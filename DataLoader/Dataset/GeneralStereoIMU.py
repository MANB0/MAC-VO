import cv2
import torch
import numpy as np
import pypose as pp

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..Interface import StereoFrame, StereoInertialFrame, StereoData, IMUData
from ..SequenceBase import SequenceBase
from Utility.PrettyPrint import Logger


class IMUCSVLoader:
    def __init__(self, csv_path: Path) -> None:
        assert csv_path.exists(), f"IMU csv file does not exist: {csv_path}"
        raw = np.genfromtxt(str(csv_path), delimiter=",", names=True)
        assert raw.size > 0, f"No IMU rows in {csv_path}"

        self.time_ns = torch.from_numpy(raw[self._pick_field(raw.dtype.names, ["timestamp", "time_ns", "time"])]).long()
        gx = raw[self._pick_field(raw.dtype.names, ["ang_vel_x", "gyro_x", "wx"])]
        gy = raw[self._pick_field(raw.dtype.names, ["ang_vel_y", "gyro_y", "wy"])]
        gz = raw[self._pick_field(raw.dtype.names, ["ang_vel_z", "gyro_z", "wz"])]
        ax = raw[self._pick_field(raw.dtype.names, ["lin_acc_x", "acc_x", "ax"])]
        ay = raw[self._pick_field(raw.dtype.names, ["lin_acc_y", "acc_y", "ay"])]
        az = raw[self._pick_field(raw.dtype.names, ["lin_acc_z", "acc_z", "az"])]

        self.gyro = torch.from_numpy(np.stack([gx, gy, gz], axis=-1)).float()
        self.acc = torch.from_numpy(np.stack([ax, ay, az], axis=-1)).float()

    @staticmethod
    def _pick_field(fields: tuple[str, ...], candidates: list[str]) -> str:
        field_set = {f.lower(): f for f in fields}
        for cand in candidates:
            if cand.lower() in field_set:
                return field_set[cand.lower()]
        raise KeyError(f"Cannot find any of {candidates} in csv fields {fields}")

    def query_range(self, start_ns: int, end_ns: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if end_ns < start_ns:
            start_ns, end_ns = end_ns, start_ns

        i0 = int(torch.searchsorted(self.time_ns, torch.tensor(start_ns), right=False).item())
        i1 = int(torch.searchsorted(self.time_ns, torch.tensor(end_ns), right=True).item())

        i0 = max(0, min(i0, self.time_ns.numel()))
        i1 = max(0, min(i1, self.time_ns.numel()))

        return self.time_ns[i0:i1], self.acc[i0:i1], self.gyro[i0:i1]

    def query_nearest(self, target_ns: int) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = int(torch.searchsorted(self.time_ns, torch.tensor(target_ns), right=False).item())
        if idx <= 0:
            nearest = 0
        elif idx >= self.time_ns.numel():
            nearest = self.time_ns.numel() - 1
        else:
            left = idx - 1
            right = idx
            nearest = left if abs(int(self.time_ns[left].item()) - target_ns) <= abs(int(self.time_ns[right].item()) - target_ns) else right

        return nearest, self.time_ns[nearest:nearest + 1], self.acc[nearest:nearest + 1], self.gyro[nearest:nearest + 1]


class MonocularDataset:
    """
    Return images in shape (1, 3, H, W), float32, normalized to [0, 1].
    """

    def __init__(self, directory: Path, image_format: str) -> None:
        self.directory = directory
        assert self.directory.exists(), f"Monocular image directory does not exist: {self.directory}"

        self.file_names = sorted(directory.glob(f"*.{image_format}"))
        assert len(self.file_names) > 0, f"No image with '.{image_format}' suffix under {self.directory}"

        self.timestamps = [int(p.stem) for p in self.file_names]

    def __len__(self) -> int:
        return len(self.file_names)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = cv2.imread(str(self.file_names[index]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image at {self.file_names[index]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        image /= 255.0
        return image


class GeneralStereoIMUSequence(SequenceBase[StereoInertialFrame]):
    @classmethod
    def name(cls) -> str:
        return "GeneralStereoIMU"

    def __init__(self, config: SimpleNamespace | dict[str, Any]) -> None:
        cfg = self.config_dict2ns(config)

        self.seq_root = Path(cfg.root)
        self.baseline = float(cfg.bl)
        self.gravity = float(getattr(cfg, "gravity", 9.81))
        self.imu_window_ns = int(getattr(cfg, "imu_window_ns", 100_000_000))
        self.imu_fallback_max_dt_ns = int(getattr(cfg, "imu_fallback_max_dt_ns", 50_000_000))
        self.auto_estimate_time_offset = bool(getattr(cfg, "auto_estimate_time_offset", True))
        self.imu_time_offset_ns = int(getattr(cfg, "imu_time_offset_ns", 0))

        self.cam_T_BS = pp.identity_SE3(1, dtype=torch.float64)
        self.imu_T_BS = pp.identity_SE3(1, dtype=torch.float64)

        self.image_l = MonocularDataset(Path(self.seq_root, "left"), cfg.format)
        self.image_r = MonocularDataset(Path(self.seq_root, "right"), cfg.format)
        assert len(self.image_l) == len(self.image_r), "Left and right images are not same length"
        assert self.image_l.timestamps == self.image_r.timestamps, "Left/right timestamps are not aligned"

        if hasattr(cfg.camera, "fx"):
            self.K = torch.tensor(
                [[
                    [cfg.camera.fx, 0.0, cfg.camera.cx],
                    [0.0, cfg.camera.fy, cfg.camera.cy],
                    [0.0, 0.0, 1.0],
                ]],
                dtype=torch.float32,
            )
        else:
            self.K = torch.tensor(np.load(Path(self.seq_root, "intrinsic.npy")), dtype=torch.float32)

        self.frame_timestamps = self.image_l.timestamps
        self.imu_loader = IMUCSVLoader(Path(self.seq_root, "imu_data.csv"))

        if self.auto_estimate_time_offset:
            cam_first, cam_last = self.frame_timestamps[0], self.frame_timestamps[-1]
            imu_first = int(self.imu_loader.time_ns[0].item())
            imu_last = int(self.imu_loader.time_ns[-1].item())

            offset_head = cam_first - imu_first
            offset_tail = cam_last - imu_last
            self.imu_time_offset_ns = int((offset_head + offset_tail) // 2)

        Logger.write(
            "info",
            (
                "GeneralStereoIMU time offset (camera_ns - imu_ns) = "
                f"{self.imu_time_offset_ns} ns, "
                f"auto_estimate={self.auto_estimate_time_offset}, "
                f"fallback_max_dt_ns={self.imu_fallback_max_dt_ns}"
            ),
        )

        super().__init__(len(self.image_l))

    def __getitem__(self, local_index: int) -> StereoInertialFrame:
        index = self.get_index(local_index)
        frame_ns_cam = self.frame_timestamps[index]
        prev_ns_cam = self.frame_timestamps[index - 1] if index > 0 else (frame_ns_cam - self.imu_window_ns)

        frame_ns_imu = frame_ns_cam - self.imu_time_offset_ns
        prev_ns_imu = prev_ns_cam - self.imu_time_offset_ns

        imu_time_ns, imu_acc, imu_gyro = self.imu_loader.query_range(prev_ns_imu, frame_ns_imu)
        if imu_time_ns.numel() == 0:
            _, nearest_time_ns, nearest_acc, nearest_gyro = self.imu_loader.query_nearest(frame_ns_imu)
            aligned_nearest_time_ns = int(nearest_time_ns[0].item()) + self.imu_time_offset_ns
            dt_ns = abs(aligned_nearest_time_ns - frame_ns_cam)

            use_nearest = (self.imu_fallback_max_dt_ns < 0) or (dt_ns <= self.imu_fallback_max_dt_ns)
            if use_nearest:
                imu_time_ns, imu_acc, imu_gyro = nearest_time_ns, nearest_acc, nearest_gyro
            else:
                imu_time_ns = torch.empty((0,), dtype=torch.long)
                imu_acc = torch.empty((0, 3), dtype=torch.float32)
                imu_gyro = torch.empty((0, 3), dtype=torch.float32)

        image_l = self.image_l[index]
        image_r = self.image_r[index]

        return StereoInertialFrame(
            idx=[local_index],
            time_ns=[frame_ns_cam],
            stereo=StereoData(
                T_BS=self.cam_T_BS,
                K=self.K,
                baseline=torch.tensor([self.baseline], dtype=torch.float32),
                width=image_l.size(-1),
                height=image_l.size(-2),
                time_ns=[frame_ns_cam],
                imageL=image_l,
                imageR=image_r,
            ),
            imu=IMUData(
                T_BS=self.imu_T_BS,
                time_ns=imu_time_ns.view(1, -1, 1),
                gravity=[self.gravity],
                acc=imu_acc.view(1, -1, 3),
                gyro=imu_gyro.view(1, -1, 3),
            ),
        )

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "root": lambda s: isinstance(s, str),
            "bl": lambda v: isinstance(v, (float, int)) and v > 0,
            "format": lambda s: isinstance(s, str),
            "gravity": lambda v: isinstance(v, (float, int)) and v > 0,
            "imu_window_ns": lambda v: isinstance(v, int) and v > 0,
            "imu_fallback_max_dt_ns": lambda v: isinstance(v, int) and v >= -1,
            "auto_estimate_time_offset": lambda v: isinstance(v, bool),
            "imu_time_offset_ns": lambda v: isinstance(v, int),
            "camera": lambda v: isinstance(v, dict) and (
                len(v) == 0
                or cls._enforce_config_spec(v, {
                    "fx": lambda x: isinstance(x, (float, int)),
                    "fy": lambda x: isinstance(x, (float, int)),
                    "cx": lambda x: isinstance(x, (float, int)),
                    "cy": lambda x: isinstance(x, (float, int)),
                }, allow_excessive_cfg=True)
            ),
        }, allow_excessive_cfg=True)
