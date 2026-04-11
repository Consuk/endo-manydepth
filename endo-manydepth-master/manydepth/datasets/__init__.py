# flake8: noqa: F401

from .cityscapes_evaldataset import CityscapesEvalDataset
from .cityscapes_preprocessed_dataset import CityscapesPreprocessedDataset
from .hamlyn_dataset import HamlynDataset
from .kitti_dataset import KITTIDepthDataset, KITTIOdomDataset, KITTIRAWDataset
from .scared_dataset import SCAREDDataset, SCAREDRAWDataset

try:
    from .c3vd_dataset import C3VDDataset
except Exception:
    C3VDDataset = None


__all__ = [
    "KITTIRAWDataset",
    "KITTIOdomDataset",
    "KITTIDepthDataset",
    "CityscapesPreprocessedDataset",
    "CityscapesEvalDataset",
    "SCAREDDataset",
    "SCAREDRAWDataset",
    "HamlynDataset",
    "C3VDDataset",
]


dataset_dict = {
    "endovis": SCAREDRAWDataset,
    "hamlyn": HamlynDataset,
}

if C3VDDataset is not None:
    dataset_dict["c3vd"] = C3VDDataset
