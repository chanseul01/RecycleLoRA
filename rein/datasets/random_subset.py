import copy

import numpy as np
from mmengine.dataset import force_full_init
from mmengine.logging import MMLogger
from mmseg.registry import DATASETS


@DATASETS.register_module()
class RandomSubsetDataset:
    """Dataset wrapper that exposes a fixed-size subset of a wrapped dataset.

    Args:
        dataset (dict | Dataset): the dataset (config or built instance) to wrap.
        num_samples (int): number of items to keep. Clipped to ``len(dataset)``.
        seed (int | None): RNG seed for the subset draw.
        lazy_init (bool): defer ``full_init`` until first use.
    """

    def __init__(self, dataset, num_samples, seed=None, lazy_init=False):
        if isinstance(dataset, dict):
            self.dataset = DATASETS.build(dataset)
        elif hasattr(dataset, "full_init") or hasattr(dataset, "__getitem__"):
            self.dataset = dataset
        else:
            raise TypeError(
                "dataset must be a dict config or a built dataset, "
                f"got {type(dataset)}"
            )

        self.num_samples = int(num_samples)
        self.seed = seed
        self._metainfo = getattr(self.dataset, "metainfo", {})
        self._fully_initialized = False
        self._indices = None
        if not lazy_init:
            self.full_init()

    @property
    def metainfo(self) -> dict:
        return copy.deepcopy(self._metainfo)

    def full_init(self):
        if self._fully_initialized:
            return
        self.dataset.full_init()
        total = len(self.dataset)
        n = min(self.num_samples, total)

        seed = self.seed
        if seed is None:
            seed = int(np.random.SeedSequence().generate_state(1)[0])
        rng = np.random.default_rng(seed)

        # keep the original ordering; the sampler shuffles during training
        self._indices = sorted(
            rng.choice(total, size=n, replace=False).tolist()
        )
        self._resolved_seed = seed
        self._fully_initialized = True

        MMLogger.get_current_instance().info(
            f"RandomSubsetDataset: seed={seed}, kept {n}/{total} samples"
        )

    @force_full_init
    def _map_index(self, idx: int) -> int:
        return self._indices[idx]

    @force_full_init
    def get_data_info(self, idx: int) -> dict:
        return self.dataset.get_data_info(self._indices[idx])

    @force_full_init
    def __getitem__(self, idx: int):
        return self.dataset[self._indices[idx]]

    @force_full_init
    def __len__(self) -> int:
        return len(self._indices)
