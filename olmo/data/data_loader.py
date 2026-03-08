from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Dict, List, Union, Tuple

import numpy as np
import omegaconf
from torch.utils.data import DataLoader, DistributedSampler

from olmo.config import BaseConfig
from olmo.data.dataset import DeterministicDataset, Dataset
from olmo.data.get_dataset import get_dataset_by_name
from olmo.data.iterable_dataset_mixture import IterableDatasetMixture
from olmo.models.sprvla.sprvla import SPRVLAConfig
from olmo.torch_util import get_global_rank, get_world_size

log = logging.getLogger(__name__)


@dataclass
class RootSizeMixture(BaseConfig):
    rate: float
    mixture: Dict[str, Optional[float]]


@dataclass
class DatasetWithKwargs(BaseConfig):
    dataset_name: str
    sampling_rate: Optional[float] = None
    root_size_factor: Optional[float] = None
    max_high_res: Optional[int] = None
    min_high_res: Optional[int] = None

    def get_kwargs(self):
        if self.max_high_res is None:
            return dict(max_high_res=self.max_high_res, min_high_res=self.min_high_res)
        else:
            return {}


@dataclass
class DataLoaderConfig(BaseConfig):
    """Configuration for a torch `DataLoader`"""

    dataset: Optional[str] = None
    """Dataset name, will be used for `get_dataset_by_name`"""

    mixture: Optional[Dict[str, float]] = None
    """Mixture of dataset names and sampling rates"""

    root_size_mixture: Optional[List[RootSizeMixture]] = None
    """Mixture-of-mixtures where sub-mixtures rates are determined by the root dataset size"""

    kwargs_mixture: Optional[List[DatasetWithKwargs]] = None

    split: str = omegaconf.MISSING
    """Dataset split to load"""

    seed: int = omegaconf.MISSING
    """Dataset seed for shuffling and augmentation"""

    pad: Optional[str] = "to_max"
    """How to pad in the collator"""

    sequence_length: Optional[int] = None
    """Max sequence length to truncate examples to in the Collator"""

    max_text_seq_len: Optional[int] = None
    """Max sequence length excluding MM tokens
    
    If set, the sequence_length is computed as `max_text_seq_len` + the max length of the MM tokens
    """

    shuffle: Optional[bool] = True
    """Should the data be shuffled"""

    start_index: int = 0
    """Example index to start at"""

    # DataLoader args
    num_workers: int = 0
    drop_last: bool = False
    pin_memory: bool = True
    prefetch_factor: Optional[int] = None
    persistent_workers: bool = False
    timeout: int = 0

    def build_eval_dataloader(
        self,
        model_config: SPRVLAConfig,
        batch_size: int,
        for_inference: bool,
        include_metadata: bool = None,
        pad_batches: bool = False,
        max_steps_for_padding=None,
        include_image: bool = False,
    ) -> DataLoader:
        assert self.mixture is None and self.root_size_mixture is None
        log.info(f"Loading eval dataset: {self.dataset}/{self.split}")
        if include_metadata is None:
            include_metadata = for_inference

        dataset = get_dataset_by_name(self.dataset, self.split)
        n_pad = 0
        if pad_batches and not self.drop_last:
            global_batch_size = batch_size*get_world_size()
            n_steps = (len(dataset) + global_batch_size - 1) // global_batch_size
            if max_steps_for_padding:
                n_steps = min(n_steps, max_steps_for_padding)
            if n_steps*global_batch_size > len(dataset):
                # Pad the dataset so that it can produce enough batches of `global_batch_size` size
                # to cover the entire dataset without dropping any examples
                # We need this if evaluating FSDP models since they will need all devices to get
                # exactly the same number of batches
                n_pad = (n_steps*global_batch_size) - len(dataset)

        if self.pad is None:
            max_seq_len = None
        elif self.max_text_seq_len:
            max_seq_len = self.max_text_seq_len + model_config.mm_preprocessor.get_max_mm_tokens(model_config.vision_backbone)
            max_seq_len = ((max_seq_len + 8 - 1) // 8) * 8
        else:
            max_seq_len = self.sequence_length
        preprocessor = model_config.build_preprocessor(
            for_inference=for_inference, is_training=False, include_image=include_image, max_seq_len=max_seq_len)
        dataset = DeterministicDataset(
            dataset=dataset,
            seed=self.seed,
            preprocessor=preprocessor,
            n_pad=n_pad
        )
        sampler = DistributedSampler(
            dataset,
            drop_last=self.drop_last,
            shuffle=self.shuffle,
            num_replicas=get_world_size(),
            rank=get_global_rank(),
            seed=self.seed,
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=model_config.build_collator(
                max_seq_len, self.pad, include_metadata=include_metadata),
            num_workers=self.num_workers,
            sampler=sampler,
            pin_memory=self.pin_memory,
            prefetch_factor=None if self.num_workers == 0 else self.prefetch_factor,
            persistent_workers=False if self.num_workers == 0 else self.persistent_workers,
            timeout=self.timeout,
        )

    def build_train_dataloader(
        self, 
        model_config: SPRVLAConfig,
        global_batch_size: int, 
        device=None
    ) -> DataLoader:
        if device is None:
            device = "cpu"
        
        total_size = 0

        if self.pad is None:
            max_seq_len = None
        elif self.max_text_seq_len:
            max_seq_len = self.max_text_seq_len + model_config.mm_preprocessor.get_max_mm_tokens(model_config.vision_backbone)
            max_seq_len = ((max_seq_len + 8 - 1) // 8) * 8
        else:
            max_seq_len = self.sequence_length
        preprocessor = model_config.build_preprocessor(
            for_inference=False, is_training=True, max_seq_len=max_seq_len)
        if self.dataset:
            ds = get_dataset_by_name(self.dataset, self.split)
            datasets = [DeterministicDataset(ds, preprocessor, self.seed)]
            rates = [1]
        else:
            mixture: Dict[str, Tuple[Dataset, float, Optional[Dict]]] = {}
            if self.kwargs_mixture:
                for task in self.kwargs_mixture:
                    log.info(f"Loading train dataset {task.dataset_name}/{self.split}")
                    dataset = get_dataset_by_name(task.dataset_name, self.split)
                    if task.sampling_rate is not None:
                        size = task.sampling_rate
                    elif task.root_size_factor < 1:
                        size = np.sqrt(len(dataset) * task.sampling_rate)
                    else:
                        size = np.sqrt(task.root_size_factor)
                    mixture[task.dataset_name] = (dataset, size, task.get_kwargs())
            elif self.mixture:
                for name, rate in self.mixture.items():
                    log.info(f"Loading train dataset {name}/{self.split}")
                    mixture[name] = (get_dataset_by_name(name, self.split), rate, None)
            else:
                total_size = 0
                for root_size_mixture in self.root_size_mixture:
                    group_datasets = {}
                    for name, as_size in root_size_mixture.mixture.items():
                        log.info(f"Loading train dataset {name}/{self.split}")
                        dataset = get_dataset_by_name(name, self.split)
                        if as_size is None:
                            size = len(dataset)
                        elif as_size <= 1:
                            size = len(dataset) * as_size
                        else:
                            size = as_size
                        total_size += size
                        log.info(f"{size} of data loaded")
                        group_datasets[name] = (dataset, np.sqrt(size))
                    total_rate = sum(x[1] for x in group_datasets.values())
                    mixture.update({name: (ds, r/total_rate*root_size_mixture.rate, None)
                                    for name, (ds, r) in group_datasets.items()})

            log.info(f"********** in total, {total_size} of data loaded **********")

            total_rate = sum(x[1] for x in mixture.values())
            mixture = sorted(mixture.items(), key=lambda x: x[0])
            rates = [rate/total_rate for (_, (_, rate, _)) in mixture]
            datasets = []
            for _, (dataset, _, kwargs) in mixture:
                datasets.append(DeterministicDataset(dataset, preprocessor, self.seed, preprocessor_kwargs=kwargs))
            log.info("Sampling rates:")
            names = list(x[0] for x in mixture)
            for ix in np.argsort(rates)[::-1]:
                log.info(f"{names[ix]}: {100*rates[ix]:0.2f}")
                # log.info(f"{names[ix]}: {total_size*rates[ix]}")
        dataset = IterableDatasetMixture(
            start_index=self.start_index,
            datasets=datasets,
            mixture_rates=rates,
            global_batch_size=global_batch_size,
            seed=self.seed,
            shuffle=self.shuffle,
            total_size=total_size,
        )
        return DataLoader(
            dataset,
            batch_size=dataset.device_batch_size,
            drop_last=self.drop_last,
            collate_fn=model_config.build_collator(
                max_seq_len, self.pad, include_metadata=False),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            prefetch_factor=None if self.num_workers == 0 else self.prefetch_factor,
            persistent_workers=self.persistent_workers,
            timeout=self.timeout,
        )

