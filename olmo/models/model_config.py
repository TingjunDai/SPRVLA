import dataclasses
import typing
from typing import Callable, Optional, List, cast, Dict, Any, Type
import numpy as np

from torch import nn
from omegaconf import OmegaConf as om

from olmo.config import BaseConfig
from olmo.io import PathOrStr, read_file
from olmo.models.model import ModelBase


def get_model_types() -> Dict[str, Type['BaseModelConfig']]:
    """Get a dictionary of model names to their classes"""
    # import here to avoid circular imports
    from olmo.models.sprvla.sprvla import SPRVLAConfig

    return {
        SPRVLAConfig._model_name: SPRVLAConfig,
    }


@dataclasses.dataclass
class BaseModelConfig(BaseConfig):
    """Base class for Model configs"""

    _model_name: typing.ClassVar[str]
    """
    Unique name of the model
    """

    model_name: str = dataclasses.field(init=False)
    """
    Unique name to used to identify the subclass when loading configs, should mirror the 
    class variable `_model_name`. We duplicate it as a field so OmegaConf saves it  
    """

    def __post_init__(self):
        self.model_name = self._model_name

    def build_preprocessor(
        self,
        for_inference,
        is_training=True,
        sequence_length: Optional[int] = None,
    ) -> Callable[[Dict, np.random.RandomState], Dict]:
        """
        Build a preprocessor that processes individual examples

        :param for_inference: If the examples will be used for inference
        :param is_training: If train-time augmentation should be applied
        :param sequence_length: Max sequence length allowed
        """
        raise NotImplementedError()

    def build_collator(
        self,
        sequence_length,
        pad_mode: str,
        include_metadata=True
    ) -> Callable[[List[Dict]], Dict]:
        """
        Build a collator to build batches from preprocessed examples,

        :param sequence_length: sequence length to use when padding or truncating
        :param pad_mode: How to truncate/pad data
        :param include_metadata: If batch should include non-tensor metadata field
                                 (usually only used or evaluation)
        """
        raise NotImplementedError()

    def build_model(self, device=None) -> ModelBase:
        """
        Build a model that takes batches from the collator as input
        """
        raise NotImplementedError()

    @classmethod
    def get_default_model_name(cls):
        return "sprvla"

    @classmethod
    def load(
        cls,
        path: PathOrStr,
        overrides: Optional[List[str]] = None,
        key: Optional[str] = None,
        validate_paths: bool = True,
    ) -> 'BaseModelConfig':
        """Load from a YAML file."""
        raw = om.create(read_file(path))
        if key is not None:
            raw = raw[key]

        # Manually resolve the correct subclass using `model_name`
        model_name = raw.get("model_name", cls.get_default_model_name())
        model_types = get_model_types()
        if model_name not in model_types:
            raise ValueError(f"Unknown model type {model_name}")
        model_cls = model_types[model_name]
        schema = om.structured(model_cls)
        raw = model_cls.update_legacy_settings(raw)
        conf = om.merge(schema, raw)
        if overrides:
            conf.merge_with_dotlist(overrides)
        return cast(model_cls, om.to_object(conf))
