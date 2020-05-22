from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import
from abc import abstractmethod

import random
import functools
import os

from transformers import AutoModelWithLMHead
from transformers import PreTrainedModel
from transformers.tokenization_utils import BatchEncoding

from .tokenizer import MaskedLMRandomTextEditsGeneratorWithTokenizer
from .....edit import TextEdit
from ......utils.cache import KeyedSingletonLoader


class MaskedLMRandomTextEditsGeneratorWithModel(
    MaskedLMRandomTextEditsGeneratorWithTokenizer
):
    # pylint: disable=too-few-public-methods
    models_cache = KeyedSingletonLoader()

    def __init__(
        self,
        model_name: str = "bert-base-cased",
        device: Optional[str] = None,
        rng: Optional[random.Random] = None,
        edits_num: Optional[Union[int, Callable[[int, Optional[int]], int]]] = None,
    ) -> None:
        super().__init__(model_name, rng, edits_num)

        self.device = device
        self.model = self.__load_model(self.model_name, device=self.device)

    def __del__(self) -> None:
        super().__del__()
        self.__unload_model(self.model_name, device=self.device)

    @abstractmethod
    def generate(self, text: str) -> List[TextEdit]:
        ...  # pragma: no cover

    def _model_encode(self, text: str) -> BatchEncoding:
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_special_tokens_mask=True,
            return_tensors="pt",
        )
        # TODO: Handle overflowing elements for long sentences.
        return encoding

    @classmethod
    def __load_model(
        cls, model_name: str, device: Optional[str] = None
    ) -> PreTrainedModel:
        key = model_name + "-" + str(device)
        model_provider = functools.partial(
            cls.__load_transformers_model, model_name, device=device
        )
        return cls.models_cache.load(key, model_provider)

    @classmethod
    def __unload_model(cls, model_name: str, device: Optional[str] = None) -> None:
        key = model_name + "-" + str(device)
        cls.models_cache.unload(key)

    @classmethod
    def __load_transformers_model(
        cls, model_name: str, **kwargs: Any
    ) -> PreTrainedModel:
        cache_dir = os.environ.get("TRANSFORMERS_CACHE_DIR", ".transformers_cache")
        model = AutoModelWithLMHead.from_pretrained(model_name, cache_dir=cache_dir)
        if not model.__class__.__name__.endswith("ForMaskedLM"):
            raise ValueError("The model provided is not a Masked LM: %s" % model_name)
        if "device" in kwargs:
            model = model.to(kwargs.get("device"))
        model.eval()
        return model
