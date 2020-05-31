from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

import functools
import os

from lm_scorer.models.abc.base import LMScorer
from lm_scorer.models.auto import AutoLMScorer

from .abc.base import ScoredTextEditsValidator
from ....utils.cache import KeyedSingletonLoader


class ValidateWithLMScore(ScoredTextEditsValidator):
    models_cache = KeyedSingletonLoader()

    def __init__(
        self,
        model_name: str = "gpt2",
        device: Optional[str] = None,
        scoring_comp: Callable[[float, float], bool] = lambda s1, s2: s1 - s2 > 0,
        scoring_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(scoring_comp, scoring_options)

        self.model_name = model_name
        self.device = device
        self.scorer = self.__load_model(self.model_name, device=self.device)

        self.last_source: Optional[Tuple[str, float]] = None

    def __del__(self) -> None:
        self.__unload_model(self.model_name, device=self.device)

    def validate(self, source_text: str, modified_text: str) -> bool:
        if self.last_source is None or self.last_source[0] != modified_text:
            source_score = self.scorer.sentence_score(
                source_text, **self.scoring_options
            )
            self.last_source = source_text, source_score

        source_score = self.last_source[1]
        modified_score = self.scorer.sentence_score(
            modified_text, **self.scoring_options
        )

        return self.scoring_comp(source_score, modified_score)

    @classmethod
    def __load_model(cls, model_name: str, device: Optional[str] = None) -> LMScorer:
        key = model_name + "-" + str(device)
        scorer_provider = functools.partial(
            cls.__load_scorer_model, model_name, device=device
        )
        return cls.models_cache.load(key, scorer_provider)

    @classmethod
    def __unload_model(cls, model_name: str, device: Optional[str] = None) -> None:
        key = model_name + "-" + str(device)
        cls.models_cache.unload(key)

    @classmethod
    def __load_scorer_model(cls, model_name: str, **kwargs) -> LMScorer:
        cache_dir = os.environ.get("TRANSFORMERS_CACHE_DIR", ".transformers_cache")
        kwargs["cache_dir"] = kwargs.get("cache_dir", cache_dir)

        scorer = AutoLMScorer.from_pretrained(model_name, **kwargs)
        return scorer
