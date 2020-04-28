from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

from .generator import Text2ErrorGenerator
from .validator import Text2ErrorValidator


class Text2Error:
    # pylint: disable=too-few-public-methods
    def __init__(
        self,
        generator: Text2ErrorGenerator,
        validator: Text2ErrorValidator,
        max_validation_iterations: int = 10000,
    ):
        self.generator = generator
        self.validator = validator
        self.max_validation_iterations = max_validation_iterations

    def __call__(self, source_text: str):
        for _ in range(self.max_validation_iterations):
            modified_text = self.generator(source_text)
            valid = self.validator(source_text, modified_text)
            if valid:
                return modified_text
        raise RuntimeError("Maximum validation iterations exceeded")
