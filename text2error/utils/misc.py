from typing import *  # pylint: disable=wildcard-import,unused-wildcard-import

T1 = TypeVar("T1")  # pylint: disable=invalid-name
T2 = TypeVar("T2")  # pylint: disable=invalid-name


def resolve_optional(variable: Optional[T1], default_value: T2) -> Union[T1, T2]:
    if variable is None:
        return default_value
    return variable


def resolve_value_or_callable(
    variable: Union[T1, Callable[..., T1]], *callable_args: Any
) -> T1:
    if callable(variable):
        return variable(*callable_args)
    return variable


def resolve_optional_value_or_callable(
    variable: Optional[Union[T1, Callable[..., T1]]],
    default_value: T1,
    *callable_args: Any
) -> T1:
    if variable is None:
        return default_value
    if callable(variable):
        return variable(*callable_args)
    return variable
