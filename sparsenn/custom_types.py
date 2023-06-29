import jax
import jax.numpy as jnp
import typing
from typing import TYPE_CHECKING, TypeVar, Any

Key = jax.random.PRNGKeyArray
Array = jnp.ndarray


class _WithRepr(type):
    def __new__(mcs, obj, string):
        out = super().__new__(mcs, string, (), {})
        # prevent the custom typing repr from doing the wrong thing
        out.__module__ = "builtins"
        return out

    def __init__(cls, obj, string):
        cls.obj = obj
        cls.string = string

    def __repr__(cls):
        return cls.string

    def __call__(cls, *args, **kwargs):
        return cls.obj(*args, **kwargs)


_T = TypeVar("_T")


def doc_repr(obj: _T, string: str) -> _T:
    if TYPE_CHECKING:
        return obj
    else:
        if getattr(typing, "GENERATING_DOCUMENTATION", False):
            return _WithRepr(obj, string)
        else:
            return obj


sentinel: Any = doc_repr(object(), "sentinel")
