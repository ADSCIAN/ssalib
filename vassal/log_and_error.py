from functools import wraps
import logging
from typing import Any, Callable, TypeVar

R = TypeVar('R')


def ignored_argument_warning(
        *ignored_args: str,
        log_level: str = 'warning'
) -> Callable[[Callable[..., R]], Callable[..., R]]:
    """
    Decorator to ignore specified keyword arguments and log a warning if they
    are passed to the function.

    Parameters
    ----------
    *ignored_args : str
        Variable length argument list specifying the keyword arguments to
        ignore.
    log_level : str
        Logging level for the message (default is 'warning').

    Returns
    -------
    Callable[[Callable[..., R]], Callable[..., R]]
        A decorator that wraps functions, ignoring specified keyword arguments
        and issuing warnings for them.
    """

    def decorator(func: Callable[..., R]) -> Callable[..., R]:
        logger = logging.getLogger(func.__name__)
        log_func = getattr(logger, log_level)

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> R:
            for arg in ignored_args:
                if arg in kwargs:
                    if kwargs[arg] is not None:
                        log_func(f"Ignored {arg}={kwargs[arg]} passed to "
                                 f"{func.__name__} method.")
                    del kwargs[arg]
            return func(*args, **kwargs)

        return wrapper

    return decorator


class DecompositionError(Exception):
    """Exception raised when a dependent method is called before decompose"""

    def __init__(
            self,
            message: str
    ) -> None:
        super().__init__(message)


class ReconstructionError(Exception):
    """Exception raised when a dependent method is called before reconstruct"""

    def __init__(
            self,
            message: str
    ) -> None:
        super().__init__(message)
