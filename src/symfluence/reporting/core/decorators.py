# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2024-2026 SYMFLUENCE Team <dev@symfluence.org>

"""
Decorators for the reporting module.

Provides reusable decorators for common patterns in visualization methods.
"""
from __future__ import annotations

from functools import wraps
from typing import Any, Callable, Dict, List, TypeVar, Union, get_type_hints

# Type variable for the return type
T = TypeVar('T')

# Default return values by type annotation
_DEFAULT_RETURNS: Dict[type, Any] = {
    type(None): None,
    dict: {},
    Dict: {},
    list: [],
    List: [],
    str: '',
}


def _infer_default_from_return_type(func: Callable) -> Any:
    """
    Infer the default return value from a function's return type annotation.

    Args:
        func: The function to inspect

    Returns:
        Appropriate default value based on return type, or None if not inferrable
    """
    try:
        hints = get_type_hints(func)
        return_type = hints.get('return')

        if return_type is None:
            return None

        # Handle Optional types (Union[X, None])
        # For Optional[X], we should return None since None is a valid return value
        origin = getattr(return_type, '__origin__', None)
        if origin is Union:
            args = getattr(return_type, '__args__', ())
            # If NoneType is one of the args, this is Optional - return None
            if type(None) in args:
                return None
            # Otherwise, use the first non-None type
            non_none_args = [a for a in args if a is not type(None)]
            if non_none_args:
                return_type = non_none_args[0]
            else:
                return None

        # Check for dict/Dict
        if return_type is dict or getattr(return_type, '__origin__', None) is dict:
            return {}

        # Check for list/List
        if return_type is list or getattr(return_type, '__origin__', None) is list:
            return []

        # Check for str
        if return_type is str:
            return ''

        # Default to None for other types
        return None

    except (TypeError, AttributeError):
        # If type inspection fails, return None
        return None


def skip_if_not_visualizing(default: Any = None, *, auto_infer: bool = True):
    """
    Decorator that skips method execution if visualization is disabled.

    This decorator eliminates the need for repetitive `if not self.visualize: return`
    checks at the beginning of visualization methods in ReportingManager.

    The decorated method must be on a class that has a `visualize` attribute.

    Args:
        default: The default value to return when visualization is disabled.
                 If not specified and auto_infer is True, the default will be
                 inferred from the method's return type annotation.
        auto_infer: If True (default), automatically infer the default return
                   value from the method's return type annotation when no
                   explicit default is provided.

    Returns:
        Decorated function that returns early if self.visualize is False

    Usage:
        @skip_if_not_visualizing()  # Auto-infers default from return type
        def visualize_domain(self) -> Optional[str]:
            ...

        @skip_if_not_visualizing(default={})  # Explicit default
        def visualize_summa_outputs(self, experiment_id: str) -> Dict[str, str]:
            ...

        @skip_if_not_visualizing(default=[])
        def visualize_benchmarks(self, results: Dict) -> List[str]:
            ...

    Example transformation:
        # Before:
        def visualize_domain(self) -> Optional[str]:
            if not self.visualize:
                return None
            self.logger.info("Creating domain visualization...")
            return self.domain_plotter.plot_domain()

        # After:
        @skip_if_not_visualizing()
        def visualize_domain(self) -> Optional[str]:
            self.logger.info("Creating domain visualization...")
            return self.domain_plotter.plot_domain()
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Determine the default return value
        if default is not None:
            return_default = default
        elif auto_infer:
            return_default = _infer_default_from_return_type(func)
        else:
            return_default = None

        @wraps(func)
        def wrapper(self, *args, **kwargs) -> T:
            # Check if visualization is enabled
            if not getattr(self, 'visualize', False):
                return return_default
            return func(self, *args, **kwargs)

        return wrapper
    return decorator


def skip_if_not_diagnostic(default: Any = None, *, auto_infer: bool = True):
    """
    Decorator that skips method execution if diagnostic mode is disabled.

    This decorator eliminates the need for repetitive `if not self.diagnostic: return`
    checks at the beginning of diagnostic methods in ReportingManager.

    The decorated method must be on a class that has a `diagnostic` attribute.

    Args:
        default: The default value to return when diagnostic mode is disabled.
                 If not specified and auto_infer is True, the default will be
                 inferred from the method's return type annotation.
        auto_infer: If True (default), automatically infer the default return
                   value from the method's return type annotation when no
                   explicit default is provided.

    Returns:
        Decorated function that returns early if self.diagnostic is False

    Usage:
        @skip_if_not_diagnostic()  # Auto-infers default from return type
        def diagnostic_domain_definition(self, basin_gdf, dem_path) -> Optional[str]:
            ...

        @skip_if_not_diagnostic(default={})  # Explicit default
        def diagnostic_model_output(self, output_nc, model_name) -> Dict[str, str]:
            ...
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        # Determine the default return value
        if default is not None:
            return_default = default
        elif auto_infer:
            return_default = _infer_default_from_return_type(func)
        else:
            return_default = None

        @wraps(func)
        def wrapper(self, *args, **kwargs) -> T:
            # Check if diagnostic mode is enabled
            if not getattr(self, 'diagnostic', False):
                return return_default
            return func(self, *args, **kwargs)

        return wrapper
    return decorator
