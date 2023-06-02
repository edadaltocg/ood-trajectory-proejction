"""
Pipeline module.
"""
from enum import Enum
from typing import Any, List, Optional, Tuple

from src.pipelines.base import Pipeline

pipeline_registry = {}


def register_pipeline(name: str, dataset_name: Optional[str] = None):
    """
    Decorator to register a new pipeline in the registry.

    Args:
        name (str): The name of the pipeline to register.
    """

    def decorator(f):
        if dataset_name is None:
            pipeline_registry[name] = f
        else:
            pipeline_registry[name + "_" + dataset_name] = f
        return f

    return decorator


def create_pipeline(name: str, dataset_name: Optional[str] = None, **kwargs) -> Pipeline:
    """
    Example::

        >>> import src
        >>> pipe = src.create_pipeline("ood_benchmark", "cifar10")
        >>> pipe.run(detector)
    """
    if dataset_name is None:
        return pipeline_registry[name](**kwargs)
    return pipeline_registry[name + "_" + dataset_name](**kwargs)


def list_pipelines() -> List[str]:
    """
    List all available pipelines.

    Returns:
        List[str]: A list of available pipelines.
    """
    return list(pipeline_registry.keys())


def list_pipeline_args(name: str) -> List[Tuple[str, Any]]:
    """
    List all available arguments for a given pipeline.

    Args:
        name (str): The name of the pipeline.

    Returns:
        list: A list of available arguments and default values for the pipeline.
    """
    import inspect

    signature = inspect.signature(pipeline_registry[name]).parameters
    return [(name, parameter.default) for name, parameter in signature.items()]


from .ood import *

PipelinesRegistry = Enum("PipelinesRegistry", dict(zip(list_pipelines(), list_pipelines())))
