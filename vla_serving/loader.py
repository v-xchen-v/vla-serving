from importlib import import_module
from typing import Any, Dict, Type
from .base_service import BaseModelService


def load_backend_class(class_path: str) -> Type[BaseModelService]:
    """
    Dynamically load a backend class from a given class path.
    Example class_path: "vla_serving.examples.dummy_service.DummyService"
    """
    module_path, class_name = class_path.rsplit('.', 1)
    module = import_module(module_path)
    backend_class = getattr(module, class_name)
    if not issubclass(backend_class, BaseModelService):
        raise ValueError(f"Loaded class {class_name} is not a subclass of BaseModelService")
    return backend_class


def build_service_from_config(cfg: Dict[str, Any]) -> BaseModelService:
    """
    Build a model service from a configuration dictionary.
    The cfg should contain 'backend_class' and 'backend_config' keys.
    """
    backend_class_path = cfg['backend_class']
    backend_config = cfg.get('backend_config', {})
    
    backend_class = load_backend_class(backend_class_path)
    service_instance = backend_class.from_config(backend_config)
    
    return service_instance