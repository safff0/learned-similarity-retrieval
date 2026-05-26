import importlib
import pkgutil

for _, _mod_name, _ in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{_mod_name}")
