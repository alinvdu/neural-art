from torch import nn

#@title Utils for ELDM
def instantiate_from_config(config):
    if not "target" in config:
        if config in ['__is_first_stage__', "__is_unconditional__"]:
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    print(string)
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)
