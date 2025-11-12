
from .fusion_base import *
from .modeling_fusion_concat import *

__regisitered_models = {}

for cls in FusionBase._subclass_cls:
    # print(cls)
    if cls.model_name in __regisitered_models:
        raise ValueError(f"Duplicate model name: {cls.model_name}")
    __regisitered_models[cls.model_name] = cls

def load_fusion_model(
    name: str,
    img_embed_size: int,
    img_embed_len: int,
    txt_embed_size: int,
    **kwargs,
) -> FusionBase:
    if name in __regisitered_models:
        return __regisitered_models[name](img_embed_size, img_embed_len, txt_embed_size, **kwargs)
    else:
        raise NotImplementedError(f"Module {name} is not regisited")

def load_fusion_class(name: str) -> FusionBase:
    if name in __regisitered_models:
        return __regisitered_models[name]
    else:
        raise NotImplementedError(f"Module {name} is not regisited")

def list_fusion_models():
    return list(__regisitered_models.keys())
