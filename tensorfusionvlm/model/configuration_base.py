import enum
from dataclasses import dataclass

from transformers import PretrainedConfig

# from llava.constants import IGNORE_INDEX

# the constants
DEFAULT_SYSTEM_PROMPT = """You are an AI visual assistant that can analyze images. There could be an object with a bounding box in the chat, where an object is surrounded with <st> and <ed> and the bounding box is in the form of [x1, y1, x2, y2], which are a list of float numbers normalized from 0 to 1, corresponding to the top left x1, top left y1, bottom right x2, and bottom right y2. Please answer the question based on the visual input. If you do not see the object, do not say anything about it."""
# 

class Role(enum.Enum):
    user = 0
    system = 1
    assistant = 2

@dataclass
class ChatHistoryItem:
    role: Role
    content: str

    @staticmethod
    def new_chat_user(msg: str):
        return ChatHistoryItem(Role.user, msg)

    @staticmethod
    def new_chat_system(msg: str):
        return ChatHistoryItem(Role.system, msg)

    @staticmethod
    def new_chat_assistant(msg: str):
        return ChatHistoryItem(Role.assistant, msg)

class TensorFusionVLMConfig(PretrainedConfig):
    """
    Common attributes (present in all subclasses):

    - vocab_size (`int`) -- The number of tokens in the vocabulary, which is also the first dimension of the
      embeddings matrix (this attribute may be missing for models that don't have a text modality like ViT).
    - hidden_size (`int`) -- The hidden size of the model.
    - num_attention_heads (`int`) -- The number of attention heads used in the multi-head attention layers of the
      model.
    - num_hidden_layers (`int`) -- The number of blocks in the model.
    """
    model_type = "tensor_fusion_vlm"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        llm_backbone: str = None,
        vision_backbone: str = None,
        fusion_backbone: str = None,
        hidden_size: int = 4096,
        llm_load_pretrained: bool = True,
        vision_load_pretrained: bool = True,
        vision_select_layer: int = -1,
        vision_select_feature: str = 'pad',
        vision_backbone_align_chckpoint: str = None,
        max_sequence_len: int = 0,
        enable_que_end: bool = False,
        patch_checkpoint_name: str = None, # if None, not enable
        patch_num_hidden_layers: int = None, 
        patch_img_project_scale: float = None,
        **kwargs,
    ):
        self.llm_backbone = llm_backbone
        self.vision_backbone = vision_backbone
        self.fusion_backbone = fusion_backbone
        self.hidden_size = hidden_size
        self.llm_load_pretrained = llm_load_pretrained
        self.vision_load_pretrained = vision_load_pretrained
        self.vision_select_layer = vision_select_layer
        self.vision_select_feature = vision_select_feature
        self.vision_backbone_align_checkpoint = vision_backbone_align_chckpoint
        self.max_sequence_len = max_sequence_len
        self.enable_que_end = enable_que_end

        # configure for patch
        self.patch_checkpoint_name = patch_checkpoint_name
        self.patch_num_hidden_layers = patch_num_hidden_layers
        self.patch_img_project_scale = patch_img_project_scale

        super().__init__(
            llm_backbone = llm_backbone,
            vision_backbone = vision_backbone,
            fusion_backbone = fusion_backbone,
            hidden_size = hidden_size,
            max_sequence_len = max_sequence_len,
            is_decoder=False,
            **kwargs,
        )