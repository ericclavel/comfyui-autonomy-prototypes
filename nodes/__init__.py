# Package init for comfyui-autonomy-prototypes
# Expose NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS for ComfyUI loader.

from .semantic_mask_to_weightmap import (
    NODE_CLASS_MAPPINGS as _SM_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _SM_DISPLAY,
)

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# Merge node maps (easy to extend as we add more nodes)
NODE_CLASS_MAPPINGS.update(_SM_MAPPINGS)
NODE_DISPLAY_NAME_MAPPINGS.update(_SM_DISPLAY)
