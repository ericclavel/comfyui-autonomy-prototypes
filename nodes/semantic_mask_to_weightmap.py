import torch
import torch.nn.functional as F

# SemanticMaskToWeightMap
# Input:  IMAGE tensor assumed to be a single-channel semantic ID mask encoded in IMAGE format (0..1 float)
#         where pixel value*255 ~= class_id. (Common for exported ID masks or segmentation outputs written as grayscale.)
# Output: MASK (0..1) suitable for conditioning modules, plus an IMAGE preview of the weight map.
#
# Notes:
# - This is intentionally simple and dependency-free (no torchvision). We build a small Gaussian kernel on the fly.
# - Batch-safe: processes all batch items in the input IMAGE.

class SemanticMaskToWeightMap:
    """Convert a grayscale semantic-ID image into a soft weight map (MASK) for diffusion conditioning."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_image": ("IMAGE",),   # float tensor [B,H,W,C] in 0..1
                "class_id": ("INT", {"default": 11, "min": 0, "max": 255, "step": 1}),
                "sigma": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 32.0, "step": 0.1}),
                "invert": ("BOOLEAN", {"default": False}),
                "edge_soften": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("weight_mask", "weight_preview")
    FUNCTION = "make_weightmap"
    CATEGORY = "Autonomy/Conditioning"

    @staticmethod
    def _make_gaussian_kernel(sigma: float, device, dtype):
        """Create a normalized 2D Gaussian kernel with size derived from sigma."""
        if sigma <= 0:
            return None
        # kernel size: odd, ~ 6*sigma rounded up
        k = int(max(3, int(6.0 * sigma) | 1))  # ensure odd
        coords = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2.0
        g = torch.exp(-(coords**2) / (2 * sigma * sigma))
        g = g / g.sum()
        kernel2d = (g[:, None] * g[None, :])
        kernel2d = kernel2d / kernel2d.sum()
        # shape for depthwise conv2d weight: [out_ch, in_ch/groups, kH, kW]
        kernel2d = kernel2d[None, None, :, :]
        return kernel2d

    def make_weightmap(self, mask_image, class_id, sigma, invert, edge_soften):
        """
        mask_image: float tensor [B,H,W,C] in 0..1
        class_id:   integer (0..255), semantic class to extract
        sigma:      gaussian blur sigma (>0 softens edges)
        invert:     invert the resulting mask
        edge_soften: if True and sigma>0, apply blur for soft transitions
        """
        # Validate and prep
        assert isinstance(mask_image, torch.Tensor), "mask_image must be a torch tensor"
        assert mask_image.ndim == 4 and mask_image.shape[-1] >= 1, "mask_image must be [B,H,W,C]"

        B, H, W, C = mask_image.shape
        device = mask_image.device
        dtype = mask_image.dtype

        # Extract first channel as grayscale ID field in 0..1 and map to [0..255] integer IDs
        gray = mask_image[..., 0]  # [B,H,W]
        ids = torch.round(gray * 255.0).to(torch.int32)  # [B,H,W]

        # Binary selection for target class
        target = (ids == int(class_id)).to(dtype)  # [B,H,W], 0 or 1 in float

        # Optional edge softening (gaussian blur)
        if edge_soften and sigma > 0.0:
            # conv2d expects [B, C, H, W]
            x = target.unsqueeze(1)  # [B,1,H,W]
            kernel = self._make_gaussian_kernel(float(sigma), device, dtype)
            if kernel is not None:
                # depthwise with groups=1 on single-channel
                x = F.conv2d(x, kernel, padding="same")
            target = x.squeeze(1)  # [B,H,W]

        # Normalize to [0..1] (already is) and optionally invert
        weight = target
        if invert:
            weight = 1.0 - weight

        # Clamp for safety
        weight = weight.clamp(0.0, 1.0)  # [B,H,W]

        # Build outputs:
        # 1) MASK expects [B,H,W] in 0..1
        weight_mask = weight

        # 2) IMAGE preview expects [B,H,W,C] in 0..1; duplicate to 3 channels
        preview = torch.stack([weight, weight, weight], dim=-1)  # [B,H,W,3]

        return (weight_mask, preview)


# ComfyUI node registration
NODE_CLASS_MAPPINGS = {
    "SemanticMaskToWeightMap": SemanticMaskToWeightMap,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SemanticMaskToWeightMap": "Semantic → Weight Map",
}
