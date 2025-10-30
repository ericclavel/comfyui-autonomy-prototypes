import torch
import torch.nn.functional as F

class SemanticMaskToWeightMap:
    """Convert a semantic mask (binary or ID map) into a soft weight map."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Accept IMAGE tensors from SAM or ID maps: [B,H,W,C], 0..1 floats (or bools)
                "mask_image": ("IMAGE",),
                # -1  -> treat non-zero as mask (binary / SAM)
                # 0-255 -> select this ID from an ID map
                "class_id": ("INT", {"default": -1, "min": -1, "max": 255, "step": 1}),
                "sigma": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 64.0, "step": 0.1}),
                "invert": ("BOOLEAN", {"default": False}),
                "edge_soften": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE")
    RETURN_NAMES = ("weight_mask", "weight_preview")
    FUNCTION = "make_weightmap"
    CATEGORY = "Autonomy/Conditioning"

    @staticmethod
    def _gauss_1d(sigma: float, k: int, device, dtype):
        coords = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2.0
        g = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
        g = g / g.sum()
        return g.view(1, 1, 1, k), g.view(1, 1, k, 1)  # (horiz, vert)

    @staticmethod
    def _kernel_size_from_sigma(sigma: float, cap: int = 129):
        if sigma <= 0:
            return 0
        k = int(max(3, int(6.0 * sigma)))
        if k % 2 == 0:
            k += 1
        return min(k, cap | 1)

    def make_weightmap(self, mask_image, class_id, sigma, invert, edge_soften):
        assert isinstance(mask_image, torch.Tensor), "mask_image must be a torch tensor"
        assert mask_image.ndim == 4 and mask_image.shape[-1] >= 1, "mask_image must be [B,H,W,C]"

        B, H, W, C = mask_image.shape
        device = mask_image.device
        in_dtype = mask_image.dtype

        # Use a safe working dtype (keep half types on CUDA)
        work_dtype = in_dtype if (mask_image.is_cuda and in_dtype in (torch.float16, torch.bfloat16)) else torch.float32

        with torch.no_grad():
            mi = mask_image.to(device=device, dtype=work_dtype)
            gray = mi[..., 0]  # [B,H,W]

            # Heuristic: detect binary-like mask (SAM) and bypass class_id
            # If class_id == -1, always treat as binary.
            use_binary = (class_id == -1)
            if not use_binary:
                # Quick binary check: values ~ {0,1} within tolerance
                mn = float(gray.min().item())
                mx = float(gray.max().item())
                use_binary = (mx <= 1.0 + 1e-4) and (mn >= -1e-4) and ((mx > 0.5 and mn < 0.5) and (mx - mn > 0.4))

            if use_binary:
                # Threshold at 0.5
                target = (gray > 0.5).to(dtype=work_dtype)
            else:
                # Treat as ID map encoded in 0..1 → 0..255
                ids = torch.round(gray * 255.0).to(torch.int32)
                target = (ids == int(class_id)).to(dtype=work_dtype)

            # Optional edge soften via separable Gaussian
            if edge_soften and sigma > 0.0:
                k = self._kernel_size_from_sigma(float(sigma))
                if k >= 3:
                    x = target.unsqueeze(1)  # [B,1,H,W]
                    try:
                        g_h, g_v = self._gauss_1d(float(sigma), k, device, work_dtype)
                        pad_v = (0, 0, k // 2, k // 2)
                        pad_h = (k // 2, k // 2, 0, 0)
                        x = F.conv2d(F.pad(x, pad_v, mode="replicate"), g_v, groups=1)
                        x = F.conv2d(F.pad(x, pad_h, mode="replicate"), g_h, groups=1)
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e):
                            x_cpu = target.unsqueeze(1).to("cpu", torch.float32)
                            g_hc, g_vc = self._gauss_1d(float(sigma), k, "cpu", torch.float32)
                            pad_v = (0, 0, k // 2, k // 2)
                            pad_h = (k // 2, k // 2, 0, 0)
                            x_cpu = F.conv2d(F.pad(x_cpu, pad_v, mode="replicate"), g_vc, groups=1)
                            x_cpu = F.conv2d(F.pad(x_cpu, pad_h, mode="replicate"), g_hc, groups=1)
                            x = x_cpu.to(device=device, dtype=work_dtype)
                        else:
                            raise
                    target = x.squeeze(1)

            weight = 1.0 - target if invert else target
            weight = weight.clamp(0.0, 1.0)

            weight_mask = weight                      # [B,H,W]
            preview = torch.stack([weight, weight, weight], dim=-1)  # [B,H,W,3]

        return (weight_mask, preview)


NODE_CLASS_MAPPINGS = {
    "SemanticMaskToWeightMap": SemanticMaskToWeightMap,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SemanticMaskToWeightMap": "Semantic → Weight Map",
}
