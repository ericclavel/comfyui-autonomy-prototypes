# ComfyUI Autonomy Prototypes

# 

# Prototype nodes and example graphs focused on ComfyUI-driven rapid prototyping for diffusion + perception workflows. Includes small, dependency-light utilities you can drop into existing graphs.

# 

# Core deliverables:

# 

# Custom node(s) under autonomy\_prototypes

# 

# Example inpaint graph (/workflows/sam\_semantic\_conditioning\_inpaint.json)

# 

# Screenshot(s) (/workflows/screenshots/prototype\_graph\_ui\_01.png)

# 

# What’s inside

# ✅ Nodes

# 

# SemanticMaskToWeightMap

# 

# Input: IMAGE tensor \[B,H,W,C] in 0..1 (grayscale semantic ID, or a binary mask).

# 

# Params:

# 

# class\_id (0–255): target semantic ID.

# 

# sigma: optional Gaussian soften for edges (separable, VRAM-friendly).

# 

# invert: invert selection.

# 

# edge\_soften: enable/disable the blur pass.

# 

# Output:

# 

# MASK (0..1) for conditioning/inpaint

# 

# IMAGE preview (3-ch)

# 

# Permanent fix included: builds Gaussian kernel in a float dtype; never inherits bool and won’t hit "arange\_cpu" not implemented for 'Bool'.

# 

# BinaryMaskToWeightMap (minimal)

# 

# Converts a binary MASK/grayscale \[0,1] image to a soft weight map with optional blur and invert.

# 

# Good when you already have a binary mask (e.g., SAM).

# 

# Folder: autonomy\_prototypes/

# Registration via \_\_init\_\_.py (ComfyUI auto-discovers this folder under ComfyUI/custom\_nodes).

# 

# Requirements

# 

# ComfyUI 0.3.27+

# 

# PyTorch 2.5.x (works with the Windows portable build)

# 

# (Optional) SAM nodes (for the example inpaint graph)

# 

# A standard inpaint checkpoint (e.g., SD 1.5/XL inpaint)

# 

# Install

# 

# Clone or copy the repo folder into ComfyUI’s custom\_nodes:

# 

# ComfyUI/

# └─ custom\_nodes/

# &nbsp;  └─ autonomy\_prototypes/

# &nbsp;     ├─ \_\_init\_\_.py

# &nbsp;     ├─ semantic\_mask\_to\_weightmap.py

# &nbsp;     └─ binary\_mask\_to\_weightmap.py

# 

# 

# Restart ComfyUI.

# 

# Confirm you can see:

# 

# Nodes → Autonomy → Conditioning → Semantic → Weight Map

# 

# Nodes → Autonomy → Conditioning → Binary → Weight Map

# 

# Quick start (inpaint with SAM)

# 

# Open the example graph:

# 

# /workflows/sam\_semantic\_conditioning\_inpaint.json

# 

# 

# Wiring in the graph:

# 

# LoadImage → feed your source image.

# 

# SAMModelLoader + SAM Parameters → SAM Image Mask

# 

# Draw a few positive/negative points to isolate a region.

# 

# Route the image (or SAM mask) into:

# 

# Option A (binary SAM mask path): BinaryMaskToWeightMap

# 

# Option B (semantic ID path): SemanticMaskToWeightMap (see tips below)

# 

# Output MASK → VAEEncodeForInpaint.mask, then your model → KSampler, → VAEDecode → Preview.

# 

# Denoise strength matters. Start around 0.5–0.7 for visible edits.

# 

# Tips \& gotchas

# 

# Feeding SAM to SemanticMaskToWeightMap

# 

# SAM’s mask is binary (0 or 1). This node rounds gray\*255, so foreground becomes 255.

# 

# Set class\_id = 255 to select the SAM foreground.

# 

# If you don’t need semantic ID selection, use BinaryMaskToWeightMap instead (simpler).

# 

# Nothing changes in the image

# 

# Check that the MASK → VAEEncodeForInpaint.mask is connected.

# 

# Use a proper inpaint checkpoint or inpaint-capable pipeline.

# 

# Increase denoise strength.

# 

# Try invert if the region being edited is flipped.

# 

# CUDA OOM or slowdowns

# 

# Lower sigma, or disable edge\_soften.

# 

# The blur is separable to be VRAM-friendly; we also fall back to CPU if needed.

# 

# Windows symlink warnings (ComfyLiterals, etc.)

# 

# Not related to these nodes. Copy extension folders manually if your Comfy build can’t create symlinks.

# 

# Repository layout

# .

# ├─ LICENSE

# ├─ README.md

# ├─ autonomy\_prototypes/                  # the actual nodes (drop this folder into ComfyUI/custom\_nodes)

# │  ├─ \_\_init\_\_.py

# │  ├─ binary\_mask\_to\_weightmap.py

# │  └─ semantic\_mask\_to\_weightmap.py

# ├─ examples/                             # (optional) small code samples

# ├─ graphs/                               # (optional) extra graphs

# ├─ samples/                              # input/output example images

# └─ workflows/

# &nbsp;  ├─ sam\_semantic\_conditioning\_inpaint.json

# &nbsp;  └─ screenshots/

# &nbsp;     └─ prototype\_graph\_ui\_01.png

# 

# Roadmap

# 

# Add class-agnostic mode to SemanticMaskToWeightMap (treat non-zero as foreground without requiring class\_id=255).

# 

# Optional edge-only weight mode (distance transform-like mask) for feathered edits.

# 

# A/B harness: simple metric logging (SSIM/LPIPS region-only).

# 

# License

# 

# MIT — see LICENSE.

# 

# Contact

# 

# Eric Clavel

# Technical Artist / TD (Houdini • Unity • ComfyUI)

# Graphs + code in this repo; feel free to open issues or PRs for small fixes.

# 

# Changelog

# 

# 2025-10-30: Initial public prototype. Fixed PyTorch dtype issue on kernel build; added separable Gaussian and CPU fallback.

