# \# ComfyUI Autonomy Prototypes

# 

# Prototype nodes and example graphs focused on ComfyUI-driven rapid prototyping for diffusion + perception workflows. Includes small, dependency-light utilities you can drop into existing graphs.

# 

# ---

# 

# \## Core deliverables

# 

# \- Custom node(s) under `autonomy\_prototypes/`

# \- Example inpaint graph: `workflows/sam\_semantic\_conditioning\_inpaint.json`

# \- Screenshot(s): `workflows/screenshots/prototype\_graph\_ui\_01.png`

# 

# ---

# 

# \## What’s inside

# 

# \### ✅ Nodes

# 

# \#### `SemanticMaskToWeightMap`

# \*\*Input:\*\* `IMAGE` tensor `\[B,H,W,C]` in `0..1` (either a grayscale semantic-ID image or a binary mask).  

# \*\*Params:\*\*

# \- `class\_id` (0–255): target semantic ID.

# \- `sigma`: optional Gaussian soften for edges (separable, VRAM-friendly).

# \- `invert`: invert selection.

# \- `edge\_soften`: enable/disable the blur pass.

# 

# \*\*Output:\*\*  

# \- `MASK` (0..1) for conditioning/inpaint  

# \- `IMAGE` preview (3-ch)

# 

# \*\*Permanent fix:\*\* Gaussian kernels are built in a float dtype (never bool); avoids the `"arange\_cpu" not implemented for 'Bool'` crash.

# 

# \#### `BinaryMaskToWeightMap` (minimal)

# Converts a binary `MASK`/grayscale `\[0,1]` image to a soft weight map with optional blur and invert.  

# Use this when you already have a binary mask (e.g., from SAM).

# 

# \*\*Folder:\*\* `autonomy\_prototypes/`  

# Registration via `\_\_init\_\_.py` (ComfyUI auto-discovers this folder under `ComfyUI/custom\_nodes`).

# 

# ---

# 

# \## Requirements (tested)

# 

# \- ComfyUI `0.3.27+`

# \- PyTorch `2.5.x` (Windows portable build OK)

# \- (Optional) SAM nodes (for the example inpaint graph)

# \- An inpaint-capable checkpoint (e.g., SDXL/SD1.5 inpaint variants)

# 

# \*\*Models used in the example graph (swap for equivalents if needed):\*\*

# \- Checkpoint: `inpaintingProductDesign\_v10.safetensors`

# \- LoRA (optional texture bias): `SDXL\_Grass\_texture\_Sa\_May.safetensors`

# \- IP-Adapter: `ip-adapter\_sdxl\_vit-h.safetensors` + Unified Loader (`VIT-G` preset)

# \- SAM: `sam\_vit\_h` (2.56 GB variant)

# 

# ---

# 

# \## Install

# 

# 1\. Clone or copy this repo into ComfyUI’s `custom\_nodes`:

# ComfyUI/

# └─ custom\_nodes/

# └─ autonomy\_prototypes/

# ├─ init.py

# ├─ semantic\_mask\_to\_weightmap.py

# └─ binary\_mask\_to\_weightmap.py

# 

# markdown

# Copy code

# 2\. Restart ComfyUI.

# 3\. Confirm you can see:

# \- `Nodes → Autonomy → Conditioning → Semantic → Weight Map`

# \- `Nodes → Autonomy → Conditioning → Binary → Weight Map`

# 

# ---

# 

# \## Reproduce the demo in ~60 seconds

# 

# 1\. Open `workflows/sam\_semantic\_conditioning\_inpaint.json`.

# 2\. In the canvas, set the \*\*LoadImage\*\* to your source (e.g., a street scene).

# 3\. Ensure \*\*CheckpointLoaderSimple\*\* points to an \*\*inpaint\*\* checkpoint.

# 4\. With \*\*SAM Parameters\*\* + \*\*SAM Image Mask\*\*, click a few positive points on the region to change (e.g., the road). Add a negative point to exclude obvious distractors if needed.

# 5\. Choose one path:

# \- \*\*Binary path (recommended with SAM):\*\* Connect SAM’s `MASK` → `BinaryMaskToWeightMap` → `VAEEncodeForInpaint.mask`.

# \- \*\*Semantic-ID path:\*\* Feed SAM’s \*\*IMAGE\*\* (binary 0/1 in first channel) → `SemanticMaskToWeightMap` and set `class\_id = 255` (SAM foreground). Output `MASK` → `VAEEncodeForInpaint.mask`.

# 6\. Set \*\*denoise\*\* in `KSampler` to `0.5–0.7` to force visible edits.

# 7\. Optional: Enable IP-Adapter with a grass reference image to bias texture.

# 8\. Run. Preview nodes should show the masked inpaint and decoded output.

# 

# ---

# 

# \## Tips \& gotchas

# 

# \- \*\*Feeding SAM to `SemanticMaskToWeightMap`\*\*  

# SAM’s mask is binary (0/1). This node rounds `gray\*255`, so foreground becomes `255`.  

# ➤ Set `class\_id = 255` to select SAM’s foreground.  

# If you don’t need class selection, prefer `BinaryMaskToWeightMap` (simpler and faster).

# 

# \- \*\*“Nothing changes” troubleshooting\*\*  

# \- Ensure `MASK` → `VAEEncodeForInpaint.mask` is connected.

# \- Use an \*\*inpaint\*\* checkpoint/pipeline.

# \- Increase denoise to `~0.6–0.8`.

# \- Try `invert = True` if edits apply to the wrong side of the mask.

# \- If edges look harsh, set `sigma = 1.5–3.0` with `edge\_soften = True`.

# 

# \- \*\*VRAM / stability\*\*  

# \- Lower `sigma` or disable `edge\_soften`.

# \- The blur is separable (two 1D passes) and has a CPU fallback on OOM.

# 

# \- \*\*SAM points shape errors\*\*  

# Provide equal counts of positive/negative points (or leave negatives empty). Mixed lengths can trigger tensor-concat size errors in some SAM node builds.

# 

# ---

# 

# \## Repository layout

# 

# .

# ├─ LICENSE

# ├─ README.md

# ├─ autonomy\_prototypes/ # the actual nodes (drop this folder into ComfyUI/custom\_nodes)

# │ ├─ init.py

# │ ├─ binary\_mask\_to\_weightmap.py

# │ └─ semantic\_mask\_to\_weightmap.py

# ├─ examples/ # (optional) small code samples

# ├─ graphs/ # (optional) extra graphs

# ├─ samples/ # input/output example images (add your reproducible assets here)

# └─ workflows/

# ├─ sam\_semantic\_conditioning\_inpaint.json

# └─ screenshots/

# └─ prototype\_graph\_ui\_01.png

# 

# yaml

# Copy code

# 

# > If you share the repo publicly, consider adding a few \*\*sample inputs\*\* (licensed images) and \*\*one output\*\* for reviewers to reproduce the run exactly.

# 

# ---

# 

# \## Known issues / roadmap

# 

# \- Add \*\*class-agnostic\*\* mode to `SemanticMaskToWeightMap` (treat any non-zero as foreground; no `class\_id=255` required).

# \- Optional \*\*edge-only\*\* weighting (distance-like falloff) for cleaner feathered edits.

# \- A/B harness: simple \*\*region-only\*\* SSIM/LPIPS logging, seeded KSampler for reproducibility.

# 

# ---

# 

# \## License

# 

# MIT — see `LICENSE`.  

# If you include sample images, ensure you have the right to redistribute them and specify the license/source in a `samples/README.md`.

# 

# ---

# 

# \## Contact

# 

# Eric Clavel  

# Technical Artist / TD (Houdini • Unity • ComfyUI)  

# Graphs + code in this repo; feel free to open issues or PRs for small fixes.

