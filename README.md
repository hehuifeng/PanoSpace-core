# PanoSpace

High-resolution single-cell insight from low-resolution spatial transcriptomics.

![PanoSpace overview](figures/fig1.png)

PanoSpace bridges the gap between spot-based spatial transcriptomics (e.g. 10x
Visium) and single-cell resolution. It combines histology-guided cell detection,
transcriptomic deconvolution, deep-learning-based super-resolution and
expression prediction to generate consistent cell-level maps across an entire
tissue section.

## Why PanoSpace?

- **Spot-to-cell reconstruction** – detect nuclei from whole-slide images and
  register them with the Visium grid.
- **Ensemble cell-type deconvolution** – run multiple backends (RCTD,
  cell2location, spatialDWLS) and fuse their outputs with the EnDecon ensemble.
- **Image-aware super resolution** – refine cell-type assignments with a
  DINOv2-based contextual classifier and assign labels to segmented cells.
- **Gene expression projection** – diffuse cell-type specific expression from
  spots to detected cells using graph-aware propagation.
- **Modular design** – each step is exposed through ``panospace.tl`` wrappers so
  workflows can be scripted or combined with the broader `scverse` ecosystem.

## Installation

PanoSpace targets **Python 3.9+**. The base installation provides the high-level
API; optional extras pull in heavier dependencies such as PyTorch-based models
or Gurobi.

### Install from PyPI (coming soon)

```bash
pip install panospace
```

### Install from source

```bash
git clone https://github.com/hehuifeng/PanoSpace.git
cd PanoSpace
pip install .
```

Append extras to enable specific backends:

```bash
# install everything required for end-to-end processing
pip install .[cellvit,annotation,prediction]

# or pick only what you need, e.g. CellViT-based detection
pip install .[cellvit]
```

Available extras:

| Extra          | Main dependencies (highlights)                      | Purpose                                      |
| -------------- | --------------------------------------------------- | -------------------------------------------- |
| ``cellvit``    | ``torch``, ``torchvision``, ``opencv-python``, ``ray`` | CellViT nuclei/cell detection backend        |
| ``annotation`` | ``scanpy``, ``anndata``, ``pytorch-lightning``, ``pot`` | Deconvolution, super-resolution, annotation |
| ``prediction`` | ``scipy`` (included via the base requirements)      | Gene expression diffusion utilities          |

> The default installation already pulls these dependencies to keep the full pipeline working out of the box. Extras simply group
> them so that environment managers (e.g. Conda, pip-tools) can reference the relevant feature set explicitly.
> **Gurobi:** ``annotation`` installs hooks for the optional MILP solver. You
> must obtain a license (free for academics) and follow [Gurobi's
> instructions](https://www.gurobi.com/academia/academic-program-and-licenses/)
> to activate it.

### Development environment

A ready-to-use Conda specification is available:

```bash
conda env create -f environment.yml
conda activate PanoSpace
pip install -e .[cellvit,annotation,prediction]
```

## Quick start

```python
import panospace as ps
from PIL import Image

# 1) Detect cells on a whole-slide image
tissue = Image.open("path/to/visium_slide.tif")
seg_adata, contours = ps.detect_cells(tissue, model="cellvit")

# 2) Deconvolve Visium spots using multiple backends and ensemble integration
deconv_adata = ps.deconv_celltype(
    adata_vis=visium_adata,
    sc_adata=reference_sc,
    celltype_key="celltype_major",
    methods=['RCTD', 'spatialDWLS', 'cell2location']
)

# 3) Super-resolve and annotate segmented cells
sr_adata = ps.superres_celltype(
    deconv_adata=deconv_adata,
    img_dir="path/to/visium_slide.tif",
)
annotated_seg = ps.celltype_annotator(
    decov_adata=deconv_adata,
    sr_deconv_adata=sr_adata,
    seg_adata=seg_adata,
)

# 4) Predict cell-level expression profiles
expr = ps.genexp_predictor(
    sc_adata=reference_sc,
    spot_adata=visium_adata,
    infered_adata=annotated_seg,
    celltype_list=list(reference_sc.obs["celltype_major"].unique()),
)
```

Each function returns an ``AnnData`` object (or tuple) ready for downstream
analysis and visualization.

## Repository layout

```
panospace/              # Python package
├── tl/                 # User-facing tools (detect, annotate, predict)
├── _core/              # Backend implementations (CellViT, EnDecon, etc.)
├── _utils/             # Shared helpers
└── _version.py         # Single-source version string

demo/                   # Jupyter notebooks reproducing the paper analyses
figures/                # Figures used in the documentation
tests/                  # Unit tests and regression checks
```

## Reproducibility

The ``demo`` directory contains notebooks to reproduce the main figures in the
paper, including:

- [10x Visium breast cancer dataset](demo/Visium_Breast_Reproducibility.ipynb)
- [10x Visium adult mouse olfactory bulb dataset](demo/Visium_bulb_Reproducibility.ipynb)

## Contributing

We welcome pull requests and bug reports. Please ensure new code is covered by
unit tests and run the test-suite before submitting:

```bash
pytest
```

## Contact

For questions or collaboration opportunities please contact Hui-Feng He
(<huifeng@mails.ccnu.edu.cn>) or Prof. Xiao-Fei Zhang (<zhangxf@ccnu.edu.cn>).
