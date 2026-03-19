# PanoSpace

**High-resolution single-cell insight from low-resolution spatial transcriptomics**

<!-- PyPI  -->
[![PyPI Version](https://img.shields.io/pypi/v/panospace.svg?color=blue&label=panospace)](https://pypi.org/project/panospace/)


![PanoSpace overview](figures/fig1.png)

PanoSpace bridges the gap between spot-based spatial transcriptomics (e.g., 10x
Visium) and single-cell resolution. It combines histology-guided cell detection,
transcriptomic deconvolution, deep-learning-based super-resolution, expression
prediction, and microenvironment analysis to generate consistent cell-level maps
across entire tissue sections.


## 📦 Installation

### System Requirements

- **OS**: Linux (strongly recommended)
- **GPU**: NVIDIA GPU with CUDA support (strongly recommended for performance)
  - CUDA 12.1+ recommended
  - Minimum 8GB GPU memory

### Installation

**Option 1: Install from PyPI (Recommended for Users)**

```bash
# Step 1: Create a conda environment
conda create -n panospace python=3.11
conda activate panospace

# Step 2: Install PanoSpace with all dependencies
pip install panospace[all]

# Step 3: Install PyTorch with CUDA support (GPU version, recommended)
pip install --extra-index-url https://download.pytorch.org/whl/cu121 torch>=2.1 torchvision>=0.15

# For CPU-only version:
# pip install torch>=2.1 torchvision>=0.15
```

**Option 2: Install from Source (Automatic Setup)**

```bash
git clone https://github.com/hehuifeng/PanoSpace-core.git
cd PanoSpace-core
bash install.sh
```

The script will automatically:
- Create conda environment with all dependencies (except PyTorch)
- Detect your GPU
- Install PyTorch via pip (with CUDA if GPU detected)
- Install PanoSpace package
- Verify the installation

**Option 3: Manual Installation from Source**

For **GPU version** (recommended):
```bash
# Step 1: Create conda environment
conda env create -f environment-gpu.yml
conda activate PanoSpace

# Step 2: Install PyTorch with CUDA support
pip install --extra-index-url https://download.pytorch.org/whl/cu121 'torch>=2.1' 'torchvision>=0.15'

# Step 3: Install PanoSpace
pip install .
```

<details>
<summary><b>Optional: Optimization Solvers (Click to expand)</b></summary>

#### Optimization Solvers for Cell Annotation

PanoSpace uses **Mixed Integer Linear Programming (MILP)** solvers for accurate cell-type annotation with spot-level quota constraints. Two solvers are supported:

**Supported Solvers:**

1. **Gurobi** (Recommended, Commercial but Free for Academia)
   - Significantly faster (10-100x speedup on large datasets)
   - Best for production use and large-scale analyses
   - Free academic license available at: https://www.gurobi.com/academia/academic-program-and-licenses/

2. **SCIP** (Open-Source, Default)
   - Automatically installed with PanoSpace
   - Produces mathematically identical results to Gurobi
   - Suitable for small to medium datasets
   - No additional setup required

**Solver Selection Logic:**

PanoSpace automatically selects the best available solver:
- If **Gurobi is installed** → Uses Gurobi (fastest)
- If **Gurobi is not available** → Uses SCIP (open-source fallback)

Both solvers implement the **same mathematical model**.

*Note: Based on our experience, Gurobi typically solves problems in under 1 minute, while SCIP may take hundreds of minutes for the same problem.*



</details>

## 🚀 Quick Start

### Basic Workflow

```python
import os
import panospace as ps
from PIL import Image

# ==============================================================================
# Step 1: Cell Detection from Histology Image
# ==============================================================================
# Load high-resolution tissue image (TIFF/PNG/JPEG format, 40x+ magnification recommended)
tissue = Image.open("path/to/visium_slide.tif")

# Perform cell segmentation using deep learning (CellViT model)
# Returns:
#   - seg_adata: AnnData object with cell segmentation results
#   - contours: Cell boundary contours for visualization
seg_adata, contours = ps.detect_cells(
    tissue,
    model="cellvit",  # Pre-trained deep learning model for cell detection
    gpu=True          # Use GPU acceleration (requires CUDA-compatible GPU)
)


# ==============================================================================
# Step 2: Cell Type Deconvolution of Visium Spots
# ==============================================================================
# Input requirements:
#   - visium_adata: AnnData with spatial transcriptomics data
#     * .X: Gene expression matrix (dense)
#     * .obsm['spatial']: Spatial coordinates of Visium spots
#   - sc_reference: Single-cell reference AnnData
#     * .X: Gene expression matrix (sparse)
#     * .obs[celltype_key]: Cell type annotations for each cell
deconv_adata = ps.deconv_celltype(
    adata_vis=visium_adata,
    sc_adata=sc_reference,
    celltype_key="celltype_major",  # Column name in sc_reference.obs containing cell type labels
    methods=['RCTD', 'spatialDWLS', 'cell2location'],  # Ensemble of deconvolution methods
    cache_dir=os.path.join(OUTPUT_DIR, 'deconv_cache'),  # Cache directory for intermediate results
    project_name='simulation_data',                      # Project identifier for caching
    resume=True,                  # Resume from cached results if available
    continue_on_error=False,      # Stop execution if any method fails
    require_nonnegative=False     # Allow negative values in deconvolution results
)


# ==============================================================================
# Step 3: Super-Resolution
# ==============================================================================
# Transform spot-level deconvolution results to whole slides
# using spatial information and histology image features
sr_adata = ps.superres_celltype(
    deconv_adata=deconv_adata,     # Output from Step 2
    img_dir="path/to/visium_slide.tif"  # Path to histology image for spatial guidance
)


# ==============================================================================
# Step 4: Cell Type Annotation for Segmented Cells
# ==============================================================================
# Assign cell types to segmented cells using:
# - Spot-level quota constraints from deconvolution results
# - Super-resolved cell type probabilities
# - MILP optimization (SCIP or Gurobi solver)
deconv_adata.uns['radius'] = 100  # Set spot radius (in pixels) for spatial transcriptomics technology

annotated_adata = ps.celltype_annotator(
    decov_adata=deconv_adata,    # Original Visium data with spot-level deconvolution results
    alpha = 0.3,
    sr_deconv_adata=sr_adata,    # Super-resolved cell type probabilities
    seg_adata=seg_adata          # Segmented cells from Step 1
)


# ==============================================================================
# Step 5: Gene Expression Prediction at Single-Cell Level
# ==============================================================================
# Predict complete gene expression profiles for each annotated cell
# using single-cell reference and spatial context
pred_adata = ps.genexp_predictor(
    sc_adata=sc_reference,              # Single-cell reference with complete transcriptome (can be same as deconvolution)
    spot_adata=deconv_adata,            # Visium spot data for spatial context
    infered_adata=annotated_adata,      # Annotated cells from Step 4
    celltype_list=list(sc_reference.obs["celltype_major"].unique())  # All cell types to predict
)
```


### Cell-Cell Interaction Analysis

```python
# ==============================================================================
# Cell-Cell Interaction Analysis
# ==============================================================================
# Define cell type pairs for interaction analysis
# Format: [(source_cell_type, target_cell_type), ...]
pairs = [
    ('Cancer_epithelial', 'CAF'),      # Cancer epithelial cells → Cancer-associated fibroblasts
    ('T_cell', 'Macrophage')           # T cells → Macrophages
]

# Analyze ligand-receptor mediated interactions between cell type pairs
# Returns: Dictionary with (source, target) tuples as keys
results = ps.analyze_interaction(
    adata=annotated_adata,
    cell_type_pairs=pairs,
    cell_type_col='pred_cell_type',  # Column in adata.obs containing cell type annotations
    radius=100.0                      # Neighborhood radius for spatial interaction (in pixels)
                                     # Interactions are counted for cells within this distance
)

# ==============================================================================
# Extract Interaction Results and Perform Correlation Analysis
# ==============================================================================
# Extract results for a specific cell pair: Cancer_epithelial → CAF
# Returns:
#   - expr_df: DataFrame of ligand/receptor gene expression in source cells
#   - target_abundance: Array of target cell abundance in each neighborhood
#   - metadata: Additional information about the interaction
expr_df, target_abundance, metadata = results[('Cancer_epithelial', 'CAF')]

# Perform correlation analysis between gene expression and target cell abundance
# Identifies potential signaling molecules driving the interaction
corr_results = ps.correlation_analysis(
    expr_df,              # Gene expression matrix (ligands/receptors)
    target_abundance      # Target cell abundance across neighborhoods
)

# Extract statistically significant genes (adjusted p-value < 0.05)
significant_genes = corr_results.query('p_adjust < 0.05')['gene'].tolist()


# ==============================================================================
# Functional Enrichment Analysis
# ==============================================================================
# Perform Gene Ontology enrichment on significant genes
# Identifies biological processes associated with cell-cell interactions
if len(significant_genes) > 0:
    go_results = ps.spatial_enrichment(
        gene_list=significant_genes,
        organism='Human',                           # Organism name for gene annotation
        gene_sets='GO_Biological_Process_2021'     # Gene set database for enrichment
    )
```

### Data Requirements

**Visium Spatial Transcriptomics Data** (`visium_adata`)
- **Format**: AnnData object
- **Required fields**:
  - `.X`: Gene expression matrix (counts values)
    - Shape: `(n_spots, n_genes)`
    - Dense or sparse matrix format supported
  - `.obsm['spatial']`: Spatial coordinates of Visium spots
    - Shape: `(n_spots, 2)`

**Single-Cell Reference Data** (`sc_reference`)
- **Format**: AnnData object
- **Required fields**:
  - `.X`: Gene expression matrix (counts values)
    - Shape: `(n_cells, n_genes)`
    - Dense or sparse matrix format supported
  - `.obs[celltype_key]`: Cell type annotations for each cell
    - Categorical or string dtype

**Histology Image**
- **Supported formats**: TIFF, PNG, JPEG
- **Magnification**: 20x or 40x 



## 📖 Citation

If you use PanoSpace in your research, please cite:

He, HF., Peng, P., Yang, ST. et al. Unlocking single-cell level and continuous whole-slide insights in spatial transcriptomics with PanoSpace. *Nat Comput Sci* (2026). https://doi.org/10.1038/s43588-025-00938-y


## 📧 Contact

- **Hui-Feng He** (<huifeng@mails.ccnu.edu.cn>)
- **Xiao-Fei Zhang** (<zhangxf@ccnu.edu.cn>)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.


---

**Note:** PanoSpace is actively under development. API changes may occur between
versions. Please check the changelog when upgrading.
