# PanoSpace

**High-resolution single-cell insight from low-resolution spatial transcriptomics**

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

### Installation

**Option 1: Automatic (Recommended)**
```bash
git clone https://github.com/hehuifeng/PanoSpace.git
cd PanoSpace
bash install.sh
```
The script will automatically detect your GPU and install the GPU-enabled version.

**Option 2: Manual**
```bash
conda env create -f environment-gpu.yml
conda activate PanoSpace
pip install -e .
```

**Verify Installation**
```bash
python -c "import panospace as ps; print('PanoSpace installed successfully!')"
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

Both solvers implement the **same mathematical model** with:
- Global cell-type quotas
- Spot-level quota constraints (ensures consistency within each spot)
- Exact 0/1 assignment (no approximation)

**Installation:**

**SCIP** (installed by default):
```bash
# Already included in environment.yml
conda activate PanoSpace
```

**Gurobi** (optional, recommended for better performance):
```bash
# Install Gurobi
conda install -c conda-forge gurobipy

# Request free academic license at: https://www.gurobi.com/academia/academic-program-and-licenses/
# Follow Gurobi's instructions to activate the license

# Verify installation
python -c "import gurobipy; print('Gurobi installed successfully!')"
```

*Note: Based on our experience, Gurobi typically solves problems in under 1 minute, while SCIP may take hundreds of minutes for the same problem.*



</details>

## 🚀 Quick Start

### Basic Workflow

```python
import panospace as ps
from PIL import Image

# 1. Detect cells from tissue image
tissue = Image.open("path/to/visium_slide.tif")
seg_adata, contours = ps.detect_cells(tissue, model="cellvit", gpu=True)

# 2. Deconvolve Visium spots
#    visium_adata: AnnData with .X (expression) and .obsm['spatial'] (coordinates)
#    sc_reference: AnnData with .X and .obs[celltype_key] (cell type labels)
deconv_adata = ps.deconv_celltype(
    adata_vis=visium_adata,
    sc_adata=sc_reference,
    celltype_key="celltype_major",  # Column name in sc_reference.obs
    methods=['RCTD', 'spatialDWLS', 'cell2location']
)

# 3. Super-resolve to cell level
sr_adata = ps.superres_celltype(
    deconv_adata=deconv_adata,
    img_dir="path/to/visium_slide.tif"
)

# 4. Annotate segmented cells
annotated_adata = ps.celltype_annotator(
    decov_adata=visium_adata,
    sr_deconv_adata=sr_adata,
    seg_adata=seg_adata
)

# 5. Predict gene expression
pred_adata = ps.genexp_predictor(
    sc_adata=sc_reference,
    spot_adata=visium_adata,
    infered_adata=annotated_adata,
    celltype_list=list(sc_reference.obs["celltype_major"].unique())
)
```


### Cell-Cell Interaction Analysis

```python
# Analyze interactions between cell pairs
pairs = [('Cancer_epithelial', 'CAF'), ('T_cell', 'Macrophage')]
results = ps.analyze_interaction(
    adata=annotated_adata,
    cell_type_pairs=pairs,
    cell_type_col='pred_cell_type',  # Column in adata.obs
    radius=100.0  # Neighborhood radius (same units as spatial coordinates)
)

# Extract results and find correlated genes
expr_df, target_abundance, _ = results[('Cancer_epithelial', 'CAF')]
corr_results = ps.correlation_analysis(expr_df, target_abundance)
significant_genes = corr_results.query('p_adjust < 0.05')['gene'].tolist()

# Functional enrichment
if len(significant_genes) > 0:
    go_results = ps.spatial_enrichment(
        gene_list=significant_genes,
        organism='Human',
        gene_sets='GO_Biological_Process_2021'
    )
```

### Data Requirements

**Visium Data** (`visium_adata`)
- AnnData object with `.X` (gene expression) and `.obsm['spatial']` (coordinates)

**Single-Cell Reference** (`sc_reference`)
- AnnData object with `.X` and `.obs[celltype_key]` (cell type labels)
- Minimum 100 cells per type, genes should overlap with Visium data

**Histology Image**
- TIFF/PNG/JPEG format, 40x+ magnification recommended



## 📖 Citation

If you use PanoSpace in your research, please cite:

He, HF., Peng, P., Yang, ST. et al. Unlocking single-cell level and continuous whole-slide insights in spatial transcriptomics with PanoSpace. *Nat Comput Sci* (2026). https://doi.org/10.1038/s43588-025-00938-y


## 📧 Contact

For questions or collaboration opportunities:

- **Hui-Feng He** (<huifeng@mails.ccnu.edu.cn>)
- **Xiao-Fei Zhang** (<zhangxf@ccnu.edu.cn>)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.


---

**Note:** PanoSpace is actively under development. API changes may occur between
versions. Please check the changelog when upgrading.
