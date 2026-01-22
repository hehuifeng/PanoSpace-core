"""panospace.tl.microenv
=======================
Microenvironment analysis for spatial transcriptomics.

This module provides tools to analyze cellular interactions and microenvironments
by computing spatial neighborhood features, correlating gene expression with
cell-type abundances, and performing enrichment analysis.

Main Functions
-------------
analyze_interaction
    Analyze gene expression correlations with cell-type abundances in neighborhoods.
compute_environment_features
    Calculate cell-type composition around each cell.
correlation_analysis
    Compute correlation between genes and a target variable.
spatial_enrichment
    Perform GO/KEGG enrichment on gene sets.

Examples
--------
Basic cell-cell interaction analysis:

>>> import panospace as ps
>>> import scanpy as sc
>>>
>>> # Load spatial data with cell-type annotations
>>> adata = sc.read_h5ad("spatial_data.h5ad")
>>>
>>> # Define cell-type pairs to analyze
>>> pairs = [('Cancer_epithelial', 'CAF'), ('T_cell', 'Macrophage')]
>>>
>>> # Analyze interactions
>>> results = ps.analyze_interaction(
>>>     adata,
>>>     cell_type_pairs=pairs,
>>>     cell_type_col='pred_cell_type',
>>>     radius=100.0
>>> )
>>>
>>> # Get results for a specific pair
>>> expr_df, target_abundance, adata_subset = results[('Cancer_epithelial', 'CAF')]

Correlation analysis to find marker genes:

>>> # Find genes correlated with target cell-type abundance
>>> corr_results = ps.correlation_analysis(
>>>     expr_df,
>>>     target_abundance,
>>>     method='pearson'
>>> )
>>>
>>> # Filter significant genes
>>> significant_genes = corr_results.query(
>>>     'correlation > 0.1 & p_adjust < 0.05'
>>> )['gene'].tolist()
>>>
>>> print(f"Found {len(significant_genes)} significant genes")

Gene set enrichment analysis:

>>> if len(significant_genes) > 0:
>>>     # Perform GO enrichment
>>>     go_results = ps.spatial_enrichment(
>>>         gene_list=significant_genes,
>>>         organism='Human',
>>>         gene_sets='GO_Biological_Process_2021',
>>>         cutoff=0.05
>>>     )
>>>
>>>     # Save results
>>>     go_results.to_csv('go_enrichment.csv', index=False)

Compute neighborhood features manually:

>>> # Calculate cell-type composition in spatial neighborhoods
>>> adata, features, cell_types = ps.compute_environment_features(
>>>     adata,
>>>     cell_type_col='pred_cell_type',
>>>     radius=100.0
>>> )
>>>
>>> print(f"Features shape: {features.shape}")  # (n_cells, n_cell_types)
>>> print(f"Cell types: {cell_types}")

Complete workflow example:

>>> def analyze_communication(adata, source_type, target_type, radius=100):
>>>     '''Analyze communication between two cell types.'''
>>>
>>>     # 1. Analyze interaction
>>>     results = ps.analyze_interaction(
>>>         adata,
>>>         [(source_type, target_type)],
>>>         radius=radius
>>>     )
>>>     expr, target, adata_sub = results[(source_type, target_type)]
>>>
>>>     # 2. Find correlated genes
>>>     corr = ps.correlation_analysis(expr, target)
>>>     top_genes = corr.query('correlation > 0.1 & p_adjust < 0.05')['gene'].tolist()
>>>
>>>     if len(top_genes) == 0:
>>>         print("No significant genes found")
>>>         return None
>>>
>>>     # 3. Calculate interaction score
>>>     import numpy as np
>>>     adata_sub.obs['interaction_score'] = np.mean(
>>>         adata_sub[:, top_genes].X, axis=1
>>>     )
>>>
>>>     # 4. Enrichment analysis
>>>     go_results = ps.spatial_enrichment(top_genes, organism='Human')
>>>
>>>     return {
>>>         'correlation': corr,
>>>         'genes': top_genes,
>>>         'enrichment': go_results,
>>>         'adata': adata_sub
>>>     }
>>>
>>> # Run analysis
>>> results = analyze_communication(adata, 'Cancer_epithelial', 'CAF', radius=100)
"""

from __future__ import annotations

import logging
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, spearmanr, hypergeom
from statsmodels.stats.multitest import multipletests

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Environment Feature Computation
# -----------------------------------------------------------------------------

def compute_environment_features(
    adata: sc.AnnData,
    cell_type_col: str = 'pred_cell_type',
    radius: float = 100.0,
) -> Tuple[sc.AnnData, np.ndarray, List[str]]:
    """Calculate cell-type composition in spatial neighborhoods.

    For each cell, counts the number of cells of each type within a given radius.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with spatial coordinates in obsm['spatial']
        and cell-type labels in obs[cell_type_col].
    cell_type_col : str, default='pred_cell_type'
        Column in adata.obs containing cell-type labels.
    radius : float, default=100.0
        Spatial radius for neighborhood definition.

    Returns
    -------
    adata : AnnData
        Input AnnData with added cell-type indicator columns.
    features : np.ndarray
        Cell-type composition matrix (n_cells x n_types), where each row
        contains counts of each cell type in the neighborhood.
    cell_types : list of str
        Sorted list of unique cell types.

    Examples
    --------
    >>> import panospace as ps
    >>> adata = ps.tl.microenv.compute_environment_features(
    ...     adata, cell_type_col='celltype', radius=100
    ... )
    """
    coordinates = adata.obsm['spatial']
    kdtree = cKDTree(coordinates)

    cell_types = adata.obs[cell_type_col].unique().tolist()
    cell_types = sorted(cell_types)

    # Add binary columns for each cell type
    for ct in cell_types:
        adata.obs[ct] = (adata.obs[cell_type_col] == ct).astype(float)

    # Compute neighborhood composition
    logger.info(f"Computing environment features for {len(coordinates)} cells...")
    features = []
    for point in coordinates:
        indices = kdtree.query_ball_point(point, radius)
        if len(indices) == 0:
            features.append(np.zeros(len(cell_types)))
        else:
            features.append(np.sum(adata[indices].obs[cell_types].values, axis=0))

    logger.info(f"Computed features for {len(features)} cells with {len(cell_types)} cell types")
    return adata, np.vstack(features), cell_types


# -----------------------------------------------------------------------------
# Gene Expression Analysis
# -----------------------------------------------------------------------------

def detect_high_expressed_genes(
    expr_df: pd.DataFrame,
    threshold: float = 3.0,
) -> List[str]:
    """Identify highly expressed genes from expression matrix.

    Parameters
    ----------
    expr_df : pd.DataFrame
        Gene expression dataframe (genes x samples) in log1p space.
    threshold : float, default=3.0
        Log-expression threshold for defining high expression.

    Returns
    -------
    list of str
        Genes with mean log-expression above threshold.
    """
    # Convert from log1p to linear space and scale
    expr_linear = 100 * (np.exp(expr_df.values) - 1) if hasattr(expr_df, 'values') else 100 * (np.exp(expr_df) - 1)
    log_mean_expr = np.log1p(np.mean(expr_linear, axis=1))
    high_expr_genes = log_mean_expr[log_mean_expr >= threshold]

    if hasattr(high_expr_genes, 'index'):
        return high_expr_genes.index.tolist()
    return expr_df.index[log_mean_expr >= threshold].tolist()


def correlation_analysis(
    expr_df: pd.DataFrame,
    target_vector: np.ndarray,
    method: Literal['pearson', 'spearman'] = 'pearson',
) -> pd.DataFrame:
    """Correlate gene expression with a target variable.

    Computes correlation between each gene's expression profile and a target
    variable (e.g., cell-type abundance), with multiple testing correction.

    Parameters
    ----------
    expr_df : pd.DataFrame
        Gene expression matrix (cells x genes).
    target_vector : np.ndarray
        Target variable vector (e.g., cell-type abundance).
    method : {'pearson', 'spearman'}, default='pearson'
        Correlation method to use.

    Returns
    -------
    pd.DataFrame
        Results dataframe with columns:
        - gene: gene name
        - correlation: correlation coefficient
        - p_value: raw p-value
        - p_adjust: FDR-adjusted p-value (Benjamini-Hochberg)
        Sorted by correlation in descending order.
    """
    results = []
    logger.info(f"Performing {method} correlation for {len(expr_df.columns)} genes...")

    for gene in expr_df.columns:
        if method == 'pearson':
            corr, pval = pearsonr(expr_df[gene], target_vector)
        else:
            corr, pval = spearmanr(expr_df[gene], target_vector)
        results.append((gene, corr, pval))

    results_df = pd.DataFrame(results, columns=['gene', 'correlation', 'p_value'])
    results_df['p_adjust'] = multipletests(results_df['p_value'], method='fdr_bh')[1]
    results_df.sort_values(by='correlation', ascending=False, inplace=True)

    logger.info(f"Correlation analysis completed. Top gene: {results_df.iloc[0]['gene']} "
               f"(corr={results_df.iloc[0]['correlation']:.3f})")
    return results_df


# -----------------------------------------------------------------------------
# Cell-Cell Interaction Analysis
# -----------------------------------------------------------------------------

def analyze_interaction(
    adata: sc.AnnData,
    cell_type_pairs: List[Tuple[str, str]],
    cell_type_col: str = 'pred_cell_type',
    radius: float = 100.0,
) -> dict:
    """Analyze interactions between pairs of cell types.

    For each pair, identifies genes in the source cell type whose expression
    correlates with the abundance of the target cell type in its neighborhood.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with spatial coordinates and cell types.
    cell_type_pairs : list of tuple
        List of (source_type, target_type) pairs to analyze.
        The source type is the cell whose genes are tested;
        the target type is the neighborhood composition analyzed.
    cell_type_col : str, default='pred_cell_type'
        Column in adata.obs containing cell-type labels.
    radius : float, default=100.0
        Spatial radius for neighborhood definition.

    Returns
    -------
    dict
        Nested dictionary with keys (source_type, target_type).
        Each value is a tuple of (expression_df, target_abundance, adata_subset):
        - expression_df: pd.DataFrame of highly expressed genes in source cells
        - target_abundance: np.ndarray of target cell-type abundance in neighborhoods
        - adata_subset: AnnData object restricted to source cells

    Examples
    --------
    >>> import panospace as ps
    >>> pairs = [('Cancer', 'CAF'), ('T_cell', 'Macrophage')]
    >>> results = ps.tl.microenv.analyze_interaction(
    ...     adata, pairs, radius=100
    ... )
    >>> expr, target, adata_cancer = results[('Cancer', 'CAF')]
    """
    adata, envirfea, cell_types = compute_environment_features(
        adata, cell_type_col, radius
    )

    analyze_dict = {}
    logger.info(f"Analyzing {len(cell_type_pairs)} cell-type pairs...")

    for source_celltype, target_celltype in cell_type_pairs:
        logger.info(f"Processing: {source_celltype} -> {target_celltype}")

        # Get source cells
        source_mask = adata.obs[cell_type_col] == source_celltype
        adata_source = adata[source_mask]

        # Extract expression matrix
        if sc.sparse.issparse(adata_source.X):
            expr = pd.DataFrame(
                adata_source.X.todense(),
                columns=adata_source.var_names,
                index=adata_source.obs_names
            )
        else:
            expr = pd.DataFrame(
                adata_source.X,
                columns=adata_source.var_names,
                index=adata_source.obs_names
            )

        # Find highly expressed genes in source cells
        index = detect_high_expressed_genes(expr.T, threshold=3)
        expr = expr[index]

        # Get target cell-type abundance in neighborhoods
        target_idx = list(cell_types).index(target_celltype)
        target_proportion = envirfea[source_mask, target_idx]

        analyze_dict[(source_celltype, target_celltype)] = [
            expr, target_proportion, adata_source
        ]

        logger.info(f"  Found {len(index)} highly expressed genes in {source_celltype}")

    return analyze_dict


# -----------------------------------------------------------------------------
# Enrichment Analysis
# -----------------------------------------------------------------------------

def spatial_enrichment(
    gene_list: List[str],
    background_genes: Optional[List[str]] = None,
    organism: Literal['Human', 'Mouse', 'Yeast', 'Fly', 'Fish', 'Worm'] = 'Human',
    gene_sets: str = 'GO_Biological_Process_2021',
    cutoff: float = 0.05,
    outdir: Optional[str] = None,
) -> pd.DataFrame:
    """Perform gene set enrichment analysis.

    Uses gseapy's Enrichr to perform over-representation analysis.

    Parameters
    ----------
    gene_list : list of str
        Gene list to test for enrichment.
    background_genes : list of str, optional
        Background gene universe. If None, uses all genes in Enrichr.
    organism : {'Human', 'Mouse', 'Yeast', 'Fly', 'Fish', 'Worm'}, default='Human'
        Organism for enrichment analysis.
    gene_sets : str, default='GO_Biological_Process_2021'
        Gene set library to use.
    cutoff : float, default=0.05
        P-value cutoff for significant terms.
    outdir : str, optional
        Directory to save Enrichr results. If None, results are not saved to disk.

    Returns
    -------
    pd.DataFrame
        Enrichment results with columns including Term, P-value, Combined Score, etc.

    Notes
    -----
    Requires gseapy to be installed: ``pip install gseapy``
    """
    try:
        import gseapy as gp
    except ImportError:
        raise ImportError(
            "gseapy is required for enrichment analysis. "
            "Install it with: pip install gseapy"
        )

    logger.info(f"Running enrichment analysis for {len(gene_list)} genes...")

    enr = gp.enrichr(
        gene_list=gene_list,
        background=background_genes,
        organism=organism,
        gene_sets=gene_sets,
        outdir=outdir,
        cutoff=cutoff
    )

    n_significant = (enr.res2d['P-value'] < cutoff).sum() if len(enr.res2d) > 0 else 0
    logger.info(f"Enrichment completed. Found {n_significant} significant terms at p<{cutoff}")

    return enr.res2d


def test_gene_overlap(
    gene_universe: List[str],
    gene_set1: List[str],
    gene_set2: List[str],
) -> Tuple[float, int]:
    """Test overlap between two gene sets using hypergeometric test.

    Parameters
    ----------
    gene_universe : list of str
        All genes in the background universe.
    gene_set1 : list of str
        First gene set.
    gene_set2 : list of str
        Second gene set.

    Returns
    -------
    p_value : float
        Right-tailed p-value from hypergeometric test.
    overlap_size : int
        Number of overlapping genes.

    Examples
    --------
    >>> import panospace as ps
    >>> pval, overlap = ps.tl.microenv.test_gene_overlap(
    ...     all_genes, gene_list_a, gene_list_b
    ... )
    """
    N = len(gene_universe)
    K = len(gene_set1)
    M = len(gene_set2)
    x = len(set(gene_set1) & set(gene_set2))

    pval = hypergeom.sf(x - 1, N, K, M)

    logger.info(f"Gene overlap test: {x}/{min(K, M)} genes overlap (p={pval:.4e})")

    return pval, x


# -----------------------------------------------------------------------------
# Backward Compatibility Aliases
# -----------------------------------------------------------------------------

# Alias for compatibility with original code
def umap_adata(
    adata: sc.AnnData,
    n_neighbors: int = 150,
    cell_type_col: str = 'pred_cell_type',
    plot: bool = False,
    save_dir: Optional[str] = None,
) -> sc.AnnData:
    """Perform UMAP dimensionality reduction and optional plotting.

    .. deprecated::
        This function is kept for backward compatibility.
        Consider using scanpy's direct functions instead.

    Notes
    -----
    Parameters ``cell_type_col``, ``plot``, and ``save_dir`` are accepted for
    compatibility but not used in this simplified version.
    """
    _ = cell_type_col, plot, save_dir  # Mark as intentionally unused
    logger.warning("umap_adata is deprecated. Use scanpy directly.")
    sc.pp.calculate_qc_metrics(adata, inplace=True, log1p=True)
    sc.tl.pca(adata)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)
    sc.tl.umap(adata)
    return adata


__all__ = [
    "analyze_interaction",
    "compute_environment_features",
    "correlation_analysis",
    "detect_high_expressed_genes",
    "spatial_enrichment",
    "test_gene_overlap",
]
