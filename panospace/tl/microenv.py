"""panospace.tl.microenv
=======================
High-level wrapper for **spatial micro-environment analysis**.  It quantifies
how the neighbourhood of a *sender* cell type influences gene expression in a
*receiver* cell type and can optionally infer ligand-receptor-target (L-R-T)
networks.

All heavy computation is delegated to specialised back-end implementations in
:pydata:`panospace._core.microenv`.  This wrapper only performs:

1. **Input validation** - detected cells + cell-type labels + predicted
   expression must exist.
2. **Parameter normalisation** - convert paths/AnnData into
   :class:`spatialdata.SpatialData`, check that the requested cell types exist
   and that a prediction layer is available.
3. **Backend dispatch** - call the selected algorithm and collect results.
4. **Result storage** - write outputs into ``sdata.uns['panospace/microenv']``
   for downstream plotting and return a typed :class:`MicroEnvResult` when
   requested.

"""
