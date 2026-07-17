# Manuscript figure scripts

These scripts reproduce the figures and supporting figure statistics used in the ESSD manuscript. Set `GSIM_PLUS_PROJECT_DIR` to the workspace containing the intermediate validation outputs and released GSIM-PLUS products before running them.

## Main-text figure mapping

| Manuscript figure | Script |
|---|---|
| Fig. 1 | `Fig1_global_station_overview.py` |
| Figs. 2-3 | `GRDC_CrossValidation/plot_GRDC_final.py` |
| Fig. 4 | `plot_taylor_professional.py` |
| Fig. 5 | `Fig8_koppen_boxplot.py` |
| Fig. 6 | `plot_lowflow_boundary_figures.py` |
| Fig. 7 | `Fig9d_dual_cdf.py` |
| Fig. 8 | `Fig9_product_quality.py` |
| Fig. 9 | `Fig10_timeseries.py` |

Run `build_essd_submission_figures.py` to regenerate and assemble the complete 600 dpi figure set after all required intermediate outputs are available.

The remaining scripts provide validation and supplementary visualizations retained for reproducibility.
