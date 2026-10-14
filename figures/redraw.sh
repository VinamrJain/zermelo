#!/bin/bash
# Redraws every paper figure from the newest launches into results/figures/, then copies the pdfs into paper/manuscript/figures/.
# Sizes are set in figures/common.py: WIDTH, ROW_HEIGHT, LEGEND_HEIGHT and the font size in rcParams.
cd "$(dirname "$0")/.."
for script in fig_comparison fig_travel fig_main fig_oracle fig_resource fig_tuning fig_cost_weight fig_opening fig_fields_paths fig_planners fig_prior_mismatch tab_metrics fig_ocean_search fig_ocean_learn fig_ocean_visit; do
  pixi run fig $script 2>&1 | grep "^/share\|^results\|Error\|error"
done
(cd results/figures/appendix && for table in tab_*_standalone.tex; do pdflatex -interaction=batchmode "$table" > /dev/null; done; rm -f *.aux *.log)
../paper/junk/sync_figures.sh > /dev/null
