#!/bin/bash
# Redraws every paper figure from the newest launches into results/figures/, then copies the pdfs into paper/manuscript/figures/.
# Sizes are set in figures/common.py: WIDTH, ROW_HEIGHT, LEGEND_HEIGHT and the font size in rcParams.
cd "$(dirname "$0")/.."
for script in fig_comparison fig_travel fig_main fig_oracle fig_resource fig_tuning fig_cost_weight fig_opening fig_fields_paths fig_planners fig_prior_mismatch tab_metrics fig_ocean_search fig_ocean_learn fig_ocean_visit; do
  pixi run fig $script 2>&1 | grep "^results\|Error\|error"
done
pixi run fig headline -- "$(ls -d results/ambient_waypoint/planning_budget/*/arm=planning_budget_steps50,seed=7 | tail -1)" "$(ls -d results/balloon_waypoint/static/*/arm=static_esi_L50,seed=7 | tail -1)" 1590 160 2>&1 | grep "^results\|Error\|error"
for where in main appendix; do
  (cd results/figures/$where && for table in tab_*_standalone.tex; do pdflatex -interaction=batchmode "$table" > /dev/null; done; rm -f *.aux *.log)
done
../paper/junk/sync_figures.sh > /dev/null
