#!/bin/bash
# Redraws every paper figure from the newest launches into results/figures/, then copies the pdfs into paper/manuscript/figures/.
# Sizes are set in figures/common.py: WIDTH, ROW_HEIGHT, LEGEND_HEIGHT and the font size in rcParams.
cd "$(dirname "$0")/.."
for script in comparison.fig_comparison comparison.fig_travel comparison.fig_main comparison.fig_oracle comparison.fig_tuning comparison.tab_metrics \
              balloon.fig_resource balloon.fig_cost_weight ambient.fig_opening ambient.fig_fields_paths ambient.fig_planners ambient.fig_prior_mismatch \
              ocean.fig_search ocean.fig_learn ocean.fig_visit; do
  pixi run fig $script 2>&1 | grep "^results\|Error\|error"
done
pixi run fig comparison.headline -- "$(ls -d results/ambient_waypoint/planning_budget/*/arm=planning_budget_steps50,seed=7 | tail -1)" "$(ls -d results/balloon_waypoint/static/*/arm=static_esi_L50,seed=7 | tail -1)" 1590 160 2>&1 | grep "^results\|Error\|error"
for where in main appendix; do
  (cd results/figures/$where && for table in tab_*_standalone.tex; do pdflatex -interaction=batchmode "$table" > /dev/null; done; rm -f *.aux *.log)
done
../paper/junk/sync_figures.sh > /dev/null
