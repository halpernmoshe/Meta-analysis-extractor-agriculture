# Reproduce the full scope-aware analysis (numbers + figures).
Set-Location $PSScriptRoot
python line_by_line_scope_aware.py
python scope_aware_paired_tost.py
python scope_aware_aggregate_tost.py
python reconciliation_analysis.py
python make_bland_altman.py
python make_fig1_fidelity.py
python make_fig2_equivalence.py
python make_fig3_reconciliation.py
