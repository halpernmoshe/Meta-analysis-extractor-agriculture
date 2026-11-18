# Re-run every analysis and figure script from the deposited key tables.
# Outputs are written in place: printed results to the console, figures to
# 4_figures/figures/, coverage tables to 3_evaluation/round2_additional_analysis/.
# To check the results against the expected outputs, run:  python verify.py
$ErrorActionPreference = 'Stop'
$env:PYTHONIOENCODING = 'utf-8'
$root = $PSScriptRoot
$scripts = @(
    '3_evaluation/line_by_line_scope_aware.py',
    '3_evaluation/scope_aware_paired_tost.py',
    '3_evaluation/scope_aware_aggregate_tost.py',
    '3_evaluation/biochar_native_control_tost.py',
    '3_evaluation/reconciliation_analysis.py',
    '3_evaluation/make_bland_altman.py',
    '3_evaluation/round2_additional_analysis/coverage_structural_complexity.py',
    '4_figures/make_fig1_fidelity.py',
    '4_figures/make_fig2_equivalence.py',
    '4_figures/make_fig3_reconciliation.py'
)
foreach ($script in $scripts) {
    Write-Host "=== $script"
    python (Join-Path $root $script)
    if ($LASTEXITCODE -ne 0) {
        throw "Failed while running $script"
    }
}
