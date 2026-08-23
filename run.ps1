# Reproduce the deposited scope-aware analysis (numbers + figures).
Set-Location $PSScriptRoot
$scripts = @(
    'line_by_line_scope_aware.py',
    'scope_aware_paired_tost.py',
    'scope_aware_aggregate_tost.py',
    'reconciliation_analysis.py',
    'make_bland_altman.py',
    'make_fig1_fidelity.py',
    'make_fig2_equivalence.py',
    'make_fig3_reconciliation.py',
    'round2_additional_analysis/coverage_structural_complexity.py'
)

foreach ($script in $scripts) {
    python $script
    if ($LASTEXITCODE -ne 0) {
        throw "Reproduction failed while running $script"
    }
}
