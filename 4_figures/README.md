# Stage 4: figures

The figure scripts read the key tables in `../2_conversion/keys/`, recompute the values they plot,
and write PNG files to `figures/`. They need matplotlib and contain no random elements.

| Script | Figure | Output |
|---|---|---|
| `make_fig1_fidelity.py` | Figure 1: reading-fidelity scatter | `figures/fig1_fidelity.png` |
| `make_fig2_equivalence.py` | Figure 2: paired equivalence forest | `figures/fig2_equivalence.png` |
| `make_fig3_reconciliation.py` | Figure 3: reconciliation (biochar control definition; Loladze per element) | `figures/fig3_reconciliation.png` |
| `../3_evaluation/make_bland_altman.py` | Figure S1: Bland-Altman agreement | `figures/figS1_bland_altman.png` |

`make_fig2_equivalence.py` and `make_fig3_reconciliation.py` print the values they plot.
`expected_outputs/` holds that printed text, without the first line, which gives the absolute path
of the written file. Figure 1 prints only the file path, so it is checked by its image alone; it
plots the matched cells summarised in Table 3. The Figure S1 values are printed by
`make_bland_altman.py` and checked against `3_evaluation/expected_outputs/EXPECTED_OUTPUT_BLAND_ALTMAN.txt`.

`python verify.py` (repository root) regenerates all four PNG files in a temporary copy and compares
each with the deposited file. On the tested environment (`../docs/TESTED_ENVIRONMENT.md`) the
regenerated files are identical to the deposited ones. A different matplotlib or FreeType version
can change the rendering. A different image is reported as a failure, unless
`python verify.py --allow-image-differences` is used; then it is a warning, and the check relies on
the printed values, which do not cover Figure 1.
