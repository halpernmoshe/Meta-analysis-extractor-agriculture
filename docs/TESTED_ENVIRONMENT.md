# Tested environment

`python verify.py`, `run_all.ps1` and `run_all.sh` (Git Bash) completed without errors on
18 November 2026 in this environment:

- Microsoft Windows NT 10.0.26200.0
- Python 3.12.10, 64-bit
- matplotlib 3.10.8
- NumPy 2.4.0
- openpyxl 3.1.5
- pandas 2.3.3

The Hui reference-side check was run with the downloaded workbook in place and passed; without the
workbook it is reported as SKIPPED. On this environment the regenerated figures are identical to the
deposited PNG files. Other operating systems and package versions were not tested.
`requirements.txt` gives minimum versions expected to work, not tested versions.
