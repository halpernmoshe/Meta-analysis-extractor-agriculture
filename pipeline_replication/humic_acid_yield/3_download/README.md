# Stage 3: Download

**Stage name**: PDF and full-text download

**Purpose**: Download full-text PDFs for screened-in papers from Stage 2. Falls back to abstract-only extraction when full text is unavailable.

**Expected inputs**:
- `../2_screen/screened_in.csv` — papers that passed abstract screening

**Expected outputs**:
- `pdfs/` — downloaded PDF files
- `download_log.csv` — per-paper download status (success / failed / abstract_only)
- `download_summary.json` — total counts and coverage statistics
