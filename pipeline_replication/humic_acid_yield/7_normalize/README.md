# Stage 7: Normalize Effectors

**Stage name**: Effector label normalization

**Purpose**: Normalize free-text moderator / effector labels to controlled vocabulary levels defined in the config (e.g., "soil drench" and "drench" both map to "soil_drench"). Enables subgroup analyses and benchmark-aligned subset filtering.

**Expected inputs**:
- `../6_adjudicate/adjudicated_kept.csv` — adjudication-passed rows
- `../config.json` — moderator definitions with expected_levels

**Expected outputs**:
- `effector_labels.json` — raw-to-normalized label mapping
- `summary_normalized.csv` — rows with normalized moderator columns
