# Meta-Analysis Data Extraction Skill

Extract quantitative data from research papers (PDFs) for systematic review and meta-analysis. This skill reads PDFs, identifies relevant data tables and figures, and outputs structured JSON with treatment means, control means, variance, sample sizes, and moderator variables.

## Usage

```
/extract-meta-analysis <config_file> <pdf_directory> <output_directory> [--ground-truth <gt_file>] [--compare <second_extraction_dir>]
```

Arguments from `$ARGUMENTS`:
- Arg 1: Path to config JSON file (or "interactive" to create one)
- Arg 2: Directory containing PDF files to process
- Arg 3: Output directory for extracted JSON files
- Optional flags:
  - `--ground-truth <path>`: CSV/JSON with ground truth for validation
  - `--compare <path>`: Directory with a second extraction for cross-method agreement

If no arguments provided, prompt the user for each.

## Step-by-Step Procedure

### Phase 1: Setup and Configuration

1. **Parse arguments** from `$ARGUMENTS`. If config path is "interactive" or missing, walk the user through PICO definition (see "Interactive Config Creation" below).

2. **Load and validate the config file**. Ensure it has at minimum: `intervention`, `control`, `primary_outcomes`, and `expected_direction`.

3. **Scan the PDF directory**. List all PDF files, report count, and confirm with user before proceeding.

4. **Create output directory** if it doesn't exist.

### Phase 2: Extraction (Per Paper)

For EACH PDF file in the input directory:

1. **Read the PDF** using the Read tool. If the PDF is large, read it in sections (first 50 pages, then remaining).

2. **Identify the paper metadata**: title, authors, year, journal, DOI if available.

3. **Locate ALL data tables and figures** containing quantitative outcomes matching the config's `primary_outcomes`. Check:
   - All tables in the paper (not just the first one)
   - Table titles, headers, and footnotes
   - Figure captions for numeric data
   - Results section text for inline statistics
   - Supplementary materials if present in the PDF

4. **For each relevant table/figure, extract EVERY row**:
   - `outcome`: The measured variable (e.g., "Zn", "yield", "biomass")
   - `treatment_mean`: Mean value under intervention condition
   - `control_mean`: Mean value under control condition
   - `effect_pct`: Calculated as `(treatment_mean - control_mean) / |control_mean| * 100`
   - `n`: Sample size per group (check Methods section, table footnotes, or figure legends)
   - `variance_type`: One of "SE", "SD", "CI_95", "LSD", "CV", or null
   - `variance_value`: The numeric variance value, or null if not reported
   - `unit`: Measurement unit (e.g., "mg/kg", "t/ha", "g/plant")
   - `source_type`: One of "table", "figure", or "text" (see Rule 11)
   - `data_source`: Where the data came from (e.g., "Table 2", "Figure 3 (estimated from graph)", "text p.5")
   - `treatment_description`: What the treatment condition is (e.g., "Elevated CO2 700 ppm")
   - `control_description`: What the control condition is (e.g., "Ambient CO2 400 ppm")
   - `moderators`: Dict of moderator variables from the config (e.g., `{"cultivar": "IR64", "soil_type": "clay"}`)

5. **Save per-paper JSON** to the output directory as `{paper_id}_extracted.json`.

### Phase 3: Summary Report

After all papers are processed:

1. Run the summarize script (`.claude/skills/extract-meta-analysis/scripts/summarize.py`) on the output directory.
2. Report: total papers processed, total observations, outcome distribution, effect size statistics, variance/sample size coverage, and flagged issues.

### Phase 4: Optional Validation

If `--ground-truth` is provided:
1. Run the validate script (`.claude/skills/extract-meta-analysis/scripts/validate.py`) with the extracted data and ground truth.
2. Report: Pearson r, ICC, MAE, direction agreement, TOST result, and per-paper tiers.

If `--compare` is provided:
1. Match observations between the two extractions by paper + outcome + moderator.
2. Report agreement statistics (r, MAE, direction match rate).

---

## Critical Extraction Rules

### Rule 1: Extract EVERY Row
If a table has 20 rows of data, you should produce approximately 20 observations. Do NOT pool, average, or summarize across rows. Each unique combination of treatment x cultivar x tissue x time point is a separate observation.

### Rule 2: Factorial Designs
Many papers use factorial designs (e.g., CO2 level x nitrogen level x cultivar). Extract ALL relevant combinations:
- If the design is CO2 x N with 2 CO2 levels and 3 N levels, and the config intervention is CO2, then:
  - Control = ambient CO2 at each N level
  - Treatment = elevated CO2 at each N level
  - This produces 3 observations (one per N level), NOT 1 pooled observation
- Record the other factors (N level) as moderators

### Rule 3: Treatment vs Control Assignment
- READ THE CONFIG carefully. The config specifies what is treatment and what is control.
- Common mistake: confusing which group is treatment vs control
- Verify: if `expected_direction` is "negative" but most of your effects are positive, you may have swapped T/C
- Sanity check: does the direction of most effects match `expected_direction`?

### Rule 4: Finding Variance (SE/SD)
Variance information can be in many places. Search ALL of these:
1. **Table cells**: Look for "mean +/- value" or parenthetical values
2. **Table footnotes**: Often says "Values are mean +/- SE" or "LSD (0.05) = ..."
3. **Figure captions**: "Error bars represent SE" or "Bars indicate +/- SD"
4. **Methods/Statistical Analysis section**: "Data are expressed as mean +/- SEM"
5. **Table headers**: Column labeled "SE" or "SD" or "+/-"

If the paper says "mean +/- SE" in one place but the table header says "SD", the table header wins (it's more specific to that data).

### Rule 5: Finding Sample Size (n)
Sample size is critical for meta-analysis weighting. Look in:
1. **Methods section**: "n = X per treatment" or "X replicates"
2. **Table footnotes**: "Each value is the mean of n determinations"
3. **Experimental design section**: "randomized block with X replications"
4. **Figure legends**: "n = X"

### Rule 6: Handle Missing Data Honestly
- If a value is not reported, set it to `null`. NEVER guess or hallucinate values.
- If variance type is ambiguous, set `variance_type` to null and note the ambiguity.
- If sample size is not found anywhere, set `n` to null.
- It is better to have null values than wrong values.

### Rule 7: Units Matter
- Record the units exactly as reported in the paper
- Common unit variations: mg/kg vs ppm vs ug/g (these are equivalent)
- Watch for papers that report "% of control" rather than absolute values
- If the paper reports "% change", calculate back to absolute means if possible, or note that these are relative values

### Rule 8: Check ALL Tables
Do not stop after the first table. Many papers have:
- Table 1: Experimental conditions (not data)
- Table 2: Main results (THIS is what you want)
- Table 3: Additional results or secondary outcomes
- Table S1-S5: Supplementary tables with more data

### Rule 9: Effect Size Calculation
Always calculate effect size as: `(treatment_mean - control_mean) / |control_mean| * 100`

This gives percent change. Example:
- Control = 50 mg/kg, Treatment = 45 mg/kg
- Effect = (45 - 50) / |50| * 100 = -10%

If control_mean is 0, set effect_pct to null (division by zero).

### Rule 10: Data from Figures
If data is presented only in figures (bar charts, scatter plots):
- Read approximate values from the figure
- Note `data_source: "Figure X (estimated from graph)"`
- These are less precise but still valuable for meta-analysis
- Look for exact values in supplementary materials first

### Rule 11: Source Type Labeling
Every observation MUST include a `source_type` field:
- `"table"` -- data read directly from a data table in the paper
- `"figure"` -- data estimated from a figure (bar chart, line graph, scatter plot). These values are inherently less precise (typical error +/-3-5pp). Always note "(estimated from figure)" in the data_source field.
- `"text"` -- data extracted from inline text in results/abstract

Figure-estimated data should be clearly labeled because:
1. Values are approximate (reading from axes introduces error)
2. Downstream users may want to exclude figure data for sensitivity analysis
3. Meta-analysis can weight figure-sourced observations with wider variance

---

## Interactive Config Creation

If the user doesn't have a config file, guide them through creating one:

1. **Study name**: "What is your meta-analysis about?"
2. **Intervention**: "What is the treatment/intervention being studied?"
3. **Control**: "What is the control/comparison condition?"
4. **Primary outcomes**: "What outcome variables are you extracting? (e.g., yield, mineral concentration, biomass)"
5. **Expected direction**: "Do you expect the intervention to increase or decrease the outcome? (positive/negative/mixed)"
6. **Moderators**: "What variables do you want to track as moderators? (e.g., species, soil type, climate)"
7. **Notes**: "Any special instructions for extraction?"

Save the config to the output directory as `extraction_config.json`.

---

## Output Format

### Per-Paper JSON
```json
{
  "paper_id": "Smith_2020",
  "title": "Effects of X on Y in Z",
  "authors": "Smith, J.; Jones, M.",
  "year": 2020,
  "journal": "Journal of Example Studies",
  "doi": "10.1234/example",
  "extraction_date": "2026-03-18",
  "config_used": "my_config.json",
  "observations": [
    {
      "outcome": "Zn",
      "source_type": "table",
      "treatment_mean": 25.3,
      "control_mean": 30.1,
      "effect_pct": -15.95,
      "n": 4,
      "variance_type": "SE",
      "variance_value": 1.2,
      "unit": "mg/kg",
      "data_source": "Table 2",
      "treatment_description": "Elevated CO2 (600 ppm)",
      "control_description": "Ambient CO2 (400 ppm)",
      "moderators": {
        "species": "Triticum aestivum",
        "tissue": "grain",
        "experiment_type": "FACE"
      }
    }
  ],
  "notes": "Variance from table footnote: 'Values are mean +/- SE (n=4)'"
}
```

### Summary JSON (auto-generated)
```json
{
  "total_papers": 25,
  "total_observations": 342,
  "outcomes": {"Zn": 120, "Fe": 95, "N": 80, "P": 47},
  "effect_size_stats": {
    "mean": -8.2,
    "median": -6.5,
    "range": [-45.2, 32.1]
  },
  "variance_coverage": 0.72,
  "sample_size_coverage": 0.89,
  "flags": ["3 observations with effect > 100%", "5 papers with no variance data"]
}
```

---

## Platform Notes

- This skill works on Windows. Use forward slashes in paths or raw strings.
- Python scripts in `scripts/` require only numpy and scipy. Install with: `pip install numpy scipy`
- For PDF reading, the Claude Code Read tool handles PDFs natively.
- Output encoding: scripts use `sys.stdout.reconfigure(encoding='utf-8')` on Windows.

---

## Tips for Best Results

1. **Start with 3-5 papers** as a pilot before running the full batch. Review the JSON output carefully.
2. **Check effect direction**: if your config says `expected_direction: "negative"` but most effects are positive, something is wrong.
3. **Variance is the hardest part**. Many papers report it inconsistently or not at all. A 60-70% variance coverage rate is normal.
4. **Factorial designs** are the main source of observation count disagreements. When in doubt, extract more granularly.
5. **Some papers are genuinely poor** and don't report enough data. It's OK to have papers with few or no extractable observations.
