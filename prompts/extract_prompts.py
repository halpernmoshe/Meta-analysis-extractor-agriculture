"""
Prompt templates for data extraction tasks

These prompts are designed to extract quantitative data from papers
with high precision and structured output.
"""

# System prompt for extraction tasks
EXTRACTION_SYSTEM_PROMPT = """You are an expert data extractor for agricultural meta-analysis.

Your task is to precisely and COMPREHENSIVELY extract quantitative data from scientific papers.
Be extremely accurate with numbers - extract exactly what is reported.
Do NOT calculate, convert, or estimate values unless explicitly instructed.
If a value is unclear or not present, mark it as null.

CRITICAL GUIDELINES:
1. Extract data from ALL tables in the paper - not just the first one or two
2. Always look for sample size (n) in the Methods section if not shown in tables
3. Always check table footnotes for variance notation (SD, SE, LSD)
4. Extract ALL outcome variables found, not just some - let the user decide what's important

VARIANCE IS CRITICAL FOR META-ANALYSIS:
- Meta-analysis REQUIRES variance data (SE, SD, or LSD) to calculate effect sizes
- WITHOUT variance, observations cannot be properly weighted
- Search EXHAUSTIVELY for variance in:
  * Table column values (look for ± symbol, e.g., "12.5 ± 0.8")
  * Separate SE/SD columns in tables
  * Table footnotes (e.g., "Values are means ± SE (n=4)")
  * LSD values in footnotes (e.g., "LSD(0.05) = 2.3")
  * Figure error bar descriptions
  * Results text (e.g., "mean of 45.2 ± 3.1 kg/ha")
- Extract the ACTUAL NUMERIC VALUES, not just the type
- If variance differs between treatments, extract each separately

SAMPLE SIZE (n) IS ALSO CRITICAL:
- Search Methods section: "n=4", "4 replicates", "4 replications per treatment"
- Search table footnotes: "Values are means of 4 replicates"
- Search experimental design: "randomized complete block with 3 replications"
- Use the SAME n for all observations from that paper if only reported once

Always output valid JSON as specified. Keep responses concise - output only the JSON structure."""


def get_extraction_prompt(
    paper_text: str,
    outcomes: list,
    moderators: list,
    control_heuristic: dict,
    design_info: dict,
    variance_info: dict,
    primary_outcome: str = None,
    domain_config: dict = None
) -> str:
    """
    Main extraction prompt for pulling all data from a paper

    Args:
        primary_outcome: The main outcome code (e.g., 'MINERAL_CONC') to prioritize
    """
    outcomes_str = "\n".join([f"- {o['code']}: {o['name']} (unit: {o.get('unit', 'as reported')})"
                              for o in outcomes])

    # Build primary outcome priority section
    primary_outcome_section = ""
    if primary_outcome:
        primary_outcome_section = f"""
=== CRITICAL: PRIMARY OUTCOME PRIORITY ===

**YOUR #1 PRIORITY IS TO EXTRACT: {primary_outcome}**

This meta-analysis is specifically about {primary_outcome}. You MUST:
1. Extract ALL {primary_outcome} observations FIRST, before extracting any other outcome types
2. Search ALL tables in the paper for {primary_outcome} data - don't stop at early tables
3. Only after extracting ALL {primary_outcome} data should you extract secondary outcomes
4. If you run out of space, it is BETTER to have complete {primary_outcome} data than partial data from all outcomes

DO NOT extract biomass (CWAD, RWAD, GWAD) or other outcomes BEFORE you have extracted
ALL available {primary_outcome} data from ALL tables in the paper.

"""
    moderators_str = "\n".join([f"- {m['code']}: {m['name']}" for m in moderators])
    
    control_desc = control_heuristic.get('description', 'Identify the baseline/control treatment')
    control_keywords = ", ".join(control_heuristic.get('keywords', ['control']))
    
    design_type = design_info.get('design_type', 'Unknown')
    design_notes = ""
    if design_type == "Split-plot":
        main_factor = design_info.get('main_plot_factor', 'unknown')
        sub_factor = design_info.get('subplot_factor', 'unknown')
        design_notes = f"""
SPLIT-PLOT DESIGN DETECTED:
- Main plot factor: {main_factor}
- Subplot factor: {sub_factor}
- Extract the appropriate error term based on which factor is being analyzed"""
    
    variance_type = variance_info.get('variance_type', 'Unknown')
    variance_notes = f"Expected variance type: {variance_type}"
    if variance_type == "LSD":
        variance_notes += "\n- For LSD, record the LSD value as pooled_variance"
        variance_notes += "\n- Note the alpha level (usually 0.05)"
    elif variance_type == "CV":
        variance_notes += "\n- For CV, record the CV% as pooled_variance"
    
    # Build domain-specific section
    domain_section = ""
    if domain_config:
        domain_name = domain_config.get('name', 'Unknown')
        domain_instructions = domain_config.get('extraction_instructions', '')
        cultivar_instructions = domain_config.get('cultivar_instructions', '')
        domain_section = f"""
=== DOMAIN: {domain_name} ===
{domain_instructions}

{cultivar_instructions}
"""

    return f"""Extract ALL quantitative data from this paper for meta-analysis.
{primary_outcome_section}
PAPER TEXT:
{paper_text[:80000]}

=== EXTRACTION INSTRUCTIONS ===
{domain_section}
CONTROL IDENTIFICATION:
{control_desc}
Keywords to identify control: {control_keywords}

EXPERIMENTAL DESIGN: {design_type}
{design_notes}

VARIANCE REPORTING:
{variance_notes}

OUTCOMES TO EXTRACT:
{outcomes_str}

=== CRITICAL: OUTCOME CLASSIFICATION ===

MINERAL_CONC means ONLY mineral/nutrient element concentrations:
- YES: N, P, K, Ca, Mg, Fe, Zn, Mn, Cu, S, B, Mo, Se, Na, Cd, Pb, Ni, Al
- YES: Nitrogen concentration, Phosphorus content, Zinc in grain, Iron in leaves
- NO: Photosynthesis rate (use PHOT)
- NO: Stomatal conductance (use COND)
- NO: Starch concentration (NOT a mineral)
- NO: Transpiration rate (NOT a mineral)
- NO: Temperature, humidity (NOT outcomes)
- NO: Chlorophyll content (NOT a mineral element)
- NO: Bacteria counts (NOT a mineral)

Only classify as MINERAL_CONC if the outcome is a mineral element concentration.
Other physiological measurements should use the appropriate code (PHOT, COND, GWAD, CWAD, etc.)
If unsure, use "CUSTOM" with a descriptive outcome_name.

MODERATORS TO EXTRACT:
{moderators_str}

=== CRITICAL: EXTRACT FROM ALL TABLES ===

You MUST extract data from EVERY table in the paper, not just the first few tables.
- Scientific papers often have biomass/growth in early tables and mineral/nutrient data in later tables
- Go through Table 1, Table 2, Table 3, Table 4, etc. - extract relevant data from ALL of them
- Do NOT stop after finding data in one or two tables
- Tables with element names (N, P, K, Ca, Mg, Fe, Zn, Mn, Cu, etc.) contain mineral concentration data

=== CRITICAL: INDIVIDUAL vs AVERAGED DATA ===

ALWAYS prefer INDIVIDUAL DATA over AVERAGED/SUMMARY data:

1. **CULTIVARS/VARIETIES**: If a paper reports data for multiple cultivars (e.g., "Masr 3", "Sakha 95", "Giza 171"):
   - Extract SEPARATE observations for EACH cultivar
   - Do NOT extract only the "average across cultivars" row
   - Each cultivar should have its own observation with its specific values
   - Set moderators.CULTIVAR to the individual cultivar name (not "Average of X, Y, Z")

2. **YEARS/SEASONS**: If a paper reports data for multiple years (e.g., 2020, 2021):
   - Extract SEPARATE observations for EACH year
   - Do NOT extract only the "average across years" row
   - Set study_year to the specific year (not "Average of 2020 and 2021")

3. **SITES/LOCATIONS**: If a paper has data from multiple sites:
   - Extract SEPARATE observations for EACH site
   - Set site_id to the specific site name

4. **IRRIGATION/STRESS LEVELS**: If a paper has multiple irrigation treatments:
   - Extract SEPARATE observations for EACH level
   - Set moderators.WATER_REGIME to the specific level

**WHEN TO USE SUMMARY DATA**: Only use summary/averaged data if:
- The paper ONLY reports averaged data (individual data not available)
- In that case, set is_summary_row: true and note what was averaged in the notes field

**EXAMPLE**: If Table 3 shows data for 3 cultivars across 2 years = 6 combinations
Extract 6 observations (one per cultivar-year combination), NOT 1 averaged observation

=== IMPORTANT: NON-SIGNIFICANT RESULTS ===

Extract data EVEN WHEN the paper reports "no significant effect", "no change", or "ns":
- Non-significant results are VALUABLE for meta-analysis
- If a paper says "No change in Ca or K was observed", still extract the Ca and K values
- The absence of a significant effect is still data that must be captured

=== IMPORTANT: EXTRACT ACTUAL VARIANCE VALUES ===

When variance is reported, extract the ACTUAL NUMERIC VALUES:
- If table shows "12.5 ± 0.8", the mean is 12.5 and variance is 0.8
- Put variance values in treatment_variance and control_variance fields
- Don't just identify variance_type - extract the actual numbers
- Check table columns for ± values, SE columns, or SD columns

=== TABLE FORMAT: AVERAGED DATA ===

Some tables show data "averaged across" factors:
- "Temperature data averaged over all CO2 concentrations"
- "CO2 data averaged over both temperatures"
- Extract these values and note the averaging in the notes field
- These are still valid observations for meta-analysis

=== FINDING SAMPLE SIZE (n) ===

Look carefully for sample size (n) in these locations:
1. Methods section - often stated as "n = X", "X replicates", "X replications", "X plants per treatment"
2. Table footnotes - may say "n = X per treatment" or "Values are means of X replicates"
3. Statistical analysis section - degrees of freedom can indicate n
4. Figure captions - may mention sample size

=== FINDING VARIANCE ===

Look for variance (SD, SE, CV, LSD) in these locations:
1. Table columns - look for "±" symbols or columns labeled "SD", "SE", "S.E.", "s.e."
2. Table footnotes - often say "Values are means ± SE" or "LSD (P=0.05) = X"
3. Figure error bars - caption often specifies if bars are SD or SE
4. Results text - may state "mean ± SD" or "SE of the mean"

=== EXTRACTION RULES ===

1. Extract EVERY treatment-control comparison for EACH outcome variable from EVERY table
2. For factorial designs, extract each treatment combination as a separate observation
3. Record variance EXACTLY as reported - do not convert
4. Note the source location (Table X, Figure Y) for each value
5. If values are unclear or estimated from figures, set confidence to "low"
6. Mark summary/mean rows with is_summary_row: true - these aggregate across years/sites
7. Extract individual year/site data when available, not just overall means
8. If n is not in the table, search Methods section and use that n for all observations

=== OUTPUT FORMAT ===

Return a JSON object with this structure:
{{
    "paper_info": {{
        "title": "paper title",
        "authors": "first author et al.",
        "year": publication year,
        "study_years": ["2018", "2019"],
        "study_sites": ["site names if multiple"]
    }},
    "control_treatment": {{
        "description": "description of control",
        "identification_method": "how control was identified"
    }},
    "observations": [
        {{
            "observation_id": 1,
            "outcome_variable": "ICASA code",
            "outcome_name": "human readable name",
            "plant_part": "Grain" | "Leaf" | etc or null,
            "treatment_description": "description of treatment",
            "control_description": "description of control",
            "treatment_mean": number,
            "control_mean": number,
            "treatment_n": sample size or null,
            "control_n": sample size or null,
            "variance_type": "SD" | "SE" | "LSD" | "CV" | null,
            "treatment_variance": number or null,
            "control_variance": number or null,
            "pooled_variance": number or null (for LSD, CV, MSE),
            "degrees_of_freedom": number or null,
            "alpha_level": 0.05 or other or null,
            "unit_reported": "unit as stated in paper",
            "data_source": "Table 2" | "Figure 3" | etc,
            "study_year": "2018" or null,
            "site_id": "site name" or null,
            "moderators": {{
                "MODERATOR_CODE": "value"
            }},
            "is_summary_row": false,
            "confidence": "high" | "medium" | "low",
            "notes": "any relevant notes"
        }}
    ],
    "extraction_notes": "any overall notes about the extraction"
}}

Return ONLY the JSON object, no other text."""


def get_table_extraction_prompt(
    paper_text: str,
    table_id: str,
    outcomes: list,
    control_heuristic: dict
) -> str:
    """
    Focused extraction from a specific table
    """
    outcomes_str = ", ".join([o['code'] for o in outcomes])
    control_keywords = ", ".join(control_heuristic.get('keywords', ['control']))

    return f"""Extract data from {table_id} in this paper.

PAPER TEXT:
{paper_text[:40000]}

TARGET OUTCOMES: {outcomes_str}
CONTROL KEYWORDS: {control_keywords}

Extract ALL rows from {table_id} that contain quantitative data.

For each row, extract:
- Treatment name/description
- Whether it's the control (based on keywords: {control_keywords})
- Mean value
- Variance value (SD, SE, or whatever is reported) - check column headers AND footnotes
- Sample size (n) - if not in table, look in Methods section for "n=X", "X replicates", etc.
- Any factor levels (year, site, variety, etc.)

IMPORTANT: Check table footnotes carefully - they often contain:
- Variance notation (e.g., "Values are means ± SE (n=3)")
- Sample size (e.g., "n = 4 per treatment")
- Significance indicators

Return as JSON:
{{
    "table_id": "{table_id}",
    "table_title": "table caption if visible",
    "column_headers": ["list of column headers"],
    "variance_type": "SD" | "SE" | "LSD" | "CV" | null,
    "variance_location": "in columns" | "in footnote" | "not present",
    "rows": [
        {{
            "row_id": 1,
            "treatment": "treatment name",
            "is_control": true/false,
            "factor_levels": {{"year": "2018", "variety": "cv1"}},
            "values": {{
                "OUTCOME_CODE": {{
                    "mean": number,
                    "variance": number or null,
                    "n": number or null,
                    "unit": "unit"
                }}
            }},
            "is_summary": true/false,
            "notes": ""
        }}
    ],
    "footnotes": "any relevant footnotes",
    "notes": "extraction notes"
}}

Return ONLY the JSON object."""


def get_figure_extraction_prompt(
    paper_text: str,
    figure_id: str,
    outcomes: list
) -> str:
    """
    Guidance for extracting data from figures
    """
    outcomes_str = ", ".join([o['code'] for o in outcomes])
    
    return f"""Analyze {figure_id} in this paper to understand what data it contains.

PAPER TEXT:
{paper_text[:40000]}

TARGET OUTCOMES: {outcomes_str}

Based on the figure caption and any description in the text, provide:

{{
    "figure_id": "{figure_id}",
    "figure_title": "figure caption",
    "figure_type": "bar chart" | "line graph" | "scatter plot" | "box plot" | "other",
    "outcomes_shown": ["list of outcomes in figure"],
    "x_axis": "what's on x-axis",
    "y_axis": "what's on y-axis with units",
    "treatments_shown": ["list of treatments/groups"],
    "has_error_bars": true/false,
    "error_bar_type": "SD" | "SE" | "CI" | "unknown" | null,
    "can_extract_values": true/false,
    "extraction_difficulty": "easy" | "medium" | "hard",
    "notes": "any notes about extracting from this figure"
}}

Return ONLY the JSON object."""


def get_moderator_extraction_prompt(
    paper_text: str,
    moderators: list
) -> str:
    """
    Extract moderator values from paper
    """
    mod_list = "\n".join([f"- {m['code']}: {m['name']} ({m.get('type', 'any')})" for m in moderators])

    return f"""Extract moderator variable values from this paper.

PAPER TEXT:
{paper_text[:40000]}

MODERATORS TO EXTRACT:
{mod_list}

For each moderator, find and extract the value(s) reported in the paper.

Return as JSON:
{{
    "moderators": {{
        "MODERATOR_CODE": {{
            "value": "single value if applicable",
            "values": ["list if multiple values"],
            "unit": "unit if applicable",
            "source": "where in paper this was found",
            "confidence": "high" | "medium" | "low",
            "notes": ""
        }}
    }},
    "study_location": {{
        "country": "country name",
        "region": "region/state",
        "site_name": "specific site if named",
        "latitude": null,
        "longitude": null,
        "coordinates_source": "reported" | "looked up" | null
    }},
    "soil_properties": {{
        "texture": "texture class if reported",
        "pH": null,
        "organic_carbon": null,
        "organic_carbon_unit": "%" | "g/kg" | null,
        "total_N": null,
        "available_P": null,
        "available_K": null
    }},
    "climate_info": {{
        "climate_description": "any climate description",
        "mean_annual_precip": null,
        "mean_annual_temp": null,
        "growing_season_precip": null
    }},
    "notes": "any notes about moderator extraction"
}}

Return ONLY the JSON object."""


def get_missing_data_prompt(
    paper_text: str,
    missing_fields: list
) -> str:
    """
    Second pass to find missing data
    """
    fields_str = ", ".join(missing_fields)

    return f"""Search this paper carefully for the following missing information:

MISSING FIELDS: {fields_str}

PAPER TEXT:
{paper_text[:40000]}

=== WHERE TO LOOK FOR SAMPLE SIZE (n) ===
- Methods section: "n = X", "X replicates", "X replications per treatment", "X plants per pot"
- Experimental design: "randomized block design with X replicates"
- Statistical analysis: degrees of freedom can indicate n
- Table footnotes: "Values are means of X replicates"
- Figure captions: "n = X per treatment"

=== WHERE TO LOOK FOR VARIANCE (SD, SE, CV, LSD) ===
- Table column headers: look for "±" or "SD" or "SE" columns
- Table footnotes: "Values are means ± SE" or "Different letters indicate significance (LSD P=0.05)"
- Results text: may report "mean ± SD" inline
- Figure captions: "Error bars represent SE (n=X)"
- LSD values often in table footnotes: "LSD(0.05) = X"

=== OTHER LOCATIONS ===
- Methods section for experimental details
- Results section for any mentioned values
- Tables and table footnotes (check ALL tables, not just first few)
- Figure captions
- Supplementary information references

Return as JSON:
{{
    "found_values": {{
        "field_name": {{
            "value": "found value",
            "source": "where found (be specific: 'Methods section, paragraph 2' or 'Table 3 footnote')",
            "confidence": "high" | "medium" | "low"
        }}
    }},
    "still_missing": ["fields that could not be found"],
    "suggestions": "any suggestions for finding missing data"
}}

Return ONLY the JSON object."""


# Compile all prompts
EXTRACT_PROMPTS = {
    'main': get_extraction_prompt,
    'table': get_table_extraction_prompt,
    'figure': get_figure_extraction_prompt,
    'moderators': get_moderator_extraction_prompt,
    'missing': get_missing_data_prompt
}
