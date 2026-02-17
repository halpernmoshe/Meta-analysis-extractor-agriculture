"""
Prompt templates for reconnaissance tasks

These prompts are designed to extract structured information from papers
during the reconnaissance phase.
"""

# System prompt for all recon tasks
RECON_SYSTEM_PROMPT = """You are an expert agricultural research analyst specializing in meta-analysis data extraction.

Your task is to carefully analyze scientific papers and extract structured information.
Be precise, thorough, and conservative - if information is unclear or not present, say so.
Always output valid JSON as specified in each prompt."""


def get_overview_prompt(paper_text: str) -> str:
    """
    Initial overview scan to understand paper content
    Used in orientation phase
    """
    return f"""Analyze this agricultural research paper and provide a structured overview.

PAPER TEXT:
{paper_text[:15000]}  

Extract the following information and return as JSON:

{{
    "title": "paper title",
    "authors": "first author et al. or full list if short",
    "year": publication year as integer,
    "study_type": "Field" | "Greenhouse" | "Growth Chamber" | "FACE" | "OTC" | "Pot" | "Other",
    "crop_species": ["list of crops studied"],
    "country": "country where study conducted",
    "intervention_type": "what treatment/intervention was tested (e.g., nitrogen fertilizer, elevated CO2, tillage)",
    "outcomes_mentioned": ["list of outcome variables mentioned (e.g., yield, biomass, protein, minerals)"],
    "has_usable_data": true/false - whether paper has extractable quantitative data,
    "data_presentation": "Tables" | "Figures" | "Both" | "Text only",
    "notes": "any important observations about this paper"
}}

Return ONLY the JSON object, no other text."""


def get_outcomes_recon_prompt(paper_text: str, target_outcomes: list) -> str:
    """
    Scan paper for specific outcome variables
    """
    outcomes_str = ", ".join(target_outcomes)
    
    return f"""Analyze this paper to identify which outcome variables are reported.

TARGET OUTCOMES TO LOOK FOR:
{outcomes_str}

PAPER TEXT:
{paper_text[:15000]}

For each outcome variable, determine:
1. Is it present in the paper?
2. Where is it reported (Table number, Figure number, or text location)?
3. What units are used?
4. Are treatment means reported?
5. Is variance/error information reported? What type (SD, SE, LSD, CV)?

Return as JSON:
{{
    "outcomes": [
        {{
            "variable": "outcome name",
            "present": true/false,
            "location": "Table 2" or "Figure 3" or "Results section" or null,
            "unit": "reported unit" or null,
            "has_means": true/false,
            "has_variance": true/false,
            "variance_type": "SD" | "SE" | "LSD" | "CV" | "CI" | "none" | null,
            "notes": "any relevant notes"
        }}
    ],
    "other_outcomes_found": ["list any other quantitative outcomes in paper not in target list"]
}}

Return ONLY the JSON object."""


def get_moderators_recon_prompt(paper_text: str, target_moderators: list) -> str:
    """
    Scan paper for moderator variable availability
    """
    mod_list = "\n".join([f"- {m}" for m in target_moderators])
    
    return f"""Analyze this paper to identify which moderator variables are reported.

MODERATOR VARIABLES TO LOOK FOR:
{mod_list}

PAPER TEXT:
{paper_text[:15000]}

For each moderator, determine if it's reported and extract the value if available.

Return as JSON:
{{
    "moderators": [
        {{
            "variable": "moderator name",
            "available": true/false,
            "value": "the value if single value" or null,
            "values": ["list of values if multiple"] or [],
            "location": "where found in paper",
            "notes": "any relevant notes"
        }}
    ],
    "coordinates_available": true/false,
    "latitude": number or null,
    "longitude": number or null
}}

Return ONLY the JSON object."""


def get_design_recon_prompt(paper_text: str) -> str:
    """
    Identify experimental design details
    Critical for variance extraction
    """
    return f"""Analyze this paper to identify the experimental design.

PAPER TEXT (focus on Methods section):
{paper_text[:15000]}

Identify:
1. Experimental design type (RCBD, CRD, Split-plot, etc.)
2. If split-plot: what factor is in main plot vs subplot
3. Number of replicates
4. Plot size if mentioned
5. Number of years/seasons
6. Number of locations/sites

Return as JSON:
{{
    "design_type": "RCBD" | "CRD" | "Split-plot" | "Strip-plot" | "Latin Square" | "Factorial" | "Other" | "Unknown",
    "design_keywords_found": ["list of design-related terms found in text"],
    "is_split_plot": true/false,
    "main_plot_factor": "factor name" or null,
    "subplot_factor": "factor name" or null,
    "replicates": number or null,
    "replicate_term_used": "blocks" | "replicates" | "reps" | other term used,
    "plot_size": "size with units" or null,
    "num_years": number or null,
    "years_list": ["2018", "2019"] or [],
    "num_sites": number or null,
    "sites_list": ["site names"] or [],
    "notes": "any important design details"
}}

Return ONLY the JSON object."""


def get_variance_recon_prompt(paper_text: str) -> str:
    """
    Identify how variance is reported
    """
    return f"""Analyze this paper to understand how variance/error is reported.

PAPER TEXT:
{paper_text[:15000]}

Look for:
1. Standard Deviation (SD)
2. Standard Error (SE, SEM)
3. Confidence Intervals (CI, 95% CI)
4. Least Significant Difference (LSD)
5. Coefficient of Variation (CV, CV%)
6. Mean Square Error (MSE)
7. Significance letters (a, b, c groupings)
8. Error bars in figures

Return as JSON:
{{
    "variance_reported": true/false,
    "variance_type": "SD" | "SE" | "LSD" | "CV" | "CI" | "MSE" | "Letters" | "Multiple" | "None",
    "variance_location": "where variance is reported (e.g., 'Table 2 footnote', 'error bars in figures')",
    "per_treatment": true/false - is variance given for each treatment or pooled?,
    "lsd_details": {{
        "present": true/false,
        "alpha": 0.05 or other value,
        "type": "Fisher's" | "Tukey" | "Duncan" | "unspecified"
    }},
    "cv_reported": true/false,
    "cv_value": number or null,
    "significance_letters": true/false,
    "notes": "any relevant details about variance reporting"
}}

Return ONLY the JSON object."""


def get_control_recon_prompt(paper_text: str, domain: str, heuristic: dict) -> str:
    """
    Identify control/baseline treatments
    """
    keywords = ", ".join(heuristic.get('keywords', ['control', 'check', 'baseline']))
    
    return f"""Analyze this paper to identify the control/baseline treatment.

INTERVENTION DOMAIN: {domain}
EXPECTED CONTROL KEYWORDS: {keywords}

PAPER TEXT:
{paper_text[:15000]}

Identify:
1. How is the control/baseline defined?
2. What treatment levels are compared?
3. Is there a clear untreated or zero-input control?

Return as JSON:
{{
    "control_identified": true/false,
    "control_description": "description of control treatment",
    "control_keywords_found": ["keywords that helped identify control"],
    "treatment_levels": [
        {{
            "name": "treatment name/level",
            "description": "brief description",
            "is_control": true/false
        }}
    ],
    "comparison_type": "treatment vs zero" | "treatment vs conventional" | "treatment vs untreated" | "dose-response" | "factorial" | "other",
    "multiple_controls": true/false,
    "notes": "any important details about treatment structure"
}}

Return ONLY the JSON object."""


def get_cultivars_recon_prompt(paper_text: str) -> str:
    """
    Identify all cultivars/varieties/genotypes in the paper
    Critical for ensuring individual data extraction
    """
    return f"""Analyze this paper to identify ALL cultivars, varieties, or genotypes studied.

PAPER TEXT:
{paper_text[:15000]}

LOOK FOR:
1. Named cultivars/varieties (e.g., "Sakha 95", "IR64", "Mihan")
2. Genotype codes (e.g., "G1", "G2", "Variety A")
3. Whether data is reported individually per cultivar or averaged across cultivars

Return as JSON:
{{
    "cultivars_found": [
        {{
            "name": "cultivar/variety name",
            "species": "crop species if different from main",
            "notes": "any relevant notes"
        }}
    ],
    "total_cultivars": number,
    "data_structure": "individual" | "averaged" | "both",
    "individual_cultivar_tables": ["Table 2", "Table 3"],
    "averaged_cultivar_tables": ["Table 1"],
    "notes": "any important notes about cultivar data presentation"
}}

Return ONLY the JSON object."""


def get_data_tables_prompt(paper_text: str) -> str:
    """
    Identify and catalog data tables in paper
    """
    return f"""Analyze this paper to identify all data tables.

PAPER TEXT:
{paper_text[:20000]}

For each table with quantitative data, identify:
1. Table number/identifier
2. What variables/outcomes it contains
3. Whether it has the raw data we need for meta-analysis
4. Whether data is for individual cultivars or averaged across cultivars

Return as JSON:
{{
    "tables": [
        {{
            "table_id": "Table 1" or "Table S1" etc,
            "title": "table title/caption",
            "outcomes_in_table": ["list of outcome variables"],
            "has_treatment_means": true/false,
            "has_variance": true/false,
            "variance_type": "SD" | "SE" | "LSD" | etc or null,
            "has_sample_size": true/false,
            "is_summary_table": true/false - is this aggregated data or individual observations?,
            "factors_in_table": ["list of factors/treatments shown"],
            "cultivar_data": "individual" | "averaged" | "none" - whether table has individual cultivar data or averages,
            "cultivars_in_table": ["list cultivar names if individual data"],
            "usability": "high" | "medium" | "low" - how useful for meta-analysis extraction,
            "notes": "any relevant notes"
        }}
    ],
    "best_tables_for_extraction": ["Table 2", "Table 3"] - ordered by usefulness,
    "figures_with_data": ["Figure 1", "Figure 2"] - figures that might have extractable data,
    "supplementary_data": true/false - is there supplementary material mentioned?
}}

Return ONLY the JSON object."""


# Compile all prompts into a dictionary for easy access
RECON_PROMPTS = {
    'overview': get_overview_prompt,
    'outcomes': get_outcomes_recon_prompt,
    'moderators': get_moderators_recon_prompt,
    'design': get_design_recon_prompt,
    'variance': get_variance_recon_prompt,
    'control': get_control_recon_prompt,
    'tables': get_data_tables_prompt,
    'cultivars': get_cultivars_recon_prompt
}
