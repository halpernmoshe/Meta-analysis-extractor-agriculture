#!/usr/bin/env python3
"""Convert PAPER_FINAL_v23.md references from APA author-date to AMA numbered style."""
import re
from pathlib import Path

MD_FILE = Path(__file__).parent / "PAPER_FINAL_v23.md"

# ── AMA-formatted references in order of first text appearance ──────────────
# Number assignments based on first in-text occurrence
AMA_REFS = {
    1: 'Borah R, Brown AW, Capers PL, Kaiser KA. Analysis of the time and workers needed to conduct systematic reviews of medical interventions using data from the PROSPERO registry. *BMJ Open*. 2017;7(4):e012545.',
    2: 'Shojania KG, Sampson M, Ansari MT, Ji J, Doucette S, Moher D. How quickly do systematic reviews go out of date? A survival analysis. *Ann Intern Med*. 2007;147(4):224-233.',
    3: 'Schmidt L, Shokraneh F, Pieper D, Mathes T. Data extraction methods for systematic review (semi)automation: update of a living systematic review. *F1000Research*. 2025.',
    4: 'Buscemi N, Hartling L, Vandermeer B, Tjosvold L, Klassen TP. Single data extraction generated more errors than double data extraction in systematic reviews. *J Clin Epidemiol*. 2006;59(7):697-703.',
    5: 'Higgins JPT, Thomas J, Chandler J, et al, eds. *Cochrane Handbook for Systematic Reviews of Interventions*. Version 6.4. Cochrane; 2023.',
    6: 'Mathes T, Klassen P, Pieper D. Frequency of data extraction errors and methods to increase data extraction quality: a methodological review. *BMC Med Res Methodol*. 2017;17:152.',
    7: 'Topp CFE, et al. AgroEcoList: a checklist to improve reporting of ecological research in agronomy. *PLoS One*. 2023;18(6):e0285478.',
    8: 'Nakagawa S, et al. A robust and readily implementable method for the meta-analysis of response ratios with and without missing standard deviations. *Ecol Lett*. 2023;26(2):232-244.',
    9: 'Gougherty AV, Clipp HL. Testing the reliability of an AI-based large language model to extract ecological information from the scientific literature. *npj Biodivers*. 2024;3(1):13.',
    10: 'Helms Andersen T, et al. Using AI tools as second reviewers in systematic reviews. *Cochrane Evid Synth Methods*. 2025.',
    11: 'Jansen T, et al. Data extraction by generative artificial intelligence. *Psychol Bull*. 2025;151(10):1280-1306.',
    12: 'Kataoka Y, et al. Automating the data extraction process for systematic reviews using GPT-4o and o3. *Res Synth Methods*. 2026;17:42-62.',
    13: 'Cao X, et al. OttoSR: automation of systematic reviews with large language models. *medRxiv*. 2025. doi:10.1101/2025.01.15.25320588',
    14: 'Gartlehner G, et al. Artificial intelligence-assisted data extraction with a large language model: a study within reviews. *Ann Intern Med*. 2025.',
    15: 'Poser PL, Klimas R, Luerweg J, et al. Improving reliability and accuracy of structured data extraction using a consensus large-language model approach. *Front Artif Intell*. 2026.',
    16: 'Khan MA, Ayub U, Naqvi SAA, et al. Collaborative large language models for automated data extraction in living systematic reviews. *J Am Med Inform Assoc*. 2025;32(4):638-647.',
    17: 'Anthropic. Claude API pricing. Accessed March 2026. https://www.anthropic.com/pricing',
    18: 'Hui X, Luo L, Chen Y, Palta JA, Wang Z. Zinc agronomic biofortification in wheat and its drivers: a global meta-analysis. *Nat Commun*. 2025;16:3913.',
    19: 'Li J, Van Gerrewey T, Geelen D. A meta-analysis of biostimulant yield effectiveness in field trials. *Front Plant Sci*. 2022;13:836702.',
    20: 'Li S, et al. Biochar increases crop yield: a meta-analysis. figshare. 2024. doi:10.6084/m9.figshare.c.6622375.v1',
    21: 'Boldorini E, Lucchi A, Tamburini G. Predator-mediated biocontrol of crop arthropod pests and their damage: a global meta-analysis. *Proc R Soc B*. 2024;291:20232522.',
    22: 'Loladze I. Hidden shift of the ionome of plants exposed to elevated CO2 depletes minerals at the base of human nutrition. *eLife*. 2014;3:e02245.',
    23: 'Koo TK, Li MY. A guideline of selecting and reporting intraclass correlation coefficients for reliability research. *J Chiropr Med*. 2016;15(2):155-163.',
    24: 'Lin LI-K. A concordance correlation coefficient to evaluate reproducibility. *Biometrics*. 1989;45(1):255-268.',
    25: 'Pustejovsky JE, Tipton E. Small-sample methods for cluster-robust variance estimation and hypothesis testing in fixed effects models. *J Bus Econ Stat*. 2018;36(4):672-683.',
    26: 'Bland JM, Altman DG. Statistical methods for assessing agreement between two methods of clinical measurement. *Lancet*. 1986;327(8476):307-310.',
    27: 'Bureau of Labor Statistics. Occupational employment and wages: social science research assistants. US Department of Labor. 2025. https://www.bls.gov/oes/current/oes194061.htm',
    28: 'Elliott JH, Turner T, Clavisi O, et al. Living systematic reviews: an emerging opportunity to narrow the evidence-practice gap. *PLoS Med*. 2014;11(2):e1001603.',
    29: 'Marshall IJ, Kuiper J, Wallace BC. RobotReviewer: evaluation of a system for automatically assessing bias in clinical trials. *J Am Med Inform Assoc*. 2016;23(1):193-201.',
    30: 'Tendal B, Higgins JPT, Juni P, Hrobjartsson A, Gotzsche PC. Multiplicity of data in trial reports and the reliability of meta-analyses: empirical study. *BMJ*. 2009;339:b3128.',
    31: 'Flemyng E, et al; Cochrane, Campbell Collaboration, JBI, CEE. Position statement on the use of artificial intelligence in the production of evidence syntheses. *Cochrane Database Syst Rev*. 2025.',
    32: 'Gartlehner G, Kahwati L, Engeli C, Hamel C, Gaisinger K, Glechner A. Data extraction for evidence synthesis using a large language model: a proof-of-concept study. *Res Synth Methods*. 2024;15(4):576-582.',
    33: 'Khraisha Q, et al. Can large language models replace humans in systematic reviews? A study of LLM performance in screening and extracting data. *Res Synth Methods*. 2024;15(4):616-626.',
    34: 'Li L, Mathrani A, Susnjak T. What level of automation is "good enough"? A benchmark of large language models for meta-analysis data extraction. *arXiv*. 2025:2507.15152.',
}

# ── Citation replacement rules ──────────────────────────────────────────────
# Each tuple: (regex_pattern, replacement)
# Order matters -- more specific patterns first to avoid partial matches

CITATION_RULES = [
    # === Section 1.1 (line 46-48) ===
    # "67.3 weeks (Borah et al., 2017)"
    (r'\(Borah et al\., 2017\)', '^1^'),
    # "within two years of publication (Shojania et al., 2007)"
    (r'\(Shojania et al\., 2007\)', '^2^'),
    # "2--8 hours per paper (Schmidt et al., 2025)"
    (r'\(Schmidt et al\., 2025\)', '^3^'),
    # "Schmidt et al. (2025) estimate" - narrative
    (r'Schmidt et al\. \(2025\)', 'Schmidt et al^3^'),
    # "falling to 8.8% only under costly dual-extraction protocols (Buscemi et al., 2006)"
    (r'\(Buscemi et al\., 2006\)', '^4^'),
    # "Buscemi et al. (2006) documented" - narrative
    (r'Buscemi et al\. \(2006\)', 'Buscemi et al^4^'),
    # "Cochrane Handbook's recommendation... (Higgins et al., 2023)"
    (r'\(Higgins et al\., 2023\)', '^5^'),
    # "Higgins et al., 2023" bare inside parens with other text (like "as recommended by...")
    # "66.8% ... (Mathes et al., 2017)"
    (r'\(Mathes et al\., 2017\)', '^6^'),

    # === Section 1.2 (line 52) ===
    # "like CONSORT (Topp et al., 2023)"
    (r'\(Topp et al\., 2023\)', '^7^'),
    # "Topp et al. (2023) documented" - narrative
    (r'Topp et al\. \(2023\)', 'Topp et al^7^'),
    # "(Nakagawa et al., 2023)"
    (r'\(Nakagawa et al\., 2023\)', '^8^'),

    # === Section 1.3 (line 56-58) ===
    # "Gougherty and Clipp (2024) reported"
    (r'Gougherty and Clipp \(2024\)', 'Gougherty and Clipp^9^'),
    # "(Gougherty and Clipp, 2024)" - not present but just in case
    (r'\(Gougherty and Clipp, 2024\)', '^9^'),
    # "Helms Andersen et al. (2025) found"
    (r'Helms Andersen et al\. \(2025\)', 'Helms Andersen et al^10^'),
    # "Jansen et al. (2025) evaluated"
    (r'Jansen et al\. \(2025\)', 'Jansen et al^11^'),
    # "Jansen et al. (2025; 26--36%)" in Discussion
    (r'Jansen et al\. \(2025; ', 'Jansen et al^11^ ('),
    # "Kataoka et al. (2026) tested"
    (r'Kataoka et al\. \(2026\)', 'Kataoka et al^12^'),
    # "Cao et al. (2025) developed"
    (r'Cao et al\. \(2025\)', 'Cao et al^13^'),
    # "Gartlehner et al. (2025) found"
    (r'Gartlehner et al\. \(2025\)', 'Gartlehner et al^14^'),
    # "Poser et al. (2026) showed"
    (r'Poser et al\. \(2026\)', 'Poser et al^15^'),
    # "Khan et al. (2025) demonstrated"
    (r'Khan et al\. \(2025\)', 'Khan et al^16^'),

    # === Section 2.1 (line 83, 111) ===
    # "(Anthropic, 2026..." - may appear multiple ways
    (r'\(Anthropic, 2026; \$5', '^17^ ($5'),
    (r'\(Anthropic, 2026\)', '^17^'),
    (r'Anthropic, 2026\)', 'Anthropic^17^)'),

    # === Section 2.2 (line 128-136) ===
    # "(Hui et al., 2025)"
    (r'\(Hui et al\., 2025\)', '^18^'),
    # "(Li et al., 2022)"
    (r'\(Li et al\., 2022\)', '^19^'),
    # "(Li et al., 2024)"
    (r'\(Li et al\., 2024\)', '^20^'),
    # "(Boldorini et al., 2024)"
    (r'\(Boldorini et al\., 2024\)', '^21^'),
    # "(Loladze, 2014)"
    (r'\(Loladze, 2014\)', '^22^'),
    # "Loladze (2014)" narrative
    (r'Loladze \(2014\)', 'Loladze^22^'),

    # === Section 2.4 (line 167-189) ===
    # "Koo and Li (2016)"
    (r'Koo and Li \(2016\)', 'Koo and Li^23^'),
    # "(Lin, 1989)"
    (r'\(Lin, 1989\)', '^24^'),
    # "(Pustejovsky & Tipton, 2018)"
    (r'\(Pustejovsky & Tipton, 2018\)', '^25^'),
    (r'\(Pustejovsky and Tipton, 2018\)', '^25^'),
    # "(Bland & Altman, 1986)"
    (r'\(Bland & Altman, 1986\)', '^26^'),
    (r'\(Bland and Altman, 1986\)', '^26^'),

    # === Section 4.6 (line 396-398) ===
    # "Bureau of Labor Statistics (2025)"
    (r'Bureau of Labor Statistics \(2025\)', 'Bureau of Labor Statistics^27^'),
    # "(Higgins et al., 2023)" already handled above

    # === Section 4.6 (line 400) ===
    # "(Elliott et al., 2014)"
    (r'\(Elliott et al\., 2014\)', '^28^'),

    # === Table 10 (line 413) ===
    # "Marshall et al. 2016" in table cell (no parens)
    (r'Marshall et al\. 2016', 'Marshall et al, 2016^29^'),
    # Other table cells - these have "Author et al. YYYY" format without parens
    (r'Jansen et al\. 2025', 'Jansen et al, 2025^11^'),
    (r'Kataoka et al\. 2026', 'Kataoka et al, 2026^12^'),
    (r'Poser et al\. 2026', 'Poser et al, 2026^15^'),
    (r'Cao et al\. 2025', 'Cao et al, 2025^13^'),
    (r'Gartlehner et al\. 2025', 'Gartlehner et al, 2025^14^'),
    (r'Khan et al\. 2025', 'Khan et al, 2025^16^'),

    # === Section 4.8 (line 424) ===
    # "Tendal et al. (2009) found"
    (r'Tendal et al\. \(2009\)', 'Tendal et al^30^'),

    # === Sections 4.9, 4.11 ===
    # "Cochrane/Campbell/JBI/CEE (2025)"
    (r'Cochrane/Campbell/JBI/CEE \(2025\)', 'Cochrane/Campbell/JBI/CEE^31^'),
]


def convert():
    text = MD_FILE.read_text(encoding='utf-8')

    # ── Apply citation replacements ─────────────────────────────────────
    for pattern, replacement in CITATION_RULES:
        text = re.sub(pattern, replacement, text)

    # ── Replace the old reference block ─────────────────────────────────
    # Find everything from "# References" to the next "---" or "## Declarations"
    ref_start = text.find('# References')
    ref_end = text.find('---\n\n## Declarations')

    if ref_start == -1 or ref_end == -1:
        print("ERROR: Could not find reference block boundaries")
        return

    new_refs = ['# References\n']
    for num in sorted(AMA_REFS.keys()):
        new_refs.append(f'{num}. {AMA_REFS[num]}')
    new_refs.append('')  # trailing newline

    text = text[:ref_start] + '\n'.join(new_refs) + '\n' + text[ref_end:]

    # ── Write output ────────────────────────────────────────────────────
    MD_FILE.write_text(text, encoding='utf-8')
    print(f"Converted {len(AMA_REFS)} references to AMA numbered style.")
    print("Done.")


if __name__ == '__main__':
    convert()
