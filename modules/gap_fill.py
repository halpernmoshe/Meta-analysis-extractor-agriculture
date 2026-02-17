"""
Gap Fill Module - Targeted extraction for missing values

This module analyzes existing extraction results and makes focused API calls
to find missing variance, sample sizes, and moderators without re-extracting
everything.

Usage:
    python -m modules.gap_fill --input ./output --papers ./input --provider google --api-key YOUR_KEY
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import re


@dataclass
class GapAnalysis:
    """Analysis of what's missing in the extraction"""
    paper_id: str
    missing_variance: List[str]  # observation_ids missing variance
    missing_n: List[str]  # observation_ids missing sample size
    missing_moderators: Dict[str, List[str]]  # moderator -> observation_ids
    total_observations: int
    variance_coverage: float
    n_coverage: float


@dataclass
class FillResult:
    """Result of gap filling for one paper"""
    paper_id: str
    variance_filled: int
    n_filled: int
    moderators_filled: Dict[str, int]
    errors: List[str]


class GapFillModule:
    """Module for filling gaps in extracted data"""

    def __init__(self, llm_client, output_dir: str, papers_dir: str):
        self.llm = llm_client
        self.output_dir = Path(output_dir)
        self.papers_dir = Path(papers_dir)
        self.summary_path = self.output_dir / 'summary.csv'
        self.filled_path = self.output_dir / 'summary_filled.csv'

    def analyze_gaps(self) -> Dict[str, GapAnalysis]:
        """Analyze existing extraction to find gaps"""
        if not self.summary_path.exists():
            raise FileNotFoundError(f"No summary.csv found at {self.summary_path}")

        df = pd.read_csv(self.summary_path)

        # Group by paper
        gaps_by_paper = {}

        for paper_id, group in df.groupby('paper_id'):
            # Find observations missing variance
            missing_var = group[
                group['treatment_variance'].isna() &
                group['control_variance'].isna() &
                group['pooled_variance'].isna()
            ]['observation_id'].tolist()

            # Find observations missing sample size
            missing_n = group[
                group['treatment_n'].isna()
            ]['observation_id'].tolist()

            # Find observations missing key moderators
            key_moderators = ['mod_COUNTRY', 'mod_CROP_SP', 'mod_STUDY_TYPE',
                            'mod_SL_PH', 'mod_STRESS_TYPE']
            missing_mods = {}
            for mod in key_moderators:
                if mod in group.columns:
                    missing = group[group[mod].isna()]['observation_id'].tolist()
                    if missing:
                        missing_mods[mod] = missing

            total = len(group)
            var_coverage = 1 - (len(missing_var) / total) if total > 0 else 1
            n_coverage = 1 - (len(missing_n) / total) if total > 0 else 1

            # Only include papers that have gaps
            if missing_var or missing_n or missing_mods:
                gaps_by_paper[paper_id] = GapAnalysis(
                    paper_id=paper_id,
                    missing_variance=missing_var,
                    missing_n=missing_n,
                    missing_moderators=missing_mods,
                    total_observations=total,
                    variance_coverage=var_coverage,
                    n_coverage=n_coverage
                )

        return gaps_by_paper

    def print_gap_summary(self, gaps: Dict[str, GapAnalysis]):
        """Print summary of gaps found"""
        print("\n" + "="*60)
        print("GAP ANALYSIS SUMMARY")
        print("="*60)

        total_missing_var = sum(len(g.missing_variance) for g in gaps.values())
        total_missing_n = sum(len(g.missing_n) for g in gaps.values())
        total_obs = sum(g.total_observations for g in gaps.values())

        print(f"\nPapers with gaps: {len(gaps)}")
        print(f"Total observations in gap papers: {total_obs}")
        print(f"Missing variance: {total_missing_var} observations")
        print(f"Missing sample size: {total_missing_n} observations")

        # Papers sorted by gap severity
        print(f"\n{'Paper':<15} {'Total':<8} {'Missing Var':<12} {'Missing n':<10}")
        print("-" * 50)

        sorted_gaps = sorted(gaps.values(),
                           key=lambda x: len(x.missing_variance),
                           reverse=True)

        for gap in sorted_gaps[:15]:  # Show top 15
            print(f"{gap.paper_id:<15} {gap.total_observations:<8} "
                  f"{len(gap.missing_variance):<12} {len(gap.missing_n):<10}")

        if len(sorted_gaps) > 15:
            print(f"... and {len(sorted_gaps) - 15} more papers")

    def _get_paper_text(self, paper_id: str) -> Optional[str]:
        """Get text content from PDF"""
        # Find PDF file
        pdf_path = None
        for ext in ['.pdf', '.PDF']:
            candidate = self.papers_dir / f"{paper_id}{ext}"
            if candidate.exists():
                pdf_path = candidate
                break

        if not pdf_path:
            # Try to find by pattern
            for f in self.papers_dir.glob(f"{paper_id}*"):
                if f.suffix.lower() == '.pdf':
                    pdf_path = f
                    break

        if not pdf_path:
            return None

        # Extract text using pdfplumber or pymupdf
        try:
            import fitz  # pymupdf
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text[:100000]  # Limit to 100k chars
        except ImportError:
            try:
                import pdfplumber
                with pdfplumber.open(pdf_path) as pdf:
                    text = ""
                    for page in pdf.pages:
                        text += page.extract_text() or ""
                return text[:100000]
            except ImportError:
                print(f"Warning: No PDF library available (install pymupdf or pdfplumber)")
                return None

    def _build_variance_prompt(self, paper_id: str, observations: List[Dict]) -> str:
        """Build targeted prompt to find variance values"""
        obs_context = []
        for obs in observations[:20]:  # Limit to 20 observations per query
            obs_context.append(
                f"- {obs.get('outcome_variable', 'Unknown')}: "
                f"treatment={obs.get('treatment_mean', '?')}, "
                f"control={obs.get('control_mean', '?')} "
                f"(from {obs.get('data_source', 'unknown source')})"
            )

        return f"""Find the variance/error values for these specific observations from paper {paper_id}.

OBSERVATIONS NEEDING VARIANCE:
{chr(10).join(obs_context)}

SEARCH LOCATIONS (in order of priority):
1. Table footnotes - look for "± SE", "± SD", "LSD(0.05)=", "Values are means ± SE (n=X)"
2. Column headers with ± symbol
3. Separate SE/SD columns in tables
4. Methods section - "Data are presented as mean ± standard error"
5. Figure legends - "Error bars represent SE/SD"
6. Results text - "mean of X ± Y"

IMPORTANT:
- Extract the ACTUAL NUMERIC VALUES, not just the type
- If LSD is reported, extract the LSD value
- If SE/SD differs between treatments, note both values
- Report variance_type as one of: SE, SD, LSD, MSE, CV

Return JSON array with format:
[
  {{
    "outcome_variable": "grain yield",
    "treatment_mean": 45.2,
    "variance_type": "SE",
    "treatment_variance": 2.3,
    "control_variance": 1.9,
    "pooled_variance": null,
    "source": "Table 2 footnote"
  }}
]

If variance truly not reported for an observation, return null for variance values.
Return ONLY the JSON array, no other text."""

    def _build_sample_size_prompt(self, paper_id: str) -> str:
        """Build targeted prompt to find sample size"""
        return f"""Find the sample size (n) / number of replicates for paper {paper_id}.

SEARCH LOCATIONS:
1. Methods section - "n=4", "4 replicates", "four replications per treatment"
2. Experimental design - "randomized complete block design with 3 replications"
3. Table footnotes - "Values are means of 4 replicates"
4. Statistical analysis - "Data were analyzed with n=5 per treatment"
5. Figure legends - "n=3 per group"

IMPORTANT:
- Look for replicates per treatment, not total sample size
- If different experiments have different n, list each
- Common terms: replicates, replications, repetitions, n, sample size

Return JSON:
{{
  "sample_size": 4,
  "source": "Methods section: 'four replications per treatment'",
  "applies_to": "all experiments" or "specific experiment name",
  "confidence": "high" or "medium" or "low"
}}

Return ONLY valid JSON, no other text."""

    def _build_moderator_prompt(self, paper_id: str, moderators: List[str]) -> str:
        """Build targeted prompt to find moderator values"""
        mod_descriptions = {
            'mod_COUNTRY': 'Country where study was conducted',
            'mod_CROP_SP': 'Crop species (scientific name preferred)',
            'mod_STUDY_TYPE': 'Study type: Field, Greenhouse, Growth Chamber',
            'mod_SL_PH': 'Soil pH value',
            'mod_SL_TEXTURE': 'Soil texture class',
            'mod_STRESS_TYPE': 'Stress treatment: Drought, Salinity, Heat, None',
            'mod_LAT': 'Latitude of study site',
            'mod_LON': 'Longitude of study site',
            'mod_CULTIVAR': 'Cultivar/variety name',
            'mod_FERT_N': 'Nitrogen fertilizer rate (kg/ha)',
        }

        mod_list = []
        for mod in moderators:
            desc = mod_descriptions.get(mod, mod.replace('mod_', ''))
            mod_list.append(f"- {mod}: {desc}")

        return f"""Find these specific moderator values for paper {paper_id}:

{chr(10).join(mod_list)}

SEARCH LOCATIONS:
1. Materials and Methods section
2. Study site description
3. Table 1 (often has site characteristics)
4. Supplementary materials

Return JSON:
{{
  "mod_COUNTRY": "Egypt",
  "mod_SL_PH": 7.8,
  "mod_STUDY_TYPE": "Field",
  ...
}}

Use null for values not found. Return ONLY valid JSON."""

    def _parse_json_response(self, response: str) -> Any:
        """Parse JSON from LLM response, handling common issues"""
        # Clean up response
        text = response.strip()

        # Remove markdown code blocks
        if text.startswith('```'):
            text = re.sub(r'^```(?:json)?\n?', '', text)
            text = re.sub(r'\n?```$', '', text)

        # Try to find JSON in response
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON object or array
            match = re.search(r'(\{[^{}]*\}|\[[^\[\]]*\])', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except:
                    pass

        return None

    def fill_paper_gaps(self, gap: GapAnalysis, df: pd.DataFrame) -> FillResult:
        """Fill gaps for a single paper"""
        result = FillResult(
            paper_id=gap.paper_id,
            variance_filled=0,
            n_filled=0,
            moderators_filled={},
            errors=[]
        )

        # Get paper text
        paper_text = self._get_paper_text(gap.paper_id)
        if not paper_text:
            result.errors.append(f"Could not read PDF for {gap.paper_id}")
            return result

        paper_df = df[df['paper_id'] == gap.paper_id].copy()

        # 1. Fill missing variance
        if gap.missing_variance:
            try:
                # Get observations needing variance
                missing_obs = paper_df[
                    paper_df['observation_id'].isin(gap.missing_variance)
                ].to_dict('records')

                prompt = self._build_variance_prompt(gap.paper_id, missing_obs)

                response = self.llm.call(
                    prompt=f"Paper text:\n{paper_text[:50000]}\n\n{prompt}",
                    system_prompt="You are a precise data extractor. Extract only values explicitly stated in the paper. Never invent or estimate values.",
                    response_format='text'
                )

                variance_data = self._parse_json_response(response)

                if variance_data and isinstance(variance_data, list):
                    for var_item in variance_data:
                        if not var_item:
                            continue
                        # Match to observations and update
                        outcome = var_item.get('outcome_variable', '')
                        trt_mean = var_item.get('treatment_mean')

                        # Find matching observation
                        for idx, row in paper_df.iterrows():
                            if (row['observation_id'] in gap.missing_variance and
                                outcome.lower() in str(row.get('outcome_variable', '')).lower()):

                                # Update variance values
                                if var_item.get('treatment_variance') is not None:
                                    df.loc[df['observation_id'] == row['observation_id'],
                                          'treatment_variance'] = var_item['treatment_variance']
                                    result.variance_filled += 1

                                if var_item.get('control_variance') is not None:
                                    df.loc[df['observation_id'] == row['observation_id'],
                                          'control_variance'] = var_item['control_variance']

                                if var_item.get('pooled_variance') is not None:
                                    df.loc[df['observation_id'] == row['observation_id'],
                                          'pooled_variance'] = var_item['pooled_variance']
                                    if var_item.get('treatment_variance') is None:
                                        result.variance_filled += 1

                                if var_item.get('variance_type'):
                                    df.loc[df['observation_id'] == row['observation_id'],
                                          'variance_type'] = var_item['variance_type']
                                break

            except Exception as e:
                result.errors.append(f"Variance fill error: {str(e)}")

        # 2. Fill missing sample size
        if gap.missing_n:
            try:
                prompt = self._build_sample_size_prompt(gap.paper_id)

                response = self.llm.call(
                    prompt=f"Paper text:\n{paper_text[:30000]}\n\n{prompt}",
                    system_prompt="You are a precise data extractor. Extract only values explicitly stated in the paper.",
                    response_format='text'
                )

                n_data = self._parse_json_response(response)

                if n_data and isinstance(n_data, dict):
                    n_value = n_data.get('sample_size')
                    if n_value is not None:
                        # Apply to all missing observations in this paper
                        for obs_id in gap.missing_n:
                            df.loc[df['observation_id'] == obs_id, 'treatment_n'] = n_value
                            df.loc[df['observation_id'] == obs_id, 'control_n'] = n_value
                            result.n_filled += 1

            except Exception as e:
                result.errors.append(f"Sample size fill error: {str(e)}")

        # 3. Fill missing moderators
        if gap.missing_moderators:
            try:
                mods_to_find = list(gap.missing_moderators.keys())
                prompt = self._build_moderator_prompt(gap.paper_id, mods_to_find)

                response = self.llm.call(
                    prompt=f"Paper text:\n{paper_text[:30000]}\n\n{prompt}",
                    system_prompt="You are a precise data extractor. Extract only values explicitly stated in the paper.",
                    response_format='text'
                )

                mod_data = self._parse_json_response(response)

                if mod_data and isinstance(mod_data, dict):
                    for mod_name, value in mod_data.items():
                        if value is not None and mod_name in gap.missing_moderators:
                            # Apply to all observations missing this moderator
                            for obs_id in gap.missing_moderators[mod_name]:
                                df.loc[df['observation_id'] == obs_id, mod_name] = value
                            result.moderators_filled[mod_name] = len(gap.missing_moderators[mod_name])

            except Exception as e:
                result.errors.append(f"Moderator fill error: {str(e)}")

        return result

    def run(self, max_papers: int = None, priority: str = 'variance') -> pd.DataFrame:
        """
        Run gap filling process

        Args:
            max_papers: Maximum number of papers to process (None = all)
            priority: 'variance', 'n', or 'all' - what to prioritize filling

        Returns:
            Updated DataFrame
        """
        print("\n" + "="*60)
        print("GAP FILL MODULE")
        print("="*60)

        # 1. Analyze gaps
        print("\nAnalyzing existing extraction for gaps...")
        gaps = self.analyze_gaps()

        if not gaps:
            print("No gaps found - extraction is complete!")
            return pd.read_csv(self.summary_path)

        self.print_gap_summary(gaps)

        # 2. Load current data
        df = pd.read_csv(self.summary_path)

        # 3. Sort papers by priority
        if priority == 'variance':
            sorted_papers = sorted(gaps.keys(),
                                 key=lambda x: len(gaps[x].missing_variance),
                                 reverse=True)
        elif priority == 'n':
            sorted_papers = sorted(gaps.keys(),
                                 key=lambda x: len(gaps[x].missing_n),
                                 reverse=True)
        else:
            sorted_papers = sorted(gaps.keys(),
                                 key=lambda x: (len(gaps[x].missing_variance) +
                                              len(gaps[x].missing_n)),
                                 reverse=True)

        if max_papers:
            sorted_papers = sorted_papers[:max_papers]

        # 4. Process each paper
        print(f"\nProcessing {len(sorted_papers)} papers...")

        total_var_filled = 0
        total_n_filled = 0
        total_mod_filled = 0

        for i, paper_id in enumerate(sorted_papers):
            gap = gaps[paper_id]
            print(f"\n[{i+1}/{len(sorted_papers)}] {paper_id}...")
            print(f"  Missing: {len(gap.missing_variance)} variance, {len(gap.missing_n)} n")

            result = self.fill_paper_gaps(gap, df)

            total_var_filled += result.variance_filled
            total_n_filled += result.n_filled
            total_mod_filled += sum(result.moderators_filled.values())

            print(f"  Filled: {result.variance_filled} variance, {result.n_filled} n, "
                  f"{sum(result.moderators_filled.values())} moderators")

            if result.errors:
                for err in result.errors:
                    print(f"  Warning: {err}")

        # 5. Save results
        print(f"\n" + "="*60)
        print("GAP FILL COMPLETE")
        print("="*60)
        print(f"Variance values filled: {total_var_filled}")
        print(f"Sample sizes filled: {total_n_filled}")
        print(f"Moderator values filled: {total_mod_filled}")

        # Save to new file (preserve original)
        df.to_csv(self.filled_path, index=False)
        print(f"\nSaved to: {self.filled_path}")

        # Also update original
        df.to_csv(self.summary_path, index=False)
        print(f"Updated: {self.summary_path}")

        return df


def main():
    """CLI entry point for gap fill module"""
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from core.llm import LLMClient
    from config import GOOGLE_API_KEY, ANTHROPIC_API_KEY

    parser = argparse.ArgumentParser(description='Fill gaps in meta-analysis extraction')
    parser.add_argument('--input', '-i', required=True, help='Output directory with summary.csv')
    parser.add_argument('--papers', '-p', required=True, help='Directory containing PDF papers')
    parser.add_argument('--provider', choices=['google', 'anthropic'], default='google')
    parser.add_argument('--api-key', help='API key (or use env variable)')
    parser.add_argument('--max-papers', type=int, help='Max papers to process')
    parser.add_argument('--priority', choices=['variance', 'n', 'all'], default='variance',
                       help='What to prioritize filling')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze gaps, do not fill')

    args = parser.parse_args()

    # Initialize LLM client
    api_key = args.api_key
    if not api_key:
        api_key = GOOGLE_API_KEY if args.provider == 'google' else ANTHROPIC_API_KEY

    if not api_key:
        print(f"Error: No API key provided for {args.provider}")
        sys.exit(1)

    llm = LLMClient(
        api_key=api_key,
        google_api_key=api_key if args.provider == 'google' else None,
        provider=args.provider
    )

    # Run gap fill
    gap_filler = GapFillModule(llm, args.input, args.papers)

    if args.analyze_only:
        gaps = gap_filler.analyze_gaps()
        gap_filler.print_gap_summary(gaps)
    else:
        gap_filler.run(max_papers=args.max_papers, priority=args.priority)


if __name__ == '__main__':
    main()
