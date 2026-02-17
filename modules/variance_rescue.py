"""
Variance Rescue Module - Vision-based extraction for missing variance values

This module targets observations where variance_type was detected but numeric
values weren't extracted. It uses vision API to read tables as images and
extract variance values from footnotes.

Usage:
    python -m modules.variance_rescue --input ./output --papers ./input --provider google
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import re
import fitz  # PyMuPDF
from io import BytesIO


@dataclass
class RescueTarget:
    """Observation needing variance rescue"""
    paper_id: str
    observation_id: str
    outcome_variable: str
    treatment_mean: float
    control_mean: float
    variance_type: str  # The type that was detected (LSD, SE, etc.)
    data_source: str  # Table name or figure reference


@dataclass
class RescueResult:
    """Result of variance rescue for one paper"""
    paper_id: str
    targets_found: int
    values_rescued: int
    tables_processed: int
    errors: List[str]


class VarianceRescueModule:
    """Module for rescuing variance values using vision-based extraction"""

    def __init__(self, llm_client, output_dir: str, papers_dir: str):
        self.llm = llm_client
        self.output_dir = Path(output_dir)
        self.papers_dir = Path(papers_dir)
        self.summary_path = self.output_dir / 'summary.csv'

    def find_rescue_targets(self) -> Dict[str, List[RescueTarget]]:
        """Find observations with variance_type but no variance values"""
        if not self.summary_path.exists():
            raise FileNotFoundError(f"No summary.csv found at {self.summary_path}")

        df = pd.read_csv(self.summary_path)

        # Find observations with variance_type but no values
        needs_rescue = df[
            df['variance_type'].notna() &
            df['treatment_variance'].isna() &
            df['control_variance'].isna() &
            df['pooled_variance'].isna()
        ]

        targets_by_paper = {}
        for _, row in needs_rescue.iterrows():
            paper_id = row['paper_id']
            if paper_id not in targets_by_paper:
                targets_by_paper[paper_id] = []

            targets_by_paper[paper_id].append(RescueTarget(
                paper_id=paper_id,
                observation_id=row['observation_id'],
                outcome_variable=row.get('outcome_variable', ''),
                treatment_mean=row.get('treatment_mean'),
                control_mean=row.get('control_mean'),
                variance_type=row['variance_type'],
                data_source=row.get('data_source', '')
            ))

        return targets_by_paper

    def print_rescue_summary(self, targets: Dict[str, List[RescueTarget]]):
        """Print summary of rescue targets"""
        print("\n" + "="*60)
        print("VARIANCE RESCUE TARGETS")
        print("="*60)

        total_targets = sum(len(t) for t in targets.values())
        print(f"\nPapers needing rescue: {len(targets)}")
        print(f"Total observations to rescue: {total_targets}")

        print(f"\n{'Paper':<15} {'Targets':<10} {'Variance Type':<15}")
        print("-" * 45)

        for paper_id, paper_targets in sorted(targets.items(),
                                               key=lambda x: len(x[1]),
                                               reverse=True)[:15]:
            var_types = set(t.variance_type for t in paper_targets)
            var_type_str = ', '.join(var_types)
            print(f"{paper_id:<15} {len(paper_targets):<10} {var_type_str:<15}")

        if len(targets) > 15:
            print(f"... and {len(targets) - 15} more papers")

    def _get_pdf_path(self, paper_id: str) -> Optional[Path]:
        """Find PDF file for paper"""
        for ext in ['.pdf', '.PDF']:
            candidate = self.papers_dir / f"{paper_id}{ext}"
            if candidate.exists():
                return candidate

        # Try pattern match
        for f in self.papers_dir.glob(f"{paper_id}*"):
            if f.suffix.lower() == '.pdf':
                return f

        return None

    def _extract_table_pages(self, pdf_path: Path, table_refs: List[str]) -> List[Tuple[int, bytes]]:
        """Extract pages containing tables as images"""
        doc = fitz.open(pdf_path)
        table_pages = []

        # Parse table references to find page numbers
        table_names = set()
        for ref in table_refs:
            # Extract table number from references like "Table 2", "Table 3_row4"
            match = re.search(r'[Tt]able\s*(\d+)', ref)
            if match:
                table_names.add(f"Table {match.group(1)}")

        # Search for pages containing these tables
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text().lower()

            # Check if this page has any of our target tables
            for table_name in table_names:
                if table_name.lower() in text:
                    # Render page as image
                    mat = fitz.Matrix(2, 2)  # 2x zoom for better quality
                    pix = page.get_pixmap(matrix=mat)
                    img_bytes = pix.tobytes("png")
                    table_pages.append((page_num + 1, img_bytes))
                    break

        doc.close()

        # If no specific tables found, try first few pages (tables often at start)
        if not table_pages:
            doc = fitz.open(pdf_path)
            for page_num in range(min(5, len(doc))):
                page = doc[page_num]
                text = page.get_text().lower()
                if 'table' in text:
                    mat = fitz.Matrix(2, 2)
                    pix = page.get_pixmap(matrix=mat)
                    img_bytes = pix.tobytes("png")
                    table_pages.append((page_num + 1, img_bytes))
            doc.close()

        return table_pages[:5]  # Limit to 5 pages

    def _build_rescue_prompt(self, targets: List[RescueTarget]) -> str:
        """Build prompt for variance rescue"""
        # Group by variance type
        by_type = {}
        for t in targets:
            if t.variance_type not in by_type:
                by_type[t.variance_type] = []
            by_type[t.variance_type].append(t)

        prompt_parts = []
        prompt_parts.append("Extract the NUMERIC variance values from this table image.")
        prompt_parts.append("\nI need the following specific values:\n")

        for var_type, type_targets in by_type.items():
            prompt_parts.append(f"\n{var_type} VALUES NEEDED:")
            if var_type == 'LSD':
                prompt_parts.append("Look for: 'LSD(0.05) = X', 'LSD = X', 'LSD0.05 = X' in table footnotes")
            elif var_type == 'SE':
                prompt_parts.append("Look for: '± X', 'SE = X', 'standard error = X', or separate SE column")
            elif var_type == 'SD':
                prompt_parts.append("Look for: '± X', 'SD = X', 'standard deviation = X'")
            elif var_type == 'MSE':
                prompt_parts.append("Look for: 'MSE = X', 'Mean Square Error = X' in footnotes")
            elif var_type == 'CV':
                prompt_parts.append("Look for: 'CV = X%', 'coefficient of variation = X'")

            for t in type_targets[:10]:  # Limit per type
                prompt_parts.append(f"  - {t.outcome_variable}: treatment={t.treatment_mean}, control={t.control_mean}")

        prompt_parts.append("\n\nIMPORTANT:")
        prompt_parts.append("- Extract the ACTUAL NUMERIC VALUE, not just confirm it exists")
        prompt_parts.append("- LSD values are usually in table footnotes, not cells")
        prompt_parts.append("- If different outcomes have different variance values, list each")
        prompt_parts.append("- If one LSD value applies to all outcomes in a table, note that")

        prompt_parts.append("\n\nReturn JSON format:")
        prompt_parts.append("""```json
{
  "variance_values": [
    {
      "outcome": "GWAD",
      "variance_type": "LSD",
      "value": 234.5,
      "applies_to": "all grain yield comparisons",
      "source": "Table 2 footnote"
    }
  ],
  "notes": "Any additional context about variance reporting"
}
```""")

        return "\n".join(prompt_parts)

    def _parse_rescue_response(self, response: str) -> List[Dict]:
        """Parse variance values from LLM response"""
        # Clean up response
        text = response.strip()

        # Remove markdown code blocks
        if '```' in text:
            match = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
            if match:
                text = match.group(1)

        try:
            data = json.loads(text)
            return data.get('variance_values', [])
        except json.JSONDecodeError:
            # Try to extract any JSON object
            match = re.search(r'\{[^{}]*"variance_values"[^{}]*\[.*?\][^{}]*\}', text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                    return data.get('variance_values', [])
                except:
                    pass

        return []

    def rescue_paper_variance(self, paper_id: str, targets: List[RescueTarget],
                               df: pd.DataFrame) -> RescueResult:
        """Rescue variance values for a single paper using vision"""
        result = RescueResult(
            paper_id=paper_id,
            targets_found=len(targets),
            values_rescued=0,
            tables_processed=0,
            errors=[]
        )

        # Get PDF path
        pdf_path = self._get_pdf_path(paper_id)
        if not pdf_path:
            result.errors.append(f"PDF not found for {paper_id}")
            return result

        # Get table references from targets
        table_refs = [t.data_source for t in targets if t.data_source]
        if not table_refs:
            table_refs = [t.observation_id for t in targets]  # Use obs_id as fallback

        # Extract table pages as images
        try:
            table_pages = self._extract_table_pages(pdf_path, table_refs)
        except Exception as e:
            result.errors.append(f"PDF extraction error: {e}")
            return result

        if not table_pages:
            result.errors.append("No table pages found")
            return result

        result.tables_processed = len(table_pages)

        # Build rescue prompt
        prompt = self._build_rescue_prompt(targets)

        # Process each table page with vision
        all_variance_values = []
        for page_num, img_bytes in table_pages:
            try:
                response = self.llm.call_vision(
                    prompt=prompt,
                    image_data=img_bytes,
                    system_prompt="You are a precise data extractor specializing in reading scientific tables. Extract exact numeric values for variance (LSD, SE, SD) from table footnotes."
                )

                values = self._parse_rescue_response(response)
                all_variance_values.extend(values)

            except Exception as e:
                result.errors.append(f"Vision API error on page {page_num}: {e}")

        # Apply rescued values to dataframe
        for val in all_variance_values:
            if not val.get('value'):
                continue

            outcome = val.get('outcome', '').lower()
            var_type = val.get('variance_type', '')
            numeric_value = val.get('value')

            # Find matching targets
            for target in targets:
                target_outcome = target.outcome_variable.lower()

                # Match by outcome or apply to all if specified
                applies_to = val.get('applies_to', '').lower()
                matches = (
                    outcome in target_outcome or
                    target_outcome in outcome or
                    'all' in applies_to
                )

                if matches:
                    # Update the dataframe
                    mask = df['observation_id'] == target.observation_id

                    if var_type in ['LSD', 'MSE']:
                        df.loc[mask, 'pooled_variance'] = numeric_value
                    else:
                        df.loc[mask, 'treatment_variance'] = numeric_value
                        df.loc[mask, 'control_variance'] = numeric_value

                    result.values_rescued += 1

        return result

    def run(self, max_papers: int = None) -> pd.DataFrame:
        """
        Run variance rescue process

        Args:
            max_papers: Maximum number of papers to process (None = all)

        Returns:
            Updated DataFrame
        """
        print("\n" + "="*60)
        print("VARIANCE RESCUE MODULE")
        print("="*60)

        # Find rescue targets
        print("\nFinding observations with variance_type but no values...")
        targets = self.find_rescue_targets()

        if not targets:
            print("No rescue targets found - all variance values present!")
            return pd.read_csv(self.summary_path)

        self.print_rescue_summary(targets)

        # Load current data
        df = pd.read_csv(self.summary_path)

        # Sort papers by number of targets
        sorted_papers = sorted(targets.keys(),
                              key=lambda x: len(targets[x]),
                              reverse=True)

        if max_papers:
            sorted_papers = sorted_papers[:max_papers]

        # Process each paper
        print(f"\nProcessing {len(sorted_papers)} papers with vision API...")

        total_rescued = 0
        total_tables = 0

        for i, paper_id in enumerate(sorted_papers):
            paper_targets = targets[paper_id]
            print(f"\n[{i+1}/{len(sorted_papers)}] {paper_id}...")
            print(f"  Targets: {len(paper_targets)} observations")

            result = self.rescue_paper_variance(paper_id, paper_targets, df)

            total_rescued += result.values_rescued
            total_tables += result.tables_processed

            print(f"  Tables processed: {result.tables_processed}")
            print(f"  Values rescued: {result.values_rescued}")

            if result.errors:
                for err in result.errors:
                    print(f"  Warning: {err}")

        # Save results
        print(f"\n" + "="*60)
        print("VARIANCE RESCUE COMPLETE")
        print("="*60)
        print(f"Papers processed: {len(sorted_papers)}")
        print(f"Tables analyzed: {total_tables}")
        print(f"Variance values rescued: {total_rescued}")

        # Save to files
        rescued_path = self.output_dir / 'summary_rescued.csv'
        df.to_csv(rescued_path, index=False)
        print(f"\nSaved to: {rescued_path}")

        df.to_csv(self.summary_path, index=False)
        print(f"Updated: {self.summary_path}")

        return df


def main():
    """CLI entry point for variance rescue module"""
    import argparse
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from core.llm import LLMClient
    from config import GOOGLE_API_KEY, ANTHROPIC_API_KEY

    parser = argparse.ArgumentParser(description='Rescue variance values using vision API')
    parser.add_argument('--input', '-i', required=True, help='Output directory with summary.csv')
    parser.add_argument('--papers', '-p', required=True, help='Directory containing PDF papers')
    parser.add_argument('--provider', choices=['google', 'anthropic'], default='google')
    parser.add_argument('--api-key', help='API key (or use env variable)')
    parser.add_argument('--max-papers', type=int, help='Max papers to process')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze targets, do not rescue')

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

    # Run variance rescue
    rescuer = VarianceRescueModule(llm, args.input, args.papers)

    if args.analyze_only:
        targets = rescuer.find_rescue_targets()
        rescuer.print_rescue_summary(targets)
    else:
        rescuer.run(max_papers=args.max_papers)


if __name__ == '__main__':
    main()
