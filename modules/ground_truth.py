"""
Ground Truth Testing Module

Compare extracted data against known values from published meta-analyses
or manually verified extractions.

Usage:
    python -m modules.ground_truth --extracted results/summary.csv --truth ground_truth.csv
"""
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import statistics


@dataclass
class ComparisonResult:
    """Result of comparing one observation"""
    paper_id: str
    observation_id: str
    field: str
    extracted_value: Any
    truth_value: Any
    match: bool
    error: Optional[float] = None  # For numeric fields
    error_pct: Optional[float] = None
    notes: str = ""


class GroundTruthTester:
    """
    Compare extracted values against ground truth data.
    
    Ground truth can come from:
    1. Published meta-analysis datasets (e.g., from supplementary materials)
    2. Manually verified extractions from a subset of papers
    3. Known values for specific papers/observations
    """
    
    def __init__(self, tolerance_pct: float = 5.0):
        """
        Args:
            tolerance_pct: Percentage tolerance for numeric comparisons (default 5%)
        """
        self.tolerance_pct = tolerance_pct
        self.results: List[ComparisonResult] = []
    
    def load_extracted(self, filepath: str) -> List[Dict]:
        """Load extracted data from CSV"""
        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            return list(reader)
    
    def load_ground_truth(self, filepath: str) -> List[Dict]:
        """
        Load ground truth data.
        
        Expected format (CSV):
            paper_id, observation_id, outcome_variable, treatment_mean, control_mean, ...
        
        Or JSON format:
            [{"paper_id": "...", "observation_id": "...", ...}, ...]
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.json':
            with open(filepath) as f:
                return json.load(f)
        else:
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                return list(reader)
    
    def compare(
        self,
        extracted: List[Dict],
        ground_truth: List[Dict],
        key_fields: List[str] = None,
        compare_fields: List[str] = None
    ) -> Dict[str, Any]:
        """
        Compare extracted data against ground truth.
        
        Args:
            extracted: List of extracted observations
            ground_truth: List of ground truth observations
            key_fields: Fields to match observations (default: paper_id + observation_id)
            compare_fields: Fields to compare values (default: means and variances)
        
        Returns:
            Summary statistics and detailed results
        """
        key_fields = key_fields or ['paper_id', 'observation_id']
        compare_fields = compare_fields or [
            'treatment_mean', 'control_mean',
            'treatment_variance', 'control_variance', 'pooled_variance',
            'treatment_n', 'control_n'
        ]
        
        # Index ground truth by key
        truth_index = {}
        for row in ground_truth:
            key = tuple(str(row.get(k, '')).strip() for k in key_fields)
            truth_index[key] = row
        
        # Index extracted by key
        extract_index = {}
        for row in extracted:
            key = tuple(str(row.get(k, '')).strip() for k in key_fields)
            extract_index[key] = row
        
        self.results = []
        
        # Compare each ground truth observation
        for key, truth_row in truth_index.items():
            if key not in extract_index:
                # Missing extraction
                self.results.append(ComparisonResult(
                    paper_id=key[0] if key else "",
                    observation_id=key[1] if len(key) > 1 else "",
                    field="OBSERVATION",
                    extracted_value=None,
                    truth_value="present",
                    match=False,
                    notes="Observation not extracted"
                ))
                continue
            
            extract_row = extract_index[key]
            
            # Compare each field
            for field in compare_fields:
                truth_val = truth_row.get(field)
                extract_val = extract_row.get(field)
                
                result = self._compare_values(
                    paper_id=key[0] if key else "",
                    observation_id=key[1] if len(key) > 1 else "",
                    field=field,
                    extracted=extract_val,
                    truth=truth_val
                )
                self.results.append(result)
        
        # Check for extra extractions not in ground truth
        for key in extract_index:
            if key not in truth_index:
                self.results.append(ComparisonResult(
                    paper_id=key[0] if key else "",
                    observation_id=key[1] if len(key) > 1 else "",
                    field="OBSERVATION",
                    extracted_value="present",
                    truth_value=None,
                    match=False,
                    notes="Extra observation (not in ground truth)"
                ))
        
        return self._summarize_results()
    
    def _compare_values(
        self,
        paper_id: str,
        observation_id: str,
        field: str,
        extracted: Any,
        truth: Any
    ) -> ComparisonResult:
        """Compare a single value"""
        
        # Handle missing values
        if truth is None or truth == '' or str(truth).lower() == 'nan':
            return ComparisonResult(
                paper_id=paper_id,
                observation_id=observation_id,
                field=field,
                extracted_value=extracted,
                truth_value=truth,
                match=True,  # No ground truth to compare
                notes="No ground truth value"
            )
        
        if extracted is None or extracted == '' or str(extracted).lower() == 'nan':
            return ComparisonResult(
                paper_id=paper_id,
                observation_id=observation_id,
                field=field,
                extracted_value=extracted,
                truth_value=truth,
                match=False,
                notes="Missing extracted value"
            )
        
        # Try numeric comparison
        try:
            ext_num = float(extracted)
            truth_num = float(truth)
            
            if truth_num == 0:
                match = ext_num == 0
                error = ext_num - truth_num
                error_pct = None
            else:
                error = ext_num - truth_num
                error_pct = abs(error / truth_num) * 100
                match = error_pct <= self.tolerance_pct
            
            return ComparisonResult(
                paper_id=paper_id,
                observation_id=observation_id,
                field=field,
                extracted_value=ext_num,
                truth_value=truth_num,
                match=match,
                error=error,
                error_pct=error_pct,
                notes=f"Error: {error_pct:.1f}%" if error_pct else ""
            )
        
        except (ValueError, TypeError):
            # String comparison
            match = str(extracted).strip().lower() == str(truth).strip().lower()
            return ComparisonResult(
                paper_id=paper_id,
                observation_id=observation_id,
                field=field,
                extracted_value=extracted,
                truth_value=truth,
                match=match,
                notes="" if match else "String mismatch"
            )
    
    def _summarize_results(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        
        total = len(self.results)
        matches = sum(1 for r in self.results if r.match)
        mismatches = total - matches
        
        # Group by field
        by_field = {}
        for r in self.results:
            if r.field not in by_field:
                by_field[r.field] = {'total': 0, 'matches': 0, 'errors': []}
            by_field[r.field]['total'] += 1
            if r.match:
                by_field[r.field]['matches'] += 1
            if r.error_pct is not None:
                by_field[r.field]['errors'].append(r.error_pct)
        
        # Calculate stats per field
        field_stats = {}
        for field, data in by_field.items():
            stats = {
                'accuracy': data['matches'] / data['total'] * 100 if data['total'] > 0 else 0,
                'total': data['total'],
                'matches': data['matches']
            }
            if data['errors']:
                stats['mean_error_pct'] = statistics.mean(data['errors'])
                stats['median_error_pct'] = statistics.median(data['errors'])
                if len(data['errors']) > 1:
                    stats['std_error_pct'] = statistics.stdev(data['errors'])
            field_stats[field] = stats
        
        # Get worst mismatches
        worst = sorted(
            [r for r in self.results if not r.match and r.error_pct is not None],
            key=lambda x: x.error_pct,
            reverse=True
        )[:10]
        
        return {
            'overall': {
                'total_comparisons': total,
                'matches': matches,
                'mismatches': mismatches,
                'accuracy_pct': matches / total * 100 if total > 0 else 0
            },
            'by_field': field_stats,
            'worst_errors': [
                {
                    'paper_id': r.paper_id,
                    'observation_id': r.observation_id,
                    'field': r.field,
                    'extracted': r.extracted_value,
                    'truth': r.truth_value,
                    'error_pct': r.error_pct
                }
                for r in worst
            ],
            'missing_observations': [
                r.paper_id for r in self.results 
                if r.field == 'OBSERVATION' and r.extracted_value is None
            ],
            'extra_observations': [
                r.paper_id for r in self.results
                if r.field == 'OBSERVATION' and r.truth_value is None
            ]
        }
    
    def save_detailed_results(self, filepath: str):
        """Save detailed comparison results to CSV"""
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'paper_id', 'observation_id', 'field',
                'extracted_value', 'truth_value', 'match',
                'error', 'error_pct', 'notes'
            ])
            for r in self.results:
                writer.writerow([
                    r.paper_id, r.observation_id, r.field,
                    r.extracted_value, r.truth_value, r.match,
                    r.error, r.error_pct, r.notes
                ])
    
    def print_report(self, summary: Dict[str, Any]):
        """Print formatted report"""
        print("\n" + "=" * 60)
        print("GROUND TRUTH COMPARISON REPORT")
        print("=" * 60)
        
        overall = summary['overall']
        print(f"\nOverall Accuracy: {overall['accuracy_pct']:.1f}%")
        print(f"  Total comparisons: {overall['total_comparisons']}")
        print(f"  Matches: {overall['matches']}")
        print(f"  Mismatches: {overall['mismatches']}")
        
        print("\n--- By Field ---")
        for field, stats in summary['by_field'].items():
            print(f"\n{field}:")
            print(f"  Accuracy: {stats['accuracy']:.1f}% ({stats['matches']}/{stats['total']})")
            if 'mean_error_pct' in stats:
                print(f"  Mean error: {stats['mean_error_pct']:.2f}%")
                print(f"  Median error: {stats['median_error_pct']:.2f}%")
        
        if summary['worst_errors']:
            print("\n--- Worst Errors ---")
            for err in summary['worst_errors'][:5]:
                print(f"  {err['paper_id']}/{err['observation_id']} - {err['field']}:")
                print(f"    Extracted: {err['extracted']}, Truth: {err['truth']} ({err['error_pct']:.1f}% error)")
        
        if summary['missing_observations']:
            print(f"\n--- Missing Observations ({len(summary['missing_observations'])}) ---")
            for pid in summary['missing_observations'][:5]:
                print(f"  {pid}")
            if len(summary['missing_observations']) > 5:
                print(f"  ... and {len(summary['missing_observations']) - 5} more")
        
        print("\n" + "=" * 60)


def create_ground_truth_template(
    extracted_csv: str,
    output_csv: str,
    sample_n: int = 10
):
    """
    Create a template for manual ground truth verification.
    
    Takes a sample of extracted observations and creates a CSV
    that can be manually verified against original papers.
    """
    import random
    
    with open(extracted_csv, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    # Sample rows
    if len(rows) > sample_n:
        sample = random.sample(rows, sample_n)
    else:
        sample = rows
    
    # Write template
    with open(output_csv, 'w', newline='') as f:
        fieldnames = [
            'paper_id', 'observation_id', 'outcome_variable',
            'treatment_mean', 'control_mean',
            'treatment_n', 'control_n',
            'variance_type', 'treatment_variance', 'control_variance',
            'pooled_variance', 'unit_reported',
            'VERIFIED', 'NOTES'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in sample:
            out_row = {k: row.get(k, '') for k in fieldnames if k not in ['VERIFIED', 'NOTES']}
            out_row['VERIFIED'] = ''  # To be filled in manually
            out_row['NOTES'] = ''
            writer.writerow(out_row)
    
    print(f"Created ground truth template with {len(sample)} observations: {output_csv}")
    print("Instructions:")
    print("  1. Open each paper listed")
    print("  2. Verify or correct each value")
    print("  3. Mark VERIFIED column as 'Y' when checked")
    print("  4. Add any notes about discrepancies")


def main():
    parser = argparse.ArgumentParser(description="Ground Truth Testing")
    subparsers = parser.add_subparsers(dest='command')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare extracted vs ground truth')
    compare_parser.add_argument('--extracted', '-e', required=True, help='Extracted data CSV')
    compare_parser.add_argument('--truth', '-t', required=True, help='Ground truth CSV/JSON')
    compare_parser.add_argument('--output', '-o', help='Output detailed results CSV')
    compare_parser.add_argument('--tolerance', type=float, default=5.0, help='Tolerance %% (default 5)')
    
    # Template command
    template_parser = subparsers.add_parser('template', help='Create verification template')
    template_parser.add_argument('--extracted', '-e', required=True, help='Extracted data CSV')
    template_parser.add_argument('--output', '-o', required=True, help='Output template CSV')
    template_parser.add_argument('--sample', '-n', type=int, default=10, help='Sample size')
    
    args = parser.parse_args()
    
    if args.command == 'compare':
        tester = GroundTruthTester(tolerance_pct=args.tolerance)
        extracted = tester.load_extracted(args.extracted)
        truth = tester.load_ground_truth(args.truth)
        summary = tester.compare(extracted, truth)
        tester.print_report(summary)
        
        if args.output:
            tester.save_detailed_results(args.output)
            print(f"\nDetailed results saved to: {args.output}")
    
    elif args.command == 'template':
        create_ground_truth_template(args.extracted, args.output, args.sample)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
