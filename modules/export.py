"""
Export module for Meta-Analysis Extraction System

Handles exporting data to JSON, CSV, and generating documentation.
"""
import json
import csv
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime

import pandas as pd

from core.state import SessionState, PaperRecon, Observation


class ExportModule:
    """
    Handles exporting extraction results
    """
    
    def to_json(
        self,
        observations: List[Observation],
        filepath: str,
        paper_id: str = None
    ):
        """
        Export observations to JSON file
        """
        data = {
            'paper_id': paper_id,
            'extraction_timestamp': datetime.now().isoformat(),
            'observation_count': len(observations),
            'observations': [obs.to_dict() for obs in observations]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def to_csv(
        self,
        extractions: Dict[str, List[Observation]],
        filepath: str
    ):
        """
        Export all observations to a single CSV file
        """
        # Flatten observations
        rows = []
        for paper_id, observations in extractions.items():
            for obs in observations:
                row = obs.to_dict()
                # Flatten moderators dict
                moderators = row.pop('moderators', {})
                for mod_key, mod_val in moderators.items():
                    row[f'mod_{mod_key}'] = mod_val
                rows.append(row)
        
        if not rows:
            # Create empty file with headers
            df = pd.DataFrame(columns=['paper_id', 'observation_id', 'outcome_variable'])
        else:
            df = pd.DataFrame(rows)
        
        # Reorder columns
        priority_cols = [
            'paper_id', 'observation_id', 'outcome_variable', 'outcome_name',
            'treatment_description', 'control_description',
            'treatment_mean', 'control_mean',
            'treatment_n', 'control_n',
            'variance_type', 'treatment_variance', 'control_variance', 'pooled_variance',
            'unit_reported', 'data_source',
            'extraction_confidence', 'validation_status'
        ]
        
        # Put priority columns first, then others
        other_cols = [c for c in df.columns if c not in priority_cols]
        col_order = [c for c in priority_cols if c in df.columns] + other_cols
        df = df[col_order]
        
        df.to_csv(filepath, index=False)
    
    def papers_to_csv(
        self,
        recon_cache: Dict[str, PaperRecon],
        filepath: str
    ):
        """
        Export paper-level summary to CSV
        """
        rows = []
        for paper_id, recon in recon_cache.items():
            row = {
                'paper_id': paper_id,
                'title': recon.title,
                'authors': recon.authors,
                'year': recon.year,
                'country': recon.country,
                'study_type': recon.study_type,
                'design_type': recon.design_type,
                'variance_type': recon.variance_type,
                'meets_criteria': recon.meets_pico_criteria,
                'exclusion_reasons': "; ".join(recon.exclusion_reasons) if recon.exclusion_reasons else "",
                'num_tables': len(recon.data_tables),
                'num_figures': len(recon.data_figures),
                'control_identified': recon.control_identified,
                'replicates': recon.replicates
            }
            
            # Add outcome availability
            for code, info in recon.outcomes_present.items():
                row[f'has_{code}'] = info.present if hasattr(info, 'present') else False
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
    
    def to_methods_doc(
        self,
        state: SessionState,
        filepath: str
    ):
        """
        Generate methods documentation in Markdown
        """
        pico = state.pico
        
        content = f"""# Meta-Analysis Data Extraction Methods

## Session Information
- Session ID: {state.session_id}
- Created: {state.created_at}
- Last Updated: {state.last_updated}

## PICO Specification

### Population
- **Crop Species**: {', '.join(pico.crop_species) if pico.crop_species else 'Not specified'}
- **Study Types**: {', '.join(pico.study_types) if pico.study_types else 'Not specified'}
- **Environments**: {', '.join(pico.environments) if pico.environments else 'Not specified'}
- **Geographic Scope**: {pico.geographic_scope or 'Not specified'}

### Intervention
- **Domain**: {pico.intervention_domain or 'Not specified'}
- **Variable**: {pico.intervention_variable or 'Not specified'}
- **Description**: {pico.intervention_description or 'Not specified'}

### Comparison (Control Definition)
- **Definition**: {pico.control_definition or 'Not specified'}
- **Identification Method**: {pico.control_heuristic or 'Not specified'}
- **Keywords**: {', '.join(pico.control_keywords) if pico.control_keywords else 'Not specified'}

### Outcomes
- **Primary**: {', '.join(pico.primary_outcomes) if pico.primary_outcomes else 'Not specified'}
- **Secondary**: {', '.join(pico.secondary_outcomes) if pico.secondary_outcomes else 'Not specified'}

### Moderators
- **Required**: {', '.join(pico.required_moderators) if pico.required_moderators else 'Not specified'}
- **Optional**: {', '.join(pico.optional_moderators) if pico.optional_moderators else 'Not specified'}

## Data Summary
- **Total Papers Screened**: {state.paper_count}
- **Papers Meeting Criteria**: {state.papers_meeting_criteria}
- **Total Observations Extracted**: {state.total_observations}
- **Observations Flagged for Review**: {state.flagged_observations}

## Decision Log

"""
        
        # Add decision log
        for i, decision in enumerate(state.decisions, 1):
            content += f"""### Decision {i}: {decision.question}
- **Phase**: {decision.phase}
- **Timestamp**: {decision.timestamp}
- **Choice**: {decision.user_choice}
- **Rationale**: {decision.rationale or 'Not recorded'}

"""
        
        content += """
## Notes

This document was automatically generated by the Meta-Analysis Extraction System.
All decisions and parameters are recorded for reproducibility.

### Variance Handling

Variance was extracted exactly as reported in each paper. The following types were encountered:
- SD (Standard Deviation)
- SE (Standard Error)
- LSD (Least Significant Difference)
- CV (Coefficient of Variation)

Conversion to a common variance metric should be performed during analysis.

### Effect Size Calculation

Effect sizes were not calculated during extraction. For meta-analysis, calculate:
- Log Response Ratio: ln(Treatment Mean / Control Mean)
- Variance of lnRR from extracted SD/SE values

See Hedges et al. (1999) for formulas.
"""
        
        with open(filepath, 'w') as f:
            f.write(content)
    
    def export_all(
        self,
        state: SessionState,
        output_dir: str
    ):
        """
        Export all results to output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create extractions subdirectory
        extractions_path = output_path / "extractions"
        extractions_path.mkdir(exist_ok=True)
        
        # Export individual paper JSONs
        for paper_id, observations in state.extractions.items():
            json_path = extractions_path / f"{paper_id}.json"
            self.to_json(observations, str(json_path), paper_id)
        
        # Export summary CSV
        self.to_csv(state.extractions, str(output_path / "summary.csv"))
        
        # Export papers CSV
        self.papers_to_csv(state.recon_cache, str(output_path / "papers.csv"))
        
        # Export session state
        state.save(str(output_path / "session_state.json"))
        
        # Export methods document
        self.to_methods_doc(state, str(output_path / "methods.md"))
        
        return {
            'extractions_dir': str(extractions_path),
            'summary_csv': str(output_path / "summary.csv"),
            'papers_csv': str(output_path / "papers.csv"),
            'session_json': str(output_path / "session_state.json"),
            'methods_md': str(output_path / "methods.md")
        }
