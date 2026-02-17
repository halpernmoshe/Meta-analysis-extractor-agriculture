"""
Validation module for Meta-Analysis Extraction System

Handles validation of extracted data.
"""
from typing import Dict, List, Optional, Any
from pathlib import Path

from core.state import Observation
from core.llm import LLMClient


class ValidateModule:
    """
    Handles validation of extracted data
    """
    
    def __init__(self, llm_client: LLMClient = None):
        self.llm = llm_client
    
    def validate_paper(
        self,
        observations: List[Observation],
        pdf_path: str = None,
        ontology: dict = None
    ) -> List[Observation]:
        """
        Validate extracted observations from a paper
        """
        validated = []
        
        for obs in observations:
            # Run validation checks
            issues = []
            
            # Check for required fields
            if not obs.treatment_mean and obs.treatment_mean != 0:
                issues.append("Missing treatment mean")
            
            # Check for plausible ranges
            if ontology:
                range_issues = self._check_ranges(obs, ontology)
                issues.extend(range_issues)
            
            # Check variance consistency
            variance_issues = self._check_variance(obs)
            issues.extend(variance_issues)
            
            # Check for outliers
            outlier_issues = self._check_outliers(obs)
            issues.extend(outlier_issues)
            
            # Update validation status
            if issues:
                obs.validation_status = 'flagged'
                obs.validation_notes = "; ".join(issues)
            else:
                obs.validation_status = 'verified'
            
            validated.append(obs)
        
        return validated
    
    def _check_ranges(
        self,
        obs: Observation,
        ontology: dict
    ) -> List[str]:
        """
        Check if values are within expected ranges
        """
        issues = []
        
        outcomes = ontology.get('outcomes', {})
        outcome_info = outcomes.get(obs.outcome_variable, {})
        
        expected_range = outcome_info.get('range')
        if expected_range and len(expected_range) == 2:
            min_val, max_val = expected_range
            
            if obs.treatment_mean is not None:
                if min_val is not None and obs.treatment_mean < min_val:
                    issues.append(f"Treatment mean {obs.treatment_mean} below expected minimum {min_val}")
                if max_val is not None and obs.treatment_mean > max_val:
                    issues.append(f"Treatment mean {obs.treatment_mean} above expected maximum {max_val}")
            
            if obs.control_mean is not None:
                if min_val is not None and obs.control_mean < min_val:
                    issues.append(f"Control mean {obs.control_mean} below expected minimum {min_val}")
                if max_val is not None and obs.control_mean > max_val:
                    issues.append(f"Control mean {obs.control_mean} above expected maximum {max_val}")
        
        return issues
    
    def _check_variance(self, obs: Observation) -> List[str]:
        """
        Check variance for consistency
        """
        issues = []
        
        # Check that variance is positive
        if obs.treatment_variance is not None and obs.treatment_variance < 0:
            issues.append("Negative treatment variance")
        
        if obs.control_variance is not None and obs.control_variance < 0:
            issues.append("Negative control variance")
        
        if obs.pooled_variance is not None and obs.pooled_variance < 0:
            issues.append("Negative pooled variance")
        
        # Check that SE is reasonable relative to mean
        if obs.variance_type == 'SE' and obs.treatment_mean and obs.treatment_variance:
            cv_approx = (obs.treatment_variance / abs(obs.treatment_mean)) * 100
            if cv_approx > 100:
                issues.append(f"Very high SE relative to mean (CV ~{cv_approx:.0f}%)")
        
        return issues
    
    def _check_outliers(self, obs: Observation) -> List[str]:
        """
        Flag potential outliers
        """
        issues = []
        
        # Check for extreme response ratios
        if obs.treatment_mean and obs.control_mean and obs.control_mean != 0:
            ratio = obs.treatment_mean / obs.control_mean
            
            if ratio > 5:
                issues.append(f"Very large effect (treatment/control = {ratio:.1f})")
            elif ratio < 0.2:
                issues.append(f"Very large negative effect (treatment/control = {ratio:.1f})")
        
        # Check for zero control (problematic for log response ratio)
        if obs.control_mean == 0:
            issues.append("Control mean is zero (cannot calculate log response ratio)")
        
        return issues
    
    def validate_batch(
        self,
        extractions: Dict[str, List[Observation]],
        ontology: dict = None,
        progress_callback=None
    ) -> Dict[str, List[Observation]]:
        """
        Validate extractions from multiple papers
        """
        validated = {}
        papers = list(extractions.keys())
        total = len(papers)
        
        for i, paper_id in enumerate(papers):
            if progress_callback:
                progress_callback(i, total, f"Validating {paper_id}...")
            
            observations = extractions[paper_id]
            validated[paper_id] = self.validate_paper(observations, ontology=ontology)
        
        if progress_callback:
            progress_callback(total, total, "Validation complete")
        
        return validated
    
    def get_validation_summary(
        self,
        extractions: Dict[str, List[Observation]]
    ) -> Dict[str, Any]:
        """
        Get summary statistics of validation results
        """
        total_obs = 0
        verified = 0
        flagged = 0
        errors = 0
        
        issues_by_type = {}
        
        for paper_id, observations in extractions.items():
            for obs in observations:
                total_obs += 1
                
                if obs.validation_status == 'verified':
                    verified += 1
                elif obs.validation_status == 'flagged':
                    flagged += 1
                    # Categorize issues
                    if obs.validation_notes:
                        for issue in obs.validation_notes.split("; "):
                            issue_type = issue.split()[0] if issue else "Unknown"
                            issues_by_type[issue_type] = issues_by_type.get(issue_type, 0) + 1
                elif obs.validation_status == 'error':
                    errors += 1
        
        return {
            'total_observations': total_obs,
            'verified': verified,
            'flagged': flagged,
            'errors': errors,
            'verification_rate': round(100 * verified / total_obs, 1) if total_obs > 0 else 0,
            'issues_by_type': issues_by_type
        }
