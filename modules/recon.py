"""
Reconnaissance module for Meta-Analysis Extraction System

Handles scanning papers to understand their content before extraction.
"""
import json
from typing import Dict, List, Optional, Any
from pathlib import Path

from core.state import PaperRecon, OutcomeInfo, ModeratorInfo, PICOSpec
from core.llm import LLMClient
from prompts.recon_prompts import (
    RECON_SYSTEM_PROMPT,
    get_overview_prompt,
    get_outcomes_recon_prompt,
    get_moderators_recon_prompt,
    get_design_recon_prompt,
    get_variance_recon_prompt,
    get_control_recon_prompt,
    get_data_tables_prompt,
    get_cultivars_recon_prompt
)


class ReconModule:
    """
    Handles reconnaissance scanning of papers
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file
        """
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
            return "\n".join(text_parts)
        except ImportError:
            # Fallback to pdfplumber
            try:
                import pdfplumber
                with pdfplumber.open(pdf_path) as pdf:
                    text_parts = []
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text:
                            text_parts.append(text)
                return "\n".join(text_parts)
            except ImportError:
                raise ImportError("Please install PyMuPDF (fitz) or pdfplumber for PDF processing")
    
    def scan_overview(self, pdf_path: str) -> PaperRecon:
        """
        Quick overview scan of a paper
        Used in orientation phase
        """
        paper_text = self.extract_text_from_pdf(pdf_path)
        paper_id = Path(pdf_path).stem
        
        # Get overview
        prompt = get_overview_prompt(paper_text)
        response = self.llm.call_worker(prompt, RECON_SYSTEM_PROMPT)
        overview = self.llm.parse_json_response(response)
        
        if 'error' in overview:
            # Return minimal recon on parse error
            return PaperRecon(
                paper_id=paper_id,
                file_path=pdf_path,
                quality_notes=f"Parse error in overview: {overview.get('error')}"
            )
        
        # Build PaperRecon from overview
        recon = PaperRecon(
            paper_id=paper_id,
            file_path=pdf_path,
            title=overview.get('title'),
            authors=overview.get('authors'),
            year=overview.get('year'),
            study_type=overview.get('study_type'),
            country=overview.get('country')
        )
        
        # Add crop species to moderators
        crops = overview.get('crop_species', [])
        if crops:
            recon.moderators_available['CROP_SP'] = ModeratorInfo(
                code='CROP_SP',
                available=True,
                values=crops
            )
        
        # Add detected outcomes
        for outcome in overview.get('outcomes_mentioned', []):
            recon.outcomes_present[outcome] = OutcomeInfo(
                icasa_code=outcome,
                present=True
            )
        
        return recon
    
    def scan_detailed(
        self, 
        pdf_path: str,
        pico: PICOSpec = None,
        ontology: dict = None
    ) -> PaperRecon:
        """
        Detailed reconnaissance scan
        Uses PICO spec to guide what to look for
        """
        paper_text = self.extract_text_from_pdf(pdf_path)
        paper_id = Path(pdf_path).stem
        
        # Initialize recon
        recon = PaperRecon(
            paper_id=paper_id,
            file_path=pdf_path
        )
        
        # 1. Overview scan
        overview_prompt = get_overview_prompt(paper_text)
        overview = self.llm.parse_json_response(
            self.llm.call_worker(overview_prompt, RECON_SYSTEM_PROMPT)
        )
        
        if 'error' not in overview:
            recon.title = overview.get('title')
            recon.authors = overview.get('authors')
            recon.year = overview.get('year')
            recon.study_type = overview.get('study_type')
            recon.country = overview.get('country')
        
        # 2. Outcomes scan
        target_outcomes = []
        if pico and pico.primary_outcomes:
            target_outcomes.extend(pico.primary_outcomes)
        if pico and pico.secondary_outcomes:
            target_outcomes.extend(pico.secondary_outcomes)
        
        if not target_outcomes:
            # Default outcomes if no PICO
            target_outcomes = ['yield', 'biomass', 'protein', 'nitrogen']
        
        outcomes_prompt = get_outcomes_recon_prompt(paper_text, target_outcomes)
        outcomes_result = self.llm.parse_json_response(
            self.llm.call_worker(outcomes_prompt, RECON_SYSTEM_PROMPT)
        )
        
        if 'error' not in outcomes_result:
            for outcome_data in outcomes_result.get('outcomes', []):
                code = outcome_data.get('variable', '')
                recon.outcomes_present[code] = OutcomeInfo(
                    icasa_code=code,
                    present=outcome_data.get('present', False),
                    location=outcome_data.get('location'),
                    unit=outcome_data.get('unit'),
                    has_means=outcome_data.get('has_means', False),
                    has_variance=outcome_data.get('has_variance', False),
                    variance_type=outcome_data.get('variance_type'),
                    notes=outcome_data.get('notes')
                )
        
        # 3. Moderators scan
        target_moderators = []
        if pico and pico.required_moderators:
            target_moderators.extend(pico.required_moderators)
        if pico and pico.optional_moderators:
            target_moderators.extend(pico.optional_moderators)
        
        if not target_moderators:
            # Default moderators
            target_moderators = ['soil pH', 'soil texture', 'N rate', 'crop species']
        
        moderators_prompt = get_moderators_recon_prompt(paper_text, target_moderators)
        moderators_result = self.llm.parse_json_response(
            self.llm.call_worker(moderators_prompt, RECON_SYSTEM_PROMPT)
        )
        
        if 'error' not in moderators_result:
            for mod_data in moderators_result.get('moderators', []):
                code = mod_data.get('variable', '')
                recon.moderators_available[code] = ModeratorInfo(
                    code=code,
                    available=mod_data.get('available', False),
                    value=mod_data.get('value'),
                    values=mod_data.get('values', []),
                    location=mod_data.get('location'),
                    notes=mod_data.get('notes')
                )
            
            # Extract coordinates if available
            if moderators_result.get('coordinates_available'):
                recon.coordinates = {
                    'latitude': moderators_result.get('latitude'),
                    'longitude': moderators_result.get('longitude')
                }
        
        # 4. Design scan
        design_prompt = get_design_recon_prompt(paper_text)
        design_result = self.llm.parse_json_response(
            self.llm.call_worker(design_prompt, RECON_SYSTEM_PROMPT)
        )
        
        if 'error' not in design_result:
            recon.design_type = design_result.get('design_type')
            recon.main_plot_factor = design_result.get('main_plot_factor')
            recon.subplot_factor = design_result.get('subplot_factor')
            recon.replicates = design_result.get('replicates')
            recon.plot_size = design_result.get('plot_size')
        
        # 5. Variance scan
        variance_prompt = get_variance_recon_prompt(paper_text)
        variance_result = self.llm.parse_json_response(
            self.llm.call_worker(variance_prompt, RECON_SYSTEM_PROMPT)
        )
        
        if 'error' not in variance_result:
            recon.variance_type = variance_result.get('variance_type')
            recon.variance_location = variance_result.get('variance_location')
            recon.variance_notes = variance_result.get('notes')
        
        # 6. Control identification
        if pico and pico.intervention_domain:
            domain = pico.intervention_domain
            heuristic = {
                'keywords': pico.control_keywords or ['control', 'check'],
                'description': pico.control_definition or 'Identify control treatment'
            }
        else:
            domain = 'general'
            heuristic = {'keywords': ['control', 'check', 'untreated']}
        
        control_prompt = get_control_recon_prompt(paper_text, domain, heuristic)
        control_result = self.llm.parse_json_response(
            self.llm.call_worker(control_prompt, RECON_SYSTEM_PROMPT)
        )
        
        if 'error' not in control_result:
            recon.control_identified = control_result.get('control_identified', False)
            recon.control_definition = control_result.get('control_description')
            recon.treatment_levels = [
                t.get('name', '') for t in control_result.get('treatment_levels', [])
            ]

        # 7. Cultivars scan - identify all cultivars/varieties in the paper
        cultivars_prompt = get_cultivars_recon_prompt(paper_text)
        cultivars_result = self.llm.parse_json_response(
            self.llm.call_worker(cultivars_prompt, RECON_SYSTEM_PROMPT)
        )

        if 'error' not in cultivars_result:
            cultivar_names = [c.get('name', '') for c in cultivars_result.get('cultivars_found', [])]
            if cultivar_names:
                recon.moderators_available['CULTIVAR'] = ModeratorInfo(
                    code='CULTIVAR',
                    available=True,
                    values=cultivar_names,
                    notes=f"Data structure: {cultivars_result.get('data_structure', 'unknown')}"
                )
            recon.cultivar_data_structure = cultivars_result.get('data_structure')
            recon.individual_cultivar_tables = cultivars_result.get('individual_cultivar_tables', [])

        # 8. Data tables scan
        tables_prompt = get_data_tables_prompt(paper_text)
        tables_result = self.llm.parse_json_response(
            self.llm.call_worker(tables_prompt, RECON_SYSTEM_PROMPT)
        )
        
        if 'error' not in tables_result:
            recon.data_tables = [
                t.get('table_id', '') for t in tables_result.get('tables', [])
                if t.get('usability') in ['high', 'medium']
            ]
            recon.data_figures = tables_result.get('figures_with_data', [])
            recon.supplementary_data = tables_result.get('supplementary_data', False)
        
        return recon
    
    def scan_batch(
        self,
        pdf_paths: List[str],
        pico: PICOSpec = None,
        ontology: dict = None,
        detailed: bool = False,
        progress_callback=None
    ) -> Dict[str, PaperRecon]:
        """
        Scan multiple papers
        """
        results = {}
        total = len(pdf_paths)
        
        for i, pdf_path in enumerate(pdf_paths):
            paper_id = Path(pdf_path).stem
            
            if progress_callback:
                progress_callback(i, total, f"Scanning {paper_id}...")
            
            try:
                if detailed:
                    recon = self.scan_detailed(pdf_path, pico, ontology)
                else:
                    recon = self.scan_overview(pdf_path)
                results[paper_id] = recon
            except Exception as e:
                # Log error but continue
                results[paper_id] = PaperRecon(
                    paper_id=paper_id,
                    file_path=pdf_path,
                    meets_pico_criteria=False,
                    exclusion_reasons=[f"Error during scan: {str(e)}"]
                )
        
        if progress_callback:
            progress_callback(total, total, "Scan complete")
        
        return results
    
    def synthesize_recon_results(
        self,
        recon_results: Dict[str, PaperRecon]
    ) -> Dict[str, Any]:
        """
        Aggregate reconnaissance results across papers
        Returns summary statistics
        """
        from collections import Counter
        
        total = len(recon_results)
        
        # Aggregate crops
        all_crops = []
        for recon in recon_results.values():
            crop_mod = recon.moderators_available.get('CROP_SP')
            if crop_mod and crop_mod.values:
                all_crops.extend(crop_mod.values)
        crop_counts = Counter(all_crops)
        
        # Aggregate study types
        study_types = [r.study_type for r in recon_results.values() if r.study_type]
        study_type_counts = Counter(study_types)
        
        # Aggregate outcomes
        all_outcomes = []
        for recon in recon_results.values():
            for code, info in recon.outcomes_present.items():
                if info.present:
                    all_outcomes.append(code)
        outcome_counts = Counter(all_outcomes)
        
        # Aggregate variance types
        variance_types = [r.variance_type for r in recon_results.values() if r.variance_type]
        variance_counts = Counter(variance_types)
        
        # Aggregate design types
        designs = [r.design_type for r in recon_results.values() if r.design_type]
        design_counts = Counter(designs)
        
        # Count moderator availability
        moderator_availability = {}
        all_mod_codes = set()
        for recon in recon_results.values():
            all_mod_codes.update(recon.moderators_available.keys())
        
        for mod_code in all_mod_codes:
            count = sum(
                1 for r in recon_results.values()
                if mod_code in r.moderators_available and r.moderators_available[mod_code].available
            )
            moderator_availability[mod_code] = {
                'count': count,
                'percentage': round(100 * count / total, 1) if total > 0 else 0
            }
        
        # Countries
        countries = [r.country for r in recon_results.values() if r.country]
        country_counts = Counter(countries)
        
        return {
            'total_papers': total,
            'crops': dict(crop_counts.most_common(10)),
            'study_types': dict(study_type_counts),
            'outcomes': dict(outcome_counts.most_common(15)),
            'variance_types': dict(variance_counts),
            'design_types': dict(design_counts),
            'moderator_availability': moderator_availability,
            'countries': dict(country_counts.most_common(10)),
            'papers_with_coordinates': sum(1 for r in recon_results.values() if r.coordinates),
            'papers_with_variance': sum(1 for r in recon_results.values() if r.variance_type and r.variance_type != 'None'),
            'papers_with_control_id': sum(1 for r in recon_results.values() if r.control_identified)
        }
