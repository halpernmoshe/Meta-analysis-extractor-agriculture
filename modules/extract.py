"""
Extraction module for Meta-Analysis Extraction System

Handles extracting quantitative data from papers.
"""
import json
from typing import Dict, List, Optional, Any
from pathlib import Path

from core.state import (
    PaperRecon, Observation, ExtractionSchema, PICOSpec,
    OutcomeField, ModeratorField
)
from core.llm import LLMClient
from prompts.extract_prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    get_extraction_prompt,
    get_table_extraction_prompt,
    get_moderator_extraction_prompt,
    get_missing_data_prompt
)


class ExtractModule:
    """
    Handles data extraction from papers
    """
    
    def __init__(self, llm_client: LLMClient, output_dir: str = None):
        self.llm = llm_client
        self.output_dir = output_dir
    
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
                raise ImportError("Please install PyMuPDF (fitz) or pdfplumber")
    
    def extract_paper(
        self,
        pdf_path: str,
        schema: ExtractionSchema,
        recon: PaperRecon,
        pico: PICOSpec,
        ontology: dict = None,
        domain_config: dict = None
    ) -> List[Observation]:
        """
        Extract all data from a single paper
        """
        paper_text = self.extract_text_from_pdf(pdf_path)
        paper_id = Path(pdf_path).stem
        
        # Prepare extraction parameters
        outcomes = [
            {'code': o.code, 'name': o.name, 'unit': o.unit}
            for o in schema.outcomes
        ]
        
        moderators = [
            {'code': m.code, 'name': m.name, 'type': m.field_type}
            for m in schema.moderators
        ]
        
        control_heuristic = schema.control_heuristic or {
            'description': pico.control_definition or 'Identify control treatment',
            'keywords': pico.control_keywords or ['control', 'check']
        }
        
        design_info = {
            'design_type': recon.design_type or 'Unknown',
            'main_plot_factor': recon.main_plot_factor,
            'subplot_factor': recon.subplot_factor,
            'replicates': recon.replicates
        }
        
        variance_info = {
            'variance_type': recon.variance_type or 'Unknown',
            'variance_location': recon.variance_location
        }

        # Run global variance scanner for better variance type detection
        global_variance_type = None
        global_variance_evidence = None
        try:
            from global_variance_scanner import scan_for_global_variance
            global_type, confidence, evidence = scan_for_global_variance(paper_text)
            if global_type and str(global_type) not in ('UNKNOWN', 'None', 'VarianceType.UNKNOWN'):
                type_str = global_type.value if hasattr(global_type, 'value') else str(global_type)
                global_variance_type = type_str
                global_variance_evidence = evidence
                # Override recon variance if we found a higher-confidence declaration
                if confidence in ('high', 'medium'):
                    variance_info['variance_type'] = type_str
                    variance_info['variance_location'] = f"Global declaration: {evidence[:100]}"
        except ImportError:
            pass  # global_variance_scanner not available
        except Exception as e:
            pass  # Continue without global scan

        # Get primary outcome for prioritization
        primary_outcome = None
        if pico.primary_outcomes:
            primary_outcome = pico.primary_outcomes[0]  # Use first primary outcome

        # Main extraction call (uses extract model for accuracy)
        extraction_prompt = get_extraction_prompt(
            paper_text=paper_text,
            outcomes=outcomes,
            moderators=moderators,
            control_heuristic=control_heuristic,
            design_info=design_info,
            variance_info=variance_info,
            primary_outcome=primary_outcome,
            domain_config=domain_config
        )

        response = self.llm.call_extract(extraction_prompt, EXTRACTION_SYSTEM_PROMPT)
        extraction_result = self.llm.parse_json_response(response)

        if 'error' in extraction_result:
            # Save FULL raw response to debug file for analysis
            if self.output_dir:
                debug_dir = Path(self.output_dir) / 'debug'
                debug_dir.mkdir(exist_ok=True)
                debug_file = debug_dir / f"{paper_id}_raw_response.txt"
                with open(debug_file, 'w', encoding='utf-8') as f:
                    f.write(f"Paper: {paper_id}\n")
                    f.write(f"Error: {extraction_result.get('error')}\n")
                    f.write(f"Response length: {len(response)} chars\n")
                    f.write(f"--- FULL RAW RESPONSE ---\n{response}\n")

            # Return empty list with error note
            return [Observation(
                observation_id=f"{paper_id}_error",
                paper_id=paper_id,
                notes=f"Extraction error: {extraction_result.get('error')}"
            )]
        
        # Convert to Observation objects
        observations = []
        for i, obs_data in enumerate(extraction_result.get('observations', [])):
            obs = self._convert_to_observation(obs_data, paper_id, i + 1)
            observations.append(obs)

        # Apply global variance type to observations without variance_type
        if global_variance_type:
            for obs in observations:
                if not obs.variance_type or obs.variance_type in ('Unknown', 'UNKNOWN', 'None'):
                    obs.variance_type = global_variance_type
                    if global_variance_evidence:
                        obs.notes = (obs.notes or '') + f" [Global variance: {global_variance_type}]"

        # If no observations extracted, try table-specific extraction
        if not observations and recon.data_tables:
            for table_id in recon.data_tables[:3]:  # Try top 3 tables
                table_obs = self._extract_from_table(
                    paper_text, table_id, outcomes, control_heuristic, paper_id
                )
                observations.extend(table_obs)
        
        return observations
    
    def _convert_to_observation(
        self,
        obs_data: dict,
        paper_id: str,
        obs_num: int
    ) -> Observation:
        """
        Convert extracted dict to Observation dataclass
        """
        obs_id = obs_data.get('observation_id', obs_num)
        
        return Observation(
            observation_id=f"{paper_id}_{obs_id}",
            paper_id=paper_id,
            treatment_description=obs_data.get('treatment_description', ''),
            control_description=obs_data.get('control_description', ''),
            is_control=obs_data.get('is_control', False),
            comparison_type=obs_data.get('comparison_type'),
            outcome_variable=obs_data.get('outcome_variable', ''),
            outcome_name=obs_data.get('outcome_name', ''),
            plant_part=obs_data.get('plant_part'),
            measurement_type=obs_data.get('measurement_type'),
            treatment_mean=self._safe_float(obs_data.get('treatment_mean')),
            treatment_n=self._safe_int(obs_data.get('treatment_n')),
            control_mean=self._safe_float(obs_data.get('control_mean')),
            control_n=self._safe_int(obs_data.get('control_n')),
            variance_type=obs_data.get('variance_type'),
            treatment_variance=self._safe_float(obs_data.get('treatment_variance')),
            control_variance=self._safe_float(obs_data.get('control_variance')),
            pooled_variance=self._safe_float(obs_data.get('pooled_variance')),
            degrees_of_freedom=self._safe_int(obs_data.get('degrees_of_freedom')),
            alpha_level=self._safe_float(obs_data.get('alpha_level')),
            unit_reported=obs_data.get('unit_reported'),
            unit_standardized=obs_data.get('unit_standardized'),
            moderators=obs_data.get('moderators', {}),
            data_source=obs_data.get('data_source'),
            data_source_type=obs_data.get('data_source_type'),
            study_year=obs_data.get('study_year'),
            site_id=obs_data.get('site_id'),
            extraction_confidence=obs_data.get('confidence', 'medium'),
            is_summary_row=obs_data.get('is_summary_row', False),
            notes=obs_data.get('notes')
        )
    
    def _extract_from_table(
        self,
        paper_text: str,
        table_id: str,
        outcomes: list,
        control_heuristic: dict,
        paper_id: str
    ) -> List[Observation]:
        """
        Extract data from a specific table
        """
        prompt = get_table_extraction_prompt(
            paper_text=paper_text,
            table_id=table_id,
            outcomes=outcomes,
            control_heuristic=control_heuristic
        )

        response = self.llm.call_extract(prompt, EXTRACTION_SYSTEM_PROMPT)
        result = self.llm.parse_json_response(response)
        
        if 'error' in result:
            return []
        
        observations = []
        variance_type = result.get('variance_type')
        
        # Find control row
        control_row = None
        for row in result.get('rows', []):
            if row.get('is_control'):
                control_row = row
                break
        
        # Extract observations
        for i, row in enumerate(result.get('rows', [])):
            if row.get('is_control') or row.get('is_summary'):
                continue
            
            for outcome_code, values in row.get('values', {}).items():
                if not values.get('mean'):
                    continue
                
                control_mean = None
                control_variance = None
                if control_row and outcome_code in control_row.get('values', {}):
                    control_mean = control_row['values'][outcome_code].get('mean')
                    control_variance = control_row['values'][outcome_code].get('variance')
                
                obs = Observation(
                    observation_id=f"{paper_id}_{table_id}_{i}_{outcome_code}",
                    paper_id=paper_id,
                    treatment_description=row.get('treatment', ''),
                    control_description=control_row.get('treatment', '') if control_row else '',
                    outcome_variable=outcome_code,
                    treatment_mean=self._safe_float(values.get('mean')),
                    control_mean=self._safe_float(control_mean),
                    treatment_n=self._safe_int(values.get('n')),
                    variance_type=variance_type,
                    treatment_variance=self._safe_float(values.get('variance')),
                    control_variance=self._safe_float(control_variance),
                    unit_reported=values.get('unit'),
                    data_source=table_id,
                    data_source_type='table',
                    moderators=row.get('factor_levels', {}),
                    is_summary_row=row.get('is_summary', False),
                    extraction_confidence='medium'
                )
                observations.append(obs)
        
        return observations
    
    def extract_moderators(
        self,
        pdf_path: str,
        moderators: List[ModeratorField]
    ) -> Dict[str, Any]:
        """
        Extract moderator values from a paper
        """
        paper_text = self.extract_text_from_pdf(pdf_path)
        
        mod_list = [
            {'code': m.code, 'name': m.name, 'type': m.field_type}
            for m in moderators
        ]
        
        prompt = get_moderator_extraction_prompt(paper_text, mod_list)
        response = self.llm.call_worker(prompt, EXTRACTION_SYSTEM_PROMPT)
        result = self.llm.parse_json_response(response)
        
        return result
    
    def fill_missing_data(
        self,
        pdf_path: str,
        missing_fields: List[str]
    ) -> Dict[str, Any]:
        """
        Second pass to find missing data
        """
        paper_text = self.extract_text_from_pdf(pdf_path)
        
        prompt = get_missing_data_prompt(paper_text, missing_fields)
        response = self.llm.call_worker(prompt, EXTRACTION_SYSTEM_PROMPT)
        result = self.llm.parse_json_response(response)
        
        return result
    
    def extract_batch(
        self,
        papers: List[Dict],
        schema: ExtractionSchema,
        recon_cache: Dict[str, PaperRecon],
        pico: PICOSpec,
        ontology: dict = None,
        progress_callback=None
    ) -> Dict[str, List[Observation]]:
        """
        Extract data from multiple papers
        """
        results = {}
        total = len(papers)
        
        for i, paper in enumerate(papers):
            paper_id = paper['id']
            pdf_path = paper['path']
            
            if progress_callback:
                progress_callback(i, total, f"Extracting from {paper_id}...")
            
            recon = recon_cache.get(paper_id, PaperRecon(paper_id=paper_id, file_path=pdf_path))
            
            try:
                observations = self.extract_paper(
                    pdf_path=pdf_path,
                    schema=schema,
                    recon=recon,
                    pico=pico,
                    ontology=ontology
                )
                results[paper_id] = observations
            except Exception as e:
                results[paper_id] = [Observation(
                    observation_id=f"{paper_id}_error",
                    paper_id=paper_id,
                    notes=f"Extraction error: {str(e)}"
                )]
        
        if progress_callback:
            progress_callback(total, total, "Extraction complete")
        
        return results
    
    @staticmethod
    def _safe_float(value) -> Optional[float]:
        """Safely convert to float"""
        if value is None:
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def _safe_int(value) -> Optional[int]:
        """Safely convert to int"""
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None
