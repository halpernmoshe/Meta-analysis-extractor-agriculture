"""
Figure Extraction Module for Meta-Analysis Extraction System

Handles vision-based extraction from figures/charts when table data is incomplete.
Only triggers when gap analysis identifies missing data that figures may contain.
"""
import json
import base64
import io
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

try:
    from PIL import Image
    import fitz  # PyMuPDF
    HAS_IMAGE_LIBS = True
except ImportError:
    HAS_IMAGE_LIBS = False

from core.state import PaperRecon, Observation, PICOSpec
from core.llm import LLMClient


class FigureExtractModule:
    """
    Handles extraction of data from figures using vision capabilities.

    Design principle: Only extract from figures when necessary (cost control).
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def check_figure_extraction_needed(
        self,
        recon: PaperRecon,
        table_observations: List[Observation],
        pico: PICOSpec
    ) -> Tuple[bool, List[str], str]:
        """
        Determine if figure extraction is needed based on gap analysis.

        Returns:
            (needs_extraction, figure_ids, reason)
        """
        reasons = []
        figures_to_extract = []

        # 1. Check if recon identified figures with data
        if not recon.data_figures:
            return False, [], "No figures with extractable data identified in recon"

        # 2. Check for factorial design gaps
        if recon.design_type in ['Split-plot', 'Factorial', 'Strip-plot']:
            # Calculate expected observations
            expected = self._calculate_expected_cells(recon)
            actual = len(table_observations)

            if actual < expected * 0.5:  # Less than half expected
                reasons.append(f"Factorial design: expected ~{expected} cells, got {actual}")
                figures_to_extract = recon.data_figures

        # 3. Check if primary outcomes are missing from tables
        primary_outcomes = set(pico.primary_outcomes) if pico.primary_outcomes else set()
        extracted_outcomes = set(obs.outcome_variable for obs in table_observations if obs.outcome_variable)

        missing_primary = primary_outcomes - extracted_outcomes
        if missing_primary:
            reasons.append(f"Missing primary outcomes: {missing_primary}")
            figures_to_extract = recon.data_figures

        # 4. Check if we only got main effects but design suggests interactions
        if recon.design_type and 'x' in str(recon.design_type).lower():
            # Likely factorial - check if we have interaction data
            has_interaction_data = any(
                obs.notes and 'interaction' in obs.notes.lower()
                for obs in table_observations
            )
            if not has_interaction_data and len(table_observations) < 20:
                reasons.append("May be missing interaction data from factorial design")
                figures_to_extract = recon.data_figures

        # 5. Check for variance gaps - figures often have error bars
        obs_with_variance = sum(1 for obs in table_observations
                                if obs.treatment_variance or obs.control_variance or obs.pooled_variance)
        if obs_with_variance < len(table_observations) * 0.3:
            # Less than 30% have variance - figures might help
            reasons.append(f"Low variance coverage: {obs_with_variance}/{len(table_observations)}")
            # Only add figures if not already added
            for fig in recon.data_figures:
                if fig not in figures_to_extract:
                    figures_to_extract.append(fig)

        needs_extraction = len(figures_to_extract) > 0 and len(reasons) > 0
        reason_str = "; ".join(reasons) if reasons else "No gaps detected"

        return needs_extraction, figures_to_extract, reason_str

    def _calculate_expected_cells(self, recon: PaperRecon) -> int:
        """Estimate expected number of observations based on design"""
        base = 1

        # Factor in treatment levels
        if recon.treatment_levels:
            base *= max(len(recon.treatment_levels), 2)

        # Factor in years
        if recon.design_type and 'year' in str(recon.design_type).lower():
            base *= 2  # Assume at least 2 years

        # Factor in replicates (for means, not individual obs)
        # Actually for meta-analysis we want treatment means, not replicate values

        # Factor in outcomes
        outcome_count = len([o for o in recon.outcomes_present.values() if o.present])
        if outcome_count > 0:
            base *= min(outcome_count, 10)  # Cap at 10 to avoid overestimate

        return max(base, 5)  # Minimum 5 expected

    def extract_figures_from_pdf(
        self,
        pdf_path: str,
        target_figures: List[str] = None,
        min_width: int = 200,
        min_height: int = 150
    ) -> List[Dict]:
        """
        Extract figure images from PDF.

        Args:
            pdf_path: Path to PDF file
            target_figures: Specific figures to extract (e.g., ["Figure 2", "Figure 3"])
                           If None, extracts all potential chart figures
            min_width: Minimum width to consider
            min_height: Minimum height to consider

        Returns:
            List of figure info dicts with image data
        """
        if not HAS_IMAGE_LIBS:
            print("Warning: PIL/PyMuPDF not available for figure extraction")
            return []

        figures = []

        try:
            doc = fitz.open(pdf_path)

            for page_num, page in enumerate(doc, 1):
                # Get page text to find figure references
                page_text = page.get_text()

                # Check if this page has target figures
                page_has_target = True
                if target_figures:
                    page_has_target = any(
                        fig.lower() in page_text.lower()
                        for fig in target_figures
                    )

                if not page_has_target:
                    continue

                # Extract embedded images from this page
                image_list = page.get_images(full=True)

                for img_idx, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]

                        # Check size
                        pil_image = Image.open(io.BytesIO(image_bytes))
                        width, height = pil_image.size

                        if width < min_width or height < min_height:
                            continue

                        # Detect figure type
                        fig_type = self._detect_figure_type(pil_image)

                        # Skip photos (we want charts/graphs)
                        if fig_type == 'photo':
                            continue

                        # Convert to PNG bytes for API
                        img_buffer = io.BytesIO()
                        # Convert to RGB if needed (some PDFs have CMYK)
                        if pil_image.mode in ('CMYK', 'P'):
                            pil_image = pil_image.convert('RGB')
                        pil_image.save(img_buffer, format='PNG')
                        img_bytes = img_buffer.getvalue()

                        # Try to identify figure number from nearby text
                        fig_id = self._identify_figure_number(page_text, img_idx)

                        fig_info = {
                            'page_num': page_num,
                            'image_idx': img_idx,
                            'figure_id': fig_id,
                            'width': width,
                            'height': height,
                            'image_data': img_bytes,  # Raw bytes for vision API
                            'fig_type': fig_type,
                            'source': 'embedded'
                        }

                        figures.append(fig_info)

                    except Exception as e:
                        continue

                # If no embedded images found but page mentions target figure,
                # render the full page as fallback
                if not image_list and page_has_target and target_figures:
                    page_fig = self._render_page_as_figure(page, page_num)
                    if page_fig:
                        page_fig['figure_id'] = f"Page {page_num} (full render)"
                        figures.append(page_fig)

            doc.close()

        except Exception as e:
            print(f"Error extracting figures: {e}")

        return figures

    def _detect_figure_type(self, image: Image.Image) -> str:
        """
        Detect if image is a chart, photo, or diagram.
        Charts typically have: white background, few colors, geometric shapes
        """
        if image.mode != 'RGB':
            try:
                image = image.convert('RGB')
            except:
                return 'unknown'

        width, height = image.size

        # Sample pixels
        try:
            pixels = list(image.getdata())
            sample_size = min(1000, len(pixels))
            sample = pixels[:sample_size]
        except:
            return 'unknown'

        # Count characteristics
        unique_colors = len(set(sample))
        white_ish = sum(1 for p in sample if sum(p) > 700)  # RGB sum > 700 is whitish
        white_ratio = white_ish / sample_size

        # Charts: white background, limited colors
        if white_ratio > 0.4 and unique_colors < 100:
            return 'chart'
        # Photos: many colors, varied backgrounds
        elif unique_colors > 500:
            return 'photo'
        else:
            return 'diagram'

    def _identify_figure_number(self, page_text: str, img_idx: int) -> str:
        """Try to identify figure number from page text"""
        # Look for Figure X, Fig. X patterns
        patterns = [
            r'Figure\s*(\d+)',
            r'Fig\.\s*(\d+)',
            r'FIGURE\s*(\d+)',
        ]

        matches = []
        for pattern in patterns:
            matches.extend(re.findall(pattern, page_text, re.IGNORECASE))

        if matches:
            # Return first match (most likely the figure on this page)
            return f"Figure {matches[0]}"

        return f"Figure (page image {img_idx})"

    def _render_page_as_figure(self, page, page_num: int) -> Optional[Dict]:
        """Render full page at high resolution as fallback"""
        try:
            mat = fitz.Matrix(2, 2)  # 2x zoom
            pix = page.get_pixmap(matrix=mat)

            img_data = pix.tobytes("png")

            return {
                'page_num': page_num,
                'image_idx': 'full_page',
                'width': pix.width,
                'height': pix.height,
                'image_data': img_data,  # Raw bytes for vision API
                'fig_type': 'full_page',
                'source': 'rendered'
            }
        except:
            return None

    def extract_data_from_figure(
        self,
        figure_info: Dict,
        paper_text: str,
        recon: PaperRecon,
        pico: PICOSpec
    ) -> List[Dict]:
        """
        Use vision API to extract data from a figure image.

        Returns list of extracted data points.
        """
        # Build context for the LLM
        outcomes_str = ", ".join(pico.primary_outcomes + (pico.secondary_outcomes or []))

        prompt = f"""Extract numerical data from this scientific figure/chart.

CONTEXT:
- Paper title: {recon.title or 'Unknown'}
- Target outcomes: {outcomes_str}
- Control definition: {pico.control_definition or 'lowest value or untreated'}
- Figure ID: {figure_info.get('figure_id', 'Unknown')}
- Expected design: {recon.design_type or 'Unknown'}

INSTRUCTIONS:
1. Identify the chart type (bar chart, line graph, scatter plot, etc.)
2. Read axis labels carefully for units
3. Extract ALL visible data points
4. For bar charts: read bar heights from y-axis scale
5. If values are labeled on bars/points, use those exact numbers (HIGH confidence)
6. If estimating from axis scale, note MEDIUM confidence
7. Look for error bars - extract those values too (they indicate SE or SD)
8. Identify which groups are treatments vs controls

OUTPUT FORMAT (JSON):
{{
    "figure_type": "bar chart",
    "y_axis_label": "Grain yield (Mg/acre)",
    "y_axis_unit": "Mg/acre",
    "x_axis_label": "Treatment",
    "has_error_bars": true,
    "error_bar_type": "SE or SD or unknown",
    "data_points": [
        {{
            "group_label": "Control",
            "subgroup": "Cultivar A",
            "value": 3.13,
            "error_value": 0.25,
            "is_control": true,
            "confidence": "high",
            "labeled_on_figure": true
        }},
        {{
            "group_label": "K2SiO3",
            "subgroup": "Cultivar A",
            "value": 3.55,
            "error_value": 0.30,
            "is_control": false,
            "confidence": "medium",
            "labeled_on_figure": false
        }}
    ],
    "notes": "Values extracted from bar heights. Error bars appear to be SE."
}}

Extract ALL visible data points. Be precise with numbers."""

        try:
            # Call vision-enabled model
            response = self.llm.call_vision(
                prompt=prompt,
                image_data=figure_info.get('image_data'),
                image_format='png',
                system_prompt="You are an expert at reading scientific charts and extracting precise numerical values."
            )

            # Parse response
            result = self._parse_json_response(response)

            if 'error' not in result:
                # Add metadata
                result['figure_id'] = figure_info.get('figure_id')
                result['page_num'] = figure_info.get('page_num')
                result['extraction_source'] = 'figure_vision'

            return result

        except Exception as e:
            return {'error': str(e), 'figure_id': figure_info.get('figure_id')}

    def convert_figure_data_to_observations(
        self,
        figure_data: Dict,
        recon: PaperRecon,
        pico: PICOSpec
    ) -> List[Observation]:
        """
        Convert extracted figure data to Observation objects.
        Pairs treatments with controls to create comparisons.
        """
        observations = []

        if 'error' in figure_data or 'data_points' not in figure_data:
            return observations

        data_points = figure_data.get('data_points', [])
        unit = figure_data.get('y_axis_unit', '')
        error_type = figure_data.get('error_bar_type', 'unknown')
        figure_id = figure_data.get('figure_id', 'Figure')

        # Separate controls and treatments
        controls = [dp for dp in data_points if dp.get('is_control', False)]
        treatments = [dp for dp in data_points if not dp.get('is_control', False)]

        # If no explicit controls identified, use lowest values
        if not controls and treatments:
            # Find the treatment with lowest value as control
            sorted_by_value = sorted(data_points, key=lambda x: x.get('value', 999))
            if sorted_by_value:
                controls = [sorted_by_value[0]]
                treatments = sorted_by_value[1:]

        # Create observations by pairing treatments with controls
        obs_id = 1
        for treatment in treatments:
            # Find matching control (same subgroup if available)
            subgroup = treatment.get('subgroup', '')
            matching_control = None

            for ctrl in controls:
                if ctrl.get('subgroup', '') == subgroup or not subgroup:
                    matching_control = ctrl
                    break

            if not matching_control and controls:
                matching_control = controls[0]  # Use first control as fallback

            if matching_control:
                # Determine variance type and values
                variance_type = None
                treatment_variance = None
                control_variance = None

                if error_type and error_type.lower() in ['se', 'sd', 'standard error', 'standard deviation']:
                    variance_type = 'SE' if 'e' in error_type.lower() else 'SD'
                    treatment_variance = treatment.get('error_value')
                    control_variance = matching_control.get('error_value')

                # Determine confidence
                confidence = 'medium'  # Default for figure extraction
                if treatment.get('labeled_on_figure') and matching_control.get('labeled_on_figure'):
                    confidence = 'high'
                elif treatment.get('confidence') == 'low' or matching_control.get('confidence') == 'low':
                    confidence = 'low'

                obs = Observation(
                    observation_id=f"{recon.paper_id}_fig_{obs_id}",
                    paper_id=recon.paper_id,
                    outcome_variable=self._infer_outcome_code(figure_data.get('y_axis_label', ''), pico),
                    outcome_name=figure_data.get('y_axis_label', ''),
                    treatment_description=treatment.get('group_label', 'Treatment'),
                    control_description=matching_control.get('group_label', 'Control'),
                    treatment_mean=treatment.get('value'),
                    control_mean=matching_control.get('value'),
                    treatment_n=recon.replicates,
                    control_n=recon.replicates,
                    variance_type=variance_type,
                    treatment_variance=treatment_variance,
                    control_variance=control_variance,
                    unit_reported=unit,
                    data_source=figure_id,
                    data_source_type='figure',
                    extraction_confidence=confidence,
                    notes=f"Extracted from figure via vision. {figure_data.get('notes', '')}"
                )

                # Add subgroup as moderator if present
                if subgroup:
                    obs.moderators = obs.moderators or {}
                    obs.moderators['subgroup'] = subgroup

                observations.append(obs)
                obs_id += 1

        return observations

    def _infer_outcome_code(self, y_label: str, pico: PICOSpec) -> str:
        """Infer outcome code from y-axis label"""
        y_lower = y_label.lower()

        # Common mappings
        if 'yield' in y_lower or 'grain' in y_lower:
            return 'GWAD'  # Grain weight at maturity
        elif 'biomass' in y_lower or 'biological' in y_lower:
            return 'CWAD'  # Crop weight at maturity
        elif 'height' in y_lower:
            return 'PLHT'
        elif 'protein' in y_lower:
            return 'PROT'
        elif '1000' in y_lower and 'grain' in y_lower:
            return 'TGW'

        # Default to first primary outcome
        if pico.primary_outcomes:
            return pico.primary_outcomes[0]

        return 'UNKNOWN'

    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON from LLM response"""
        try:
            # Try to find JSON in response
            match = re.search(r'\{[\s\S]*\}', response)
            if match:
                return json.loads(match.group())
        except:
            pass

        # Try fixing common issues
        try:
            match = re.search(r'\{[\s\S]*\}', response)
            if match:
                fixed = re.sub(r',(\s*[\]\}])', r'\1', match.group())
                return json.loads(fixed)
        except:
            pass

        return {'error': 'Could not parse JSON response', 'raw': response[:500]}

    def extract_and_convert(
        self,
        pdf_path: str,
        paper_text: str,
        recon: PaperRecon,
        table_observations: List[Observation],
        pico: PICOSpec,
        progress_callback=None
    ) -> Tuple[List[Observation], Dict]:
        """
        Main entry point: Check if figure extraction needed, extract if so.

        Returns:
            (additional_observations, extraction_report)
        """
        # Check if extraction is needed
        needed, figure_ids, reason = self.check_figure_extraction_needed(
            recon, table_observations, pico
        )

        report = {
            'extraction_attempted': needed,
            'reason': reason,
            'figures_identified': figure_ids,
            'figures_processed': 0,
            'observations_added': 0,
            'errors': []
        }

        if not needed:
            return [], report

        # Extract figures from PDF
        if progress_callback:
            progress_callback(0, len(figure_ids), "Extracting figures from PDF...")

        figures = self.extract_figures_from_pdf(pdf_path, figure_ids)
        report['figures_found'] = len(figures)

        if not figures:
            report['errors'].append("No figures could be extracted from PDF")
            return [], report

        # Process each figure
        all_observations = []

        for i, fig_info in enumerate(figures[:3]):  # Limit to 3 figures max (cost control)
            if progress_callback:
                progress_callback(i + 1, min(len(figures), 3),
                                 f"Processing {fig_info.get('figure_id', f'figure {i+1}')}...")

            try:
                # Extract data from figure
                fig_data = self.extract_data_from_figure(fig_info, paper_text, recon, pico)

                if 'error' in fig_data:
                    report['errors'].append(f"{fig_info.get('figure_id')}: {fig_data['error']}")
                    continue

                # Convert to observations
                observations = self.convert_figure_data_to_observations(fig_data, recon, pico)

                all_observations.extend(observations)
                report['figures_processed'] += 1

            except Exception as e:
                report['errors'].append(f"{fig_info.get('figure_id')}: {str(e)}")

        report['observations_added'] = len(all_observations)

        return all_observations, report
