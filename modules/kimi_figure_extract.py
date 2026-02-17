"""
Kimi K2.5 Figure Extraction Module for Meta-Analysis

Uses Kimi K2.5's vision capabilities to extract quantitative data from:
- Bar charts
- Line graphs
- Scatter plots
- Tables rendered as images

Key Features:
- Native multimodal vision via Moonshot API
- Thinking mode for complex figure analysis
- Error bar extraction (SE, SD, 95% CI)
- Batch page processing
- Automatic figure detection
"""

import os
import json
import base64
import re
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime

# PDF processing
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    print("Warning: PyMuPDF not installed. Install with: pip install PyMuPDF")

# OpenAI-compatible API client
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


@dataclass
class FigureDataPoint:
    """Single data point extracted from a figure."""
    series_name: str
    category: str  # x-axis value or group
    mean: Optional[float] = None
    error_upper: Optional[float] = None  # Upper error bound or error value
    error_lower: Optional[float] = None  # Lower error bound (if asymmetric)
    error_value: Optional[float] = None  # Symmetric error (SE/SD)
    is_control: bool = False
    confidence: str = "medium"  # high, medium, low
    labeled_on_figure: bool = False  # True if value was explicitly labeled


@dataclass
class FigureExtraction:
    """Complete extraction from a single figure."""
    figure_id: str
    page_num: int
    figure_type: str  # bar chart, line graph, scatter plot, table, etc.
    x_axis_label: Optional[str] = None
    x_axis_categories: List[str] = field(default_factory=list)
    y_axis_label: Optional[str] = None
    y_axis_unit: Optional[str] = None
    error_bar_type: Optional[str] = None  # SE, SD, 95% CI, SEM, none, unknown
    legend: List[str] = field(default_factory=list)
    data_points: List[FigureDataPoint] = field(default_factory=list)
    notes: str = ""
    raw_response: Optional[str] = None
    extraction_time: Optional[float] = None


@dataclass
class FigureObservation:
    """Observation ready for meta-analysis (treatment vs control comparison)."""
    observation_id: str
    paper_id: str
    outcome_name: str
    treatment_description: str
    control_description: str
    treatment_mean: Optional[float] = None
    control_mean: Optional[float] = None
    treatment_n: Optional[int] = None
    control_n: Optional[int] = None
    variance_type: Optional[str] = None  # SE, SD
    treatment_variance: Optional[float] = None
    control_variance: Optional[float] = None
    unit: Optional[str] = None
    figure_id: str = ""
    page_num: int = 0
    confidence: str = "medium"
    notes: str = ""


class KimiFigureExtractor:
    """
    Extract quantitative data from scientific figures using Kimi K2.5 vision.

    Usage:
        extractor = KimiFigureExtractor()

        # Extract from specific pages
        extractions = extractor.extract_figures_from_pdf(
            "paper.pdf",
            pages=[3, 4, 5],
            context="CO2 effects on plant minerals"
        )

        # Convert to observations for meta-analysis
        observations = extractor.to_observations(extractions, paper_id="Smith_2020")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        mode: str = "thinking",  # "thinking" for thorough, "instant" for fast
        dpi: int = 200,  # Resolution for page rendering
        max_retries: int = 3
    ):
        """
        Initialize the Kimi figure extractor.

        Args:
            api_key: Moonshot API key (defaults to MOONSHOT_API_KEY env var)
            mode: "thinking" for thorough analysis, "instant" for faster extraction
            dpi: DPI for rendering PDF pages to images
            max_retries: Number of retries on API errors
        """
        self.api_key = api_key or os.environ.get("MOONSHOT_API_KEY")
        if not self.api_key:
            raise ValueError("MOONSHOT_API_KEY not found. Set it in .env or pass api_key parameter.")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.moonshot.ai/v1"
        )

        self.mode = mode
        self.dpi = dpi
        self.max_retries = max_retries
        self.model = "kimi-k2.5"

        # Kimi K2.5 parameters
        if mode == "thinking":
            self.temperature = 1.0
            self.thinking_config = {"thinking": {"type": "enabled"}}
        else:
            self.temperature = 0.6
            self.thinking_config = {"thinking": {"type": "disabled"}}

    def _page_to_base64(self, pdf_path: str, page_num: int) -> Optional[str]:
        """Convert a PDF page to base64-encoded PNG."""
        if not HAS_FITZ:
            raise ImportError("PyMuPDF required. Install with: pip install PyMuPDF")

        try:
            doc = fitz.open(pdf_path)
            if page_num >= len(doc):
                doc.close()
                return None

            page = doc[page_num]
            mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
            pix = page.get_pixmap(matrix=mat)
            png_bytes = pix.tobytes("png")
            doc.close()

            return base64.b64encode(png_bytes).decode("utf-8")
        except Exception as e:
            print(f"  Error converting page {page_num}: {e}")
            return None

    def _build_extraction_prompt(self, context: str = "") -> str:
        """Build the figure extraction prompt."""
        return f"""Analyze this scientific figure and extract ALL quantitative data points.

CONTEXT: {context if context else "Scientific research paper with quantitative data"}

INSTRUCTIONS:
1. IDENTIFY the figure type: bar chart, line graph, scatter plot, table, box plot, etc.
2. READ axis labels carefully - note the units
3. EXTRACT every visible data point:
   - For bar charts: read bar heights precisely from the y-axis scale
   - For line graphs: read each data point at every x-value
   - For scatter plots: estimate coordinates of each point
   - For tables: extract all numeric values
4. IDENTIFY error bars if present:
   - Read error bar lengths carefully
   - Determine type: SE, SD, SEM, 95% CI (often stated in caption or legend)
   - Record upper and lower bounds (may be asymmetric)
5. DETERMINE which groups are Control vs Treatment:
   - Control: usually "ambient", "0", "untreated", "control", lowest value
   - Treatment: elevated, positive dose, treated condition
6. If values are LABELED directly on bars/points, use those exact numbers (high confidence)
7. If ESTIMATING from axis scale, note medium confidence
8. If unclear or difficult to read, note low confidence

OUTPUT FORMAT (JSON only, no other text):
{{
    "figure_type": "bar chart | line graph | scatter plot | table | box plot | other",
    "x_axis_label": "exact label from figure",
    "x_axis_categories": ["Cat1", "Cat2", ...],
    "y_axis_label": "exact label from figure",
    "y_axis_unit": "unit if visible (mg/kg, %, g, etc.)",
    "error_bar_type": "SE | SD | SEM | 95% CI | unknown | none",
    "legend": ["Series1", "Series2", ...],
    "data_points": [
        {{
            "series_name": "Control or specific series name",
            "category": "x-axis category or value",
            "mean": 12.5,
            "error_upper": 14.0,
            "error_lower": 11.0,
            "error_value": 1.5,
            "is_control": true,
            "confidence": "high | medium | low",
            "labeled_on_figure": true
        }}
    ],
    "notes": "Any observations about data quality, reading difficulty, or important context"
}}

CRITICAL:
- Extract ALL data points visible in the figure, not just a subset
- Be precise with numbers - read carefully from the axis scale
- If there is NO figure or NO data on this page, return {{"figure_type": "none", "data_points": [], "notes": "No figure with data on this page"}}
- Output ONLY valid JSON, no explanations before or after"""

    def _parse_json_response(self, response: str) -> Dict:
        """Robustly parse JSON from LLM response."""
        if not response:
            return {"error": "Empty response", "data_points": []}

        # Strategy 1: Extract from code blocks
        if "```json" in response:
            match = re.search(r'```json\s*([\s\S]*?)```', response)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    pass

        if "```" in response:
            match = re.search(r'```\s*([\s\S]*?)```', response)
            if match:
                try:
                    return json.loads(match.group(1).strip())
                except json.JSONDecodeError:
                    pass

        # Strategy 2: Find JSON object with proper brace matching
        start_idx = response.find('{')
        if start_idx != -1:
            depth = 0
            in_string = False
            escape_next = False

            for i, char in enumerate(response[start_idx:], start_idx):
                if escape_next:
                    escape_next = False
                    continue
                if char == '\\' and in_string:
                    escape_next = True
                    continue
                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if not in_string:
                    if char == '{':
                        depth += 1
                    elif char == '}':
                        depth -= 1
                        if depth == 0:
                            try:
                                return json.loads(response[start_idx:i+1])
                            except json.JSONDecodeError:
                                break

        # Strategy 3: Simple brace finding
        start = response.find('{')
        end = response.rfind('}')
        if start != -1 and end > start:
            try:
                text = response[start:end+1]
                # Fix trailing commas
                text = re.sub(r',\s*([}\]])', r'\1', text)
                return json.loads(text)
            except json.JSONDecodeError:
                pass

        return {"error": "Could not parse JSON", "raw": response[:500], "data_points": []}

    def extract_figure_from_image(
        self,
        image_b64: str,
        context: str = "",
        page_num: int = 0
    ) -> FigureExtraction:
        """
        Extract data from a single figure image.

        Args:
            image_b64: Base64-encoded PNG image
            context: Context about the paper/study
            page_num: Page number for reference

        Returns:
            FigureExtraction with extracted data
        """
        prompt = self._build_extraction_prompt(context)

        start_time = time.time()

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=16384,
                    temperature=self.temperature,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                                }
                            ]
                        }
                    ],
                    extra_body=self.thinking_config
                )

                content = response.choices[0].message.content
                result = self._parse_json_response(content)

                extraction_time = time.time() - start_time

                # Convert to FigureExtraction
                data_points = []
                for dp in result.get("data_points", []):
                    data_points.append(FigureDataPoint(
                        series_name=dp.get("series_name", ""),
                        category=str(dp.get("category", "")),
                        mean=dp.get("mean"),
                        error_upper=dp.get("error_upper"),
                        error_lower=dp.get("error_lower"),
                        error_value=dp.get("error_value"),
                        is_control=dp.get("is_control", False),
                        confidence=dp.get("confidence", "medium"),
                        labeled_on_figure=dp.get("labeled_on_figure", False)
                    ))

                return FigureExtraction(
                    figure_id=f"page_{page_num+1}",
                    page_num=page_num + 1,
                    figure_type=result.get("figure_type", "unknown"),
                    x_axis_label=result.get("x_axis_label"),
                    x_axis_categories=result.get("x_axis_categories", []),
                    y_axis_label=result.get("y_axis_label"),
                    y_axis_unit=result.get("y_axis_unit"),
                    error_bar_type=result.get("error_bar_type"),
                    legend=result.get("legend", []),
                    data_points=data_points,
                    notes=result.get("notes", ""),
                    raw_response=content if "error" in result else None,
                    extraction_time=extraction_time
                )

            except Exception as e:
                if attempt < self.max_retries - 1:
                    print(f"  Attempt {attempt+1} failed: {e}. Retrying...")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return FigureExtraction(
                        figure_id=f"page_{page_num+1}",
                        page_num=page_num + 1,
                        figure_type="error",
                        notes=f"Extraction failed after {self.max_retries} attempts: {str(e)}"
                    )

    def extract_figures_from_pdf(
        self,
        pdf_path: str,
        pages: Optional[List[int]] = None,
        context: str = "",
        skip_no_figure: bool = True,
        progress_callback: Optional[callable] = None
    ) -> List[FigureExtraction]:
        """
        Extract figures from specific pages of a PDF.

        Args:
            pdf_path: Path to PDF file
            pages: List of page numbers (0-indexed). If None, scans pages 2-10.
            context: Context about the paper/study for better extraction
            skip_no_figure: If True, don't include pages with no figures in results
            progress_callback: Optional callback(current, total, message)

        Returns:
            List of FigureExtraction objects
        """
        if not HAS_FITZ:
            raise ImportError("PyMuPDF required. Install with: pip install PyMuPDF")

        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        doc.close()

        # Default: scan pages 2-10 (0-indexed: 2-9) where figures usually appear
        if pages is None:
            pages = list(range(2, min(10, num_pages)))

        paper_name = Path(pdf_path).stem
        print(f"Extracting figures from: {paper_name}")
        print(f"  Pages to scan: {[p+1 for p in pages]}")

        extractions = []

        for i, page_num in enumerate(pages):
            if progress_callback:
                progress_callback(i, len(pages), f"Processing page {page_num+1}")

            print(f"  Page {page_num+1}/{num_pages}...", end=" ")

            # Convert page to image
            image_b64 = self._page_to_base64(pdf_path, page_num)
            if not image_b64:
                print("skipped (conversion failed)")
                continue

            # Extract from image
            extraction = self.extract_figure_from_image(
                image_b64,
                context=f"{context}. Paper: {paper_name}, Page {page_num+1}",
                page_num=page_num
            )

            # Report result
            if extraction.figure_type == "none" or not extraction.data_points:
                print(f"no figure data")
                if skip_no_figure:
                    continue
            else:
                print(f"{extraction.figure_type}, {len(extraction.data_points)} data points")

            extractions.append(extraction)

        print(f"  Total extractions: {len(extractions)}")
        return extractions

    def to_observations(
        self,
        extractions: List[FigureExtraction],
        paper_id: str,
        control_keywords: List[str] = None,
        default_n: Optional[int] = None
    ) -> List[FigureObservation]:
        """
        Convert figure extractions to meta-analysis observations.

        Creates treatment-control pairs for each outcome.

        Args:
            extractions: List of FigureExtraction objects
            paper_id: Paper identifier
            control_keywords: Keywords to identify control groups
            default_n: Default sample size if not extractable

        Returns:
            List of FigureObservation objects ready for meta-analysis
        """
        if control_keywords is None:
            control_keywords = ["control", "ambient", "0", "untreated", "placebo", "ck", "370"]

        observations = []
        obs_count = 0

        for extraction in extractions:
            if not extraction.data_points or extraction.figure_type in ["none", "error"]:
                continue

            # Group data points by category (x-axis value)
            by_category = {}
            for dp in extraction.data_points:
                cat = dp.category
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append(dp)

            # Also group by series for multi-series figures
            by_series = {}
            for dp in extraction.data_points:
                series = dp.series_name
                if series not in by_series:
                    by_series[series] = []
                by_series[series].append(dp)

            # Determine variance type
            variance_type = None
            if extraction.error_bar_type:
                et = extraction.error_bar_type.upper()
                if "SE" in et or "SEM" in et:
                    variance_type = "SE"
                elif "SD" in et:
                    variance_type = "SD"
                elif "CI" in et:
                    variance_type = "CI_95"

            # Strategy 1: Explicit control flag
            controls = [dp for dp in extraction.data_points if dp.is_control]
            treatments = [dp for dp in extraction.data_points if not dp.is_control]

            # Strategy 2: Keyword-based control identification
            if not controls:
                for dp in extraction.data_points:
                    label = (dp.series_name + " " + dp.category).lower()
                    if any(kw in label for kw in control_keywords):
                        controls.append(dp)
                    else:
                        treatments.append(dp)

            # Strategy 3: If still no controls, use first category as control
            if not controls and by_category:
                first_cat = list(by_category.keys())[0]
                controls = by_category[first_cat]
                treatments = [dp for dp in extraction.data_points if dp.category != first_cat]

            # Create observations by pairing treatments with matching controls
            for treatment in treatments:
                # Find matching control (same series or same category pattern)
                matching_control = None

                # Try to match by series (for multi-category figures)
                for ctrl in controls:
                    if ctrl.series_name == treatment.series_name:
                        matching_control = ctrl
                        break

                # Fall back to first control
                if not matching_control and controls:
                    matching_control = controls[0]

                if not matching_control:
                    continue

                obs_count += 1

                # Calculate error values
                treat_var = treatment.error_value
                if treat_var is None and treatment.error_upper and treatment.mean:
                    treat_var = treatment.error_upper - treatment.mean

                ctrl_var = matching_control.error_value
                if ctrl_var is None and matching_control.error_upper and matching_control.mean:
                    ctrl_var = matching_control.error_upper - matching_control.mean

                # Determine confidence
                confidence = "medium"
                if treatment.confidence == "high" and matching_control.confidence == "high":
                    confidence = "high"
                elif treatment.confidence == "low" or matching_control.confidence == "low":
                    confidence = "low"

                # Create outcome name from y-axis label
                outcome_name = extraction.y_axis_label or "Unknown"
                if extraction.y_axis_unit:
                    outcome_name = f"{outcome_name} ({extraction.y_axis_unit})"

                obs = FigureObservation(
                    observation_id=f"{paper_id}_fig_p{extraction.page_num}_{obs_count}",
                    paper_id=paper_id,
                    outcome_name=outcome_name,
                    treatment_description=f"{treatment.series_name} {treatment.category}".strip(),
                    control_description=f"{matching_control.series_name} {matching_control.category}".strip(),
                    treatment_mean=treatment.mean,
                    control_mean=matching_control.mean,
                    treatment_n=default_n,
                    control_n=default_n,
                    variance_type=variance_type,
                    treatment_variance=treat_var,
                    control_variance=ctrl_var,
                    unit=extraction.y_axis_unit,
                    figure_id=extraction.figure_id,
                    page_num=extraction.page_num,
                    confidence=confidence,
                    notes=f"From {extraction.figure_type}. {extraction.notes}"
                )
                observations.append(obs)

        return observations

    def save_extractions(
        self,
        extractions: List[FigureExtraction],
        output_path: str
    ):
        """Save extractions to JSON file."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "mode": self.mode,
            "extractions": [asdict(e) for e in extractions]
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"Saved extractions to: {output_path}")

    def save_observations(
        self,
        observations: List[FigureObservation],
        output_path: str,
        format: str = "json"
    ):
        """Save observations to JSON or CSV."""
        if format == "json":
            data = {
                "timestamp": datetime.now().isoformat(),
                "total_observations": len(observations),
                "observations": [asdict(o) for o in observations]
            }
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        elif format == "csv":
            import csv
            fieldnames = [
                "observation_id", "paper_id", "outcome_name",
                "treatment_description", "control_description",
                "treatment_mean", "control_mean",
                "treatment_n", "control_n",
                "variance_type", "treatment_variance", "control_variance",
                "unit", "figure_id", "page_num", "confidence", "notes"
            ]

            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for obs in observations:
                    writer.writerow(asdict(obs))

        print(f"Saved {len(observations)} observations to: {output_path}")


def test_extraction(pdf_path: str, pages: List[int] = None, context: str = ""):
    """Test figure extraction on a single PDF."""
    print("=" * 70)
    print("KIMI K2.5 FIGURE EXTRACTION TEST")
    print("=" * 70)

    extractor = KimiFigureExtractor(mode="thinking")

    extractions = extractor.extract_figures_from_pdf(
        pdf_path,
        pages=pages,
        context=context
    )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    def safe_print(text):
        """Print with Unicode fallback for Windows console."""
        try:
            print(text)
        except UnicodeEncodeError:
            print(text.encode('ascii', 'replace').decode())

    total_data_points = 0
    for ext in extractions:
        safe_print(f"\nPage {ext.page_num}: {ext.figure_type}")
        if ext.y_axis_label:
            safe_print(f"  Y-axis: {ext.y_axis_label} ({ext.y_axis_unit or 'no unit'})")
        if ext.error_bar_type:
            safe_print(f"  Error bars: {ext.error_bar_type}")
        safe_print(f"  Data points: {len(ext.data_points)}")
        total_data_points += len(ext.data_points)

        if ext.data_points and len(ext.data_points) <= 10:
            for dp in ext.data_points:
                ctrl = " [CONTROL]" if dp.is_control else ""
                err = f" +/- {dp.error_value}" if dp.error_value else ""
                safe_print(f"    - {dp.series_name} / {dp.category}: {dp.mean}{err}{ctrl}")
        elif ext.data_points:
            safe_print(f"    (showing first 5 of {len(ext.data_points)})")
            for dp in ext.data_points[:5]:
                ctrl = " [CONTROL]" if dp.is_control else ""
                err = f" +/- {dp.error_value}" if dp.error_value else ""
                safe_print(f"    - {dp.series_name} / {dp.category}: {dp.mean}{err}{ctrl}")

    print(f"\nTotal data points extracted: {total_data_points}")

    # Convert to observations
    paper_id = Path(pdf_path).stem
    observations = extractor.to_observations(extractions, paper_id)
    print(f"Observations created: {len(observations)}")

    return extractions, observations


def batch_extract(
    pdf_dir: str,
    output_dir: str,
    context: str = "",
    pages: List[int] = None,
    limit: int = None
):
    """Run batch extraction on a directory of PDFs."""
    pdf_dir = Path(pdf_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = list(pdf_dir.glob("*.pdf"))
    if limit:
        pdfs = pdfs[:limit]

    print(f"Processing {len(pdfs)} PDFs...")

    extractor = KimiFigureExtractor(mode="thinking")

    all_observations = []

    for i, pdf_path in enumerate(pdfs):
        print(f"\n[{i+1}/{len(pdfs)}] {pdf_path.name}")

        try:
            extractions = extractor.extract_figures_from_pdf(
                str(pdf_path),
                pages=pages,
                context=context
            )

            # Save extractions for this paper
            paper_id = pdf_path.stem
            extractor.save_extractions(
                extractions,
                str(output_dir / f"{paper_id}_figure_extractions.json")
            )

            # Convert to observations
            observations = extractor.to_observations(extractions, paper_id)
            all_observations.extend(observations)

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    # Save all observations
    extractor.save_observations(
        all_observations,
        str(output_dir / "all_figure_observations.json"),
        format="json"
    )
    extractor.save_observations(
        all_observations,
        str(output_dir / "all_figure_observations.csv"),
        format="csv"
    )

    print(f"\n{'=' * 70}")
    print(f"BATCH EXTRACTION COMPLETE")
    print(f"{'=' * 70}")
    print(f"PDFs processed: {len(pdfs)}")
    print(f"Total observations: {len(all_observations)}")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python kimi_figure_extract.py <pdf_path> [pages] [context]")
        print("       python kimi_figure_extract.py --batch <pdf_dir> <output_dir> [context]")
        print("\nExamples:")
        print("  python kimi_figure_extract.py paper.pdf")
        print("  python kimi_figure_extract.py paper.pdf 3,4,5")
        print("  python kimi_figure_extract.py paper.pdf 3,4,5 'CO2 effects on wheat'")
        print("  python kimi_figure_extract.py --batch ./papers ./output 'CO2 effects'")
        sys.exit(1)

    if sys.argv[1] == "--batch":
        if len(sys.argv) < 4:
            print("Batch mode requires: <pdf_dir> <output_dir>")
            sys.exit(1)

        pdf_dir = sys.argv[2]
        output_dir = sys.argv[3]
        context = sys.argv[4] if len(sys.argv) > 4 else ""

        batch_extract(pdf_dir, output_dir, context)

    else:
        pdf_path = sys.argv[1]
        pages = None
        context = ""

        if len(sys.argv) > 2:
            try:
                pages = [int(p) - 1 for p in sys.argv[2].split(",")]
            except ValueError:
                context = sys.argv[2]

        if len(sys.argv) > 3:
            context = sys.argv[3]

        extractions, observations = test_extraction(pdf_path, pages, context)

        # Save results
        output_dir = Path(pdf_path).parent / "figure_extraction_output"
        output_dir.mkdir(exist_ok=True)

        extractor = KimiFigureExtractor()
        extractor.save_extractions(extractions, str(output_dir / f"{Path(pdf_path).stem}_extractions.json"))
        extractor.save_observations(observations, str(output_dir / f"{Path(pdf_path).stem}_observations.csv"), format="csv")
