"""
Single-LLM Baseline Extractor

Sends each PDF to ONE model with a SIMPLE prompt.
No recon, no consensus, no challenge routing, no table targeting.
This is the naive baseline to compare against the full consensus pipeline.

Usage:
    python single_llm_baseline.py --model claude --config configs/loladze_co2_minerals.json --input ../Loladze/validated --output output/baseline_claude_loladze
    python single_llm_baseline.py --model kimi --config configs/loladze_co2_minerals.json --input ../Loladze/validated --output output/baseline_kimi_loladze
    python single_llm_baseline.py --model gemini --config configs/loladze_co2_minerals.json --input ../Loladze/validated --output output/baseline_gemini_loladze
"""
import sys, os, json, time, argparse, re
from pathlib import Path
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from dotenv import load_dotenv
load_dotenv()

import fitz  # PyMuPDF


def extract_pdf_text(pdf_path: str) -> str:
    """Extract text from PDF using PyMuPDF only (no Kimi fallback for baseline)."""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    except Exception as e:
        print(f"  ERROR reading PDF: {e}")
        return ""


def build_simple_prompt(config: dict) -> str:
    """Build a minimal extraction prompt from config. No recon, no table targeting."""
    outcomes = ", ".join(config.get("primary_outcomes", ["all reported outcomes"]))
    intervention = config.get("intervention", "treatment")
    control = config.get("control", "control")

    return f"""You are extracting quantitative data from a scientific research paper for meta-analysis.

TASK: Extract all observations comparing {intervention} versus {control}.

TARGET OUTCOME: {outcomes}

For each observation, extract:
- element/variable name
- tissue or plant part
- control mean (numeric value)
- treatment mean (numeric value)
- sample size (n)
- variance type (SE, SD, LSD, or null)
- variance value (numeric, or null if not reported)
- unit of measurement
- any relevant moderators (cultivar, year, site, treatment details)

IMPORTANT RULES:
- Extract EVERY row from data tables. Do not average or pool across cultivars, years, treatments, or sites.
- If a table has 20 data rows, you should have approximately 20 observations.
- Return null for values you cannot find. Do NOT guess or fabricate values.

Return your results as a JSON object with this structure:
{{
    "paper_info": {{
        "title": "paper title",
        "authors": "first author et al.",
        "year": YYYY,
        "species": "species name"
    }},
    "observations": [
        {{
            "element": "variable name",
            "tissue": "grain | leaf | root | shoot | whole plant",
            "treatment_mean": number,
            "control_mean": number,
            "treatment_variance": number or null,
            "control_variance": number or null,
            "variance_type": "SE | SD | LSD | null",
            "n": number or null,
            "unit": "unit string",
            "data_source": "Table X or Figure Y",
            "treatment_description": "description",
            "control_description": "description",
            "moderators": {{}},
            "confidence": "high | medium | low",
            "notes": ""
        }}
    ]
}}

Return ONLY valid JSON, no other text."""


def call_claude(pdf_text: str, prompt: str) -> str:
    """Call Claude Sonnet 4 with simple prompt (streaming for long responses)."""
    import anthropic
    client = anthropic.Anthropic()

    # Truncate to 150K chars like the pipeline does
    text_chunk = pdf_text[:150000]

    result_text = ""
    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=32768,
        messages=[{
            "role": "user",
            "content": f"PAPER TEXT:\n\n{text_chunk}\n\n{prompt}"
        }]
    ) as stream:
        for text in stream.text_stream:
            result_text += text

    return result_text


def call_kimi(pdf_text: str, prompt: str) -> str:
    """Call Kimi K2.5 with simple prompt."""
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ.get("MOONSHOT_API_KEY") or os.environ.get("KIMI_API_KEY"),
        base_url="https://api.moonshot.ai/v1"
    )

    text_chunk = pdf_text[:200000]

    response = client.chat.completions.create(
        model="kimi-k2.5",
        temperature=1.0,
        messages=[
            {"role": "system", "content": f"PAPER TEXT:\n\n{text_chunk}"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=32768,
        extra_body={"thinking": {"type": "enabled"}}
    )
    return response.choices[0].message.content


def call_gemini(pdf_text: str, prompt: str) -> str:
    """Call Gemini 2.5 Flash with simple prompt."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    text_chunk = pdf_text[:500000]

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"PAPER TEXT:\n\n{text_chunk}\n\n{prompt}",
        config=types.GenerateContentConfig(
            max_output_tokens=32768,
            temperature=0.1,
        )
    )
    return response.text


def parse_json_response(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code block
    patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'(\{.*\})',
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    return {"error": "Failed to parse JSON", "raw": text[:500]}


def run_baseline(model: str, config_path: str, input_dir: str, output_dir: str):
    """Run single-model baseline extraction on all PDFs in input_dir."""

    # Load config
    with open(config_path) as f:
        config = json.load(f)

    print(f"{'='*70}")
    print(f"SINGLE-LLM BASELINE EXTRACTION")
    print(f"{'='*70}")
    print(f"Model: {model}")
    print(f"Config: {config['name']}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*70}")

    # Build prompt once (same for all papers — no per-paper recon)
    prompt = build_simple_prompt(config)

    # Select model function
    model_fn = {
        "claude": call_claude,
        "kimi": call_kimi,
        "gemini": call_gemini,
    }[model]

    # Ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)

    # Find all PDFs
    pdfs = sorted(Path(input_dir).glob("*.pdf"))
    print(f"\nFound {len(pdfs)} PDFs")

    results = []
    total_obs = 0
    errors = 0
    total_cost = 0.0

    for i, pdf_path in enumerate(pdfs):
        paper_id = pdf_path.stem
        output_file = Path(output_dir) / f"{paper_id}_baseline.json"

        # Skip if already extracted
        if output_file.exists():
            try:
                with open(output_file) as f:
                    existing = json.load(f)
                n_obs = len(existing.get("observations", []))
                print(f"  [{i+1}/{len(pdfs)}] SKIP {paper_id} (already done, {n_obs} obs)")
                total_obs += n_obs
                results.append({"paper": paper_id, "observations": n_obs, "status": "cached"})
                continue
            except:
                pass

        print(f"  [{i+1}/{len(pdfs)}] {paper_id}...", end=" ", flush=True)

        # Extract PDF text
        pdf_text = extract_pdf_text(str(pdf_path))
        if not pdf_text:
            print("NO TEXT")
            results.append({"paper": paper_id, "observations": 0, "status": "no_text"})
            errors += 1
            continue

        # Call model
        start_time = time.time()
        try:
            raw_response = model_fn(pdf_text, prompt)
            elapsed = time.time() - start_time

            # Parse JSON
            data = parse_json_response(raw_response)

            if "error" in data:
                print(f"PARSE ERROR ({elapsed:.1f}s)")
                # Save raw response for debugging
                data["_raw_response"] = raw_response[:2000]
                results.append({"paper": paper_id, "observations": 0, "status": "parse_error"})
                errors += 1
            else:
                n_obs = len(data.get("observations", []))
                total_obs += n_obs
                print(f"{n_obs} obs ({elapsed:.1f}s)")
                results.append({"paper": paper_id, "observations": n_obs, "status": "ok"})

            # Save result
            data["_meta"] = {
                "model": model,
                "baseline": True,
                "elapsed_seconds": elapsed,
                "pdf_text_length": len(pdf_text),
                "timestamp": datetime.now().isoformat()
            }
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"ERROR: {e} ({elapsed:.1f}s)")
            results.append({"paper": paper_id, "observations": 0, "status": f"error: {e}"})
            errors += 1

        # Brief pause between API calls
        time.sleep(1)

    # Summary
    print(f"\n{'='*70}")
    print(f"BASELINE COMPLETE: {model}")
    print(f"{'='*70}")
    print(f"Papers: {len(pdfs)}")
    print(f"Total observations: {total_obs}")
    print(f"Errors: {errors}")
    print(f"Avg obs/paper: {total_obs / max(1, len(pdfs) - errors):.1f}")

    # Save summary
    summary = {
        "model": model,
        "config": config["name"],
        "timestamp": datetime.now().isoformat(),
        "papers": len(pdfs),
        "total_observations": total_obs,
        "errors": errors,
        "results": results
    }
    summary_path = Path(output_dir) / "_baseline_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print(f"Summary saved to {summary_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single-LLM baseline extraction")
    parser.add_argument("--model", required=True, choices=["claude", "kimi", "gemini"],
                       help="Which LLM to use")
    parser.add_argument("--config", required=True, help="Path to config JSON")
    parser.add_argument("--input", required=True, help="Directory with PDFs")
    parser.add_argument("--output", required=True, help="Output directory")

    args = parser.parse_args()
    run_baseline(args.model, args.config, args.input, args.output)
