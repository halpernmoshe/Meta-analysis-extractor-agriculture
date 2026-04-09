#!/usr/bin/env python3
"""
Seed the multi-role pilot with a small amount of structured content derived
from the existing per-paper reports, so the merger produces non-empty outputs.

This is not the final reader. It is a bootstrap demonstrator.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(r"C:\Users\moshe\Dropbox\Testing metaanalyis program\meta_analysis_extractor")
PILOT_DIR = ROOT / "pipeline_replication" / "codex" / "outputs" / "multi_role_pilot"


SEEDS = {
    "019_Baxter_1994": {
        "design_agent": {
            "claims": [
                {
                    "claim_id": "baxter_target",
                    "claim_key": "target::foliar_concentration",
                    "claim_text": "Benchmark target is foliar concentration, not total content.",
                    "element": "multi",
                    "outcome": "concentration",
                    "tissue": "leaf",
                    "arm": "ambient_vs_elevated_co2",
                    "timepoint": "final harvest per species",
                    "direction": "",
                    "unit": "mg/g structural DW",
                    "source_channel": "methods+report",
                    "source_locator": "report",
                    "benchmark_comparable": True,
                    "confidence": "high",
                    "evidence_quote": "GT requires foliar concentration data.",
                    "notes": "",
                }
            ],
            "constraints": [
                {
                    "constraint_type": "construct",
                    "constraint_text": "Do not substitute mg/plant total content for foliar concentration.",
                    "applies_to_claim_keys": ["target::foliar_concentration"],
                    "confidence": "high",
                    "source_locator": "report",
                }
            ],
        },
        "table_agent": {
            "claims": [
                {
                    "claim_id": "baxter_table",
                    "claim_key": "target::foliar_concentration",
                    "claim_text": "Extractable table data are total nutrient content per whole plant.",
                    "element": "multi",
                    "outcome": "total_content",
                    "tissue": "whole",
                    "arm": "ambient_vs_elevated_co2",
                    "timepoint": "final harvest",
                    "direction": "mixed",
                    "unit": "mg/plant",
                    "source_channel": "table",
                    "source_locator": "Table 1",
                    "benchmark_comparable": False,
                    "confidence": "high",
                    "evidence_quote": "These are content values, not concentrations.",
                    "notes": "",
                }
            ]
        },
        "figure_agent": {
            "claims": [
                {
                    "claim_id": "baxter_fig",
                    "claim_key": "target::foliar_concentration",
                    "claim_text": "Relevant concentration data exist only in figures.",
                    "element": "multi",
                    "outcome": "concentration",
                    "tissue": "leaf",
                    "arm": "ambient_vs_elevated_co2",
                    "timepoint": "multiple timepoints",
                    "direction": "",
                    "unit": "mg/g structural DW",
                    "source_channel": "figure",
                    "source_locator": "Figures 1-2",
                    "benchmark_comparable": True,
                    "confidence": "high",
                    "evidence_quote": "The GT entries are based on leaf blade concentration data read from Figures 1 and 2.",
                    "notes": "",
                }
            ]
        },
        "benchmark_agent": {
            "claims": [
                {
                    "claim_id": "baxter_benchmark",
                    "claim_key": "target::foliar_concentration",
                    "claim_text": "Table extraction is not benchmark-comparable because it is total content rather than foliar concentration.",
                    "element": "multi",
                    "outcome": "benchmark_target",
                    "tissue": "leaf",
                    "arm": "ambient_vs_elevated_co2",
                    "timepoint": "final harvest per species",
                    "direction": "",
                    "unit": "mg/g structural DW",
                    "source_channel": "benchmark_alignment",
                    "source_locator": "report",
                    "benchmark_comparable": True,
                    "confidence": "high",
                    "evidence_quote": "This is a data modality mismatch: total content per plant vs foliar concentration.",
                    "notes": "",
                }
            ]
        },
    },
    "026_Seneweera_1997": {
        "design_agent": {
            "claims": [
                {
                    "claim_id": "seneweera_target",
                    "claim_key": "target::grain_ca_zn_fe",
                    "claim_text": "Benchmark includes grain Ca, Zn, and Fe by P-level, distinct from blade averages.",
                    "element": "Ca/Zn/Fe",
                    "outcome": "grain_concentration",
                    "tissue": "grain",
                    "arm": "350_vs_700_co2",
                    "timepoint": "final harvest",
                    "direction": "",
                    "unit": "mixed",
                    "source_channel": "design",
                    "source_locator": "report",
                    "benchmark_comparable": True,
                    "confidence": "high",
                    "evidence_quote": "GT contains grain Ca/Zn/Fe at 6 P levels.",
                    "notes": "",
                }
            ]
        },
        "table_agent": {
            "claims": [
                {
                    "claim_id": "seneweera_table",
                    "claim_key": "target::grain_ca_zn_fe",
                    "claim_text": "Table extraction captured blade and grain N/P but not grain Ca/Zn/Fe.",
                    "element": "Ca/Zn/Fe",
                    "outcome": "grain_concentration",
                    "tissue": "leaf",
                    "arm": "350_vs_700_co2",
                    "timepoint": "56 DAP",
                    "direction": "mixed",
                    "unit": "table_only",
                    "source_channel": "table",
                    "source_locator": "Table 1 / Table 2",
                    "benchmark_comparable": False,
                    "confidence": "high",
                    "evidence_quote": "The validator had no grain Ca or Zn AI candidates and fell back to leaf values.",
                    "notes": "",
                }
            ]
        },
        "figure_agent": {
            "claims": [
                {
                    "claim_id": "seneweera_fig",
                    "claim_key": "target::grain_ca_zn_fe",
                    "claim_text": "Grain Ca, Zn, and Fe targets are figure-only and require digitization.",
                    "element": "Ca/Zn/Fe",
                    "outcome": "grain_concentration",
                    "tissue": "grain",
                    "arm": "350_vs_700_co2",
                    "timepoint": "final harvest",
                    "direction": "",
                    "unit": "figure_only",
                    "source_channel": "figure",
                    "source_locator": "Figures 3b, 4a, 4b",
                    "benchmark_comparable": True,
                    "confidence": "high",
                    "evidence_quote": "Grain Ca/Zn/Fe were never extracted; figures were not digitized.",
                    "notes": "",
                }
            ]
        },
        "benchmark_agent": {
            "claims": [
                {
                    "claim_id": "seneweera_benchmark",
                    "claim_key": "target::grain_ca_zn_fe",
                    "claim_text": "Leaf values are not benchmark-comparable substitutes for grain values.",
                    "element": "Ca/Zn/Fe",
                    "outcome": "benchmark_target",
                    "tissue": "grain",
                    "arm": "350_vs_700_co2",
                    "timepoint": "final harvest",
                    "direction": "",
                    "unit": "",
                    "source_channel": "benchmark_alignment",
                    "source_locator": "report",
                    "benchmark_comparable": True,
                    "confidence": "high",
                    "evidence_quote": "Wrong tissue matched; should be 0 matches.",
                    "notes": "",
                }
            ]
        },
    },
    "035_Oksanen_2005": {
        "design_agent": {
            "claims": [
                {
                    "claim_id": "oksanen_target",
                    "claim_key": "target::ec_only_leaf",
                    "claim_text": "Benchmark target is EC-only leaf concentration, not EC+EO.",
                    "element": "multi",
                    "outcome": "leaf_concentration",
                    "tissue": "leaf",
                    "arm": "EC_only",
                    "timepoint": "2000-2001 pooled",
                    "direction": "",
                    "unit": "table values",
                    "source_channel": "design",
                    "source_locator": "Table 8",
                    "benchmark_comparable": True,
                    "confidence": "high",
                    "evidence_quote": "Loladze 2014 used the EC-only arm pooled across both clones.",
                    "notes": "",
                }
            ]
        },
        "table_agent": {
            "claims": [
                {
                    "claim_id": "oksanen_table",
                    "claim_key": "target::ec_only_leaf",
                    "claim_text": "Table extraction captured both EC and EC+EO arms from Table 8.",
                    "element": "multi",
                    "outcome": "leaf_concentration",
                    "tissue": "leaf",
                    "arm": "EC+EO",
                    "timepoint": "2000-2001 pooled",
                    "direction": "mixed",
                    "unit": "mg/g or ug/g DW",
                    "source_channel": "table",
                    "source_locator": "Table 8",
                    "benchmark_comparable": False,
                    "confidence": "high",
                    "evidence_quote": "Our system extracted both EC and EC+EO arms.",
                    "notes": "",
                }
            ]
        },
        "benchmark_agent": {
            "claims": [
                {
                    "claim_id": "oksanen_benchmark",
                    "claim_key": "target::ec_only_leaf",
                    "claim_text": "EC+EO is a nearby but non-equivalent arm for a pure CO2 benchmark.",
                    "element": "multi",
                    "outcome": "benchmark_target",
                    "tissue": "leaf",
                    "arm": "EC_only",
                    "timepoint": "2000-2001 pooled",
                    "direction": "",
                    "unit": "",
                    "source_channel": "benchmark_alignment",
                    "source_locator": "report",
                    "benchmark_comparable": True,
                    "confidence": "high",
                    "evidence_quote": "Prefer the EC-only arm when matching CO2-effect papers.",
                    "notes": "",
                }
            ]
        },
    },
    "015_Pleijel_2009": {
        "design_agent": {
            "claims": [
                {
                    "claim_id": "pleijel_target",
                    "claim_key": "target::grain_zn_pure_co2",
                    "claim_text": "Benchmark target is pure CO2 grain Zn comparisons, excluding O3-only experiments.",
                    "element": "Zn",
                    "outcome": "grain_concentration",
                    "tissue": "grain",
                    "arm": "pure_co2",
                    "timepoint": "experiment-specific",
                    "direction": "",
                    "unit": "mg/kg",
                    "source_channel": "design",
                    "source_locator": "Table 1",
                    "benchmark_comparable": True,
                    "confidence": "high",
                    "evidence_quote": "O3-only experiments were properly excluded.",
                    "notes": "",
                }
            ]
        },
        "narrative_agent": {
            "claims": [
                {
                    "claim_id": "pleijel_narrative",
                    "claim_key": "target::grain_zn_pure_co2",
                    "claim_text": "Narrative and title support grain Zn dilution under elevated CO2.",
                    "element": "Zn",
                    "outcome": "grain_concentration",
                    "tissue": "grain",
                    "arm": "pure_co2",
                    "timepoint": "experiment-specific",
                    "direction": "negative",
                    "unit": "",
                    "source_channel": "narrative",
                    "source_locator": "title/abstract/report",
                    "benchmark_comparable": True,
                    "confidence": "high",
                    "evidence_quote": "Yield dilution of grain Zn in wheat grown in open-top chamber experiments with elevated CO2.",
                    "notes": "",
                }
            ]
        },
        "table_agent": {
            "claims": [
                {
                    "claim_id": "pleijel_table",
                    "claim_key": "target::grain_zn_pure_co2",
                    "claim_text": "Table extraction produced exact benchmark-matching Zn observations.",
                    "element": "Zn",
                    "outcome": "grain_concentration",
                    "tissue": "grain",
                    "arm": "pure_co2",
                    "timepoint": "experiment-specific",
                    "direction": "negative",
                    "unit": "mg/kg",
                    "source_channel": "table",
                    "source_locator": "Table 1",
                    "benchmark_comparable": True,
                    "confidence": "high",
                    "evidence_quote": "All three GT-matched observations show zero error.",
                    "notes": "",
                }
            ]
        },
        "benchmark_agent": {
            "claims": [
                {
                    "claim_id": "pleijel_benchmark",
                    "claim_key": "target::grain_zn_pure_co2",
                    "claim_text": "The extracted pure-CO2 grain Zn claims are benchmark-comparable and clean.",
                    "element": "Zn",
                    "outcome": "benchmark_target",
                    "tissue": "grain",
                    "arm": "pure_co2",
                    "timepoint": "experiment-specific",
                    "direction": "negative",
                    "unit": "mg/kg",
                    "source_channel": "benchmark_alignment",
                    "source_locator": "report",
                    "benchmark_comparable": True,
                    "confidence": "high",
                    "evidence_quote": "Perfect numerical accuracy on all 3 GT-matched observations.",
                    "notes": "",
                }
            ]
        },
    },
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main() -> None:
    for paper_id, role_map in SEEDS.items():
        paper_dir = PILOT_DIR / paper_id
        for role, additions in role_map.items():
            path = paper_dir / f"{role}.json"
            if not path.exists():
                continue
            data = load_json(path)
            payload = data.get("output_schema", {})
            payload["claims"] = additions.get("claims", [])
            payload["constraints"] = additions.get("constraints", [])
            payload["contradictions"] = additions.get("contradictions", [])
            payload["notes"] = additions.get("notes", "")
            data["output_schema"] = payload
            write_json(path, data)
            print(path)


if __name__ == "__main__":
    main()
