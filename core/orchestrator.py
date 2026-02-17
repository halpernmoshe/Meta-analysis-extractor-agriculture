"""
Orchestrator for Meta-Analysis Extraction System

Supports two modes:
- Interactive: User makes decisions via GUI dialogs
- Auto: Uses predefined config, logs all decisions to file
"""
import json
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

from core.state import (
    SessionState, Phase, PICOSpec, PaperRecon,
    Observation, ExtractionSchema, Decision,
    OutcomeField, ModeratorField
)
from core.llm import create_llm_client
from modules.recon import ReconModule
from modules.extract import ExtractModule
from modules.validate import ValidateModule
from modules.export import ExportModule
from modules.figure_extract import FigureExtractModule
from modules.gap_fill import GapFillModule
from modules.variance_rescue import VarianceRescueModule
from variance_pipeline import VariancePipeline
from gui.dialogue import DialogueManager
from config import (
    SAMPLE_SIZE_ORIENTATION, SAMPLE_SIZE_PILOT,
    MODEL_COSTS, AVG_INPUT_TOKENS_RECON, AVG_OUTPUT_TOKENS_RECON,
    AVG_INPUT_TOKENS_EXTRACT, AVG_OUTPUT_TOKENS_EXTRACT,
    CALLS_PER_PAPER_RECON, CALLS_PER_PAPER_EXTRACT
)

# Try to import domain configs
try:
    from domains import get_domain, list_domains, DOMAINS
    HAS_DOMAINS = True
except ImportError:
    HAS_DOMAINS = False
    DOMAINS = {}


# Default configuration for autonomous mode (CO2/mineral studies)
DEFAULT_AUTO_CONFIG = {
    "description": "Elevated CO2 effects on plant mineral concentrations",
    "pico": {
        "crop_species": ["All"],  # Accept all species
        "study_types": ["Field", "Greenhouse", "Growth Chamber", "FACE", "OTC"],
        "intervention_domain": "Elevated CO2",
        "intervention_variable": "CO2_CONC",
        "control_definition": "Ambient CO2 (~400 ppm)",
        "control_heuristic": "min",
        "control_keywords": ["ambient", "aCO2", "control", "360", "380", "400"],
        "primary_outcomes": ["MINERAL_CONC"],
        "secondary_outcomes": ["CWAD", "GWAD", "RWAD", "NUPT", "PROT"],
        "required_moderators": [
            "CO2_CONC", "CROP_SP", "CULTIVAR", "COUNTRY", "LAT", "LON",
            "STUDY_TYPE", "FERT_N", "PLANT_PART", "DURATION", "CO2_ELEV"
        ]
    },
    "models": {
        "recon": "auto",  # Use provider default
        "extract": "auto"
    },
    "resume_if_exists": True,
    "skip_confirmations": True
}


class Orchestrator:
    """Main orchestrator for meta-analysis extraction"""

    # Checkpoint save frequency during long operations
    CHECKPOINT_INTERVAL = 5  # Save every N papers during recon/extraction

    def __init__(self, input_dir: str, output_dir: str, api_key: str = None,
                 provider: str = None, auto_mode: bool = False,
                 auto_config_path: Optional[str] = None,
                 domain: Optional[str] = None):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.provider = provider or "anthropic"
        self.auto_mode = auto_mode
        self.domain_config = None

        # Load domain configuration if specified
        if domain and HAS_DOMAINS:
            try:
                self.domain_config = get_domain(domain)
                print(f"Loaded domain config: {self.domain_config.get('name', domain)}")
            except ValueError as e:
                print(f"Warning: {e}. Available domains: {list_domains()}")

        # Load auto config
        if auto_mode:
            if auto_config_path:
                with open(auto_config_path) as f:
                    self.auto_config = json.load(f)
                print(f"Loaded custom config from: {auto_config_path}")
            else:
                self.auto_config = DEFAULT_AUTO_CONFIG.copy()
                print("Using default auto config (CO2/mineral studies)")
            self._init_auto_log()
        else:
            self.auto_config = None

        # Create LLM client with specified provider
        if self.provider == "google":
            self.llm = create_llm_client(provider="google", google_api_key=api_key)
        else:
            self.llm = create_llm_client(api_key=api_key, provider="anthropic")
        self.dialogue = DialogueManager()
        self.recon = ReconModule(self.llm)
        self.extract = ExtractModule(self.llm, output_dir=self.output_dir)
        self.validate = ValidateModule(self.llm)
        self.export = ExportModule()
        self.figure_extract = FigureExtractModule(self.llm)
        self.gap_fill = GapFillModule(self.llm, str(self.output_dir), str(self.input_dir))
        self.variance_rescue = VarianceRescueModule(self.llm, str(self.output_dir), str(self.input_dir))

        ontology_path = Path(__file__).parent.parent / "data" / "ontology.json"
        with open(ontology_path) as f:
            self.ontology = json.load(f)

        # Check for existing session state
        self.state = None
        self.resumed = False
        self.resume_phase = None

        self.papers = self._discover_papers()

        # Try to load existing state
        state_file = self.output_dir / "session_state.json"
        if state_file.exists():
            self._handle_existing_state(state_file)

        # Create new state if not resuming
        if self.state is None:
            self.state = SessionState(
                session_id=datetime.now().strftime("%Y%m%d_%H%M%S"),
                created_at=datetime.now().isoformat(),
                last_updated=datetime.now().isoformat(),
                input_directory=str(input_dir),
                output_directory=str(output_dir)
            )

        self.state.paper_count = len(self.papers)

    def _init_auto_log(self):
        """Initialize the auto-decision log file"""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.auto_log_path = self.output_dir / "auto_decisions.log"
        with open(self.auto_log_path, 'a') as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"AUTO MODE RUN - {datetime.now().isoformat()}\n")
            f.write(f"{'='*60}\n")
            f.write(f"Config: {self.auto_config.get('description', 'Custom')}\n\n")

    def _log_auto_decision(self, phase: str, decision: str, value: str):
        """Log an automatic decision"""
        msg = f"[{phase}] {decision}: {value}"
        print(f"  AUTO: {msg}")
        if hasattr(self, 'auto_log_path'):
            with open(self.auto_log_path, 'a') as f:
                f.write(f"{datetime.now().strftime('%H:%M:%S')} {msg}\n")

    def _handle_existing_state(self, state_file: Path):
        """Handle existing session state - ask user whether to resume"""
        try:
            loaded_state = SessionState.load(str(state_file))

            # Show what we found
            print(f"\n{'='*60}")
            print("EXISTING SESSION FOUND")
            print(f"{'='*60}")
            print(f"Session ID: {loaded_state.session_id}")
            print(f"Last phase: {loaded_state.current_phase.value}")
            print(f"Papers scanned: {len(loaded_state.recon_cache)}/{loaded_state.paper_count}")
            print(f"Papers extracted: {len(loaded_state.extractions)}")
            print(f"Last updated: {loaded_state.last_updated}")
            print(f"{'='*60}\n")

            # In auto mode, check config for resume behavior
            if self.auto_mode:
                resume = self.auto_config.get('resume_if_exists', True)
                if resume:
                    self._log_auto_decision("RESUME", "Resuming existing session", "Yes")
                    self.state = loaded_state
                    self.resumed = True
                    self.resume_phase = loaded_state.current_phase
                    print(f"Resuming from phase: {self.resume_phase.value}")
                else:
                    self._log_auto_decision("RESUME", "Starting fresh", "Config says no resume")
                    import shutil
                    backup_file = self.output_dir / f"session_state_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    shutil.copy(state_file, backup_file)
                    print(f"Previous session backed up to: {backup_file}")
                return

            # Interactive mode - ask user what to do
            choice = self.dialogue.ask_choice(
                "Resume Session",
                "What would you like to do?",
                [
                    "Resume from where we left off",
                    "Start fresh (discard previous progress)"
                ]
            )

            if choice and "Resume" in choice:
                self.state = loaded_state
                self.resumed = True
                self.resume_phase = loaded_state.current_phase
                print(f"Resuming from phase: {self.resume_phase.value}")
            else:
                # Back up old state before starting fresh
                backup_file = self.output_dir / f"session_state_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                import shutil
                shutil.copy(state_file, backup_file)
                print(f"Previous session backed up to: {backup_file}")

        except Exception as e:
            print(f"Warning: Could not load existing state: {e}")
            print("Starting fresh session.")
    
    def _discover_papers(self) -> List[Dict]:
        papers = []
        for pdf_path in sorted(self.input_dir.glob("*.pdf")):
            papers.append({
                'id': pdf_path.stem,
                'path': str(pdf_path),
                'filename': pdf_path.name
            })
        return papers

    def _estimate_cost(self, model_id: str, num_papers: int, task_type: str) -> float:
        """Estimate cost for a given model and number of papers"""
        if model_id not in MODEL_COSTS:
            return 0.0

        costs = MODEL_COSTS[model_id]

        if task_type == 'recon':
            input_tokens = AVG_INPUT_TOKENS_RECON * CALLS_PER_PAPER_RECON * num_papers
            output_tokens = AVG_OUTPUT_TOKENS_RECON * CALLS_PER_PAPER_RECON * num_papers
        else:  # extraction
            input_tokens = AVG_INPUT_TOKENS_EXTRACT * CALLS_PER_PAPER_EXTRACT * num_papers
            output_tokens = AVG_OUTPUT_TOKENS_EXTRACT * CALLS_PER_PAPER_EXTRACT * num_papers

        input_cost = (input_tokens / 1_000_000) * costs['input_per_1m']
        output_cost = (output_tokens / 1_000_000) * costs['output_per_1m']

        return input_cost + output_cost

    def _select_models(self) -> bool:
        """Ask user to select models for recon and extraction"""
        num_papers = len(self.papers)

        # In auto mode, use config models or provider defaults
        if self.auto_mode:
            # Check if config specifies models
            config_models = self.auto_config.get('models', {})
            config_recon = config_models.get('recon', 'auto')
            config_extract = config_models.get('extract', 'auto')

            # Use config models if specified (not 'auto')
            if config_recon != 'auto':
                selected_recon = config_recon
            else:
                selected_recon = self.llm.recon_model

            if config_extract != 'auto':
                selected_extract = config_extract
            else:
                selected_extract = self.llm.extract_model

            # Apply the selected models
            self.llm.set_models(recon_model=selected_recon, extract_model=selected_extract)
            self._log_auto_decision("MODELS", "Recon model", selected_recon)
            self._log_auto_decision("MODELS", "Extract model", selected_extract)
            self.state.recon_model = selected_recon
            self.state.extract_model = selected_extract
            return True

        # Build cost estimates table
        model_ids = list(MODEL_COSTS.keys())
        model_names = [MODEL_COSTS[m]['name'] for m in model_ids]

        print(f"\n{'='*70}")
        print("MODEL SELECTION")
        print(f"{'='*70}")
        print(f"\nYou have {num_papers} papers to process.\n")
        print("Cost estimates by model combination:")
        print("-" * 70)
        print(f"{'Recon Model':<15} {'Extract Model':<15} {'Recon Cost':<12} {'Extract Cost':<12} {'Total':<10}")
        print("-" * 70)

        # Show some common combinations
        combinations = [
            ("claude-3-5-haiku-20241022", "claude-3-5-haiku-20241022"),
            ("claude-3-5-haiku-20241022", "claude-sonnet-4-20250514"),
            ("claude-sonnet-4-20250514", "claude-sonnet-4-20250514"),
            ("claude-3-5-haiku-20241022", "claude-opus-4-20250514"),
            ("claude-3-5-haiku-20241022", "claude-opus-4-5-20251101"),
        ]

        for recon_m, extract_m in combinations:
            recon_cost = self._estimate_cost(recon_m, num_papers, 'recon')
            extract_cost = self._estimate_cost(extract_m, num_papers, 'extraction')
            total = recon_cost + extract_cost
            recon_name = MODEL_COSTS[recon_m]['name']
            extract_name = MODEL_COSTS[extract_m]['name']
            print(f"{recon_name:<15} {extract_name:<15} ${recon_cost:<11.2f} ${extract_cost:<11.2f} ${total:<9.2f}")

        print("-" * 70)
        print("\nRecommendation: Haiku for recon (speed), Sonnet for extraction (accuracy)")
        print()

        # Ask for recon model
        recon_options = [f"{MODEL_COSTS[m]['name']} (${self._estimate_cost(m, num_papers, 'recon'):.2f})" for m in model_ids]
        recon_choice = self.dialogue.ask_choice(
            "Recon Model",
            "Select model for reconnaissance (scanning papers):",
            recon_options
        )
        if not recon_choice:
            return False

        recon_idx = recon_options.index(recon_choice)
        selected_recon = model_ids[recon_idx]

        # Ask for extraction model
        extract_options = [f"{MODEL_COSTS[m]['name']} (${self._estimate_cost(m, num_papers, 'extraction'):.2f})" for m in model_ids]
        extract_choice = self.dialogue.ask_choice(
            "Extraction Model",
            "Select model for data extraction (needs accuracy):",
            extract_options
        )
        if not extract_choice:
            return False

        extract_idx = extract_options.index(extract_choice)
        selected_extract = model_ids[extract_idx]

        # Calculate and show total
        total_cost = (self._estimate_cost(selected_recon, num_papers, 'recon') +
                      self._estimate_cost(selected_extract, num_papers, 'extraction'))

        print(f"\n{'='*70}")
        print(f"Selected: {MODEL_COSTS[selected_recon]['name']} (recon) + {MODEL_COSTS[selected_extract]['name']} (extract)")
        print(f"Estimated total cost: ${total_cost:.2f}")
        print(f"{'='*70}\n")

        # Apply models to LLM client and save to state
        self.llm.set_models(recon_model=selected_recon, extract_model=selected_extract)
        self.state.recon_model = selected_recon
        self.state.extract_model = selected_extract

        return self.dialogue.ask_confirm("Confirm", f"Proceed with estimated cost of ${total_cost:.2f}?")
    
    def run(self):
        print(f"\n{'='*60}")
        print("META-ANALYSIS DATA EXTRACTION SYSTEM")
        if self.auto_mode:
            print(">>> AUTONOMOUS MODE - No user interaction required <<<")
        print(f"{'='*60}")
        print(f"Found {len(self.papers)} papers in {self.input_dir}")
        print(f"Output: {self.output_dir}")
        print(f"Provider: {self.provider}")
        if self.resumed:
            print(f"RESUMING from: {self.resume_phase.value}")
        if self.auto_mode:
            print(f"Config: {self.auto_config.get('description', 'Custom')}")
            print(f"Decision log: {self.auto_log_path}")
        print(f"{'='*60}\n")

        if not self.papers:
            self.dialogue.show_error("No Papers", f"No PDFs in {self.input_dir}")
            return

        # Model selection with cost estimates (skip if resuming with saved models)
        if self.resumed and self.state.recon_model and self.state.extract_model:
            # Restore saved model selection
            self.llm.set_models(recon_model=self.state.recon_model, extract_model=self.state.extract_model)
            print(f"Using saved models: {self.state.recon_model} (recon), {self.state.extract_model} (extract)")
        else:
            if not self._select_models():
                print("Model selection cancelled.")
                return

        phases = [
            (Phase.ORIENTATION, self._phase_orientation),
            (Phase.PICO_POPULATION, self._phase_pico_population),
            (Phase.PICO_INTERVENTION, self._phase_pico_intervention),
            (Phase.PICO_COMPARISON, self._phase_pico_comparison),
            (Phase.PICO_OUTCOMES, self._phase_pico_outcomes),
            (Phase.MODERATORS, self._phase_moderators),
            (Phase.FULL_RECON, self._phase_full_recon),
            (Phase.GAP_ANALYSIS, self._phase_gap_analysis),
            (Phase.SCHEMA_REVIEW, self._phase_schema_review),
            (Phase.PILOT_EXTRACTION, self._phase_pilot_extraction),
            (Phase.PILOT_REVIEW, self._phase_pilot_review),
            (Phase.FULL_EXTRACTION, self._phase_full_extraction),
            (Phase.GAP_FILL, self._phase_gap_fill),
            (Phase.VARIANCE_EXTRACTION, self._phase_variance_extraction),
            (Phase.VARIANCE_RESCUE, self._phase_variance_rescue),
            (Phase.VARIANCE_VERIFICATION, self._phase_variance_verification),
            (Phase.VALIDATION, self._phase_validation),
            (Phase.EXPORT, self._phase_export),
        ]

        # Find starting point for resume
        phase_order = [p for p, _ in phases]
        start_idx = 0
        if self.resumed:
            if self.resume_phase == Phase.COMPLETE:
                # Session already complete - just show results
                print("Session already complete. Nothing to do.")
                print(f"\n{'='*60}")
                print("COMPLETE")
                print(f"{'='*60}")
                print(f"Results: {self.output_dir}")
                print(f"{'='*60}")
                return
            elif self.resume_phase in phase_order:
                start_idx = phase_order.index(self.resume_phase)
                print(f"Skipping {start_idx} completed phase(s)...")

        for i, (phase, handler) in enumerate(phases):
            # Skip completed phases when resuming
            if i < start_idx:
                continue

            self.state.current_phase = phase
            self.state.update_timestamp()
            print(f"\n--- {phase.value} ---")

            try:
                if not handler():
                    print("Stopped by user.")
                    self._save_session()  # Save on user stop
                    print(f"Progress saved. You can resume later.")
                    break
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                self._save_session()  # Save on error
                print(f"Progress saved. You can resume later.")
                if self.auto_mode:
                    self._log_auto_decision("ERROR", "Error occurred", str(e))
                    print("Auto mode: continuing despite error...")
                    continue
                if not self.dialogue.ask_confirm("Error", f"{e}\n\nContinue?"):
                    break

            # Checkpoint save after each phase completes successfully
            self._save_session()
            print(f"[Checkpoint saved after {phase.value}]")

        self.state.current_phase = Phase.COMPLETE
        self._save_session()

        print(f"\n{'='*60}")
        print("COMPLETE")
        print(f"{'='*60}")
        print(f"Results: {self.output_dir}")
        if self.auto_mode:
            print(f"Decision log: {self.auto_log_path}")
            # Write final summary to log
            with open(self.auto_log_path, 'a') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"COMPLETED: {datetime.now().isoformat()}\n")
                f.write(f"Total papers: {len(self.papers)}\n")
                f.write(f"Papers extracted: {len(self.state.extractions)}\n")
                f.write(f"Total observations: {self.state.total_observations}\n")
                f.write(f"{'='*60}\n")
        print(f"{'='*60}\n")
    
    def _phase_orientation(self) -> bool:
        n = min(SAMPLE_SIZE_ORIENTATION, len(self.papers))
        sample = self.papers[:n]

        progress = self.dialogue.show_progress("Orientation", f"Scanning {n} papers...", 0, "")
        for i, p in enumerate(sample):
            progress.update((i/n)*100, f"Scanning {p['filename']}...")
            recon = self.recon.scan_overview(p['path'])
            recon.paper_id = p['id']
            recon.file_path = p['path']
            self.state.recon_cache[p['id']] = recon
        progress.close()

        synthesis = self.recon.synthesize_recon_results(self.state.recon_cache)

        if self.auto_mode:
            self._log_auto_decision("ORIENTATION", "Papers scanned", f"{n}/{len(self.papers)}")
            self._log_auto_decision("ORIENTATION", "Crops found", str(list(synthesis.get('crops', {}).keys())))
            self._log_auto_decision("ORIENTATION", "Proceed to PICO", "Yes (auto)")
            return True

        self.dialogue.show_summary("Orientation", {
            "Papers": f"{n}/{len(self.papers)}",
            "Crops": synthesis.get('crops', {}),
            "Study Types": synthesis.get('study_types', {}),
            "Outcomes": synthesis.get('outcomes', {})
        })

        return self.dialogue.ask_confirm("Proceed?", "Continue to PICO definition?")
    
    def _phase_pico_population(self) -> bool:
        if self.auto_mode:
            pico_cfg = self.auto_config.get('pico', {})
            self.state.pico.crop_species = pico_cfg.get('crop_species', ['All'])
            self.state.pico.study_types = pico_cfg.get('study_types',
                ["Field", "Greenhouse", "Growth Chamber", "FACE", "OTC"])
            self._log_auto_decision("PICO_POPULATION", "Crop species", str(self.state.pico.crop_species))
            self._log_auto_decision("PICO_POPULATION", "Study types", str(self.state.pico.study_types))
            self._record_decision(Phase.PICO_POPULATION, "Population", [],
                f"{self.state.pico.crop_species}, {self.state.pico.study_types} (AUTO)")
            return True

        crops = list(self.recon.synthesize_recon_results(self.state.recon_cache).get('crops', {}).keys())
        if not crops:
            crops = ["Wheat", "Maize", "Rice", "Soybean", "Barley", "Other"]

        selected = self.dialogue.ask_multiselect("PICO: Population", "Select crops:", crops, min_select=1)
        if not selected:
            return False
        self.state.pico.crop_species = selected

        study_opts = ["Field trials only", "Field + Greenhouse", "All"]
        st = self.dialogue.ask_choice("PICO: Study Types", "Include which study types?", study_opts)
        if not st:
            return False

        if "only" in st:
            self.state.pico.study_types = ["Field"]
        elif "Greenhouse" in st:
            self.state.pico.study_types = ["Field", "Greenhouse"]
        else:
            self.state.pico.study_types = ["Field", "Greenhouse", "Growth Chamber", "FACE", "OTC"]

        self._record_decision(Phase.PICO_POPULATION, "Population", [], f"{selected}, {st}")
        return True
    
    def _phase_pico_intervention(self) -> bool:
        if self.auto_mode:
            pico_cfg = self.auto_config.get('pico', {})
            self.state.pico.intervention_domain = pico_cfg.get('intervention_domain', 'Elevated CO2')
            self.state.pico.intervention_variable = pico_cfg.get('intervention_variable', 'CO2_CONC')
            self._log_auto_decision("PICO_INTERVENTION", "Domain", self.state.pico.intervention_domain)
            self._log_auto_decision("PICO_INTERVENTION", "Variable", self.state.pico.intervention_variable)
            self._record_decision(Phase.PICO_INTERVENTION, "Intervention", [],
                f"{self.state.pico.intervention_domain} (AUTO)")
            return True

        domains = ["Nitrogen fertilizer", "Phosphorus fertilizer", "Organic amendments",
                   "Elevated CO2", "Tillage", "Irrigation", "Biostimulants", "Other"]

        selected = self.dialogue.ask_choice("PICO: Intervention", "Intervention type?", domains, allow_custom=True)
        if not selected:
            return False

        self.state.pico.intervention_domain = selected
        if "nitrogen" in selected.lower():
            self.state.pico.intervention_variable = "FERT_N"
        elif "co2" in selected.lower():
            self.state.pico.intervention_variable = "CO2_CONC"

        self._record_decision(Phase.PICO_INTERVENTION, "Intervention", domains, selected)
        return True
    
    def _phase_pico_comparison(self) -> bool:
        if self.auto_mode:
            pico_cfg = self.auto_config.get('pico', {})
            self.state.pico.control_definition = pico_cfg.get('control_definition', 'Ambient CO2 (~400 ppm)')
            self.state.pico.control_heuristic = pico_cfg.get('control_heuristic', 'min')
            self.state.pico.control_keywords = pico_cfg.get('control_keywords',
                ["ambient", "aCO2", "control", "360", "380", "400"])
            self._log_auto_decision("PICO_COMPARISON", "Control definition", self.state.pico.control_definition)
            self._log_auto_decision("PICO_COMPARISON", "Control keywords", str(self.state.pico.control_keywords))
            self._record_decision(Phase.PICO_COMPARISON, "Control", [],
                f"{self.state.pico.control_definition} (AUTO)")
            return True

        domain = self.state.pico.intervention_domain.lower()

        if "nitrogen" in domain or "fertilizer" in domain:
            opts = ["Zero input (0 kg/ha)", "Lowest rate", "Farmer practice", "Custom"]
            keywords = ["0", "zero", "unfertilized", "control", "N0"]
            heuristic = "min"
        elif "co2" in domain:
            opts = ["Ambient CO2 (~400 ppm)", "Lowest CO2", "Custom"]
            keywords = ["ambient", "aCO2", "control"]
            heuristic = "min"
        else:
            opts = ["Untreated control", "Conventional practice", "Custom"]
            keywords = ["control", "untreated", "check"]
            heuristic = "keyword"

        selected = self.dialogue.ask_choice("PICO: Control", "Control definition?", opts, allow_custom=True)
        if not selected:
            return False

        self.state.pico.control_definition = selected
        self.state.pico.control_heuristic = heuristic
        self.state.pico.control_keywords = keywords

        self._record_decision(Phase.PICO_COMPARISON, "Control", opts, selected)
        return True
    
    def _phase_pico_outcomes(self) -> bool:
        if self.auto_mode:
            pico_cfg = self.auto_config.get('pico', {})
            self.state.pico.primary_outcomes = pico_cfg.get('primary_outcomes', ['MINERAL_CONC'])
            self.state.pico.secondary_outcomes = pico_cfg.get('secondary_outcomes',
                ['CWAD', 'GWAD', 'RWAD', 'NUPT', 'PROT'])
            self._log_auto_decision("PICO_OUTCOMES", "Primary outcomes", str(self.state.pico.primary_outcomes))
            self._log_auto_decision("PICO_OUTCOMES", "Secondary outcomes", str(self.state.pico.secondary_outcomes))
            self._record_decision(Phase.PICO_OUTCOMES, "Outcomes", [],
                f"P:{self.state.pico.primary_outcomes}, S:{self.state.pico.secondary_outcomes} (AUTO)")
            return True

        outcomes = list(self.ontology.get('outcomes', {}).keys())
        display = [f"{c}: {self.ontology['outcomes'][c].get('name', c)}" for c in outcomes]

        primary = self.dialogue.ask_multiselect("PICO: Primary Outcomes",
            "Select PRIMARY outcomes:", display, min_select=1, max_select=3)
        if not primary:
            return False

        self.state.pico.primary_outcomes = [p.split(":")[0] for p in primary]

        remaining = [d for d in display if d not in primary]
        secondary = self.dialogue.ask_multiselect("PICO: Secondary Outcomes",
            "Select SECONDARY outcomes:", remaining, min_select=0)
        self.state.pico.secondary_outcomes = [s.split(":")[0] for s in (secondary or [])]

        self._record_decision(Phase.PICO_OUTCOMES, "Outcomes", [], f"P:{primary}, S:{secondary}")
        return True
    
    def _phase_moderators(self) -> bool:
        if self.auto_mode:
            pico_cfg = self.auto_config.get('pico', {})
            self.state.pico.required_moderators = pico_cfg.get('required_moderators', [
                "CO2_CONC", "CROP_SP", "CULTIVAR", "COUNTRY", "LAT", "LON",
                "STUDY_TYPE", "FERT_N", "PLANT_PART", "DURATION", "CO2_ELEV"
            ])
            self._log_auto_decision("MODERATORS", "Selected moderators", str(self.state.pico.required_moderators))
            self._record_decision(Phase.MODERATORS, "Moderators", [],
                f"{self.state.pico.required_moderators} (AUTO)")
            return True

        mods = []
        for cat, items in self.ontology.get('moderators', {}).items():
            for code, info in items.items():
                mods.append(f"{code}: {info.get('name', code)} [{cat}]")

        selected = self.dialogue.ask_multiselect("Moderators", "Select moderators:", mods, min_select=1)
        if not selected:
            return False

        self.state.pico.required_moderators = [s.split(":")[0] for s in selected]
        self._record_decision(Phase.MODERATORS, "Moderators", mods, str(selected))
        return True
    
    def _phase_full_recon(self) -> bool:
        # Skip papers already in recon_cache (for resume)
        papers_to_scan = [p for p in self.papers if p['id'] not in self.state.recon_cache]
        already_done = len(self.papers) - len(papers_to_scan)

        if already_done > 0:
            print(f"Resuming: {already_done} papers already scanned, {len(papers_to_scan)} remaining")

        if not papers_to_scan:
            print("All papers already scanned.")
        else:
            progress = self.dialogue.show_progress("Full Recon", f"Scanning {len(papers_to_scan)} papers...", 0, "")

            for i, p in enumerate(papers_to_scan):
                total_done = already_done + i + 1
                progress.update((total_done/len(self.papers))*100, f"{p['filename']} ({total_done}/{len(self.papers)})")

                recon = self.recon.scan_detailed(p['path'], self.state.pico, self.ontology)
                recon.paper_id = p['id']
                recon.file_path = p['path']
                recon.meets_pico_criteria = self._check_pico(recon)
                self.state.recon_cache[p['id']] = recon

                # Checkpoint save every N papers
                if (i + 1) % self.CHECKPOINT_INTERVAL == 0:
                    self._save_session()
                    print(f"  [Checkpoint: {total_done}/{len(self.papers)} papers scanned]")

            progress.close()

        meeting = sum(1 for r in self.state.recon_cache.values() if r.meets_pico_criteria)
        self.state.papers_meeting_criteria = meeting
        self.dialogue.show_info("Recon Complete", f"{meeting}/{len(self.papers)} meet criteria")
        return True
    
    def _phase_gap_analysis(self) -> bool:
        synthesis = self.recon.synthesize_recon_results(self.state.recon_cache)

        if self.auto_mode:
            self._log_auto_decision("GAP_ANALYSIS", "Papers meeting criteria",
                f"{self.state.papers_meeting_criteria}/{len(self.papers)}")
            self._log_auto_decision("GAP_ANALYSIS", "Outcomes found", str(list(synthesis.get('outcomes', {}).keys())))
            self._log_auto_decision("GAP_ANALYSIS", "Proceed to extraction", "Yes (auto)")
            return True

        self.dialogue.show_summary("Gap Analysis", {
            "Meeting Criteria": f"{self.state.papers_meeting_criteria}/{len(self.papers)}",
            "Outcomes": synthesis.get('outcomes', {}),
            "Variance": synthesis.get('variance_types', {}),
            "Countries": synthesis.get('countries', {})
        })
        return self.dialogue.ask_confirm("Proceed?", "Continue to extraction?")
    
    def _phase_schema_review(self) -> bool:
        self.state.schema = self._build_schema()
        text = self._format_schema()

        if self.auto_mode:
            self._log_auto_decision("SCHEMA_REVIEW", "Schema built", "Auto-approved")
            print(f"  Schema:\n{text}")
            return True

        return self.dialogue.ask_confirm("Schema", f"{text}\n\nProceed?")
    
    def _phase_pilot_extraction(self) -> bool:
        pilots = [p for p in self.papers
                  if self.state.recon_cache.get(p['id'], PaperRecon(p['id'], p['path'])).meets_pico_criteria
                 ][:SAMPLE_SIZE_PILOT]

        if not pilots:
            self.dialogue.show_warning("No Papers", "No papers meet criteria")
            return False

        # Check which pilots already extracted (for resume)
        to_extract = [p for p in pilots if p['id'] not in self.state.extractions]

        if not to_extract:
            total = sum(len(o) for o in self.state.extractions.values())
            self.dialogue.show_info("Pilot Complete", f"Already extracted. {total} observations from {len(pilots)} papers")
            return True

        progress = self.dialogue.show_progress("Pilot", f"Extracting from {len(to_extract)} papers...", 0, "")

        for i, p in enumerate(to_extract):
            progress.update(((i+1)/len(to_extract))*100, f"{p['filename']}...")
            recon = self.state.recon_cache[p['id']]
            obs = self.extract.extract_paper(p['path'], self.state.schema, recon,
                                              self.state.pico, self.ontology, self.domain_config)

            # DISABLED: Figure extraction - tables only mode
            # try:
            #     fig_obs, fig_report = self.figure_extract.extract_and_convert(
            #         pdf_path=p['path'],
            #         paper_text="",
            #         recon=recon,
            #         table_observations=obs,
            #         pico=self.state.pico
            #     )
            #     if fig_obs:
            #         obs.extend(fig_obs)
            #         print(f"  + Added {len(fig_obs)} observations from figures")
            # except Exception as fig_e:
            #     print(f"  Note: Figure extraction skipped: {fig_e}")

            self.state.extractions[p['id']] = obs
            # Save after each pilot paper (they're expensive)
            self._save_session()

        progress.close()

        total = sum(len(o) for o in self.state.extractions.values())
        self.dialogue.show_info("Pilot Complete", f"{total} observations from {len(pilots)} papers")
        return True
    
    def _phase_pilot_review(self) -> bool:
        pilot_dir = self.output_dir / "pilot"
        pilot_dir.mkdir(parents=True, exist_ok=True)

        for pid, obs in self.state.extractions.items():
            self.export.to_json(obs, str(pilot_dir / f"{pid}.json"), pid)

        total = sum(len(o) for o in self.state.extractions.values())

        if self.auto_mode:
            self._log_auto_decision("PILOT_REVIEW", "Pilot observations", str(total))
            self._log_auto_decision("PILOT_REVIEW", "Pilot JSONs saved", str(pilot_dir))
            self._log_auto_decision("PILOT_REVIEW", "Proceed to full extraction", "Yes (auto)")
            return True

        return self.dialogue.ask_confirm("Pilot Review",
            f"{total} observations extracted.\nReview JSONs in {pilot_dir}\n\nProceed?")
    
    def _phase_full_extraction(self) -> bool:
        # Get papers that meet criteria and haven't been extracted yet
        meeting_criteria = [p for p in self.papers
                           if self.state.recon_cache.get(p['id'], PaperRecon(p['id'], p['path'])).meets_pico_criteria]
        to_extract = [p for p in meeting_criteria if p['id'] not in self.state.extractions]
        already_done = len(meeting_criteria) - len(to_extract)

        if already_done > 0:
            print(f"Resuming: {already_done} papers already extracted, {len(to_extract)} remaining")

        if not to_extract:
            self.dialogue.show_info("Complete", "All papers already extracted")
            return True

        progress = self.dialogue.show_progress("Extraction", f"Extracting {len(to_extract)} papers...", 0, "")
        figure_extraction_count = 0

        for i, p in enumerate(to_extract):
            total_done = already_done + i + 1
            progress.update((total_done/len(meeting_criteria))*100, f"{p['filename']} ({total_done}/{len(meeting_criteria)})")

            try:
                recon = self.state.recon_cache[p['id']]
                obs = self.extract.extract_paper(p['path'], self.state.schema, recon,
                                                  self.state.pico, self.ontology, self.domain_config)

                # DISABLED: Figure extraction - tables only mode
                # try:
                #     fig_obs, fig_report = self.figure_extract.extract_and_convert(
                #         pdf_path=p['path'],
                #         paper_text="",  # Already extracted in recon
                #         recon=recon,
                #         table_observations=obs,
                #         pico=self.state.pico,
                #         progress_callback=lambda cur, tot, msg: progress.update(
                #             (total_done/len(meeting_criteria))*100, f"{p['filename']} - {msg}"
                #         )
                #     )
                #
                #     if fig_obs:
                #         obs.extend(fig_obs)
                #         figure_extraction_count += 1
                #         if self.auto_mode:
                #             self._log_auto_decision("FIGURE_EXTRACT", p['id'],
                #                 f"Added {len(fig_obs)} obs from figures: {fig_report.get('reason', 'gap detected')}")
                #         else:
                #             print(f"  + Added {len(fig_obs)} observations from figures")
                #
                # except Exception as fig_e:
                #     # Figure extraction is optional - don't fail if it errors
                #     print(f"  Note: Figure extraction skipped for {p['id']}: {fig_e}")

                self.state.extractions[p['id']] = obs

            except Exception as e:
                print(f"Error {p['id']}: {e}")
                self.state.extractions[p['id']] = [Observation(
                    observation_id=f"{p['id']}_error",
                    paper_id=p['id'],
                    notes=f"Error: {e}"
                )]

            # Checkpoint save every N papers
            if (i + 1) % self.CHECKPOINT_INTERVAL == 0:
                self._save_session()
                print(f"  [Checkpoint: {total_done}/{len(meeting_criteria)} papers extracted]")

        progress.close()

        total = sum(len(o) for o in self.state.extractions.values())
        self.state.total_observations = total

        summary_msg = f"{total} observations from {len(self.state.extractions)} papers"
        if figure_extraction_count > 0:
            summary_msg += f"\n({figure_extraction_count} papers had figures extracted)"
        self.dialogue.show_info("Complete", summary_msg)
        return True

    def _phase_gap_fill(self) -> bool:
        """Run gap fill to find missing variance, sample sizes, and moderators"""
        # First export so summary.csv exists for gap analysis
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.export.export_all(self.state, str(self.output_dir))

        # Check if gap fill should run
        run_gap_fill = False
        if self.auto_mode:
            # In auto mode, run gap fill if enabled in config (default: True)
            run_gap_fill = self.auto_config.get('gap_fill', {}).get('enabled', True)
            self._log_auto_decision("GAP_FILL", "Run gap fill", "Yes" if run_gap_fill else "No (disabled in config)")
        else:
            # Ask user in interactive mode
            run_gap_fill = self.dialogue.ask_confirm(
                "Gap Fill",
                "Would you like to run targeted gap filling to find missing variance, sample sizes, and moderators?\n\n"
                "This makes additional API calls to look more closely at papers with missing data."
            )

        if not run_gap_fill:
            return True

        try:
            # Analyze gaps
            gaps = self.gap_fill.analyze_gaps()

            if not gaps:
                print("No gaps found - extraction is complete!")
                return True

            self.gap_fill.print_gap_summary(gaps)

            # Get settings from config
            max_papers = None
            priority = 'variance'
            if self.auto_mode:
                gap_config = self.auto_config.get('gap_fill', {})
                max_papers = gap_config.get('max_papers', None)
                priority = gap_config.get('priority', 'variance')
                self._log_auto_decision("GAP_FILL", "Settings",
                    f"max_papers={max_papers}, priority={priority}")

            # Run gap fill
            df = self.gap_fill.run(max_papers=max_papers, priority=priority)

            # Reload observations from filled CSV into state
            # This updates the state so validation and export use filled data
            self._reload_observations_from_csv()

            if self.auto_mode:
                self._log_auto_decision("GAP_FILL", "Complete",
                    f"Processed {len(gaps)} papers")

        except Exception as e:
            print(f"Gap fill error: {e}")
            if self.auto_mode:
                self._log_auto_decision("GAP_FILL", "Error", str(e))
            # Gap fill is optional - continue even if it fails

        return True

    def _reload_observations_from_csv(self):
        """Reload observations from summary.csv after gap fill"""
        import pandas as pd
        summary_path = self.output_dir / 'summary.csv'
        if not summary_path.exists():
            return

        df = pd.read_csv(summary_path)

        # Update state.extractions with filled values
        for paper_id, group in df.groupby('paper_id'):
            if paper_id in self.state.extractions:
                # Update each observation
                for obs in self.state.extractions[paper_id]:
                    row = group[group['observation_id'] == obs.observation_id]
                    if not row.empty:
                        row = row.iloc[0]
                        # Update variance if filled
                        if pd.notna(row.get('treatment_variance')) and obs.treatment_variance is None:
                            obs.treatment_variance = row['treatment_variance']
                        if pd.notna(row.get('control_variance')) and obs.control_variance is None:
                            obs.control_variance = row['control_variance']
                        if pd.notna(row.get('pooled_variance')) and obs.pooled_variance is None:
                            obs.pooled_variance = row['pooled_variance']
                        if pd.notna(row.get('variance_type')) and not obs.variance_type:
                            obs.variance_type = row['variance_type']
                        # Update n if filled
                        if pd.notna(row.get('treatment_n')) and obs.treatment_n is None:
                            obs.treatment_n = int(row['treatment_n'])
                        if pd.notna(row.get('control_n')) and obs.control_n is None:
                            obs.control_n = int(row['control_n'])

    def _phase_variance_extraction(self) -> bool:
        """Dedicated variance extraction pass - focuses ONLY on finding variance values"""
        # Check if variance extraction should run
        run_extraction = False
        if self.auto_mode:
            run_extraction = self.auto_config.get('variance_extraction', {}).get('enabled', True)
            self._log_auto_decision("VARIANCE_EXTRACTION", "Run variance extraction",
                                   "Yes" if run_extraction else "No (disabled in config)")
        else:
            run_extraction = self.dialogue.ask_confirm(
                "Variance Extraction",
                "Would you like to run a dedicated variance extraction pass?\n\n"
                "This makes focused API calls to find variance values (SE, SD, LSD)\n"
                "in table footnotes, figure captions, and Methods sections."
            )

        if not run_extraction:
            return True

        try:
            import fitz  # PyMuPDF

            # Count observations missing variance
            missing_count = 0
            papers_with_missing = set()
            for paper_id, observations in self.state.extractions.items():
                for obs in observations:
                    has_variance = (obs.treatment_variance is not None or
                                   obs.control_variance is not None or
                                   obs.pooled_variance is not None)
                    if not has_variance:
                        missing_count += 1
                        papers_with_missing.add(paper_id)

            if missing_count == 0:
                print("All observations already have variance values!")
                return True

            print(f"\n{'='*60}")
            print("VARIANCE EXTRACTION PASS")
            print(f"{'='*60}")
            print(f"Observations missing variance: {missing_count}")
            print(f"Papers to process: {len(papers_with_missing)}")
            print()

            # Build variance extraction prompt
            VARIANCE_EXTRACTION_PROMPT = '''You are extracting ONLY variance information from this scientific paper.

PAPER TEXT:
{paper_text}

TASK: Find ALL variance/error values reported in this paper.

SEARCH THESE LOCATIONS CAREFULLY:
1. TABLE FOOTNOTES - Look for "Values are means ± SE", "LSD(0.05) = X", "SD shown in parentheses"
2. METHODS SECTION - Look for "data presented as mean ± SE", "standard error of the mean"
3. FIGURE CAPTIONS - Look for "Error bars represent SE", "bars indicate SD"
4. RESULTS TEXT - Look for "mean of 12.5 ± 1.2"

EXISTING OBSERVATIONS TO MATCH:
{observations_list}

For each observation above, find the variance value if possible.

OUTPUT FORMAT (JSON):
{{
    "global_variance_type": "SE" or "SD" or "LSD" or null,
    "global_variance_evidence": "quote from paper",
    "variance_values": [
        {{
            "observation_id": "matching ID from list above",
            "variance_type": "SE" or "SD" or "LSD" or "CV",
            "treatment_variance": number or null,
            "control_variance": number or null,
            "pooled_variance": number or null,
            "source": "Table 1 footnote" or "Methods section",
            "evidence": "quote showing this value"
        }}
    ]
}}

IMPORTANT:
- Extract ACTUAL NUMERIC VALUES, not just types
- Match to observation_ids from the list above
- If a single LSD value applies to all rows, report it as pooled_variance
- If variance differs by treatment, report treatment_variance and control_variance separately
'''

            # Get PDF paths
            pdf_paths = {p['id']: p['path'] for p in self.papers}

            # Process each paper with missing variance
            total_filled = 0
            progress = self.dialogue.show_progress("Variance Extraction",
                f"Processing {len(papers_with_missing)} papers...", 0, "")

            for idx, paper_id in enumerate(papers_with_missing):
                progress.update(
                    (idx / len(papers_with_missing)) * 100,
                    f"Extracting from {paper_id}..."
                )

                pdf_path = pdf_paths.get(paper_id)
                if not pdf_path:
                    continue

                # Read PDF text
                try:
                    doc = fitz.open(pdf_path)
                    paper_text = "\n".join(page.get_text() for page in doc)
                    doc.close()
                except Exception as e:
                    continue

                # Build observations list for this paper
                observations = self.state.extractions.get(paper_id, [])
                obs_list_str = "\n".join([
                    f"- {obs.observation_id}: {obs.outcome_name or obs.outcome_variable}, "
                    f"treatment={obs.treatment_mean}, control={obs.control_mean}"
                    for obs in observations
                    if obs.treatment_variance is None and obs.control_variance is None and obs.pooled_variance is None
                ])

                if not obs_list_str:
                    continue

                # Call LLM for variance extraction
                prompt = VARIANCE_EXTRACTION_PROMPT.format(
                    paper_text=paper_text[:60000],
                    observations_list=obs_list_str
                )

                try:
                    response = self.llm.call_extract(prompt, "Extract variance values precisely. Output valid JSON only.")
                    result = self.llm.parse_json_response(response)

                    if 'error' in result:
                        continue

                    # Apply extracted variance to observations
                    global_type = result.get('global_variance_type')
                    variance_values = result.get('variance_values', [])

                    # Create lookup by observation_id
                    obs_lookup = {obs.observation_id: obs for obs in observations}

                    for var_data in variance_values:
                        obs_id = var_data.get('observation_id')
                        if obs_id and obs_id in obs_lookup:
                            obs = obs_lookup[obs_id]

                            # Update variance values
                            if var_data.get('treatment_variance') is not None and obs.treatment_variance is None:
                                obs.treatment_variance = var_data['treatment_variance']
                                total_filled += 1

                            if var_data.get('control_variance') is not None and obs.control_variance is None:
                                obs.control_variance = var_data['control_variance']
                                total_filled += 1

                            if var_data.get('pooled_variance') is not None and obs.pooled_variance is None:
                                obs.pooled_variance = var_data['pooled_variance']
                                total_filled += 1

                            # Update variance type
                            if var_data.get('variance_type') and not obs.variance_type:
                                obs.variance_type = var_data['variance_type']

                            # Add notes
                            source = var_data.get('source', '')
                            if source:
                                obs.notes = (obs.notes or '') + f" [Variance from: {source}]"

                    # Apply global type to remaining observations without type
                    if global_type:
                        for obs in observations:
                            if not obs.variance_type:
                                obs.variance_type = global_type

                except Exception as e:
                    continue

            progress.close()

            # Show summary
            print(f"\n{'='*60}")
            print("VARIANCE EXTRACTION COMPLETE")
            print(f"{'='*60}")
            print(f"Variance values filled: {total_filled}")
            print(f"{'='*60}")

            if self.auto_mode:
                self._log_auto_decision("VARIANCE_EXTRACTION", "Complete", f"Filled: {total_filled}")

            # Update summary CSV
            self._update_summary_csv()

        except ImportError as e:
            print(f"Variance extraction not available: {e}")
        except Exception as e:
            print(f"Variance extraction error: {e}")
            import traceback
            traceback.print_exc()

        return True

    def _phase_variance_rescue(self) -> bool:
        """Run variance rescue using vision API for missing variance values"""
        # Check if variance rescue should run
        run_rescue = False
        if self.auto_mode:
            # In auto mode, run if enabled in config (default: True)
            run_rescue = self.auto_config.get('variance_rescue', {}).get('enabled', True)
            self._log_auto_decision("VARIANCE_RESCUE", "Run variance rescue",
                                   "Yes" if run_rescue else "No (disabled in config)")
        else:
            # Ask user in interactive mode
            run_rescue = self.dialogue.ask_confirm(
                "Variance Rescue",
                "Would you like to run vision-based variance rescue?\n\n"
                "This uses the vision API to read table images and extract\n"
                "variance values (LSD, SE, SD) from table footnotes."
            )

        if not run_rescue:
            return True

        try:
            # Find rescue targets
            targets = self.variance_rescue.find_rescue_targets()

            if not targets:
                print("No variance rescue targets found!")
                return True

            self.variance_rescue.print_rescue_summary(targets)

            # Get settings from config
            max_papers = None
            if self.auto_mode:
                rescue_config = self.auto_config.get('variance_rescue', {})
                max_papers = rescue_config.get('max_papers', None)
                self._log_auto_decision("VARIANCE_RESCUE", "Settings",
                    f"max_papers={max_papers}")

            # Run variance rescue
            df = self.variance_rescue.run(max_papers=max_papers)

            # Reload observations from rescued CSV into state
            self._reload_observations_from_csv()

            if self.auto_mode:
                self._log_auto_decision("VARIANCE_RESCUE", "Complete",
                    f"Processed {len(targets)} papers")

        except Exception as e:
            print(f"Variance rescue error: {e}")
            import traceback
            traceback.print_exc()
            if self.auto_mode:
                self._log_auto_decision("VARIANCE_RESCUE", "Error", str(e))
            # Variance rescue is optional - continue even if it fails

        return True

    def _phase_variance_verification(self) -> bool:
        """Run variance verification pipeline using GRIM, GRIMMER, and cross-validation"""
        # Check if variance verification should run
        run_verification = False
        if self.auto_mode:
            # In auto mode, run if enabled in config (default: True)
            run_verification = self.auto_config.get('variance_verification', {}).get('enabled', True)
            self._log_auto_decision("VARIANCE_VERIFICATION", "Run variance verification",
                                   "Yes" if run_verification else "No (disabled in config)")
        else:
            # Ask user in interactive mode
            run_verification = self.dialogue.ask_confirm(
                "Variance Verification",
                "Would you like to run variance verification?\n\n"
                "This applies mathematical checks (GRIM, GRIMMER, CV bounds)\n"
                "and cross-validates SE/SD types within each paper."
            )

        if not run_verification:
            return True

        try:
            # Initialize the variance pipeline
            pipeline = VariancePipeline()

            # Get list of PDF paths for text extraction
            pdf_paths = {p['id']: p['path'] for p in self.papers}

            # Process each paper's extractions
            total_verified = 0
            total_flagged = 0
            papers_processed = 0

            progress = self.dialogue.show_progress("Variance Verification",
                f"Verifying {len(self.state.extractions)} papers...", 0, "")

            for paper_id, observations in self.state.extractions.items():
                papers_processed += 1
                progress.update(
                    (papers_processed / len(self.state.extractions)) * 100,
                    f"Verifying {paper_id}..."
                )

                # Skip papers with no observations
                if not observations:
                    continue

                # Get paper text for global variance scanning
                pdf_path = pdf_paths.get(paper_id)
                full_text = ""
                if pdf_path:
                    try:
                        import fitz
                        doc = fitz.open(pdf_path)
                        full_text = "\n".join(page.get_text() for page in doc)
                        doc.close()
                    except Exception:
                        pass

                # Get journal name from recon cache
                journal = None
                if paper_id in self.state.recon_cache:
                    recon = self.state.recon_cache[paper_id]
                    # Journal might be in citation or title
                    journal = getattr(recon, 'journal', None)

                # Convert observations to the format expected by variance_pipeline
                # The pipeline processes individual values, not paired treatment/control
                extraction_dicts = []
                obs_map = {}  # Map index to observation for updating later

                for idx, obs in enumerate(observations):
                    # Process treatment values if present
                    if obs.treatment_mean is not None and obs.treatment_variance is not None:
                        extraction_dicts.append({
                            'mean': obs.treatment_mean,
                            'variance_value': obs.treatment_variance,
                            'claimed_type': obs.variance_type or 'UNKNOWN',
                            'n': obs.treatment_n,
                            'source': f"{obs.observation_id}_treatment",
                        })
                        obs_map[len(extraction_dicts) - 1] = (idx, 'treatment')

                    # Process control values if present
                    if obs.control_mean is not None and obs.control_variance is not None:
                        extraction_dicts.append({
                            'mean': obs.control_mean,
                            'variance_value': obs.control_variance,
                            'claimed_type': obs.variance_type or 'UNKNOWN',
                            'n': obs.control_n,
                            'source': f"{obs.observation_id}_control",
                        })
                        obs_map[len(extraction_dicts) - 1] = (idx, 'control')

                    # Process pooled variance if present
                    if obs.pooled_variance is not None and obs.treatment_mean is not None:
                        extraction_dicts.append({
                            'mean': obs.treatment_mean,
                            'variance_value': obs.pooled_variance,
                            'claimed_type': obs.variance_type or 'UNKNOWN',
                            'n': obs.treatment_n,
                            'source': f"{obs.observation_id}_pooled",
                        })
                        obs_map[len(extraction_dicts) - 1] = (idx, 'pooled')

                if not extraction_dicts:
                    # No variance values to verify - just apply global type if found
                    from global_variance_scanner import scan_for_global_variance
                    global_type, confidence, evidence = scan_for_global_variance(full_text)
                    if global_type and str(global_type) not in ('UNKNOWN', 'None'):
                        for obs in observations:
                            if not obs.variance_type:
                                obs.variance_type = str(global_type.value) if hasattr(global_type, 'value') else str(global_type)
                                obs.validation_notes = (obs.validation_notes or "") + f"Global type: {global_type}"
                    continue

                # Run the variance pipeline
                report = pipeline.process_paper(
                    paper_id=paper_id,
                    full_text=full_text,
                    extractions=extraction_dicts,
                    journal=journal
                )

                # Apply global variance type to observations without variance
                if report.global_variance_type and report.global_variance_type not in ('UNKNOWN', None):
                    for obs in observations:
                        if not obs.variance_type:
                            obs.variance_type = report.global_variance_type
                            obs.validation_notes = (obs.validation_notes or "") + f"Global: {report.global_evidence[:50]}"

                # Update observations with verification results
                for result_idx, result in enumerate(report.extractions):
                    if result_idx in obs_map:
                        obs_idx, value_type = obs_map[result_idx]
                        obs = observations[obs_idx]

                        # Update variance type if we have a better determination
                        if result.final_type and result.final_type not in ('UNKNOWN', None):
                            if not obs.variance_type or obs.variance_type == 'UNKNOWN':
                                obs.variance_type = result.final_type

                        # Add verification notes
                        verification_notes = []
                        for flag in result.verification_flags:
                            verification_notes.append(flag)

                        if result.cv_value:
                            verification_notes.append(f"CV={result.cv_value:.1f}%")

                        if verification_notes:
                            existing_notes = obs.validation_notes or ""
                            obs.validation_notes = existing_notes + " | " + ", ".join(verification_notes)

                        # Update confidence based on verification
                        if result.final_confidence == "high":
                            total_verified += 1
                        elif result.requires_review:
                            total_flagged += 1
                            obs.validation_status = "flagged"
                            if result.review_reasons:
                                obs.validation_notes = (obs.validation_notes or "") + f" Review: {', '.join(result.review_reasons)}"

            progress.close()

            # Show summary
            summary_msg = (
                f"Verified {total_verified} observations\n"
                f"Flagged {total_flagged} for review\n"
                f"Processed {papers_processed} papers"
            )
            self.dialogue.show_info("Variance Verification Complete", summary_msg)

            if self.auto_mode:
                self._log_auto_decision("VARIANCE_VERIFICATION", "Complete",
                    f"Verified: {total_verified}, Flagged: {total_flagged}")

        except ImportError as e:
            print(f"Variance verification not available: {e}")
            if self.auto_mode:
                self._log_auto_decision("VARIANCE_VERIFICATION", "Skipped", f"Import error: {e}")
        except Exception as e:
            print(f"Variance verification error: {e}")
            import traceback
            traceback.print_exc()
            if self.auto_mode:
                self._log_auto_decision("VARIANCE_VERIFICATION", "Error", str(e))
            # Variance verification is optional - continue even if it fails

        return True

    def _phase_validation(self) -> bool:
        progress = self.dialogue.show_progress("Validation", "Validating...", 0, "")
        
        validated = self.validate.validate_batch(
            self.state.extractions, self.ontology,
            lambda i, t, s: progress.update((i/t)*100 if t else 0, s)
        )
        self.state.extractions = validated
        progress.close()
        
        summary = self.validate.get_validation_summary(validated)
        self.state.flagged_observations = summary['flagged']
        self.dialogue.show_summary("Validation", summary)
        return True
    
    def _phase_export(self) -> bool:
        progress = self.dialogue.show_progress("Export", "Exporting...", 0, "")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.export.export_all(self.state, str(self.output_dir))
        
        progress.update(100, "Complete!")
        progress.close()
        
        self.dialogue.show_info("Export Complete", 
            f"Saved to {self.output_dir}:\n- extractions/*.json\n- summary.csv\n- papers.csv\n- methods.md")
        return True
    
    # Helpers
    
    def _check_pico(self, recon: PaperRecon) -> bool:
        reasons = []
        if self.state.pico.study_types:
            if recon.study_type and recon.study_type not in self.state.pico.study_types:
                reasons.append(f"Study type: {recon.study_type}")
        recon.exclusion_reasons = reasons
        return len(reasons) == 0
    
    def _build_schema(self) -> ExtractionSchema:
        schema = ExtractionSchema()
        
        for code in self.state.pico.primary_outcomes:
            info = self.ontology.get('outcomes', {}).get(code, {})
            schema.outcomes.append(OutcomeField(
                code=code, name=info.get('name', code), required=True,
                unit=info.get('unit'), range=info.get('range')
            ))
        
        for code in self.state.pico.secondary_outcomes:
            info = self.ontology.get('outcomes', {}).get(code, {})
            schema.outcomes.append(OutcomeField(
                code=code, name=info.get('name', code), required=False,
                unit=info.get('unit'), range=info.get('range')
            ))
        
        for code in self.state.pico.required_moderators:
            for cat, mods in self.ontology.get('moderators', {}).items():
                if code in mods:
                    info = mods[code]
                    schema.moderators.append(ModeratorField(
                        code=code, name=info.get('name', code),
                        field_type=info.get('type', 'categorical'),
                        values=info.get('values') if isinstance(info.get('values'), list) else None,
                        unit=info.get('unit')
                    ))
                    break
        
        schema.control_heuristic = {
            'rule': self.state.pico.control_heuristic,
            'definition': self.state.pico.control_definition,
            'keywords': self.state.pico.control_keywords
        }
        return schema
    
    def _format_schema(self) -> str:
        lines = ["OUTCOMES:"]
        for o in self.state.schema.outcomes:
            lines.append(f"  - {o.code}: {o.name} ({'REQUIRED' if o.required else 'optional'})")
        lines.append("\nMODERATORS:")
        for m in self.state.schema.moderators:
            lines.append(f"  - {m.code}: {m.name}")
        lines.append(f"\nCONTROL: {self.state.schema.control_heuristic.get('definition', 'N/A')}")
        return "\n".join(lines)
    
    def _record_decision(self, phase: Phase, question: str, options: List[str], choice: str):
        self.state.decisions.append(Decision(
            timestamp=datetime.now().isoformat(),
            phase=phase.value,
            question=question,
            options=options,
            user_choice=choice
        ))
    
    def _save_session(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.state.save(str(self.output_dir / "session_state.json"))
