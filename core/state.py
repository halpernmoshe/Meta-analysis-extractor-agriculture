"""
State management for Meta-Analysis Extraction System

Contains all data classes for session state, PICO specification,
paper reconnaissance, observations, and extraction schema.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union
from enum import Enum
from datetime import datetime
import json


class Phase(Enum):
    """Phases of the extraction workflow"""
    INIT = "init"
    ORIENTATION = "orientation"
    PICO_POPULATION = "pico_population"
    PICO_INTERVENTION = "pico_intervention"
    PICO_COMPARISON = "pico_comparison"
    PICO_OUTCOMES = "pico_outcomes"
    MODERATORS = "moderators"
    FULL_RECON = "full_recon"
    GAP_ANALYSIS = "gap_analysis"
    SCHEMA_REVIEW = "schema_review"
    PILOT_EXTRACTION = "pilot_extraction"
    PILOT_REVIEW = "pilot_review"
    FULL_EXTRACTION = "full_extraction"
    GAP_FILL = "gap_fill"
    VARIANCE_EXTRACTION = "variance_extraction"
    VARIANCE_RESCUE = "variance_rescue"
    VARIANCE_VERIFICATION = "variance_verification"
    VALIDATION = "validation"
    EXPORT = "export"
    COMPLETE = "complete"


@dataclass
class PICOSpec:
    """
    Agricultural PICO specification
    
    P - Population: crops, environments, study types
    I - Intervention: treatment being studied
    C - Comparison: control/baseline definition
    O - Outcomes: response variables to extract
    """
    # Population
    crop_species: List[str] = field(default_factory=list)
    crop_groups: List[str] = field(default_factory=list)
    environments: List[str] = field(default_factory=list)
    study_types: List[str] = field(default_factory=list)
    geographic_scope: Optional[str] = None
    
    # Intervention
    intervention_domain: Optional[str] = None
    intervention_variable: Optional[str] = None
    intervention_description: Optional[str] = None
    
    # Comparison
    control_definition: Optional[str] = None
    control_heuristic: Optional[str] = None
    control_keywords: List[str] = field(default_factory=list)
    comparison_types: List[str] = field(default_factory=list)
    
    # Outcomes
    primary_outcomes: List[str] = field(default_factory=list)
    secondary_outcomes: List[str] = field(default_factory=list)
    outcome_measurement_types: Dict[str, str] = field(default_factory=dict)
    
    # Moderators (selected for extraction)
    required_moderators: List[str] = field(default_factory=list)
    optional_moderators: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            'population': {
                'crop_species': self.crop_species,
                'crop_groups': self.crop_groups,
                'environments': self.environments,
                'study_types': self.study_types,
                'geographic_scope': self.geographic_scope
            },
            'intervention': {
                'domain': self.intervention_domain,
                'variable': self.intervention_variable,
                'description': self.intervention_description
            },
            'comparison': {
                'definition': self.control_definition,
                'heuristic': self.control_heuristic,
                'keywords': self.control_keywords,
                'types': self.comparison_types
            },
            'outcomes': {
                'primary': self.primary_outcomes,
                'secondary': self.secondary_outcomes,
                'measurement_types': self.outcome_measurement_types
            },
            'moderators': {
                'required': self.required_moderators,
                'optional': self.optional_moderators
            }
        }


@dataclass
class OutcomeInfo:
    """Information about an outcome variable found in a paper"""
    icasa_code: str
    present: bool = False
    location: Optional[str] = None  # "Table 2", "Figure 3", etc.
    unit: Optional[str] = None
    has_means: bool = False
    has_variance: bool = False
    variance_type: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class ModeratorInfo:
    """Information about a moderator variable found in a paper"""
    code: str
    available: bool = False
    value: Optional[Any] = None
    values: List[Any] = field(default_factory=list)
    location: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class PaperRecon:
    """Reconnaissance results for a single paper"""
    paper_id: str
    file_path: str
    citation: Optional[str] = None
    title: Optional[str] = None
    authors: Optional[str] = None
    year: Optional[int] = None
    
    # Content assessment
    outcomes_present: Dict[str, OutcomeInfo] = field(default_factory=dict)
    moderators_available: Dict[str, ModeratorInfo] = field(default_factory=dict)
    
    # Design info
    design_type: Optional[str] = None
    main_plot_factor: Optional[str] = None
    subplot_factor: Optional[str] = None
    replicates: Optional[int] = None
    plot_size: Optional[str] = None
    
    # Variance info
    variance_type: Optional[str] = None
    variance_location: Optional[str] = None
    variance_notes: Optional[str] = None
    
    # Control info
    control_definition: Optional[str] = None
    control_identified: bool = False
    treatment_levels: List[str] = field(default_factory=list)
    
    # Study context
    study_type: Optional[str] = None
    country: Optional[str] = None
    region: Optional[str] = None
    coordinates: Optional[Dict[str, float]] = None
    
    # Quality flags
    meets_pico_criteria: bool = True
    exclusion_reasons: List[str] = field(default_factory=list)
    quality_notes: Optional[str] = None
    
    # Data locations
    data_tables: List[str] = field(default_factory=list)
    data_figures: List[str] = field(default_factory=list)
    supplementary_data: bool = False

    # Cultivar/variety info
    cultivar_data_structure: Optional[str] = None  # "individual", "averaged", or "both"
    individual_cultivar_tables: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'paper_id': self.paper_id,
            'file_path': self.file_path,
            'citation': self.citation,
            'title': self.title,
            'authors': self.authors,
            'year': self.year,
            'outcomes_present': {k: v.__dict__ if hasattr(v, '__dict__') else v 
                                for k, v in self.outcomes_present.items()},
            'moderators_available': {k: v.__dict__ if hasattr(v, '__dict__') else v 
                                    for k, v in self.moderators_available.items()},
            'design_type': self.design_type,
            'main_plot_factor': self.main_plot_factor,
            'subplot_factor': self.subplot_factor,
            'replicates': self.replicates,
            'variance_type': self.variance_type,
            'variance_location': self.variance_location,
            'control_definition': self.control_definition,
            'control_identified': self.control_identified,
            'treatment_levels': self.treatment_levels,
            'study_type': self.study_type,
            'country': self.country,
            'region': self.region,
            'coordinates': self.coordinates,
            'meets_pico_criteria': self.meets_pico_criteria,
            'exclusion_reasons': self.exclusion_reasons,
            'data_tables': self.data_tables,
            'data_figures': self.data_figures
        }


@dataclass
class Observation:
    """
    A single extracted observation (one treatment-control comparison for one outcome)
    """
    observation_id: str
    paper_id: str
    
    # Treatment/Control identification
    treatment_description: str = ""
    control_description: str = ""
    is_control: bool = False
    comparison_type: Optional[str] = None
    
    # Outcome identification
    outcome_variable: str = ""  # ICASA code
    outcome_name: str = ""  # Human readable
    plant_part: Optional[str] = None
    measurement_type: Optional[str] = None  # concentration, content, proportion
    
    # Treatment values
    treatment_mean: Optional[float] = None
    treatment_n: Optional[int] = None
    
    # Control values
    control_mean: Optional[float] = None
    control_n: Optional[int] = None
    
    # Variance (as reported - no conversion)
    variance_type: Optional[str] = None  # SD, SE, LSD, CV, etc.
    treatment_variance: Optional[float] = None
    control_variance: Optional[float] = None
    pooled_variance: Optional[float] = None  # For LSD, CV, MSE
    degrees_of_freedom: Optional[int] = None
    alpha_level: Optional[float] = 0.05  # For LSD
    
    # Units
    unit_reported: Optional[str] = None
    unit_standardized: Optional[str] = None
    conversion_factor: Optional[float] = None
    
    # Moderators (flexible dict for any moderators)
    moderators: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    data_source: Optional[str] = None  # "Table 2", "Figure 3"
    data_source_type: Optional[str] = None  # table, figure, text
    study_year: Optional[str] = None
    site_id: Optional[str] = None
    
    # Quality flags
    extraction_confidence: str = "high"  # high, medium, low
    validation_status: Optional[str] = None  # verified, flagged, error
    validation_notes: Optional[str] = None
    is_summary_row: bool = False  # Flag if this is an aggregate, not individual observation
    notes: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            'observation_id': self.observation_id,
            'paper_id': self.paper_id,
            'treatment_description': self.treatment_description,
            'control_description': self.control_description,
            'is_control': self.is_control,
            'comparison_type': self.comparison_type,
            'outcome_variable': self.outcome_variable,
            'outcome_name': self.outcome_name,
            'plant_part': self.plant_part,
            'measurement_type': self.measurement_type,
            'treatment_mean': self.treatment_mean,
            'treatment_n': self.treatment_n,
            'control_mean': self.control_mean,
            'control_n': self.control_n,
            'variance_type': self.variance_type,
            'treatment_variance': self.treatment_variance,
            'control_variance': self.control_variance,
            'pooled_variance': self.pooled_variance,
            'degrees_of_freedom': self.degrees_of_freedom,
            'alpha_level': self.alpha_level,
            'unit_reported': self.unit_reported,
            'unit_standardized': self.unit_standardized,
            'conversion_factor': self.conversion_factor,
            'moderators': self.moderators,
            'data_source': self.data_source,
            'data_source_type': self.data_source_type,
            'study_year': self.study_year,
            'site_id': self.site_id,
            'extraction_confidence': self.extraction_confidence,
            'validation_status': self.validation_status,
            'validation_notes': self.validation_notes,
            'is_summary_row': self.is_summary_row,
            'notes': self.notes
        }


@dataclass
class OutcomeField:
    """Definition of an outcome field in the extraction schema"""
    code: str
    name: str
    required: bool = True
    unit: Optional[str] = None
    range: Optional[List[float]] = None
    notes: Optional[str] = None


@dataclass
class ModeratorField:
    """Definition of a moderator field in the extraction schema"""
    code: str
    name: str
    field_type: str = "categorical"  # categorical, continuous, free_text
    values: Optional[List[str]] = None
    unit: Optional[str] = None
    range: Optional[List[float]] = None
    notes: Optional[str] = None


@dataclass
class ExtractionSchema:
    """
    Defines what to extract from papers
    Built from PICO spec and recon results
    """
    outcomes: List[OutcomeField] = field(default_factory=list)
    moderators: List[ModeratorField] = field(default_factory=list)
    control_heuristic: Dict[str, Any] = field(default_factory=dict)
    design_handling: Dict[str, str] = field(default_factory=dict)
    variance_extraction: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            'outcomes': [{'code': o.code, 'name': o.name, 'required': o.required,
                         'unit': o.unit, 'range': o.range} for o in self.outcomes],
            'moderators': [{'code': m.code, 'name': m.name, 'type': m.field_type,
                           'values': m.values, 'unit': m.unit} for m in self.moderators],
            'control_heuristic': self.control_heuristic,
            'design_handling': self.design_handling,
            'variance_extraction': self.variance_extraction
        }


@dataclass
class Decision:
    """Audit trail of user decisions"""
    timestamp: str
    phase: str
    question: str
    options: List[str]
    user_choice: str
    rationale: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'phase': self.phase,
            'question': self.question,
            'options': self.options,
            'user_choice': self.user_choice,
            'rationale': self.rationale
        }


@dataclass
class SessionState:
    """
    Complete session state - serializable for reproducibility

    This captures everything about the meta-analysis extraction session,
    allowing it to be saved, resumed, and fully documented.
    """
    session_id: str
    created_at: str
    last_updated: str

    # Current phase
    current_phase: Phase = Phase.INIT

    # Configuration
    input_directory: str = ""
    output_directory: str = ""
    paper_count: int = 0

    # Model selection (for resume)
    recon_model: Optional[str] = None
    extract_model: Optional[str] = None
    
    # PICO specification (built through dialogue)
    pico: PICOSpec = field(default_factory=PICOSpec)
    
    # Reconnaissance cache (what we learned about each paper)
    recon_cache: Dict[str, PaperRecon] = field(default_factory=dict)
    
    # Extraction schema (derived from PICO + recon)
    schema: Optional[ExtractionSchema] = None
    
    # Results
    extractions: Dict[str, List[Observation]] = field(default_factory=dict)
    
    # Audit trail
    decisions: List[Decision] = field(default_factory=list)
    
    # Summary statistics
    papers_meeting_criteria: int = 0
    total_observations: int = 0
    flagged_observations: int = 0
    
    def to_dict(self) -> dict:
        """Convert entire state to dictionary for JSON serialization"""
        return {
            'session_id': self.session_id,
            'created_at': self.created_at,
            'last_updated': self.last_updated,
            'current_phase': self.current_phase.value,
            'input_directory': self.input_directory,
            'output_directory': self.output_directory,
            'paper_count': self.paper_count,
            'recon_model': self.recon_model,
            'extract_model': self.extract_model,
            'pico': self.pico.to_dict(),
            'recon_cache': {k: v.to_dict() for k, v in self.recon_cache.items()},
            'schema': self.schema.to_dict() if self.schema else None,
            'extractions': {
                paper_id: [obs.to_dict() for obs in obs_list]
                for paper_id, obs_list in self.extractions.items()
            },
            'decisions': [d.to_dict() for d in self.decisions],
            'papers_meeting_criteria': self.papers_meeting_criteria,
            'total_observations': self.total_observations,
            'flagged_observations': self.flagged_observations
        }
    
    def save(self, filepath: str):
        """Save state to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    @classmethod
    def load(cls, filepath: str) -> 'SessionState':
        """Load state from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Reconstruct state
        state = cls(
            session_id=data['session_id'],
            created_at=data['created_at'],
            last_updated=data['last_updated']
        )
        state.current_phase = Phase(data['current_phase'])
        state.input_directory = data['input_directory']
        state.output_directory = data['output_directory']
        state.paper_count = data['paper_count']
        state.recon_model = data.get('recon_model')
        state.extract_model = data.get('extract_model')

        # Reconstruct PICO spec
        pico_data = data.get('pico', {})
        state.pico = PICOSpec(
            crop_species=pico_data.get('population', {}).get('crop_species', []),
            crop_groups=pico_data.get('population', {}).get('crop_groups', []),
            environments=pico_data.get('population', {}).get('environments', []),
            study_types=pico_data.get('population', {}).get('study_types', []),
            geographic_scope=pico_data.get('population', {}).get('geographic_scope'),
            intervention_domain=pico_data.get('intervention', {}).get('domain'),
            intervention_variable=pico_data.get('intervention', {}).get('variable'),
            intervention_description=pico_data.get('intervention', {}).get('description'),
            control_definition=pico_data.get('comparison', {}).get('definition'),
            control_heuristic=pico_data.get('comparison', {}).get('heuristic'),
            control_keywords=pico_data.get('comparison', {}).get('keywords', []),
            comparison_types=pico_data.get('comparison', {}).get('types', []),
            primary_outcomes=pico_data.get('outcomes', {}).get('primary', []),
            secondary_outcomes=pico_data.get('outcomes', {}).get('secondary', []),
            outcome_measurement_types=pico_data.get('outcomes', {}).get('measurement_types', {}),
            required_moderators=pico_data.get('moderators', {}).get('required', []),
            optional_moderators=pico_data.get('moderators', {}).get('optional', [])
        )

        # Reconstruct recon cache
        for paper_id, recon_data in data.get('recon_cache', {}).items():
            recon = PaperRecon(
                paper_id=recon_data.get('paper_id', paper_id),
                file_path=recon_data.get('file_path', ''),
                citation=recon_data.get('citation'),
                title=recon_data.get('title'),
                authors=recon_data.get('authors'),
                year=recon_data.get('year'),
                design_type=recon_data.get('design_type'),
                main_plot_factor=recon_data.get('main_plot_factor'),
                subplot_factor=recon_data.get('subplot_factor'),
                replicates=recon_data.get('replicates'),
                variance_type=recon_data.get('variance_type'),
                variance_location=recon_data.get('variance_location'),
                control_definition=recon_data.get('control_definition'),
                control_identified=recon_data.get('control_identified', False),
                treatment_levels=recon_data.get('treatment_levels', []),
                study_type=recon_data.get('study_type'),
                country=recon_data.get('country'),
                region=recon_data.get('region'),
                coordinates=recon_data.get('coordinates'),
                meets_pico_criteria=recon_data.get('meets_pico_criteria', True),
                exclusion_reasons=recon_data.get('exclusion_reasons', []),
                data_tables=recon_data.get('data_tables', []),
                data_figures=recon_data.get('data_figures', [])
            )
            # Reconstruct outcomes_present
            for code, outcome_data in recon_data.get('outcomes_present', {}).items():
                if isinstance(outcome_data, dict):
                    recon.outcomes_present[code] = OutcomeInfo(
                        icasa_code=outcome_data.get('icasa_code', code),
                        present=outcome_data.get('present', False),
                        location=outcome_data.get('location'),
                        unit=outcome_data.get('unit'),
                        has_means=outcome_data.get('has_means', False),
                        has_variance=outcome_data.get('has_variance', False),
                        variance_type=outcome_data.get('variance_type'),
                        notes=outcome_data.get('notes')
                    )
            # Reconstruct moderators_available
            for code, mod_data in recon_data.get('moderators_available', {}).items():
                if isinstance(mod_data, dict):
                    recon.moderators_available[code] = ModeratorInfo(
                        code=mod_data.get('code', code),
                        available=mod_data.get('available', False),
                        value=mod_data.get('value'),
                        values=mod_data.get('values', []),
                        location=mod_data.get('location'),
                        notes=mod_data.get('notes')
                    )
            state.recon_cache[paper_id] = recon

        # Reconstruct schema if present
        schema_data = data.get('schema')
        if schema_data:
            from core.state import ExtractionSchema, OutcomeField, ModeratorField
            state.schema = ExtractionSchema(
                control_heuristic=schema_data.get('control_heuristic', {}),
                design_handling=schema_data.get('design_handling', {}),
                variance_extraction=schema_data.get('variance_extraction', {})
            )
            for o in schema_data.get('outcomes', []):
                state.schema.outcomes.append(OutcomeField(
                    code=o['code'], name=o['name'], required=o.get('required', True),
                    unit=o.get('unit'), range=o.get('range')
                ))
            for m in schema_data.get('moderators', []):
                state.schema.moderators.append(ModeratorField(
                    code=m['code'], name=m['name'], field_type=m.get('type', 'categorical'),
                    values=m.get('values'), unit=m.get('unit')
                ))

        # Reconstruct extractions
        for paper_id, obs_list in data.get('extractions', {}).items():
            state.extractions[paper_id] = []
            for obs_data in obs_list:
                obs = Observation(
                    observation_id=obs_data.get('observation_id', ''),
                    paper_id=obs_data.get('paper_id', paper_id),
                    treatment_description=obs_data.get('treatment_description', ''),
                    control_description=obs_data.get('control_description', ''),
                    is_control=obs_data.get('is_control', False),
                    comparison_type=obs_data.get('comparison_type'),
                    outcome_variable=obs_data.get('outcome_variable', ''),
                    outcome_name=obs_data.get('outcome_name', ''),
                    plant_part=obs_data.get('plant_part'),
                    measurement_type=obs_data.get('measurement_type'),
                    treatment_mean=obs_data.get('treatment_mean'),
                    treatment_n=obs_data.get('treatment_n'),
                    control_mean=obs_data.get('control_mean'),
                    control_n=obs_data.get('control_n'),
                    variance_type=obs_data.get('variance_type'),
                    treatment_variance=obs_data.get('treatment_variance'),
                    control_variance=obs_data.get('control_variance'),
                    pooled_variance=obs_data.get('pooled_variance'),
                    degrees_of_freedom=obs_data.get('degrees_of_freedom'),
                    alpha_level=obs_data.get('alpha_level', 0.05),
                    unit_reported=obs_data.get('unit_reported'),
                    unit_standardized=obs_data.get('unit_standardized'),
                    conversion_factor=obs_data.get('conversion_factor'),
                    moderators=obs_data.get('moderators', {}),
                    data_source=obs_data.get('data_source'),
                    data_source_type=obs_data.get('data_source_type'),
                    study_year=obs_data.get('study_year'),
                    site_id=obs_data.get('site_id'),
                    extraction_confidence=obs_data.get('extraction_confidence', 'high'),
                    validation_status=obs_data.get('validation_status'),
                    validation_notes=obs_data.get('validation_notes'),
                    is_summary_row=obs_data.get('is_summary_row', False),
                    notes=obs_data.get('notes')
                )
                state.extractions[paper_id].append(obs)

        # Reconstruct decisions
        for dec_data in data.get('decisions', []):
            state.decisions.append(Decision(
                timestamp=dec_data.get('timestamp', ''),
                phase=dec_data.get('phase', ''),
                question=dec_data.get('question', ''),
                options=dec_data.get('options', []),
                user_choice=dec_data.get('user_choice', ''),
                rationale=dec_data.get('rationale')
            ))

        # Summary stats
        state.papers_meeting_criteria = data.get('papers_meeting_criteria', 0)
        state.total_observations = data.get('total_observations', 0)
        state.flagged_observations = data.get('flagged_observations', 0)

        return state
    
    def update_timestamp(self):
        """Update the last_updated timestamp"""
        self.last_updated = datetime.now().isoformat()
