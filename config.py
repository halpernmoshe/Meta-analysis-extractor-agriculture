"""
Configuration for Meta-Analysis Extraction System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY", "")

# LLM Provider: "anthropic" or "google"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "anthropic")

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# Cost estimates per 100 papers (as of early 2025):
#   - Haiku 3.5:  ~$16   (fast, good for recon)
#   - Sonnet 4:   ~$60   (balanced, recommended for extraction)
#   - Opus 4:     ~$300  (best quality, use for difficult papers)
#
# Recommended strategy:
#   - RECON_MODEL = Haiku (cheap, just scanning)
#   - EXTRACT_MODEL = Sonnet (needs accuracy)
#   - Or use Sonnet for everything if budget allows
# =============================================================================

# Model choices - uncomment your preferred configuration:

# OPTION 1: Budget (~$35/100 papers) - Haiku for recon, Sonnet for extraction
RECON_MODEL = "claude-3-5-haiku-20241022"
EXTRACT_MODEL = "claude-sonnet-4-20250514"

# OPTION 2: Balanced (~$60/100 papers) - Sonnet for everything
# RECON_MODEL = "claude-sonnet-4-20250514"
# EXTRACT_MODEL = "claude-sonnet-4-20250514"

# OPTION 3: Quality (~$300/100 papers) - Opus for everything
# RECON_MODEL = "claude-opus-4-20250514"
# EXTRACT_MODEL = "claude-opus-4-20250514"

# OPTION 4: Maximum quality for extraction only (~$150/100 papers)
# RECON_MODEL = "claude-sonnet-4-20250514"
# EXTRACT_MODEL = "claude-opus-4-20250514"

# =============================================================================
# GEMINI MODEL CONFIGURATION (Google)
# =============================================================================
# Latest Gemini models (January 2026):
#   - gemini-3-flash-preview:   Best value, 3x faster than 2.5 Pro ($0.50/1M in)
#   - gemini-3-pro-preview:     Best reasoning, complex tasks ($2.00/1M in)
#   - gemini-2.5-flash:         Stable, great price-performance ($0.15/1M in)
#   - gemini-2.5-flash-lite:    Fastest, cheapest ($0.10/1M in)
#   - gemini-2.5-pro:           Stable reasoning model ($1.25/1M in)
#
# All Gemini models have 1M token context window (vs 200K for Claude)
# =============================================================================

# OPTION G1: Best value - Gemini 3 Flash (recommended)
GEMINI_RECON_MODEL = "gemini-2.5-flash-lite"
GEMINI_EXTRACT_MODEL = "gemini-3-flash-preview"

# OPTION G2: Maximum quality - Gemini 3 Pro
# GEMINI_RECON_MODEL = "gemini-2.5-flash"
# GEMINI_EXTRACT_MODEL = "gemini-3-pro-preview"

# OPTION G3: Cheapest - All Flash-Lite
# GEMINI_RECON_MODEL = "gemini-2.5-flash-lite"
# GEMINI_EXTRACT_MODEL = "gemini-2.5-flash-lite"

# OPTION G4: Stable only (no preview models)
# GEMINI_RECON_MODEL = "gemini-2.5-flash-lite"
# GEMINI_EXTRACT_MODEL = "gemini-2.5-flash"

# =============================================================================
# MOONSHOT KIMI MODEL CONFIGURATION
# =============================================================================
# Kimi K2 models (January 2026):
#   - kimi-k2-turbo-preview:   Fastest, 60-100 tok/s ($0.06/1M in, $0.60/1M out)
#   - kimi-k2-0905-preview:    256K context, stable ($0.06/1M in, $0.60/1M out)
#   - kimi-k2-thinking-turbo:  With reasoning/thinking ($0.06/1M in, $0.60/1M out)
#   - kimi-k2.5:               Multimodal/vision ($0.60/1M in, $3.00/1M out)
#
# All models have 256K token context window
# =============================================================================

KIMI_RECON_MODEL = "kimi-k2-turbo-preview"
KIMI_EXTRACT_MODEL = "kimi-k2-turbo-preview"
KIMI_VISION_MODEL = "kimi-k2.5"  # For PDF-as-images extraction

# Legacy names for compatibility
ORCHESTRATOR_MODEL = EXTRACT_MODEL
WORKER_MODEL = RECON_MODEL

# Temperature settings
RECON_TEMPERATURE = 0.0  # Deterministic for consistency
EXTRACT_TEMPERATURE = 0.0  # Deterministic for accuracy
ORCHESTRATOR_TEMPERATURE = 0.3  # Some flexibility for dialogue

# Token limits
MAX_TOKENS_RECON = 4096
MAX_TOKENS_EXTRACT = 32768  # Increased to prevent truncation

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
ONTOLOGY_PATH = DATA_DIR / "ontology.json"
FEW_SHOT_DIR = DATA_DIR / "few_shot_examples"

# Default directories
DEFAULT_INPUT_DIR = BASE_DIR / "input"
DEFAULT_OUTPUT_DIR = BASE_DIR / "output"

# Processing settings
SAMPLE_SIZE_ORIENTATION = 5  # Papers to scan in orientation phase
SAMPLE_SIZE_PILOT = 3  # Papers to extract in pilot phase
BATCH_SIZE = 10  # Papers to process in parallel (future)

# Retry settings
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds

# Validation thresholds
OUTLIER_SD_THRESHOLD = 3.0  # Flag values > 3 SD from mean
MIN_CONFIDENCE_FOR_INCLUSION = "medium"  # low, medium, high

# Export settings
EXPORT_FORMAT_JSON = True
EXPORT_FORMAT_CSV = True
GENERATE_METHODS_DOC = True

# =============================================================================
# COST ESTIMATION (per 1M tokens, as of early 2025)
# =============================================================================
MODEL_COSTS = {
    # Anthropic Claude models
    "claude-3-5-haiku-20241022": {
        "name": "Haiku 3.5",
        "input_per_1m": 0.80,
        "output_per_1m": 4.00,
        "provider": "anthropic",
    },
    "claude-sonnet-4-20250514": {
        "name": "Sonnet 4",
        "input_per_1m": 3.00,
        "output_per_1m": 15.00,
        "provider": "anthropic",
    },
    "claude-opus-4-20250514": {
        "name": "Opus 4",
        "input_per_1m": 15.00,
        "output_per_1m": 75.00,
        "provider": "anthropic",
    },
    "claude-opus-4-5-20251101": {
        "name": "Opus 4.5",
        "input_per_1m": 15.00,
        "output_per_1m": 75.00,
        "provider": "anthropic",
    },
    # Google Gemini models (January 2026 pricing)
    "gemini-3-flash-preview": {
        "name": "Gemini 3 Flash",
        "input_per_1m": 0.50,
        "output_per_1m": 3.00,
        "provider": "google",
    },
    "gemini-3-pro-preview": {
        "name": "Gemini 3 Pro",
        "input_per_1m": 2.00,
        "output_per_1m": 12.00,
        "provider": "google",
    },
    "gemini-2.5-flash": {
        "name": "Gemini 2.5 Flash",
        "input_per_1m": 0.15,
        "output_per_1m": 0.60,
        "provider": "google",
    },
    "gemini-2.5-flash-lite": {
        "name": "Gemini 2.5 Flash-Lite",
        "input_per_1m": 0.10,
        "output_per_1m": 0.40,
        "provider": "google",
    },
    "gemini-2.5-pro": {
        "name": "Gemini 2.5 Pro",
        "input_per_1m": 1.25,
        "output_per_1m": 5.00,
        "provider": "google",
    },
    # Moonshot Kimi models (January 2026 pricing)
    "kimi-k2-turbo-preview": {
        "name": "Kimi K2 Turbo",
        "input_per_1m": 0.06,
        "output_per_1m": 0.60,
        "provider": "moonshot",
    },
    "kimi-k2-0905-preview": {
        "name": "Kimi K2 0905",
        "input_per_1m": 0.06,
        "output_per_1m": 0.60,
        "provider": "moonshot",
    },
    "kimi-k2-thinking-turbo": {
        "name": "Kimi K2 Thinking",
        "input_per_1m": 0.06,
        "output_per_1m": 0.60,
        "provider": "moonshot",
    },
    "kimi-k2.5": {
        "name": "Kimi K2.5 Vision",
        "input_per_1m": 0.60,
        "output_per_1m": 3.00,
        "provider": "moonshot",
    },
}

# Average tokens per API call (estimated)
AVG_INPUT_TOKENS_RECON = 8000  # PDF text + prompt
AVG_OUTPUT_TOKENS_RECON = 800  # JSON response
AVG_INPUT_TOKENS_EXTRACT = 12000  # PDF text + schema + prompt
AVG_OUTPUT_TOKENS_EXTRACT = 1500  # Detailed JSON response

# API calls per paper (estimated)
CALLS_PER_PAPER_RECON = 7  # overview, outcomes, moderators, design, variance, control, tables
CALLS_PER_PAPER_EXTRACT = 2  # main extraction + possible table extraction
