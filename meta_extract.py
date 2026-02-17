#!/usr/bin/env python3
"""
Meta-Analysis Data Extraction System

Usage:
    python meta_extract.py --input ./papers --output ./results
    python meta_extract.py -i ./papers -o ./results --provider google
"""
import argparse
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def ask_provider_choice():
    """Ask user to choose between available providers"""
    print("\n" + "="*60)
    print("SELECT LLM PROVIDER")
    print("="*60)
    print("\n1. Google Gemini (Recommended)")
    print("   - 1M token context window (5x larger than Claude)")
    print("   - Gemini 3 Flash: Fast, great value ($0.50/1M tokens)")
    print("   - Better at reading complex tables")
    print("\n2. Anthropic Claude")
    print("   - 200K token context window")
    print("   - Sonnet 4: Reliable extraction ($3/1M tokens)")
    print("="*60)

    while True:
        choice = input("\nEnter 1 for Gemini or 2 for Claude: ").strip()
        if choice == '1':
            return 'google'
        elif choice == '2':
            return 'anthropic'
        else:
            print("Please enter 1 or 2")


def main():
    parser = argparse.ArgumentParser(
        description="Meta-Analysis Data Extraction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Interactive mode (asks for user input at each step):
    python meta_extract.py --input ./papers --output ./results

    # Specify provider directly:
    python meta_extract.py --input ./papers --output ./results --provider google

    # AUTONOMOUS MODE - runs without user interaction:
    python meta_extract.py --input ./papers --output ./results --provider google --auto

    # Autonomous mode with custom config:
    python meta_extract.py --input ./papers --output ./results --auto --config my_config.json

The system will:
1. Scan your PDF papers to understand the corpus
2. Guide you through PICO specification via popup dialogues (or use defaults in --auto mode)
3. Extract quantitative data for meta-analysis
4. Output JSON files per paper and summary CSV

Autonomous Mode (--auto):
- Uses predefined defaults for CO2/mineral studies (or custom config via --config)
- Logs all automatic decisions to output/auto_decisions.log
- Resumes from previous session if exists
- Continues on errors instead of stopping

Supported providers:
- google: Gemini models (Flash-Lite, Flash, Pro) - 1M token context window!
- anthropic: Claude models (Haiku, Sonnet, Opus)
        """
    )

    parser.add_argument('-i', '--input', type=str, required=True,
                        help='Input directory containing PDF papers')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output directory for results')
    parser.add_argument('--api-key', type=str, default=None,
                        help='API key (overrides .env file)')
    parser.add_argument('--provider', type=str, default=None,
                        choices=['anthropic', 'google'],
                        help='LLM provider: anthropic (Claude) or google (Gemini)')
    parser.add_argument('--auto', action='store_true',
                        help='Run in autonomous mode without user interaction')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to auto-mode config JSON (optional, uses defaults if not provided)')
    parser.add_argument('--domain', type=str, default=None,
                        help='Meta-analysis domain (e.g., silicon_wheat, elevated_co2). '
                             'Provides domain-specific extraction instructions.')

    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    if not input_dir.is_dir():
        print(f"Error: Input path is not a directory: {input_dir}")
        sys.exit(1)

    pdfs = list(input_dir.glob("*.pdf"))
    if not pdfs:
        print(f"Error: No PDF files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(pdfs)} PDF files in {input_dir}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check available API keys
    anthropic_key = args.api_key if args.provider == 'anthropic' else os.getenv('ANTHROPIC_API_KEY')
    google_key = args.api_key if args.provider == 'google' else os.getenv('GOOGLE_API_KEY')

    # Determine provider
    provider = args.provider

    if provider is None:
        # Check what's configured in .env
        env_provider = os.getenv('LLM_PROVIDER', '').strip()

        if env_provider in ['anthropic', 'google']:
            provider = env_provider
        elif google_key and anthropic_key:
            # Both keys available - ask user
            provider = ask_provider_choice()
        elif google_key:
            provider = 'google'
        elif anthropic_key:
            provider = 'anthropic'
        else:
            print("Error: No API keys found.")
            print("Set ANTHROPIC_API_KEY or GOOGLE_API_KEY in .env file")
            print("Or use --api-key with --provider")
            sys.exit(1)

    # Get the appropriate API key
    if provider == 'google':
        api_key = args.api_key or google_key
        if not api_key:
            print("Error: No Google API key found.")
            print("Set GOOGLE_API_KEY in .env or use --api-key")
            sys.exit(1)
    else:
        api_key = args.api_key or anthropic_key
        if not api_key:
            print("Error: No Anthropic API key found.")
            print("Set ANTHROPIC_API_KEY in .env or use --api-key")
            sys.exit(1)

    # Display provider info
    print("\n" + "-"*60)
    if provider == 'google':
        print("Using: Google Gemini")
        print("  Recon model:   gemini-2.5-flash-lite (fastest)")
        print("  Extract model: gemini-3-flash-preview (best value)")
        print("  Context:       1,000,000 tokens")
    else:
        print("Using: Anthropic Claude")
        print("  Recon model:   claude-3-5-haiku")
        print("  Extract model: claude-sonnet-4")
        print("  Context:       200,000 tokens")
    print("-"*60 + "\n")

    try:
        from core.orchestrator import Orchestrator
    except ImportError as e:
        print(f"Import error: {e}")
        print("Run from meta_analysis_extractor directory")
        print("Install requirements: pip install -r requirements.txt")
        if provider == 'google':
            print("For Gemini: pip install google-generativeai")
        sys.exit(1)

    orchestrator = Orchestrator(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        api_key=api_key,
        provider=provider,
        auto_mode=args.auto,
        auto_config_path=args.config,
        domain=args.domain
    )
    orchestrator.run()


if __name__ == "__main__":
    main()
