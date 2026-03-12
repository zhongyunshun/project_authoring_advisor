"""Text-to-Speech using Deepgram Aura-2 API.

Usage:
    python -m voice_agent.tts input.txt -o output.wav --voice female
    python -m voice_agent.tts input.txt --voice male --model aura-2-orion-en
    python -m voice_agent.tts input.txt --list-voices
"""

from __future__ import annotations

import argparse
import sys
import logging
from pathlib import Path

from deepgram import DeepgramClient

from voice_agent.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Default voices per gender (Aura-2, English)
MALE_VOICES: list[str] = [
    "aura-2-apollo-en", "aura-2-arcas-en", "aura-2-aries-en",
    "aura-2-atlas-en", "aura-2-draco-en", "aura-2-hermes-en",
    "aura-2-hyperion-en", "aura-2-janus-en", "aura-2-jupiter-en",
    "aura-2-mars-en", "aura-2-neptune-en", "aura-2-odysseus-en",
    "aura-2-orion-en", "aura-2-orpheus-en", "aura-2-pluto-en",
    "aura-2-saturn-en", "aura-2-zeus-en",
]

FEMALE_VOICES: list[str] = [
    "aura-2-asteria-en", "aura-2-athena-en", "aura-2-aurora-en",
    "aura-2-callista-en", "aura-2-cordelia-en", "aura-2-cora-en",
    "aura-2-delia-en", "aura-2-electra-en", "aura-2-harmonia-en",
    "aura-2-helena-en", "aura-2-hera-en", "aura-2-iris-en",
    "aura-2-juno-en", "aura-2-luna-en", "aura-2-minerva-en",
    "aura-2-ophelia-en", "aura-2-pandora-en", "aura-2-phoebe-en",
    "aura-2-selene-en", "aura-2-thalia-en", "aura-2-theia-en",
    "aura-2-vesta-en",
]

DEFAULT_MALE: str = "aura-2-orion-en"
DEFAULT_FEMALE: str = "aura-2-asteria-en"


def list_voices() -> None:
    """Print all available Deepgram Aura-2 TTS voices to stdout.

    Voices are grouped by gender. The default voice for each gender is
    marked with ``(default)``.
    """
    print("Male voices:")
    for v in MALE_VOICES:
        default = " (default)" if v == DEFAULT_MALE else ""
        print(f"  {v}{default}")
    print("\nFemale voices:")
    for v in FEMALE_VOICES:
        default = " (default)" if v == DEFAULT_FEMALE else ""
        print(f"  {v}{default}")


def synthesize(text: str, output_path: Path, model: str, api_key: str) -> None:
    """Convert text to audio using Deepgram TTS and save to a file.

    The output audio format is determined by the file extension of
    ``output_path``:
        - ``.wav`` — linear16 PCM in a WAV container
        - ``.mp3`` — MP3 encoding
        - ``.ogg`` — Opus encoding in an OGG container

    Args:
        text: The text content to synthesize into speech.
        output_path: Destination file path for the generated audio.
        model: Deepgram Aura-2 voice model identifier
            (e.g. ``"aura-2-asteria-en"``).
        api_key: Deepgram API key for authentication.
    """
    client = DeepgramClient(api_key=api_key)

    # Determine output format from extension
    ext = output_path.suffix.lower()
    container = "wav"
    encoding = "linear16"
    if ext == ".mp3":
        container = "none"
        encoding = "mp3"
    elif ext == ".ogg":
        container = "ogg"
        encoding = "opus"

    logger.info("Generating audio with model: %s", model)
    logger.info("Output: %s (format: %s)", output_path, ext)

    audio_iter = client.speak.v1.audio.generate(
        text=text,
        model=model,
        encoding=encoding,
        container=container,
    )

    with open(output_path, "wb") as f:
        for chunk in audio_iter:
            f.write(chunk)

    logger.info("Audio saved to: %s", output_path)


def main() -> None:
    """Entry point for the TTS command-line tool.

    Parses command-line arguments, reads the input text file, selects a
    voice model based on ``--voice`` or ``--model``, and writes the
    synthesized audio to ``voice_agent/tts/outputs/``.

    Raises:
        SystemExit: If ``DEEPGRAM_API_KEY`` is not configured, the input
            file is missing or empty, or required arguments are not provided.
    """
    parser = argparse.ArgumentParser(description="Text-to-Speech using Deepgram")
    parser.add_argument("input", nargs="?", help="Input text file path")
    parser.add_argument("-o", "--output", help="Output audio file (default: <input>.wav)")
    parser.add_argument("--voice", choices=["male", "female"], default="female",
                        help="Voice gender (default: female)")
    parser.add_argument("--model", help="Specific Deepgram voice model (overrides --voice)")
    parser.add_argument("--list-voices", action="store_true", help="List available voices")
    args = parser.parse_args()

    if args.list_voices:
        list_voices()
        return

    if not args.input:
        parser.error("input file is required (use --list-voices to see available voices)")

    settings = get_settings()
    api_key = settings.stt.api_key
    if not api_key:
        logger.error("DEEPGRAM_API_KEY is not set. Export it or add it to voice_agent/.env")
        sys.exit(1)

    # Read input text
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        sys.exit(1)

    text = input_path.read_text(encoding="utf-8").strip()
    if not text:
        logger.error("Input file is empty")
        sys.exit(1)

    # Determine voice model
    if args.model:
        model = args.model
    elif args.voice == "male":
        model = DEFAULT_MALE
    else:
        model = DEFAULT_FEMALE

    # Determine output path — default to voice_agent/tts/outputs/
    output_dir = Path(__file__).parent / "tts" / "outputs"
    if args.output:
        output_path = output_dir / Path(args.output).name
    else:
        output_path = output_dir / input_path.with_suffix(".wav").name

    output_path.parent.mkdir(parents=True, exist_ok=True)

    synthesize(text, output_path, model, api_key)


if __name__ == "__main__":
    main()
