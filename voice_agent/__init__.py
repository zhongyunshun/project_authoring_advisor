"""Voice agent — real-time STT and TTS using Deepgram.

Speech-to-Text (live microphone transcription):
    python -m voice_agent.app

-------------------------------------------------------------------------------------------------------------

Text-to-Speech (supports .wav, .mp3, and .ogg format):

    It uses Deepgram Aura-2 models (their latest generation with improved naturalness, support English only):
    - Female default: aura-2-asteria-en
    - Male default: aura-2-orion-en

    # List all available voices
    python -m voice_agent.tts --list-voices

    # Female voice (default)
    python -m voice_agent.tts input.txt -o output.wav

    # Male voice
    python -m voice_agent.tts input.txt -o output.wav --voice male

    # Pick a specific voice
    python -m voice_agent.tts input.txt -o output.wav --model aura-2-zeus-en

    # MP3 output (based on file extension)
    python -m voice_agent.tts input.txt -o output.mp3 --voice female
"""
