# generate_audio.py

import argparse
import wave
import os
from piper.voice import PiperVoice
import numpy as np # Ensure numpy is imported

# --- Configuration ---
PIPER_MODEL_PATH = "./model/en_US-hfc_female-medium.onnx"
OUTPUT_DIR = "./assets"

def generate_audio(text_to_speak: str, output_filename: str):
    """
    Generates a WAV file from text using Piper TTS.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not output_filename.lower().endswith('.wav'):
        output_filename += '.wav'
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    print("Loading Piper voice model...")
    try:
        voice = PiperVoice.load(PIPER_MODEL_PATH)
    except Exception as e:
        print(f"Error: Could not load the Piper model from '{PIPER_MODEL_PATH}'")
        print(f"Details: {e}")
        return

    print(f"Synthesizing audio for: '{text_to_speak}'")
    
    with wave.open(output_path, 'wb') as wav_file:
        # Set the parameters for the WAV file
        wav_file.setnchannels(1)  # Mono audio
        wav_file.setsampwidth(2)  # 16-bit audio
        wav_file.setframerate(voice.config.sample_rate)
        
        # Synthesize the audio and write the raw bytes to the file
        for chunk in voice.synthesize(text_to_speak):
            # --- THIS IS THE CRITICAL FIX ---
            # Convert the numpy array of audio samples to raw bytes
            wav_file.writeframes(chunk.audio_int16_array.tobytes())

    print("-" * 50)
    print(f"Successfully generated audio file at: {output_path}")
    print("-" * 50)

if __name__ == "__main__":
    text = "Which report would you like to discuss? You can say, for example, 'the first one', or 'the most recent one' or 'the second one'."
    filename = "feedback_prompt_for_selection"
    generate_audio(text, filename)