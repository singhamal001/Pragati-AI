from piper.voice import PiperVoice

PIPER_MODEL_PATH = './model/en_US-hfc_female-medium.onnx'

print("Loading model...")
try:
    voice = PiperVoice.load(PIPER_MODEL_PATH)
    print("Synthesizing a test sentence...")
    # Get the generator
    audio_generator = voice.synthesize("test")
    
    # Get the VERY FIRST chunk from the generator
    first_chunk = next(audio_generator)
    
    print("\n--- OBJECT INSPECTION ---")
    print(f"The object type is: {type(first_chunk)}")
    print("\nAttributes and methods available in this object:")
    # Use dir() to list everything inside the object
    print(dir(first_chunk))
    print("-------------------------\n")

except Exception as e:
    print(f"An error occurred: {e}")