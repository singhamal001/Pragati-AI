import sounddevice as sd
import numpy as np
import speech_recognition as sr

def speak(text, piper_voice):
    """
    Converts text to speech using the loaded Piper TTS model.
    Takes the loaded piper_voice object as an argument.
    """
    print(f"\n< Gemma: {text}")
    if not piper_voice or not text or not text.strip():
        return

    try:
        print("Synthesizing audio...")
        audio_generator = piper_voice.synthesize(text)
        audio_arrays = [chunk.audio_int16_array for chunk in audio_generator]
        audio_data = np.concatenate(audio_arrays)
        sd.play(audio_data, samplerate=piper_voice.config.sample_rate)
        sd.wait()
    except Exception as e:
        print(f"An error occurred during TTS playback: {e}")

def listen(recognizer, microphone, transcriber):
    """
    Listens for a user's response, transcribes it, and returns the text.
    Takes initialized recognizer, microphone, and transcriber objects as arguments.
    """
    with microphone as source:
        print("Listening for your answer...")
        try:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.listen(source)
            print("Processing your answer...")
            wav_bytes = audio_data.get_wav_data()
            user_answer = transcriber(wav_bytes)["text"].strip()
            
            if user_answer:
                print(f"> You said: {user_answer}")
                return user_answer
            else:
                speak("I'm sorry, I didn't catch that. Could you please repeat your answer?", None) # Pass None for piper_voice to avoid error
                return None

        except sr.UnknownValueError:
            speak("I couldn't understand that. Let's try again.", None)
            return None