# stt_google.py
import os
from google.cloud import speech
from dotenv import load_dotenv

load_dotenv()

def google_stt(audio_path):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    client = speech.SpeechClient()

    with open(audio_path, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="ko-KR"  
    )

    response = client.recognize(config=config, audio=audio)

    if not response.results:
        print("인식 결과 없음")
        return None

    transcript = response.results[0].alternatives[0].transcript
    print("Google STT 결과:", transcript)
    return transcript
