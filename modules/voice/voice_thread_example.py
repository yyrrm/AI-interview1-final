# voice_thread_example.py — DEFAULT MIC VERSION (최종)

import threading
import queue
import time
import numpy as np
import traceback
import os

from modules.voice.voice_module import (
    record_until_silence,
    preprocess_audio,
)
from modules.voice.stt_google import google_stt
from modules.shared_flags import RUNNING

voice_result_queue = queue.Queue(maxsize=5)


# ======================================
# 🎤 Voice Thread Worker
# ======================================
def voice_worker(rate=16000):
    print("🎧 Voice Thread Started")
    print("🎤 기본 마이크(Default Input Device) 사용")

    while RUNNING:
        try:
            print("\n🎤 말하면 녹음 시작...")

            audio_path = record_until_silence(
                output_path="temp.wav",
                rate=rate,
                silence_limit=1.2
            )

            if audio_path is None:
                print("❌ 녹음 실패 — 다음 반복")
                continue

            print(f"🎙 녹음 완료 → {audio_path}")
            print("📁 파일 크기:", os.path.getsize(audio_path), "bytes")

            # 전처리
            preprocess_audio(audio_path, rate)
            print("🔧 전처리 완료")

            # STT
            print("⏳ STT 처리 중...")
            text = google_stt(audio_path) or "(음성 없음)"

            print(f"\n[🎤 Voice Recognized]\n>> {text}")

            # 결과 패킹
            result = {
                "text": text,
                "timestamp": time.time()
            }

            if voice_result_queue.full():
                voice_result_queue.get_nowait()

            voice_result_queue.put(result)

        except Exception as e:
            print("❌ Voice Thread Error:", e)
            traceback.print_exc()
            time.sleep(0.5)

    print("🎧 Voice Thread Stopped")


def start_voice_thread():
    t = threading.Thread(target=voice_worker, daemon=True)
    t.start()
    print("🚀 voice_thread_example 실행됨!")
    return t
