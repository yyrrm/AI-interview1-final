import pyaudio
import wave
import numpy as np
import webrtcvad
import collections

def record_until_silence(output_path="temp.wav", rate=16000, silence_limit=1.0):
    vad = webrtcvad.Vad(2)  # 0~3, 숫자 클수록 민감
    p = pyaudio.PyAudio()

    CHUNK = int(rate * 0.02)   # 20ms 단위
    FORMAT = pyaudio.paInt16
    CHANNELS = 1

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=rate,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("🎤 말하면 녹음 시작됨... (침묵 1초면 자동 종료)")

    frames = []
    silence_chunks = int(silence_limit / 0.02)
    silence_counter = 0

    while True:
        data = stream.read(CHUNK)
        frames.append(data)

        pcm = np.frombuffer(data, dtype=np.int16)
        is_speech = vad.is_speech(data, rate)

        if not is_speech:
            silence_counter += 1
        else:
            silence_counter = 0

        if silence_counter > silence_chunks:
            print("🛑 말이 멈춰서 녹음 종료됩니다.")
            break

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(output_path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(rate)
    wf.writeframes(b"".join(frames))
    wf.close()

    return output_path
