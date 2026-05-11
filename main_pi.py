"""
pi_client_optimized.py
======================
الـ Pi بيعمل بس:
  1. PPN wake word (Porcupine) — بدون TFLite / MFCC
  2. Silero VAD streaming — يبعت الـ PCM فوراً لما الكلام يخلص
  3. HTTP streaming POST للسيرفر
  4. يشغّل الـ TTS اللي رجع

pip install pvporcupine pyaudio numpy requests sounddevice
"""

import struct
import numpy as np
import pyaudio
import pvporcupine
import torch
import requests
import sounddevice as sd
import threading
import queue

# ─── Config ───────────────────────────────────────────────────
SERVER_URL      = "https://7501-01khejztq1640gknejy92er4gq.cloudspaces.litng.ai/"
USER_ID         = "pi-living-room-001"
PPN_ACCESS_KEY  = "o03bvBbHZkULR6O6MOvvDWQ1UUMKrvv6ErIUp8YZ2KJov90RA1CYaA=="
PPN_MODEL_PATH  = "wa-nees_en_raspberry-pi_v4_0_0.ppn"   # ملفك الـ .ppn
RATE            = 16_000
CHANNELS        = 1
# Porcupine يشتغل على frame_length ثابت
# ─────────────────────────────────────────────────────────────


# ─── Load models once at startup ──────────────────────────────
print("⏳ Loading Porcupine…")
porcupine = pvporcupine.create(
    access_key=PPN_ACCESS_KEY,
    keyword_paths=[PPN_MODEL_PATH],
)
FRAME_LEN = porcupine.frame_length   # عادةً 512

print("⏳ Loading Silero VAD…")
vad_model, utils = torch.hub.load(
    "snakers4/silero-vad", "silero_vad", force_reload=False
)
(get_speech_timestamps, _, _, VADIterator, _) = utils
vad_model.eval()

# VADIterator أسرع من استدعاء vad_model مباشرة — مصمم للـ streaming
vad_iter = VADIterator(
    vad_model,
    threshold=0.5,
    sampling_rate=RATE,
    min_silence_duration_ms=600,   # 600ms صمت = نهاية الكلام
    speech_pad_ms=100,
)

p = pyaudio.PyAudio()
print("✅ Ready — listening for 'يا ونيس'…")


def int2float(pcm_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    return arr / 32768.0


def stream_audio_to_server(pcm_frames: list[bytes]) -> None:
    """بعت الـ PCM للسيرفر وشغّل الـ TTS فوراً."""
    full_pcm = b"".join(pcm_frames)
    print(f"📤 Sending {len(full_pcm)/1024:.1f} KB…")

    try:
        resp = requests.post(
            f"{SERVER_URL}/{USER_ID}",
            data=full_pcm,                          # raw bytes مباشرة
            headers={"Content-Type": "audio/octet-stream"},
            stream=True,
            timeout=(3, 30),
        )
        resp.raise_for_status()

        # استقبل TTS chunks وشغّلهم فوراً
        tts_buf = bytearray()
        for chunk in resp.iter_content(chunk_size=4096):
            tts_buf.extend(chunk)

        if tts_buf:
            audio = np.frombuffer(tts_buf, dtype=np.int16)
            print("🔊 Playing response…")
            sd.play(audio, samplerate=24_000)
            sd.wait()

    except Exception as e:
        print(f"❌ Server error: {e}")


# ─── Main loop ────────────────────────────────────────────────
stream = p.open(
    format=pyaudio.paInt16,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=FRAME_LEN,
)

try:
    while True:
        # ── مرحلة 1: Wake Word بالـ PPN فقط ──────────────────
        raw = stream.read(FRAME_LEN, exception_on_overflow=False)
        pcm_frame = struct.unpack_from(f"{FRAME_LEN}h", raw)
        result = porcupine.process(pcm_frame)

        if result < 0:
            continue   # لسه ما سمعش الـ wake word

        print("🎤 Wake word detected! Listening…")
        vad_iter.reset_states()   # reset بين كل utterance

        # ── مرحلة 2: VAD Streaming — اجمع الكلام فوراً ───────
        speech_frames: list[bytes] = []
        silence_count  = 0
        speech_started = False
        MAX_SILENCE_FRAMES = int(0.8 * RATE / FRAME_LEN)  # 0.8 ثانية صمت

        while True:
            raw = stream.read(FRAME_LEN, exception_on_overflow=False)
            audio_f32 = int2float(raw)
            tensor    = torch.from_numpy(audio_f32).unsqueeze(0)

            # VADIterator.أسرع — بيرجع dict أو None
            with torch.no_grad():
                vad_out = vad_iter(tensor, return_seconds=False)

            if vad_out and "start" in vad_out:
                speech_started = True
                silence_count  = 0

            if speech_started:
                speech_frames.append(raw)

            if vad_out and "end" in vad_out:
                # الكلام خلص — ابعت فوراً
                break

            if speech_started:
                silence_count += 1
                if silence_count > MAX_SILENCE_FRAMES:
                    break   # ضمان — fallback لو VAD ما اشتغلش

            if not speech_started and len(speech_frames) > int(5 * RATE / FRAME_LEN):
                break   # مفيش كلام بعد 5 ثواني

        if speech_frames:
            # شغّل في thread منفصل عشان نرجع نسمع فوراً
            t = threading.Thread(
                target=stream_audio_to_server,
                args=(speech_frames,),
                daemon=True,
            )
            t.start()
            t.join()  # استنى الرد قبل ما نرجع للـ wake word

except KeyboardInterrupt:
    print("\n--- Stopped ---")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    porcupine.delete()