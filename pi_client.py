"""
pi_client.py — Younes Raspberry Pi Client
==========================================
الـ Pi بيعمل بس:
  1. PPN wake word (Porcupine) — بدون TFLite / MFCC
  2. Silero VAD streaming — يبعت الـ PCM فوراً لما الكلام يخلص
  3. HTTP streaming POST للسيرفر
  4. يشغّل الـ TTS اللي رجع

مُحسَّن لـ Raspberry Pi 4 (ARM CPU, no GPU)
"""

# ═══════════════════════════════════════════════════════════════
#  STANDARD LIBRARY
# ═══════════════════════════════════════════════════════════════
import struct
import threading
import logging
import time
from typing import List

# ═══════════════════════════════════════════════════════════════
#  THIRD-PARTY
# ═══════════════════════════════════════════════════════════════
import numpy as np
import pyaudio
import pvporcupine
import torch
import requests
import sounddevice as sd

# ─────────────────────────── logging ────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("younes-pi")

# ═══════════════════════════════════════════════════════════════
#  CONFIG — غيّر هنا بس
# ═══════════════════════════════════════════════════════════════
SERVER_URL     = "https://7501-01khejztq1640gknejy92er4gq.cloudspaces.litng.ai/transcribe"
USER_ID        = "pi-living-room-001"
PPN_ACCESS_KEY = "o03bvBbHZkULR6O6MOvvDWQ1UUMKrvv6ErIUp8YZ2KJov90RA1CYaA=="
PPN_MODEL_PATH = "wa-nees_en_raspberry-pi_v4_0_0.ppn"   # نسخة Raspberry Pi من Picovoice Console

RATE       = 16_000
CHANNELS   = 1
TTS_RATE   = 24_000

# VAD tuning
VAD_THRESHOLD          = 0.5
VAD_MIN_SILENCE_MS     = 600    # ms صمت = نهاية الكلام
VAD_SPEECH_PAD_MS      = 100
MAX_SILENCE_FRAMES_SEC = 0.8    # ثانية fallback لو VAD ما اشتغلش
MAX_NO_SPEECH_SEC      = 5.0    # ثواني max استنى لو مفيش كلام

# ═══════════════════════════════════════════════════════════════
#  LOAD MODELS — مرة واحدة عند الـ startup
# ═══════════════════════════════════════════════════════════════
log.info("⏳ Loading Porcupine wake word engine…")
porcupine = pvporcupine.create(
    access_key=PPN_ACCESS_KEY,
    keyword_paths=[PPN_MODEL_PATH],
)
FRAME_LEN = porcupine.frame_length  # عادةً 512 sample

log.info("⏳ Loading Silero VAD…")
vad_model, utils = torch.hub.load(
    "snakers4/silero-vad",
    "silero_vad",
    force_reload=False,
    trust_repo=True,
)
(_, _, _, VADIterator, _) = utils
vad_model.eval()

# VADIterator مصمم للـ streaming — أسرع من vad_model() مباشرة
vad_iter = VADIterator(
    vad_model,
    threshold=VAD_THRESHOLD,
    sampling_rate=RATE,
    min_silence_duration_ms=VAD_MIN_SILENCE_MS,
    speech_pad_ms=VAD_SPEECH_PAD_MS,
)

# PyAudio instance
pa = pyaudio.PyAudio()

log.info("✅ Models loaded — listening for 'يا ونيس'…")


# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════

def pcm_bytes_to_float32(pcm_bytes: bytes) -> np.ndarray:
    """PCM-16 bytes → float32 numpy array in [-1, 1]."""
    arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    return arr / 32768.0


def play_pcm(pcm_bytes: bytes) -> None:
    """Play raw PCM-16 24kHz audio."""
    audio = np.frombuffer(pcm_bytes, dtype=np.int16)
    sd.play(audio, samplerate=TTS_RATE)
    sd.wait()


# ═══════════════════════════════════════════════════════════════
#  SERVER COMMUNICATION
# ═══════════════════════════════════════════════════════════════

def send_audio_and_play(pcm_frames: List[bytes]) -> None:
    """
    بعت الـ PCM للسيرفر كـ raw bytes واستقبل TTS streaming.
    شغّل الصوت فوراً لما يكمل.
    """
    full_pcm = b"".join(pcm_frames)
    log.info("📤 Sending %.1f KB to server…", len(full_pcm) / 1024)
    t0 = time.time()

    try:
        resp = requests.post(
            f"{SERVER_URL}/{USER_ID}",
            data=full_pcm,
            headers={"Content-Type": "audio/octet-stream"},
            stream=True,
            timeout=(5, 60),    # 5s connect, 60s read
        )
        resp.raise_for_status()

        # اجمع الـ TTS chunks
        tts_buf = bytearray()
        for chunk in resp.iter_content(chunk_size=4096):
            if chunk:
                tts_buf.extend(chunk)

        log.info("✅ Response received in %.2fs — playing…", time.time() - t0)

        if tts_buf:
            play_pcm(bytes(tts_buf))

    except requests.exceptions.ConnectionError:
        log.error("❌ Cannot reach server — check URL and Lightning Studio is running")
    except requests.exceptions.Timeout:
        log.error("❌ Server timeout — check vLLM is running on port 8000")
    except requests.exceptions.HTTPError as e:
        log.error("❌ HTTP error: %s", e)
    except Exception as e:
        log.error("❌ Unexpected error: %s", e)


# ═══════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════

stream = pa.open(
    format=pyaudio.paInt16,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=FRAME_LEN,
)

try:
    while True:

        # ── Phase 1: Wake Word Detection (PPN) ────────────────
        raw = stream.read(FRAME_LEN, exception_on_overflow=False)
        pcm_frame = struct.unpack_from(f"{FRAME_LEN}h", raw)
        result = porcupine.process(pcm_frame)

        if result < 0:
            continue    # لسه ما سمعش الـ wake word

        log.info("🎤 Wake word detected! Recording…")
        vad_iter.reset_states()

        # ── Phase 2: VAD Streaming — اجمع الكلام ──────────────
        speech_frames: List[bytes] = []
        silence_count   = 0
        speech_started  = False
        no_speech_count = 0

        MAX_SILENCE_FRAMES  = int(MAX_SILENCE_FRAMES_SEC * RATE / FRAME_LEN)
        MAX_NO_SPEECH_FRAMES = int(MAX_NO_SPEECH_SEC * RATE / FRAME_LEN)

        while True:
            raw = stream.read(FRAME_LEN, exception_on_overflow=False)
            audio_f32 = pcm_bytes_to_float32(raw)
            tensor = torch.from_numpy(audio_f32).unsqueeze(0)

            with torch.no_grad():
                vad_out = vad_iter(tensor, return_seconds=False)

            # كلام بدأ
            if vad_out and "start" in vad_out:
                speech_started = True
                silence_count  = 0
                log.info("🗣  Speech started…")

            if speech_started:
                speech_frames.append(raw)

            # كلام خلص — ابعت فوراً
            if vad_out and "end" in vad_out:
                log.info("🔇 Speech ended — sending.")
                break

            # Fallback: صمت طويل بعد ما الكلام بدأ
            if speech_started:
                silence_count += 1
                if silence_count > MAX_SILENCE_FRAMES:
                    log.info("🔇 Silence fallback — sending.")
                    break

            # Fallback: مفيش كلام خالص بعد MAX_NO_SPEECH_SEC
            if not speech_started:
                no_speech_count += 1
                if no_speech_count > MAX_NO_SPEECH_FRAMES:
                    log.info("⏭  No speech detected — back to listening.")
                    break

        # ── Phase 3: بعت للسيرفر في thread منفصل ─────────────
        if speech_frames:
            t = threading.Thread(
                target=send_audio_and_play,
                args=(speech_frames,),
                daemon=True,
            )
            t.start()
            t.join()    # استنى الرد قبل ما نرجع للـ wake word

except KeyboardInterrupt:
    log.info("\n--- Stopped by user ---")

finally:
    stream.stop_stream()
    stream.close()
    pa.terminate()
    porcupine.delete()
    log.info("🧹 Cleanup done.")
