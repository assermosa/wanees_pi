"""
pi_main.py — Wanees AI Voice Assistant
Raspberry Pi Edge Client

Flow:
  1. Rolling buffer يسمع باستمرار
  2. Wake word "ونيس" → TFLite classifier
  3. Beep للتنبيه
  4. WebRTC VAD يسجل لحد ما تصمت
  5. يرسل PCM للسيرفر عبر WebSocket
  6. يستقبل TTS ويشغله
"""

import asyncio
import json
import logging
import os
import time
import numpy as np
import sounddevice as sd
import tensorflow as tf
import websockets
from openwakeword.utils import AudioFeatures

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("wanees-pi")

# ─────────────────────────────────────────────
#  CONFIG — عدّل هنا بس
# ─────────────────────────────────────────────
SERVER_URI       = "ws://<YOUR_SERVER_IP>:8765"   # ← IP السيرفر
USER_ID          = "pi-living-room-001"            # ← ID الجهاز
WAKEWORD_MODEL   = "/home/pi/younes/wanees.tflite" # ← مسار الموديل
WAKEWORD_THRESH  = 0.4                             # ← الـ threshold

SAMPLE_RATE      = 16_000   # Hz
CHUNK_SIZE       = 1_280    # 80ms per chunk
BUFFER_SECONDS   = 1        # حجم الـ rolling buffer
BUFFER_SAMPLES   = SAMPLE_RATE * BUFFER_SECONDS

TTS_RATE         = 24_000   # sample rate للصوت الراجع

# VAD settings
VAD_AGGRESSIVENESS = 2        # 0=permissive → 3=aggressive
VAD_SILENCE_SEC    = 1.2      # ثواني صمت قبل ما يوقف التسجيل
VAD_MAX_SEC        = 10.0     # أقصى مدة تسجيل
VAD_MIN_SEC        = 0.8      # أقل مدة تسجيل صالحة

# ─────────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────────

def load_models():
    """Load TFLite classifier + AudioFeatures embedding extractor."""
    log.info("⏳ Loading TFLite wake word model...")
    interpreter = tf.lite.Interpreter(model_path=WAKEWORD_MODEL)
    interpreter.allocate_tensors()
    inp_det = interpreter.get_input_details()
    out_det = interpreter.get_output_details()
    log.info("✅ TFLite loaded")

    log.info("⏳ Loading AudioFeatures (embedding extractor)...")
    audio_features = AudioFeatures()
    log.info("✅ AudioFeatures loaded")

    return interpreter, inp_det, out_det, audio_features


# ─────────────────────────────────────────────
#  WAKE WORD SCORING
# ─────────────────────────────────────────────

def get_wakeword_score(
    audio_int16: np.ndarray,
    interpreter: tf.lite.Interpreter,
    inp_det: list,
    out_det: list,
    af: AudioFeatures,
) -> float:
    """
    Extract embedding from 1-second audio chunk
    then run TFLite classifier → returns float score 0.0–1.0
    """
    # Ensure exactly 1 second
    if len(audio_int16) < BUFFER_SAMPLES:
        audio_int16 = np.pad(
            audio_int16, (0, BUFFER_SAMPLES - len(audio_int16)))
    audio_int16 = audio_int16[:BUFFER_SAMPLES]

    # embed_clips expects shape (N, samples) int16
    clips = audio_int16.reshape(1, BUFFER_SAMPLES)
    emb   = af.embed_clips(clips)                  # (1, frames, 96)
    emb   = emb.mean(axis=(0, 1)).reshape(1, -1).astype(np.float32)

    interpreter.set_tensor(inp_det[0]["index"], emb)
    interpreter.invoke()
    return float(interpreter.get_tensor(out_det[0]["index"])[0][0])


# ─────────────────────────────────────────────
#  BEEP
# ─────────────────────────────────────────────

def play_beep(freq: int = 880, duration: float = 0.15):
    """Play a short confirmation beep."""
    t    = np.linspace(0, duration, int(TTS_RATE * duration), False)
    beep = (np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    sd.play(beep, samplerate=TTS_RATE)
    sd.wait()


# ─────────────────────────────────────────────
#  VAD — WebRTC
# ─────────────────────────────────────────────

def record_with_vad() -> bytes:
    """
    Record audio using WebRTC VAD.
    Stops when VAD_SILENCE_SEC of continuous silence is detected
    or VAD_MAX_SEC total is reached.
    Returns raw PCM-16 bytes.
    """
    try:
        import webrtcvad
        vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        use_webrtc = True
        log.info("🎙 Recording with WebRTC VAD...")
    except ImportError:
        use_webrtc = False
        log.warning("⚠️  webrtcvad not installed — using energy VAD fallback")

    # WebRTC VAD needs exactly 10ms, 20ms, or 30ms frames
    # At 16kHz: 20ms = 320 samples
    VAD_FRAME  = 320
    recorded   = []
    silent_ms  = 0
    total_ms   = 0
    silence_limit_ms = int(VAD_SILENCE_SEC * 1000)
    max_ms           = int(VAD_MAX_SEC     * 1000)
    frame_ms         = int(VAD_FRAME / SAMPLE_RATE * 1000)  # 20ms

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=VAD_FRAME,
    ) as stream:

        while total_ms < max_ms:
            frame, _ = stream.read(VAD_FRAME)
            frame_np  = frame[:, 0]
            recorded.append(frame_np.copy())
            total_ms += frame_ms

            if use_webrtc:
                try:
                    is_speech = vad.is_speech(
                        frame_np.tobytes(), SAMPLE_RATE)
                except Exception:
                    is_speech = _energy_vad(frame_np)
            else:
                is_speech = _energy_vad(frame_np)

            if is_speech:
                silent_ms = 0
            else:
                silent_ms += frame_ms

            # Stop if silence after minimum speech
            if (silent_ms >= silence_limit_ms and
                    total_ms >= VAD_MIN_SEC * 1000):
                log.info(
                    f"🔇 Silence detected — {total_ms/1000:.1f}s recorded")
                break

    audio = np.concatenate(recorded, axis=0)
    log.info(f"📼 Recorded {len(audio)/SAMPLE_RATE:.1f}s")
    return audio.tobytes()


def _energy_vad(frame: np.ndarray, threshold: float = 0.01) -> bool:
    """Simple RMS energy VAD fallback."""
    rms = np.sqrt(np.mean(frame.astype(np.float32) ** 2)) / 32768.0
    return rms > threshold


# ─────────────────────────────────────────────
#  SERVER COMMUNICATION
# ─────────────────────────────────────────────

async def send_to_server(pcm_bytes: bytes):
    """
    Send PCM audio to Wanees server via WebSocket.
    Receives and plays back TTS response.
    """
    log.info(f"📡 Connecting to {SERVER_URI}...")

    try:
        async with websockets.connect(
            SERVER_URI,
            ping_interval=20,
            ping_timeout=30,
            close_timeout=10,
        ) as ws:

            # ── Send header + raw audio ────────────────────────
            await ws.send(json.dumps({
                "type":     "audio",
                "user_id":  USER_ID,
                "encoding": "pcm16_16k_mono",
            }, ensure_ascii=False))
            await ws.send(pcm_bytes)

            log.info("📨 Audio sent — waiting for response...")

            # ── Receive events + audio ─────────────────────────
            audio_buffer = bytearray()

            async for message in ws:

                if isinstance(message, bytes):
                    # TTS audio chunk
                    audio_buffer.extend(message)

                else:
                    event    = json.loads(message)
                    evt_type = event.get("type")

                    if evt_type == "asr":
                        log.info(f"📝 You said : {event.get('text')}")

                    elif evt_type == "llm":
                        log.info(f"💬 Wanees   : {event.get('text')}")

                    elif evt_type == "emergency":
                        log.warning(f"🚨 Emergency: {event.get('text')}")

                    elif evt_type == "tts_end":
                        log.info("🔊 Playing response...")
                        if audio_buffer:
                            audio_np = np.frombuffer(
                                audio_buffer, dtype=np.int16)
                            sd.play(audio_np, samplerate=TTS_RATE)
                            sd.wait()
                            log.info("✅ Playback done")
                        else:
                            log.warning("⚠️  No audio received")
                        break

                    elif evt_type == "error":
                        log.error(f"❌ Server: {event.get('message')}")
                        break

    except websockets.exceptions.ConnectionRefusedError:
        log.error("❌ Cannot connect to server — is it running?")
    except websockets.exceptions.ConnectionClosedError as e:
        log.error(f"❌ Connection closed unexpectedly: {e}")
    except Exception as e:
        log.exception(f"❌ Unexpected error: {e}")


# ─────────────────────────────────────────────
#  MAIN LOOP
# ─────────────────────────────────────────────

async def main():
    # ── Load models ────────────────────────────────────────────
    interpreter, inp_det, out_det, audio_features = load_models()

    # ── Rolling buffer ─────────────────────────────────────────
    buffer       = np.zeros(BUFFER_SAMPLES, dtype=np.int16)
    # Score every N chunks to reduce CPU load
    SCORE_EVERY  = 4   # score every 4 chunks = every 320ms
    chunk_count  = 0
    last_trigger = 0   # timestamp of last detection (cooldown)
    COOLDOWN_SEC = 2.0 # minimum seconds between detections

    log.info("=" * 50)
    log.info('👂 Listening for "ونيس" or "يا ونيس"...')
    log.info(f"   Threshold : {WAKEWORD_THRESH}")
    log.info(f"   Server    : {SERVER_URI}")
    log.info("=" * 50)

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=CHUNK_SIZE,
    ) as stream:

        while True:
            # ── Read chunk ─────────────────────────────────────
            chunk, _ = stream.read(CHUNK_SIZE)
            chunk_np  = chunk[:, 0]

            # ── Update rolling buffer ──────────────────────────
            buffer = np.roll(buffer, -CHUNK_SIZE)
            buffer[-CHUNK_SIZE:] = chunk_np
            chunk_count += 1

            # ── Score every N chunks ───────────────────────────
            if chunk_count % SCORE_EVERY != 0:
                continue

            score = get_wakeword_score(
                buffer.copy(),
                interpreter, inp_det, out_det,
                audio_features,
            )

            # ── Debug: show score when it's rising ────────────
            if score > 0.2:
                log.debug(f"Score: {score:.3f}")

            # ── Wake word detected ─────────────────────────────
            now = time.time()
            if score >= WAKEWORD_THRESH and (now - last_trigger) > COOLDOWN_SEC:
                last_trigger = now
                log.info(f"\n🎯 Wake word detected! score={score:.3f}")

                # ── Beep ───────────────────────────────────────
                play_beep()

                # ── Record until silence (VAD) ─────────────────
                pcm_bytes = record_with_vad()

                # ── Validate minimum length ────────────────────
                min_bytes = int(VAD_MIN_SEC * SAMPLE_RATE * 2)
                if len(pcm_bytes) < min_bytes:
                    log.warning("⚠️  Recording too short — ignoring")
                else:
                    # ── Send to server ─────────────────────────
                    await send_to_server(pcm_bytes)

                # ── Reset buffer to avoid re-trigger ──────────
                buffer = np.zeros(BUFFER_SAMPLES, dtype=np.int16)
                log.info(f'\n👂 Back to listening for "ونيس"...\n')

            # ── Yield control to event loop ────────────────────
            await asyncio.sleep(0)


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # ── Check dependencies ─────────────────────────────────────
    print("\n" + "=" * 50)
    print("  Wanees AI — Raspberry Pi Client")
    print("=" * 50)

    missing = []
    for pkg in ["sounddevice", "tensorflow", "websockets",
                "openwakeword", "numpy"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    try:
        import webrtcvad
        print("  VAD      : WebRTC ✅")
    except ImportError:
        print("  VAD      : Energy fallback ⚠️  (pip install webrtcvad)")

    if missing:
        print(f"\n❌ Missing packages: {missing}")
        print(f"   pip install {' '.join(missing)}")
        exit(1)

    print(f"  Model    : {WAKEWORD_MODEL}")
    print(f"  Server   : {SERVER_URI}")
    print(f"  Threshold: {WAKEWORD_THRESH}")
    print(f"  User ID  : {USER_ID}")

    if not os.path.exists(WAKEWORD_MODEL):
        print(f"\n❌ Model not found: {WAKEWORD_MODEL}")
        print("   Copy wanees.tflite to the Pi first:")
        print("   scp wanees.tflite pi@<PI_IP>:/home/pi/younes/")
        exit(1)

    print("\n✅ All checks passed — starting...\n")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Stopped by user")