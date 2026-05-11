"""
pi_client_optimized.py
======================
الـ Pi بيعمل بس:
  1. PPN wake word (Porcupine) — بدون TFLite / MFCC
  2. Silero VAD streaming — يبعت الـ PCM فوراً لما الكلام يخلص
  3. HTTP streaming POST للسيرفر
  4. يشغّل الـ TTS اللي رجع
  5. بيعمل Beep لما يتم الكشف عن الـ Wake Word

pip install pvporcupine pyaudio numpy requests sounddevice torch
"""

import struct
import math
import numpy as np
import pyaudio
import pvporcupine
import torch
import requests
import sounddevice as sd
import threading
import time

# ─── Config ───────────────────────────────────────────────────
SERVER_URL      = "https://7501-01khejztq1640gknejy92er4gq.cloudspaces.litng.ai/transcribe"
USER_ID         = "pi-living-room-001"
PPN_ACCESS_KEY  = "o03bvBbHZkULR6O6MOvvDWQ1UUMKrvv6ErIUp8YZ2KJov90RA1CYaA=="
PPN_MODEL_PATH  = "wa-nees_en_raspberry-pi_v4_0_0.ppn"   # ملفك الـ .ppn
RATE            = 16_000
CHANNELS        = 1

# Beep settings
BEEP_FREQUENCY  = 880       # Hz  — A5 note, clear and friendly
BEEP_DURATION   = 0.18      # seconds
BEEP_VOLUME     = 0.6       # 0.0 → 1.0
BEEP_SAMPLE_RATE = 44_100   # Hz

# VAD settings
VAD_THRESHOLD           = 0.5
VAD_MIN_SILENCE_MS      = 600    # ms of silence to consider speech ended
VAD_SPEECH_PAD_MS       = 100
PRE_SPEECH_BUFFER_SEC   = 0.3    # seconds of audio to keep before speech starts
MAX_SPEECH_SEC          = 10.0   # hard cap on recording duration
NO_SPEECH_TIMEOUT_SEC   = 5.0    # give up if no speech after wake word
# ─────────────────────────────────────────────────────────────


# ─── Beep Generator ───────────────────────────────────────────
def generate_beep(
    frequency: float = BEEP_FREQUENCY,
    duration: float  = BEEP_DURATION,
    volume: float    = BEEP_VOLUME,
    sample_rate: int = BEEP_SAMPLE_RATE,
    fade_ms: int     = 15,
) -> np.ndarray:
    """
    Generate a smooth sine-wave beep with fade-in/fade-out to avoid clicks.
    Returns int16 numpy array ready for sounddevice.
    """
    num_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)
    wave = np.sin(2 * math.pi * frequency * t).astype(np.float32)

    # Fade in/out to remove clicks
    fade_samples = int(sample_rate * fade_ms / 1000)
    fade_samples = min(fade_samples, num_samples // 2)
    fade_in  = np.linspace(0.0, 1.0, fade_samples)
    fade_out = np.linspace(1.0, 0.0, fade_samples)
    wave[:fade_samples]  *= fade_in
    wave[-fade_samples:] *= fade_out

    wave *= volume
    return (wave * 32767).astype(np.int16)


BEEP_AUDIO = generate_beep()   # pre-generate once at startup


def play_beep() -> None:
    """Play wake-word confirmation beep (non-blocking)."""
    try:
        sd.play(BEEP_AUDIO, samplerate=BEEP_SAMPLE_RATE)
        sd.wait()
    except Exception as e:
        print(f"⚠ Beep playback error: {e}")


# ─── Load models once at startup ──────────────────────────────
print("⏳ Loading Porcupine…")
porcupine = pvporcupine.create(
    access_key=PPN_ACCESS_KEY,
    keyword_paths=[PPN_MODEL_PATH],
)
FRAME_LEN = porcupine.frame_length   # usually 512

print("⏳ Loading Silero VAD…")
vad_model, utils = torch.hub.load(
    "snakers4/silero-vad", "silero_vad", force_reload=False
)
(get_speech_timestamps, _, _, VADIterator, _) = utils
vad_model.eval()

vad_iter = VADIterator(
    vad_model,
    threshold=VAD_THRESHOLD,
    sampling_rate=RATE,
    min_silence_duration_ms=VAD_MIN_SILENCE_MS,
    speech_pad_ms=VAD_SPEECH_PAD_MS,
)

p = pyaudio.PyAudio()
print("✅ Ready — listening for 'يا ونيس'…\n")


# ─── Helpers ──────────────────────────────────────────────────
def int2float(pcm_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32)
    return arr / 32768.0


def stream_audio_to_server(pcm_frames: list) -> None:
    """Send captured PCM to server and play back TTS response."""
    full_pcm = b"".join(pcm_frames)
    duration  = len(full_pcm) / (RATE * 2)   # 2 bytes per int16 sample
    print(f"📤 Sending {len(full_pcm) / 1024:.1f} KB  ({duration:.1f}s of audio)…")

    try:
        resp = requests.post(
            f"{SERVER_URL}/{USER_ID}",
            data=full_pcm,
            headers={"Content-Type": "audio/octet-stream"},
            stream=True,
            timeout=(5, 60),
        )
        resp.raise_for_status()

        tts_buf = bytearray()
        for chunk in resp.iter_content(chunk_size=4096):
            tts_buf.extend(chunk)

        if tts_buf:
            audio = np.frombuffer(tts_buf, dtype=np.int16)
            print("🔊 Playing Younes' response…")
            sd.play(audio, samplerate=24_000)
            sd.wait()
            print("✅ Done. Back to listening…\n")
        else:
            print("⚠ Server returned empty audio.\n")

    except requests.exceptions.Timeout:
        print("❌ Server timeout — is the Lightning server running?\n")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot reach server — check SERVER_URL and network.\n")
    except Exception as e:
        print(f"❌ Server error: {e}\n")


# ─── VAD Recording ────────────────────────────────────────────
def record_speech_with_vad(stream) -> list:
    """
    Record audio after wake word using Silero VADIterator.
    Returns list of raw PCM byte-frames (int16, 16kHz mono).
    Fixes:
      - Pre-speech buffer to avoid clipping word starts
      - Correct timeout logic
      - ended flag to avoid double-appending after VAD "end"
    """
    # Reset VAD state between utterances
    try:
        vad_iter.reset_states()
    except AttributeError:
        vad_model.reset_states()   # older Silero versions

    speech_frames: list  = []
    pre_buffer:    list  = []
    speech_started = False
    speech_ended   = False

    pre_buffer_max   = int(PRE_SPEECH_BUFFER_SEC * RATE / FRAME_LEN)
    max_frames       = int(MAX_SPEECH_SEC        * RATE / FRAME_LEN)
    no_speech_frames = int(NO_SPEECH_TIMEOUT_SEC * RATE / FRAME_LEN)

    total_frames = 0

    while True:
        raw      = stream.read(FRAME_LEN, exception_on_overflow=False)
        f32      = int2float(raw)
        tensor   = torch.from_numpy(f32).unsqueeze(0)
        total_frames += 1

        with torch.no_grad():
            vad_out = vad_iter(tensor, return_seconds=False)

        # ── Pre-speech rolling buffer ────────────────────────
        if not speech_started:
            pre_buffer.append(raw)
            if len(pre_buffer) > pre_buffer_max:
                pre_buffer.pop(0)

        # ── Speech start event ───────────────────────────────
        if vad_out and "start" in vad_out:
            speech_started = True
            speech_frames.extend(pre_buffer)   # include audio before VAD fires
            pre_buffer = []
            print("🗣  Speech started…")

        # ── Accumulate frames while speaking ─────────────────
        if speech_started and not speech_ended:
            speech_frames.append(raw)

        # ── Speech end event ─────────────────────────────────
        if vad_out and "end" in vad_out:
            speech_ended = True
            print("🔇 Speech ended.")
            break

        # ── Hard cap: max recording duration ─────────────────
        if len(speech_frames) >= max_frames:
            print(f"⏱ Max recording duration ({MAX_SPEECH_SEC}s) reached.")
            break

        # ── No speech timeout ─────────────────────────────────
        if not speech_started and total_frames >= no_speech_frames:
            print(f"⏱ No speech detected in {NO_SPEECH_TIMEOUT_SEC}s — returning to wake word.")
            break

    return speech_frames


# ─── Main Loop ────────────────────────────────────────────────
stream = p.open(
    format=pyaudio.paInt16,
    channels=CHANNELS,
    rate=RATE,
    input=True,
    frames_per_buffer=FRAME_LEN,
)

try:
    while True:
        # ── Phase 1: Wake Word Detection (Porcupine) ──────────
        raw       = stream.read(FRAME_LEN, exception_on_overflow=False)
        pcm_frame = struct.unpack_from(f"{FRAME_LEN}h", raw)
        result    = porcupine.process(pcm_frame)

        if result < 0:
            continue   # wake word not detected yet

        # ── Wake word confirmed ────────────────────────────────
        print("🔔 Wake word detected!")

        # Play beep in a separate thread so we don't block the mic stream.
        # The mic keeps buffering in pyaudio's ring buffer during beep playback.
        beep_thread = threading.Thread(target=play_beep, daemon=True)
        beep_thread.start()
        beep_thread.join()   # wait for beep to finish before recording
                             # (~180ms) — ensures beep doesn't bleed into ASR

        # ── Phase 2: VAD-based speech capture ─────────────────
        speech_frames = record_speech_with_vad(stream)

        if not speech_frames:
            print("⚠ No speech captured — back to listening.\n")
            continue

        # ── Phase 3: Send to server & play TTS ────────────────
        # Run in the main thread (join) so wake word detection
        # doesn't restart while Younes is still talking.
        stream_audio_to_server(speech_frames)

except KeyboardInterrupt:
    print("\n\n--- Stopped by user ---")

finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    porcupine.delete()
    print("Cleanup done. Goodbye!")
