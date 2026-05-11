# Younes — Raspberry Pi Setup Guide

## الـ Pi المدعوم
- Raspberry Pi 4 (2GB RAM minimum, 4GB recommended)
- OS: Raspberry Pi OS Bullseye 64-bit
- Python: 3.9 أو 3.10

---

## خطوة 1 — System Dependencies

```bash
sudo apt-get update
sudo apt-get install -y \
    portaudio19-dev \
    python3-pyaudio \
    libatlas-base-dev \
    libasound2-dev \
    ffmpeg
```

---

## خطوة 2 — PyTorch CPU (لازم يتنصب قبل الباقي)

```bash
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu
```

> ⚠️ متعملش `pip install torch` من غير الـ URL ده — هينزل نسخة CUDA كبيرة جداً

---

## خطوة 3 — باقي الـ Libraries

```bash
pip install -r requirements_pi.txt
```

---

## خطوة 4 — Silero VAD (بيتحمل تلقائي أول مرة)

مش محتاج تنصبه — الكود بيعمل `torch.hub.load` تلقائياً أول ما تشغّل.
بس لازم يكون عندك إنترنت أول مرة بس.

---

## خطوة 5 — ملف الـ Wake Word (.ppn)

1. روح على: https://console.picovoice.ai
2. Wake Words ← الكلمة بتاعتك
3. Download ← **Platform: Raspberry Pi** ← Version: v4
4. حط الـ `.ppn` في نفس فولدر `pi_client.py`

---

## خطوة 6 — شغّل الكود

```bash
python pi_client.py
```

---

## Troubleshooting

| المشكلة | الحل |
|---------|------|
| `PorcupineInvalidArgumentError` | الـ .ppn مش نسخة Raspberry Pi — حمّل التاني |
| `OSError: No Default Input Device` | `sudo raspi-config` ← Audio ← اختار الميك |
| `Connection refused` | تأكد Lightning Studio شغّال والـ port 7501 مفتوح |
| `Timeout` | تأكد vLLM شغّال على السيرفر على port 8000 |
