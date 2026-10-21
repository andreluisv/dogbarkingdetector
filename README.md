# Dog Barking Detector

This repository contains a browser-based YAMNet detector and a separate,
lightweight headless listener for Raspberry Pi.

## Raspberry Pi quick start

Install ALSA and Python, then run from the project directory:

```sh
sudo apt install -y alsa-utils python3-venv
python3 -m venv .venv
.venv/bin/pip install -r requirements-yamnet.txt
cp .env.example .env
nano .env
.venv/bin/python main.py
```

The default `DETECTION_MODE=yamnet` uses local YAMNet inference. Its model is
downloaded and cached the first time it runs; subsequent inference runs locally.
Use a 64-bit Raspberry Pi OS on a newer Pi (Pi 5 recommended). The original Pi 1
is not suitable for this TensorFlow model. For the lightweight volume trigger,
set `DETECTION_MODE=rms` and install only `requirements.txt`.

Set `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `.env`, and set
`MIC_DEVICE=auto` to use automatic selection. It finds ALSA capture devices and
tries them in order, retrying discovery if none are found or capture stops. To
pin a specific microphone, set a device such as `MIC_DEVICE=plughw:2,0`.
Set `VERBOSE=true` to show informational/debug logs; `VERBOSE=false` hides them
while warnings and errors remain visible.

## macOS

macOS uses FFmpeg/AVFoundation instead of ALSA. Install FFmpeg with
`brew install ffmpeg`, install the YAMNet dependencies with
`.venv/bin/pip install -r requirements-yamnet.txt`, set `MIC_DEVICE=auto` in
`.env`, then start with `.venv/bin/python main.py`. Automatic selection tries the
audio inputs reported by FFmpeg. Linux Raspberry Pi continues to use `arecord`/ALSA.

On macOS, grant microphone access to the terminal app running the script in
**System Settings → Privacy & Security → Microphone**.

## Adjust tolerance

In YAMNet mode, `YAMNET_CONFIDENCE_THRESHOLD` (default `0.7`) works like the
browser's tolerance: lower it to detect more sounds, raise it to require stronger
model confidence. It accepts values from `0.0` to `1.0`. In lightweight `rms`
mode, use `SENSITIVITY_THRESHOLD` instead; it is a microphone-volume threshold,
not a model confidence.

`CHUNK_SECONDS` sets the audio clip length (default 3 seconds), and
`COOLDOWN_SECONDS` limits how often alerts are sent (default 60 seconds).

## Start automatically on boot

Create `/etc/systemd/system/dogbarkingdetector.service`, replacing `<USER>` with
the Pi account that owns the project:

```ini
[Unit]
Description=Dog Barking Detector
After=network-online.target sound.target

[Service]
User=<USER>
WorkingDirectory=/home/<USER>/dogbarkingdetector
EnvironmentFile=/home/<USER>/dogbarkingdetector/.env
ExecStart=/home/<USER>/dogbarkingdetector/.venv/bin/python /home/<USER>/dogbarkingdetector/main.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

Enable and check the service:

```sh
sudo systemctl daemon-reload
sudo systemctl enable --now dogbarkingdetector
journalctl -u dogbarkingdetector -f
```

The service account needs access to the microphone (usually via the `audio`
group). Keep `.env` private because it contains the bot token.

## Browser version

The browser app is in [`docs/`](docs/) and uses TensorFlow.js with YAMNet for
local sound classification. See the [live demo](https://dogbarkingdetector.com/).

Licensed under the [MIT License](LICENSE).
