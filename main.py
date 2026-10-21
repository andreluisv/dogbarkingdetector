import io
import logging
import math
import os
import re
import struct
import subprocess
import sys
import time
import wave
from dataclasses import dataclass

import requests
from dotenv import load_dotenv


LOGGER = logging.getLogger("dogbarkingdetector")
RECONNECT_DELAY_SECONDS = 2
YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"
DOG_CLASS_NAMES = {
    67: "Animal",
    68: "Domestic animals",
    69: "Dog",
    70: "Bark",
    71: "Yip",
    72: "Howl",
    73: "Bow-wow",
    74: "Growling",
    75: "Whimper",
    117: "Canidae",
}
SPECIFIC_DOG_CLASS_IDS = {69, 70, 71, 72, 73, 74, 75, 117}


@dataclass(frozen=True)
class Config:
    telegram_bot_token: str
    telegram_chat_id: str
    microphone_device: str
    verbose: bool
    detection_mode: str
    sample_rate: int
    chunk_seconds: int
    sensitivity_threshold: int
    yamnet_confidence_threshold: float
    cooldown_seconds: int

    @classmethod
    def from_environment(cls):
        load_dotenv()

        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
        if not token or not chat_id:
            raise ValueError("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set in .env")

        config = cls(
            telegram_bot_token=token,
            telegram_chat_id=chat_id,
            microphone_device=os.getenv("MIC_DEVICE", "auto").strip(),
            verbose=_boolean_setting("VERBOSE", False),
            detection_mode=os.getenv("DETECTION_MODE", "rms").strip().lower(),
            sample_rate=_integer_setting("SAMPLE_RATE", 16000),
            chunk_seconds=_integer_setting("CHUNK_SECONDS", 3),
            sensitivity_threshold=_integer_setting("SENSITIVITY_THRESHOLD", 1200),
            yamnet_confidence_threshold=_float_setting("YAMNET_CONFIDENCE_THRESHOLD", 0.7),
            cooldown_seconds=_integer_setting("COOLDOWN_SECONDS", 60),
        )
        if not config.microphone_device:
            raise ValueError("MIC_DEVICE must not be empty")
        if config.detection_mode not in {"rms", "yamnet"}:
            raise ValueError("DETECTION_MODE must be 'rms' or 'yamnet'")
        if not 8000 <= config.sample_rate <= 44100:
            raise ValueError("SAMPLE_RATE must be between 8000 and 44100")
        if config.detection_mode == "yamnet" and config.sample_rate != 16000:
            raise ValueError("YAMNet requires SAMPLE_RATE=16000")
        if not 1 <= config.chunk_seconds <= 10:
            raise ValueError("CHUNK_SECONDS must be between 1 and 10")
        if not 0 <= config.sensitivity_threshold <= 32767:
            raise ValueError("SENSITIVITY_THRESHOLD must be between 0 and 32767")
        if not 0.0 <= config.yamnet_confidence_threshold <= 1.0:
            raise ValueError("YAMNET_CONFIDENCE_THRESHOLD must be between 0.0 and 1.0")
        if config.cooldown_seconds < 0:
            raise ValueError("COOLDOWN_SECONDS must not be negative")
        return config


def _integer_setting(name, default):
    value = os.getenv(name, str(default))
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError("{} must be an integer".format(name)) from exc


def _float_setting(name, default):
    value = os.getenv(name, str(default))
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError("{} must be a number".format(name)) from exc


def _boolean_setting(name, default):
    value = os.getenv(name, str(default)).strip().lower()
    if value in {"true", "1", "yes", "on"}:
        return True
    if value in {"false", "0", "no", "off"}:
        return False
    raise ValueError("{} must be true or false".format(name))


def find_yamnet_detection(frame_scores, confidence_threshold):
    if hasattr(frame_scores, "numpy"):
        frame_scores = frame_scores.numpy()
    if hasattr(frame_scores, "tolist"):
        frame_scores = frame_scores.tolist()

    highest_scores = {}
    for frame in frame_scores:
        for class_id in DOG_CLASS_NAMES:
            if class_id < len(frame):
                highest_scores[class_id] = max(
                    highest_scores.get(class_id, 0.0),
                    float(frame[class_id]),
                )

    has_specific_dog_sound = any(
        highest_scores.get(class_id, 0.0) > 0.01
        for class_id in SPECIFIC_DOG_CLASS_IDS
    )
    confident_classes = [
        (class_id, score)
        for class_id, score in highest_scores.items()
        if score > confidence_threshold
    ]
    if not has_specific_dog_sound or not confident_classes:
        return None
    class_id, score = max(confident_classes, key=lambda item: item[1])
    return DOG_CLASS_NAMES[class_id], score


class YamnetDetector:
    def __init__(self):
        try:
            import tensorflow as tf
            import tensorflow_hub as hub
        except ImportError as exc:
            raise RuntimeError(
                "YAMNet needs the optional dependencies; install requirements-yamnet.txt"
            ) from exc

        LOGGER.info("Loading YAMNet model (downloaded once, then cached locally)")
        self.tf = tf
        self.model = hub.load(YAMNET_MODEL_URL)
        LOGGER.info("YAMNet model is ready for local inference")

    def detect(self, pcm_data, confidence_threshold):
        samples = [
            sample / 32768.0
            for (sample,) in struct.iter_unpack("<h", pcm_data)
        ]
        waveform = self.tf.convert_to_tensor(samples, dtype=self.tf.float32)
        scores, _, _ = self.model(waveform)
        return find_yamnet_detection(scores, confidence_threshold)


def _parse_capture_devices(output):
    devices = []
    for line in output.splitlines():
        match = re.match(r"^\s*card\s+(\d+):.*\bdevice\s+(\d+):", line)
        if match:
            device = "plughw:{},{}".format(match.group(1), match.group(2))
            if device not in devices:
                devices.append(device)
    return devices


def _parse_avfoundation_devices(output):
    devices = []
    in_audio_section = False
    for line in output.splitlines():
        if "AVFoundation audio devices:" in line:
            in_audio_section = True
            continue
        if "AVFoundation " in line and " devices:" in line:
            in_audio_section = False
        if in_audio_section:
            match = re.search(r"\[(\d+)\]\s+(.+?)\s*$", line)
            if match:
                devices.append(":{}".format(match.group(1)))
    return devices


def _capture_backend(platform=None):
    platform = platform or sys.platform
    if platform == "darwin":
        return "avfoundation"
    if platform.startswith("linux"):
        return "alsa"
    raise RuntimeError("Automatic microphone capture is supported on macOS and Linux")


def get_capture_devices(config):
    if config.microphone_device.lower() != "auto":
        return [config.microphone_device]
    backend = _capture_backend()
    try:
        if backend == "avfoundation":
            result = subprocess.run(
                ["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
            devices = _parse_avfoundation_devices(result.stderr)
            if not devices:
                LOGGER.warning("No macOS audio input devices found; will retry")
            return devices

        result = subprocess.run(
            ["arecord", "-l"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        LOGGER.warning("Could not list %s capture devices: %s", backend, exc)
        return []

    devices = _parse_capture_devices(result.stdout)
    if not devices:
        LOGGER.warning("No ALSA capture microphones found; will retry")
    return devices


def calculate_rms(pcm_data):
    if len(pcm_data) % 2:
        raise ValueError("16-bit PCM data must contain an even number of bytes")
    if not pcm_data:
        return 0.0
    sample_count = len(pcm_data) // 2
    square_sum = sum(sample * sample for (sample,) in struct.iter_unpack("<h", pcm_data))
    return math.sqrt(square_sum / sample_count)


def make_wav(pcm_data, sample_rate):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)
    return buffer.getvalue()


class AudioCapture:
    def __init__(self, config, device):
        backend = _capture_backend()
        if backend == "avfoundation":
            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "avfoundation",
                "-i",
                device,
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(config.sample_rate),
                "-f",
                "s16le",
                "-acodec",
                "pcm_s16le",
                "pipe:1",
            ]
        else:
            command = [
                "arecord",
                "-q",
                "-D",
                device,
                "-t",
                "raw",
                "-f",
                "S16_LE",
                "-r",
                str(config.sample_rate),
                "-c",
                "1",
            ]
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self.config = config

    def read_chunk(self):
        target_bytes = self.config.sample_rate * self.config.chunk_seconds * 2
        chunks = []
        bytes_read = 0
        while bytes_read < target_bytes:
            data = self.process.stdout.read(target_bytes - bytes_read)
            if not data:
                raise RuntimeError(
                    "Audio capture ended (exit status: {})".format(self.process.poll())
                )
            chunks.append(data)
            bytes_read += len(data)
        return b"".join(chunks)

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        if self.process.stdout:
            self.process.stdout.close()


def send_telegram_audio(config, wav_data, caption):
    url = "https://api.telegram.org/bot{}/sendDocument".format(config.telegram_bot_token)
    response = requests.post(
        url,
        data={
            "chat_id": config.telegram_chat_id,
            "caption": caption,
        },
        files={"document": ("bark.wav", io.BytesIO(wav_data), "audio/wav")},
        timeout=(5, 20),
    )
    response.raise_for_status()
    result = response.json()
    if not result.get("ok"):
        raise RuntimeError("Telegram rejected the audio upload: {}".format(result))


def run(config):
    last_alert = 0.0
    last_level_log = 0.0
    yamnet = YamnetDetector() if config.detection_mode == "yamnet" else None
    LOGGER.info(
        "Microphone: %s; detection: %s; audio: %d Hz in %d-second windows",
        config.microphone_device,
        config.detection_mode,
        config.sample_rate,
        config.chunk_seconds,
    )

    while True:
        devices = get_capture_devices(config)
        for device in devices:
            capture = None
            try:
                capture = AudioCapture(config, device)
                LOGGER.info("Listening on %s", device)
                while True:
                    pcm_data = capture.read_chunk()
                    rms = calculate_rms(pcm_data)
                    now = time.monotonic()
                    if yamnet:
                        detection = yamnet.detect(
                            pcm_data,
                            config.yamnet_confidence_threshold,
                        )
                        if detection is None:
                            continue
                        label, confidence = detection
                        caption = "Possible dog sound: {} ({:.1f}% confidence)".format(
                            label,
                            confidence * 100,
                        )
                        LOGGER.info(
                            "Detected %s (confidence %.3f)",
                            label,
                            confidence,
                        )
                    else:
                        if now - last_level_log >= 10:
                            LOGGER.info(
                                "Audio level RMS %.0f (trigger threshold %d)",
                                rms,
                                config.sensitivity_threshold,
                            )
                            last_level_log = now
                        if rms < config.sensitivity_threshold:
                            continue
                        caption = "Possible loud sound (audio RMS: {:.0f})".format(rms)

                    if now - last_alert < config.cooldown_seconds:
                        continue

                    last_alert = now
                    try:
                        send_telegram_audio(
                            config,
                            make_wav(pcm_data, config.sample_rate),
                            caption,
                        )
                        LOGGER.info("Sent Telegram alert")
                    except (requests.RequestException, RuntimeError, ValueError) as exc:
                        error = str(exc).replace(config.telegram_bot_token, "[redacted]")
                        LOGGER.error("Could not send Telegram alert: %s", error)
            except (OSError, RuntimeError) as exc:
                LOGGER.warning(
                    "Audio capture failed on %s: %s; trying another microphone",
                    device,
                    exc,
                )
            finally:
                if capture is not None:
                    capture.close()
            time.sleep(RECONNECT_DELAY_SECONDS)
        if not devices:
            time.sleep(RECONNECT_DELAY_SECONDS)


def main():
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        config = Config.from_environment()
        logging.getLogger().setLevel(logging.DEBUG if config.verbose else logging.WARNING)
        run(config)
    except (OSError, RuntimeError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        LOGGER.info("Stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
