import io
import struct
import unittest
import wave
from unittest.mock import patch

from main import (
    Config,
    _capture_backend,
    _parse_avfoundation_devices,
    _parse_capture_devices,
    calculate_rms,
    find_yamnet_detection,
    make_wav,
)


class AudioHelperTests(unittest.TestCase):
    def test_config_reads_yamnet_confidence_threshold(self):
        environment = {
            "TELEGRAM_BOT_TOKEN": "test-token",
            "TELEGRAM_CHAT_ID": "test-chat",
            "DETECTION_MODE": "yamnet",
            "YAMNET_CONFIDENCE_THRESHOLD": "0.7",
            "SAMPLE_RATE": "16000",
        }
        with patch.dict("os.environ", environment, clear=True):
            config = Config.from_environment()
        self.assertEqual(config.detection_mode, "yamnet")
        self.assertEqual(config.yamnet_confidence_threshold, 0.7)

    def test_config_reads_verbose_flag(self):
        environment = {
            "TELEGRAM_BOT_TOKEN": "test-token",
            "TELEGRAM_CHAT_ID": "test-chat",
            "VERBOSE": "true",
        }
        with patch.dict("os.environ", environment, clear=True):
            config = Config.from_environment()
        self.assertTrue(config.verbose)

    def test_verbose_flag_defaults_to_false_and_rejects_invalid_values(self):
        environment = {
            "TELEGRAM_BOT_TOKEN": "test-token",
            "TELEGRAM_CHAT_ID": "test-chat",
        }
        with patch.dict("os.environ", environment, clear=True):
            self.assertFalse(Config.from_environment().verbose)
            with patch.dict("os.environ", {"VERBOSE": "sometimes"}):
                with self.assertRaisesRegex(ValueError, "VERBOSE must be true or false"):
                    Config.from_environment()

    def test_parse_capture_devices(self):
        output = (
            "**** List of CAPTURE Hardware Devices ****\n"
            "card 1: Headset [USB Headset], device 0: USB Audio [USB Audio]\n"
            "card 2: Mic [USB Mic], device 1: USB Audio [USB Audio]\n"
        )
        self.assertEqual(
            _parse_capture_devices(output),
            ["plughw:1,0", "plughw:2,1"],
        )

    def test_parse_avfoundation_audio_devices_only(self):
        output = (
            "[AVFoundation indev] AVFoundation video devices:\n"
            "[AVFoundation indev] [0] FaceTime HD Camera\n"
            "[AVFoundation indev] AVFoundation audio devices:\n"
            "[AVFoundation indev] [0] MacBook Pro Microphone\n"
            "[AVFoundation indev] [1] USB Microphone\n"
        )
        self.assertEqual(_parse_avfoundation_devices(output), [":0", ":1"])

    def test_capture_backend_for_macos_and_linux(self):
        self.assertEqual(_capture_backend("darwin"), "avfoundation")
        self.assertEqual(_capture_backend("linux"), "alsa")
        with self.assertRaises(RuntimeError):
            _capture_backend("win32")

    def test_yamnet_detection_uses_confidence_threshold(self):
        frame_scores = [[0.0] * 118]
        frame_scores[0][70] = 0.71
        self.assertEqual(
            find_yamnet_detection(frame_scores, 0.7),
            ("Bark", 0.71),
        )
        self.assertIsNone(find_yamnet_detection(frame_scores, 0.72))

    def test_yamnet_detection_requires_dog_related_class(self):
        frame_scores = [[0.0] * 118]
        frame_scores[0][5] = 0.99
        self.assertIsNone(find_yamnet_detection(frame_scores, 0.7))

    def test_calculate_rms(self):
        pcm_data = struct.pack("<hhhh", 3, -3, 4, -4)
        self.assertAlmostEqual(calculate_rms(pcm_data), 3.5355, places=4)

    def test_calculate_rms_empty_audio(self):
        self.assertEqual(calculate_rms(b""), 0.0)

    def test_calculate_rms_rejects_incomplete_sample(self):
        with self.assertRaises(ValueError):
            calculate_rms(b"\x00")

    def test_make_wav_writes_mono_16_bit_audio(self):
        pcm_data = struct.pack("<hh", 100, -100)
        wav_data = make_wav(pcm_data, 16000)

        with wave.open(io.BytesIO(wav_data), "rb") as wav_file:
            self.assertEqual(wav_file.getnchannels(), 1)
            self.assertEqual(wav_file.getsampwidth(), 2)
            self.assertEqual(wav_file.getframerate(), 16000)
            self.assertEqual(wav_file.readframes(2), pcm_data)


if __name__ == "__main__":
    unittest.main()
