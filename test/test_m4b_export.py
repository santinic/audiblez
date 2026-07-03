import json
import math
import struct
import subprocess
import tempfile
import unittest
import wave
import zlib
from pathlib import Path

from audiblez.core import create_index_file, create_m4b


def _png_bytes(width=16, height=16):
    def chunk(kind, data):
        payload = kind + data
        crc = zlib.crc32(payload) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + payload + struct.pack(">I", crc)

    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            row.extend((x * 16 % 256, y * 16 % 256, 128))
        rows.append(bytes(row))

    return b"".join([
        b"\x89PNG\r\n\x1a\n",
        chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)),
        chunk(b"IDAT", zlib.compress(b"".join(rows))),
        chunk(b"IEND", b""),
    ])


def _write_tone(path, frequency):
    sample_rate = 24000
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        frames = bytearray()
        for i in range(sample_rate):
            value = int(9000 * math.sin(2 * math.pi * frequency * i / sample_rate))
            frames.extend(struct.pack("<h", value))
        wav.writeframes(frames)


class M4BExportTest(unittest.TestCase):
    def test_create_m4b_encodes_aac_chapters_and_cover_art(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_folder = Path(tmp)
            chapter_files = [output_folder / "chapter_1.wav", output_folder / "chapter_2.wav"]
            _write_tone(chapter_files[0], 440)
            _write_tone(chapter_files[1], 660)

            create_index_file("Patch Test", "Audiblez", chapter_files, output_folder)
            create_m4b(chapter_files, "patch-test.epub", _png_bytes(), output_folder)

            output_file = output_folder / "patch-test.m4b"
            self.assertTrue(output_file.exists())
            self.assertFalse((output_folder / "patch-test_wav_list.txt").exists())
            self.assertFalse((output_folder / "cover").exists())

            probe = subprocess.run([
                "ffprobe",
                "-v", "error",
                "-show_format",
                "-show_streams",
                "-show_chapters",
                "-of", "json",
                output_file,
            ], capture_output=True, text=True, check=True)
            info = json.loads(probe.stdout)

            audio_stream = next(s for s in info["streams"] if s["codec_type"] == "audio")
            self.assertEqual(audio_stream["codec_name"], "aac")
            self.assertEqual(audio_stream["sample_rate"], "24000")
            self.assertEqual(audio_stream["channels"], 1)

            cover_stream = next(s for s in info["streams"] if s.get("codec_type") == "video")
            self.assertEqual(cover_stream["codec_name"], "mjpeg")
            self.assertEqual(cover_stream["disposition"]["attached_pic"], 1)

            self.assertEqual(len(info["chapters"]), 2)
            self.assertEqual(info["format"]["tags"]["title"], "Patch Test")
            self.assertEqual(info["format"]["tags"]["artist"], "Audiblez")
            self.assertEqual(info["format"]["tags"]["major_brand"], "M4A ")


if __name__ == "__main__":
    unittest.main()
