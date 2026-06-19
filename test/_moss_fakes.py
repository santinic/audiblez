"""Shared test doubles for the MOSS resident pipe co-process.

``FakeMossProcess`` speaks the "MOSS resident pipe v1" wire protocol
(docs/moss-coprocess-spec.md) so tests can exercise the Python engine
machinery (request/response round-trips, restart, circuit-breaker) without a
real llama.cpp binary, GGUF weights, or a GPU.

Usage — basic round-trip::

    fake = FakeMossProcess()
    fake.start()
    req = make_synth_request(0, "Hello world.", "/tmp/test-0.wav")
    fake.send(req)
    resp = fake.recv()
    assert resp["status"] == "ok"
    fake.shutdown()

Usage — scripted failure modes::

    fake = FakeMossProcess(error_texts={"poison"}, max_errors_before_death=0)
    # → returns status:error for the text "poison" (child stays alive)

    fake = FakeMossProcess(die_after=1)
    # → closes stdout after the first response (simulates child death / EOF)

    fake = FakeMossProcess(hang_texts={"slow"})
    # → never writes a response for that text (simulates timeout)

    fake = FakeMossProcess(poison_pill_texts={"boom"}, poison_deaths=2)
    # → closes stdout (dies) on every request for "boom", up to poison_deaths times,
    #   then stays alive (so the caller can count K consecutive deaths)

Wire protocol summary (from docs/moss-coprocess-spec.md):
    REQUEST  (parent→child, stdin):
        {"v":1, "id":N, "op":"synth", "text":"...", "wav_out":"/abs/path.wav",
         "seed":S, "sampling":{...}, "max_new_tokens":2048}
    OK       (child→parent, stdout):
        {"v":1, "id":N, "status":"ok", "wav_out":"...", "frames":F,
         "sample_rate":24000}
    ERROR    (child→parent, stdout):
        {"v":1, "id":N, "status":"error", "code":"gen_failed|empty_audio|bad_request",
         "message":"..."}
    SHUTDOWN (parent→child):
        {"op":"shutdown"} → {"status":"bye"} then child exits 0
"""

import io
import json
import os
import struct
import tempfile
import threading
import time
from typing import Any

__all__ = [
    "FakeMossProcess",
    "FakeMossAdapter",
    "make_synth_request",
    "make_ok_response",
    "make_error_response",
    "SAMPLING_DEFAULTS",
    "SAMPLE_RATE",
    "fake_wav_reader",
]

# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------

SAMPLE_RATE: int = 24_000
PROTOCOL_VERSION: int = 1

#: Sampling param defaults from the spec (pin these in the cache key).
SAMPLING_DEFAULTS: dict[str, float] = {
    "text_temperature": 1.5,
    "text_top_k": 50,
    "audio_temperature": 1.7,
    "audio_top_p": 0.8,
    "audio_top_k": 25,
    "audio_repetition_penalty": 1.0,
}

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def make_synth_request(
    request_id: int,
    text: str,
    wav_out: str,
    seed: int = 42,
    sampling: dict[str, float] | None = None,
    max_new_tokens: int = 2048,
) -> dict[str, Any]:
    """Return a well-formed synth REQUEST dict (parent→child)."""
    return {
        "v": PROTOCOL_VERSION,
        "id": request_id,
        "op": "synth",
        "text": text,
        "wav_out": wav_out,
        "seed": seed,
        "sampling": sampling if sampling is not None else dict(SAMPLING_DEFAULTS),
        "max_new_tokens": max_new_tokens,
    }


def make_ok_response(
    request_id: int,
    wav_out: str,
    frames: int = 24_000,
) -> dict[str, Any]:
    """Return a well-formed OK response dict (child→parent)."""
    return {
        "v": PROTOCOL_VERSION,
        "id": request_id,
        "status": "ok",
        "wav_out": wav_out,
        "frames": frames,
        "sample_rate": SAMPLE_RATE,
    }


def make_error_response(
    request_id: int,
    code: str = "gen_failed",
    message: str = "synthesis failed",
) -> dict[str, Any]:
    """Return a well-formed ERROR response dict (child→parent).

    ``code`` must be one of: ``gen_failed``, ``empty_audio``, ``bad_request``.
    """
    return {
        "v": PROTOCOL_VERSION,
        "id": request_id,
        "status": "error",
        "code": code,
        "message": message,
    }


# ---------------------------------------------------------------------------
# Minimal 16-bit WAV writer (stdlib only)
# ---------------------------------------------------------------------------

def _write_minimal_wav(path: str, n_frames: int = 24_000, sample_rate: int = SAMPLE_RATE) -> None:
    """Write a silent 16-bit mono WAV to *path*.

    Uses only stdlib ``struct`` — no soundfile / numpy dependency.
    ``n_frames`` defaults to 1 second at 24 kHz.
    """
    n_channels = 1
    bits_per_sample = 16
    byte_rate = sample_rate * n_channels * bits_per_sample // 8
    block_align = n_channels * bits_per_sample // 8
    data_size = n_frames * block_align
    with open(path, "wb") as f:
        # RIFF header
        f.write(b"RIFF")
        f.write(struct.pack("<I", 36 + data_size))  # file size - 8
        f.write(b"WAVE")
        # fmt chunk
        f.write(b"fmt ")
        f.write(struct.pack("<I", 16))              # chunk size
        f.write(struct.pack("<H", 1))               # PCM
        f.write(struct.pack("<H", n_channels))
        f.write(struct.pack("<I", sample_rate))
        f.write(struct.pack("<I", byte_rate))
        f.write(struct.pack("<H", block_align))
        f.write(struct.pack("<H", bits_per_sample))
        # data chunk
        f.write(b"data")
        f.write(struct.pack("<I", data_size))
        f.write(b"\x00" * data_size)


# ---------------------------------------------------------------------------
# FakeMossProcess
# ---------------------------------------------------------------------------

class FakeMossProcess:
    """In-process fake that mimics the MOSS resident child's I/O contract.

    The fake runs a background thread (``_serve``) that reads JSON lines from
    ``self.stdin`` and writes JSON lines to ``self.stdout``, just as the real
    child process would.  Tests interact via ``send`` / ``recv``.

    Failure modes (scriptable via constructor kwargs):
    - ``error_texts``: set of strings → respond with status:error (child healthy)
    - ``die_after``: int → close stdout (EOF) after this many successful responses
    - ``hang_texts``: set of strings → never write a response (caller times out)
    - ``poison_pill_texts``: set of strings → close stdout on *each* matching
      request, up to ``poison_deaths`` times total, then resume normally
    - ``spawn_fails``: bool → raise OSError from ``start()`` (simulates Popen fail)

    Attributes mirroring subprocess.Popen (read by the real engine):
    - ``stdin``  — writable pipe end (parent writes requests here)
    - ``stdout`` — readable pipe end (parent reads responses here)
    - ``stderr`` — /dev/null (stderr drain satisfied; no real logs needed)
    - ``poll()`` — returns None while alive, returncode once dead
    - ``kill()`` / ``terminate()`` — mark the fake as dead
    """

    def __init__(
        self,
        *,
        error_texts: set[str] | None = None,
        die_after: int | None = None,
        hang_texts: set[str] | None = None,
        poison_pill_texts: set[str] | None = None,
        poison_deaths: int = 2,
        spawn_fails: bool = False,
        wav_frames: int = 24_000,
        emit_ready: bool = False,
    ) -> None:
        # emit_ready=True: write {"status":"ready"} immediately on startup so that
        # MossProcess._wait_ready() (which expects that line before any request) passes.
        self._emit_ready: bool = emit_ready
        self._error_texts: set[str] = error_texts or set()
        self._die_after: int | None = die_after
        self._hang_texts: set[str] = hang_texts or set()
        self._poison_pill_texts: set[str] = poison_pill_texts or set()
        self._poison_deaths_remaining: int = poison_deaths
        self._spawn_fails: bool = spawn_fails
        self._wav_frames: int = wav_frames

        # Popen-compatible pipes (in-memory)
        self._parent_w, self._child_r = _make_pipe()   # parent writes → child reads
        self._child_w, self._parent_r = _make_pipe()   # child writes → parent reads

        # Public attributes that look like subprocess.Popen
        self.stdin = self._parent_w    # parent writes requests here
        self.stdout = self._parent_r   # parent reads responses here
        self.stderr = open(os.devnull, "rb")  # satisfy drain; safe to ignore
        self.returncode: int | None = None

        self._alive: bool = False
        self._thread: threading.Thread | None = None
        self._responses_sent: int = 0
        self._tmpfiles: list[str] = []
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Popen-compatible interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Spawn the background serve thread (analogous to Popen succeeding)."""
        if self._spawn_fails:
            raise OSError("FakeMossProcess: spawn_fails=True")
        self._alive = True
        self._thread = threading.Thread(target=self._serve, daemon=True, name="FakeMoss")
        self._thread.start()

    def poll(self) -> int | None:
        """Return None while alive, 0 on clean exit, 1 on kill."""
        return self.returncode

    def kill(self) -> None:
        """Terminate immediately (non-zero returncode)."""
        with self._lock:
            self._alive = False
            self.returncode = 1
        # Unblock any blocking reads on the parent side
        try:
            self._child_w.close()
        except Exception:
            pass

    def terminate(self) -> None:
        """Graceful termination (same as kill in the fake)."""
        self.kill()

    def wait(self, timeout: float | None = None) -> int | None:
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        return self.returncode

    # ------------------------------------------------------------------
    # Convenience helpers for tests
    # ------------------------------------------------------------------

    def send(self, request: dict[str, Any]) -> None:
        """Serialize *request* and write one JSON line to the fake's stdin."""
        line = (json.dumps(request) + "\n").encode()
        self.stdin.write(line)
        self.stdin.flush()

    def recv(self, timeout: float = 5.0) -> dict[str, Any] | None:
        """Read and deserialize one JSON response line from the fake's stdout.

        Returns None if EOF is reached before a line arrives (child died).
        Raises TimeoutError if *timeout* seconds elapse with no data.
        """
        result: dict[str, Any] | None = None
        done = threading.Event()
        exc_holder: list[Exception] = []

        def _read() -> None:
            try:
                line = self.stdout.readline()
                if line:
                    nonlocal result
                    result = json.loads(line.decode().strip())
            except Exception as e:
                exc_holder.append(e)
            finally:
                done.set()

        t = threading.Thread(target=_read, daemon=True)
        t.start()
        if not done.wait(timeout):
            raise TimeoutError(f"FakeMossProcess: no response within {timeout}s")
        if exc_holder:
            raise exc_holder[0]
        return result

    def shutdown(self) -> dict[str, Any] | None:
        """Send shutdown request and read the bye response."""
        try:
            self.send({"op": "shutdown"})
            return self.recv(timeout=2.0)
        except Exception:
            return None
        finally:
            self.kill()

    def close_pipes(self) -> None:
        """Close all pipe file objects held by the parent side.

        Call this in tearDown after kill()/shutdown() to suppress
        ResourceWarning about unclosed files in tests.
        """
        for fobj in (self.stdin, self.stdout, self.stderr,
                     self._child_r, self._child_w):
            try:
                fobj.close()
            except Exception:
                pass

    def cleanup(self) -> None:
        """Remove any temporary WAV files written during the session."""
        for path in self._tmpfiles:
            try:
                os.unlink(path)
            except OSError:
                pass
        self._tmpfiles.clear()

    # ------------------------------------------------------------------
    # Internal serve loop (runs in background thread)
    # ------------------------------------------------------------------

    def _serve(self) -> None:
        try:
            if self._emit_ready:
                # MossProcess._wait_ready() reads lines until it sees {"status":"ready"}.
                self._write_line(json.dumps({"status": "ready"}))
            for raw_line in self._child_r:
                with self._lock:
                    if not self._alive:
                        break
                try:
                    req = json.loads(raw_line.decode().strip())
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

                op = req.get("op")
                if op == "shutdown":
                    resp = {"status": "bye"}
                    self._write_line(json.dumps(resp))
                    break

                if op != "synth":
                    # unknown op — send bad_request error
                    err = make_error_response(
                        req.get("id", -1), code="bad_request", message=f"unknown op: {op}"
                    )
                    self._write_line(json.dumps(err))
                    continue

                req_id = req.get("id", -1)
                text = req.get("text", "")
                wav_out = req.get("wav_out", "")

                # --- poison pill: die (close stdout) ---
                if text in self._poison_pill_texts:
                    with self._lock:
                        if self._poison_deaths_remaining > 0:
                            self._poison_deaths_remaining -= 1
                            self._alive = False
                            self.returncode = 1
                    # Close child write end so parent sees EOF
                    try:
                        self._child_w.close()
                    except Exception:
                        pass
                    break

                # --- hang: never respond ---
                if text in self._hang_texts:
                    # block indefinitely (test must impose a timeout)
                    time.sleep(9999)
                    continue

                # --- error: respond with status:error, child stays alive ---
                if text in self._error_texts:
                    err = make_error_response(req_id, code="gen_failed",
                                              message="fake error for text")
                    self._write_line(json.dumps(err))
                    continue

                # --- die_after: close stdout after N successful responses ---
                if self._die_after is not None and self._responses_sent >= self._die_after:
                    with self._lock:
                        self._alive = False
                        self.returncode = 1
                    try:
                        self._child_w.close()
                    except Exception:
                        pass
                    break

                # --- normal: write WAV and OK response ---
                wav_path = wav_out if wav_out else self._make_tmp_wav()
                _write_minimal_wav(wav_path, n_frames=self._wav_frames)
                ok = make_ok_response(req_id, wav_out=wav_path, frames=self._wav_frames)
                self._write_line(json.dumps(ok))
                with self._lock:
                    self._responses_sent += 1

        except Exception:
            pass
        finally:
            with self._lock:
                self._alive = False
                if self.returncode is None:
                    self.returncode = 0
            try:
                self._child_w.close()
            except Exception:
                pass

    def _write_line(self, line: str) -> None:
        try:
            self._child_w.write((line + "\n").encode())
            self._child_w.flush()
        except (BrokenPipeError, OSError):
            pass

    def _make_tmp_wav(self) -> str:
        fd, path = tempfile.mkstemp(suffix=".wav", prefix="fakemoss_")
        os.close(fd)
        self._tmpfiles.append(path)
        return path


# ---------------------------------------------------------------------------
# Pipe helpers
# ---------------------------------------------------------------------------

def _make_pipe() -> tuple[io.RawIOBase, io.RawIOBase]:
    """Return a (write_end, read_end) pair of binary pipe objects."""
    r_fd, w_fd = os.pipe()
    return (
        open(w_fd, "wb", buffering=0),   # write end
        open(r_fd, "rb", buffering=0),   # read end
    )


# ---------------------------------------------------------------------------
# WAV reader compatible with MossProcess._wav_reader injection
# ---------------------------------------------------------------------------

def fake_wav_reader(path: str) -> tuple[Any, int]:
    """Read a minimal 16-bit mono WAV written by FakeMossProcess.

    Returns (float32_array, sample_rate) matching the soundfile.read interface
    that MossProcess uses, but without requiring the soundfile package.
    Suitable for injection as MossProcess(wav_reader=fake_wav_reader).
    """
    import array as _array
    with open(path, "rb") as f:
        f.seek(40)                                   # data-chunk size
        data_size = struct.unpack("<I", f.read(4))[0]
        raw = f.read(data_size)
    shorts = _array.array("h", raw)                  # signed int16
    floats = [s / 32768.0 for s in shorts]           # int16 → float32
    import numpy as _np
    return _np.array(floats, dtype=_np.float32), SAMPLE_RATE


# ---------------------------------------------------------------------------
# MossProcess-duck-typed adapter for integration tests
# ---------------------------------------------------------------------------

class FakeMossAdapter:
    """Wraps FakeMossProcess to expose the MossProcess public interface.

    MossProcess interface used by the synth closure:
      .synth(text, timeout) -> np.ndarray (1-D float32 @ 24 kHz)
      .start() -> self
      .close()
      .restart() -> self
      .alive() -> bool
      .spawns (int) — incremented on each start()/restart()

    This lets ``_build_llamacpp_synth(moss_factory=lambda: FakeMossAdapter(...))``
    work without any subprocess or GPU.

    The ``_wav_reader`` is wired to ``fake_wav_reader`` automatically so the float32
    conversion path inside MossProcess._handle_response is exercised with the same
    code, but reading the fake's struct-written WAVs instead of real soundfile output.

    Usage in integration tests::

        def moss_factory():
            return FakeMossAdapter(error_texts={"bad"})

        synth = _build_llamacpp_synth(
            voice="af_sky", lang_code="a",
            moss_factory=moss_factory,
            work_dir=tmp_dir,
        )
    """

    def __init__(self, **fake_kwargs: Any) -> None:
        # Do NOT set emit_ready here: FakeMossAdapter handles its own protocol directly
        # and does not go through MossProcess._wait_ready() — the ready handshake is a
        # MossProcess detail not relevant to FakeMossAdapter.synth().
        fake_kwargs.pop("emit_ready", None)   # harmless if passed in by mistake
        self._kwargs = fake_kwargs
        self._proc: FakeMossProcess | None = None
        self.spawns: int = 0
        self._work_dir: str = tempfile.mkdtemp(prefix="fakemoss_adapter_")
        self._tmpfiles: list[str] = []
        self._req_id: int = 0
        self._lock = threading.Lock()

    # -- MossProcess-compatible lifecycle --
    # _build_llamacpp_synth calls: factory() → start() → [synth loop] → restart()/close()
    # factory() must return an un-started adapter; start() launches the fake.

    def start(self) -> "FakeMossAdapter":
        """Launch a fresh FakeMossProcess.  Raises MossRunAborted when spawn_fails=True,
        mirroring MossProcess.start() which wraps OSError → MossRunAborted."""
        if self._kwargs.get("spawn_fails"):
            try:
                from audiblez.core import MossRunAborted
            except ImportError as exc:
                raise OSError("FakeMossAdapter: spawn_fails=True (MossRunAborted not available)") from exc
            raise MossRunAborted("FakeMossAdapter: spawn_fails=True")
        fake = FakeMossProcess(**self._kwargs)
        fake.start()
        with self._lock:
            self._proc = fake
            self.spawns += 1
        return self

    def restart(self) -> "FakeMossAdapter":
        self.close()
        return self.start()

    def alive(self) -> bool:
        proc = self._proc
        return proc is not None and proc.returncode is None

    def close(self) -> None:
        with self._lock:
            proc = self._proc
            self._proc = None
        if proc is None:
            return
        try:
            if proc.returncode is None:
                proc.shutdown()
        except Exception:
            pass
        try:
            proc.kill()
            proc.close_pipes()
            proc.cleanup()
        except Exception:
            pass

    def synth(self, text: str, timeout: float) -> Any:
        """Issue one synth request; mirrors MossProcess.synth() contract.

        Returns a 1-D float32 numpy array @ 24 kHz.
        Raises MossError / MossPermanentError / MossDeath (imported lazily
        from audiblez.core so this module stays import-light until needed).
        """
        import numpy as _np
        try:
            from audiblez.core import MossError, MossPermanentError, MossDeath
        except ImportError:
            MossError = Exception  # type: ignore[assignment,misc]
            MossPermanentError = Exception  # type: ignore[assignment,misc]
            MossDeath = Exception  # type: ignore[assignment,misc]

        proc = self._proc
        if proc is None or proc.returncode is not None:
            raise MossDeath("FakeMossAdapter: child not running")  # type: ignore[misc]

        with self._lock:
            self._req_id += 1
            req_id = self._req_id

        wav_out = os.path.join(self._work_dir, f"moss-req-{req_id}.wav")
        if os.path.exists(wav_out):
            os.unlink(wav_out)

        req = make_synth_request(req_id, text, wav_out)
        proc.send(req)
        try:
            resp = proc.recv(timeout=timeout)
        except TimeoutError as exc:
            raise MossDeath(  # type: ignore[misc]
                f"FakeMossAdapter: timeout on request {req_id}") from exc

        if resp is None:
            # EOF: the child died (die_after, poison pill, etc.)
            with self._lock:
                self._proc = None
            raise MossDeath(f"FakeMossAdapter: EOF / child dead on request {req_id}")  # type: ignore[misc]

        status = resp.get("status")
        if status == "error":
            code = resp.get("code", "gen_failed")
            msg = resp.get("message", "")
            if code in ("bad_request", "empty_audio"):
                raise MossPermanentError(f"MOSS {code}: {msg}")  # type: ignore[misc]
            raise MossError(f"MOSS {code}: {msg}")  # type: ignore[misc]

        if status != "ok":
            raise MossError(f"unexpected status {status!r}")  # type: ignore[misc]

        frames = resp.get("frames", 0)
        if not frames or frames <= 0:
            raise MossPermanentError("MOSS reported ok with no frames")  # type: ignore[misc]

        if not os.path.exists(wav_out):
            raise MossDeath(f"FakeMossAdapter: wav_out missing after ok: {wav_out}")  # type: ignore[misc]

        audio, _sr = fake_wav_reader(wav_out)
        audio = _np.asarray(audio, dtype=_np.float32).reshape(-1)
        if audio.size == 0:
            raise MossPermanentError("MOSS wav decoded to empty audio")  # type: ignore[misc]

        return audio

    def cleanup(self) -> None:
        """Close adapter, close any lingering FakeMossProcess pipes, and remove temp dir."""
        self.close()
        import shutil
        shutil.rmtree(self._work_dir, ignore_errors=True)

    def __del__(self) -> None:
        """Best-effort cleanup on GC to suppress ResourceWarning on unclosed pipe fds."""
        try:
            self.close()
        except Exception:
            pass
