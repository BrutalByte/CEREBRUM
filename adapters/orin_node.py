"""
OrinNode — AGX Orin SSH perception bridge for CEREBRUM.

Connects to an NVIDIA AGX Orin (or any Linux edge device) via SSH and exposes
its GPU-accelerated perception capabilities to the CEREBRUM pipeline.

Two usage modes
---------------
**Command mode** (no server required on the Orin):
  Run Whisper or a vision script directly via SSH, upload/download files via
  SFTP, return PerceptionResult objects. Zero setup on the Orin side beyond
  having faster-whisper or a vision CLI installed.

**Tunnel mode** (Orin runs ollama/llama-server/any OpenAI-compatible server):
  SSH port-forward the Orin's inference server to a local port. Existing
  VisionBackend / AudioBackend HTTP clients connect through the tunnel
  transparently — no changes to PerceptionAdapter needed.

Quick start
-----------
    from adapters.orin_node import OrinPerceptionNode
    from adapters.networkx_adapter import NetworkXAdapter
    import networkx as nx

    node = OrinPerceptionNode(host="192.168.1.50", user="nvidia",
                              key_path="~/.ssh/id_rsa")
    node.connect()

    # Command mode — Whisper on a local file (uploads then runs on Orin GPU)
    pa = node.create_perception_adapter(NetworkXAdapter(nx.DiGraph()))
    edges = pa.ingest_audio("meeting.wav", context="project_meeting")

    # Tunnel mode — forward Orin's ollama port and use normal VisionBackend
    with node.open_tunnel(remote_port=11434, local_port=11434):
        edges = pa.ingest_image("frame.jpg")

    node.disconnect()

Install
-------
    pip install "cerebrum-kg-core[orin]"   # adds paramiko>=3.0.0
"""
from __future__ import annotations

import io
import json
import logging
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from adapters.perception_adapter import (
    AudioBackend,
    PerceptionAdapter,
    PerceptionResult,
    VisionBackend,
)

logger = logging.getLogger("cerebrum.orin")

# Default commands run on the Orin.  Override at OrinPerceptionNode() init.
_DEFAULT_WHISPER_CMD = (
    "python3 -m faster_whisper {remote_path} "
    "--model {model} --output_format json --output_dir /tmp/ 2>/dev/null"
    " && cat /tmp/{stem}.json"
)
_DEFAULT_VISION_CMD = (
    "python3 -c \""
    "from transformers import pipeline as P; import sys, json; "
    "p=P('image-to-text', model='{model}'); "
    "r=p('{remote_path}')[0]; "
    "print(json.dumps({{'text': r.get('generated_text','')}}))\" 2>/dev/null"
)


# ---------------------------------------------------------------------------
# OrinNode — low-level SSH/SFTP wrapper
# ---------------------------------------------------------------------------

class OrinNode:
    """
    Low-level SSH connection to an AGX Orin (or any Linux host).

    Wraps paramiko to provide exec_command, SFTP upload/download, and
    SSH port-forwarding as a context manager.

    Parameters
    ----------
    host     : Hostname or IP address (Tailscale address works fine).
    user     : SSH username (usually "nvidia" on JetPack).
    port     : SSH port (default 22).
    key_path : Path to private key file. If None, uses SSH agent / default keys.
    password : Password authentication (prefer key-based instead).
    timeout  : Connection timeout in seconds.
    """

    def __init__(
        self,
        host: str,
        user: str = "nvidia",
        port: int = 22,
        key_path: Optional[str] = None,
        password: Optional[str] = None,
        timeout: float = 10.0,
    ) -> None:
        self._host = host
        self._user = user
        self._port = port
        self._key_path = str(Path(key_path).expanduser()) if key_path else None
        self._password = password
        self._timeout = timeout
        self._client: Any = None   # paramiko.SSHClient
        self._sftp: Any = None     # paramiko.SFTPClient

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open SSH connection to the Orin. Call before any other method."""
        try:
            import paramiko
        except ImportError:
            raise ImportError(
                "pip install 'cerebrum-kg-core[orin]'  # adds paramiko"
            )
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        connect_kwargs: Dict[str, Any] = {
            "hostname": self._host,
            "username": self._user,
            "port": self._port,
            "timeout": self._timeout,
        }
        if self._key_path:
            connect_kwargs["key_filename"] = self._key_path
        if self._password:
            connect_kwargs["password"] = self._password
        client.connect(**connect_kwargs)
        self._client = client
        self._sftp = client.open_sftp()
        logger.info("Connected to Orin at %s@%s:%s", self._user, self._host, self._port)

    def disconnect(self) -> None:
        """Close SSH and SFTP connections."""
        if self._sftp:
            self._sftp.close()
            self._sftp = None
        if self._client:
            self._client.close()
            self._client = None

    def __enter__(self) -> "OrinNode":
        self.connect()
        return self

    def __exit__(self, *_: Any) -> None:
        self.disconnect()

    @property
    def connected(self) -> bool:
        return self._client is not None

    def _require_connected(self) -> None:
        if not self.connected:
            raise RuntimeError("Not connected — call connect() first.")

    # ------------------------------------------------------------------
    # Command execution
    # ------------------------------------------------------------------

    def exec(self, command: str, timeout: float = 60.0) -> Tuple[str, str, int]:
        """
        Run a shell command on the Orin.

        Returns (stdout, stderr, exit_code).
        Raises RuntimeError if the command times out.
        """
        self._require_connected()
        stdin, stdout, stderr = self._client.exec_command(command, timeout=timeout)
        stdout_str = stdout.read().decode(errors="replace")
        stderr_str = stderr.read().decode(errors="replace")
        exit_code = stdout.channel.recv_exit_status()
        return stdout_str, stderr_str, exit_code

    # ------------------------------------------------------------------
    # SFTP helpers
    # ------------------------------------------------------------------

    def upload(self, local_path: str, remote_path: str) -> None:
        """Copy a local file to the Orin."""
        self._require_connected()
        self._sftp.put(local_path, remote_path)

    def download(self, remote_path: str, local_path: str) -> None:
        """Copy a file from the Orin to a local path."""
        self._require_connected()
        self._sftp.get(remote_path, local_path)

    def read_remote(self, remote_path: str) -> bytes:
        """Read a remote file's contents into memory without a local copy."""
        self._require_connected()
        buf = io.BytesIO()
        self._sftp.getfo(remote_path, buf)
        return buf.getvalue()

    def remote_tmp(self, suffix: str = "") -> str:
        """Return a unique /tmp/ path on the Orin."""
        return f"/tmp/cerebrum_{int(time.time() * 1000)}{suffix}"

    def ls(self, remote_dir: str) -> List[str]:
        """List filenames in a remote directory."""
        self._require_connected()
        try:
            return self._sftp.listdir(remote_dir)
        except Exception:
            return []

    def file_exists(self, remote_path: str) -> bool:
        self._require_connected()
        try:
            self._sftp.stat(remote_path)
            return True
        except FileNotFoundError:
            return False

    # ------------------------------------------------------------------
    # SSH port forwarding
    # ------------------------------------------------------------------

    @contextmanager
    def open_tunnel(self, remote_port: int, local_port: int) -> Iterator[None]:
        """
        Forward Orin's remote_port to localhost:local_port.

        Useful for accessing the Orin's ollama/llama-server through a normal
        HTTP backend without exposing the port over the network.

        Usage::

            with node.open_tunnel(remote_port=11434, local_port=11434):
                backend = VisionBackend(endpoint="http://localhost:11434/v1")
                result = backend.perceive("image.jpg")
        """
        self._require_connected()
        transport = self._client.get_transport()
        stop_event = threading.Event()

        def _forward_handler(chan: Any, src_addr: Any, dest_addr: Any) -> None:
            sock = socket.socket()
            try:
                sock.connect(dest_addr)
                while not stop_event.is_set():
                    import select
                    rlist, _, _ = select.select([sock, chan], [], [], 0.1)
                    if sock in rlist:
                        data = sock.recv(1024)
                        if not data:
                            break
                        chan.send(data)
                    if chan in rlist:
                        data = chan.recv(1024)
                        if not data:
                            break
                        sock.send(data)
            finally:
                sock.close()
                chan.close()

        def _accept_loop() -> None:
            while not stop_event.is_set():
                try:
                    chan = transport.accept(0.5)
                    if chan is None:
                        continue
                    src_addr = chan.origin_addr
                    dest_addr = ("127.0.0.1", local_port)
                    t = threading.Thread(
                        target=_forward_handler,
                        args=(chan, src_addr, dest_addr),
                        daemon=True,
                    )
                    t.start()
                except Exception:
                    break

        try:
            transport.request_port_forward("", local_port)
            accept_thread = threading.Thread(target=_accept_loop, daemon=True)
            accept_thread.start()
            logger.info("Tunnel open: localhost:%s → orin:%s", local_port, remote_port)
            yield
        finally:
            stop_event.set()
            try:
                transport.cancel_port_forward("", local_port)
            except Exception:
                pass
            logger.info("Tunnel closed: localhost:%s", local_port)


# ---------------------------------------------------------------------------
# OrinPerceptionNode — perception-specific operations
# ---------------------------------------------------------------------------

class OrinPerceptionNode(OrinNode):
    """
    Extends OrinNode with GPU-accelerated perception on the Orin.

    Runs Whisper and vision inference via SSH commands, uploads/downloads
    files as needed, and returns PerceptionResult objects that plug directly
    into PerceptionAdapter.

    Parameters
    ----------
    whisper_model  : Whisper model to use on the Orin ("large-v3", "medium", etc.)
    vision_model   : HuggingFace model ID for image-to-text (used in command mode).
    vision_cmd     : Override the vision SSH command template.
                     Template vars: {remote_path}, {model}.
    whisper_cmd    : Override the Whisper SSH command template.
                     Template vars: {remote_path}, {model}, {stem}.
    remote_work_dir: Working directory on the Orin for temp files.
    """

    def __init__(
        self,
        host: str,
        user: str = "nvidia",
        port: int = 22,
        key_path: Optional[str] = None,
        password: Optional[str] = None,
        timeout: float = 10.0,
        whisper_model: str = "large-v3",
        vision_model: str = "Salesforce/blip-image-captioning-large",
        vision_cmd: Optional[str] = None,
        whisper_cmd: Optional[str] = None,
        remote_work_dir: str = "/tmp/cerebrum_orin",
    ) -> None:
        super().__init__(host, user, port, key_path, password, timeout)
        self._whisper_model = whisper_model
        self._vision_model = vision_model
        self._vision_cmd = vision_cmd or _DEFAULT_VISION_CMD
        self._whisper_cmd = whisper_cmd or _DEFAULT_WHISPER_CMD
        self._remote_work_dir = remote_work_dir

    def connect(self) -> None:
        super().connect()
        # Ensure work directory exists on the Orin
        self.exec(f"mkdir -p {self._remote_work_dir}")

    # ------------------------------------------------------------------
    # Audio perception (Whisper)
    # ------------------------------------------------------------------

    def perceive_audio(self, remote_path: str) -> PerceptionResult:
        """
        Run Whisper on a file already on the Orin. Returns transcript as PerceptionResult.
        """
        stem = Path(remote_path).stem
        cmd = self._whisper_cmd.format(
            remote_path=remote_path,
            model=self._whisper_model,
            stem=stem,
        )
        stdout, stderr, code = self.exec(cmd, timeout=120.0)
        if code != 0 or not stdout.strip():
            logger.warning("Whisper failed (exit %d): %s", code, stderr[:200])
            return PerceptionResult(
                raw_text="", confidence=0.0,
                provenance=f"audio:whisper-{self._whisper_model}@orin",
                metadata={"error": stderr[:200]},
            )
        try:
            data = json.loads(stdout)
            text = data.get("text", stdout).strip()
            return PerceptionResult(
                raw_text=text,
                confidence=0.85,
                provenance=f"audio:whisper-{self._whisper_model}@orin",
            )
        except json.JSONDecodeError:
            return PerceptionResult(
                raw_text=stdout.strip(),
                confidence=0.75,
                provenance=f"audio:whisper-{self._whisper_model}@orin",
            )

    def perceive_local_audio(self, local_path: str) -> PerceptionResult:
        """Upload a local audio file to the Orin, run Whisper, return transcript."""
        remote_path = f"{self._remote_work_dir}/{Path(local_path).name}"
        self.upload(local_path, remote_path)
        result = self.perceive_audio(remote_path)
        self.exec(f"rm -f {remote_path}")
        return result

    # ------------------------------------------------------------------
    # Vision perception
    # ------------------------------------------------------------------

    def perceive_image(self, remote_path: str) -> PerceptionResult:
        """Run vision inference on a file already on the Orin."""
        cmd = self._vision_cmd.format(
            remote_path=remote_path,
            model=self._vision_model,
        )
        stdout, stderr, code = self.exec(cmd, timeout=60.0)
        if code != 0 or not stdout.strip():
            logger.warning("Vision inference failed (exit %d): %s", code, stderr[:200])
            return PerceptionResult(
                raw_text="", confidence=0.0,
                provenance=f"vision:{self._vision_model}@orin",
                metadata={"error": stderr[:200]},
            )
        try:
            data = json.loads(stdout)
            text = data.get("text", stdout).strip()
        except json.JSONDecodeError:
            text = stdout.strip()
        return PerceptionResult(
            raw_text=text,
            confidence=0.80,
            provenance=f"vision:{self._vision_model}@orin",
        )

    def perceive_local_image(self, local_path: str) -> PerceptionResult:
        """Upload a local image to the Orin, run vision inference, return result."""
        remote_path = f"{self._remote_work_dir}/{Path(local_path).name}"
        self.upload(local_path, remote_path)
        result = self.perceive_image(remote_path)
        self.exec(f"rm -f {remote_path}")
        return result

    # ------------------------------------------------------------------
    # PerceptionAdapter integration
    # ------------------------------------------------------------------

    def create_perception_adapter(
        self,
        graph_adapter: Any,
        pipeline: Any = None,
        confidence_threshold: float = 0.6,
        llm_fn: Any = None,
    ) -> PerceptionAdapter:
        """
        Return a PerceptionAdapter wired to this Orin node (command mode).

        The returned adapter's ingest_audio() and ingest_image() methods
        automatically upload files to the Orin, run inference on its GPU,
        and write the extracted triples into graph_adapter.

        For tunnel mode instead, call open_tunnel() and build VisionBackend /
        AudioBackend pointing at localhost with the tunneled port.
        """
        node = self

        class _OrinAudioBackend(AudioBackend):
            def perceive(self, input_data: Any) -> PerceptionResult:
                path = str(input_data)
                if node.file_exists(path):
                    return node.perceive_audio(path)
                return node.perceive_local_audio(path)

        class _OrinVisionBackend(VisionBackend):
            def perceive(self, input_data: Any) -> PerceptionResult:
                path = str(input_data)
                if node.file_exists(path):
                    return node.perceive_image(path)
                return node.perceive_local_image(path)

        return PerceptionAdapter(
            graph_adapter=graph_adapter,
            pipeline=pipeline,
            vision=_OrinVisionBackend(),
            audio=_OrinAudioBackend(),
            confidence_threshold=confidence_threshold,
            llm_fn=llm_fn,
        )

    # ------------------------------------------------------------------
    # Directory watcher
    # ------------------------------------------------------------------

    def watch_directory(
        self,
        remote_dir: str,
        modality: str = "auto",
        poll_interval: float = 2.0,
    ) -> Iterator[PerceptionResult]:
        """
        Poll remote_dir on the Orin for new files and yield PerceptionResults.

        Designed for sensor drop-folders: camera capture scripts or audio
        recording daemons write files there, and this generator picks them up.

        Usage::

            pa = node.create_perception_adapter(adapter)
            for result in node.watch_directory("/data/orin_camera"):
                pa._process_result(result, context="live_camera")
        """
        from adapters.perception_adapter import _VISION_EXTS, _AUDIO_EXTS
        seen: set = set()

        while True:
            try:
                filenames = self.ls(remote_dir)
                for name in sorted(filenames):
                    if name in seen:
                        continue
                    seen.add(name)
                    remote_path = f"{remote_dir}/{name}"
                    ext = Path(name).suffix.lower()

                    effective_modality = modality
                    if effective_modality == "auto":
                        if ext in _VISION_EXTS:
                            effective_modality = "vision"
                        elif ext in _AUDIO_EXTS:
                            effective_modality = "audio"
                        else:
                            effective_modality = "skip"

                    if effective_modality == "vision":
                        yield self.perceive_image(remote_path)
                    elif effective_modality == "audio":
                        yield self.perceive_audio(remote_path)
            except Exception as exc:
                logger.warning("watch_directory error: %s", exc)

            time.sleep(poll_interval)

    def watch_as_stream_source(
        self,
        remote_dir: str,
        modality: str = "auto",
        poll_interval: float = 2.0,
        context: str = "",
    ) -> "OrinStreamSource":
        """
        Return a StreamSource that watches remote_dir and yields StreamEvents.
        Plug into StreamAdapter.add_source() for live CEREBRUM ingestion.
        """
        return OrinStreamSource(
            node=self,
            remote_dir=remote_dir,
            modality=modality,
            poll_interval=poll_interval,
            context=context,
        )


# ---------------------------------------------------------------------------
# OrinStreamSource — StreamAdapter-compatible source
# ---------------------------------------------------------------------------

class OrinStreamSource:
    """
    Watches a directory on the Orin for new files and emits StreamEvents.

    Drop-in for StreamAdapter.add_source().  Internally calls
    OrinPerceptionNode.watch_directory() in a generator loop and converts
    PerceptionResult → StreamEvent via triple extraction.

    Usage::

        node = OrinPerceptionNode(host="192.168.1.50")
        node.connect()

        stream_adapter = StreamAdapter(time_window_seconds=300)
        stream_adapter.add_source(
            node.watch_as_stream_source("/data/captures", context="lab_camera")
        )
        stream_adapter.start()
    """

    def __init__(
        self,
        node: OrinPerceptionNode,
        remote_dir: str,
        modality: str = "auto",
        poll_interval: float = 2.0,
        context: str = "",
    ) -> None:
        self._node = node
        self._remote_dir = remote_dir
        self._modality = modality
        self._poll_interval = poll_interval
        self._context = context
        self._running = True

    def read(self) -> Iterator[Any]:
        from core.stream_engine import StreamEvent
        from adapters.perception_adapter import _extract_triples_heuristic, _extract_triples_llm

        for result in self._node.watch_directory(
            self._remote_dir, self._modality, self._poll_interval
        ):
            if not self._running:
                break
            if not result.raw_text:
                continue
            triples = _extract_triples_heuristic(result.raw_text)
            for subj, rel, obj in triples:
                yield StreamEvent(
                    source=subj,
                    relation=rel,
                    target=obj,
                    timestamp=time.time(),
                    metadata={
                        "provenance": result.provenance,
                        "orin_dir": self._remote_dir,
                    },
                )

    def stop(self) -> None:
        self._running = False

    def __repr__(self) -> str:
        return f"OrinStreamSource(host={self._node._host}, dir={self._remote_dir})"
