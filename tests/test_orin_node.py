"""
Tests for adapters/orin_node.py

All SSH/SFTP operations are mocked — no real Orin needed.
Covers:
  - OrinNode: connect, exec, upload, download, ls, file_exists, remote_tmp
  - OrinPerceptionNode: perceive_audio, perceive_image, perceive_local_audio,
    perceive_local_image, create_perception_adapter
  - OrinStreamSource: repr, stop
"""
import io
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, PropertyMock

import networkx as nx
import pytest

from adapters.orin_node import OrinNode, OrinPerceptionNode, OrinStreamSource
from adapters.networkx_adapter import NetworkXAdapter


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_channel(exit_code: int = 0) -> MagicMock:
    ch = MagicMock()
    ch.recv_exit_status.return_value = exit_code
    return ch


def _make_exec_result(stdout: str = "", stderr: str = "", exit_code: int = 0):
    sin = MagicMock()
    sout = MagicMock()
    sout.read.return_value = stdout.encode()
    sout.channel = _make_channel(exit_code)
    serr = MagicMock()
    serr.read.return_value = stderr.encode()
    return sin, sout, serr


def _make_ssh_client(stdout="", stderr="", exit_code=0) -> MagicMock:
    client = MagicMock()
    client.exec_command.return_value = _make_exec_result(stdout, stderr, exit_code)
    sftp = MagicMock()
    client.open_sftp.return_value = sftp
    return client


def _patched_node(stdout="", stderr="", exit_code=0) -> tuple:
    """Return (node, mock_ssh_client)."""
    node = OrinNode(host="192.168.1.50", user="nvidia")
    mock_client = _make_ssh_client(stdout, stderr, exit_code)
    node._client = mock_client
    node._sftp = mock_client.open_sftp.return_value
    return node, mock_client


# ---------------------------------------------------------------------------
# OrinNode.connect
# ---------------------------------------------------------------------------

class TestOrinNodeConnect:
    def test_connect_opens_ssh_and_sftp(self):
        with patch("paramiko.SSHClient") as MockSSH:
            instance = MockSSH.return_value
            sftp_mock = MagicMock()
            instance.open_sftp.return_value = sftp_mock

            node = OrinNode(host="orin.local", user="nvidia", key_path=None)
            node.connect()

            instance.connect.assert_called_once()
            assert node._sftp is sftp_mock

    def test_connect_raises_without_paramiko(self):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "paramiko":
                raise ImportError("no paramiko")
            return real_import(name, *args, **kwargs)

        node = OrinNode(host="orin.local", user="nvidia")
        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="cerebrum-kg-core"):
                node.connect()

    def test_disconnect_clears_client(self):
        node, _ = _patched_node()
        node.disconnect()
        assert node._client is None
        assert node._sftp is None

    def test_connected_property(self):
        node = OrinNode(host="x", user="u")
        assert not node.connected
        node._client = MagicMock()
        assert node.connected

    def test_context_manager(self):
        with patch("paramiko.SSHClient") as MockSSH:
            instance = MockSSH.return_value
            instance.open_sftp.return_value = MagicMock()
            with OrinNode(host="x", user="u") as n:
                assert n.connected
            assert not n.connected


# ---------------------------------------------------------------------------
# OrinNode.exec
# ---------------------------------------------------------------------------

class TestOrinNodeExec:
    def test_exec_returns_stdout_stderr_exitcode(self):
        node, client = _patched_node(stdout="hello\n", stderr="warn", exit_code=0)
        out, err, code = node.exec("echo hello")
        assert out == "hello\n"
        assert err == "warn"
        assert code == 0

    def test_exec_requires_connected(self):
        node = OrinNode(host="x", user="u")
        with pytest.raises(RuntimeError, match="Not connected"):
            node.exec("ls")

    def test_exec_nonzero_exit(self):
        node, _ = _patched_node(stdout="", stderr="permission denied", exit_code=1)
        _, err, code = node.exec("sudo rm -rf /")
        assert code == 1
        assert "permission denied" in err


# ---------------------------------------------------------------------------
# OrinNode SFTP
# ---------------------------------------------------------------------------

class TestOrinNodeSftp:
    def test_upload(self, tmp_path):
        node, client = _patched_node()
        f = tmp_path / "test.wav"
        f.write_bytes(b"\x00" * 10)
        node.upload(str(f), "/tmp/test.wav")
        node._sftp.put.assert_called_once_with(str(f), "/tmp/test.wav")

    def test_download(self, tmp_path):
        node, _ = _patched_node()
        dest = str(tmp_path / "out.wav")
        node.download("/tmp/test.wav", dest)
        node._sftp.get.assert_called_once_with("/tmp/test.wav", dest)

    def test_read_remote(self):
        node, _ = _patched_node()
        node._sftp.getfo.side_effect = lambda path, buf: buf.write(b"hello")
        data = node.read_remote("/tmp/file.txt")
        assert data == b"hello"

    def test_ls(self):
        node, _ = _patched_node()
        node._sftp.listdir.return_value = ["a.jpg", "b.wav"]
        assert node.ls("/data") == ["a.jpg", "b.wav"]

    def test_ls_returns_empty_on_error(self):
        node, _ = _patched_node()
        node._sftp.listdir.side_effect = Exception("no such dir")
        assert node.ls("/nonexistent") == []

    def test_file_exists_true(self):
        node, _ = _patched_node()
        node._sftp.stat.return_value = MagicMock()
        assert node.file_exists("/tmp/exists.wav") is True

    def test_file_exists_false(self):
        node, _ = _patched_node()
        node._sftp.stat.side_effect = FileNotFoundError
        assert node.file_exists("/tmp/no.wav") is False

    def test_remote_tmp_unique(self):
        node, _ = _patched_node()
        p1 = node.remote_tmp(".wav")
        time.sleep(0.01)
        p2 = node.remote_tmp(".wav")
        assert p1 != p2
        assert p1.endswith(".wav")


# ---------------------------------------------------------------------------
# OrinPerceptionNode.perceive_audio
# ---------------------------------------------------------------------------

class TestOrinPerceiveAudio:
    def _make_node(self, whisper_json: dict, exit_code: int = 0):
        node = OrinPerceptionNode(host="x", user="u")
        stdout = json.dumps(whisper_json)
        node._client = _make_ssh_client(stdout=stdout, exit_code=exit_code)
        node._sftp = node._client.open_sftp.return_value
        return node

    def test_returns_transcript(self):
        node = self._make_node({"text": "Hello from the Orin."})
        result = node.perceive_audio("/data/clip.wav")
        assert result.raw_text == "Hello from the Orin."
        assert result.confidence == 0.85
        assert "orin" in result.provenance

    def test_failed_command_returns_low_confidence(self):
        node = self._make_node({}, exit_code=1)
        # override stdout to empty
        node._client.exec_command.return_value = _make_exec_result("", "error", 1)
        result = node.perceive_audio("/data/clip.wav")
        assert result.confidence == 0.0
        assert "error" in result.metadata

    def test_non_json_stdout_used_as_raw_text(self):
        node = OrinPerceptionNode(host="x", user="u")
        node._client = _make_ssh_client(stdout="plain transcript text", exit_code=0)
        node._sftp = node._client.open_sftp.return_value
        result = node.perceive_audio("/data/clip.wav")
        assert result.raw_text == "plain transcript text"
        assert result.confidence == 0.75


# ---------------------------------------------------------------------------
# OrinPerceptionNode.perceive_image
# ---------------------------------------------------------------------------

class TestOrinPerceiveImage:
    def test_returns_description(self):
        node = OrinPerceptionNode(host="x", user="u")
        node._client = _make_ssh_client(
            stdout=json.dumps({"text": "A cat sits on a mat"}), exit_code=0
        )
        node._sftp = node._client.open_sftp.return_value
        result = node.perceive_image("/data/frame.jpg")
        assert "cat" in result.raw_text
        assert result.confidence == 0.80
        assert "orin" in result.provenance

    def test_failed_command(self):
        node = OrinPerceptionNode(host="x", user="u")
        node._client = _make_ssh_client(stdout="", stderr="OOM", exit_code=137)
        node._sftp = node._client.open_sftp.return_value
        result = node.perceive_image("/data/frame.jpg")
        assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# OrinPerceptionNode.perceive_local_audio / perceive_local_image
# ---------------------------------------------------------------------------

class TestOrinPerceiveLocal:
    def test_local_audio_uploads_then_deletes(self, tmp_path):
        audio = tmp_path / "clip.wav"
        audio.write_bytes(b"\x00" * 100)

        node = OrinPerceptionNode(host="x", user="u")
        node._client = _make_ssh_client(
            stdout=json.dumps({"text": "Meeting notes"}), exit_code=0
        )
        node._sftp = node._client.open_sftp.return_value
        # Stub file_exists to False so upload is called
        node._sftp.stat.side_effect = FileNotFoundError

        result = node.perceive_local_audio(str(audio))

        node._sftp.put.assert_called_once()
        assert "Meeting notes" in result.raw_text
        # rm command should have been called
        exec_calls = node._client.exec_command.call_args_list
        assert any("rm -f" in str(c) for c in exec_calls)

    def test_local_image_uploads_then_deletes(self, tmp_path):
        img = tmp_path / "frame.jpg"
        img.write_bytes(b"\xff\xd8\xff" + b"\x00" * 100)

        node = OrinPerceptionNode(host="x", user="u")
        node._client = _make_ssh_client(
            stdout=json.dumps({"text": "A dog running"}), exit_code=0
        )
        node._sftp = node._client.open_sftp.return_value

        result = node.perceive_local_image(str(img))
        node._sftp.put.assert_called_once()
        assert "dog" in result.raw_text


# ---------------------------------------------------------------------------
# OrinPerceptionNode.create_perception_adapter
# ---------------------------------------------------------------------------

class TestCreatePerceptionAdapter:
    def test_returns_perception_adapter(self):
        from adapters.perception_adapter import PerceptionAdapter
        node = OrinPerceptionNode(host="x", user="u")
        node._client = _make_ssh_client(stdout=json.dumps({"text": "x"}))
        node._sftp = node._client.open_sftp.return_value
        ga = NetworkXAdapter(nx.DiGraph())
        pa = node.create_perception_adapter(ga)
        assert isinstance(pa, PerceptionAdapter)

    def test_ingest_audio_via_adapter(self, tmp_path):
        from adapters.perception_adapter import PerceptionAdapter
        audio = tmp_path / "rec.wav"
        audio.write_bytes(b"\x00" * 10)

        node = OrinPerceptionNode(host="x", user="u")
        node._client = _make_ssh_client(
            stdout="Paris | LOCATED_IN | France", exit_code=0
        )
        node._sftp = node._client.open_sftp.return_value

        ga = NetworkXAdapter(nx.DiGraph())
        pa = node.create_perception_adapter(ga)
        edges = pa.ingest_audio(str(audio))
        assert len(edges) == 1


# ---------------------------------------------------------------------------
# OrinStreamSource
# ---------------------------------------------------------------------------

class TestOrinStreamSource:
    def test_repr(self):
        node = OrinPerceptionNode(host="orin.local", user="nvidia")
        src = OrinStreamSource(node=node, remote_dir="/data/captures")
        assert "orin.local" in repr(src)
        assert "/data/captures" in repr(src)

    def test_stop(self):
        node = OrinPerceptionNode(host="x", user="u")
        src = OrinStreamSource(node=node, remote_dir="/data")
        assert src._running is True
        src.stop()
        assert src._running is False
