"""Test suite for npc_sysenv module - system environment utilities."""

import os
import tempfile
import shutil
import pytest


class TestCheckInternetConnection:
    """Test internet connectivity check."""

    def test_check_internet_connection_returns_bool(self):
        """Test that check_internet_connection returns boolean"""
        from npcpy.npc_sysenv import check_internet_connection

        result = check_internet_connection(timeout=2)
        assert isinstance(result, bool)

    def test_check_internet_connection_with_short_timeout(self):
        """Test with very short timeout"""
        from npcpy.npc_sysenv import check_internet_connection

        # Very short timeout might fail, but should not raise
        result = check_internet_connection(timeout=0.001)
        assert isinstance(result, bool)


class TestGetLocallyAvailableModels:
    """Test model availability detection."""

    def test_get_locally_available_models_empty_dir(self):
        """Test with empty directory (no .env file)"""
        from npcpy.npc_sysenv import get_locally_available_models

        temp_dir = tempfile.mkdtemp()
        try:
            result = get_locally_available_models(temp_dir, airplane_mode=True)
            assert isinstance(result, dict)
        finally:
            shutil.rmtree(temp_dir)

    def test_get_locally_available_models_with_env(self):
        """Test with .env file containing API keys"""
        from npcpy.npc_sysenv import get_locally_available_models

        temp_dir = tempfile.mkdtemp()
        try:
            # Create .env file
            env_content = """
OPENAI_API_KEY=sk-test123
ANTHROPIC_API_KEY=
GEMINI_API_KEY=test-gemini-key
"""
            env_path = os.path.join(temp_dir, ".env")
            with open(env_path, "w") as f:
                f.write(env_content)

            result = get_locally_available_models(temp_dir, airplane_mode=True)
            assert isinstance(result, dict)
        finally:
            shutil.rmtree(temp_dir)


class TestMiniMaxProvider:
    """Test MiniMax model registration and compatible API routing."""

    def test_target_models_use_minimax_provider(self):
        """Both target model IDs should be registered with MiniMax."""
        from npcpy.npc_sysenv import MINIMAX_MODELS, lookup_provider

        assert MINIMAX_MODELS == frozenset({"MiniMax-M3", "MiniMax-M2.7"})
        assert lookup_provider("MiniMax-M3") == "minimax"
        assert lookup_provider("MiniMax-M2.7") == "minimax"

    @pytest.mark.parametrize(
        ("api_url", "expected_base", "expected_model"),
        [
            (None, "https://api.minimax.io/v1", "openai/MiniMax-M2.7"),
            (
                "https://api.minimaxi.com/v1",
                "https://api.minimaxi.com/v1",
                "openai/MiniMax-M2.7",
            ),
            (
                "https://api.minimax.io/anthropic",
                "https://api.minimax.io/anthropic",
                "anthropic/MiniMax-M2.7",
            ),
            (
                "https://api.minimaxi.com/anthropic",
                "https://api.minimaxi.com/anthropic",
                "anthropic/MiniMax-M2.7",
            ),
        ],
    )
    def test_compatible_api_routing(
        self, monkeypatch, api_url, expected_base, expected_model
    ):
        """The existing API URL setting should select region and protocol."""
        from types import SimpleNamespace
        from npcpy.gen import response as response_module

        captured = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="ok", tool_calls=None)
                    )
                ]
            )

        monkeypatch.setattr(response_module, "completion", fake_completion)
        monkeypatch.delenv("MINIMAX_API_URL", raising=False)
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")

        result = response_module.get_litellm_response(
            prompt="Hello",
            model="MiniMax-M2.7",
            provider="minimax",
            api_url=api_url,
        )

        assert result["response"] == "ok"
        assert captured["model"] == expected_model
        assert captured["api_base"] == expected_base
        assert captured["api_key"] == "test-key"

    def test_compatible_request_paths(self):
        """LiteLLM should append each protocol's request path to its Base URL."""
        import json
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from threading import Thread
        from npcpy.gen import response as response_module

        class CaptureHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.server.paths.append(self.path)
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                if self.path.endswith("/chat/completions"):
                    payload = {
                        "id": "chatcmpl-test",
                        "object": "chat.completion",
                        "created": 0,
                        "model": "MiniMax-M3",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "ok"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
                    }
                else:
                    payload = {
                        "id": "msg_test",
                        "type": "message",
                        "role": "assistant",
                        "model": "MiniMax-M2.7",
                        "content": [{"type": "text", "text": "ok"}],
                        "stop_reason": "end_turn",
                        "stop_sequence": None,
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    }
                body = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), CaptureHandler)
        server.paths = []
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host = f"http://127.0.0.1:{server.server_port}"

        try:
            openai_result = response_module.get_litellm_response(
                prompt="Hello",
                model="MiniMax-M3",
                provider="minimax",
                api_url=f"{host}/v1",
                api_key="test-key",
                timeout=5,
            )
            anthropic_result = response_module.get_litellm_response(
                prompt="Hello",
                model="MiniMax-M2.7",
                provider="minimax",
                api_url=f"{host}/anthropic",
                api_key="test-key",
                timeout=5,
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

        assert openai_result["response"] == "ok"
        assert anthropic_result["response"] == "ok"
        assert server.paths == [
            "/v1/chat/completions",
            "/anthropic/v1/messages",
        ]


class TestPlatformDetection:
    """Test platform detection variables."""

    def test_on_windows_is_bool(self):
        """Test ON_WINDOWS is a boolean"""
        from npcpy.npc_sysenv import ON_WINDOWS

        assert isinstance(ON_WINDOWS, bool)

    def test_platform_matches_system(self):
        """Test platform detection matches actual system"""
        import platform
        from npcpy.npc_sysenv import ON_WINDOWS

        if platform.system() == "Windows":
            assert ON_WINDOWS is True
        else:
            assert ON_WINDOWS is False


class TestGlobalStateVariables:
    """Test global state variables are properly initialized."""

    def test_running_flag_exists(self):
        """Test running flag is initialized"""
        from npcpy.npc_sysenv import running

        assert isinstance(running, bool)

    def test_is_recording_flag_exists(self):
        """Test is_recording flag is initialized"""
        from npcpy.npc_sysenv import is_recording

        assert isinstance(is_recording, bool)

    def test_recording_data_is_list(self):
        """Test recording_data is a list"""
        from npcpy.npc_sysenv import recording_data

        assert isinstance(recording_data, list)

    def test_buffer_data_is_list(self):
        """Test buffer_data is a list"""
        from npcpy.npc_sysenv import buffer_data

        assert isinstance(buffer_data, list)


class TestRenderMarkdown:
    """Test markdown rendering functionality."""

    def test_render_markdown_exists(self):
        """Test render_markdown function exists"""
        from npcpy.npc_sysenv import render_markdown

        assert callable(render_markdown)

    def test_render_markdown_with_string(self):
        """Test render_markdown with simple string"""
        from npcpy.npc_sysenv import render_markdown

        # Should not raise, output depends on rich availability
        try:
            render_markdown("# Test Header\n\nSome **bold** text")
        except Exception as e:
            # May fail if rich not installed, that's ok
            pytest.skip(f"render_markdown requires rich: {e}")


class TestEnvironmentSetup:
    """Test environment variable setup."""

    def test_pythonwarnings_set(self):
        """Test PYTHONWARNINGS is set to ignore"""
        # The module sets this on import
        from npcpy import npc_sysenv  # noqa

        assert os.environ.get("PYTHONWARNINGS") == "ignore"

    def test_sdl_audiodriver_set(self):
        """Test SDL_AUDIODRIVER is set to dummy"""
        from npcpy import npc_sysenv  # noqa

        assert os.environ.get("SDL_AUDIODRIVER") == "dummy"


class TestOptionalImports:
    """Test optional module imports are handled gracefully."""

    def test_readline_import_handled(self):
        """Test readline import is handled (may be None on some systems)"""
        # Just importing should not raise
        from npcpy.npc_sysenv import readline

        # readline may be None or the module
        assert readline is None or hasattr(readline, "add_history")

    def test_rich_imports_handled(self):
        """Test rich imports are handled gracefully"""
        from npcpy.npc_sysenv import Console, Markdown, Syntax

        # These may be None if rich not installed
        if Console is not None:
            assert callable(Console)
        if Markdown is not None:
            assert callable(Markdown)


# =============================================================================
# Platform-Specific Path Tests (Issue #95)
# =============================================================================

class TestPlatformPaths:
    """Test platform-specific path functions."""

    def test_get_data_dir_returns_string(self):
        """get_data_dir should return a string path."""
        from npcpy.npc_sysenv import get_data_dir
        result = get_data_dir()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_config_dir_returns_string(self):
        """get_config_dir should return a string path."""
        from npcpy.npc_sysenv import get_config_dir
        result = get_config_dir()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_models_dir_returns_string(self):
        """get_models_dir should return a string path."""
        from npcpy.npc_sysenv import get_models_dir
        result = get_models_dir()
        assert isinstance(result, str)
        assert 'models' in result.lower() or 'npcsh' in result.lower()


class TestMLXDiscovery:
    """Test MLX model discovery (Issue #193)."""

    def test_mlx_discovery_function_runs(self):
        """Test that model discovery runs without error."""
        from npcpy.npc_sysenv import get_locally_available_models

        temp_dir = tempfile.mkdtemp()
        try:
            # MLX discovery happens inside get_locally_available_models
            result = get_locally_available_models(temp_dir, airplane_mode=True)
            assert isinstance(result, dict)
        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
