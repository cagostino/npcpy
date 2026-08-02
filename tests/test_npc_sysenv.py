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
    """Test MiniMax model registration in npc_sysenv."""

    def test_target_models_use_minimax_provider(self):
        """Both target model IDs should be registered with MiniMax."""
        from npcpy.npc_sysenv import MINIMAX_MODELS, lookup_provider

        assert MINIMAX_MODELS == frozenset({"MiniMax-M3", "MiniMax-M2.7"})
        assert lookup_provider("MiniMax-M3") == "minimax"
        assert lookup_provider("MiniMax-M2.7") == "minimax"


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
