"""Tests for MCP server integration and skills functionality."""

import asyncio
import os
import tempfile
import shutil
import sys
import types
from contextlib import asynccontextmanager
import pytest
from pathlib import Path

from npcpy.npc_compiler import NPC, Jinx, Team, _parse_skill_md, _compile_skill_to_jinx


class TestSkillParsing:
    """Test skill file parsing."""

    def test_parse_skill_md_basic(self, tmp_path):
        """Test parsing a basic SKILL.md file."""
        skill_content = """---
name: test-skill
description: A test skill for unit testing.
---
# Test Skill

## section-one
Content for section one.

## section-two
Content for section two.
"""
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(skill_content)

        result = _parse_skill_md(str(skill_file))

        assert result is not None
        assert result['name'] == 'test-skill'
        assert result['description'] == 'A test skill for unit testing.'
        assert 'section-one' in result['sections']
        assert 'section-two' in result['sections']
        assert 'Content for section one' in result['sections']['section-one']

    def test_parse_skill_md_missing_frontmatter(self, tmp_path):
        """Test parsing fails gracefully without proper frontmatter."""
        skill_content = """# No Frontmatter Skill

## section
Some content.
"""
        skill_file = tmp_path / "SKILL.md"
        skill_file.write_text(skill_content)

        result = _parse_skill_md(str(skill_file))
        # Should return None or handle gracefully
        assert result is None or result.get('name') is None

    def test_compile_skill_to_jinx(self, tmp_path):
        """Test compiling a skill to a Jinx."""
        skill_data = {
            'name': 'compiled-skill',
            'description': 'A skill that gets compiled.',
            'sections': {
                'intro': 'Introduction content.',
                'details': 'Detailed content here.'
            },
            'scripts': [],
            'references': [],
            'assets': [],
            'file_context': []
        }

        jinx = _compile_skill_to_jinx(skill_data, source_path=str(tmp_path / "test.skill"))

        assert jinx is not None
        assert jinx.jinx_name == 'compiled-skill'
        assert 'compiled' in jinx.description.lower() or 'skill' in jinx.description.lower()


class TestSkillsWithNPC:
    """Test skills integration with NPC."""

    def test_npc_with_skill_directory(self, tmp_path):
        """Test NPC loading skills from a directory structure."""
        # Create skill directory structure
        skills_dir = tmp_path / "npc_team" / "jinxes" / "skills" / "code-review"
        skills_dir.mkdir(parents=True)

        skill_content = """---
name: code-review
description: Use when reviewing code for quality and best practices.
---
# Code Review Skill

## checklist
- Check for security vulnerabilities
- Verify error handling
- Review naming conventions

## security
Focus on OWASP top 10 vulnerabilities.
"""
        (skills_dir / "SKILL.md").write_text(skill_content)

        # Create NPC file
        npc_content = """name: reviewer
primary_directive: You review code for quality and security issues.
model: llama3.2
provider: ollama
jinxes:
  - skills/code-review
"""
        npc_file = tmp_path / "npc_team" / "reviewer.npc"
        (tmp_path / "npc_team").mkdir(exist_ok=True)
        npc_file.write_text(npc_content)

        # Load NPC - skills should be available
        npc = NPC(file=str(npc_file))
        assert npc.name == "reviewer"
        # The jinxes_spec should include the skill reference
        assert npc.jinxes_spec is not None


class TestMCPClientNPC:
    """Test MCPClientNPC class."""

    @pytest.fixture
    def remote_mcp_fakes(self, monkeypatch):
        """Replace MCP transports and sessions with hermetic async fakes."""
        from npcpy import serve

        calls = {"sse": [], "streamable-http": [], "session_args": []}

        @asynccontextmanager
        async def sse_client(url):
            calls["sse"].append(url)
            yield ("sse-read", "sse-write")

        @asynccontextmanager
        async def streamablehttp_client(url):
            calls["streamable-http"].append(url)
            yield ("http-read", "http-write", lambda: "session-id")

        class FakeClientSession:
            def __init__(self, *args):
                calls["session_args"].append(args)

            async def __aenter__(self):
                return self

            async def __aexit__(self, exc_type, exc, traceback):
                return False

            async def initialize(self):
                return None

            async def list_tools(self):
                return types.SimpleNamespace(tools=[])

        sse_module = types.ModuleType("mcp.client.sse")
        sse_module.sse_client = sse_client
        streamable_module = types.ModuleType("mcp.client.streamable_http")
        streamable_module.streamablehttp_client = streamablehttp_client
        monkeypatch.setitem(sys.modules, "mcp.client.sse", sse_module)
        monkeypatch.setitem(sys.modules, "mcp.client.streamable_http", streamable_module)
        monkeypatch.setattr(serve, "ClientSession", FakeClientSession)
        return calls

    @staticmethod
    async def _connect_and_close(client, server_spec):
        try:
            await client._connect_async(server_spec)
        finally:
            if client._exit_stack is not None:
                await client._exit_stack.aclose()

    def test_remote_url_preserves_sse_default(self, remote_mcp_fakes):
        """A URL without an explicit transport remains an SSE connection."""
        from npcpy.serve import MCPClientNPC

        client = MCPClientNPC(debug=False)
        asyncio.run(self._connect_and_close(client, {"url": "https://example.com/sse"}))

        assert remote_mcp_fakes["sse"] == ["https://example.com/sse"]
        assert remote_mcp_fakes["streamable-http"] == []
        assert remote_mcp_fakes["session_args"] == [("sse-read", "sse-write")]

    def test_remote_url_supports_streamable_http(self, remote_mcp_fakes):
        """Streamable HTTP uses only the transport's read and write streams."""
        from npcpy.serve import MCPClientNPC

        client = MCPClientNPC(debug=False)
        asyncio.run(self._connect_and_close(client, {
            "url": "https://search.parallel.ai/mcp",
            "transport": "streamable-http",
        }))

        assert remote_mcp_fakes["sse"] == []
        assert remote_mcp_fakes["streamable-http"] == ["https://search.parallel.ai/mcp"]
        assert remote_mcp_fakes["session_args"] == [("http-read", "http-write")]

    def test_remote_url_rejects_unknown_transport(self, remote_mcp_fakes):
        """Unknown transport names fail before a remote connection is opened."""
        from npcpy.serve import MCPClientNPC

        client = MCPClientNPC(debug=False)
        with pytest.raises(ValueError, match="expected 'sse' or 'streamable-http'"):
            asyncio.run(self._connect_and_close(client, {
                "url": "https://example.com/mcp",
                "transport": "websocket",
            }))

        assert remote_mcp_fakes["sse"] == []
        assert remote_mcp_fakes["streamable-http"] == []
        assert remote_mcp_fakes["session_args"] == []

    def test_remote_url_rejects_non_string_transport(self, remote_mcp_fakes):
        """Transport configuration must be a documented string value."""
        from npcpy.serve import MCPClientNPC

        client = MCPClientNPC(debug=False)
        with pytest.raises(ValueError, match="must be 'sse' or 'streamable-http'"):
            asyncio.run(self._connect_and_close(client, {
                "url": "https://example.com/mcp",
                "transport": True,
            }))

        assert remote_mcp_fakes["sse"] == []
        assert remote_mcp_fakes["streamable-http"] == []
        assert remote_mcp_fakes["session_args"] == []

    def test_mcp_client_import(self):
        """Test that MCPClientNPC can be imported."""
        from npcpy.serve import MCPClientNPC
        client = MCPClientNPC(debug=False)
        assert client is not None
        assert client.available_tools_llm == []
        assert client.tool_map == {}

    def test_mcp_client_connect_nonexistent_server(self):
        """Test MCPClientNPC handles missing server gracefully."""
        from npcpy.serve import MCPClientNPC
        client = MCPClientNPC(debug=False)
        result = client.connect_sync('/nonexistent/path/to/server.py')
        assert result is False

    @pytest.fixture
    def simple_mcp_server(self, tmp_path):
        """Create a simple MCP server for testing."""
        server_content = '''"""Simple MCP server for testing."""
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Test Server")

@mcp.tool()
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@mcp.tool()
def echo_message(message: str) -> str:
    """Echo back a message."""
    return f"Echo: {message}"

if __name__ == "__main__":
    mcp.run()
'''
        server_file = tmp_path / "test_mcp_server.py"
        server_file.write_text(server_content)
        return str(server_file)

    @pytest.mark.skipif(
        not shutil.which("python") or True,  # Skip by default - requires mcp package
        reason="MCP integration test - requires mcp package and running server"
    )
    def test_mcp_client_connect_to_server(self, simple_mcp_server):
        """Test MCPClientNPC can connect to a real server."""
        from npcpy.serve import MCPClientNPC
        client = MCPClientNPC(debug=False)

        try:
            result = client.connect_sync(simple_mcp_server)
            if result:
                assert len(client.available_tools_llm) > 0
                assert 'add_numbers' in client.tool_map or 'echo_message' in client.tool_map
                client.disconnect_sync()
        except Exception:
            # MCP server may not be available in test environment
            pytest.skip("MCP server connection failed - likely missing mcp package")


class TestMCPWithNPC:
    """Test MCP integration with NPC get_llm_response."""

    def test_npc_accepts_tools_and_tool_map(self):
        """Test that NPC.get_llm_response accepts tools and tool_map parameters."""
        npc = NPC(
            name="test_assistant",
            primary_directive="You are a helpful assistant.",
            model="llama3.2",
            provider="ollama"
        )

        # Create mock tools schema and map (simulating what MCPClientNPC provides)
        mock_tools = [
            {
                "type": "function",
                "function": {
                    "name": "test_tool",
                    "description": "A test tool",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "input": {"type": "string"}
                        },
                        "required": ["input"]
                    }
                }
            }
        ]
        mock_tool_map = {
            "test_tool": lambda input: f"Processed: {input}"
        }

        # This should not raise an error - just verify the API accepts these params
        # We don't actually call the LLM here to avoid network dependencies
        assert hasattr(npc, 'get_llm_response')
        import inspect
        sig = inspect.signature(npc.get_llm_response)
        assert 'tools' in sig.parameters
        assert 'tool_map' in sig.parameters


class TestMCPToolExecution:
    """Test MCP tool execution flow."""

    def test_tool_map_callable(self):
        """Test that tool_map functions are callable."""
        def mock_search(query: str) -> str:
            return f"Results for: {query}"

        def mock_notify(message: str, channel: str = "general") -> str:
            return f"Sent '{message}' to #{channel}"

        tool_map = {
            "search_database": mock_search,
            "send_notification": mock_notify
        }

        # Verify tools work
        assert tool_map["search_database"]("test query") == "Results for: test query"
        assert tool_map["send_notification"]("hello", "alerts") == "Sent 'hello' to #alerts"


class TestReadmeExamples:
    """Verify README examples are syntactically correct."""

    def test_skills_example_structure(self, tmp_path):
        """Test the skills example from README creates valid structure."""
        # Create structure from README example
        skills_dir = tmp_path / "npc_team" / "jinxes" / "skills" / "code-review"
        skills_dir.mkdir(parents=True)

        skill_md = """---
name: code-review
description: Use when reviewing code for quality, security, and best practices.
---
# Code Review Skill

## checklist
- Check for security vulnerabilities (SQL injection, XSS, etc.)
- Verify error handling and edge cases
- Review naming conventions and code clarity

## security
Focus on OWASP top 10 vulnerabilities...
"""
        (skills_dir / "SKILL.md").write_text(skill_md)

        npc_yaml = """name: reviewer
primary_directive: You review code for quality and security issues.
model: llama3.2
provider: ollama
jinxes:
  - skills/code-review
"""
        npc_file = tmp_path / "npc_team" / "reviewer.npc"
        npc_file.write_text(npc_yaml)

        # Verify files exist
        assert (skills_dir / "SKILL.md").exists()
        assert npc_file.exists()

        # Verify skill parses correctly
        result = _parse_skill_md(str(skills_dir / "SKILL.md"))
        assert result is not None
        assert result['name'] == 'code-review'

    def test_mcp_example_imports(self):
        """Test that MCP example imports work."""
        # These imports should work
        from npcpy.npc_compiler import NPC
        from npcpy.serve import MCPClientNPC

        # Verify classes exist and can be instantiated
        npc = NPC(
            name='Assistant',
            primary_directive='You help users with tasks.',
            model='llama3.2',
            provider='ollama'
        )
        assert npc is not None

        mcp = MCPClientNPC(debug=False)
        assert mcp is not None
        assert hasattr(mcp, 'connect_sync')
        assert hasattr(mcp, 'disconnect_sync')
        assert hasattr(mcp, 'available_tools_llm')
        assert hasattr(mcp, 'tool_map')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
