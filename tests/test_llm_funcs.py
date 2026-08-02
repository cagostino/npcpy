import os
import tempfile
import pytest
from npcpy.llm_funcs import get_llm_response, gen_image, execute_llm_command, check_llm_command, breathe, resolve_model_provider
from npcpy.npc_compiler import NPC


def test_get_llm_response_basic():
    """Test basic LLM response"""
    response = get_llm_response(
        prompt="What is 2+2? Answer only with the number.",
        model="llama3.2:latest",
        provider="ollama"
    )
    assert response is not None
    print(f"Response: {response}")


def test_get_llm_response_with_messages():
    """Test LLM response with conversation history"""
    messages = [
        {"role": "user", "content": "Hi there!"},
        {"role": "assistant", "content": "Hello! How can I help you?"},
    ]
    
    response = get_llm_response(
        prompt="What did I just say?",
        messages=messages,
        model="llama3.2:latest",
        provider="ollama"
    )
    assert response is not None
    print(f"Conversation response: {response}")


def test_get_llm_response_with_attachments():
    """Test LLM response with file attachments"""
    temp_dir = tempfile.mkdtemp()
    test_file = os.path.join(temp_dir, "test.txt")
    
    with open(test_file, "w") as f:
        f.write("This is a test document with important information.")
    
    try:
        response = get_llm_response(
            prompt="What does the attached file contain?",
            attachments=[test_file],
            model="llama3.2:latest", 
            provider="ollama"
        )
        assert response is not None
        print(f"Attachment response: {response}")
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_execute_llm_command():
    """Test LLM command execution"""
    try:
        result = execute_llm_command(
            command="Tell me a very short joke",
            model="llama3.2:latest",
            provider="ollama"
        )
        assert result is not None
        print(f"Command result: {result}")
    except Exception as e:
        print(f"Command execution failed: {e}")


def test_check_llm_command():
    """Test LLM command checking"""
    try:
        result = check_llm_command(
            command="Calculate 5 * 7",
            model="llama3.2:latest",
            provider="ollama"
        )
        assert result is not None
        print(f"Command check result: {result}")
    except Exception as e:
        print(f"Command check failed: {e}")


def test_get_llm_response_transformers():
    result = get_llm_response(
        prompt="hello",
        model="Qwen/Qwen3-1.7b", 
        provider="transformers"
    )

    result = get_llm_response(
        prompt="what is 2+2",
        model="qwen3/qwen3-1.7b",
        provider="transformers",
        messages=[{"role": "user", "content": "hi"}]
    )

    result = get_llm_response(
        prompt="test",
        provider="transformers"
    )
def test_gen_image():
    """Test image generation"""
    try:

        result = gen_image(
            prompt="A really red circle that is redder than the reddest red redded in redding",
            model="dall-e-3",
            provider="openai"
        )
        if result:
            print(f"Generated image type: {type(result)}")
        else:
            print("Image generation returned None (expected without API key)")
    except Exception as e:
        print(f"Image generation failed (expected without API key): {e}")


def test_get_llm_response_with_npc():
    """Test LLM response using NPC"""
    try:
        test_npc = NPC(
            name="test_npc",
            primary_directive="You are a helpful test assistant",
            model="llama3.2:latest",
            provider="ollama"
        )
        
        response = get_llm_response(
            prompt="Hello, introduce yourself briefly.",
            npc=test_npc
        )
        assert response is not None
        print(f"NPC response: {response}")
    except Exception as e:
        print(f"NPC test failed: {e}")


def test_streaming_response():
    """Test streaming LLM response"""
    try:
        response = get_llm_response(
            prompt="Count from 1 to 3",
            model="llama3.2:latest",
            provider="ollama",
            stream=True
        )
        assert response is not None
        print(f"Streaming response: {response}")
    except Exception as e:
        print(f"Streaming failed: {e}")
        
def test_breathe():
    messages = [
    {'role': 'user', 'content': 'I need a function to add two numbers'},
    {'role': 'assistant', 'content': 'def add(a, b): return a + b'},
    {'role': 'user', 'content': 'Now make it work with strings too'}
    ]
    result = breathe(messages)
    print("Test 2 - Coding:")
    print(result)
    print()

    
    messages = [
        {'role': 'user', 'content': 'I want to build a todo app'},
        {'role': 'assistant', 'content': 'What features do you need?'},
        {'role': 'user', 'content': 'Add tasks, mark complete, delete'},
        {'role': 'assistant', 'content': 'Use Flask for backend and React for frontend?'},
        {'role': 'user', 'content': 'Just Flask for now, keep it simple'}
    ]
    result = breathe(messages)
    print("Test 3 - Project:")
    print(result)


class TestMiniMaxRouting:
    """Test MiniMax API URL / protocol normalization in llm_funcs."""

    def test_resolve_minimax_by_model_id(self, monkeypatch):
        """lookup_provider returns minimax for exact model IDs."""
        monkeypatch.delenv("MINIMAX_API_URL", raising=False)
        m, p, url, key = resolve_model_provider(model="MiniMax-M2.7")
        assert p == "openai"
        assert url == "https://api.minimax.io/v1"

    def test_resolve_minimax_api_key_from_env(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_API_URL", raising=False)
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        m, p, url, key = resolve_model_provider(model="MiniMax-M3", provider="minimax")
        assert p == "openai"
        assert key == "test-key"

    @pytest.mark.parametrize(
        ("api_url", "expected_provider", "expected_url"),
        [
            (None, "openai", "https://api.minimax.io/v1"),
            ("https://api.minimaxi.com/v1", "openai", "https://api.minimaxi.com/v1"),
            ("https://api.minimax.io/anthropic", "anthropic", "https://api.minimax.io/anthropic"),
            ("https://api.minimaxi.com/anthropic", "anthropic", "https://api.minimaxi.com/anthropic"),
        ],
    )
    def test_resolve_minimax_protocol(self, monkeypatch, api_url, expected_provider, expected_url):
        monkeypatch.delenv("MINIMAX_API_URL", raising=False)
        m, p, url, key = resolve_model_provider(
            model="MiniMax-M2.7",
            provider="minimax",
            api_url=api_url,
            api_key="test-key",
        )
        assert p == expected_provider
        assert url == expected_url

    def test_get_llm_response_passes_minimax_routing(self, monkeypatch):
        """get_llm_response normalizes minimax to openai/anthropic before calling get_litellm_response."""
        import npcpy.llm_funcs as llm_funcs_module

        monkeypatch.delenv("MINIMAX_API_URL", raising=False)
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")

        captured = {}

        def fake_get_litellm_response(*args, **kwargs):
            captured.update(kwargs)
            return {"response": "ok", "messages": []}

        monkeypatch.setattr(llm_funcs_module, "get_litellm_response", fake_get_litellm_response)

        result = get_llm_response(
            "Hello",
            model="MiniMax-M2.7",
            provider="minimax",
            api_url="https://api.minimax.io/anthropic",
        )

        assert result["response"] == "ok"
        assert captured["provider"] == "anthropic"
        assert captured["api_url"] == "https://api.minimax.io/anthropic"
        assert captured["api_key"] == "test-key"
        assert captured["model"] == "MiniMax-M2.7"

    def test_minimax_request_paths(self):
        """LiteLLM should append each protocol's request path to the MiniMax Base URL."""
        import json
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from threading import Thread

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
            openai_result = get_llm_response(
                "Hello",
                model="MiniMax-M3",
                provider="minimax",
                api_url=f"{host}/v1",
                api_key="test-key",
                timeout=5,
            )
            anthropic_result = get_llm_response(
                "Hello",
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

