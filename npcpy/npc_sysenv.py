from datetime import datetime
from dotenv import load_dotenv
import logging
import re
import os
import socket
import concurrent.futures 
import platform
import sqlite3
import subprocess
import sys
from typing import Dict, List
import textwrap
import json

import requests
ON_WINDOWS = platform.system() == "Windows"
ON_MACOS = platform.system() == "Darwin"

def get_data_dir() -> str:
    """Get the data directory."""
    if ON_WINDOWS:
        base = os.environ.get('LOCALAPPDATA', os.path.expanduser('~/AppData/Local'))
        return os.path.join(base, 'npcsh')
    elif ON_MACOS:
        return os.path.expanduser('~/Library/Application Support/npcsh')
    else:
        xdg_data = os.environ.get('XDG_DATA_HOME', os.path.expanduser('~/.local/share'))
        return os.path.join(xdg_data, 'npcsh')

def get_config_dir() -> str:
    """Get the platform-specific config directory."""
    if ON_WINDOWS:
        base = os.environ.get('APPDATA', os.path.expanduser('~/AppData/Roaming'))
        return os.path.join(base, 'npcsh')
    elif ON_MACOS:
        return os.path.expanduser('~/Library/Application Support/npcsh')
    else:
        xdg_config = os.environ.get('XDG_CONFIG_HOME', os.path.expanduser('~/.config'))
        return os.path.join(xdg_config, 'npcsh')

def get_models_dir() -> str:
    """Get the directory for storing models."""
    return os.path.join(get_data_dir(), 'npc_team', 'models')

def get_images_dir() -> str:
    """Get the directory for storing generated images."""
    return os.path.join(get_data_dir(), 'npc_team', 'images')

def get_jobs_dir() -> str:
    """Get the directory for cron/scheduled jobs."""
    return os.path.join(get_data_dir(), 'npc_team', 'jobs')

def get_triggers_dir() -> str:
    """Get the directory for trigger scripts."""
    return os.path.join(get_data_dir(), 'npc_team', 'triggers')

def get_videos_dir() -> str:
    """Get the directory for generated videos."""
    return os.path.join(get_data_dir(), 'npc_team', 'videos')

def get_attachments_dir() -> str:
    """Get the directory for attachments."""
    return os.path.join(get_data_dir(), 'npc_team', 'attachments')

def get_logs_dir() -> str:
    """Get the directory for logs."""
    return os.path.join(get_data_dir(), 'npc_team', 'logs')

try:
    if not ON_WINDOWS:
        import termios
        import tty
        import pty
        import select
        import signal
except ImportError:
    termios = None
    tty = None
    pty = None
    select = None
    signal = None

try:
    import readline
except ImportError:
    readline = None
    logging.warning('no readline support, some features may not work as desired.')

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.syntax import Syntax
except ImportError:
    Console = None
    Markdown = None
    Syntax = None

import warnings
import time

running = True
is_recording = False
recording_data = []
buffer_data = []
last_speech_time = 0

warnings.filterwarnings("ignore", module="whisper.transcribe")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", module="torch.serialization")
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["SDL_AUDIODRIVER"] = "dummy"

def check_internet_connection(timeout=5):
    """
    Checks for internet connectivity by trying to connect to a well-known host.
    """
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=timeout)
        return True
    except OSError:
        return False

def get_locally_available_models(project_directory, airplane_mode=False, gguf_dir=None):
    available_models = {}
    env_path = os.path.join(project_directory, ".env")
    env_vars = {}
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    if "=" in line:
                        key, value = line.split("=", 1)
                        env_vars[key.strip()] = value.strip().strip("\"'")

    internet_available = check_internet_connection()
    if not internet_available:
        logging.info("No internet connection detected. External API calls will be skipped.")
        airplane_mode = True
    else:
        logging.info("Internet connection detected. Proceeding based on 'airplane_mode' parameter.")

    airplane_mode = False
    if not airplane_mode:
        timeout_seconds = 3.5

        provider_model_attrs = {
            "anthropic": ("ANTHROPIC_API_KEY", "anthropic_models"),
            "moonshot": ("MOONSHOT_API_KEY", "moonshot_models"),
            "openai": ("OPENAI_API_KEY", "open_ai_chat_completion_models"),
            "gemini": ("GEMINI_API_KEY", "gemini_models"),
            "deepseek": ("DEEPSEEK_API_KEY", "deepseek_models"),
            "groq": ("GROQ_API_KEY", "groq_models"),
            "mistral": ("MISTRAL_API_KEY", "mistral_chat_models"),
            "xai": ("XAI_API_KEY", "xai_models"),
            "perplexity": ("PERPLEXITY_API_KEY", "perplexity_models"),
            "together": ("TOGETHER_API_KEY", "together_ai_models"),
            "fireworks_ai": ("FIREWORKS_API_KEY", "fireworks_ai_models"),
            "cerebras": ("CEREBRAS_API_KEY", "cerebras_models"),
            "ai21": ("AI21_API_KEY", "ai21_models"),
            "azure": ("AZURE_API_KEY", "azure_models"),
            "cohere": ("COHERE_API_KEY", "cohere_models"),
            "openrouter": ("OPENROUTER_API_KEY", "openrouter_models"),
            "novita": ("NOVITA_API_KEY", "novita_models"),
            "hyperbolic": ("HYPERBOLIC_API_KEY", "hyperbolic_models"),
            "sambanova": ("SAMBANOVA_API_KEY", "sambanova_models"),
            "nebius": ("NEBIUS_API_KEY", "nebius_models"),
        }

        try:
            import litellm
        except Exception as e:
            logging.info(f"litellm not available for model listing: {e}")
            litellm = None

        for provider, (env_var, attr_name) in provider_model_attrs.items():
            if env_var not in env_vars and not os.environ.get(env_var):
                continue
            if litellm is None:
                continue
            try:
                model_set = getattr(litellm, attr_name, None)
                if not model_set:
                    continue
                for model_id in model_set:
                    clean_id = model_id.split("/")[-1] if "/" in model_id else model_id
                    available_models[clean_id] = provider
            except Exception as e:
                logging.info(f"{provider.capitalize()} models not indexed: {e}")
    try:
        import ollama
        timeout_seconds = 0.5 
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ollama_executor:
            def fetch_ollama_models():
                return ollama.list()

            future = ollama_executor.submit(fetch_ollama_models)
            models = future.result(timeout=timeout_seconds) 

        for model in models.models:
            if "embed" not in model.model:
                mod = model.model
                available_models[mod] = "ollama"
    except (ImportError, concurrent.futures.TimeoutError, Exception) as e:
        logging.info(f"Error loading Ollama models or timed out: {e}")

    models_dir = get_models_dir()
    gguf_dirs = [
        os.path.join(models_dir, 'gguf'),
        models_dir,
        os.path.expanduser('~/models'),
        os.path.expanduser('~/.cache/huggingface/hub'),
    ]
    resolved_gguf = gguf_dir or os.environ.get('GGUF_DIR')
    if resolved_gguf:
        gguf_dirs.insert(0, os.path.expanduser(resolved_gguf))

    seen_paths = set()
    for scan_dir in gguf_dirs:
        if not os.path.isdir(scan_dir):
            continue
        try:
            for root, dirs, files in os.walk(scan_dir):
                for f in files:
                    if f.endswith(('.gguf', '.ggml')) and not f.startswith('.'):
                        full_path = os.path.join(root, f)
                        if full_path not in seen_paths:
                            seen_paths.add(full_path)
                            available_models[full_path] = "llamacpp"
        except Exception as e:
            logging.info(f"Error scanning GGUF directory {scan_dir}: {e}")

    try:
        import requests
        response = requests.get('http://127.0.0.1:1234/v1/models', timeout=1)
        if response.ok:
            data = response.json()
            for model in data.get('data', []):
                model_id = model.get('id', model.get('name', 'unknown'))
                available_models[model_id] = "lmstudio"
    except Exception as e:
        logging.debug(f"LM Studio not available: {e}")

    try:
        import requests
        response = requests.get('http://127.0.0.1:8080/v1/models', timeout=1)
        if response.ok:
            data = response.json()
            for model in data.get('data', []):
                model_id = model.get('id', model.get('name', 'unknown'))
                available_models[model_id] = "llamacpp-server"
    except Exception as e:
        logging.debug(f"llama.cpp server not available: {e}")

    try:
        import requests
        response = requests.get('http://127.0.0.1:8000/v1/models', timeout=1)
        if response.ok:
            data = response.json()
            for model in data.get('data', []):
                model_id = model.get('id', model.get('name', 'unknown'))
                available_models[model_id] = "omlx"
    except Exception as e:
        logging.debug(f"OMLX server not available: {e}")

    try:
        import requests
        response = requests.get('http://127.0.0.1:5000/v1/models', timeout=1)
        if response.ok:
            data = response.json()
            for model in data.get('data', []):
                model_id = model.get('id', model.get('name', 'unknown'))
                if model_id not in available_models:
                    available_models[model_id] = "omlx"
    except Exception as e:
        logging.debug(f"OMLX server (port 5000) not available: {e}")

    lora_dirs = [
        get_models_dir(),
    ]
    for scan_dir in lora_dirs:
        if not os.path.isdir(scan_dir):
            continue
        try:
            for item in os.listdir(scan_dir):
                item_path = os.path.join(scan_dir, item)
                if os.path.isdir(item_path):
                    adapter_config = os.path.join(item_path, 'adapter_config.json')
                    if os.path.exists(adapter_config):
                        available_models[item_path] = "lora"
                        logging.debug(f"Found LoRA adapter: {item_path}")
        except Exception as e:
            logging.debug(f"Error scanning LoRA directory {scan_dir}: {e}")

    return available_models

def log_action(action: str, detail: str = "") -> None:
    """
    Function Description:
        This function logs an action with optional detail.
    Args:
        action: The action to log.
        detail: Additional detail to log.
    Keyword Args:
        None
    Returns:
        None
    """
    logging.info(f"{action}: {detail}")

def preprocess_code_block(code_text):
    """
    Preprocess code block text to remove leading spaces.
    """
    lines = code_text.split("\n")
    return "\n".join(line.lstrip() for line in lines)

def preprocess_markdown(md_text):
    """
    Preprocess markdown text to handle code blocks separately.
    """
    lines = md_text.split("\n")
    processed_lines = []

    inside_code_block = False
    current_code_block = []

    for line in lines:
        if line.startswith("```"):  
            if inside_code_block:

                processed_lines.append("```")
                processed_lines.extend(
                    textwrap.dedent("\n".join(current_code_block)).split("\n")
                )
                processed_lines.append("```")
                current_code_block = []
            inside_code_block = not inside_code_block
        elif inside_code_block:
            current_code_block.append(line)
        else:
            processed_lines.append(line)

    return "\n".join(processed_lines)

def request_user_input(input_request: Dict[str, str]) -> str:
    """
    Request and get input from user.

    Args:
        input_request: Dict with reason and prompt for input

    Returns:
        User's input text
    """
    print(f"\nAdditional input needed: {input_request['reason']}")
    return input(f"{input_request['prompt']}: ")

def render_markdown(text: str) -> None:
    """
    Renders markdown text, but handles code blocks as plain syntax-highlighted text.
    """
    lines = text.split("\n")
    console = Console()

    inside_code_block = False
    code_lines = []
    prose_lines = []
    lang = None

    def _flush_prose():
        if not prose_lines:
            return
        import re
        block = "\n".join(ln.rstrip('\r') for ln in prose_lines)
        block = re.sub(r'\n{3,}', '\n\n', block)

        _box_re = re.compile(r'[─-╿]')

        def _is_structured(ln: str) -> bool:
            """Box-drawing art or 4-space-indented code: must be rendered verbatim."""
            s = ln.strip()
            return bool(s) and (bool(_box_re.search(ln)) or ln.startswith('    '))

        lines = block.split('\n')
        segments: list = []
        struct_acc: list = []
        prose_acc: list = []

        def _commit_struct():
            if struct_acc:
                segments.append((True, list(struct_acc)))
                struct_acc.clear()

        def _commit_prose():
            if prose_acc:
                acc = list(prose_acc)
                while acc and not acc[0].strip():
                    acc.pop(0)
                while acc and not acc[-1].strip():
                    acc.pop()
                if acc:
                    segments.append((False, acc))
                prose_acc.clear()

        in_struct = False
        for i, ln in enumerate(lines):
            if _is_structured(ln):
                if not in_struct:
                    _commit_prose()
                    in_struct = True
                struct_acc.append(ln)
            elif not ln.strip():
                if in_struct:
                    j = i + 1
                    while j < len(lines) and not lines[j].strip():
                        j += 1
                    if j < len(lines) and _is_structured(lines[j]):
                        pass
                    else:
                        _commit_struct()
                        in_struct = False
                        prose_acc.append(ln)
                else:
                    prose_acc.append(ln)
            else:
                if in_struct:
                    _commit_struct()
                    in_struct = False
                prose_acc.append(ln)

        if in_struct:
            _commit_struct()
        else:
            _commit_prose()

        for is_str, seg_lines in segments:
            seg = '\n'.join(seg_lines)
            if not seg.strip():
                continue
            if is_str:
                sys.stdout.write(seg + '\n')
                sys.stdout.flush()
            else:
                console.print(Markdown(seg))

        prose_lines.clear()

    for line in lines:
        if line.startswith("```"):
            if inside_code_block:
                code = "\n".join(code_lines)
                if code.strip():
                    syntax = Syntax(
                        code, lang or "python", theme="monokai", line_numbers=False
                    )
                    console.print(syntax)
                code_lines = []
            else:
                _flush_prose()
                lang = line[3:].strip() or None
            inside_code_block = not inside_code_block
        elif inside_code_block:
            code_lines.append(line)
        else:
            prose_lines.append(line)

    _flush_prose()

def get_directory_npcs(directory: str = None) -> List[str]:
    """
    Function Description:
        This function retrieves a list of valid NPCs from the database.
    Args:
        db_path: The path to the database file.
    Keyword Args:
        None
    Returns:
        A list of valid NPCs.
    """
    if directory is None:
        directory = os.path.expanduser("./npc_team")
    npcs = []
    for filename in os.listdir(directory):
        if filename.endswith(".npc"):
            npcs.append(filename[:-4])
    return npcs

def get_db_npcs(db_path: str) -> List[str]:
    """
    Function Description:
        This function retrieves a list of valid NPCs from the database.
    Args:
        db_path: The path to the database file.
    Keyword Args:
        None
    Returns:
        A list of valid NPCs.
    """
    if "~" in db_path:
        db_path = os.path.expanduser(db_path)
    db_conn = sqlite3.connect(db_path)
    cursor = db_conn.cursor()
    cursor.execute("SELECT name FROM compiled_npcs")
    npcs = [row[0] for row in cursor.fetchall()]
    db_conn.close()
    return npcs

def guess_mime_type(filename):
    """Guess the MIME type of a file based on its extension."""
    extension = os.path.splitext(filename)[1].lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".webp": "image/webp",
        ".pdf": "application/pdf",
        ".txt": "text/plain",
        ".csv": "text/csv",
        ".json": "application/json",
        ".md": "text/markdown",
    }
    return mime_types.get(extension, "application/octet-stream")

def get_model_and_provider(command: str, available_models: list) -> tuple:
    """
    Function Description:
        Extracts model and provider from command and autocompletes if possible.
    Args:
        command : str : Command string
        available_models : list : List of available models
    Keyword Args:
        None
    Returns:
        model_name : str : Model name
        provider : str : Provider
        cleaned_command : str : Clean

    """

    model_match = re.search(r"@(\S+)", command)
    if model_match:
        model_name = model_match.group(1)

        matches = [m for m in available_models if m.startswith(model_name)]
        if matches:
            if len(matches) == 1:
                model_name = matches[0]  

            provider = lookup_provider(model_name)
            if provider:

                cleaned_command = command.replace(
                    f"@{model_match.group(1)}", ""
                ).strip()

                return model_name, provider, cleaned_command
            else:
                return None, None, command  
        else:
            return None, None, command  
    else:
        return None, None, command  

def render_code_block(code: str, language: str = None) -> None:
    """Render a code block with syntax highlighting using rich, left-justified with no line numbers"""
    from rich.syntax import Syntax
    from rich.console import Console

    console = Console(highlight=True)
    code = code.strip()

    if code.split("\n", 1)[0].lower() in ["python", "bash", "javascript"]:
        code = code.split("\n", 1)[1]
    syntax = Syntax(
        code, language or "python", theme="monokai", line_numbers=False, padding=0
    )
    console.print(syntax)

def print_and_process_stream_with_markdown(response, model, provider, show=False, rerender=True):
    import sys

    str_output = ""
    dot_count = 0
    tool_call_data = {"id": None, "function_name": None, "arguments": ""}
    interrupted = False

    if isinstance(response, str):
        render_markdown(response)
        print('\n') 
        return response 

    if rerender:
        sys.stdout.write('\033[s')
        sys.stdout.flush()

    try:
        for chunk in response:

            if provider == "ollama":

                if "message" in chunk and "tool_calls" in chunk["message"]:
                    for tool_call in chunk["message"]["tool_calls"]:
                        if "id" in tool_call:
                            tool_call_data["id"] = tool_call["id"]
                        if "function" in tool_call:
                            if "name" in tool_call["function"]:
                                tool_call_data["function_name"] = tool_call["function"]["name"]
                            if "arguments" in tool_call["function"]:
                                if isinstance(tool_call["function"]["arguments"], dict):
                                    tool_call_data["arguments"] += json.dumps(tool_call["function"]["arguments"])
                                else:
                                    tool_call_data["arguments"] += tool_call["function"]["arguments"]
                chunk_content = chunk["message"]["content"] if "message" in chunk and "content" in chunk["message"] else ""
                reasoning_content = chunk['message'].get('thinking', '') if "message" in chunk and "thinking" in chunk['message'] else ""
                if show:
                    if len(reasoning_content) > 0:
                        print(reasoning_content, end="", flush=True)
                    if chunk_content != "":
                        print(chunk_content, end="", flush=True)
                else:
                    print('.', end="", flush=True)
                    dot_count += 1

            else:
                for c in chunk.choices:
                    if hasattr(c.delta, "tool_calls") and c.delta.tool_calls:
                        for tool_call in c.delta.tool_calls:
                            if tool_call.id:
                                tool_call_data["id"] = tool_call.id
                            if tool_call.function:
                                if hasattr(tool_call.function, "name") and tool_call.function.name:
                                    tool_call_data["function_name"] = tool_call.function.name
                                if hasattr(tool_call.function, "arguments") and tool_call.function.arguments:
                                    tool_call_data["arguments"] += tool_call.function.arguments

                chunk_content = ''
                reasoning_content = ''
                for c in chunk.choices:
                    if hasattr(c.delta, "reasoning_content"):
                        reasoning_content += c.delta.reasoning_content

                chunk_content += "".join(
                    c.delta.content for c in chunk.choices if c.delta.content
                )
                if show:
                    if reasoning_content is not None:
                        print(reasoning_content, end="", flush=True)
                    if chunk_content != "":
                        print(chunk_content, end="", flush=True)
                else:
                    print('.', end="", flush=True)
                    dot_count += 1

            if not chunk_content:
                continue
            str_output += chunk_content

    except KeyboardInterrupt:
        interrupted = True
        print('\n⚠️ Stream interrupted by user')

    if tool_call_data["id"] or tool_call_data["function_name"] or tool_call_data["arguments"]:
        str_output += "\n\n"
        if tool_call_data["id"]:
            str_output += f"**ID:** {tool_call_data['id']}\n\n"
        if tool_call_data["function_name"]:
            str_output += f"**Function:** {tool_call_data['function_name']}\n\n"
        if tool_call_data["arguments"]:
            try:
                args_parsed = json.loads(tool_call_data["arguments"])
                str_output += f"**Arguments:**\n```json\n{json.dumps(args_parsed, indent=2)}\n```"
            except Exception:
                str_output += f"**Arguments:** `{tool_call_data['arguments']}`"

    if interrupted:
        str_output += "\n\n[⚠️ Response interrupted by user]"

    if rerender:
        sys.stdout.write('\033[u')
        sys.stdout.write('\033[J')
        sys.stdout.flush()
        render_markdown(str_output)
    print('\n')

    return str_output

def print_and_process_stream(response, model, provider):

    str_output = ""
    dot_count = 0  
    tool_call_data = {"id": None, "function_name": None, "arguments": ""}
    interrupted = False

    thinking_part=True
    thinking_str=''
    if isinstance(response, str):
        render_markdown(response)  
        print('\n') 
        return response 
    try:
        for chunk in response:

            if provider == "ollama":

                if "message" in chunk and "tool_calls" in chunk["message"]:
                    for tool_call in chunk["message"]["tool_calls"]:
                        if "id" in tool_call:
                            tool_call_data["id"] = tool_call["id"]
                        if "function" in tool_call:
                            if "name" in tool_call["function"]:
                                tool_call_data["function_name"] = tool_call["function"]["name"]
                            if "arguments" in tool_call["function"]:
                                if isinstance(tool_call["function"]["arguments"], dict):
                                    tool_call_data["arguments"] += json.dumps(tool_call["function"]["arguments"])
                                else:
                                    tool_call_data["arguments"] += tool_call["function"]["arguments"]                
                chunk_content = chunk["message"]["content"] if "message" in chunk and "content" in chunk["message"] else ""
                reasoning_content = chunk['message'].get('thinking', '') if "message" in chunk and "thinking" in chunk['message'] else ""

                if len(reasoning_content) > 0:
                    print(reasoning_content, end="", flush=True)
                    thinking_part = True
                if chunk_content != "":
                    print(chunk_content, end="", flush=True)

            else:
                for c in chunk.choices:
                    if hasattr(c.delta, "tool_calls") and c.delta.tool_calls:
                        for tool_call in c.delta.tool_calls:
                            if tool_call.id:
                                tool_call_data["id"] = tool_call.id
                            if tool_call.function:
                                if hasattr(tool_call.function, "name") and tool_call.function.name:
                                    tool_call_data["function_name"] = tool_call.function.name
                                if hasattr(tool_call.function, "arguments") and tool_call.function.arguments:
                                    tool_call_data["arguments"] += tool_call.function.arguments

                chunk_content = ''
                reasoning_content = ''
                for c in chunk.choices:
                    if hasattr(c.delta, "reasoning_content"):        
                        reasoning_content += c.delta.reasoning_content

                chunk_content += "".join(
                    c.delta.content for c in chunk.choices if c.delta.content
                )
                if reasoning_content is not None:
                    if thinking_part:
                        thinking_str +='<think>'
                        thinking_part=False
                        print('<think>')
                    print(reasoning_content, end="", flush=True)
                    thinking_str+=reasoning_content

                if chunk_content != "":
                    if len(thinking_str) >0 and not thinking_part and '</think>' not in thinking_str:

                        thinking_str+='</think>'
                        print('</think>')
                    print(chunk_content, end="", flush=True)

            if not chunk_content:
                continue
            str_output += chunk_content

    except KeyboardInterrupt:
        interrupted = True
        print('\n⚠️ Stream interrupted by user')

    if tool_call_data["id"] or tool_call_data["function_name"] or tool_call_data["arguments"]:
        str_output += "\n\n"
        if tool_call_data["id"]:
            str_output += f"**ID:** {tool_call_data['id']}\n\n"
        if tool_call_data["function_name"]:
            str_output += f"**Function:** {tool_call_data['function_name']}\n\n"
        if tool_call_data["arguments"]:
            try:
                args_parsed = json.loads(tool_call_data["arguments"])
                str_output += f"**Arguments:**\n```json\n{json.dumps(args_parsed, indent=2)}\n```"
            except Exception:
                str_output += f"**Arguments:** `{tool_call_data['arguments']}`"

    if interrupted:
        str_output += "\n\n[⚠️ Response interrupted by user]"

    return thinking_str+str_output   
def get_system_message(npc, team=None, tool_capable=False) -> str:

    if npc is None:
        return "You are a helpful assistant"
    if npc.plain_system_message:
        return npc.primary_directive

    system_message = f"""
.
..
...
....
.....
......
.......
........
.........
..........
Hello!
Welcome to the team.
You are the {npc.name} NPC with the following primary directive: {npc.primary_directive}.
Users may refer to you by your assistant name, {npc.name} and you should
consider this to be your core identity.
The current working directory is {os.getcwd()}.
The current date and time are : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

    if hasattr(npc, 'kg_data') and npc.kg_data:
        memory_context = npc.get_memory_context()
        if memory_context:
            system_message += f"\n\nMemory Context:\n{memory_context}\n"

    if getattr(npc, "db_conn", None) is not None:
        db_path = None
        if hasattr(npc.db_conn, "url") and npc.db_conn.url:
            db_path = npc.db_conn.url.database
        elif hasattr(npc.db_conn, "database"):
            db_path = npc.db_conn.database
        system_message += """What follows is information about the database connection. If you are asked to execute queries with tools, use this information. 
        If you are asked for help with debugging queries, use this information. 
        Do not unnecessarily reference that you possess this information unless it is
        specifically relevant to the request.

        DB Connection Information:        
        """
        if db_path:
            system_message += f"\nDatabase path: {db_path}\n"
        if npc.tables is not None:
            system_message += f"\nDatabase tables: {npc.tables}\n"

    if team is not None and npc.name == getattr(team, 'forenpc_name', 'sibiji'):
        team_context = team.context if hasattr(team, "context") and team.context else ""
        team_preferences = team.shared_context.get('preferences', '') if hasattr(team, "shared_context") else ""
        system_message += f"\nTeam context: {team_context}\n"
        if team_preferences:
            system_message += f"Team preferences: {team_preferences}\n"

        if hasattr(team, 'npcs') and team.npcs:
            members = []
            for name, member in team.npcs.items():
                if name != npc.name:
                    directive = getattr(member, 'primary_directive', '')
                    desc = directive[:50].strip() if directive else ''
                    members.append(f"  - @{name}: {desc}")
            if members:
                system_message += "\nTeam members available for delegation:\n" + "\n".join(members) + "\n"

    if hasattr(npc, 'jinxes_dict') and npc.jinxes_dict:
        tool_lines = []
        for jname, jinx in npc.jinxes_dict.items():
            desc = getattr(jinx, 'description', '') or ''
            tool_lines.append(f"  - {jname}: {desc.strip()}")
        if tool_lines:
            system_message += "\nYou have access to the following jinxes:\n"
            system_message += "\n".join(tool_lines) + "\n"
            if tool_capable:
                system_message += "\nUse the provided function calling interface to invoke tools when they are relevant to the request. For multi-step tasks, call one tool at a time and use its result to inform the next step.\n"
            else:
                jinx_names_str = ", ".join(npc.jinxes_dict.keys())
                jinx_instructions = f"""
                if you are in the [ReAct loop] and you are asked to use jinxes, refer to these guidelines:
              [BEGIN GUIDELINES FOR JINX EXECUTION]
                  Use jinxes when appropriate. For example:

                    - If you are asked about something up-to-date or dynamic (e.g., latest exchange rates)
                    - If the user asks you to read or edit a file
                    - If the user asks for code that should be executed
                    - If the user requests to open, search, download or scrape, which involve actual system or web actions
                    - If they request a screenshot, audio, or image manipulation
                    - Situations requiring file parsing (e.g., CSV or JSON loading)
                    - Scripted workflows or pipelines, e.g., generate a chart, fetch data, summarize from source, etc.

                    You MUST use a jinx if the request directly refers to a tool the AI cannot handle directly (e.g., 'run', 'open', 'search', etc).

                    You do not need to use a jinx if:

                    - the user asks a simple question like 'what is 2+2' or 'who invented linux', essentially any question which only requires general knowledge.
                    - The user asks you to write them a story (unless they separately specify saving it to a file, then you should directly write the story to be output through a jinx to said file.)
                    To invoke a jinx, return the action 'invoke_jinx' along with the jinx specific name.
                    An example for a jinx-specific return would be:
                    """ +"""
                    {
                        "action": "invoke_jinx",
                        "jinx_name": "file_reader",
                        "explanation": "Read the contents of <full_filename_path_from_user_request> and <detailed explanation of how to accomplish the problem outlined in the request>."
                    }

                    Do not use the jinx names as the action keys. You must use the action 'invoke_jinx' to invoke a jinx!
                    Do not invent jinx names. Use only those provided.

                Respond with a single JSON object only.
                To use a jinx, set action to jinx, jinx_name to one of [{jinx_names_str}], and inputs with the required parameters.

              [END GUIDELINES FOR JINX EXECUTION]

"""
                system_message += jinx_instructions

    return system_message

def load_env_from_execution_dir() -> None:
    """
    Function Description:
        This function loads environment variables from a .env file in the current execution directory.
    Args:
        None
    Keyword Args:
        None
    Returns:
        None
    """

    execution_dir = os.path.abspath(os.getcwd())
    env_path = os.path.join(execution_dir, ".env")
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path)
        logging.info(f"Loaded .env file from {execution_dir}")
    else:
        logging.warning(f"Warning: No .env file found in {execution_dir}")

def lookup_provider(model: str) -> str:
    """
    Determine the provider based on the model name.
    Checks custom providers first, then falls back to known providers.

    Args:
        model: The model name

    Returns:
        The provider name or None if not found
    """
    if not model:
        return None
    if os.path.isdir(os.path.expanduser(model)):
        adapter_config = os.path.join(os.path.expanduser(model), 'adapter_config.json')
        if os.path.exists(adapter_config):
            return "lora"

    if model == "deepseek-chat" or model == "deepseek-reasoner":
        return "deepseek"

    if model.startswith("airllm-"):
        return "airllm"

    ollama_prefixes = [
        "llama", "deepseek", "qwen", "llava", 
        "phi", "mistral", "mixtral", "dolphin", 
        "codellama", "gemma",]
    if any(model.startswith(prefix) for prefix in ollama_prefixes):
        return "ollama"

    openai_prefixes = ["gpt-", "dall-e-", "whisper-", "o1"]
    if any(model.startswith(prefix) for prefix in openai_prefixes):
        return "openai"

    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("gemini"):
        return "gemini"
    if "diffusion" in model:
        return "diffusers"

    return None

load_env_from_execution_dir()
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", None)
gemini_api_key = os.getenv("GEMINI_API_KEY", None)

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", None)
openai_api_key = os.getenv("OPENAI_API_KEY", None)

def resolve_team_dir(team_path=None):
    """Resolve the team directory from a team_path identifier.
    None -> <data_dir>/npc_team/
    Otherwise treat as absolute path.
    """
    if not team_path:
        return os.path.join(get_data_dir(), "npc_team")
    return team_path

def _git(args, cwd, timeout=15):
    """Run a git command and return stdout."""
    result = subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"git {args[0]} failed")
    return result.stdout.strip()

def team_sync_status(team_path=None):
    """Get sync status for an npc_team directory."""
    team_dir = resolve_team_dir(team_path)

    if not os.path.exists(team_dir):
        return {"status": "unavailable", "error": "Team directory not found"}

    git_dir = os.path.join(team_dir, ".git")
    if not os.path.exists(git_dir):
        return {"status": "uninitialized"}

    status_out = _git(["status", "--porcelain"], team_dir)
    modified = [l[3:] for l in status_out.split("\n") if l.strip()] if status_out else []

    has_remote = False
    try:
        remotes = _git(["remote"], team_dir)
        has_remote = "origin" in remotes
    except Exception:
        pass

    if not has_remote:
        status = "ahead" if modified else "up-to-date"
        return {"status": status, "modified": modified, "ahead": len(modified), "behind": 0}

    try:
        _git(["fetch", "origin"], team_dir)
    except Exception:
        pass

    ahead, behind = 0, 0
    try:
        branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], team_dir)
        counts = _git(["rev-list", "--left-right", "--count", f"origin/{branch}...HEAD"], team_dir)
        parts = counts.split()
        behind = int(parts[0]) if len(parts) > 0 else 0
        ahead = int(parts[1]) if len(parts) > 1 else 0
    except Exception:
        pass

    if ahead > 0 and behind > 0:
        status = "diverged"
    elif behind > 0:
        status = "behind"
    elif ahead > 0 or modified:
        status = "ahead"
    else:
        status = "up-to-date"

    return {"status": status, "modified": modified, "ahead": ahead, "behind": behind}

def team_sync_init(team_path=None):
    """Initialize git in an npc_team directory."""
    team_dir = resolve_team_dir(team_path)
    os.makedirs(team_dir, exist_ok=True)

    git_dir = os.path.join(team_dir, ".git")
    if not os.path.exists(git_dir):
        _git(["init"], team_dir)
        _git(["add", "."], team_dir)
        _git(["commit", "-m", "Initial commit"], team_dir)

    return {"success": True, "error": None}

def team_sync_pull(team_path=None):
    """Pull/rebase from upstream for an npc_team directory."""
    team_dir = resolve_team_dir(team_path)

    if not os.path.exists(os.path.join(team_dir, ".git")):
        return {"error": "Not a git repo. Initialize first."}

    has_remote = False
    try:
        remotes = _git(["remote"], team_dir)
        has_remote = "origin" in remotes
    except Exception:
        pass

    if not has_remote:
        return {"error": "No remote configured. Add an upstream remote first."}

    try:
        _git(["pull", "--rebase", "origin", "main"], team_dir)
        return {"success": True, "error": None}
    except RuntimeError:
        status_out = _git(["status", "--porcelain"], team_dir)
        conflicts = [l[3:] for l in status_out.split("\n") if l.startswith("UU") or l.startswith("AA")]
        if conflicts:
            return {"conflicts": conflicts, "error": None}
        raise

def team_sync_resolve(team_path=None, file_path=None, resolution="ours", content=None):
    """Resolve a merge conflict in an npc_team directory."""
    team_dir = resolve_team_dir(team_path)

    if not file_path or not resolution:
        return {"error": "file and resolution are required"}

    if resolution == "ours":
        _git(["checkout", "--ours", file_path], team_dir)
    elif resolution == "theirs":
        _git(["checkout", "--theirs", file_path], team_dir)
    else:
        if content is not None:
            full_path = os.path.join(team_dir, file_path)
            with open(full_path, "w") as f:
                f.write(content)

    _git(["add", file_path], team_dir)

    status_out = _git(["status", "--porcelain"], team_dir)
    remaining = [l for l in status_out.split("\n") if l.startswith("UU") or l.startswith("AA")]
    if not remaining:
        try:
            _git(["rebase", "--continue"], team_dir)
        except Exception:
            pass

    return {"success": True, "error": None}

def team_sync_commit(team_path=None, message="Update NPC team"):
    """Commit current state of an npc_team directory."""
    team_dir = resolve_team_dir(team_path)
    _git(["add", "."], team_dir)
    _git(["commit", "-m", message], team_dir)
    return {"success": True, "error": None}

def team_sync_diff(team_path=None, file_path=None):
    """Get diff for an npc_team directory."""
    team_dir = resolve_team_dir(team_path)
    args = ["diff", "--", file_path] if file_path else ["diff"]
    diff = _git(args, team_dir)
    return {"diff": diff, "error": None}
