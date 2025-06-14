# --- START OF FILE routes.py ---

from typing import Callable, Dict, Any, List, Optional, Union
import functools
import os
import traceback
import shlex
import time
from datetime import datetime
from sqlalchemy import create_engine
import logging 

from npcpy.npc_sysenv import (
    render_code_block, render_markdown,
    NPCSH_VISION_MODEL, NPCSH_VISION_PROVIDER, NPCSH_API_URL,
    NPCSH_CHAT_MODEL, NPCSH_CHAT_PROVIDER, NPCSH_STREAM_OUTPUT,
    NPCSH_IMAGE_GEN_MODEL, NPCSH_IMAGE_GEN_PROVIDER,
    NPCSH_EMBEDDING_MODEL, NPCSH_EMBEDDING_PROVIDER,
    NPCSH_REASONING_MODEL, NPCSH_REASONING_PROVIDER,
    NPCSH_SEARCH_PROVIDER,
)
from npcpy.data.load import load_file_contents

from npcpy.llm_funcs import (
    get_llm_response,
    gen_image,
    gen_video,
)
from npcpy.npc_compiler import NPC, Team, Jinx
from npcpy.npc_compiler import initialize_npc_project


from npcpy.work.plan import execute_plan_command
from npcpy.work.trigger import execute_trigger_command
from npcpy.work.desktop import perform_action


from npcpy.memory.search import execute_rag_command, execute_search_command, execute_brainblast_command
from npcpy.memory.command_history import CommandHistory


from npcpy.memory.knowledge_graph import breathe
from npcpy.memory.sleep import sleep, forget

from npcpy.modes.guac import enter_guac_mode
from npcpy.modes.plonk import execute_plonk_command
from npcpy.modes.serve import start_flask_server
from npcpy.modes.alicanto import alicanto
from npcpy.modes.spool import enter_spool_mode
from npcpy.modes.wander import enter_wander_mode
from npcpy.modes.yap import enter_yap_mode



from npcpy.mix.debate import run_debate
from npcpy.data.image import capture_screenshot
from npcpy.npc_compiler import NPC, Team, Jinx
from npcpy.npc_compiler import initialize_npc_project
from npcpy.data.web import search_web

class CommandRouter:
    def __init__(self):
        self.routes = {}
        self.help_info = {}
        self.shell_only = {}

    def route(self, command: str, help_text: str = "", shell_only: bool = False) -> Callable:
        def wrapper(func):
            self.routes[command] = func
            self.help_info[command] = help_text
            self.shell_only[command] = shell_only

            @functools.wraps(func)
            def wrapped_func(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapped_func
        return wrapper

    def get_route(self, command: str) -> Optional[Callable]:
        return self.routes.get(command)

    def execute(self, command_str: str, **kwargs) -> Any:
        command_name = command_str.split()[0].lstrip('/')
        route_func = self.get_route(command_name)
        if route_func:
            return route_func(command=command_str, **kwargs)
        return None

    def get_commands(self) -> List[str]:
        return list(self.routes.keys())

    def get_help(self, command: str = None) -> Dict[str, str]:
        if command:
            if command in self.help_info:
                return {command: self.help_info[command]}
            return {}
        return self.help_info

router = CommandRouter()

def get_help_text():
    commands = router.get_commands()
    help_info = router.help_info
    shell_only = router.shell_only
    commands.sort()
    output = "# Available Commands\n\n"
    for cmd in commands:
        help_text = help_info.get(cmd, "")
        shell_only_text = " (Shell only)" if shell_only.get(cmd, False) else ""
        output += f"/{cmd}{shell_only_text} - {help_text}\n\n"
    output += """
# Note
- Bash commands and programs can be executed directly (try bash first, then LLM).
- Use '/exit' or '/quit' to exit the current NPC mode or the npcsh shell.
- Jinxs defined for the current NPC or Team can also be used like commands (e.g., /screenshot).
"""
    return output

def safe_get(kwargs, key, default=None):
    return kwargs.get(key, default)

@router.route("breathe", "Condense context on a regular cadence", shell_only=True)
def breathe_handler(command: str, **kwargs):
    messages = safe_get(kwargs, "messages", [])
    npc = safe_get(kwargs, "npc")
    try:
        result = run_breathe_cycle(messages=messages, npc=npc, **kwargs)
        if isinstance(result, dict): return result
        return {"output": str(result), "messages": messages}
    except NameError:
         return {"output": "Breathe function (run_breathe_cycle) not available.", "messages": messages}
    except Exception as e:
        traceback.print_exc()
        return {"output": f"Error during breathe: {e}", "messages": messages}

@router.route("compile", "Compile NPC profiles")
def compile_handler(command: str, **kwargs):
    messages = safe_get(kwargs, "messages", [])
    npc_team_dir = safe_get(kwargs, 'current_path', './npc_team')
    parts = command.split()
    npc_file_path_arg = parts[1] if len(parts) > 1 else None
    output = ""
    try:
        if npc_file_path_arg:
            npc_full_path = os.path.abspath(npc_file_path_arg)
            if os.path.exists(npc_full_path):
                npc = NPC(npc_full_path)
                output = f"Compiled NPC: {npc_full_path}"
            else:
                output = f"Error: NPC file not found: {npc_full_path}"
        else:
            npc = NPC(npc_full_path)

            output = f"Compiled all NPCs in directory: {npc_team_dir}"
    except NameError:
        output = "Compile functions (compile_npc_file, compile_team_npcs) not available."
    except Exception as e:
        traceback.print_exc()
        output = f"Error compiling: {e}"
    return {"output": output, "messages": messages, "npc": npc}



@router.route("flush", "Flush the last N messages", shell_only=True)
def flush_handler(command: str, **kwargs):
    messages = safe_get(kwargs, "messages", [])
    try:
        parts = command.split()
        n = int(parts[1]) if len(parts) > 1 else 1
    except (ValueError, IndexError):
        return {"output": "Usage: /flush [number_of_messages_to_flush]", "messages": messages}

    if n <= 0:
        return {"output": "Error: Number of messages must be positive.", "messages": messages}

    new_messages = list(messages)
    original_len = len(new_messages)
    removed_count = 0

    if new_messages and new_messages[0].get("role") == "system":
        system_message = new_messages[0]
        working_messages = new_messages[1:]
        num_to_remove = min(n, len(working_messages))
        if num_to_remove > 0:
            final_messages = [system_message] + working_messages[:-num_to_remove]
            removed_count = num_to_remove
        else:
            final_messages = [system_message]
    else:
        num_to_remove = min(n, original_len)
        if num_to_remove > 0:
            final_messages = new_messages[:-num_to_remove]
            removed_count = num_to_remove
        else:
            final_messages = []

    output = f"Flushed {removed_count} message(s). Context is now {len(final_messages)} messages."
    return {"output": output, "messages": final_messages}

@router.route("guac", "Enter guac mode")
def guac_handler(command,  **kwargs):
    '''
    Guac ignores input npc and npc_team dirs and manually sets them to be at ~/.npcsh/guac/
    
    '''
    config_dir = safe_get(kwargs, 'config_dir', None)
    plots_dir = safe_get(kwargs, 'plots_dir', None)
    refresh_period = safe_get(kwargs, 'refresh_period', 100)
    lang = safe_get(kwargs, 'lang', None)
    messages = safe_get(kwargs, "messages", [])
    db_conn = safe_get(kwargs, 'db_conn', create_engine('sqlite:///'+os.path.expanduser('~/npcsh_history.db')))
    
    npc_file = '~/.npcsh/guac/npc_team/guac.npc'
    npc_team_dir = os.path.expanduser('~/.npcsh/guac/npc_team/')
    
    npc = NPC(file=npc_file, db_conn=db_conn)

    team = Team(npc_team_dir, db_conn=db_conn)

    
    enter_guac_mode(npc=npc, 
                    team=team, 
                    config_dir=config_dir, 
                    plots_dir=plots_dir,
                    npc_team_dir=npc_team_dir,
                    refresh_period=refresh_period, lang=lang)
    
    return {"output": 'Exiting Guac Mode', "messages": safe_get(kwargs, "messages", [])}


@router.route("help", "Show help information")
def help_handler(command, **kwargs):
    return {"output": get_help_text(), "messages": safe_get(kwargs, "messages", [])}

@router.route("init", "Initialize NPC project")
def init_handler(command: str, **kwargs):
    messages = safe_get(kwargs, "messages", [])
    try:
        parts = shlex.split(command)
        directory = "."
        templates = None
        context = None
        # Basic parsing example (needs improvement for robust flag handling)
        if len(parts) > 1 and not parts[1].startswith("-"):
            directory = parts[1]
        # Add logic here to parse -t, -ctx flags if needed

        initialize_npc_project(
            directory=directory,
            templates=templates,
            context=context,
            model=safe_get(kwargs, 'model'),
            provider=safe_get(kwargs, 'provider')
        )
        output = f"NPC project initialized in {os.path.abspath(directory)}."
    except NameError:
        output = "Init function (initialize_npc_project) not available."
    except Exception as e:
        traceback.print_exc()
        output = f"Error initializing project: {e}"
    return {"output": output, "messages": messages}


@router.route("ots", "Take screenshot and optionally analyze with vision model")
def ots_handler(command: str, **kwargs):
    command_parts = command.split()
    image_paths = []
    npc = safe_get(kwargs, 'npc')
    vision_model = safe_get(kwargs, 'model', NPCSH_VISION_MODEL)
    vision_provider = safe_get(kwargs, 'provider', NPCSH_VISION_PROVIDER)
    if vision_model == NPCSH_CHAT_MODEL: vision_model = NPCSH_VISION_MODEL
    if vision_provider == NPCSH_CHAT_PROVIDER: vision_provider = NPCSH_VISION_PROVIDER

    messages = safe_get(kwargs, 'messages', [])
    stream = safe_get(kwargs, 'stream', NPCSH_STREAM_OUTPUT)

    try:
        if len(command_parts) > 1:
            for img_path_arg in command_parts[1:]:
                full_path = os.path.abspath(img_path_arg)
                if os.path.exists(full_path):
                    image_paths.append(full_path)
                else:
                    return {"output": f"Error: Image file not found at {full_path}", "messages": messages}
        else:
            screenshot_info = capture_screenshot(npc=npc)
            if screenshot_info and "file_path" in screenshot_info:
                image_paths.append(screenshot_info["file_path"])
                print(f"Screenshot captured: {screenshot_info.get('filename', os.path.basename(screenshot_info['file_path']))}")
            else:
                 return {"output": "Error: Failed to capture screenshot.", "messages": messages}

        if not image_paths:
            return {"output": "No valid images found or captured.", "messages": messages}

        user_prompt = safe_get(kwargs, 'stdin_input')
        if user_prompt is None:
            try:
                user_prompt = input(
                    "Enter a prompt for the LLM about these images (or press Enter to skip): "
                )
            except EOFError:
                 user_prompt = "Describe the image(s)."

        if not user_prompt or not user_prompt.strip():
            user_prompt = "Describe the image(s)."

        response_data = get_llm_response(
            prompt=user_prompt,
            model=vision_model,
            provider=vision_provider,
            messages=messages,
            images=image_paths,
            stream=stream,
            npc=npc,
            api_url=safe_get(kwargs, 'api_url'),
            api_key=safe_get(kwargs, 'api_key')
        )
        return {"output": response_data.get('response'), "messages": response_data.get('messages')}

    except Exception as e:
        traceback.print_exc()
        return {"output": f"Error during /ots command: {e}", "messages": messages}


@router.route("plan", "Execute a plan command")
def plan_handler(command: str, **kwargs):
    messages = safe_get(kwargs, "messages", [])
    user_command = " ".join(command.split()[1:])
    if not user_command:
        return {"output": "Usage: /plan <description_of_plan>", "messages": messages}
    try:
        return execute_plan_command(command=user_command, **kwargs)
    except NameError:
         return {"output": "Plan function (execute_plan_command) not available.", "messages": messages}
    except Exception as e:
        traceback.print_exc()
        return {"output": f"Error executing plan: {e}", "messages": messages}

@router.route("pti", "Use pardon-the-interruption mode to interact with the LLM")
def plonk_handler(command: str, **kwargs):
    return

@router.route("plonk", "Use vision model to interact with GUI")
def plonk_handler(command: str, **kwargs):
    messages = safe_get(kwargs, "messages", [])
    request_str = " ".join(command.split()[1:])
    if not request_str:
        return {"output": "Usage: /plonk <task_description>", "messages": messages}

    action_space = {
            "click": {"x": "int (0-100)", "y": "int (0-100)"},
            "type": {"text": "string"},
            "scroll": {"direction": "up/down/left/right", "amount": "int"},
            "bash": {"command": "string"},
            "wait": {"duration": "int (seconds)"}
        }
    try:
        result = execute_plonk_command(
            request=request_str,
            action_space=action_space,
            model=safe_get(kwargs, 'model', NPCSH_VISION_MODEL),
            provider=safe_get(kwargs, 'provider', NPCSH_VISION_PROVIDER),
            npc=safe_get(kwargs, 'npc')
            )
        if isinstance(result, dict) and "output" in result:
            result_messages = result.get("messages", messages)
            return {"output": result["output"], "messages": result_messages}
        else:
            return {"output": str(result), "messages": messages}
    except NameError:
         return {"output": "Plonk function (execute_plonk_command) not available.", "messages": messages}
    except Exception as e:
        traceback.print_exc()
        return {"output": f"Error executing plonk command: {e}", "messages": messages}
@router.route("brainblast", "Execute an advanced chunked search on command history")
def brainblast_handler(command: str, **kwargs):
    messages = safe_get(kwargs, "messages", [])
    
    # Parse command to get the search query
    parts = shlex.split(command)
    search_query = " ".join(parts[1:]) if len(parts) > 1 else ""
    
    if not search_query:
        return {"output": "Usage: /brainblast <search_terms>", "messages": messages}
    
    # Get the command history instance
    command_history = kwargs.get('command_history')
    if not command_history:
        # Create a new one if not provided
        db_path = safe_get(kwargs, "history_db_path", os.path.expanduser('~/npcsh_history.db'))
        try:
            command_history = CommandHistory(db_path)
        except Exception as e:
            return {"output": f"Error connecting to command history: {e}", "messages": messages}
    
    try:
        # Remove messages from kwargs to avoid duplicate argument error
        if 'messages' in kwargs:
            del kwargs['messages']
            
        # Execute the brainblast command
        return execute_brainblast_command(
            command=search_query,
            command_history=command_history,
            messages=messages,
            top_k=safe_get(kwargs, 'top_k', 5),
            **kwargs
        )
    
    except Exception as e:
        traceback.print_exc()
        return {"output": f"Error executing brainblast command: {e}", "messages": messages}

@router.route("rag", "Execute a RAG command using ChromaDB embeddings with optional file input (-f/--file)")
def rag_handler(command: str, **kwargs):
    messages = safe_get(kwargs, "messages", [])
    
    # Parse command with shlex to properly handle quoted strings
    parts = shlex.split(command)
    user_command = []
    file_paths = []
    
    # Process arguments
    i = 1  # Skip the first element which is "rag"
    while i < len(parts):
        if parts[i] == "-f" or parts[i] == "--file":
            # We found a file flag, get the file path
            if i + 1 < len(parts):
                file_paths.append(parts[i + 1])
                i += 2  # Skip both the flag and the path
            else:
                return {"output": "Error: -f/--file flag needs a file path", "messages": messages}
        else:
            # This is part of the user query
            user_command.append(parts[i])
            i += 1
    
    user_command = " ".join(user_command)
    
    vector_db_path = safe_get(kwargs, "vector_db_path", os.path.expanduser('~/npcsh_chroma.db'))
    embedding_model = safe_get(kwargs, "embedding_model", NPCSH_EMBEDDING_MODEL)
    embedding_provider = safe_get(kwargs, "embedding_provider", NPCSH_EMBEDDING_PROVIDER)
    
    if not user_command and not file_paths:
        return {"output": "Usage: /rag [-f file_path] <query>", "messages": messages}
    
    try:
        # Process files if provided
        file_contents = []
        for file_path in file_paths:
            try:
                chunks = load_file_contents(file_path)
                file_name = os.path.basename(file_path)
                file_contents.extend([f"[{file_name}] {chunk}" for chunk in chunks])
            except Exception as file_err:
                file_contents.append(f"Error processing file {file_path}: {str(file_err)}")
        
        # Execute the RAG command
        return execute_rag_command(
            command=user_command,
            vector_db_path=vector_db_path,
            embedding_model=embedding_model,
            embedding_provider=embedding_provider,
            file_contents=file_contents if file_paths else None,
            **kwargs
        )
    
    except Exception as e:
        traceback.print_exc()
        return {"output": f"Error executing RAG command: {e}", "messages": messages}
@router.route("roll", "generate a video")
def roll_handler(command: str, **kwargs):
    messages = safe_get(kwargs, "messages", [])
    prompt = " ".join(command.split()[1:])
    num_frames = safe_get(kwargs, 'num_frames', 125)
    width = safe_get(kwargs, 'width', 256)
    height = safe_get(kwargs, 'height', 256)
    output_path = safe_get(kwargs, 'output_path', "output.mp4")    
    if not prompt:
        return {"output": "Usage: /roll <your prompt>", "messages": messages}
    try:
        result = gen_video(
            prompt=prompt,
            model=safe_get(kwargs, 'model', NPCSH_VISION_MODEL),
            provider=safe_get(kwargs, 'provider', NPCSH_VISION_PROVIDER),
            npc=safe_get(kwargs, 'npc'),
            num_frames = num_frames,
            width = width,
            height = height,
            output_path=output_path,
            
            **safe_get(kwargs, 'api_kwargs', {})
        )
        return result
    except Exception as e:
        traceback.print_exc()
        return {"output": f"Error generating video: {e}", "messages": messages}
    

@router.route("sample", "Send a prompt directly to the LLM")
def sample_handler(command: str, **kwargs):
    messages = safe_get(kwargs, "messages", [])
    prompt = " ".join(command.split()[1:])
    if not prompt:
        return {"output": "Usage: /sample <your prompt>", "messages": messages}

    try:
        result = get_llm_response(
            prompt=prompt,
            provider=safe_get(kwargs, 'provider'),
            model=safe_get(kwargs, 'model'),
            images=safe_get(kwargs, 'attachments'),
            npc=safe_get(kwargs, 'npc'),
            team=safe_get(kwargs, 'team'),
            messages=messages,
            api_url=safe_get(kwargs, 'api_url'),
            api_key=safe_get(kwargs, 'api_key'),
            context=safe_get(kwargs, 'context'),
            stream=safe_get(kwargs, 'stream')
        )
        return result
    except Exception as e:
        traceback.print_exc()
        return {"output": f"Error sampling LLM: {e}", "messages": messages}

@router.route("search", "Execute a web search command")
def search_handler(command: str, **kwargs):
    """    
    Executes a search command.
    # search commands will bel ike :
    # '/search -p default = google "search term" '
    # '/search -p perplexity ..
    # '/search -p google ..
    # extract provider if its there
    # check for either -p or --p        
    """
    messages = safe_get(kwargs, "messages", [])
    query = " ".join(command.split()[1:])
    
    if not query:
        return {"output": "Usage: /search <query>", "messages": messages}
    search_provider = safe_get(kwargs, 'search_provider', NPCSH_SEARCH_PROVIDER)
    try:
        search_results = search_web(query, provider=search_provider)
        output = "\n".join([f"- {res}" for res in search_results]) if search_results else "No results found."
    except Exception as e:
        traceback.print_exc()
        output = f"Error during web search: {e}"
    return {"output": output, "messages": messages}



@router.route("serve", "Set configuration values")
def serve_handler(command: str, **kwargs):
    #print('calling serve handler')
    #print(kwargs)

    port   = safe_get(kwargs, "port", 5337)
    #print(port, type(port))
    messages = safe_get(kwargs, "messages", [])
    cors = safe_get(kwargs, "cors", None)
    if cors:
        cors_origins = [origin.strip() for origin in cors.split(",")]
    else:
        cors_origins = None

        start_flask_server(
            port=port, 
            cors_origins=cors_origins,
        )


    return {"output": None, "messages": messages}

@router.route("set", "Set configuration values")
def set_handler(command: str, **kwargs):
    messages = safe_get(kwargs, "messages", [])
    parts = command.split(maxsplit=1)
    if len(parts) < 2 or '=' not in parts[1]:
        return {"output": "Usage: /set <key>=<value>", "messages": messages}

    key_value = parts[1]
    key, value = key_value.split('=', 1)
    key = key.strip()
    value = value.strip().strip('"\'')

    try:
        set_npcsh_config_value(key, value)
        output = f"Configuration value '{key}' set."
    except NameError:
        output = "Set function (set_npcsh_config_value) not available."
    except Exception as e:
        traceback.print_exc()
        output = f"Error setting configuration '{key}': {e}"
    return {"output": output, "messages": messages}

@router.route("sleep", "Pause execution for N seconds")
def sleep_handler(command: str, **kwargs):
    messages = safe_get(kwargs, "messages", [])
    parts = command.split()
    try:
        seconds = float(parts[1]) if len(parts) > 1 else 1.0
        if seconds < 0: raise ValueError("Duration must be non-negative")
        time.sleep(seconds)
        output = f"Slept for {seconds} seconds."
    except (ValueError, IndexError):
        output = "Usage: /sleep <seconds>"
    except Exception as e:
        traceback.print_exc()
        output = f"Error during sleep: {e}"
    return {"output": output, "messages": messages}

@router.route("spool", "Enter interactive chat (spool) mode")
def spool_handler(command: str, **kwargs):
    try:
        return enter_spool_mode(
            model=safe_get(kwargs, 'model', NPCSH_CHAT_MODEL),
            provider=safe_get(kwargs, 'provider', NPCSH_CHAT_PROVIDER),
            npc=safe_get(kwargs, 'npc'),
            messages=safe_get(kwargs, 'messages'),
            conversation_id=safe_get(kwargs, 'conversation_id'),
            stream=safe_get(kwargs, 'stream', NPCSH_STREAM_OUTPUT),
            files=safe_get(kwargs, 'files'),
        )
    except Exception as e:
        traceback.print_exc()
        return {"output": f"Error entering spool mode: {e}", "messages": safe_get(kwargs, "messages", [])}


@router.route("jinxs", "Show available jinxs for the current NPC/Team")
def jinxs_handler(command: str, **kwargs):
    npc = safe_get(kwargs, 'npc')
    team = safe_get(kwargs, 'team')
    output = "Available Jinxs:\n"
    jinxs_listed = set()

    def format_jinx(name, jinx_obj):
        desc = getattr(jinx_obj, 'description', 'No description available.')
        return f"- /{name}: {desc}\n"

    if npc and isinstance(npc, NPC) and hasattr(npc, 'jinxs_dict') and npc.jinxs_dict:
        output += f"\n--- Jinxs for NPC: {npc.name} ---\n"
        for name, jinx in sorted(npc.jinxs_dict.items()):
            output += format_jinx(name, jinx)
            jinxs_listed.add(name)

    if team and hasattr(team, 'jinxs_dict') and team.jinxs_dict:
         team_has_jinxs = False
         team_output = ""
         for name, jinx in sorted(team.jinxs_dict.items()):
             if name not in jinxs_listed:
                 team_output += format_jinx(name, jinx)
                 team_has_jinxs = True
         if team_has_jinxs:
             output += f"\n--- Jinxs for Team: {getattr(team, 'name', 'Unnamed Team')} ---\n"
             output += team_output

    if not jinxs_listed and not (team and hasattr(team, 'jinxs_dict') and team.jinxs_dict):
        output = "No jinxs available for the current context."

    return {"output": output.strip(), "messages": safe_get(kwargs, "messages", [])}

@router.route("trigger", "Execute a trigger command")
def trigger_handler(command: str, **kwargs):
    messages = safe_get(kwargs, "messages", [])
    user_command = " ".join(command.split()[1:])
    if not user_command:
        return {"output": "Usage: /trigger <trigger_description>", "messages": messages}
    try:
        return execute_trigger_command(command=user_command, **kwargs)
    except NameError:
        return {"output": "Trigger function (execute_trigger_command) not available.", "messages": messages}
    except Exception as e:
        traceback.print_exc()
        return {"output": f"Error executing trigger: {e}", "messages": messages}
@router.route("vixynt", "Generate images from text descriptions")
def vixynt_handler(command: str, **kwargs):
    npc = safe_get(kwargs, 'npc')
    model = safe_get(kwargs, 'model', NPCSH_IMAGE_GEN_MODEL)
    provider = safe_get(kwargs, 'provider', NPCSH_IMAGE_GEN_PROVIDER)
    height = safe_get(kwargs, 'height', 1024)
    width = safe_get(kwargs, 'width', 1024)
    filename = safe_get(kwargs, 'output_filename', None)
    attachments = None
    if model == NPCSH_CHAT_MODEL: model = NPCSH_IMAGE_GEN_MODEL
    if provider == NPCSH_CHAT_PROVIDER: provider = NPCSH_IMAGE_GEN_PROVIDER

    messages = safe_get(kwargs, 'messages', [])

    filename = None

    prompt_parts = []
    try:
        parts = shlex.split(command)
        for part in parts[1:]:
            if part.startswith("filename="):
                filename = part.split("=", 1)[1]
            elif part.startswith("height="):
                try: 
                    height = int(part.split("=", 1)[1])
                except ValueError:
                    pass
            elif part.startswith("width="):
                try: 
                    width = int(part.split("=", 1)[1])
                except ValueError: 
                    pass
            elif part.startswith("attachments="):  # New parameter for image editing
                # split at comma
                attachments = part.split("=", 1)[1].split(",")

            else:
                prompt_parts.append(part)
    except Exception as parse_err:
        return {"output": f"Error parsing arguments: {parse_err}. Usage: /vixynt <prompt> [filename=...] [height=...] [width=...] [input=...for editing]", "messages": messages}
    user_prompt = " ".join(prompt_parts)
    if not user_prompt:
        return {"output": "Usage: /vixynt <prompt> [filename=...] [height=...] [width=...] [attachments=... for editing]", "messages": messages}

    try:
        image = gen_image(
            prompt=user_prompt,
            model=model,
            provider=provider,
            npc=npc,
            height=height,
            width=width,
            input_images=attachments  
        )
        if filename is None:
            # Generate a filename based on the prompt and the date time
            os.makedirs(os.path.expanduser("~/.npcsh/images/"), exist_ok=True)
            filename = (
                os.path.expanduser("~/.npcsh/images/")
                + f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )    
        image.save(filename)
        image.show()

        if attachments:
            output = f"Image edited and saved to: {filename}"
        else:
            output = f"Image generated and saved to: {filename}"
    except Exception as e:
        traceback.print_exc()
        output = f"Error {'editing' if attachments else 'generating'} image: {e}"

    return {"output": output, "messages": messages}

@router.route("wander", "Enter wander mode (experimental)")
def wander_handler(command: str, **kwargs):
    messages = safe_get(kwargs, "messages", [])
    command_parts = shlex.split(command)
    
    # Extract main parameters
    if len(command_parts) > 1:
        problem = " ".join(command_parts[1:])
    else:
        problem = ""
    
    # Get environment from kwargs if present
    environment = safe_get(kwargs, 'environment')
    include_events = not safe_get(kwargs, 'no_events', False)
    num_events = safe_get(kwargs, 'num_events', 3)
    
    if not problem:
        return {"output": "Usage: /wander <problem or topic to explore> [--environment 'description'] [--no-events] [--num-events N]", "messages": messages}

    try:
        # Get conversation history from the messages if it exists
        previous_insights = ""
        if messages and len(messages) > 1:
            # Extract the recent conversation history to provide context
            recent_messages = messages[-3:] if len(messages) > 3 else messages
            previous_insights = "\n\n".join([msg.get("content", "") for msg in recent_messages if msg.get("role") != "system"])
            
            if previous_insights:
                problem = f"{problem}\n\nPrevious insights to build upon:\n{previous_insights}"
        
        result = enter_wander_mode(
            problem=problem,
            npc=safe_get(kwargs, 'npc'),
            model=safe_get(kwargs, 'model', NPCSH_CHAT_MODEL),
            provider=safe_get(kwargs, 'provider', NPCSH_CHAT_PROVIDER),
            environment=environment,
            include_events=include_events,
            num_events=num_events,
            **safe_get(kwargs, 'api_kwargs', {})
        )
        
        # Convert the result to a string or extract the last insight
        if isinstance(result, list) and result:
            # Get the last session's insight
            last_session = result[-1]
            output = last_session.get("insight", "Wander mode completed.")
        elif isinstance(result, dict):
            output = result.get("insight", "Wander mode completed.")
        elif isinstance(result, tuple) and len(result) > 0:
            output = str(result[-1])
        elif result is not None:
            output = str(result)
        else:
            output = "Wander mode completed."
        
        # Add this insight to the message history
        if messages is not None:
            messages.append({
                "role": "assistant",
                "content": output
            })
            
        return {"output": output, "messages": messages}
    except NameError:
        return {"output": "Wander function (enter_wander_mode) not available.", "messages": messages}
    except Exception as e:
        traceback.print_exc()
        return {"output": f"Error during wander mode: {e}", "messages": messages}

@router.route("yap", "Enter voice chat (yap) mode", shell_only=True)
def whisper_handler(command: str, **kwargs):
    try:
        return enter_yap_mode(
            messages=safe_get(kwargs, 'messages'),
            npc=safe_get(kwargs, 'npc'),
            model=safe_get(kwargs, 'model', NPCSH_CHAT_MODEL),
            provider=safe_get(kwargs, 'provider', NPCSH_CHAT_PROVIDER),
            team=safe_get(kwargs, 'team'),
            stream=safe_get(kwargs, 'stream', NPCSH_STREAM_OUTPUT),
            conversation_id=safe_get(kwargs, 'conversation_id')
            )
    except Exception as e:
        traceback.print_exc()
        return {"output": f"Error entering yap mode: {e}", "messages": safe_get(kwargs, "messages", [])}

@router.route("alicanto", "Conduct deep research with multiple perspectives, identifying gold insights and cliff warnings")
def alicanto_handler(command: str, **kwargs):
    messages = safe_get(kwargs, "messages", [])
    
    # Parse command with shlex to properly handle quoted strings
    parts = shlex.split(command)
    
    # Process arguments
    query = ""
    num_npcs = safe_get(kwargs, 'num_npcs', 5)
    depth = safe_get(kwargs, 'depth', 3)
    exploration_factor = safe_get(kwargs, 'exploration', 0.3)
    creativity_factor = safe_get(kwargs, 'creativity', 0.5)
    output_format = safe_get(kwargs, 'format', 'report')
    
    # Parse command-line arguments
    i = 1  # Skip "alicanto" command
    while i < len(parts):
        if parts[i].startswith('--'):
            option = parts[i][2:]  # Remove '--'
            if option in ['num-npcs', 'npcs']:
                if i + 1 < len(parts) and parts[i + 1].isdigit():
                    num_npcs = int(parts[i + 1])
                    i += 2
                else:
                    i += 1
            elif option in ['depth', 'd']:
                if i + 1 < len(parts) and parts[i + 1].isdigit():
                    depth = int(parts[i + 1])
                    i += 2
                else:
                    i += 1
            elif option in ['exploration', 'e']:
                if i + 1 < len(parts) and parts[i + 1].replace('.', '', 1).isdigit():
                    exploration_factor = float(parts[i + 1])
                    i += 2
                else:
                    i += 1
            elif option in ['creativity', 'c']:
                if i + 1 < len(parts) and parts[i + 1].replace('.', '', 1).isdigit():
                    creativity_factor = float(parts[i + 1])
                    i += 2
                else:
                    i += 1
            elif option in ['format', 'f']:
                if i + 1 < len(parts):
                    output_format = parts[i + 1]
                    i += 2
                else:
                    i += 1
            else:
                # Skip unknown option
                i += 1
        else:
            # This is part of the request
            query += parts[i] + " "
            i += 1
    
    query = query.strip()
    
    # Also apply any kwargs that were passed directly (these override command line args)
    if 'num_npcs' in kwargs:
        try:
            num_npcs = int(kwargs['num_npcs'])
        except ValueError:
            return {"output": "Error: num_npcs must be an integer", "messages": messages}
    
    if 'depth' in kwargs:
        try:
            depth = int(kwargs['depth'])
        except ValueError:
            return {"output": "Error: depth must be an integer", "messages": messages}
    
    if 'exploration' in kwargs:
        try:
            exploration_factor = float(kwargs['exploration'])
        except ValueError:
            return {"output": "Error: exploration must be a float", "messages": messages}
            
    if 'creativity' in kwargs:
        try:
            creativity_factor = float(kwargs['creativity'])
        except ValueError:
            return {"output": "Error: creativity must be a float", "messages": messages}
    
    if not query:
        return {"output": "Usage: /alicanto <research query> [--num-npcs N] [--depth N] [--exploration 0.3] [--creativity 0.5] [--format report|summary|full]", "messages": messages}
    
    try:
        logging.info(f"Starting Alicanto research on: {query}")
        result = alicanto(
            request=query,
            num_npcs=num_npcs,
            depth=depth,
            memory=3,
            context=None,
            model=safe_get(kwargs, 'model', NPCSH_CHAT_MODEL),
            provider=safe_get(kwargs, 'provider', NPCSH_CHAT_PROVIDER),
            exploration_factor=exploration_factor,
            creativity_factor=creativity_factor,
            output_format=output_format
        )
        
        # Format the output based on the result type
        if isinstance(result, dict):
            if "integration" in result:
                output = result["integration"]
            else:
                output = "Alicanto research completed. Full results available in returned data."
        else:
            output = result
            
        return {"output": output, "messages": messages, "alicanto_result": result}
    except Exception as e:
        traceback.print_exc()
        logging.error(f"Error during Alicanto research: {e}")
        return {"output": f"Error during Alicanto research: {e}", "messages": messages}