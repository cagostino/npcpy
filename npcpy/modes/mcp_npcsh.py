#!/usr/bin/env python
# npcsh_mcp.py

import os
import sys
import atexit
import subprocess
import shlex
import re
from datetime import datetime
import argparse
import importlib.metadata
import textwrap
from typing import Optional, List, Dict, Any, Tuple, Union, Generator, Callable
try:
    
    from inspect import isgenerator

except: 
    pass
import shutil
import asyncio
import json
from contextlib import AsyncExitStack

from termcolor import colored, cprint
import chromadb

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from npcpy.modes._state import ShellState, initial_state


from npcpy.npc_sysenv import (
    print_and_process_stream_with_markdown,
    setup_npcsh_config,
    is_npcsh_initialized,
    initialize_base_npcs_if_needed,
    orange,
    interactive_commands,
    BASH_COMMANDS,
    log_action,
    render_markdown,
    get_locally_available_models,
    start_interactive_session,
    get_model_and_provider,
)
from npcpy.routes import router
from npcpy.data.image import capture_screenshot
from npcpy.memory.command_history import (
    CommandHistory,
    save_conversation_message,
)
from npcpy.npc_compiler import NPC, Team
from npcpy.gen.embeddings import get_embeddings
from npcpy.gen.response import get_litellm_response, get_ollama_response
from npcpy.llm_funcs import check_llm_command

import readline

VERSION = importlib.metadata.version("npcpy")

TERMINAL_EDITORS = ["vim", "emacs", "nano"]
EMBEDDINGS_DB_PATH = os.path.expanduser("~/npcsh_chroma.db")
HISTORY_DB_DEFAULT_PATH = os.path.expanduser("~/npcsh_history.db")
READLINE_HISTORY_FILE = os.path.expanduser("~/.npcsh_readline_history")
DEFAULT_NPC_TEAM_PATH = os.path.expanduser("~/.npcsh/npc_team/")
PROJECT_NPC_TEAM_PATH = "./npc_team/"

chroma_client = chromadb.PersistentClient(path=EMBEDDINGS_DB_PATH)

class CommandNotFoundError(Exception):
    pass

class MCPClientNPC:
    def __init__(self, debug: bool = True):
        self.debug = debug
        self.session: Optional[ClientSession] = None 
        self._exit_stack = asyncio.new_event_loop().run_until_complete(self._init_stack())
        self.available_tools_llm: List[Dict[str, Any]] = [] 
        self.tool_map: Dict[str, Callable] = {}  # Map of tool names to functions
        self.server_script_path: Optional[str] = None

    async def _init_stack(self):
        return AsyncExitStack()

    def _log(self, message: str, color: str = "cyan") -> None:
        if self.debug:
            cprint(f"[MCP Client] {message}", color, file=sys.stderr)

    async def _connect_async(self, server_script_path: str) -> None:
        self._log(f"Attempting async connection to MCP server: {server_script_path}")
        self.server_script_path = server_script_path
        command_parts = []
        abs_server_script_path = os.path.abspath(server_script_path)
        if not os.path.exists(abs_server_script_path):
            raise FileNotFoundError(f"MCP server script not found: {abs_server_script_path}")

        if abs_server_script_path.endswith('.py'):
            command_parts = [sys.executable, abs_server_script_path]
        elif abs_server_script_path.endswith('.js'):
            command_parts = ["node", abs_server_script_path]
        elif os.access(abs_server_script_path, os.X_OK):
            command_parts = [abs_server_script_path]
        else:
            raise ValueError(f"Unsupported MCP server script type or not executable: {abs_server_script_path}")

        server_params = StdioServerParameters(command=command_parts[0], args=command_parts[1:], env=os.environ.copy())
        if self.session: 
            await self._exit_stack.aclose()
            self._exit_stack = AsyncExitStack()
        
        stdio_transport = await self._exit_stack.enter_async_context(stdio_client(server_params))
        read, write = stdio_transport
        self.session = await self._exit_stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        
        # Get available tools from MCP server
        response = await self.session.list_tools()
        raw_tools_from_server = response.tools
        
        # Reset our tool containers
        self.available_tools_llm = []
        self.tool_map = {}
        
        # Process MCP tools
        if raw_tools_from_server:
            for mcp_tool_obj in raw_tools_from_server:
                parameters_schema = getattr(mcp_tool_obj, "inputSchema", None)
                if parameters_schema is None:
                    parameters_schema = {"type": "object", "properties": {}}
                
                tool_name = mcp_tool_obj.name
                
                # Create tool definition for LLM
                tool_definition_for_llm = {
                    "type": "function",
                    "function": {
                        "name": tool_name, 
                        "description": mcp_tool_obj.description or f"MCP tool named {tool_name}", 
                        "parameters": parameters_schema 
                    }
                }
                self.available_tools_llm.append(tool_definition_for_llm)
                
                # Create a tool execution function and add to the tool map
                # Use a closure to capture the correct tool name for each function
                def make_tool_executor(t_name):
                    return lambda *args, **kwargs: self.execute_tool(t_name, kwargs if kwargs else args[0] if args else {})

                
                self.tool_map[tool_name] = make_tool_executor(tool_name)

        tool_names_for_log = [t['function']['name'] for t in self.available_tools_llm]
        self._log(f"MCP Connection successful. Discovered and added {len(tool_names_for_log)} tools: {', '.join(tool_names_for_log) if tool_names_for_log else 'None'}")

    def execute_tool(self, tool_name: str, args):
        """Execute an MCP tool using the session."""
        if not self.session:
            return {"error": "No active MCP session"}
        
        # Ensure args is a dictionary
        if not isinstance(args, dict):
            args = args if isinstance(args, dict) else {}
        
        # Special handling for lookup_provider - this is a built-in function, not an MCP tool
        if tool_name == 'lookup_provider':
            # Import the actual lookup_provider function
            from npcpy.npc_sysenv import lookup_provider
            try:
                # Call the actual function directly
                model_name = args.get('model')
                if model_name:
                    result = lookup_provider(model_name)
                    return {"provider": result, "model": model_name}
                else:
                    return {"error": "Model name not provided for lookup_provider"}
            except Exception as e:
                return {"error": f"Error executing lookup_provider: {str(e)}"}
        
        # For actual MCP tools, use the correct API
        async def _execute_async():
            try:
                # The correct method to call an MCP tool might be different
                # It could be something like:
                result = await self.session.call_tool(tool_name, args)
                # Or:
                # result = await self.session.invoke_tool(tool_name, **args)
                # Check your MCP client API documentation
                return result
            except Exception as e:
                return {"error": f"Error executing MCP tool {tool_name}: {str(e)}"}
        
        # Run the async function
        loop = asyncio.get_event_loop_policy().get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(_execute_async())

    def connect_to_server_sync(self, server_script_path: str) -> bool:
        loop = asyncio.get_event_loop_policy().get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        loop.run_until_complete(self._connect_async(server_script_path))
        return True 

    async def _disconnect_async(self):
        if self.session:
            self._log("Disconnecting MCP session.")
            await self._exit_stack.aclose()
            self.session = None
            self.available_tools_llm = []
            self.tool_map = {}
            self.server_script_path = None
        else:
            self._log("No active MCP session to disconnect.", "yellow")

    def disconnect_sync(self):
        loop = asyncio.get_event_loop_policy().get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        loop.run_until_complete(self._disconnect_async())
        self.session = None
        self.available_tools_llm = []
        self.tool_map = {}
        
def readline_safe_prompt(prompt: str) -> str:
    ansi_escape = re.compile(r"(\033\[[0-9;]*[a-zA-Z])")
    return ansi_escape.sub(r"\001\1\002", prompt)

def print_tools_info(tools_list: List[Dict[str, Any]]): 
    output = "Available tools:\n"
    for tool_item in tools_list:
        name = "UnknownTool"
        description = "No description"
        if isinstance(tool_item, dict):
            if 'function' in tool_item and 'name' in tool_item['function']:
                name = tool_item['function']['name']
            if 'function' in tool_item and 'description' in tool_item['function']:
                description = tool_item['function']['description']
        output += f"  {name}\n   Description: {description}\n"
    return output

def open_terminal_editor(command: str) -> str:
    os.system(command)
    return 'Terminal editor closed.'

def get_multiline_input(prompt: str) -> str:
    lines = []
    current_prompt = prompt
    while True:
        line = input(current_prompt) 
        if line.endswith("\\"):
            lines.append(line[:-1])
            current_prompt = readline_safe_prompt("> ")
        else:
            lines.append(line)
            break
    return "\n".join(lines)

def split_by_pipes(command: str) -> List[str]:
    parts = []
    current = ""
    in_single_quote = False
    in_double_quote = False
    escape = False
    for char in command:
        if escape:
            current += char
            escape = False
        elif char == '\\':
            escape = True
            current += char
        elif char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            current += char
        elif char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            current += char
        elif char == '|' and not in_single_quote and not in_double_quote:
            parts.append(current.strip())
            current = ""
        else:
            current += char
    if current:
        parts.append(current.strip())
    return parts

def parse_command_safely(cmd: str) -> List[str]:
    return shlex.split(cmd)

def get_file_color(filepath: str) -> tuple:
    if not os.path.exists(filepath):
        return "grey", []
    if os.path.isdir(filepath):
        return "blue", ["bold"]
    if os.access(filepath, os.X_OK) and not os.path.isdir(filepath):
        return "green", ["bold"]
    if filepath.endswith((".zip", ".tar", ".gz", ".bz2", ".xz", ".7z")):
        return "red", []
    if filepath.endswith((".py", ".pyw")):
        return "yellow", []
    return "white", []

def format_file_listing(output: str) -> str:
    colored_lines = []
    current_dir = os.getcwd()
    for line in output.strip().split("\n"):
        parts = line.split()
        if not parts:
            colored_lines.append(line)
            continue
        filepath_guess = parts[-1]
        potential_path = os.path.join(current_dir, filepath_guess)
        color, attrs = get_file_color(potential_path)
        colored_filepath = colored(filepath_guess, color, attrs=attrs)
        if len(parts) > 1 :
            colored_line = " ".join(parts[:-1] + [colored_filepath])
        else:
            colored_line = colored_filepath
        colored_lines.append(colored_line)
    return "\n".join(colored_lines)

def wrap_text(text: str, width: int = 80) -> str:
    lines = []
    for paragraph in text.split("\n"):
        if len(paragraph) > width:
            lines.extend(textwrap.wrap(paragraph, width=width, replace_whitespace=False, drop_whitespace=False))
        else:
            lines.append(paragraph)
    return "\n".join(lines)

def setup_readline_completer() -> str:
    readline.read_history_file(READLINE_HISTORY_FILE) 
    readline.set_history_length(1000)
    readline.parse_and_bind("set enable-bracketed-paste on")
    readline.parse_and_bind(r'"\e[A": history-search-backward')
    readline.parse_and_bind(r'"\e[B": history-search-forward')
    if sys.platform == "darwin":
        readline.parse_and_bind("bind ^I rl_complete")
    else:
        readline.parse_and_bind("tab: complete")
    return READLINE_HISTORY_FILE

def save_readline_history_on_exit():
    readline.write_history_file(READLINE_HISTORY_FILE)

valid_commands_list_global = list(router.routes.keys()) + list(interactive_commands.keys()) + ["cd", "exit", "quit"] + BASH_COMMANDS

def completer_func(text: str, state_idx: int) -> Optional[str]:
    buffer = readline.get_line_buffer()
    line_parts = parse_command_safely(buffer) 
    is_command_start = not line_parts or (len(line_parts) == 1 and not buffer.endswith(' '))
    if is_command_start and not text.startswith('-'):
        cmd_matches = [cmd + ' ' for cmd in valid_commands_list_global if cmd.startswith(text)]
        if state_idx < len(cmd_matches):
            return cmd_matches[state_idx]
        else:
            return None
    if text and (not text.startswith('/') or os.path.exists(os.path.dirname(text))):
        basedir = os.path.dirname(text)
        prefix = os.path.basename(text)
        search_dir = basedir if basedir else '.'
        matches = [os.path.join(basedir, f) + ('/' if os.path.isdir(os.path.join(search_dir, f)) else ' ')
                    for f in os.listdir(search_dir) if f.startswith(prefix)]
        if state_idx < len(matches):
            return matches[state_idx]
        else:
            return None
    return None

def store_command_embeddings(command: str, output: Any, state: ShellState):
    if not chroma_client or not state.embedding_model or not state.embedding_provider:
        return
    if not command and not output:
        return
    output_str = str(output) if output else ""
    if not command and not output_str:
        return
    texts_to_embed = [command, output_str]
    embeddings = get_embeddings(texts_to_embed, state.embedding_model, state.embedding_provider)
    if not embeddings or len(embeddings) != 2:
        return
    timestamp = datetime.now().isoformat()
    npc_name = state.npc.name if isinstance(state.npc, NPC) else str(state.npc)
    metadata = [{"type": "command", "timestamp": timestamp, "path": state.current_path, "npc": npc_name, "conversation_id": state.conversation_id},
                {"type": "response", "timestamp": timestamp, "path": state.current_path, "npc": npc_name, "conversation_id": state.conversation_id}]
    collection_name = f"{state.embedding_provider}_{state.embedding_model}_embeddings"
    collection = chroma_client.get_or_create_collection(collection_name)
    ids = [f"cmd_{timestamp}_{hash(command)}", f"resp_{timestamp}_{hash(output_str)}"]
    collection.add(embeddings=embeddings, documents=texts_to_embed, metadatas=metadata, ids=ids)

def handle_interactive_command(cmd_parts: List[str], state: ShellState) -> Tuple[ShellState, str]:
    command_name = cmd_parts[0]
    print(f"Starting interactive {command_name} session...", file=sys.stderr)
    return_code = start_interactive_session(interactive_commands[command_name], cmd_parts[1:])
    output = f"Interactive {command_name} session ended with code {return_code}"
    return state, output

def handle_cd_command(cmd_parts: List[str], state: ShellState) -> Tuple[ShellState, str]:
    target_path = cmd_parts[1] if len(cmd_parts) > 1 else os.path.expanduser("~")
    os.chdir(target_path) 
    state.current_path = os.getcwd()
    output = f"Changed directory to {state.current_path}"
    return state, output

def handle_bash_command(cmd_parts: List[str], cmd_str: str, stdin_input: Optional[str], state: ShellState) -> Tuple[ShellState, str]:
    command_name = cmd_parts[0]
    if command_name in TERMINAL_EDITORS:
        return state, open_terminal_editor(cmd_str)
    
    process = subprocess.Popen(
        cmd_parts, 
        stdin=subprocess.PIPE if stdin_input is not None else None,
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True, 
        cwd=state.current_path
    )
    stdout, stderr = process.communicate(input=stdin_input)

    if process.returncode != 0:
        err_msg = stderr.strip() if stderr else f"Command '{cmd_str}' failed (code {process.returncode})."
        return state, (stdout.strip() + ("\n" + colored(f"stderr: {err_msg}", "red") if err_msg else "")).strip()
    
    output = stdout.strip() if stdout else ""
    if stderr:
        print(colored(f"stderr: {stderr.strip()}", "yellow"), file=sys.stderr)
    if command_name in ["ls", "find", "dir"]:
        output = format_file_listing(output)
    elif not output and process.returncode == 0 and not stderr:
        output = ""
    return state, output

def execute_slash_command_local(command: str, stdin_input: Optional[str], state: ShellState, stream: bool) -> Tuple[ShellState, Any]:
    command_parts = command.split()
    command_name = command_parts[0].lstrip('/')
    
    handler = router.get_route(command_name)
    if handler:
        handler_kwargs = {'stream': stream, 'npc': state.npc, 'team': state.team, 'messages': state.messages, 
                          'model': state.chat_model, 'provider': state.chat_provider, 'api_url': state.api_url, 'api_key': state.api_key}
        if stdin_input is not None:
            handler_kwargs['stdin_input'] = stdin_input
        result_dict = handler(command, **handler_kwargs)
        if isinstance(result_dict, dict):
            output = result_dict.get("output") or result_dict.get("response")
            state.messages = result_dict.get("messages", state.messages)
            return state, output
        else:
            return state, result_dict

    active_npc = state.npc if isinstance(state.npc, NPC) else None
    if active_npc and command_name in active_npc.tools_dict:
        return state, active_npc.tools_dict[command_name].run(*(command_parts[1:]), state=state, stdin_input=stdin_input, messages=state.messages)
    
    if state.team and command_name in state.team.tools_dict:
        return state, state.team.tools_dict[command_name].run(*(command_parts[1:]), state=state, stdin_input=stdin_input, messages=state.messages)

    if state.team and command_name in state.team.npcs:
        state.npc = state.team.npcs[command_name]
        return state, f"Switched to NPC: {state.npc.name}"
    
    return state, colored(f"Unknown slash command or tool: {command_name}", "red")

def process_pipeline_command(cmd_segment: str, stdin_input: Optional[str], state: ShellState, stream_final: bool) -> Tuple[ShellState, Any]:
    if not cmd_segment:
        return state, stdin_input

    available_models_all = get_locally_available_models(state.current_path)
    model_override, provider_override, cmd_cleaned = get_model_and_provider(cmd_segment, [i for _, i in available_models_all.items()])
    cmd_to_process = cmd_cleaned.strip()
    if not cmd_to_process:
        return state, stdin_input

    exec_model = model_override or state.chat_model    
    exec_provider = provider_override or state.chat_provider 

    if cmd_to_process.startswith("/"):
        return execute_slash_command_local(cmd_to_process, stdin_input, state, stream_final)
    
    cmd_parts = parse_command_safely(cmd_to_process)
    if not cmd_parts:
        return state, stdin_input
    command_name = cmd_parts[0]

    if command_name in interactive_commands:
        return handle_interactive_command(cmd_parts, state)
    
    if command_name == "cd":
        return handle_cd_command(cmd_parts, state)
    
    if command_name in BASH_COMMANDS:
        return handle_bash_command(cmd_parts, cmd_to_process, stdin_input, state)
    else:
        full_llm_cmd = f"{cmd_to_process} {stdin_input}" if stdin_input else cmd_to_process
        
        mcp_tools_for_upstream = None
        if hasattr(state, 'mcp_client') and state.mcp_client and state.mcp_client.available_tools_llm:
            mcp_tools_for_upstream = state.mcp_client.available_tools_llm
        if hasattr(state.mcp_client, 'tool_map'):
            mcp_tool_map = state.mcp_client.tool_map

        result_dict = check_llm_command(
            command=full_llm_cmd,
            model=exec_model,
            provider=exec_provider,
            api_url=state.api_url,
            api_key=state.api_key,
            npc=state.npc,
            team=state.team,
            messages=state.messages,
            images=state.attachments,
            stream=stream_final,
            tools=mcp_tools_for_upstream, 
            tool_map=mcp_tool_map, 
            
            context=state 
        )
        state.messages = result_dict.get("messages", state.messages)
        return state, result_dict.get("output")

def check_mode_switch(command:str , state: ShellState) -> Tuple[bool, ShellState]:
    if command in ['/cmd', '/agent', '/chat', '/ride']:
        state.current_mode = command[1:]
        return True, state     
    return False, state

def execute_command(command: str, state: ShellState) -> Tuple[ShellState, Any]:
    print(f'execute_command {command} {state.current_mode}', file=sys.stderr)
    if not command.strip():
        return state, ""
    
    mode_change, state = check_mode_switch(command, state)
    if mode_change:
        return state, f'Mode changed to {state.current_mode}.'

    original_command_for_embedding = command
    
    if state.current_mode == 'agent':
        commands = split_by_pipes(command)
        stdin_for_next = None
        final_output = None
        current_state = state 
        for i, cmd_segment in enumerate(commands):
            is_last_command = (i == len(commands) - 1)
            stream_this_segment = is_last_command and state.stream_output
            current_state, output = process_pipeline_command(cmd_segment.strip(), stdin_for_next, current_state, stream_this_segment)
            if is_last_command:
                final_output = output
            if isinstance(output, str):
                stdin_for_next = output
            elif isgenerator(output):
                if not stream_this_segment:
                    full_stream_output = "".join(map(str, output))
                    stdin_for_next = full_stream_output
                    if is_last_command:
                        final_output = full_stream_output
                else:
                    stdin_for_next = None
                    final_output = output
            elif output is not None:
                stdin_for_next = str(output)
            else:
                stdin_for_next = None
        if final_output is not None and not (isgenerator(final_output) and current_state.stream_output):
            store_command_embeddings(original_command_for_embedding, final_output, current_state)
        return current_state, final_output

    elif state.current_mode == 'chat':
        chat_messages = state.messages + [{"role": "user", "content": command}]
        provider_to_use = state.chat_provider
        if not provider_to_use and state.chat_model:
            from npcpy.npc_sysenv import lookup_provider
            provider_to_use = lookup_provider(state.chat_model)
        
        if provider_to_use == "ollama":
            response_dict = get_ollama_response(model=state.chat_model, messages=chat_messages, npc=state.npc, stream=state.stream_output, api_url=state.api_url)
        else:
            response_dict = get_litellm_response(model=state.chat_model, provider=provider_to_use, messages=chat_messages, npc=state.npc, stream=state.stream_output, api_key=state.api_key, api_url=state.api_url)
        
        state.messages = response_dict.get("messages", state.messages)
        return state, response_dict.get("response")

    elif state.current_mode == 'cmd':
        mcp_tools_for_upstream = None
        if hasattr(state, 'mcp_client') and state.mcp_client and state.mcp_client.available_tools_llm:
            mcp_tools_for_upstream = state.mcp_client.available_tools_llm

        result_dict = check_llm_command(
            command=command,
            model=state.chat_model,
            provider=state.chat_provider,
            api_url=state.api_url,
            api_key=state.api_key,
            npc=state.npc,
            team=state.team,
            messages=state.messages,
            images=state.attachments,
            stream=state.stream_output,
            tools=mcp_tools_for_upstream, 
            context=state 
        )
        state.messages = result_dict.get("messages", state.messages)
        return state, result_dict.get("output")

    elif state.current_mode == 'ride':     
        print('Ride mode to be implemented soon', file=sys.stderr)   
        return state, None
    
    return state, "Unknown execution mode."

def check_deprecation_warnings():
    if os.getenv("NPCSH_MODEL"):
        cprint("Deprecation Warning: NPCSH_MODEL/PROVIDER deprecated. Use NPCSH_CHAT_MODEL/PROVIDER.", "yellow", file=sys.stderr)

def print_welcome_message():
    print("""
Welcome to \033[1;94mnpc\033[0m\033[1;38;5;202msh\033[0m (MCP Edition)!
\033[1;94m                    \033[0m\033[1;38;5;202m               \\\\
\033[1;94m _ __   _ __    ___ \033[0m\033[1;38;5;202m ___  | |___    \\\\
\033[1;94m| '_ \\ | '_ \\  / __|\033[0m\033[1;38;5;202m/ __/ | |_ _|    \\\\
\033[1;94m| | | || |_) |( |__ \033[0m\033[1;38;5;202m\_  \ | | | |    //
\033[1;94m|_| |_|| .__/  \___|\033[0m\033[1;38;5;202m|___/ |_| |_|   //
       \033[1;94m| |          \033[0m\033[1;38;5;202m              //
       \033[1;94m| |
       \033[1;94m|_|

Type '/help' for commands. MCP server might be auto-connected.
    """)

def setup_shell_mcp(cli_args: argparse.Namespace, shell_initial_state: ShellState) -> Tuple[CommandHistory, ShellState]:
    check_deprecation_warnings()
    setup_npcsh_config()

    db_path = os.getenv("NPCSH_DB_PATH", HISTORY_DB_DEFAULT_PATH)
    db_path = os.path.expanduser(db_path)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    command_history = CommandHistory(db_path)

    readline.set_completer(completer_func)
    setup_readline_completer() 
    atexit.register(save_readline_history_on_exit) 
    atexit.register(command_history.close)        

    npc_directory = PROJECT_NPC_TEAM_PATH if os.path.exists(PROJECT_NPC_TEAM_PATH) else DEFAULT_NPC_TEAM_PATH
    os.makedirs(npc_directory, exist_ok=True)

    if not is_npcsh_initialized():
        print("Initializing NPCSH...", file=sys.stderr)
        initialize_base_npcs_if_needed(db_path)
        print("NPCSH initialization complete. Restart or source ~/.npcshrc.", file=sys.stderr)

    current_shell_state = shell_initial_state 
    
    team = Team(team_path=npc_directory)
    current_shell_state.team = team 
    
    default_npc_name = getattr(team, 'default_npc_name', None)
    if default_npc_name and default_npc_name in team.npcs:
        current_shell_state.npc = team.npcs[default_npc_name]
    elif team.npcs:
        current_shell_state.npc = next(iter(team.npcs.values()))
        cprint(f"No default NPC in team, using '{current_shell_state.npc.name}'.", "yellow", file=sys.stderr)
    else:
        sibiji_path = os.path.join(DEFAULT_NPC_TEAM_PATH, "sibiji.npc")
        if os.path.exists(sibiji_path):
            current_shell_state.npc = NPC(file=sibiji_path)
        else:
            cprint(f"Warning: No NPCs in team and 'sibiji.npc' not found.", "red", file=sys.stderr)

    mcp_server_path = cli_args.mcp_server_path or os.getenv("NPCSH_MCP_SERVER_PATH")
    if not mcp_server_path and current_shell_state.team:
        team_mcp_servers = getattr(current_shell_state.team, 'mcp_servers', [])
        default_mcp_name = getattr(current_shell_state.team, 'default_mcp_server_name', None)
        processed_team_mcp_servers: List[Dict[str, str]] = []
        if isinstance(team_mcp_servers, list):
            for item in team_mcp_servers:
                if isinstance(item, dict) and 'script_path' in item:
                    processed_team_mcp_servers.append({'name': item.get('name', os.path.basename(item['script_path'])), 'script_path': item['script_path']})
                elif isinstance(item, str):
                    processed_team_mcp_servers.append({'name': os.path.basename(item), 'script_path': item})
        
        if default_mcp_name:
            for server_info in processed_team_mcp_servers:
                if server_info.get('name') == default_mcp_name:
                    mcp_server_path = server_info.get('script_path')
                    break
        elif len(processed_team_mcp_servers) == 1:
            mcp_server_path = processed_team_mcp_servers[0].get('script_path')
        
        if mcp_server_path and not os.path.isabs(mcp_server_path) and hasattr(current_shell_state.team, 'path') and current_shell_state.team.path:
             mcp_server_path = os.path.join(current_shell_state.team.path, mcp_server_path)

    if mcp_server_path:
        cprint(f"Attempting to connect to MCP server: {mcp_server_path}", "yellow", file=sys.stderr)
        mcp_client = MCPClientNPC()
        if mcp_client.connect_to_server_sync(mcp_server_path):
            if hasattr(current_shell_state, 'mcp_client'): # Check if the imported ShellState has the attr
                current_shell_state.mcp_client = mcp_client
                atexit.register(mcp_client.disconnect_sync)
            else:
                cprint("CRITICAL ERROR: Imported ShellState object from npcpy.modes._state does not have 'mcp_client' attribute.", "red", file=sys.stderr)
                cprint("Please add 'mcp_client: Optional[Any] = None' to its definition.", "red", file=sys.stderr)
                # Decide if script should exit here if mcp_client cannot be set on state
                # For now, it will continue but mcp features might fail later if state.mcp_client is accessed


    if sys.stdin.isatty():
        print_welcome_message()
    return command_history, current_shell_state

def process_result(user_input: str, result_state: ShellState, output: Any, command_history: CommandHistory):
    npc_name = result_state.npc.name if isinstance(result_state.npc, NPC) else str(result_state.npc)
    team_name = result_state.team.name if isinstance(result_state.team, Team) else str(result_state.team)
    save_conversation_message(command_history, result_state.conversation_id, "user", user_input,
                              wd=result_state.current_path, model=result_state.chat_model, provider=result_state.chat_provider,
                              npc=npc_name, team=team_name, attachments=result_state.attachments)
    result_state.attachments = None
    final_output_str = None
    
    if result_state.stream_output and (isgenerator(output) or hasattr(output, '__aiter__')):
        try:
            final_output_str = print_and_process_stream_with_markdown(output, result_state.chat_model, result_state.chat_provider)
        except AttributeError as e:
            if isinstance(output, str):
                if len(output) > 0:
                    final_output_str = output
                    render_markdown(final_output_str) 
    elif output is not None:
        final_output_str = str(output)
        render_markdown(final_output_str) 
    
    if final_output_str is not None:
        is_new_assistant_message = not result_state.messages or result_state.messages[-1].get("role") != "assistant"
        if is_new_assistant_message:
            result_state.messages.append({"role": "assistant", "content": final_output_str})
        elif result_state.messages[-1].get("content") is None:
            result_state.messages[-1]["content"] = final_output_str

    print() 
    if final_output_str:
        save_conversation_message(command_history, result_state.conversation_id, "assistant", final_output_str,
                                  wd=result_state.current_path, model=result_state.chat_model, provider=result_state.chat_provider,
                                  npc=npc_name, team=team_name)

def run_repl(command_history: CommandHistory, shell_state_instance: ShellState):
    state = shell_state_instance 
    print(f'Using {state.current_mode} mode. Use /agent, /cmd, or /chat to switch.', file=sys.stderr)
    if state.npc:
        print(f'Current NPC: {state.npc.name if isinstance(state.npc, NPC) else state.npc}', file=sys.stderr)
    if hasattr(state, 'mcp_client') and state.mcp_client and state.mcp_client.server_script_path:
        print(f'MCP Client connected to: {state.mcp_client.server_script_path}', file=sys.stderr)
    elif hasattr(state, 'mcp_client') and state.mcp_client:
        print(f'MCP Client active but not initially connected.', file=sys.stderr)

    while True:
        cwd_colored = colored(os.path.basename(state.current_path), "blue")
        prompt_npc_part = f":{orange(state.npc.name)}" if isinstance(state.npc, NPC) else (f":{orange(str(state.npc))}" if state.npc else "")
        prompt = readline_safe_prompt(f"{cwd_colored}{prompt_npc_part}> ")
        user_input = get_multiline_input(prompt).strip() 
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!", file=sys.stderr)
            break
        state.current_path = os.getcwd()
        state, output = execute_command(user_input, state)
        process_result(user_input, state, output, command_history)

def run_non_interactive(command_history: CommandHistory, shell_state_instance: ShellState):
    state = shell_state_instance 
    for line in sys.stdin: 
        user_input = line.strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
            break
        state.current_path = os.getcwd()
        state, output = execute_command(user_input, state)
        if state.stream_output and (isgenerator(output) or hasattr(output, '__aiter__')):
             for chunk in output:
                 print(str(chunk), end='')
             print()
        elif output is not None:
            print(output)

def main() -> None:
    parser = argparse.ArgumentParser(description="npcsh (MCP Edition) - An NPC-powered shell with MCP.")
    parser.add_argument("-v", "--version", action="version", version=f"npcsh_mcp version {VERSION}")
    parser.add_argument("-c", "--command", type=str, help="Execute a single command and exit.")
    parser.add_argument("--mcp-server-path", type=str, help="Path to an MCP server script to connect to.")
    args = parser.parse_args()

    command_history, shell_state_instance = setup_shell_mcp(args, initial_state)

    if args.command:
         state_for_cmd = shell_state_instance 
         state_for_cmd.current_path = os.getcwd()
         final_state, output = execute_command(args.command, state_for_cmd)
         if final_state.stream_output and (isgenerator(output) or hasattr(output, '__aiter__')):
              for chunk in output:
                  print(str(chunk), end='')
              print()
         elif output is not None:
              print(output)
    elif not sys.stdin.isatty():
        run_non_interactive(command_history, shell_state_instance)
    else:
        try:
            run_repl(command_history, shell_state_instance)
        except KeyboardInterrupt:
            print("\nnpcsh terminated by user (KeyboardInterrupt).", file=sys.stderr)
        except EOFError:
            print("\nnpcsh terminated (EOF).", file=sys.stderr)

if __name__ == "__main__":
    main()