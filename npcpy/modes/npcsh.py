# Standard Library Imports
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
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
from inspect import isgenerator
import platform
try:
    from termcolor import colored
except: 
    pass

try:
    import chromadb
except ImportError:
    chromadb = None

# Local Application Imports
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
from npcpy.memory.knowledge_graph import breathe
from npcpy.memory.sleep import sleep, forget
from npcpy.memory.memory_integration import register_memory_commands, execute_breathe, execute_sleep, execute_forget
from npcpy.npc_compiler import NPC, Team
from npcpy.llm_funcs import check_llm_command, get_llm_response, execute_llm_command
from npcpy.gen.embeddings import get_embeddings
try:
    import readline
except:
    print('no readline support, some features may not work as desired. ')
# --- Constants ---
try:
    VERSION = importlib.metadata.version("npcpy")
except importlib.metadata.PackageNotFoundError:
    VERSION = "unknown"

TERMINAL_EDITORS = ["vim", "emacs", "nano"]
EMBEDDINGS_DB_PATH = os.path.expanduser("~/npcsh_chroma.db")
HISTORY_DB_DEFAULT_PATH = os.path.expanduser("~/npcsh_history.db")
READLINE_HISTORY_FILE = os.path.expanduser("~/.npcsh_readline_history")
DEFAULT_NPC_TEAM_PATH = os.path.expanduser("~/.npcsh/npc_team/")
PROJECT_NPC_TEAM_PATH = "./npc_team/"

# --- Global Clients ---
try:
    chroma_client = chromadb.PersistentClient(path=EMBEDDINGS_DB_PATH) if chromadb else None
except Exception as e:
    print(f"Warning: Failed to initialize ChromaDB client at {EMBEDDINGS_DB_PATH}: {e}")
    chroma_client = None

# --- Custom Exceptions ---
class CommandNotFoundError(Exception):
    pass


from npcpy.modes._state import initial_state, ShellState

def readline_safe_prompt(prompt: str) -> str:
    ansi_escape = re.compile(r"(\033\[[0-9;]*[a-zA-Z])")
    return ansi_escape.sub(r"\001\1\002", prompt)

def print_jinxs(jinxs):
    output = "Available jinxs:\n"
    for jinx in jinxs:
        output += f"  {jinx.jinx_name}\n"
        output += f"   Description: {jinx.description}\n"
        output += f"   Inputs: {jinx.inputs}\n"
    return output

def open_terminal_editor(command: str) -> str:
    try:
        os.system(command)
        return 'Terminal editor closed.'
    except Exception as e:
        return f"Error opening terminal editor: {e}"

def get_multiline_input(prompt: str) -> str:
    lines = []
    current_prompt = prompt
    while True:
        try:
            line = input(current_prompt)
            if line.endswith("\\"):
                lines.append(line[:-1])
                current_prompt = readline_safe_prompt("> ")
            else:
                lines.append(line)
                break
        except EOFError:
            print("Goodbye!")
            sys.exit(0)
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
            in_double_quote = not in_single_quote
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
    try:
        return shlex.split(cmd)
    except ValueError as e:
        if "No closing quotation" in str(e):
            if cmd.count('"') % 2 == 1:
                cmd += '"'
            elif cmd.count("'") % 2 == 1:
                cmd += "'"
            try:
                return shlex.split(cmd)
            except ValueError:
                return cmd.split()
        else:
            return cmd.split()

def get_file_color(filepath: str) -> tuple:
    if not os.path.exists(filepath):
         return "grey", []
    if os.path.isdir(filepath):
        return "blue", ["bold"]
    elif os.access(filepath, os.X_OK) and not os.path.isdir(filepath):
        return "green", ["bold"]
    elif filepath.endswith((".zip", ".tar", ".gz", ".bz2", ".xz", ".7z")):
        return "red", []
    elif filepath.endswith((".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff")):
        return "magenta", []
    elif filepath.endswith((".py", ".pyw")):
        return "yellow", []
    elif filepath.endswith((".sh", ".bash", ".zsh")):
        return "green", []
    elif filepath.endswith((".c", ".cpp", ".h", ".hpp")):
        return "cyan", []
    elif filepath.endswith((".js", ".ts", ".jsx", ".tsx")):
        return "yellow", []
    elif filepath.endswith((".html", ".css", ".scss", ".sass")):
        return "magenta", []
    elif filepath.endswith((".md", ".txt", ".log")):
        return "white", []
    elif os.path.basename(filepath).startswith("."):
        return "cyan", []
    else:
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
             # Handle cases like 'ls -l' where filename is last
             colored_line = " ".join(parts[:-1] + [colored_filepath])
        else:
             # Handle cases where line is just the filename
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

# --- Readline Setup and Completion ---

def setup_readline() -> str:
    try:
        readline.read_history_file(READLINE_HISTORY_FILE)

        readline.set_history_length(1000)
        readline.parse_and_bind("set enable-bracketed-paste on")
        readline.parse_and_bind(r'"\e[A": history-search-backward')
        readline.parse_and_bind(r'"\e[B": history-search-forward')
        readline.parse_and_bind(r'"\C-r": reverse-search-history')
        readline.parse_and_bind(r'\C-e: end-of-line')
        readline.parse_and_bind(r'\C-a: beginning-of-line')
        if sys.platform == "darwin":
            readline.parse_and_bind("bind ^I rl_complete")
        else:
            readline.parse_and_bind("tab: complete")

        return READLINE_HISTORY_FILE


    except FileNotFoundError:
        pass
    except OSError as e:
        print(f"Warning: Could not read readline history file {READLINE_HISTORY_FILE}: {e}")

def save_readline_history():
    try:
        readline.write_history_file(READLINE_HISTORY_FILE)
    except OSError as e:
        print(f"Warning: Could not write readline history file {READLINE_HISTORY_FILE}: {e}")


# --- Placeholder for actual valid commands ---
# This should be populated dynamically based on router, builtins, and maybe PATH executables
valid_commands_list = list(router.routes.keys()) + list(interactive_commands.keys()) + ["cd", "exit", "quit"] + BASH_COMMANDS

def complete(text: str, state: int) -> Optional[str]:
    try:
        buffer = readline.get_line_buffer()
    except:
        print('couldnt get readline buffer')
    line_parts = parse_command_safely(buffer) # Use safer parsing
    word_before_cursor = ""
    if len(line_parts) > 0 and not buffer.endswith(' '):
         current_word = line_parts[-1]
    else:
         current_word = "" # Completing after a space

    try:
        # Command completion (start of line or after pipe/semicolon)
        # This needs refinement to detect context better
        is_command_start = not line_parts or (len(line_parts) == 1 and not buffer.endswith(' ')) # Basic check
        if is_command_start and not text.startswith('-'): # Don't complete options as commands
            cmd_matches = [cmd + ' ' for cmd in valid_commands_list if cmd.startswith(text)]
            # Add executables from PATH? (Can be slow)
            # path_executables = [f + ' ' for f in shutil.get_exec_path() if os.path.basename(f).startswith(text)]
            # cmd_matches.extend(path_executables)
            return cmd_matches[state]

        # File/Directory completion (basic)
        # Improve context awareness (e.g., after 'cd', 'ls', 'cat', etc.)
        if text and (not text.startswith('/') or os.path.exists(os.path.dirname(text))):
             basedir = os.path.dirname(text)
             prefix = os.path.basename(text)
             search_dir = basedir if basedir else '.'
             try:
                 matches = [os.path.join(basedir, f) + ('/' if os.path.isdir(os.path.join(search_dir, f)) else ' ')
                            for f in os.listdir(search_dir) if f.startswith(prefix)]
                 return matches[state]
             except OSError: # Handle permission denied etc.
                  return None

    except IndexError:
        return None
    except Exception: # Catch broad exceptions during completion
        return None

    return None


# --- Command Execution Logic ---

def store_command_embeddings(command: str, output: Any, state: ShellState):
    if not chroma_client or not state.embedding_model or not state.embedding_provider:
        if not chroma_client: print("Warning: ChromaDB client not available for embeddings.", file=sys.stderr)
        return
    if not command and not output:
        return

    try:
        output_str = str(output) if output else ""
        if not command and not output_str: return # Avoid empty embeddings

        texts_to_embed = [command, output_str]

        embeddings = get_embeddings(
            texts_to_embed,
            state.embedding_model,
            state.embedding_provider,
        )

        if not embeddings or len(embeddings) != 2:
             print(f"Warning: Failed to generate embeddings for command: {command[:50]}...", file=sys.stderr)
             return

        timestamp = datetime.now().isoformat()
        npc_name = state.npc.name if isinstance(state.npc, NPC) else state.npc

        metadata = [
            {
                "type": "command", "timestamp": timestamp, "path": state.current_path,
                "npc": npc_name, "conversation_id": state.conversation_id,
            },
            {
                "type": "response", "timestamp": timestamp, "path": state.current_path,
                "npc": npc_name, "conversation_id": state.conversation_id,
            },
        ]

        collection_name = f"{state.embedding_provider}_{state.embedding_model}_embeddings"
        try:
            collection = chroma_client.get_or_create_collection(collection_name)
            ids = [f"cmd_{timestamp}_{hash(command)}", f"resp_{timestamp}_{hash(output_str)}"]

            collection.add(
                embeddings=embeddings,
                documents=texts_to_embed,
                metadatas=metadata,
                ids=ids,
            )
        except Exception as e:
            print(f"Warning: Failed to add embeddings to collection '{collection_name}': {e}", file=sys.stderr)

    except Exception as e:
        print(f"Warning: Failed to store embeddings: {e}", file=sys.stderr)


def handle_interactive_command(cmd_parts: List[str], state: ShellState) -> Tuple[ShellState, str]:
    command_name = cmd_parts[0]
    print(f"Starting interactive {command_name} session...")
    try:
        return_code = start_interactive_session(
            interactive_commands[command_name], cmd_parts[1:]
        )
        output = f"Interactive {command_name} session ended with return code {return_code}"
    except Exception as e:
        output = f"Error starting interactive session {command_name}: {e}"
    return state, output

def handle_cd_command(cmd_parts: List[str], state: ShellState) -> Tuple[ShellState, str]:
    original_path = os.getcwd()
    target_path = cmd_parts[1] if len(cmd_parts) > 1 else os.path.expanduser("~")
    try:
        os.chdir(target_path)
        state.current_path = os.getcwd()
        output = f"Changed directory to {state.current_path}"
    except FileNotFoundError:
        output = colored(f"cd: no such file or directory: {target_path}", "red")
    except Exception as e:
        output = colored(f"cd: error changing directory: {e}", "red")
        os.chdir(original_path) # Revert if error

    return state, output


def handle_bash_command(
    cmd_parts: List[str],
    cmd_str: str,
    stdin_input: Optional[str],
    state: ShellState,
    ) -> Tuple[ShellState, str]:

    command_name = cmd_parts[0]

    if command_name in TERMINAL_EDITORS:
        output = open_terminal_editor(cmd_str)
        return state, output

    try:
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
            err_msg = stderr.strip() if stderr else f"Command '{cmd_str}' failed with return code {process.returncode}."
            # If it failed because command not found, raise specific error for fallback
            if "No such file or directory" in err_msg or "command not found" in err_msg:
                 raise CommandNotFoundError(err_msg)
            # Otherwise, return the error output
            full_output = stdout.strip() + ("\n" + colored(f"stderr: {err_msg}", "red") if err_msg else "")
            return state, full_output.strip()


        output = stdout.strip() if stdout else ""
        if stderr:
             # Log stderr but don't necessarily include in piped output unless requested
             print(colored(f"stderr: {stderr.strip()}", "yellow"), file=sys.stderr)


        if command_name in ["ls", "find", "dir"]:
            output = format_file_listing(output)
        elif not output and process.returncode == 0 and not stderr:
             output = "" # No output is valid, don't print success message if piping

        return state, output

    except FileNotFoundError:
        raise CommandNotFoundError(f"Command not found: {command_name}")
    except PermissionError as e:
         return state, colored(f"Error executing '{cmd_str}': Permission denied. {e}", "red")
    except Exception as e:
        return state, colored(f"Error executing command '{cmd_str}': {e}", "red")


def execute_slash_command(command: str, stdin_input: Optional[str], state: ShellState, stream: bool) -> Tuple[ShellState, Any]:
    """Executes slash commands using the router or checking NPC/Team jinxs."""
    command_parts = command.split()
    command_name = command_parts[0].lstrip('/')
    handler = router.get_route(command_name)
    #print(handler)
    if handler:
        # Prepare kwargs for the handler
        handler_kwargs = {
            'stream': stream,
            'npc': state.npc, 
            'team': state.team,
            'messages': state.messages,
            'model': state.chat_model, 
            'provider': state.chat_provider,
            'api_url': state.api_url,
            'api_key': state.api_key,
        }
        #print(handler_kwargs, command)
        if stdin_input is not None:
            handler_kwargs['stdin_input'] = stdin_input

        try:
            result_dict = handler(command, **handler_kwargs)

            if isinstance(result_dict, dict):
                #some respond with output, some with response, needs to be fixed upstream
                output = result_dict.get("output") or result_dict.get("response")
                state.messages = result_dict.get("messages", state.messages)
                return state, output
            else:
                return state, result_dict

        except Exception as e:
            import traceback
            print(f"Error executing slash command '{command_name}':", file=sys.stderr)
            traceback.print_exc()
            return state, colored(f"Error executing slash command '{command_name}': {e}", "red")

    active_npc = state.npc if isinstance(state.npc, NPC) else None
    jinx_to_execute = None
    executor = None
    if active_npc and command_name in active_npc.jinxs_dict:
        jinx_to_execute = active_npc.jinxs_dict[command_name]
        executor = active_npc
    elif state.team and command_name in state.team.jinxs_dict:
        jinx_to_execute = state.team.jinxs_dict[command_name]
        executor = state.team

    if jinx_to_execute:
        args = command_parts[1:]
        try:
            jinx_output = jinx_to_execute.run(
                *args,
                state=state,
                stdin_input=stdin_input,
                messages=state.messages # Pass messages explicitly if needed
            )
            return state, jinx_output
        except Exception as e:
            import traceback
            print(f"Error executing jinx '{command_name}':", file=sys.stderr)
            traceback.print_exc()
            return state, colored(f"Error executing jinx '{command_name}': {e}", "red")

    if state.team and command_name in state.team.npcs:
        new_npc = state.team.npcs[command_name]
        state.npc = new_npc # Update state directly
        return state, f"Switched to NPC: {new_npc.name}"

    return state, colored(f"Unknown slash command or jinx: {command_name}", "red")


def process_pipeline_command(
    cmd_segment: str,
    stdin_input: Optional[str],
    state: ShellState,
    stream_final: bool
    ) -> Tuple[ShellState, Any]:

    if not cmd_segment:
        return state, stdin_input

    available_models_all = get_locally_available_models(state.current_path)
    available_models_all_list = [item for key, item in available_models_all.items()]
    model_override, provider_override, cmd_cleaned = get_model_and_provider(
        cmd_segment, available_models_all_list
    )
    cmd_to_process = cmd_cleaned.strip()
    if not cmd_to_process:
         return state, stdin_input

    exec_model = model_override or state.chat_model    
    exec_provider = provider_override or state.chat_provider 

    if cmd_to_process.startswith("/"):
        #print(cmd_to_process)
        return execute_slash_command(cmd_to_process, stdin_input, state, stream_final)
    else:
        try:
            cmd_parts = parse_command_safely(cmd_to_process)
            if not cmd_parts:
                 return state, stdin_input

            command_name = cmd_parts[0]

            if command_name in interactive_commands:
                return handle_interactive_command(cmd_parts, state)
            elif command_name == "cd":
                return handle_cd_command(cmd_parts, state)
            else:
                try:
                    bash_state, bash_output = handle_bash_command(cmd_parts, cmd_to_process, stdin_input, state)
                    return bash_state, bash_output
                except CommandNotFoundError:
                    full_llm_cmd = f"{cmd_to_process} {stdin_input}" if stdin_input else cmd_to_process

                    llm_result = check_llm_command(
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
                        context=None 
                       
                    )
                    if isinstance(llm_result, dict):
                        state.messages = llm_result.get("messages", state.messages)
                        output = llm_result.get("output")
                        return state, output
                    else:
                        return state, llm_result

                except Exception as bash_err:
                     return state, colored(f"Bash execution failed: {bash_err}", "red")

        except Exception as e:
            import traceback
            traceback.print_exc()
            return state, colored(f"Error processing command '{cmd_segment[:50]}...': {e}", "red")
def check_mode_switch(command:str , state: ShellState):
    if command in ['/cmd', '/agent', '/chat', '/ride']:
        state.current_mode = command[1:]
        return True, state     

    return False, state
def execute_command(
    command: str,
    state: ShellState,
    ) -> Tuple[ShellState, Any]:

    if not command.strip():
        return state, ""
    mode_change, state = check_mode_switch(command, state)
    if mode_change:
        return state, 'Mode changed.'

    original_command_for_embedding = command
    commands = split_by_pipes(command)
    stdin_for_next = None
    final_output = None
    current_state = state 
    if state.current_mode == 'agent':
        for i, cmd_segment in enumerate(commands):
            is_last_command = (i == len(commands) - 1)
            stream_this_segment = is_last_command and state.stream_output # Use state's stream setting

            try:
                current_state, output = process_pipeline_command(
                    cmd_segment.strip(),
                    stdin_for_next,
                    current_state, 
                    stream_final=stream_this_segment
                )

                if is_last_command:
                    final_output = output # Capture the output of the last command

                if isinstance(output, str):
                    stdin_for_next = output
                elif isgenerator(output):
                    if not stream_this_segment: # If intermediate output is a stream, consume for piping
                        full_stream_output = "".join(map(str, output))
                        stdin_for_next = full_stream_output
                        if is_last_command: final_output = full_stream_output
                    else: # Final output is a stream, don't consume, can't pipe
                        stdin_for_next = None
                        final_output = output
                elif output is not None: # Try converting other types to string
                    try: stdin_for_next = str(output)
                    except Exception:
                        print(f"Warning: Cannot convert output to string for piping: {type(output)}", file=sys.stderr)
                        stdin_for_next = None
                else: # Output was None
                    stdin_for_next = None


            except Exception as pipeline_error:
                import traceback
                traceback.print_exc()
                error_msg = colored(f"Error in pipeline stage {i+1} ('{cmd_segment[:50]}...'): {pipeline_error}", "red")
                # Return the state as it was when the error occurred, and the error message
                return current_state, error_msg

        # Store embeddings using the final state
        if final_output is not None and not (isgenerator(final_output) and current_state.stream_output):
            store_command_embeddings(original_command_for_embedding, final_output, current_state)

        # Return the final state and the final output
        return current_state, final_output
    elif state.current_mode == 'chat':



        response = get_llm_response(
            command, 
            model = state.chat_model, 
            provider = state.chat_provider, 
            npc= state.npc ,
            stream = state.stream_output,
            messages = state.messages
        )
        
        state.messages = response['messages']     
        
        return state, response['response']
    elif state.current_mode == 'cmd':

        response = execute_llm_command(command, 
                                                 model = state.chat_model, 
                                                 provider = state.chat_provider, 
                                                 npc = state.npc, 
                                                 stream = state.stream_output, 
                                                 messages = state.messages) 
        state.messages = response['messages']     
        return state, response['response']

    elif state.current_mode == 'ride':     
        print('To be implemented soon')   
        return state, final_output
    


# --- Main Application Logic ---

def check_deprecation_warnings():
    if os.getenv("NPCSH_MODEL"):
        cprint(
            "Deprecation Warning: NPCSH_MODEL/PROVIDER deprecated. Use NPCSH_CHAT_MODEL/PROVIDER.",
            "yellow",
        )

def print_welcome_message():
    print(
            """
Welcome to \033[1;94mnpc\033[0m\033[1;38;5;202msh\033[0m!
\033[1;94m                    \033[0m\033[1;38;5;202m               \\\\
\033[1;94m _ __   _ __    ___ \033[0m\033[1;38;5;202m ___  | |___    \\\\
\033[1;94m| '_ \ | '_ \  / __|\033[0m\033[1;38;5;202m/ __/ | |_ _|    \\\\
\033[1;94m| | | || |_) |( |__ \033[0m\033[1;38;5;202m\_  \ | | | |    //
\033[1;94m|_| |_|| .__/  \___|\033[0m\033[1;38;5;202m|___/ |_| |_|   //
       \033[1;94m| |          \033[0m\033[1;38;5;202m              //
       \033[1;94m| |
       \033[1;94m|_|

Begin by asking a question, issuing a bash command, or typing '/help' for more information.

            """
        )


def setup_shell() -> Tuple[CommandHistory, Team, Optional[NPC]]:
    check_deprecation_warnings()
    setup_npcsh_config()

    db_path = os.getenv("NPCSH_DB_PATH", HISTORY_DB_DEFAULT_PATH)
    db_path = os.path.expanduser(db_path)
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    command_history = CommandHistory(db_path)

    try:
        readline.set_completer(complete)
   
        history_file = setup_readline()
        atexit.register(save_readline_history)
        atexit.register(command_history.close)        
    except:
        pass

    npc_directory = PROJECT_NPC_TEAM_PATH if os.path.exists(PROJECT_NPC_TEAM_PATH) else DEFAULT_NPC_TEAM_PATH
    os.makedirs(npc_directory, exist_ok=True)

    if not is_npcsh_initialized():
        print("Initializing NPCSH...")
        initialize_base_npcs_if_needed(db_path)
        print("NPCSH initialization complete. Restart or source ~/.npcshrc.")

    team = Team(team_path=npc_directory)
    sibiji_path = os.path.join(DEFAULT_NPC_TEAM_PATH, "sibiji.npc")
    default_npc = None
    if os.path.exists(sibiji_path):
         try:
             default_npc = NPC(file=sibiji_path)
         except Exception as e:
              print(f"Warning: Could not load default NPC 'sibiji': {e}", file=sys.stderr)
    else:
         print(f"Warning: Default NPC file not found: {sibiji_path}", file=sys.stderr)
         if team.npcs:
              default_npc = next(iter(team.npcs.values()))
              print(f"Using '{default_npc.name}' as default NPC.")

    if sys.stdin.isatty():
        print_welcome_message()

    return command_history, team, default_npc


def process_result(
    user_input: str,
    result_state: ShellState,
    output: Any,
    command_history: CommandHistory):

    npc_name = result_state.npc.name if isinstance(result_state.npc, NPC) else result_state.npc
    team_name = result_state.team.name if isinstance(result_state.team, Team) else result_state.team
    save_conversation_message(
        command_history,
        result_state.conversation_id,
        "user",
        user_input,
        wd=result_state.current_path,
        model=result_state.chat_model, # Log primary chat model? Or specific used one?
        provider=result_state.chat_provider,
        npc=npc_name,
        team=team_name,
        attachments=result_state.attachments,
    )
    
    result_state.attachments = None # Clear attachments after logging user message

    final_output_str = None
    if result_state.stream_output:
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
    if final_output_str and result_state.messages and result_state.messages[-1].get("role") != "assistant":
        result_state.messages.append({"role": "assistant", "content": final_output_str})

    #print(result_state.messages)

    print() # Add spacing after output

    if final_output_str:
        save_conversation_message(
            command_history,
            result_state.conversation_id,
            "assistant",
            final_output_str,
            wd=result_state.current_path,
            model=result_state.chat_model,
            provider=result_state.chat_provider,
            npc=npc_name,
            team=team_name,
        )

def run_repl(command_history: CommandHistory, initial_state: ShellState):
    state = initial_state
    print(f'Using {state.current_mode} mode. Use /agent, /cmd, or /chat to switch to other modes')
    print(f'To switch to a different NPC, type /<npc_name>')
    is_windows = platform.system().lower().startswith("win")

    def exit_shell(state):
        print("\nGoodbye!")
        print('beginning knowledge consolidation')
        try:
            breathe_result = breathe(state.messages, state.chat_model, state.chat_provider, state.npc)
            print(breathe_result)
        except KeyboardInterrupt:
            print("Knowledge consolidation interrupted. Exiting immediately.")
        sys.exit(0)

    while True:
        try:
            if is_windows:
                cwd_part = os.path.basename(state.current_path)
                if isinstance(state.npc, NPC):
                    prompt_end = f":{state.npc.name}> "
                else:
                    prompt_end = ":npcsh> "
                prompt = f"{cwd_part}{prompt_end}"
            else:
                cwd_colored = colored(os.path.basename(state.current_path), "blue")
                if isinstance(state.npc, NPC):
                    prompt_end = f":🤖{orange(state.npc.name)}> "
                else:
                    prompt_end = f":🤖{colored('npc', 'blue', attrs=['bold'])}{colored('sh', 'yellow')}> "
                prompt = readline_safe_prompt(f"{cwd_colored}{prompt_end}")
            cwd_colored = colored(os.path.basename(state.current_path), "blue")
            if isinstance(state.npc, NPC):
                prompt_end = f":🤖{orange(state.npc.name)}> "
            else:
                prompt_end = f":🤖{colored('npc', 'blue', attrs=['bold'])}{colored('sh', 'yellow')}> "
            prompt = readline_safe_prompt(f"{cwd_colored}{prompt_end}")

            user_input = get_multiline_input(prompt).strip()
            # Handle Ctrl+Z (ASCII SUB, '\x1a') as exit (Windows and Unix)
            if user_input == "\x1a":
                exit_shell(state)

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                if isinstance(state.npc, NPC):
                    print(f"Exiting {state.npc.name} mode.")
                    state.npc = None
                    continue
                else:
                    exit_shell(state)

            state.current_path = os.getcwd()
            state, output = execute_command(user_input, state)
            process_result(user_input, state, output, command_history)

        except KeyboardInterrupt:
            if is_windows:
                # On Windows, Ctrl+C cancels the current input line, show prompt again
                print("^C")
                continue
            else:
                # On Unix, Ctrl+C exits the shell as before
                exit_shell(state)
        except EOFError:
            # Ctrl+D: exit shell cleanly
            exit_shell(state)

def run_non_interactive(command_history: CommandHistory, initial_state: ShellState):
    state = initial_state
    # print("Running in non-interactive mode...", file=sys.stderr) # Optional debug

    for line in sys.stdin:
        user_input = line.strip()
        if not user_input:
            continue
        if user_input.lower() in ["exit", "quit"]:
             break

        state.current_path = os.getcwd()
        state, output = execute_command(user_input, state)
        # Non-interactive: just print raw output, don't process results complexly
        if state.stream_output and isgenerator(output):
             for chunk in output: print(str(chunk), end='')
             print()
        elif output is not None:
             print(output)
        # Maybe still log history?
        # process_result(user_input, state, output, command_history)

def main() -> None:
    parser = argparse.ArgumentParser(description="npcsh - An NPC-powered shell.")
    parser.add_argument(
        "-v", "--version", action="version", version=f"npcsh version {VERSION}"
    )
    parser.add_argument(
         "-c", "--command", type=str, help="Execute a single command and exit."
    )
    args = parser.parse_args()

    command_history, team, default_npc = setup_shell()

    initial_state.npc = default_npc 
    initial_state.team = team
    #import pdb 
    #pdb.set_trace()
    if args.command:
         state = initial_state
         state.current_path = os.getcwd()
         final_state, output = execute_command(args.command, state)
         if final_state.stream_output and isgenerator(output):
              for chunk in output: print(str(chunk), end='')
              print()
         elif output is not None:
              print(output)

    elif not sys.stdin.isatty():
        run_non_interactive(command_history, initial_state)
    else:
        run_repl(command_history, initial_state)

if __name__ == "__main__":
    main()