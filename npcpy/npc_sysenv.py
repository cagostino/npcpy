import re
from datetime import datetime

import os
import sqlite3
from dotenv import load_dotenv
import logging
from typing import List, Any

import subprocess
import platform

try:
    import nltk
except:
    print("Error importing nltk")
import numpy as np

import filecmp

import shutil
import tempfile
import pandas as pd

try:
    from sentence_transformers import util
except Exception as e:
    print(f"Error importing sentence_transformers: {e}")




from typing import Dict, Any, List, Optional, Union
import numpy as np
from colorama import Fore, Back, Style
import re
import tempfile
import sqlite3
from datetime import datetime
import logging
import textwrap
import subprocess
from termcolor import colored
import sys
# --- Platform-specific imports and guards ---
import sys
import platform

ON_WINDOWS = platform.system() == "Windows"

# Try/except for termios, tty, pty, select, signal
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

# Try/except for readline
try:
    import readline
except ImportError:
    readline = None
    print('no readline support, some features may not work as desired.')

# Try/except for rich imports
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

# Global variables
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



def get_locally_available_models(project_directory):
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

    if "ANTHROPIC_API_KEY" in env_vars or os.environ.get("ANTHROPIC_API_KEY"):
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=env_vars.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY"))
            models = client.models.list()
            for model in models.data:
                available_models[model.id] = 'anthropic'
                    
        except:
            print("anthropic models not indexed")
    if "OPENAI_API_KEY" in env_vars or os.environ.get("OPENAI_API_KEY"):
        try:
            import openai

            openai.api_key = env_vars.get("OPENAI_API_KEY", None) or os.environ.get("OPENAI_API_KEY", None)
            models = openai.models.list()

            for model in models.data:
                if (
                    (
                        "gpt" in model.id
                        or "o1" in model.id
                        or "o3" in model.id
                        or "chat" in model.id
                    )
                    and "audio" not in model.id
                    and "realtime" not in model.id
                ):
                    available_models[model.id] = "openai"
        except:
            print("openai models not indexed")

    if "GEMINI_API_KEY" in env_vars or os.environ.get("GEMINI_API_KEY"):
        try:

            from google import genai
        
            #print('GEMINI_API_KEY', genai)

            client = genai.Client(api_key = env_vars.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY"))
            print(client)
            models= []
            #import pdb 
            #pdb.set_trace()
            
            for m in client.models.list():
                print('sd',m)
                for action in m.supported_actions:
                    if action == "generateContent":
                        if 'models/' in m.name:
                            
                            models.append(m.name.split('/')[1])
            for model in set(models):
                if "gemini" in model:
                    available_models[model] = "gemini"
        except Exception as e:
            print(f"gemini models not indexed: {e}")
    if "DEEPSEEK_API_KEY" in env_vars or os.environ.get("DEEPSEEK_API_KEY"):
        available_models['deepseek-chat'] = 'deepseek'
        available_models['deepseek-reasoner'] = 'deepseek'        
    try:
        import ollama
        models = ollama.list()
        for model in models.models:

            if "embed" not in model.model:
                mod = model.model
                available_models[mod] = "ollama"
    except Exception as e:
        print(f"Error loading ollama models: {e}")
    #print("locally available models", available_models)
        
    return available_models


def validate_bash_command(command_parts: list) -> bool:
    """
    Function Description:
        Validate if the command sequence is a valid bash command with proper arguments/flags.
    Args:
        command_parts : list : Command parts
    Keyword Args:
        None
    Returns:
        bool : bool : Boolean
    """
    if not command_parts:
        return False

    COMMAND_PATTERNS = {
        "cat": {
            "flags": ["-n", "-b", "-E", "-T", "-s", "--number", "-A", "--show-all"],
            "requires_arg": True,
        },
        "find": {
            "flags": [
                "-name",
                "-type",
                "-size",
                "-mtime",
                "-exec",
                "-print",
                "-delete",
                "-maxdepth",
                "-mindepth",
                "-perm",
                "-user",
                "-group",
            ],
            "requires_arg": True,
        },
        "who": {
            "flags": [
                "-a",
                "-b",
                "-d",
                "-H",
                "-l",
                "-p",
                "-q",
                "-r",
                "-s",
                "-t",
                "-u",
                "--all",
                "--count",
                "--heading",
            ],
            "requires_arg": True,
        },
        "open": {
            "flags": ["-a", "-e", "-t", "-f", "-F", "-W", "-n", "-g", "-h"],
            "requires_arg": True,
        },
        "which": {"flags": ["-a", "-s", "-v"], "requires_arg": True},
    }

    base_command = command_parts[0]

    if base_command not in COMMAND_PATTERNS:
        return True  # Allow other commands to pass through

    pattern = COMMAND_PATTERNS[base_command]
    args = []
    flags = []

    for i in range(1, len(command_parts)):
        part = command_parts[i]
        if part.startswith("-"):
            flags.append(part)
            if part not in pattern["flags"]:
                return False  # Invalid flag
        else:
            args.append(part)

    # Check if 'who' has any arguments (it shouldn't)
    if base_command == "who" and args:
        return False

    # Handle 'which' with '-a' flag
    if base_command == "which" and "-a" in flags:
        return True  # Allow 'which -a' with or without arguments.

    # Check if any required arguments are missing
    if pattern.get("requires_arg", False) and not args:
        return False

    return True




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



def start_interactive_session(command: list) -> int:
    """
    Starts an interactive session. Only works on Unix. On Windows, print a message and return 1.
    """
    if ON_WINDOWS or termios is None or tty is None or pty is None or select is None or signal is None:
        print("Interactive terminal sessions are not supported on Windows.")
        return 1
    # Save the current terminal settings
    old_tty = termios.tcgetattr(sys.stdin)
    try:
        # Create a pseudo-terminal
        master_fd, slave_fd = pty.openpty()

        # Start the process
        p = subprocess.Popen(
            command,
            stdin=slave_fd,
            stdout=slave_fd,
            stderr=slave_fd,
            shell=True,
            preexec_fn=os.setsid,  # Create a new process group
        )

        # Set the terminal to raw mode
        tty.setraw(sys.stdin.fileno())

        def handle_timeout(signum, frame):
            raise TimeoutError("Process did not terminate in time")

        while p.poll() is None:
            r, w, e = select.select([sys.stdin, master_fd], [], [], 0.1)
            if sys.stdin in r:
                d = os.read(sys.stdin.fileno(), 10240)
                os.write(master_fd, d)
            elif master_fd in r:
                o = os.read(master_fd, 10240)
                if o:
                    os.write(sys.stdout.fileno(), o)
                else:
                    break

        # Wait for the process to terminate with a timeout
        signal.signal(signal.SIGALRM, handle_timeout)
        signal.alarm(5)  # 5 second timeout
        try:
            p.wait()
        except TimeoutError:
            print("\nProcess did not terminate. Force killing...")
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            time.sleep(1)
            if p.poll() is None:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        finally:
            signal.alarm(0)

    finally:
        # Restore the terminal settings
        termios.tcsetattr(sys.stdin, termios.TCSAFLUSH, old_tty)

    return p.returncode

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
        if line.startswith("```"):  # Toggle code block
            if inside_code_block:
                # Close code block, unindent, and append
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


BASH_COMMANDS = [
    "npc",
    "npm",
    "npx",
    "open",
    "alias",
    "bg",
    "bind",
    "break",
    "builtin",
    "case",
    "command",
    "compgen",
    "complete",
    "continue",
    "declare",
    "dirs",
    "disown",
    "echo",
    "enable",
    "eval",
    "exec",
    "exit",
    "export",
    "fc",
    "fg",
    "getopts",
    "hash",
    "help",
    "history",
    "if",
    "jobs",
    "kill",
    "let",
    "local",
    "logout",
    "ollama",
    "popd",
    "printf",
    "pushd",
    "pwd",
    "read",
    "readonly",
    "return",
    "set",
    "shift",
    "shopt",
    "source",
    "suspend",
    "test",
    "times",
    "trap",
    "type",
    "typeset",
    "ulimit",
    "umask",
    "unalias",
    "unset",
    "until",
    "wait",
    "while",
    # Common Unix commands
    "ls",
    "cp",
    "mv",
    "rm",
    "mkdir",
    "rmdir",
    "touch",
    "cat",
    "less",
    "more",
    "head",
    "tail",
    "grep",
    "find",
    "sed",
    "awk",
    "sort",
    "uniq",
    "wc",
    "diff",
    "chmod",
    "chown",
    "chgrp",
    "ln",
    "tar",
    "gzip",
    "gunzip",
    "zip",
    "unzip",
    "ssh",
    "scp",
    "rsync",
    "wget",
    "curl",
    "ping",
    "netstat",
    "ifconfig",
    "route",
    "traceroute",
    "ps",
    "top",
    "htop",
    "kill",
    "killall",
    "su",
    "sudo",
    "whoami",
    "who",
    "w",
    "last",
    "finger",
    "uptime",
    "free",
    "df",
    "du",
    "mount",
    "umount",
    "fdisk",
    "mkfs",
    "fsck",
    "dd",
    "cron",
    "at",
    "systemctl",
    "service",
    "journalctl",
    "man",
    "info",
    "whatis",
    "whereis",
    "which",
    "date",
    "cal",
    "bc",
    "expr",
    "screen",
    "tmux",
    "git",
    "vim",
    "emacs",
    "nano",
    "pip",
]


interactive_commands = {
    "ipython": ["ipython"],
    "python": ["python", "-i"],
    "sqlite3": ["sqlite3"],
    "r": ["R", "--interactive"],
}

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
    lang = None

    for line in lines:
        if line.startswith("```"):
            if inside_code_block:
                # End of code block - render the collected code
                code = "\n".join(code_lines)
                if code.strip():
                    syntax = Syntax(
                        code, lang or "python", theme="monokai", line_numbers=False
                    )
                    console.print(syntax)
                code_lines = []
            else:
                # Start of code block - get language if specified
                lang = line[3:].strip() or None
            inside_code_block = not inside_code_block
        elif inside_code_block:
            code_lines.append(line)
        else:
            # Regular markdown
            console.print(Markdown(line))






def execute_set_command(command: str, value: str) -> str:
    """
    Function Description:
        This function sets a configuration value in the .npcshrc file.
    Args:
        command: The command to execute.
        value: The value to set.
    Keyword Args:
        None
    Returns:
        A message indicating the success or failure of the operation.
    """

    config_path = os.path.expanduser("~/.npcshrc")

    # Map command to environment variable name
    var_map = {
        "model": "NPCSH_CHAT_MODEL",
        "provider": "NPCSH_CHAT_PROVIDER",
        "db_path": "NPCSH_DB_PATH",
    }

    if command not in var_map:
        return f"Unknown setting: {command}"

    env_var = var_map[command]

    # Read the current configuration
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            lines = f.readlines()
    else:
        lines = []

    # Check if the property exists and update it, or add it if it doesn't exist
    property_exists = False
    for i, line in enumerate(lines):
        if line.startswith(f"export {env_var}="):
            lines[i] = f"export {env_var}='{value}'\n"
            property_exists = True
            break

    if not property_exists:
        lines.append(f"export {env_var}='{value}'\n")

    # Save the updated configuration
    with open(config_path, "w") as f:
        f.writelines(lines)

    return f"{command.capitalize()} has been set to: {value}"

# Function to check and download NLTK data if necessary
def ensure_nltk_punkt() -> None:
    """
    Function Description:
        This function ensures that the NLTK 'punkt' tokenizer is downloaded.
    Args:
        None
    Keyword Args:
        None
    Returns:
        None
    """

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        print("Downloading NLTK 'punkt' tokenizer...")
        nltk.download("punkt")

def get_shell_config_file() -> str:
    """

    Function Description:
        This function returns the path to the shell configuration file.
    Args:
        None
    Keyword Args:
        None
    Returns:
        The path to the shell configuration file.
    """
    # Check the current shell
    shell = os.environ.get("SHELL", "")

    if "zsh" in shell:
        return os.path.expanduser("~/.zshrc")
    elif "bash" in shell:
        # On macOS, use .bash_profile for login shells
        if platform.system() == "Darwin":
            return os.path.expanduser("~/.bash_profile")
        else:
            return os.path.expanduser("~/.bashrc")
    else:
        # Default to .bashrc if we can't determine the shell
        return os.path.expanduser("~/.bashrc")



def add_npcshrc_to_shell_config() -> None:
    """
    Function Description:
        This function adds the sourcing of the .npcshrc file to the user's shell configuration file.
    Args:
        None
    Keyword Args:
        None
    Returns:
        None
    """

    if os.getenv("NPCSH_INITIALIZED") is not None:
        return
    config_file = get_shell_config_file()
    npcshrc_line = "\n# Source NPCSH configuration\nif [ -f ~/.npcshrc ]; then\n    . ~/.npcshrc\nfi\n"

    with open(config_file, "a+") as shell_config:
        shell_config.seek(0)
        content = shell_config.read()
        if "source ~/.npcshrc" not in content and ". ~/.npcshrc" not in content:
            shell_config.write(npcshrc_line)
            print(f"Added .npcshrc sourcing to {config_file}")
        else:
            print(f".npcshrc already sourced in {config_file}")

def ensure_npcshrc_exists() -> str:
    """
    Function Description:
        This function ensures that the .npcshrc file exists in the user's home directory.
    Args:
        None
    Keyword Args:
        None
    Returns:
        The path to the .npcshrc file.
    """

    npcshrc_path = os.path.expanduser("~/.npcshrc")
    if not os.path.exists(npcshrc_path):
        with open(npcshrc_path, "w") as npcshrc:
            npcshrc.write("# NPCSH Configuration File\n")
            npcshrc.write("export NPCSH_INITIALIZED=0\n")
            npcshrc.write("export NPCSH_DEFAULT_MODE='agent'\n")
            npcshrc.write("export NPCSH_CHAT_PROVIDER='ollama'\n")
            npcshrc.write("export NPCSH_CHAT_MODEL='llama3.2'\n")
            npcshrc.write("export NPCSH_REASONING_PROVIDER='ollama'\n")
            npcshrc.write("export NPCSH_REASONING_MODEL='deepseek-r1'\n")

            npcshrc.write("export NPCSH_EMBEDDING_PROVIDER='ollama'\n")
            npcshrc.write("export NPCSH_EMBEDDING_MODEL='nomic-embed-text'\n")
            npcshrc.write("export NPCSH_VISION_PROVIDER='ollama'\n")
            npcshrc.write("export NPCSH_VISION_MODEL='llava7b'\n")
            npcshrc.write(
                "export NPCSH_IMAGE_GEN_MODEL='runwayml/stable-diffusion-v1-5'\n"
            )

            npcshrc.write("export NPCSH_IMAGE_GEN_PROVIDER='diffusers'\n")
            npcshrc.write(
                "export NPCSH_VIDEO_GEN_MODEL='runwayml/stable-diffusion-v1-5'\n"
            )

            npcshrc.write("export NPCSH_VIDEO_GEN_PROVIDER='diffusers'\n")

            npcshrc.write("export NPCSH_API_URL=''\n")
            npcshrc.write("export NPCSH_DB_PATH='~/npcsh_history.db'\n")
            npcshrc.write("export NPCSH_VECTOR_DB_PATH='~/npcsh_chroma.db'\n")
            npcshrc.write("export NPCSH_STREAM_OUTPUT=0")
    return npcshrc_path



def setup_npcsh_config() -> None:
    """
    Function Description:
        This function initializes the NPCSH configuration.
    Args:
        None
    Keyword Args:
        None
    Returns:
        None
    """

    ensure_npcshrc_exists()
    add_npcshrc_to_shell_config()


def is_npcsh_initialized() -> bool:
    """
    Function Description:
        This function checks if the NPCSH initialization flag is set.
    Args:
        None
    Keyword Args:
        None
    Returns:
        A boolean indicating whether NPCSH is initialized.
    """

    return os.environ.get("NPCSH_INITIALIZED", None) == "1"


def set_npcsh_initialized() -> None:
    """
    Function Description:
        This function sets the NPCSH initialization flag in the .npcshrc file.
    Args:
        None
    Keyword Args:
        None
    Returns:

        None
    """

    npcshrc_path = ensure_npcshrc_exists()

    with open(npcshrc_path, "r+") as npcshrc:
        content = npcshrc.read()
        if "export NPCSH_INITIALIZED=0" in content:
            content = content.replace(
                "export NPCSH_INITIALIZED=0", "export NPCSH_INITIALIZED=1"
            )
            npcshrc.seek(0)
            npcshrc.write(content)
            npcshrc.truncate()

    # Also set it for the current session
    os.environ["NPCSH_INITIALIZED"] = "1"
    print("NPCSH initialization flag set in .npcshrc")


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


def get_npc_path(npc_name: str, db_path: str) -> str:
    # First, check in project npc_team directory
    project_npc_team_dir = os.path.abspath("./npc_team")
    project_npc_path = os.path.join(project_npc_team_dir, f"{npc_name}.npc")

    # Then, check in global npc_team directory
    user_npc_team_dir = os.path.expanduser("~/.npcsh/npc_team")
    global_npc_path = os.path.join(user_npc_team_dir, f"{npc_name}.npc")

    # Check database for compiled NPCs
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            query = f"SELECT source_path FROM compiled_npcs WHERE name = '{npc_name}'"
            cursor.execute(query)
            result = cursor.fetchone()
            if result:
                return result[0]

    except Exception as e:
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                query = f"SELECT source_path FROM compiled_npcs WHERE name = {npc_name}"
                cursor.execute(query)
                result = cursor.fetchone()
                if result:
                    return result[0]
        except Exception as e:
            print(f"Database query error: {e}")

    # Fallback to file paths
    if os.path.exists(project_npc_path):
        return project_npc_path

    if os.path.exists(global_npc_path):
        return global_npc_path

    raise ValueError(f"NPC file not found: {npc_name}")


def initialize_base_npcs_if_needed(db_path: str) -> None:
    """
    Function Description:
        This function initializes the base NPCs if they are not already in the database.
    Args:
        db_path: The path to the database file.
    Keyword Args:

        None
    Returns:
        None
    """

    if is_npcsh_initialized():
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create the compiled_npcs table if it doesn't exist
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS compiled_npcs (
            name TEXT PRIMARY KEY,
            source_path TEXT NOT NULL,
            compiled_content TEXT
        )
        """
    )

    # Get the path to the npc_team directory in the package
    package_dir = os.path.dirname(__file__)
    package_npc_team_dir = os.path.join(package_dir, "npc_team")

    

    # User's global npc_team directory
    user_npc_team_dir = os.path.expanduser("~/.npcsh/npc_team")

    user_jinxs_dir = os.path.join(user_npc_team_dir, "jinxs")
    user_templates_dir = os.path.join(user_npc_team_dir, "templates")
    os.makedirs(user_npc_team_dir, exist_ok=True)
    os.makedirs(user_jinxs_dir, exist_ok=True)
    os.makedirs(user_templates_dir, exist_ok=True)
    # Copy NPCs from package to user directory
    for filename in os.listdir(package_npc_team_dir):
        if filename.endswith(".npc"):
            source_path = os.path.join(package_npc_team_dir, filename)
            destination_path = os.path.join(user_npc_team_dir, filename)
            if not os.path.exists(destination_path) or file_has_changed(
                source_path, destination_path
            ):
                shutil.copy2(source_path, destination_path)
                print(f"Copied NPC {filename} to {destination_path}")
        if filename.endswith(".ctx"):
            source_path = os.path.join(package_npc_team_dir, filename)
            destination_path = os.path.join(user_npc_team_dir, filename)
            if not os.path.exists(destination_path) or file_has_changed(
                source_path, destination_path
            ):
                shutil.copy2(source_path, destination_path)
                print(f"Copied ctx {filename} to {destination_path}")

    # Copy jinxs from package to user directory
    package_jinxs_dir = os.path.join(package_npc_team_dir, "jinxs")
    if os.path.exists(package_jinxs_dir):
        for filename in os.listdir(package_jinxs_dir):
            if filename.endswith(".jinx"):
                source_jinx_path = os.path.join(package_jinxs_dir, filename)
                destination_jinx_path = os.path.join(user_jinxs_dir, filename)
                if (not os.path.exists(destination_jinx_path)) or file_has_changed(
                    source_jinx_path, destination_jinx_path
                ):
                    shutil.copy2(source_jinx_path, destination_jinx_path)
                    print(f"Copied jinx {filename} to {destination_jinx_path}")

    templates = os.path.join(package_npc_team_dir, "templates")
    if os.path.exists(templates):
        for folder in os.listdir(templates):
            os.makedirs(os.path.join(user_templates_dir, folder), exist_ok=True)
            for file in os.listdir(os.path.join(templates, folder)):
                if file.endswith(".npc"):
                    source_template_path = os.path.join(templates, folder, file)

                    destination_template_path = os.path.join(
                        user_templates_dir, folder, file
                    )
                    if not os.path.exists(
                        destination_template_path
                    ) or file_has_changed(
                        source_template_path, destination_template_path
                    ):
                        shutil.copy2(source_template_path, destination_template_path)
                        print(f"Copied template {file} to {destination_template_path}")
    conn.commit()
    conn.close()
    set_npcsh_initialized()
    add_npcshrc_to_shell_config()


def file_has_changed(source_path: str, destination_path: str) -> bool:
    """
    Function Description:
        This function compares two files to determine if they are different.
    Args:
        source_path: The path to the source file.
        destination_path: The path to the destination file.
    Keyword Args:
        None
    Returns:
        A boolean indicating whether the files are different
    """

    # Compare file modification times or contents to decide whether to update the file
    return not filecmp.cmp(source_path, destination_path, shallow=False)


def is_valid_npc(npc: str, db_path: str) -> bool:
    """
    Function Description:
        This function checks if an NPC is valid based on the database.
    Args:
        npc: The name of the NPC.
        db_path: The path to the database file.
    Keyword Args:
        None
    Returns:
        A boolean indicating whether the NPC is valid.
    """

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM compiled_npcs WHERE name = ?", (npc,))
    result = cursor.fetchone()
    conn.close()
    return result is not None


def execute_python(code: str) -> str:
    """
    Function Description:
        This function executes Python code and returns the output.
    Args:
        code: The Python code to execute.
    Keyword Args:
        None
    Returns:
        The output of the code execution.
    """

    try:
        result = subprocess.run(
            ["python", "-c", code], capture_output=True, text=True, timeout=30
        )
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "Error: Execution timed out"


def execute_r(code: str) -> str:
    """
    Function Description:
        This function executes R code and returns the output.
    Args:
        code: The R code to execute.
    Keyword Args:
        None
    Returns:
        The output of the code execution.
    """

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".R", delete=False
        ) as temp_file:
            temp_file.write(code)
            temp_file_path = temp_file.name

        result = subprocess.run(
            ["Rscript", temp_file_path], capture_output=True, text=True, timeout=30
        )
        os.unlink(temp_file_path)
        return result.stdout if result.returncode == 0 else f"Error: {result.stderr}"
    except subprocess.TimeoutExpired:
        os.unlink(temp_file_path)
        return "Error: Execution timed out"


def execute_sql(code: str) -> str:
    """
    Function Description:
        This function executes SQL code and returns the output.
    Args:
        code: The SQL code to execute.
    Keyword Args:
        None
    Returns:
        result: The output of the code execution.
    """
    # use pandas to run the sql
    try:
        result = pd.read_sql_query(code, con=sqlite3.connect("npcsh_history.db"))
        return result
    except Exception as e:
        return f"Error: {e}"


def list_directory(args: List[str]) -> None:
    """
    Function Description:
        This function lists the contents of a directory.
    Args:
        args: The command arguments.
    Keyword Args:
        None
    Returns:
        None
    """
    directory = args[0] if args else "."
    try:
        files = os.listdir(directory)
        for f in files:
            print(f)
    except Exception as e:
        print(f"Error listing directory: {e}")


def read_file(args: List[str]) -> None:
    """
    Function Description:
        This function reads the contents of a file.
    Args:
        args: The command arguments.
    Keyword Args:
        None
    Returns:
        None
    """

    if not args:
        print("Usage: /read <filename>")
        return
    filename = args[0]
    try:
        with open(filename, "r") as file:
            content = file.read()
            print(content)
    except Exception as e:
        print(f"Error reading file: {e}")




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

import os
import json
from pathlib import Path


def change_directory(command_parts: list, messages: list) -> dict:
    """
    Function Description:
        Changes the current directory.
    Args:
        command_parts : list : Command parts
        messages : list : Messages
    Keyword Args:
        None
    Returns:
        dict : dict : Dictionary

    """

    try:
        if len(command_parts) > 1:
            new_dir = os.path.expanduser(command_parts[1])
        else:
            new_dir = os.path.expanduser("~")
        os.chdir(new_dir)
        return {
            "messages": messages,
            "output": f"Changed directory to {os.getcwd()}",
        }
    except FileNotFoundError:
        return {
            "messages": messages,
            "output": f"Directory not found: {new_dir}",
        }
    except PermissionError:
        return {"messages": messages, "output": f"Permission denied: {new_dir}"}

def ensure_dirs_exist(*dirs):
    """Ensure all specified directories exist"""
    for dir_path in dirs:
        os.makedirs(os.path.expanduser(dir_path), exist_ok=True)

def init_db_tables(db_path="~/npcsh_history.db"):
    """Initialize necessary database tables"""
    db_path = os.path.expanduser(db_path)
    with sqlite3.connect(db_path) as conn:
        # NPC log table for storing all kinds of entries
        conn.execute("""
            CREATE TABLE IF NOT EXISTS npc_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id TEXT,  
                entry_type TEXT,
                content TEXT,
                metadata TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Pipeline runs table for tracking pipeline executions
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pipeline_name TEXT,
                step_name TEXT,
                output TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Compiled NPCs table for storing compiled NPC content
        conn.execute("""
            CREATE TABLE IF NOT EXISTS compiled_npcs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                source_path TEXT,
                compiled_content TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()




def orange(text: str) -> str:
    """
    Function Description:
        Returns orange text.
    Args:
        text : str : Text
    Keyword Args:
        None
    Returns:
        text : str : Text

    """
    return f"\033[38;2;255;165;0m{text}{Style.RESET_ALL}"

def get_npcshrc_path_windows():
    return Path.home() / ".npcshrc"


def read_rc_file_windows(path):
    """Read shell-style rc file"""
    config = {}
    if not path.exists():
        return config

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # Match KEY='value' or KEY="value" format
                match = re.match(r'^([A-Z_]+)\s*=\s*[\'"](.*?)[\'"]$', line)
                if match:
                    key, value = match.groups()
                    config[key] = value
    return config


def get_setting_windows(key, default=None):
    # Try environment variable first
    if env_value := os.getenv(key):
        return env_value

    # Fall back to .npcshrc file
    config = read_rc_file_windows(get_npcshrc_path_windows())
    return config.get(key, default)

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
        # Autocomplete model name
        matches = [m for m in available_models if m.startswith(model_name)]
        if matches:
            if len(matches) == 1:
                model_name = matches[0]  # Complete the name if only one match
            # Find provider for the (potentially autocompleted) model
            provider = lookup_provider(model_name)
            if provider:
                # Remove the model tag from the command
                cleaned_command = command.replace(
                    f"@{model_match.group(1)}", ""
                ).strip()
                # print(cleaned_command, 'cleaned_command')
                return model_name, provider, cleaned_command
            else:
                return None, None, command  # Provider not found
        else:
            return None, None, command  # No matching model
    else:
        return None, None, command  # No model specified

def render_code_block(code: str, language: str = None) -> None:
    """Render a code block with syntax highlighting using rich, left-justified with no line numbers"""
    from rich.syntax import Syntax
    from rich.console import Console

    console = Console(highlight=True)
    code = code.strip()
    # If code starts with a language identifier, remove it
    if code.split("\n", 1)[0].lower() in ["python", "bash", "javascript"]:
        code = code.split("\n", 1)[1]
    syntax = Syntax(
        code, language or "python", theme="monokai", line_numbers=False, padding=0
    )
    console.print(syntax)
def print_and_process_stream_with_markdown(response, model, provider):
    str_output = ""
    dot_count = 0  # Keep track of how many dots we've printed
    tool_call_data = {"id": None, "function_name": None, "arguments": ""}

    for chunk in response:
        # Get chunk content based on provider
        print('.', end="", flush=True)
        dot_count += 1
        
        # Extract tool call info based on provider
        if provider == "ollama":
            # Ollama tool call extraction
            if "message" in chunk and "tool_calls" in chunk["message"]:
                for tool_call in chunk["message"]["tool_calls"]:
                    if "id" in tool_call:
                        tool_call_data["id"] = tool_call["id"]
                    if "function" in tool_call:
                        if "name" in tool_call["function"]:
                            tool_call_data["function_name"] = tool_call["function"]["name"]
                        if "arguments" in tool_call["function"]:
                            tool_call_data["arguments"] += tool_call["function"]["arguments"]
            
            chunk_content = chunk["message"]["content"] if "message" in chunk and "content" in chunk["message"] else ""
        else:
            # LiteLLM tool call extraction
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
            
            if reasoning_content:
                chunk_content = reasoning_content
                    
            chunk_content += "".join(
                c.delta.content for c in chunk.choices if c.delta.content
            )
        
        if not chunk_content:
            continue
        str_output += chunk_content
    
    # Clear the dots by returning to the start of line and printing spaces
    print('\r' + ' ' * dot_count*2 + '\r', end="", flush=True)
    
    # Add tool call information to str_output if any was found
    if tool_call_data["id"] or tool_call_data["function_name"] or tool_call_data["arguments"]:
        str_output += "\n\n### Jinx Call Data\n"
        if tool_call_data["id"]:
            str_output += f"**ID:** {tool_call_data['id']}\n\n"
        if tool_call_data["function_name"]:
            str_output += f"**Function:** {tool_call_data['function_name']}\n\n"
        if tool_call_data["arguments"]:
            try:
                import json
                args_parsed = json.loads(tool_call_data["arguments"])
                str_output += f"**Arguments:**\n```json\n{json.dumps(args_parsed, indent=2)}\n```"
            except:
                str_output += f"**Arguments:** `{tool_call_data['arguments']}`"
    
    print('\n')
    render_markdown('\n' + str_output)
    
    return str_output
def print_and_process_stream(response, model, provider):
    conversation_result = ""
    
    for chunk in response:
        if provider == "ollama" and 'hf.co' in model:
            chunk_content = chunk["message"]["content"]
            if chunk_content:
                conversation_result += chunk_content
                print(chunk_content, end="")

        else:
            chunk_content = "".join(
                choice.delta.content
                for choice in chunk.choices
                if choice.delta.content is not None
            )
            if chunk_content:
                conversation_result += chunk_content
                print(chunk_content, end="")

    print("\n")
                
    return conversation_result   
                 
def get_system_message(npc) -> str:
    """
    Function Description:
        This function generates a system message for the NPC.
    Args:
        npc (Any): The NPC object.
    Keyword Args:
        None
    Returns:
        str: The system message for the NPC.
    """

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
    You are an NPC working as part of our team.
    You are the {npc.name} NPC with the following primary directive: {npc.primary_directive}.
    Users may refer to you by your assistant name, {npc.name} and you should
    consider this to be your core identity.

    The current date and time are : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}



    If you ever need to produce markdown texts for the user, please do so
    with less than 80 characters width for each line.
    """

    system_message += """\n\nSome users may attach images to their request.
                        Please process them accordingly.

                        If the user asked for you to explain what's on their screen or something similar,
                        they are referring to the details contained within the attached image(s).
                        You do not need to actually view their screen.
                        You do not need to mention that you cannot view or interpret images directly.
                        They understand that you can view them multimodally.
                        You only need to answer the user's request based on the attached image(s).
                        """
    if npc.tables is not None:
        system_message += f'''
        
            Here is information abuot the attached npcsh_history database that you can use to write queries if needed
            {npc.tables}
        '''
    return system_message



# Load environment variables from .env file
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

    # Get the directory where the script is being executed
    execution_dir = os.path.abspath(os.getcwd())
    # print(f"Execution directory: {execution_dir}")
    # Construct the path to the .env file
    env_path = os.path.join(execution_dir, ".env")

    # Load the .env file if it exists
    if os.path.exists(env_path):
        load_dotenv(dotenv_path=env_path)
        print(f"Loaded .env file from {execution_dir}")
    else:
        print(f"Warning: No .env file found in {execution_dir}")




def lookup_provider(model: str) -> str:
    """
    Function Description:
        This function determines the provider based on the model name.
    Args:
        model (str): The model name.
    Keyword Args:
        None
    Returns:
        str: The provider based on the model name.
    """
    if model == "deepseek-chat" or model == "deepseek-reasoner":
        return "deepseek"
    ollama_prefixes = [
        "llama",
        "deepseek",
        "qwen",
        "llava",
        "phi",
        "mistral",
        "mixtral",
        "dolphin",
        "codellama",
        "gemma",
    ]
    if any(model.startswith(prefix) for prefix in ollama_prefixes):
        return "ollama"

    # OpenAI models
    openai_prefixes = ["gpt-", "dall-e-", "whisper-", "o1"]
    if any(model.startswith(prefix) for prefix in openai_prefixes):
        return "openai"

    # Anthropic models
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

NPCSH_CHAT_MODEL = os.environ.get("NPCSH_CHAT_MODEL", "llama3.2")
# print("NPCSH_CHAT_MODEL", NPCSH_CHAT_MODEL)
NPCSH_CHAT_PROVIDER = os.environ.get("NPCSH_CHAT_PROVIDER", "ollama")
# print("NPCSH_CHAT_PROVIDER", NPCSH_CHAT_PROVIDER)
NPCSH_DB_PATH = os.path.expanduser(
    os.environ.get("NPCSH_DB_PATH", "~/npcsh_history.db")
)
NPCSH_VECTOR_DB_PATH = os.path.expanduser(
    os.environ.get("NPCSH_VECTOR_DB_PATH", "~/npcsh_chroma.db")
)
#DEFAULT MODES = ['CHAT', 'AGENT', 'CODE', ]

NPCSH_DEFAULT_MODE = os.path.expanduser(os.environ.get("NPCSH_DEFAULT_MODE", "agent"))
NPCSH_VISION_MODEL = os.environ.get("NPCSH_VISION_MODEL", "llava:7b")
NPCSH_VISION_PROVIDER = os.environ.get("NPCSH_VISION_PROVIDER", "ollama")
NPCSH_IMAGE_GEN_MODEL = os.environ.get(
    "NPCSH_IMAGE_GEN_MODEL", "runwayml/stable-diffusion-v1-5"
)
NPCSH_IMAGE_GEN_PROVIDER = os.environ.get("NPCSH_IMAGE_GEN_PROVIDER", "diffusers")
NPCSH_VIDEO_GEN_MODEL = os.environ.get(
    "NPCSH_VIDEO_GEN_MODEL", "damo-vilab/text-to-video-ms-1.7b"
)
NPCSH_VIDEO_GEN_PROVIDER = os.environ.get("NPCSH_VIDEO_GEN_PROVIDER", "diffusers")

NPCSH_EMBEDDING_MODEL = os.environ.get("NPCSH_EMBEDDING_MODEL", "nomic-embed-text")
NPCSH_EMBEDDING_PROVIDER = os.environ.get("NPCSH_EMBEDDING_PROVIDER", "ollama")
NPCSH_REASONING_MODEL = os.environ.get("NPCSH_REASONING_MODEL", "deepseek-r1")
NPCSH_REASONING_PROVIDER = os.environ.get("NPCSH_REASONING_PROVIDER", "ollama")
NPCSH_STREAM_OUTPUT = eval(os.environ.get("NPCSH_STREAM_OUTPUT", "0")) == 1
NPCSH_API_URL = os.environ.get("NPCSH_API_URL", None)
NPCSH_SEARCH_PROVIDER = os.environ.get("NPCSH_SEARCH_PROVIDER", "duckduckgo")

READLINE_HISTORY_FILE = os.path.expanduser("~/.npcsh_history")
def setup_readline() -> str:
    if readline is None:
        return None
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
    if readline is None:
        return
    try:
        readline.write_history_file(READLINE_HISTORY_FILE)
    except OSError as e:
        print(f"Warning: Could not write readline history file {READLINE_HISTORY_FILE}: {e}")


