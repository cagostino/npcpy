import argparse
import sys
import os
import sqlite3
import traceback
from typing import Optional

from npcpy.npc_sysenv import (
    NPCSH_CHAT_MODEL, NPCSH_CHAT_PROVIDER,
    NPCSH_API_URL, NPCSH_DB_PATH, NPCSH_STREAM_OUTPUT,
    print_and_process_stream_with_markdown,
    render_markdown,
)
from npcpy.npc_compiler import NPC, Team
from npcpy.routes import router
from npcpy.llm_funcs import check_llm_command

def load_npc_by_name(npc_name: str = "sibiji", db_path: str = NPCSH_DB_PATH) -> Optional[NPC]:
    if not npc_name:
        npc_name = "sibiji"

    project_npc_path = os.path.abspath(f"./npc_team/{npc_name}.npc")
    global_npc_path = os.path.expanduser(f"~/.npcsh/npc_team/{npc_name}.npc")

    chosen_path = None
    if os.path.exists(project_npc_path):
        chosen_path = project_npc_path
    elif os.path.exists(global_npc_path):
        chosen_path = global_npc_path
    elif os.path.exists(f"npcs/{npc_name}.npc"):
         chosen_path = f"npcs/{npc_name}.npc"

    if chosen_path:
        try:
            db_conn = sqlite3.connect(db_path)
            npc = NPC(file=chosen_path, db_conn=db_conn)
            return npc
        except Exception as e:
            print(f"Warning: Failed to load NPC '{npc_name}' from {chosen_path}: {e}", file=sys.stderr)
            return None
    else:
        print(f"Warning: NPC file for '{npc_name}' not found in project or global paths.", file=sys.stderr)
        if npc_name != "sibiji":
            return load_npc_by_name("sibiji", db_path)
        return None

def main():
    parser = argparse.ArgumentParser(
        description="NPC Command Line Utilities. Call a command or provide a prompt for the default NPC.",
        usage="npc <command> [command_args...] | <prompt> [--npc NAME] [--model MODEL] [--provider PROV]"
    )
    parser.add_argument(
        "--model", "-m", help="LLM model to use (overrides NPC/defaults)", type=str, default=None
    )
    parser.add_argument(
        "--provider", "-pr", help="LLM provider to use (overrides NPC/defaults)", type=str, default=None
    )
    parser.add_argument(
        "-n", "--npc", help="Name of the NPC to use (default: sibiji)", type=str, default="sibiji"
    )

    # No subparsers setup at first - we'll conditionally create them

    # First, get any arguments without parsing commands
    args, all_args = parser.parse_known_args()

    # Check if the first argument is a known command
    is_valid_command = False
    command_name = None
    if all_args and all_args[0] in router.get_commands():
        is_valid_command = True
        command_name = all_args[0]
        all_args = all_args[1:]  # Remove the command from arguments

    # Only set up subparsers if we have a valid command
    if is_valid_command:
        subparsers = parser.add_subparsers(dest="command", title="Available Commands",
                                         help="Run 'npc <command> --help' for command-specific help")

        for cmd_name, help_text in router.help_info.items():
            if router.shell_only.get(cmd_name, False):
                continue

            cmd_parser = subparsers.add_parser(cmd_name, help=help_text, add_help=False)
            cmd_parser.add_argument('command_args', nargs=argparse.REMAINDER,
                                    help='Arguments passed directly to the command handler')

        # Re-parse with command subparsers
        args = parser.parse_args([command_name] + all_args)
        command_args = args.command_args if hasattr(args, 'command_args') else []
        unknown_args = []
    else:
        # Treat all arguments as a prompt
        args.command = None
        command_args = []
        unknown_args = all_args

    npc_instance = load_npc_by_name(args.npc, NPCSH_DB_PATH)

    effective_model = args.model or NPCSH_CHAT_MODEL
    effective_provider = args.provider or NPCSH_CHAT_PROVIDER

    extras = {}

    # Process command args if we have a valid command
    if is_valid_command:
        # Parse command args properly
        if command_args:
            i = 0
            while i < len(command_args):
                arg = command_args[i]
                if arg.startswith("--"):
                    param = arg[2:]  # Remove --
                    if "=" in param:
                        param_name, param_value = param.split("=", 1)
                        extras[param_name] = param_value
                        i += 1
                    elif i + 1 < len(command_args) and not command_args[i+1].startswith("--"):
                        extras[param] = command_args[i+1]
                        i += 2
                    else:
                        extras[param] = True
                        i += 1
                else:
                    i += 1
            
        handler = router.get_route(command_name)
        if not handler:
            print(f"Error: Command '{command_name}' recognized but no handler found.", file=sys.stderr)
            sys.exit(1)

        full_command_str = command_name
        if command_args:
            full_command_str += " " + " ".join(command_args)
        
        handler_kwargs = {
            "model": effective_model,
            "provider": effective_provider,
            "npc": npc_instance,
            "api_url": NPCSH_API_URL,
            "stream": NPCSH_STREAM_OUTPUT,
            "messages": [],
            "team": None,
            "current_path": os.getcwd(),
            **extras
        }

        try:
            result = handler(command=full_command_str, **handler_kwargs)

            if isinstance(result, dict):
                output = result.get("output") or result.get("response")
                
                if NPCSH_STREAM_OUTPUT and not isinstance(output, str):
                     print_and_process_stream_with_markdown(output, effective_model, effective_provider)
                elif output is not None:
                     render_markdown(str(output))
            elif result is not None:
                render_markdown(str(result))
            else:
                print(f"Command '{command_name}' executed.")

        except Exception as e:
            print(f"Error executing command '{command_name}': {e}", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)
    else:
        # Process as a prompt
        prompt = " ".join(unknown_args)

        if not prompt:
            # If no prompt and no command, show help
            parser.print_help()
            sys.exit(1)

        print(f"Processing prompt: '{prompt}' with NPC: '{args.npc}'...")
        try:
            response_data = check_llm_command(
                command=prompt,
                model=effective_model,
                provider=effective_provider,
                npc=npc_instance,
                stream=NPCSH_STREAM_OUTPUT,
                messages=[],
                team=None,
                api_url=NPCSH_API_URL,
            )

            if isinstance(response_data, dict):
                output = response_data.get("output")
                if NPCSH_STREAM_OUTPUT and hasattr(output, '__iter__') and not isinstance(output, (str, bytes, dict, list)):
                    print_and_process_stream_with_markdown(output, effective_model, effective_provider)
                elif output is not None:
                    render_markdown(str(output))
            elif response_data is not None:
                 render_markdown(str(response_data))

        except Exception as e:
            print(f"Error processing prompt: {e}", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)

if __name__ == "__main__":
    main()