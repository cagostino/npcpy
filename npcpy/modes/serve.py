from flask import Flask, request, jsonify, Response
import configparser  # Add this with your other imports
from flask_sse import sse
import redis

from flask_cors import CORS
import os
import sqlite3
from datetime import datetime
import json
from pathlib import Path
import yaml
from dotenv import load_dotenv

from PIL import Image
from PIL import ImageFile

from npcpy.npc_sysenv import get_locally_available_models
from npcpy.memory.command_history import (
    CommandHistory,
    save_conversation_message,
)
from npcpy.npc_compiler import  Jinx, NPC

from npcpy.llm_funcs import (
    get_llm_response,    
)
from npcpy.npc_compiler import NPC
import base64

import json
import os
from pathlib import Path

# Path for storing settings
SETTINGS_FILE = Path(os.path.expanduser("~/.npcshrc"))

# Configuration
db_path = os.path.expanduser("~/npcsh_history.db")
user_npc_directory = os.path.expanduser("~/.npcsh/npc_team")
# Make project_npc_directory a function that updates based on current path
# instead of a static path relative to server launch directory
def get_project_npc_directory(current_path=None):
    """
    Get the project NPC directory based on the current path
    
    Args:
        current_path: The current path where project NPCs should be looked for
        
    Returns:
        Path to the project's npc_team directory
    """
    if current_path:
        return os.path.join(current_path, "npc_team")
    else:
        # Fallback to the old behavior if no path provided
        return os.path.abspath("./npc_team")

def load_project_env(current_path):
    """
    Load environment variables from a project's .env file
    
    Args:
        current_path: The current project directory path
    
    Returns:
        Dictionary of environment variables that were loaded
    """
    if not current_path:
        return {}
    
    env_path = os.path.join(current_path, ".env")
    loaded_vars = {}
    
    if os.path.exists(env_path):
        print(f"Loading project environment from {env_path}")
        # Load the environment variables into the current process
        # Note: load_dotenv returns a boolean, not a dictionary
        success = load_dotenv(env_path, override=True)
        
        if success:
            # Manually build a dictionary of loaded variables
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        if "=" in line:
                            key, value = line.split("=", 1)
                            loaded_vars[key.strip()] = value.strip().strip("\"'")
            
            print(f"Loaded {len(loaded_vars)} variables from project .env file")
        else:
            print(f"Failed to load environment variables from {env_path}")
    else:
        print(f"No .env file found at {env_path}")
    
    return loaded_vars

# Initialize components


app = Flask(__name__)
app.config["REDIS_URL"] = "redis://localhost:6379"
app.register_blueprint(sse, url_prefix="/stream")

redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True)

available_models = {}
CORS(
    app,
    origins=["http://localhost:5173"],
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    supports_credentials=True,
)


def get_db_connection():
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


extension_map = {
    "PNG": "images",
    "JPG": "images",
    "JPEG": "images",
    "GIF": "images",
    "SVG": "images",
    "MP4": "videos",
    "AVI": "videos",
    "MOV": "videos",
    "WMV": "videos",
    "MPG": "videos",
    "MPEG": "videos",
    "DOC": "documents",
    "DOCX": "documents",
    "PDF": "documents",
    "PPT": "documents",
    "PPTX": "documents",
    "XLS": "documents",
    "XLSX": "documents",
    "TXT": "documents",
    "CSV": "documents",
    "ZIP": "archives",
    "RAR": "archives",
    "7Z": "archives",
    "TAR": "archives",
    "GZ": "archives",
    "BZ2": "archives",
    "ISO": "archives",
}
def load_npc_by_name_and_source(name, source, db_conn=None, current_path=None):
    """
    Loads an NPC from either project or global directory based on source
    
    Args:
        name: The name of the NPC to load
        source: Either 'project' or 'global' indicating where to look for the NPC
        db_conn: Optional database connection
        current_path: The current path where project NPCs should be looked for
    
    Returns:
        NPC object or None if not found
    """
    if not db_conn:
        db_conn = get_db_connection()
    
    # Determine which directory to search
    if source == 'project':
        npc_directory = get_project_npc_directory(current_path)
        print(f"Looking for project NPC in: {npc_directory}")
    else:  # Default to global if not specified or unknown
        npc_directory = user_npc_directory
        print(f"Looking for global NPC in: {npc_directory}")
    
    # Look for the NPC file in the appropriate directory
    npc_path = os.path.join(npc_directory, f"{name}.npc")
    
    if os.path.exists(npc_path):
        try:
            npc = NPC(file=npc_path, db_conn=db_conn)
            return npc
        except Exception as e:
            print(f"Error loading NPC {name} from {source}: {str(e)}")
            return None
    else:
        print(f"NPC file not found: {npc_path}")
        return None

def fetch_messages_for_conversation(conversation_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    query = """
        SELECT role, content, timestamp
        FROM conversation_history
        WHERE conversation_id = ?
        ORDER BY timestamp ASC
    """
    cursor.execute(query, (conversation_id,))
    messages = cursor.fetchall()
    conn.close()

    return [
        {
            "role": message["role"],
            "content": message["content"],
            "timestamp": message["timestamp"],
        }
        for message in messages
    ]


@app.route("/api/attachments/<message_id>", methods=["GET"])
def get_message_attachments(message_id):
    """Get all attachments for a message"""
    try:
        command_history = CommandHistory(db_path)
        attachments = command_history.get_message_attachments(message_id)
        return jsonify({"attachments": attachments, "error": None})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/attachment/<attachment_id>", methods=["GET"])
def get_attachment(attachment_id):
    """Get specific attachment data"""
    try:
        command_history = CommandHistory(db_path)
        data, name, type = command_history.get_attachment_data(attachment_id)

        if data:
            # Convert binary data to base64 for sending
            base64_data = base64.b64encode(data).decode("utf-8")
            return jsonify(
                {"data": base64_data, "name": name, "type": type, "error": None}
            )
        return jsonify({"error": "Attachment not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/capture_screenshot", methods=["GET"])
def capture():
    # Capture screenshot using NPC-based method
    screenshot = capture_screenshot(None, full=True)

    # Ensure screenshot was captured successfully
    if not screenshot:
        print("Screenshot capture failed")
        return None

    return jsonify({"screenshot": screenshot})


@app.route("/api/settings/global", methods=["GET", "OPTIONS"])
def get_global_settings():
    if request.method == "OPTIONS":
        return "", 200

    try:
        npcshrc_path = os.path.expanduser("~/.npcshrc")

        # Default settings
        global_settings = {
            "model": "llama3.2",
            "provider": "ollama",
            "embedding_model": "nomic-embed-text",
            "embedding_provider": "ollama",
            "search_provider": "perplexity",
            "NPCSH_LICENSE_KEY": "",
            "default_folder": os.path.expanduser("~/.npcsh/"),
        }
        global_vars = {}

        if os.path.exists(npcshrc_path):
            with open(npcshrc_path, "r") as f:
                for line in f:
                    # Skip comments and empty lines
                    line = line.split("#")[0].strip()
                    if not line:
                        continue

                    if "=" not in line:
                        continue

                    # Split on first = only
                    key, value = line.split("=", 1)
                    key = key.strip()
                    if key.startswith("export "):
                        key = key[7:]

                    # Clean up the value - handle quoted strings properly
                    value = value.strip()
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]

                    # Map environment variables to settings
                    key_mapping = {
                        "NPCSH_MODEL": "model",
                        "NPCSH_PROVIDER": "provider",
                        "NPCSH_EMBEDDING_MODEL": "embedding_model",
                        "NPCSH_EMBEDDING_PROVIDER": "embedding_provider",
                        "NPCSH_SEARCH_PROVIDER": "search_provider",
                        "NPCSH_LICENSE_KEY": "NPCSH_LICENSE_KEY",
                        "NPCSH_STREAM_OUTPUT": "NPCSH_STREAM_OUTPUT",
                        "NPC_STUDIO_DEFAULT_FOLDER": "default_folder",
                    }

                    if key in key_mapping:
                        global_settings[key_mapping[key]] = value
                    else:
                        global_vars[key] = value

        print("Global settings loaded from .npcshrc")
        print(global_settings)
        return jsonify(
            {
                "global_settings": global_settings,
                "global_vars": global_vars,
                "error": None,
            }
        )

    except Exception as e:
        print(f"Error in get_global_settings: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/settings/global", methods=["POST", "OPTIONS"])
def save_global_settings():
    if request.method == "OPTIONS":
        return "", 200

    try:
        data = request.json
        npcshrc_path = os.path.expanduser("~/.npcshrc")

        key_mapping = {
            "model": "NPCSH_CHAT_MODEL",
            "provider": "NPCSH_CHAT_PROVIDER",
            "embedding_model": "NPCSH_EMBEDDING_MODEL",
            "embedding_provider": "NPCSH_EMBEDDING_PROVIDER",
            "search_provider": "NPCSH_SEARCH_PROVIDER",
            "NPCSH_LICENSE_KEY": "NPCSH_LICENSE_KEY",
            "NPCSH_STREAM_OUTPUT": "NPCSH_STREAM_OUTPUT",
            "default_folder": "NPC_STUDIO_DEFAULT_FOLDER",
        }

        os.makedirs(os.path.dirname(npcshrc_path), exist_ok=True)
        print(data)
        with open(npcshrc_path, "w") as f:
            # Write settings as environment variables
            for key, value in data.get("global_settings", {}).items():
                if key in key_mapping and value:
                    # Quote value if it contains spaces
                    if " " in str(value):
                        value = f'"{value}"'
                    f.write(f"export {key_mapping[key]}={value}\n")

            # Write custom variables
            for key, value in data.get("global_vars", {}).items():
                if key and value:
                    if " " in str(value):
                        value = f'"{value}"'
                    f.write(f"export {key}={value}\n")

        return jsonify({"message": "Global settings saved successfully", "error": None})

    except Exception as e:
        print(f"Error in save_global_settings: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/settings/project", methods=["GET", "OPTIONS"])  # Add OPTIONS
def get_project_settings():
    if request.method == "OPTIONS":
        return "", 200

    try:
        current_dir = request.args.get("path")
        if not current_dir:
            return jsonify({"error": "No path provided"}), 400

        env_path = os.path.join(current_dir, ".env")
        env_vars = {}

        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        if "=" in line:
                            key, value = line.split("=", 1)
                            env_vars[key.strip()] = value.strip().strip("\"'")

        return jsonify({"env_vars": env_vars, "error": None})

    except Exception as e:
        print(f"Error in get_project_settings: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/settings/project", methods=["POST", "OPTIONS"])  # Add OPTIONS
def save_project_settings():
    if request.method == "OPTIONS":
        return "", 200

    try:
        current_dir = request.args.get("path")
        if not current_dir:
            return jsonify({"error": "No path provided"}), 400

        data = request.json
        env_path = os.path.join(current_dir, ".env")

        with open(env_path, "w") as f:
            for key, value in data.get("env_vars", {}).items():
                f.write(f"{key}={value}\n")

        return jsonify(
            {"message": "Project settings saved successfully", "error": None}
        )

    except Exception as e:
        print(f"Error in save_project_settings: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/models", methods=["GET"])
def get_models():
    """
    Endpoint to retrieve available models based on the current project path.
    Checks for local configurations (.env) and Ollama.
    """
    global available_models
    current_path = request.args.get("currentPath")
    if not current_path:
        # Fallback to a default path or user home if needed,
        # but ideally the frontend should always provide it.
        current_path = os.path.expanduser("~/.npcsh")  # Or handle error
        print("Warning: No currentPath provided for /api/models, using default.")
        # return jsonify({"error": "currentPath parameter is required"}), 400

    try:
        # Reuse the existing function to detect models
        available_models = get_locally_available_models(current_path)

        # Optionally, add more details or format the response if needed
        # Example: Add a display name
        formatted_models = []
        for m, p in available_models.items():
            # Basic formatting, customize as needed
            text_only = (
                "(text only)"
                if p == "ollama"
                and m in ["llama3.2", "deepseek-v3", "phi4"]
                else ""
            )
            # Handle specific known model names for display
            display_model = m
            if "claude-3-5-haiku-latest" in m:
                display_model = "claude-3.5-haiku"
            elif "claude-3-5-sonnet-latest" in m:
                display_model = "claude-3.5-sonnet"
            elif "gemini-1.5-flash" in m:
                display_model = "gemini-1.5-flash"  # Handle multiple versions if needed
            elif "gemini-2.0-flash-lite-preview-02-05" in m:
                display_model = "gemini-2.0-flash-lite-preview"

            display_name = f"{display_model} | {p} {text_only}".strip()

            formatted_models.append(
                {
                    "value": m,  # Use the actual model ID as the value
                    "provider": p,
                    "display_name": display_name,
                }
            )

        return jsonify({"models": formatted_models, "error": None})

    except Exception as e:
        print(f"Error getting available models: {str(e)}")
        import traceback

        traceback.print_exc()
        # Return an empty list or a specific error structure
        return jsonify({"models": [], "error": str(e)}), 500

@app.route('/api/<command>', methods=['POST'])
def api_command(command):
    data = request.json or {}
    
    # Check if command exists
    handler = router.get_route(command)
    if not handler:
        return jsonify({"error": f"Unknown command: {command}"})
    
    # Check if it's shell-only
    if router.shell_only.get(command, False):
        return jsonify({"error": f"Command {command} is only available in shell mode"})
    
    # Execute the command handler
    try:
        # Convert positional args from JSON 
        args = data.get('args', [])
        kwargs = data.get('kwargs', {})
        
        # Add command name back to the command string
        command_str = command
        if args:
            command_str += " " + " ".join(str(arg) for arg in args)
            
        result = handler(command_str, **kwargs)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})
@app.route("/api/stream", methods=["POST"])
def stream():
    data = request.json
    print(data)
    commandstr = data.get("commandstr")
    conversation_id = data.get("conversationId")
    model = data.get("model", None)
    provider = data.get("provider", None)
    if provider is None:
        provider = available_models.get(model)
        
    npc_name = data.get("npc", None)
    npc_source = data.get("npcSource", "global")  # Default to global if not specified
    team = data.get("team", None)
    current_path = data.get("currentPath")
    
    # Load project-specific environment variables if currentPath is provided
    if current_path:
        loaded_vars = load_project_env(current_path)
        print(f"Loaded project env variables for stream request: {list(loaded_vars.keys())}")
    
    # Load the NPC if a name was provided
    npc_object = None
    if npc_name:
        db_conn = get_db_connection()
        # Pass the current_path parameter when loading project NPCs
        npc_object = load_npc_by_name_and_source(npc_name, npc_source, db_conn, current_path)
        
        if not npc_object and npc_source == 'project':
            # Try global as fallback
            print(f"NPC {npc_name} not found in project directory, trying global...")
            npc_object = load_npc_by_name_and_source(npc_name, 'global', db_conn)
            
        if npc_object:
            print(f"Successfully loaded NPC {npc_name} from {npc_source} directory")
        else:
            print(f"Warning: Could not load NPC {npc_name}")
    print(npc_object, type(npc_object))
    attachments = data.get("attachments", [])

    command_history = CommandHistory(db_path)

    # Process attachments and save them properly
    images = []
    print(attachments)

    from io import BytesIO
    from PIL import Image

    attachments_loaded = []

    if attachments:
        for attachment in attachments:
            extension = attachment["name"].split(".")[-1]
            extension_mapped = extension_map.get(extension.upper(), "others")
            file_path = os.path.expanduser(
                "~/.npcsh/" + extension_mapped + "/" + attachment["name"]
            )

            if extension_mapped == "images":
                # Open the image file and save it to the file path
                ImageFile.LOAD_TRUNCATED_IMAGES = True
                img = Image.open(attachment["path"])

                # Save the image to a BytesIO buffer (to extract binary data)
                img_byte_arr = BytesIO()
                img.save(img_byte_arr, format="PNG")  # or the appropriate format
                img_byte_arr.seek(0)  # Rewind the buffer to the beginning

                # Save the image to a file
                img.save(file_path, optimize=True, quality=50)

                # Add to images list for LLM processing
                images.append({"filename": attachment["name"], "file_path": file_path})

                # Add the image data (in binary form) to attachments_loaded
                attachments_loaded.append(
                    {
                        "name": attachment["name"],
                        "type": extension_mapped,
                        "data": img_byte_arr.read(),  # Read binary data from the buffer
                        "size": os.path.getsize(file_path),
                    }
                )

    messages = fetch_messages_for_conversation(conversation_id)
    if len(messages) == 0 and npc_object is not None:
        messages = [{'role': 'system', 'content': npc_object.get_system_prompt()}]
    elif len(messages)>0 and messages[0]['role'] != 'system' and npc_object is not None:
        # If the first message is not a system prompt, we need to add it
        messages.insert(0, {'role': 'system', 'content': npc_object.get_system_prompt()})
    elif len(messages) > 0 and npc_object is not None:
        messages[0]['content'] = npc_object.get_system_prompt() 
    # if we switch between npcs mid conversation, need to change the system prompt
    if npc_object is not None and messages and messages[0]['role'] == 'system':
        messages[0]['content'] = npc_object.get_system_prompt()
    print("messages ", messages)
    print("commandstr ", commandstr)
    message_id = command_history.generate_message_id()

    save_conversation_message(
        command_history,
        conversation_id,
        "user",
        commandstr,
        wd=current_path,
        model=model,
        provider=provider,
        npc=npc_name,
        team=team,
        attachments=attachments_loaded,
        message_id=message_id,
    )
    message_id = command_history.generate_message_id()

    stream_response = get_llm_response(
        commandstr,
        messages=messages,
        images=images,
        model=model,
        provider=provider,
        npc=npc_object,  # Pass the NPC object instead of just the name
        stream=True,
    )


    def event_stream():
        complete_response = []
        dot_count = 0
        tool_call_data = {"id": None, "function_name": None, "arguments": ""}

        for response_chunk in stream_response['response']:
            # Print progress dots for terminal feedback
            print('.', end="", flush=True)
            dot_count += 1
            
            if "hf.co" in model or provider == 'ollama':
                print("streaming from hf model through ollama")
                chunk_content = response_chunk["message"]["content"] if "message" in response_chunk and "content" in response_chunk["message"] else ""
                
                # Extract tool call info for Ollama
                if "message" in response_chunk and "tool_calls" in response_chunk["message"]:
                    for tool_call in response_chunk["message"]["tool_calls"]:
                        if "id" in tool_call:
                            tool_call_data["id"] = tool_call["id"]
                        if "function" in tool_call:
                            if "name" in tool_call["function"]:
                                tool_call_data["function_name"] = tool_call["function"]["name"]
                            if "arguments" in tool_call["function"]:
                                tool_call_data["arguments"] += tool_call["function"]["arguments"]
                
                if chunk_content:
                    complete_response.append(chunk_content)
                    
                # Keep original structure but add tool calls data
                chunk_data = {
                    "id": None,
                    "object": None,
                    "created": response_chunk["created_at"],
                    "model": response_chunk["model"],
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": chunk_content,
                                "role": response_chunk["message"]["role"],
                            },
                            "finish_reason": response_chunk.get("done_reason"),
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"

            else:
                # For LiteLLM format
                chunk_content = ""
                reasoning_content = ""
                
                # Extract tool call info for LiteLLM
                for choice in response_chunk.choices:
                    if hasattr(choice.delta, "tool_calls") and choice.delta.tool_calls:
                        for tool_call in choice.delta.tool_calls:
                            if tool_call.id:
                                tool_call_data["id"] = tool_call.id
                            if tool_call.function:
                                if hasattr(tool_call.function, "name") and tool_call.function.name:
                                    tool_call_data["function_name"] = tool_call.function.name
                                if hasattr(tool_call.function, "arguments") and tool_call.function.arguments:
                                    tool_call_data["arguments"] += tool_call.function.arguments
                
                # Check for reasoning content (thoughts)
                for choice in response_chunk.choices:
                    if hasattr(choice.delta, "reasoning_content"):
                        reasoning_content += choice.delta.reasoning_content
                
                # Get regular content
                chunk_content = "".join(
                    choice.delta.content
                    for choice in response_chunk.choices
                    if choice.delta.content is not None
                )
                
                if chunk_content:
                    complete_response.append(chunk_content)
                    
                # Keep original structure but add reasoning content
                chunk_data = {
                    "id": response_chunk.id,
                    "object": response_chunk.object,
                    "created": response_chunk.created,
                    "model": response_chunk.model,
                    "choices": [
                        {
                            "index": choice.index,
                            "delta": {
                                "content": choice.delta.content,
                                "role": choice.delta.role,
                                "reasoning_content": reasoning_content if hasattr(choice.delta, "reasoning_content") else None,
                            },
                            "finish_reason": choice.finish_reason,
                        }
                        for choice in response_chunk.choices
                    ],
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                save_conversation_message(
                    command_history,
                    conversation_id,
                    "assistant",
                    ''.join(complete_response),
                    wd=current_path,
                    model=model,
                    provider=provider,
                    npc = npc_object.name or '',
                    team=team,
                    message_id=message_id,  # Save with the same message_id
                )

        # Clear the dots by returning to the start of line and printing spaces
        print('\r' + ' ' * dot_count*2 + '\r', end="", flush=True)
        print('\n')

        # Send completion message
        yield f"data: {json.dumps({'type': 'message_stop'})}\n\n"
        save_conversation_message(
            command_history,
            conversation_id,
            "assistant",
            ''.join(complete_response),
            wd=current_path,
            model=model,
            provider=provider,
            npc=npc_object.name or '',
            team=team,
            message_id=message_id,
        )
        
    response = Response(event_stream(), mimetype="text/event-stream")

    return response


@app.route("/api/npc_team_global")
def get_npc_team_global():
    try:
        db_conn = get_db_connection()
        global_npc_directory = os.path.expanduser("~/.npcsh/npc_team")

        npc_data = []

        # Use existing helper to get NPCs from the global directory
        for file in os.listdir(global_npc_directory):
            if file.endswith(".npc"):
                npc_path = os.path.join(global_npc_directory, file)
                npc = NPC(file=npc_path, db_conn=db_conn)

                # Serialize the NPC data - updated for the new Jinx structure
                serialized_npc = {
                    "name": npc.name,
                    "primary_directive": npc.primary_directive,
                    "model": npc.model,
                    "provider": npc.provider,
                    "api_url": npc.api_url,
                    "use_global_jinxs": npc.use_global_jinxs,
                    "jinxs": [
                        {
                            "jinx_name": jinx.jinx_name,
                            "inputs": jinx.inputs,
                            "steps": [
                                {
                                    "name": step.get("name", f"step_{i}"),
                                    "engine": step.get("engine", "natural"),
                                    "code": step.get("code", "")
                                }
                                for i, step in enumerate(jinx.steps)
                            ]
                        }
                        for jinx in npc.jinxs
                    ],
                }
                npc_data.append(serialized_npc)

        return jsonify({"npcs": npc_data, "error": None})

    except Exception as e:
        print(f"Error loading global NPCs: {str(e)}")
        return jsonify({"npcs": [], "error": str(e)})


@app.route("/api/jinxs/global", methods=["GET"])
def get_global_jinxs():
    # try:
    user_home = os.path.expanduser("~")
    jinxs_dir = os.path.join(user_home, ".npcsh", "npc_team", "jinxs")
    jinxs = []
    if os.path.exists(jinxs_dir):
        for file in os.listdir(jinxs_dir):
            if file.endswith(".jinx"):
                with open(os.path.join(jinxs_dir, file), "r") as f:
                    jinx_data = yaml.safe_load(f)
                    jinxs.append(jinx_data)
            print("file", file)

    return jsonify({"jinxs": jinxs})


# except Exception as e:
#    return jsonify({"error": str(e)}), 500


@app.route("/api/jinxs/project", methods=["GET"])
def get_project_jinxs():
    current_path = request.args.get(
        "currentPath"
    )  # Correctly retrieves `currentPath` from query params
    if not current_path:
        return jsonify({"jinxs": []})

    if not current_path.endswith("npc_team"):
        current_path = os.path.join(current_path, "npc_team")

    jinxs_dir = os.path.join(current_path, "jinxs")
    jinxs = []
    if os.path.exists(jinxs_dir):
        for file in os.listdir(jinxs_dir):
            if file.endswith(".jinx"):
                with open(os.path.join(jinxs_dir, file), "r") as f:
                    jinx_data = yaml.safe_load(f)
                    jinxs.append(jinx_data)
    return jsonify({"jinxs": jinxs})


@app.route("/api/jinxs/save", methods=["POST"])
def save_jinx():
    try:
        data = request.json
        jinx_data = data.get("jinx")
        is_global = data.get("isGlobal")
        current_path = data.get("currentPath")
        jinx_name = jinx_data.get("jinx_name")

        if not jinx_name:
            return jsonify({"error": "Jinx name is required"}), 400

        if is_global:
            jinxs_dir = os.path.join(
                os.path.expanduser("~"), ".npcsh", "npc_team", "jinxs"
            )
        else:
            if not current_path.endswith("npc_team"):
                current_path = os.path.join(current_path, "npc_team")
            jinxs_dir = os.path.join(current_path, "jinxs")

        os.makedirs(jinxs_dir, exist_ok=True)

        # Full jinx structure
        jinx_yaml = {
            "description": jinx_data.get("description", ""),
            "inputs": jinx_data.get("inputs", []),
            "steps": jinx_data.get("steps", []),
        }

        file_path = os.path.join(jinxs_dir, f"{jinx_name}.jinx")
        with open(file_path, "w") as f:
            yaml.safe_dump(jinx_yaml, f, sort_keys=False)

        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/save_npc", methods=["POST"])
def save_npc():
    try:
        data = request.json
        npc_data = data.get("npc")
        is_global = data.get("isGlobal")
        current_path = data.get("currentPath")

        if not npc_data or "name" not in npc_data:
            return jsonify({"error": "Invalid NPC data"}), 400

        # Determine the directory based on whether it's global or project
        if is_global:
            npc_directory = os.path.expanduser("~/.npcsh/npc_team")
        else:
            npc_directory = os.path.join(current_path, "npc_team")

        # Ensure the directory exists
        os.makedirs(npc_directory, exist_ok=True)

        # Create the YAML content
        yaml_content = f"""name: {npc_data['name']}
primary_directive: "{npc_data['primary_directive']}"
model: {npc_data['model']}
provider: {npc_data['provider']}
api_url: {npc_data.get('api_url', '')}
use_global_jinxs: {str(npc_data.get('use_global_jinxs', True)).lower()}
"""

        # Save the file
        file_path = os.path.join(npc_directory, f"{npc_data['name']}.npc")
        with open(file_path, "w") as f:
            f.write(yaml_content)

        return jsonify({"message": "NPC saved successfully", "error": None})

    except Exception as e:
        print(f"Error saving NPC: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/npc_team_project", methods=["GET"])
def get_npc_team_project():
    try:
        db_conn = get_db_connection()

        project_npc_directory = request.args.get("currentPath")
        if not project_npc_directory.endswith("npc_team"):
            project_npc_directory = os.path.join(project_npc_directory, "npc_team")

        npc_data = []

        for file in os.listdir(project_npc_directory):
            print(file)
            if file.endswith(".npc"):
                npc_path = os.path.join(project_npc_directory, file)
                npc = NPC(file=npc_path, db_conn=db_conn)

                # Serialize the NPC data, updated for new Jinx structure
                serialized_npc = {
                    "name": npc.name,
                    "primary_directive": npc.primary_directive,
                    "model": npc.model,
                    "provider": npc.provider,
                    "api_url": npc.api_url,
                    "use_global_jinxs": npc.use_global_jinxs,
                    "jinxs": [
                        {
                            "jinx_name": jinx.jinx_name,
                            "inputs": jinx.inputs,
                            "steps": [
                                {
                                    "name": step.get("name", f"step_{i}"),
                                    "engine": step.get("engine", "natural"),
                                    "code": step.get("code", "")
                                }
                                for i, step in enumerate(jinx.steps)
                            ]
                        }
                        for jinx in npc.jinxs
                    ],
                }
                npc_data.append(serialized_npc)

        print(npc_data)
        return jsonify({"npcs": npc_data, "error": None})

    except Exception as e:
        print(f"Error fetching NPC team: {str(e)}")
        return jsonify({"npcs": [], "error": str(e)})


@app.route("/api/get_attachment_response", methods=["POST"])
def get_attachment_response():
    data = request.json
    attachments = data.get("attachments", [])
    messages = data.get("messages")  # Get conversation ID
    conversation_id = data.get("conversationId")
    current_path = data.get("currentPath")
    command_history = CommandHistory(db_path)
    model = data.get("model")
    npc_name = data.get("npc")
    npc_source = data.get("npcSource", "global")
    team = data.get("team")
    provider = data.get("provider")
    message_id = data.get("messageId")
    
    # Load project-specific environment variables if currentPath is provided
    if current_path:
        loaded_vars = load_project_env(current_path)
        print(f"Loaded project env variables for attachment response: {list(loaded_vars.keys())}")
    
    # Load the NPC if a name was provided
    npc_object = None
    if npc_name:
        db_conn = get_db_connection()
        # Pass the current_path parameter when loading project NPCs
        npc_object = load_npc_by_name_and_source(npc_name, npc_source, db_conn, current_path)
        
        if not npc_object and npc_source == 'project':
            # Try global as fallback
            print(f"NPC {npc_name} not found in project directory, trying global...")
            npc_object = load_npc_by_name_and_source(npc_name, 'global', db_conn)
            
        if npc_object:
            print(f"Successfully loaded NPC {npc_name} from {npc_source} directory")
        else:
            print(f"Warning: Could not load NPC {npc_name}")
    
    images = []
    for attachment in attachments:
        extension = attachment["name"].split(".")[-1]
        extension_mapped = extension_map.get(extension.upper(), "others")
        file_path = os.path.expanduser(
            "~/.npcsh/" + extension_mapped + "/" + attachment["name"]
        )
        if extension_mapped == "images":
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            img = Image.open(attachment["path"])
            img.save(file_path, optimize=True, quality=50)
            images.append({"filename": attachment["name"], "file_path": file_path})

    message_to_send = messages[-1]["content"][0]

    response = get_llm_response(
        message_to_send,
        images=images,
        messages=messages,
        model=model,
        npc=npc_object,  # Pass the NPC object instead of just the name
    )
    messages = response["messages"]
    response = response["response"]

    # Save new messages
    save_conversation_message(
        command_history, conversation_id, "user", message_to_send, wd=current_path, team=team, 
        model=model, provider=provider, npc=npc, attachments=attachments
        
    )

    save_conversation_message(
        command_history,
        conversation_id,
        "assistant",
        response,
        wd=current_path,
        team=team,
        model=model,
        provider=provider,
        npc=npc,
        attachments=attachments,
        message_id=message_id, ) # Save with the same message_id    )
    return jsonify(
        {
            "status": "success",
            "message": response,
            "conversationId": conversation_id,
            "messages": messages,  # Optionally return fetched messages
        }
    )


@app.route("/api/execute", methods=["POST"])
def execute():
    try:
        data = request.json
        command = data.get("commandstr")
        current_path = data.get("currentPath")
        conversation_id = data.get("conversationId")
        model = data.get("model")
        print("model", model)
        npc_name = data.get("npc")
        npc_source = data.get("npcSource", "global")
        team = data.get("team")
        print("npc", npc_name)
        
        # Load project-specific environment variables if currentPath is provided
        if current_path:
            loaded_vars = load_project_env(current_path)
            print(f"Loaded project env variables for execute request: {list(loaded_vars.keys())}")
        
        # Load the NPC if a name was provided
        npc_object = None
        if npc_name:
            db_conn = get_db_connection()
            # Pass the current_path parameter when loading project NPCs
            npc_object = load_npc_by_name_and_source(npc_name, npc_source, db_conn, current_path)
            
            if not npc_object and npc_source == 'project':
                # Try global as fallback
                print(f"NPC {npc_name} not found in project directory, trying global...")
                npc_object = load_npc_by_name_and_source(npc_name, 'global', db_conn)
                
            if npc_object:
                print(f"Successfully loaded NPC {npc_name} from {npc_source} directory")
            else:
                print(f"Warning: Could not load NPC {npc_name}")
                
        # Clean command
        command = command.strip().replace('"', "").replace("'", "").replace("`", "")

        if not command:
            return (
                jsonify(
                    {
                        "error": "No command provided",
                        "output": "Error: No command provided",
                    }
                ),
                400,
            )

        command_history = CommandHistory(db_path)

        # Fetch conversation history
        if conversation_id:
            conn = get_db_connection()
            cursor = conn.cursor()

            # Get all messages for this conversation in order
            cursor.execute(
                """
                SELECT role, content, timestamp
                FROM conversation_history
                WHERE conversation_id = ?
                ORDER BY timestamp ASC
            """,
                (conversation_id,),
            )

            conversation_messages = cursor.fetchall()

            # Format messages for LLM
            messages = [
                {
                    "role": msg["role"],
                    "content": msg["content"],
                }
                for msg in conversation_messages
            ]

            conn.close()
        else:
            messages = []

        # Execute command with conversation history

        result = execute_command(
            command=command,
            command_history=command_history,
            db_path=db_path,
            npc_compiler=npc_compiler,
            conversation_id=conversation_id,
            messages=messages,
            model=model,
            npc=npc_object,  # Pass the NPC object instead of just the name
        )


        # Save new messages
        save_conversation_message(
            command_history, conversation_id, "user", command, wd=current_path,
            model=model,
            provider=available_models.get(model),
            npc=npc,
            attachments=[],
            message_id=command_history.generate_message_id(),
            team=team,
            
        )

        save_conversation_message(
            command_history,
            conversation_id,
            "assistant",
            result.get("output", ""),
            wd=current_path,
            model=model,
            provider=available_models.get(model),
            npc=npc,
            attachments=[],
            team=team,
            
            
        )

        return jsonify(
            {
                "output": result.get("output", ""),
                "currentPath": os.getcwd(),
                "error": None,
                "messages": messages,  # Return updated messages
            }
        )

    except Exception as e:
        print(f"Error executing command: {str(e)}")
        import traceback

        traceback.print_exc()
        return (
            jsonify(
                {
                    "error": str(e),
                    "output": f"Error: {str(e)}",
                    "currentPath": data.get("currentPath", None),
                }
            ),
            500,
        )


def get_conversation_history(conversation_id):
    """Fetch all messages for a conversation in chronological order."""
    if not conversation_id:
        return []

    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        query = """
            SELECT role, content, timestamp
            FROM conversation_history
            WHERE conversation_id = ?
            ORDER BY timestamp ASC
        """
        cursor.execute(query, (conversation_id,))
        messages = cursor.fetchall()

        return [
            {
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg["timestamp"],
            }
            for msg in messages
        ]
    finally:
        conn.close()


@app.route("/api/conversations", methods=["GET"])
def get_conversations():
    try:
        path = request.args.get("path")
        if not path:
            return jsonify({"error": "No path provided", "conversations": []}), 400

        conn = get_db_connection()
        try:
            cursor = conn.cursor()

            query = """
            SELECT DISTINCT conversation_id,
                   MIN(timestamp) as start_time,
                   GROUP_CONCAT(content) as preview
            FROM conversation_history
            WHERE directory_path = ?
            GROUP BY conversation_id
            ORDER BY start_time DESC
            """

            cursor.execute(query, [path])
            conversations = cursor.fetchall()

            return jsonify(
                {
                    "conversations": [
                        {
                            "id": conv["conversation_id"],
                            "timestamp": conv["start_time"],
                            "preview": (
                                conv["preview"][:100] + "..."
                                if conv["preview"] and len(conv["preview"]) > 100
                                else conv["preview"]
                            ),
                        }
                        for conv in conversations
                    ],
                    "error": None,
                }
            )

        finally:
            conn.close()

    except Exception as e:
        print(f"Error getting conversations: {str(e)}")
        return jsonify({"error": str(e), "conversations": []}), 500


@app.route("/api/conversation/<conversation_id>/messages", methods=["GET"])
def get_conversation_messages(conversation_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Modified query to ensure proper ordering and deduplication
        query = """
            WITH ranked_messages AS (
                SELECT
                    ch.*,
                    GROUP_CONCAT(ma.id) as attachment_ids,
                    ROW_NUMBER() OVER (
                        PARTITION BY ch.role, strftime('%s', ch.timestamp)
                        ORDER BY ch.id DESC
                    ) as rn
                FROM conversation_history ch
                LEFT JOIN message_attachments ma
                    ON ch.message_id = ma.message_id
                WHERE ch.conversation_id = ?
                GROUP BY ch.id, ch.timestamp
            )
            SELECT *
            FROM ranked_messages
            WHERE rn = 1
            ORDER BY timestamp ASC, id ASC
        """

        cursor.execute(query, [conversation_id])
        messages = cursor.fetchall()
        #print(messages)

        return jsonify(
            {
                "messages": [
                    {
                        "message_id": msg["message_id"],
                        "role": msg["role"],
                        "content": msg["content"],
                        "timestamp": msg["timestamp"],
                        "model": msg["model"],
                        "provider": msg["provider"],
                        "npc": msg["npc"],
                        "attachments": (
                            get_message_attachments(msg["message_id"])
                            if msg["attachment_ids"]
                            else []
                        ),
                    }
                    for msg in messages
                ],
                "error": None,
            }
        )

    except Exception as e:
        print(f"Error getting conversation messages: {str(e)}")
        return jsonify({"error": str(e), "messages": []}), 500
    finally:
        conn.close()


@app.route("/api/stream_raw", methods=["POST"])
def stream_raw():
    """SSE stream that takes messages, models, providers, and attachments from frontend."""
    data = request.json
    commandstr = data.get("commandstr")
    conversation_id = data.get("conversationId")
    model = data.get("model", None)
    provider = data.get("provider", None)
    save_to_sqlite3 = data.get("saveToSqlite3", False)
    npc = data.get("npc", None)
    attachments = data.get("attachments", [])
    current_path = data.get("currentPath")
    print(data)

    messages = data.get("messages", [])
    print("messages", messages)
    command_history = CommandHistory(db_path)

    images = []
    attachments_loaded = []

    if attachments:
        for attachment in attachments:
            extension = attachment["name"].split(".")[-1]
            extension_mapped = extension_map.get(extension.upper(), "others")
            file_path = os.path.expanduser(
                "~/.npcsh/" + extension_mapped + "/" + attachment["name"]
            )

            if extension_mapped == "images":
                # Open the image file and save it to the file path
                ImageFile.LOAD_TRUNCATED_IMAGES = True
                img = Image.open(attachment["path"])

                # Save the image to a BytesIO buffer (to extract binary data)
                img_byte_arr = BytesIO()
                img.save(img_byte_arr, format="PNG")  # or the appropriate format
                img_byte_arr.seek(0)  # Rewind the buffer to the beginning

                # Save the image to a file
                img.save(file_path, optimize=True, quality=50)

                # Add to images list for LLM processing
                images.append({"filename": attachment["name"], "file_path": file_path})

                # Add the image data (in binary form) to attachments_loaded
                attachments_loaded.append(
                    {
                        "name": attachment["name"],
                        "type": extension_mapped,
                        "data": img_byte_arr.read(),  # Read binary data from the buffer
                        "size": os.path.getsize(file_path),
                    }
                )
    if save_to_sqlite3:
        if len(messages) == 0:
            # load the conversation messages
            messages = fetch_messages_for_conversation(conversation_id)
        if not messages:
            return jsonify({"error": "No messages provided"}), 400
        messages.append({"role": "user", "content": commandstr})
        message_id = command_history.generate_message_id()

        save_conversation_message(
            command_history,
            conversation_id,
            "user",
            commandstr,
            wd=current_path,
            model=model,
            provider=provider,
            npc=npc,
            attachments=attachments_loaded,
            message_id=message_id,
        )
        message_id = command_history.generate_message_id()

    stream_response = get_stream(
        messages,
        images=images,
        model=model,
        provider=provider,
        npc=npc if isinstance(npc, NPC) else None,
    )

    """else:

        stream_response = execute_command_stream(
            commandstr,
            command_history,
            db_path,
            npc_compiler,
            model=model,
            provider=provider,
            messages=messages,
            images=images,  # Pass the processed images
        )  # Get all conversation messages so far
    """
    final_response = ""  # To accumulate the assistant's response
    complete_response = []  # List to store all chunks
    def event_stream():
        str_output = ""
        tool_call_data = {"id": None, "function_name": None, "arguments": ""}

        for response_chunk in stream_response['response']:
            chunk_content = ""
            
            if "hf.co" in model or provider == 'ollama':
                print("streaming from hf model through ollama")
                chunk_content = response_chunk["message"]["content"]
                
                # Extract tool call info for Ollama
                if "message" in response_chunk and "tool_calls" in response_chunk["message"]:
                    for tool_call in response_chunk["message"]["tool_calls"]:
                        if "id" in tool_call:
                            tool_call_data["id"] = tool_call["id"]
                        if "function" in tool_call:
                            if "name" in tool_call["function"]:
                                tool_call_data["function_name"] = tool_call["function"]["name"]
                            if "arguments" in tool_call["function"]:
                                tool_call_data["arguments"] += tool_call["function"]["arguments"]
                
                if chunk_content:
                    complete_response.append(chunk_content)
                    
                # Keep original structure but add tool calls data
                chunk_data = {
                    "id": None,
                    "object": None,
                    "created": response_chunk["created_at"],
                    "model": response_chunk["model"],
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": chunk_content,
                                "role": response_chunk["message"]["role"],
                            },
                            "finish_reason": response_chunk.get("done_reason"),
                        }
                    ],
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"

            else:
                # For LiteLLM format
                chunk_content = ""
                reasoning_content = ""
                
                # Extract tool call info for LiteLLM
                for choice in response_chunk.choices:
                    if hasattr(choice.delta, "tool_calls") and choice.delta.tool_calls:
                        for tool_call in choice.delta.tool_calls:
                            if tool_call.id:
                                tool_call_data["id"] = tool_call.id
                            if tool_call.function:
                                if hasattr(tool_call.function, "name") and tool_call.function.name:
                                    tool_call_data["function_name"] = tool_call.function.name
                                if hasattr(tool_call.function, "arguments") and tool_call.function.arguments:
                                    tool_call_data["arguments"] += tool_call.function.arguments
                
                # Check for reasoning content (thoughts)
                for choice in response_chunk.choices:
                    if hasattr(choice.delta, "reasoning_content"):
                        reasoning_content += choice.delta.reasoning_content
                
                # Get regular content
                chunk_content = "".join(
                    choice.delta.content
                    for choice in response_chunk.choices
                    if choice.delta.content is not None
                )
                
                if chunk_content:
                    complete_response.append(chunk_content)
                    
                # Keep original structure but add reasoning content
                chunk_data = {
                    "id": response_chunk.id,
                    "object": response_chunk.object,
                    "created": response_chunk.created,
                    "model": response_chunk.model,
                    "choices": [
                        {
                            "index": choice.index,
                            "delta": {
                                "content": choice.delta.content,
                                "role": choice.delta.role,
                                "reasoning_content": reasoning_content if hasattr(choice.delta, "reasoning_content") else None,
                            },
                            "finish_reason": choice.finish_reason,
                        }
                        for choice in response_chunk.choices
                    ],
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
                save_conversation_message(
                    command_history,
                    conversation_id,
                    "assistant",
                    chunk_content,
                    wd=current_path,
                    model=model,
                    provider=provider,
                    npc = npc.name or '',
                    team=team,
                    message_id=message_id,  # Save with the same message_id
                )

        # Clear and render markdown
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
        
        yield f"data: {json.dumps({'type': 'message_stop'})}\n\n"
        full_content = command_history.get_full_message_content(message_id)
        command_history.update_message_content(message_id, full_content)

    response = Response(event_stream(), mimetype="text/event-stream")
    return response



@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS")
    response.headers.add("Access-Control-Allow-Credentials", "true")
    return response


def get_db_connection():
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


extension_map = {
    "PNG": "images",
    "JPG": "images",
    "JPEG": "images",
    "GIF": "images",
    "SVG": "images",
    "MP4": "videos",
    "AVI": "videos",
    "MOV": "videos",
    "WMV": "videos",
    "MPG": "videos",
    "MPEG": "videos",
    "DOC": "documents",
    "DOCX": "documents",
    "PDF": "documents",
    "PPT": "documents",
    "PPTX": "documents",
    "XLS": "documents",
    "XLSX": "documents",
    "TXT": "documents",
    "CSV": "documents",
    "ZIP": "archives",
    "RAR": "archives",
    "7Z": "archives",
    "TAR": "archives",
    "GZ": "archives",
    "BZ2": "archives",
    "ISO": "archives",
}


def fetch_messages_for_conversation(conversation_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    query = """
        SELECT role, content, timestamp
        FROM conversation_history
        WHERE conversation_id = ?
        ORDER BY timestamp ASC
    """
    cursor.execute(query, (conversation_id,))
    messages = cursor.fetchall()
    conn.close()

    return [
        {
            "role": message["role"],
            "content": message["content"],
            "timestamp": message["timestamp"],
        }
        for message in messages
    ]


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "error": None})


def start_flask_server(
    port=5337,
    cors_origins=None,
    static_files=None, 
    debug = False
):
    try:
        # Ensure the database tables exist


        command_history = CommandHistory(db_path)


        # Only apply CORS if origins are specified
        if cors_origins:
            from flask_cors import CORS

            CORS(
                app,
                origins=cors_origins,
                allow_headers=["Content-Type", "Authorization"],
                methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                supports_credentials=True,
                
            )

        # Run the Flask app on all interfaces
        print(f"Starting Flask server on http://0.0.0.0:{port}")
        app.run(host="0.0.0.0", port=port, debug=debug)
    except Exception as e:
        print(f"Error starting server: {str(e)}")


if __name__ == "__main__":
    start_flask_server()
