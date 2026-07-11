import datetime
from flask import Flask, request, jsonify, Response
try:
    from flask_sse import sse
except ImportError:
    sse = None
try:
    import redis
except ImportError:
    redis = None
import threading
import uuid
import sys
import traceback
import glob
import re
import time
import asyncio
import argparse
from typing import Optional, List, Dict, Callable, Any
from contextlib import AsyncExitStack
import io
from flask_cors import CORS
import os
import sqlite3
import json
from pathlib import Path
import yaml
from dotenv import load_dotenv
try:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
except ImportError:
    ClientSession = None
    StdioServerParameters = None
    stdio_client = None
from PIL import Image
from PIL import ImageFile
from io import BytesIO
import networkx as nx
from collections import defaultdict
import numpy as np
import pandas as pd 
import subprocess
import platform
try:
    import ollama 
except Exception:
    pass
from jinja2 import Environment, FileSystemLoader, Template, Undefined, DictLoader
class SilentUndefined(Undefined):
    def _fail_with_undefined_error(self, *args, **kwargs):
        return ""
from npcpy.db import generate_message_id, ensure_engine
from npcpy.memory.knowledge_graph import (
    find_similar_facts_chroma,
)
from npcpy.gen.response import calculate_cost
from npcpy.memory.search import execute_rag_command
from npcpy.data.load import load_file_contents
from npcpy.data.web import search_web
from npcpy.data.image import capture_screenshot
import base64
import shutil
import uuid
from npcpy.llm_funcs import gen_image, gen_video, breathe                                                                                                                                                                
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from npcpy.npc_sysenv import (
    get_data_dir, get_models_dir,
    get_images_dir, get_videos_dir,
    get_attachments_dir, get_logs_dir, lookup_provider,
    get_locally_available_models,
    team_sync_status, team_sync_init, team_sync_pull,
    team_sync_resolve, team_sync_commit, team_sync_diff,
)
from npcpy.npc_compiler import  Jinx, NPC, Team, load_jinxes_from_directory, build_jinx_tool_catalog, initialize_npc_project, load_yaml_file
from npcpy.llm_funcs import (
    get_llm_response, check_llm_command
)
from npcpy.gen.embeddings import get_embeddings
from termcolor import cprint
from npcpy.tools import auto_tools
from npcpy.streaming import (
    StreamConfig, StreamEvent,
    clean_messages_for_llm,
    ensure_system_prompt,
    parse_stream_chunk,
    format_sse_event,
    format_sse_raw,
    resolve_npc_tools,
    execute_tool,
    create_chat_stream,
    create_tool_agent_stream,
    create_jinx_stream,
)
import json
import os
from pathlib import Path
from flask_cors import CORS
cancellation_flags = {}
cancellation_lock = threading.Lock()
# Pending permission requests from /api/stream to the Rust/frontend shell.
permission_requests = {}
permission_lock = threading.Lock()
class ServeState:
    """Minimal server-side execution context for jinxes and tools.
    Minimal server-side execution context for jinxes and tools."""
    def __init__(
        self,
        npc=None,
        team=None,
        conversation_id=None,
        chat_model=None,
        chat_provider=None,
        current_path=None,
        search_provider=None,
        embedding_model=None,
        embedding_provider=None,
    ):
        self.npc = npc
        self.team = team
        self.conversation_id = conversation_id
        self.chat_model = chat_model
        self.chat_provider = chat_provider
        self.current_path = current_path or os.getcwd()
        self.search_provider = search_provider
        self.embedding_model = embedding_model
        self.embedding_provider = embedding_provider
def _setup_stream(data):
    stream_id = data.get("streamId") or str(uuid.uuid4())
    with cancellation_lock:
        cancellation_flags[stream_id] = False
    return stream_id
def _cleanup_stream(stream_id, mcp_state_key=None):
    with cancellation_lock:
        cancellation_flags.pop(stream_id, None)
    if mcp_state_key and hasattr(app, 'mcp_clients') and mcp_state_key in app.mcp_clients:
        print(f"[CLEANUP] Removing MCP state for {mcp_state_key}")
        del app.mcp_clients[mcp_state_key]
def _serialize_jinxes_from_dir(directory):
    jinx_data = []
    for jinx in load_jinxes_from_directory(directory):
        d = jinx.to_dict()
        if jinx._source_path:
            rel = os.path.relpath(jinx._source_path, directory)
            d["path"] = rel[:-5] if rel.endswith(".jinx") else rel
            d["source_path"] = jinx._source_path
        jinx_data.append(d)
    return jinx_data
def normalize_path_for_db(path_str):
    """
    Normalize a path for consistent database storage/querying.
    Converts backslashes to forward slashes for cross-platform compatibility.
    This ensures Windows paths match Unix paths in the database.
    """
    if not path_str:
        return path_str
    normalized = path_str.replace('\\', '/')
    normalized = normalized.rstrip('/')
    return normalized
class MCPClientNPC:
    def __init__(self, debug: bool = True):
        self.debug = debug
        self.session: Optional[ClientSession] = None
        self._exit_stack: Optional[AsyncExitStack] = None
        self.available_tools_llm: List[Dict[str, Any]] = []
        self.tool_map: Dict[str, Callable] = {}
        self.server_script_path: Optional[str] = None
        self.server_spec = None
    def _log(self, message: str, color: str = "cyan") -> None:
        if self.debug:
            cprint(f"[MCP Client] {message}", color, file=sys.stderr)
    def _get_loop(self):
        """Get or create a usable event loop for the current thread."""
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                return loop
        except RuntimeError:
            pass
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop
    def _cleanup_sync(self):
        """Clean up any existing session and exit stack synchronously."""
        if self._exit_stack is not None:
            try:
                loop = self._get_loop()
                loop.run_until_complete(self._exit_stack.aclose())
            except Exception as e:
                self._log(f"Cleanup warning: {e}", "yellow")
            self._exit_stack = None
        self.session = None
        self.available_tools_llm = []
        self.tool_map = {}
    async def _connect_async(self, server_spec) -> None:
        """Connect to an MCP server.
        server_spec can be:
          - str: path to a script file (legacy)
          - str: command string starting with python/npx/uvx/node/docker
          - dict with 'path': local script file
          - dict with 'command' + 'args': arbitrary stdio command (npx, docker, uvx, node, etc.)
          - dict with 'url': SSE/HTTP remote server
        """
        if isinstance(server_spec, str):
            if _is_command_string(server_spec):
                import shlex
                parts = shlex.split(server_spec.strip())
                server_spec = {"command": parts[0], "args": parts[1:]}
            else:
                server_spec = {"path": server_spec}
        self.server_spec = server_spec
        extra_env = server_spec.get("env", {})
        env = {**os.environ, **extra_env}
        if self.session and self._exit_stack:
            try:
                await self._exit_stack.aclose()
            except Exception as e:
                self._log(f"Old session cleanup warning: {e}", "yellow")
            self._exit_stack = None
            self.session = None
        self._exit_stack = AsyncExitStack()
        if "url" in server_spec:
            from mcp.client.sse import sse_client
            url = server_spec["url"]
            self._log(f"Connecting to SSE server: {url}")
            self.server_script_path = url
            sse_transport = await self._exit_stack.enter_async_context(sse_client(url))
            self.session = await self._exit_stack.enter_async_context(ClientSession(*sse_transport))
        elif "command" in server_spec:
            command = server_spec["command"]
            args = server_spec.get("args", [])
            self._log(f"Connecting via command: {command} {' '.join(str(a) for a in args)}")
            self.server_script_path = f"{command}:{' '.join(str(a) for a in args)}"
            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=env,
            )
            stdio_transport = await self._exit_stack.enter_async_context(stdio_client(server_params))
            self.session = await self._exit_stack.enter_async_context(ClientSession(*stdio_transport))
        elif "path" in server_spec:
            abs_path = os.path.abspath(os.path.expanduser(server_spec["path"]))
            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"MCP server script not found: {abs_path}")
            self.server_script_path = abs_path
            self._log(f"Attempting to connect to MCP server: {abs_path}")
            if abs_path.endswith('.py'):
                cmd_parts = [sys.executable, abs_path]
            elif os.access(abs_path, os.X_OK):
                cmd_parts = [abs_path]
            else:
                raise ValueError(f"Unsupported MCP server script type or not executable: {abs_path}")
            server_params = StdioServerParameters(
                command=cmd_parts[0],
                args=[abs_path],
                env=env,
                cwd=os.path.dirname(abs_path) or "."
            )
            stdio_transport = await self._exit_stack.enter_async_context(stdio_client(server_params))
            self.session = await self._exit_stack.enter_async_context(ClientSession(*stdio_transport))
        else:
            raise ValueError(f"Invalid MCP server spec: must have 'path', 'command', or 'url'. Got: {server_spec}")
        await self.session.initialize()
        response = await self.session.list_tools()
        self.available_tools_llm = []
        self.tool_map = {}
        if response.tools:
            for mcp_tool in response.tools:
                tool_def = {
                    "type": "function",
                    "function": {
                        "name": mcp_tool.name,
                        "description": mcp_tool.description or f"MCP tool: {mcp_tool.name}",
                        "parameters": getattr(mcp_tool, "inputSchema", {"type": "object", "properties": {}})
                    }
                }
                self.available_tools_llm.append(tool_def)
                def make_tool_func(tool_name_closure):
                    async def tool_func(**kwargs):
                        if not self.session:
                            return {"error": "No MCP session"}
                        self._log(f"About to call MCP tool {tool_name_closure}")
                        try:
                            cleaned_kwargs = {k: (None if v == 'None' else v) for k, v in kwargs.items()}
                            result = await asyncio.wait_for(
                                self.session.call_tool(tool_name_closure, cleaned_kwargs),
                                timeout=30.0
                            )
                            self._log(f"MCP tool {tool_name_closure} returned: {type(result)}")
                            return result
                        except asyncio.TimeoutError:
                            self._log(f"Tool {tool_name_closure} timed out after 30 seconds", "red")
                            return {"error": f"Tool {tool_name_closure} timed out"}
                        except Exception as e:
                            self._log(f"Tool {tool_name_closure} error: {e}", "red")
                            return {"error": str(e)}
                    def sync_wrapper(**kwargs):
                        self._log(f"Sync wrapper called for {tool_name_closure}")
                        loop = self._get_loop()
                        return loop.run_until_complete(tool_func(**kwargs))
                    return sync_wrapper
                self.tool_map[mcp_tool.name] = make_tool_func(mcp_tool.name)
        tool_names = list(self.tool_map.keys())
        self._log(f"Connection successful. Tools: {', '.join(tool_names) if tool_names else 'None'}")
    def connect_sync(self, server_spec) -> bool:
        """Connect synchronously. server_spec: str (path) or dict with path/command/url."""
        self._cleanup_sync()
        loop = self._get_loop()
        try:
            loop.run_until_complete(self._connect_async(server_spec))
            return True
        except Exception as e:
            cprint(f"MCP connection failed: {e}", "red", file=sys.stderr)
            self._cleanup_sync()
            return False
    def disconnect_sync(self):
        self._cleanup_sync()
    def is_connected(self) -> bool:
        """Check if the session is still alive."""
        if self.session is None or self._exit_stack is None:
            return False
        loop = self._get_loop()
        try:
            loop.run_until_complete(asyncio.wait_for(self.session.list_tools(), timeout=5.0))
            return True
        except Exception as e:
            self._log(f"Health check failed: {e}", "yellow")
            return False
def get_llm_response_with_handling(prompt, npc,model, provider, messages, tools, stream, team, context=None, **kwargs):
    """Unified LLM response with basic exception handling (inlined from corca to avoid that dependency)."""
    try:
        return get_llm_response(
            prompt=prompt,
            npc=npc,
            model=model,
            provider=provider,
            messages=messages,
            tools=tools,
            auto_process_tool_calls=False,
            stream=stream,
            team=team,
            context=context,
            **kwargs,
        )
    except Exception as e:
        print(f"[LLM ERROR] First attempt failed: {e}")
        import traceback
        traceback.print_exc()
        try:
            return get_llm_response(
                prompt=prompt,
                npc=npc,
                model=model,
                provider=provider,
                messages=messages,
                tools=tools,
                auto_process_tool_calls=False,
                stream=stream,
                team=team,
                context=context,
                **kwargs,
            )
        except Exception as e2:
            print(f"[LLM ERROR] Second attempt failed: {e2}")
            traceback.print_exc()
            raise
class MCPServerManager:
    """
    Simple in-process tracker for launching/stopping MCP servers.
    Currently uses subprocess.Popen to start a Python stdio MCP server script.
    """
    def __init__(self):
        self._procs = {}
        self._lock = threading.Lock()
    def start(self, server_path: str, env_vars: dict = None):
        server_path = os.path.expanduser(server_path)
        proc_env = os.environ.copy()
        if env_vars:
            proc_env.update(env_vars)
        is_command = _is_command_string(server_path)
        stripped = server_path.strip()
        if is_command:
            import shlex
            cmd = shlex.split(stripped)
            key = stripped
            cwd = os.getcwd()
        else:
            abs_path = os.path.abspath(server_path)
            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"MCP server script not found at {abs_path}")
            cmd = [sys.executable, abs_path]
            key = abs_path
            cwd = os.path.dirname(abs_path) or "."
        with self._lock:
            existing = self._procs.get(key)
            if existing and existing.poll() is None:
                return {"status": "running", "pid": existing.pid, "serverPath": key}
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=proc_env,
            )
            self._procs[key] = proc
            return {"status": "started", "pid": proc.pid, "serverPath": key}
    def _resolve_key(self, server_path: str) -> str:
        """Resolve server_path to the key used in _procs."""
        if _is_command_string(server_path):
            return server_path.strip()
        return os.path.abspath(os.path.expanduser(server_path))
    def stop(self, server_path: str):
        key = self._resolve_key(server_path)
        with self._lock:
            proc = self._procs.get(key)
            if not proc:
                return {"status": "not_found", "serverPath": key}
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            del self._procs[key]
            return {"status": "stopped", "serverPath": key}
    def status(self, server_path: str):
        key = self._resolve_key(server_path)
        with self._lock:
            proc = self._procs.get(key)
            if not proc:
                return {"status": "not_started", "serverPath": key}
            running = proc.poll() is None
            return {
                "status": "running" if running else "exited",
                "serverPath": key,
                "pid": proc.pid,
                "returncode": None if running else proc.returncode,
            }
    def running(self):
        with self._lock:
            return {
                path: {
                    "pid": proc.pid,
                    "status": "running" if proc.poll() is None else "exited",
                    "returncode": None if proc.poll() is None else proc.returncode,
                }
                for path, proc in self._procs.items()
            }
mcp_server_manager = MCPServerManager()
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
        success = load_dotenv(env_path, override=True)
        if success:
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
def _load_kg_from_yaml_stores(store_paths):
    """Aggregate .knowledge.yaml stores from explicit paths into DataFrames."""
    from npcpy.memory.knowledge_store import KnowledgeStore
    concepts = []
    facts = []
    links = []
    for store_dir in store_paths:
        store = KnowledgeStore(store_dir)
        data = store.load()
        for c in data.get('concepts', []):
            concepts.append({
                'name': c.get('name'),
                'description': c.get('description', ''),
                'generation': c.get('generation', 0),
                'created_at': c.get('created_at'),
            })
        for m in data.get('memories', []):
            stmt = m.get('final_memory') or m.get('initial_memory', '')
            if stmt:
                facts.append({
                    'statement': stmt,
                    'source_text': m.get('source_id', ''),
                    'type': m.get('source_type', 'memory'),
                    'generation': 0,
                    'memory_id': m.get('id'),
                    'npc_name': m.get('npc', ''),
                    'team_name': m.get('team', ''),
                })
        for l in data.get('links', []):
            links.append({
                'source': l.get('from'),
                'target': l.get('to'),
                'link_type': l.get('type', 'memory_to_memory'),
                'weight': 1,
            })
    concepts_df = pd.DataFrame(concepts) if concepts else pd.DataFrame(columns=['name', 'description', 'generation', 'created_at'])
    facts_df = pd.DataFrame(facts) if facts else pd.DataFrame(columns=['statement', 'source_text', 'type', 'generation', 'memory_id', 'npc_name', 'team_name'])
    links_df = pd.DataFrame(links) if links else pd.DataFrame(columns=['source', 'target', 'link_type', 'weight'])
    return concepts_df, facts_df, links_df
def _load_kg_from_yaml(workspace):
    """DEPRECATED — kept for backward compat until callers migrate to storePaths."""
    return _load_kg_from_yaml_stores([])
def load_kg_data(store_paths=None):
    """Load KG data from a specific list of .knowledge.yaml store directories.
    store_paths is a list of directory paths."""
    return _load_kg_from_yaml_stores(store_paths or [])
def _get_registered_stores():
    """Read the configured KG registry YAML and return store directory paths."""
    registry_path = app.config.get('KG_REGISTRY_PATH')
    if not registry_path:
        return []
    registry_path = os.path.expanduser(registry_path)
    if not os.path.exists(registry_path):
        return []
    try:
        with open(registry_path, 'r') as f:
            data = yaml.safe_load(f) or {}
        stores = data.get('stores', [])
        return [str(s) for s in stores if isinstance(s, str) and s]
    except Exception:
        return []
app = Flask(__name__)
app.config["REDIS_URL"] = "redis://localhost:6379"
app.config['DB_PATH'] = ''
app.jinx_conversation_contexts ={}
redis_client = redis.Redis(host="localhost", port=6379, decode_responses=True) if redis else None
available_models = {}
CORS(
    app,
    origins=["http://localhost:5173"],
    allow_headers=["Content-Type", "Authorization"],
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    supports_credentials=True,
)
def _is_command_string(s: str) -> bool:
    """Check if a string is a command (python/npx/uvx/node/docker) vs a file path."""
    stripped = s.strip()
    if not stripped or ' ' not in stripped:
        return False
    first_word = stripped.split()[0]
    basename = os.path.basename(first_word)
    return basename in ('python', 'python3', 'npx', 'uvx', 'node', 'docker') or ' -m ' in stripped
def get_db_connection():
    engine = create_engine('sqlite:///' + app.config.get('DB_PATH'))
    return engine
def get_db_session():
    engine = get_db_connection()
    Session = sessionmaker(bind=engine)
    return Session()
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        return super().default(obj)
def resolve_mcp_server_path(current_path=None, explicit_path=None, force_global=False):
    """
    Resolve an MCP server path.
    Supports both file paths and command strings:
      - Command strings (python -m ..., npx ..., etc.) are passed through as-is
      - File paths are resolved to absolute paths
      - Fallback: use `python -m npcpy.mcp_server` (the module, not a deployed script)
    """
    if explicit_path:
        if _is_command_string(explicit_path):
            return explicit_path.strip()
        abs_path = os.path.abspath(os.path.expanduser(explicit_path))
        if os.path.exists(abs_path):
            return abs_path
    team_path = None
    if current_path:
        candidate = os.path.join(current_path, "npc_team")
        if os.path.isdir(candidate):
            team_path = candidate
    if team_path:
        return f"{sys.executable} -m npcpy.mcp_server --team {team_path}"
    return f"{sys.executable} -m npcpy.mcp_server"
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
def load_npc_by_name_and_source(name, source, current_path=None):
    """
    Loads an NPC from either project or global directory based on source.
    Database features are opt-in via NPC.initialize_db(); no DB connection is
    opened implicitly here.
    Args:
        name: The name of the NPC to load
        source: Either 'project' or 'global' indicating where to look for the NPC
        current_path: The current path where project NPCs should be looked for
    Returns:
        NPC object or None if not found
    """
    if source == 'project':
        directories = [get_project_npc_directory(current_path)]
    else:
        directories = [
            app.config['user_npc_directory'],
        ]
    for npc_directory in directories:
        if not npc_directory or not os.path.exists(npc_directory):
            continue
        npc_path = os.path.join(npc_directory, f"{name}.npc")
        if not os.path.exists(npc_path):
            if not any(os.path.exists(os.path.join(npc_directory, p)) for p in (
                'agents.md', 'AGENTS.md', 'CLAUDE.md', 'agents'
            )):
                continue
        try:
            team = Team(team_path=npc_directory)
            npc = team.npcs.get(name)
            if npc is not None:
                return npc
        except Exception as e:
            print(f"Error loading team from {npc_directory} while resolving NPC {name}: {str(e)}")
            continue
    print(f"NPC file not found: {name}.npc in {directories}")
    return None
@app.route('/api/kg/generations')
def list_generations():
    return jsonify({"generations": [0]})
@app.route('/api/kg/graph')
def get_graph_data():
    store_paths = request.args.getlist('storePaths')
    concepts_df, facts_df, links_df = load_kg_data(store_paths)
    nodes = []
    nodes.extend([{'id': name, 'type': 'concept'} for name in concepts_df['name']])
    has_memory_id = 'memory_id' in facts_df.columns
    for _, row in facts_df.iterrows():
        node = {'id': row['statement'], 'type': 'fact'}
        if has_memory_id:
            mid = row.get('memory_id')
            try:
                if mid is not None and not pd.isna(mid):
                    node['memory_id'] = int(mid)
            except Exception:
                pass
        nodes.append(node)
    links = [{'source': row['source'], 'target': row['target']} for _, row in links_df.iterrows()]
    return jsonify(graph={'nodes': nodes, 'links': links})
@app.route('/api/kg/network-stats')
def get_network_stats():
    store_paths = request.args.getlist('storePaths')
    _, _, links_df = load_kg_data(store_paths)
    G = nx.DiGraph()
    for _, link in links_df.iterrows():
        G.add_edge(link['source'], link['target'])
    n_nodes = G.number_of_nodes()
    if n_nodes == 0:
        return jsonify(stats={'nodes': 0, 'edges': 0, 'density': 0, 'avg_degree': 0, 'node_degrees': {}})
    degrees = dict(G.degree())
    stats = {
        'nodes': n_nodes, 'edges': G.number_of_edges(), 'density': nx.density(G),
        'avg_degree': np.mean(list(degrees.values())) if degrees else 0, 'node_degrees': degrees
    }
    return jsonify(stats=stats)
@app.route('/api/kg/cooccurrence')
def get_cooccurrence_network():
    store_paths = request.args.getlist('storePaths')
    min_cooccurrence = request.args.get('min_cooccurrence', 2, type=int)
    _, _, links_df = load_kg_data(store_paths)
    fact_to_concepts = defaultdict(set)
    for _, link in links_df.iterrows():
        if link['type'] == 'fact_to_concept':
            fact_to_concepts[link['source']].add(link['target'])
    cooccurrence = defaultdict(int)
    for concepts in fact_to_concepts.values():
        concepts_list = list(concepts)
        for i, c1 in enumerate(concepts_list):
            for c2 in concepts_list[i+1:]:
                pair = tuple(sorted((c1, c2)))
                cooccurrence[pair] += 1
    G_cooccur = nx.Graph()
    for (c1, c2), weight in cooccurrence.items():
        if weight >= min_cooccurrence:
            G_cooccur.add_edge(c1, c2, weight=weight)
    if G_cooccur.number_of_nodes() == 0:
        return jsonify(network={'nodes': [], 'links': []})
    components = list(nx.connected_components(G_cooccur))
    node_to_community = {node: i for i, component in enumerate(components) for node in component}
    nodes = [{'id': node, 'type': 'concept', 'community': node_to_community.get(node, 0)} for node in G_cooccur.nodes()]
    links = [{'source': u, 'target': v, 'weight': d['weight']} for u, v, d in G_cooccur.edges(data=True)]
    return jsonify(network={'nodes': nodes, 'links': links})
@app.route('/api/kg/centrality')
def get_centrality_data():
    store_paths = request.args.getlist('storePaths')
    concepts_df, _, links_df = load_kg_data(store_paths)
    G = nx.Graph()
    fact_concept_links = links_df[links_df['type'] == 'fact_to_concept']
    for _, link in fact_concept_links.iterrows():
        if link['target'] in concepts_df['name'].values:
            G.add_edge(link['source'], link['target'])
    concept_degree = {node: cent for node, cent in nx.degree_centrality(G).items() if node in concepts_df['name'].values}
    return jsonify(centrality={'degree': concept_degree})
@app.route('/api/kg/search')
def search_kg():
    """Search facts and concepts by keyword"""
    try:
        q = request.args.get('q', '').strip().lower()
        store_paths = request.args.getlist('storePaths')
        search_type = request.args.get('type', 'both')
        limit = request.args.get('limit', 50, type=int)
        if not q:
            return jsonify({"error": "Query parameter 'q' is required"}), 400
        concepts_df, facts_df, links_df = load_kg_data(store_paths)
        results = {"facts": [], "concepts": [], "query": q}
        if search_type in ('both', 'fact'):
            for _, row in facts_df.iterrows():
                statement = str(row.get('statement', '')).lower()
                source_text = str(row.get('source_text', '')).lower()
                if q in statement or q in source_text:
                    results["facts"].append({
                        "statement": row.get('statement'),
                        "source_text": row.get('source_text'),
                        "type": row.get('type'),
                        "generation": row.get('generation'),
                        "origin": row.get('origin')
                    })
                    if len(results["facts"]) >= limit:
                        break
        if search_type in ('both', 'concept'):
            for _, row in concepts_df.iterrows():
                name = str(row.get('name', '')).lower()
                description = str(row.get('description', '')).lower()
                if q in name or q in description:
                    results["concepts"].append({
                        "name": row.get('name'),
                        "description": row.get('description'),
                        "generation": row.get('generation'),
                        "origin": row.get('origin')
                    })
                    if len(results["concepts"]) >= limit:
                        break
        return jsonify(results)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route('/api/kg/embed', methods=['POST'])
def embed_kg_facts():
    """Embed existing facts from YAML stores to Chroma for semantic search"""
    try:
        data = request.get_json() or {}
        store_paths = data.get('storePaths', [])
        batch_size = data.get('batch_size', 10)
        _, facts_df, _ = load_kg_data(store_paths)
        if facts_df.empty:
            return jsonify({"message": "No facts to embed", "count": 0})
        chroma_db_path = app.config.get('CHROMA_DB_PATH')
        if not chroma_db_path:
            return jsonify({"error": "CHROMA_DB_PATH not configured"}), 500
        from npcpy.memory.knowledge_graph import store_fact_with_embedding
        import hashlib
        embedded_count = 0
        skipped_count = 0
        statements = facts_df['statement'].dropna().tolist()
        for i in range(0, len(statements), batch_size):
            batch = statements[i:i + batch_size]
            try:
                embeddings = get_embeddings(batch)
            except Exception as e:
                print(f"Failed to get embeddings for batch {i}: {e}")
                continue
            for j, statement in enumerate(batch):
                fact_id = hashlib.md5(statement.encode()).hexdigest()
                try:
                    existing = chroma_collection.get(ids=[fact_id])
                    if existing and existing.get('ids'):
                        skipped_count += 1
                        continue
                except Exception:
                    pass
                row = facts_df[facts_df['statement'] == statement].iloc[0] if len(facts_df[facts_df['statement'] == statement]) > 0 else None
                metadata = {
                    "generation": int(row.get('generation', 0)) if row is not None and pd.notna(row.get('generation')) else 0,
                    "origin": str(row.get('origin', '')) if row is not None else '',
                    "type": str(row.get('type', '')) if row is not None else '',
                }
                result = store_fact_with_embedding(
                    chroma_collection, statement, metadata, embeddings[j]
                )
                if result:
                    embedded_count += 1
        return jsonify({
            "message": f"Embedded {embedded_count} facts, skipped {skipped_count} existing",
            "embedded": embedded_count,
            "skipped": skipped_count,
            "total_facts": len(statements)
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route('/api/kg/search/semantic')
def search_kg_semantic():
    """Semantic search for facts using vector similarity"""
    try:
        q = request.args.get('q', '').strip()
        store_paths = request.args.getlist('storePaths')
        limit = request.args.get('limit', 10, type=int)
        if not q:
            return jsonify({"error": "Query parameter 'q' is required"}), 400
        chroma_db_path = app.config.get('CHROMA_DB_PATH')
        if not chroma_db_path:
            return jsonify({"error": "CHROMA_DB_PATH not configured", "facts": [], "query": q}), 500
        try:
            query_embedding = get_embeddings([q])[0]
        except Exception as e:
            return jsonify({
                "error": f"Failed to generate embedding: {str(e)}",
                "facts": [],
                "query": q
            }), 200
        similar_facts = find_similar_facts_chroma(
            chroma_collection,
            q,
            query_embedding=query_embedding,
            n_results=limit,
            metadata_filter=None
        )
        results = {
            "facts": [
                {
                    "statement": f["fact"],
                    "distance": f.get("distance"),
                    "metadata": f.get("metadata", {}),
                    "id": f.get("id")
                }
                for f in similar_facts
            ],
            "query": q,
            "total": len(similar_facts)
        }
        return jsonify(results)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route('/api/kg/facts')
def get_kg_facts():
    """Get facts from YAML stores."""
    try:
        store_paths = request.args.getlist('storePaths')
        limit = request.args.get('limit', 100, type=int)
        offset = request.args.get('offset', 0, type=int)
        _, facts_df, _ = load_kg_data(store_paths)
        facts = []
        for i, row in facts_df.iloc[offset:offset+limit].iterrows():
            mid = row.get('memory_id') if 'memory_id' in facts_df.columns else None
            try:
                mid_int = int(mid) if mid is not None and not pd.isna(mid) else None
            except Exception:
                mid_int = None
            facts.append({
                "statement": row.get('statement'),
                "source_text": row.get('source_text'),
                "type": row.get('type'),
                "generation": row.get('generation'),
                "origin": row.get('origin'),
                "memory_id": mid_int,
            })
        return jsonify({
            "facts": facts,
            "total": len(facts_df),
            "offset": offset,
            "limit": limit
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route('/api/kg/concepts')
def get_kg_concepts():
    """Get concepts from YAML stores."""
    try:
        store_paths = request.args.getlist('storePaths')
        limit = request.args.get('limit', 100, type=int)
        concepts_df, _, _ = load_kg_data(store_paths)
        concepts = []
        for _, row in concepts_df.head(limit).iterrows():
            concepts.append({
                "name": row.get('name'),
                "description": row.get('description'),
                "generation": row.get('generation'),
                "origin": row.get('origin')
            })
        return jsonify({
            "concepts": concepts,
            "total": len(concepts_df)
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route('/api/kg/node', methods=['POST'])
def add_kg_node():
    """Add a new concept or fact to the selected YAML stores."""
    try:
        data = request.get_json()
        node_id = data.get('id')
        node_type = data.get('type', 'concept')
        properties = data.get('properties', {})
        store_paths = data.get('storePaths', [])
        if not node_id:
            return jsonify({"error": "Missing node id"}), 400
        if not store_paths:
            return jsonify({"error": "storePaths are required"}), 400
        added = 0
        for sp in store_paths:
            store = KnowledgeStore(sp)
            d = store.load()
            if node_type == 'fact':
                exists = any((m.get('final_memory') or m.get('initial_memory')) == node_id for m in d.get('memories', []))
                if not exists:
                    d['memories'].append({
                        'id': _make_id(), 'initial_memory': node_id, 'final_memory': node_id,
                        'source_type': 'manual', 'status': 'auto-extracted',
                        'created_at': _utcnow(),
                    })
                    added += 1
            else:
                exists = any(c.get('name') == node_id for c in d.get('concepts', []))
                if not exists:
                    d['concepts'].append({
                        'id': _make_id(), 'name': node_id,
                        'description': properties.get('description', ''),
                        'created_at': _utcnow(),
                    })
                    added += 1
            store.save(d)
        return jsonify({"success": True, "id": node_id, "added_to": added})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route('/api/kg/node/<path:node_id>', methods=['PUT'])
def update_kg_node(node_id):
    """Update a concept description across selected YAML stores."""
    try:
        data = request.get_json()
        properties = data.get('properties', {})
        store_paths = data.get('storePaths', [])
        if not store_paths:
            return jsonify({"error": "storePaths are required"}), 400
        updated = 0
        for sp in store_paths:
            store = KnowledgeStore(sp)
            d = store.load()
            for c in d.get('concepts', []):
                if c.get('name') == node_id:
                    c['description'] = properties.get('description', c.get('description', ''))
                    updated += 1
            store.save(d)
        return jsonify({"success": True, "updated": updated})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route('/api/kg/node/<path:node_id>', methods=['DELETE'])
def delete_kg_node(node_id):
    """Delete a concept/fact and its links from selected YAML stores."""
    try:
        data = request.get_json() or {}
        store_paths = data.get('storePaths', [])
        if not store_paths:
            return jsonify({"error": "storePaths are required"}), 400
        removed = 0
        for sp in store_paths:
            store = KnowledgeStore(sp)
            d = store.load()
            pre_concepts = len(d.get('concepts', []))
            pre_memories = len(d.get('memories', []))
            d['concepts'] = [c for c in d.get('concepts', []) if c.get('name') != node_id]
            d['memories'] = [m for m in d.get('memories', []) if (m.get('final_memory') or m.get('initial_memory')) != node_id]
            d['links'] = [l for l in d.get('links', []) if l.get('from') != node_id and l.get('to') != node_id]
            store.save(d)
            removed += (pre_concepts - len(d['concepts'])) + (pre_memories - len(d['memories']))
        return jsonify({"success": True, "removed": removed})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route('/api/kg/edge', methods=['POST'])
def add_kg_edge():
    """Add a new edge/link to selected YAML stores."""
    try:
        data = request.get_json()
        source = data.get('source')
        target = data.get('target')
        edge_type = data.get('type', 'related_to')
        store_paths = data.get('storePaths', [])
        if not source or not target:
            return jsonify({"error": "Missing source or target"}), 400
        if not store_paths:
            return jsonify({"error": "storePaths are required"}), 400
        added = 0
        for sp in store_paths:
            store = KnowledgeStore(sp)
            d = store.load()
            exists = any(l.get('from') == source and l.get('to') == target for l in d.get('links', []))
            if not exists:
                d['links'].append({
                    'id': _make_id(), 'from': source, 'to': target,
                    'relation': edge_type, 'type': 'manual',
                    'created_at': _utcnow(),
                })
                added += 1
            store.save(d)
        return jsonify({"success": True, "added_to": added})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route('/api/kg/edge/<path:source_id>/<path:target_id>', methods=['DELETE'])
def delete_kg_edge(source_id, target_id):
    """Delete an edge from selected YAML stores."""
    try:
        data = request.get_json() or {}
        store_paths = data.get('storePaths', [])
        if not store_paths:
            return jsonify({"error": "storePaths are required"}), 400
        removed = 0
        for sp in store_paths:
            store = KnowledgeStore(sp)
            d = store.load()
            pre = len(d.get('links', []))
            d['links'] = [l for l in d.get('links', []) if not (l.get('from') == source_id and l.get('to') == target_id)]
            if len(d['links']) < pre:
                removed += 1
            store.save(d)
        return jsonify({"success": True, "removed": removed})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route('/api/kg/trigger', methods=['POST'])
def trigger_kg_process():
    """Trigger a KG process (sleep or dream) on selected YAML stores."""
    try:
        data = request.get_json() or {}
        process_type = data.get('process_type', 'sleep')
        store_paths = data.get('storePaths', [])
        model = data.get('model') or None
        provider = data.get('provider') or None
        if not store_paths:
            return jsonify({"error": "storePaths are required"}), 400
        total_changes = {"concepts_added": 0, "links_added": 0}
        for sp in store_paths:
            store = KnowledgeStore(sp)
            if process_type == 'sleep':
                result = store.sleep(model=model, provider=provider)
            elif process_type == 'dream':
                result = store.dream(model=model, provider=provider)
            elif process_type == 'evolve':
                result = store.assimilate(model=model, provider=provider)
            else:
                return jsonify({"error": f"Unknown process type: {process_type}. Use 'sleep', 'dream', or 'evolve'."}), 400
            total_changes["concepts_added"] += result.get("concepts_added", 0)
            total_changes["links_added"] += result.get("links_added", 0)
        return jsonify({
            "success": True,
            "process_type": process_type,
            "changes": total_changes
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route('/api/kg/ingest', methods=['POST'])
def ingest_to_kg():
    """Ingest text content into selected YAML stores."""
    try:
        data = request.get_json() or {}
        content_text = data.get('content', '')
        context = data.get('context', '')
        store_paths = data.get('storePaths', [])
        model = data.get('model') or None
        provider = data.get('provider') or None
        if not content_text or not content_text.strip():
            return jsonify({"error": "content is required"}), 400
        if not store_paths:
            return jsonify({"error": "storePaths are required"}), 400
        total_facts = 0
        total_concepts = 0
        for sp in store_paths:
            store = KnowledgeStore(sp)
            result = store.create(model=model, provider=provider, context=context, content_text=content_text)
            total_facts += result.get('facts', 0)
            total_concepts += result.get('concepts', 0)
        return jsonify({
            "success": True,
            "facts": total_facts,
            "concepts": total_concepts
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route('/api/kg/pipeline/run', methods=['POST'])
def run_kg_pipeline():
    """Run a KG pipeline step (create/assimilate/sleep/dream) on .knowledge.yaml stores.
    Streams NDJSON log lines."""
    from npcpy.memory.knowledge_store import KnowledgeStore
    data = request.get_json() or {}
    step = data.get('step')
    store_paths = data.get('storePaths', [])
    model = data.get('model') or None
    provider = data.get('provider') or None
    context = data.get('context', '')
    content_text = data.get('contentText') or None
    operations = data.get('operations') or None
    num_seeds = data.get('numSeeds')
    if num_seeds is not None:
        try:
            num_seeds = int(num_seeds)
        except (ValueError, TypeError):
            num_seeds = 3
    if not step or not store_paths:
        return jsonify({"error": "step and storePaths are required"}), 400
    def generate():
        job_id = data.get('jobId', f"kg_{int(time.time()*1000)}")
        for store_dir in store_paths:
            store = KnowledgeStore(store_dir)
            yield json.dumps({
                "jobId": job_id,
                "kind": "start",
                "message": f"Starting {step} on {store_dir}",
                "timestamp": int(time.time() * 1000),
            }) + "\n"
            try:
                if step == 'create':
                    result = store.create(model=model, provider=provider, npc=None, context=context, content_text=content_text)
                elif step == 'assimilate':
                    result = store.assimilate(model=model, provider=provider, npc=None, context=context)
                elif step == 'sleep':
                    result = store.sleep(model=model, provider=provider, npc=None, context=context, operations=operations)
                elif step == 'dream':
                    result = store.dream(model=model, provider=provider, npc=None, context=context, num_seeds=num_seeds)
                else:
                    yield json.dumps({
                        "jobId": job_id,
                        "kind": "error",
                        "message": f"Unknown step: {step}",
                        "timestamp": int(time.time() * 1000),
                    }) + "\n"
                    continue
                yield json.dumps({
                    "jobId": job_id,
                    "kind": "finish",
                    "message": f"Finished {step} on {store_dir}: {result.get('concepts_added', 0)} concepts, {result.get('links_added', 0)} links",
                    "data": result,
                    "timestamp": int(time.time() * 1000),
                }) + "\n"
            except Exception as e:
                traceback.print_exc()
                yield json.dumps({
                    "jobId": job_id,
                    "kind": "error",
                    "message": str(e),
                    "timestamp": int(time.time() * 1000),
                }) + "\n"
        yield json.dumps({
            "jobId": job_id,
            "kind": "done",
            "message": "All stores processed",
            "timestamp": int(time.time() * 1000),
        }) + "\n"
    return Response(generate(), mimetype='application/x-ndjson')
@app.route('/api/kg/query', methods=['POST'])
def query_kg():
    """Query the knowledge graph with a natural language question. Returns an LLM response grounded in KG facts.
    Modes:
      - "keyword" (default): plain keyword-overlap scoring (original behavior)
      - "traversal": Poisson-sampled depth/breadth traversal via an ephemeral KGIndividual
      - "sememolution": delegate to a persisted population by population_id; ranks multiple candidates
    """
    try:
        data = request.get_json() or {}
        question = data.get('question', '')
        top_k = data.get('top_k', 15)
        mode = (data.get('mode') or 'keyword').lower()
        lambda_depth = float(data.get('lambda_depth', 2.0))
        lambda_breadth = float(data.get('lambda_breadth', 5.0))
        similarity_threshold = float(data.get('similarity_threshold', 0.6))
        population_id = data.get('population_id')
        if not question.strip():
            return jsonify({"error": "question is required"}), 400
        db_path = app.config.get('DB_PATH')
        engine = create_engine('sqlite:///' + db_path)
        model = app.config.get('DEFAULT_MODEL', None)
        provider = app.config.get('DEFAULT_PROVIDER', None)
        if mode == 'sememolution' and population_id:
            from npcpy.memory.kg_population import load_population, save_population
            mgr = load_population(engine, population_id)
            if not mgr:
                return jsonify({"error": f"population '{population_id}' not found"}), 404
            if model: mgr.model = model
            if provider: mgr.provider = provider
            rankings = mgr.query_and_rank(question)
            try: save_population(engine, population_id, population_id, mgr)
            except Exception: pass
            return jsonify({
                "mode": "sememolution",
                "population_id": population_id,
                "candidates": [
                    {
                        "rank": c.get('rank'),
                        "individual_id": c['individual'].individual_id,
                        "response": c['response'],
                        "n_facts": c['n_facts'],
                        "context_facts": c['context_facts'],
                        "genome": {
                            "lambda_depth": c['individual'].genome.lambda_depth,
                            "lambda_breadth": c['individual'].genome.lambda_breadth,
                            "similarity_threshold": c['individual'].genome.similarity_threshold,
                        },
                    }
                    for c in rankings
                ],
            })
        store_paths = data.get('storePaths', [])
        if not store_paths:
            return jsonify({"error": "storePaths are required"}), 400
        concepts_df, facts_df, _ = load_kg_data(store_paths)
        facts = []
        for _, row in facts_df.iterrows():
            facts.append({"statement": row.get('statement', '')})
        concepts = []
        for _, row in concepts_df.iterrows():
            concepts.append({"name": row.get('name', '')})
        if not facts:
            return jsonify({"error": "Knowledge graph is empty. Ingest some data first."}), 400
        if mode == 'traversal':
            from npcpy.memory.kg_population import KGGenome, KGIndividual, SememolutionPopulation
            genome = KGGenome(
                lambda_depth=lambda_depth,
                lambda_breadth=lambda_breadth,
                similarity_threshold=similarity_threshold,
            )
            existing_kg = {
                'facts': facts,
                'concepts': concepts,
                'concept_links': [],
                'fact_to_concept_links': {},
                'fact_to_fact_links': [],
            }
            ind = KGIndividual(individual_id='ephemeral', genome=genome, kg_data=existing_kg)
            if not model:
                return jsonify({"error": "No model specified for knowledge graph search."}), 400
            mgr = SememolutionPopulation(model=model, provider=provider or "ollama", population_size=1, sample_size=1)
            relevant_facts = mgr.search_individual(ind, question)[:top_k]
            relevant_concepts = [c.get('name', '') for c in concepts[:20]]
            if not relevant_facts:
                relevant_facts = [f.get('statement', '') for f in facts[-top_k:]]
        else:
            q_words = set(question.lower().split())
            scored_facts = []
            for f in facts:
                stmt = (f.get('statement', '') or '').lower()
                score = sum(1 for w in q_words if w in stmt)
                if score > 0:
                    scored_facts.append((score, f))
            scored_facts.sort(key=lambda x: -x[0])
            relevant_facts = [f['statement'] for _, f in scored_facts[:top_k]]
            relevant_concepts = [c.get('name', '') for c in concepts[:20]]
            if not relevant_facts:
                relevant_facts = [f.get('statement', '') for f in facts[-top_k:]]
        kg_context = "Known facts:\n" + "\n".join(f"- {f}" for f in relevant_facts)
        if relevant_concepts:
            kg_context += "\n\nKey concepts: " + ", ".join(relevant_concepts)
        from npcpy.llm_funcs import get_llm_response
        prompt = f"""Based on the following knowledge graph data, answer the user's question.
Use only the provided facts to ground your response. If the facts don't contain enough information, say so.
{kg_context}
User question: {question}
Answer:"""
        response = get_llm_response(
            prompt,
            model=model,
            provider=provider,
        )
        answer = response.get('response', '') if isinstance(response, dict) else str(response)
        return jsonify({
            "answer": answer,
            "sources": relevant_facts[:5],
            "concepts": relevant_concepts[:10]
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route('/api/kg/rollback', methods=['POST'])
def rollback_kg():
    """Rollback selected YAML stores by clearing concepts and links (keeps memories)."""
    try:
        data = request.get_json() or {}
        store_paths = data.get('storePaths', [])
        if not store_paths:
            return jsonify({"error": "storePaths are required"}), 400
        cleared = 0
        for sp in store_paths:
            store = KnowledgeStore(sp)
            d = store.load()
            d['concepts'] = []
            d['links'] = []
            d['last_evolved_at'] = None
            store.save(d)
            cleared += 1
        return jsonify({
            "success": True,
            "cleared": cleared
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route('/api/kg/populations', methods=['GET'])
def list_kg_populations():
    try:
        engine = create_engine('sqlite:///' + app.config.get('DB_PATH'))
        from npcpy.memory.kg_population import list_populations
        return jsonify({"populations": list_populations(engine)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route('/api/kg/population', methods=['POST'])
def create_kg_population():
    """Create a fresh population of KGIndividuals with random genomes.
    Body:
      { "name": "my_pop", "population_size": 20, "model": "...", "provider": "...",
        "sample_size": 10, "seed_from_kg": true, "mutation_rate": 0.15, ... }
    """
    try:
        data = request.get_json() or {}
        name = (data.get('name') or '').strip() or f"pop_{int(time.time())}"
        population_id = (data.get('id') or name).replace(' ', '_')
        pop_size = int(data.get('population_size', 20))
        sample_size = int(data.get('sample_size', 10))
        model = data.get('model') or app.config.get('DEFAULT_MODEL')
        if not model:
            return jsonify({"error": "No model specified. Set model in request or DEFAULT_MODEL config."}), 400
        provider = data.get('provider') or app.config.get('DEFAULT_PROVIDER') or "ollama"
        seed_from_kg = bool(data.get('seed_from_kg', True))
        engine = create_engine('sqlite:///' + app.config.get('DB_PATH'))
        from npcpy.memory.kg_population import SememolutionPopulation, save_population, _ensure_population_schema
        _ensure_population_schema(engine)
        mgr = SememolutionPopulation(
            engine=engine, model=model, provider=provider,
            population_size=pop_size, sample_size=sample_size,
        )
        for k_src, k_dst in (('mutation_rate', 'mutation_rate'),
                             ('crossover_rate', 'crossover_rate'),
                             ('tournament_size', 'tournament_size'),
                             ('elitism_count', 'elitism_count')):
            if k_src in data:
                setattr(mgr.ga.config, k_dst, type(getattr(mgr.ga.config, k_dst))(data[k_src]))
        mgr.initialize()
        if seed_from_kg:
            import copy as _copy
            for ind in mgr.ga.population:
                ind.kg_data = _copy.deepcopy(existing)
        save_population(engine, population_id, name, mgr)
        return jsonify({"success": True, "id": population_id, "name": name, "stats": mgr.get_stats()})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route('/api/kg/population/<population_id>', methods=['GET'])
def get_kg_population(population_id):
    try:
        engine = create_engine('sqlite:///' + app.config.get('DB_PATH'))
        from npcpy.memory.kg_population import load_population, _genome_to_dict
        mgr = load_population(engine, population_id)
        if not mgr:
            return jsonify({"error": "not found"}), 404
        individuals = [
            {
                'individual_id': ind.individual_id,
                'fitness': ind.fitness,
                'wins': ind.wins,
                'total_queries': ind.total_queries,
                'facts': len(ind.kg_data.get('facts', [])),
                'concepts': len(ind.kg_data.get('concepts', [])),
                'generation': ind.kg_data.get('generation', 0),
                'genome': _genome_to_dict(ind.genome),
            }
            for ind in mgr.ga.population
        ]
        return jsonify({
            "id": population_id,
            "model": mgr.model,
            "provider": mgr.provider,
            "sample_size": mgr.sample_size,
            "stats": mgr.get_stats(),
            "individuals": individuals,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route('/api/kg/population/<population_id>', methods=['DELETE'])
def delete_kg_population(population_id):
    try:
        engine = create_engine('sqlite:///' + app.config.get('DB_PATH'))
        from npcpy.memory.kg_population import delete_population
        return jsonify({"success": delete_population(engine, population_id)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route('/api/kg/population/<population_id>/individual/<individual_id>', methods=['GET'])
def get_kg_individual(population_id, individual_id):
    try:
        engine = create_engine('sqlite:///' + app.config.get('DB_PATH'))
        from npcpy.memory.kg_population import load_population, _genome_to_dict
        mgr = load_population(engine, population_id)
        if not mgr:
            return jsonify({"error": "population not found"}), 404
        ind = next((i for i in mgr.ga.population if i.individual_id == individual_id), None)
        if not ind:
            return jsonify({"error": "individual not found"}), 404
        return jsonify({
            'individual_id': ind.individual_id,
            'fitness': ind.fitness,
            'wins': ind.wins,
            'total_queries': ind.total_queries,
            'genome': _genome_to_dict(ind.genome),
            'kg_data': ind.kg_data,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route('/api/kg/population/<population_id>/individual/<individual_id>/genome', methods=['PUT'])
def update_kg_individual_genome(population_id, individual_id):
    try:
        data = request.get_json() or {}
        engine = create_engine('sqlite:///' + app.config.get('DB_PATH'))
        from npcpy.memory.kg_population import update_individual_genome
        ok = update_individual_genome(engine, population_id, individual_id, data)
        if not ok:
            return jsonify({"error": "not found"}), 404
        return jsonify({"success": True})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route('/api/kg/population/<population_id>/evolve', methods=['POST'])
def evolve_kg_population(population_id):
    """Advance the population by one generation (select, crossover, mutate)."""
    try:
        engine = create_engine('sqlite:///' + app.config.get('DB_PATH'))
        from npcpy.memory.kg_population import load_population, save_population
        mgr = load_population(engine, population_id)
        if not mgr:
            return jsonify({"error": "not found"}), 404
        mgr.evolve_generation()
        save_population(engine, population_id, population_id, mgr)
        return jsonify({"success": True, "stats": mgr.get_stats()})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route("/api/knowledge/load", methods=["GET"])
def knowledge_load():
    """Load the .knowledge.yaml for the current directory or a list of directories."""
    try:
        from npcpy.memory.knowledge_store import KnowledgeStore
        dirs_param = request.args.get("dirs")
        if dirs_param:
            dirs = [d.strip() for d in dirs_param.split(",") if d.strip()]
            all_memories = []
            all_knowledge = []
            for d in dirs:
                store = KnowledgeStore(d)
                data = store.load()
                all_memories.extend(data.get("memories", []))
                all_knowledge.extend(data.get("knowledge", []))
            return jsonify({
                "memories": all_memories,
                "knowledge": all_knowledge,
            })
        current_path = request.args.get("currentPath", os.getcwd())
        store = KnowledgeStore(current_path)
        data = store.load()
        return jsonify({
            "memories": data.get("memories", []),
            "knowledge": data.get("knowledge", []),
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route("/api/knowledge/search", methods=["GET"])
def knowledge_search():
    """Search memories in the local .knowledge.yaml."""
    try:
        current_path = request.args.get("currentPath", os.getcwd())
        q = request.args.get("q", "")
        limit = int(request.args.get("limit", 20))
        from npcpy.memory.knowledge_store import KnowledgeStore
        store = KnowledgeStore(current_path)
        results = store.search_memories(q, limit=limit)
        return jsonify({"memories": results, "count": len(results)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route("/api/knowledge/memories", methods=["GET"])
def knowledge_memories():
    """Get memories from local .knowledge.yaml with optional status filter."""
    try:
        current_path = request.args.get("currentPath", os.getcwd())
        status = request.args.get("status")
        limit = request.args.get("limit")
        limit = int(limit) if limit else None
        from npcpy.memory.knowledge_store import KnowledgeStore
        store = KnowledgeStore(current_path)
        mems = store.get_memories(status=status, limit=limit)
        return jsonify({"memories": mems, "count": len(mems)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route("/api/knowledge/links", methods=["GET"])
def knowledge_links():
    """Get links for a memory or all links from local .knowledge.yaml."""
    try:
        current_path = request.args.get("currentPath", os.getcwd())
        mem_id = request.args.get("mem_id")
        from npcpy.memory.knowledge_store import KnowledgeStore
        store = KnowledgeStore(current_path)
        if mem_id:
            result = store.get_links_for_memory(mem_id)
            return jsonify(result)
        return jsonify({"links": store.get_all_links()})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route("/api/knowledge/link", methods=["POST"])
def knowledge_link_create():
    """Create a directed link in local .knowledge.yaml."""
    try:
        data = request.json or {}
        current_path = data.get("currentPath", os.getcwd())
        from_mem = data.get("from")
        to_mem = data.get("to")
        relation = data.get("relation", "related_to")
        agent = data.get("agent", "")
        if not from_mem or not to_mem:
            return jsonify({"error": "from and to are required"}), 400
        from npcpy.memory.knowledge_store import KnowledgeStore
        store = KnowledgeStore(current_path)
        link_id = store.append_link(from_mem, to_mem, relation, agent=agent)
        return jsonify({"success": True, "link_id": link_id})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route("/api/knowledge/context", methods=["GET"])
def knowledge_context():
    """Return formatted context string from local .knowledge.yaml."""
    try:
        current_path = request.args.get("currentPath", os.getcwd())
        max_memories = int(request.args.get("max_memories", 10))
        from npcpy.memory.knowledge_store import KnowledgeStore
        store = KnowledgeStore(current_path)
        ctx = store.build_context(max_memories=max_memories)
        return jsonify({"context": ctx})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route("/api/knowledge/all_memories", methods=["GET"])
def knowledge_all_memories():
    """Return aggregated memories from all registered knowledge stores."""
    try:
        limit = request.args.get("limit")
        limit = int(limit) if limit else None
        from npcpy.memory.knowledge_store import KnowledgeStore
        dirs = _get_registered_stores()
        all_memories = []
        for d in dirs:
            fp = os.path.join(d, ".knowledge.yaml")
            if not os.path.exists(fp):
                continue
            store = KnowledgeStore(d)
            data = store.load()
            for mem in data.get("memories", []):
                mem["_directory"] = d
                all_memories.append(mem)
        if limit:
            all_memories = all_memories[:limit]
        return jsonify({"memories": all_memories, "count": len(all_memories), "sources": len(dirs)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route("/api/knowledge/all_search", methods=["GET"])
def knowledge_all_search():
    """Search across all registered knowledge stores."""
    try:
        q = request.args.get("q", "").lower()
        limit = int(request.args.get("limit", 20))
        from npcpy.memory.knowledge_store import KnowledgeStore
        dirs = _get_registered_stores()
        results = []
        for d in dirs:
            fp = os.path.join(d, ".knowledge.yaml")
            if not os.path.exists(fp):
                continue
            store = KnowledgeStore(d)
            data = store.load()
            for mem in data.get("memories", []):
                txt = (mem.get("initial_memory", "") + " " + mem.get("final_memory", "")).lower()
                if q in txt:
                    mem["_directory"] = d
                    results.append(mem)
            if len(results) >= limit:
                break
        return jsonify({"memories": results[:limit], "count": len(results[:limit])})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route("/api/knowledge/memory/update", methods=["POST"])
def knowledge_memory_update():
    """Update a memory's status and/or final_memory in local .knowledge.yaml."""
    try:
        data = request.json or {}
        current_path = data.get("currentPath", os.getcwd())
        mem_id = data.get("id")
        status = data.get("status")
        final_memory = data.get("final_memory")
        if not mem_id or not status:
            return jsonify({"error": "id and status are required"}), 400
        from npcpy.memory.knowledge_store import KnowledgeStore
        store = KnowledgeStore(current_path)
        changed = store.update_memory(mem_id, status, final_memory)
        return jsonify({"success": changed})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route("/api/knowledge/memory/delete", methods=["POST"])
def knowledge_memory_delete():
    """Delete a memory from local .knowledge.yaml."""
    try:
        data = request.json or {}
        current_path = data.get("currentPath", os.getcwd())
        mem_id = data.get("id")
        if not mem_id:
            return jsonify({"error": "id is required"}), 400
        from npcpy.memory.knowledge_store import KnowledgeStore
        store = KnowledgeStore(current_path)
        data_yaml = store.load()
        original_len = len(data_yaml.get("memories", []))
        data_yaml["memories"] = [m for m in data_yaml.get("memories", []) if m.get("id") != mem_id]
        store.save(data_yaml)
        deleted = len(data_yaml["memories"]) < original_len
        return jsonify({"success": deleted})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route("/api/attachments/<message_id>", methods=["GET"])
def get_message_attachments(message_id):
    try:
        engine = get_db_connection()
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT id, attachment_name, attachment_type, attachment_size, file_path FROM message_attachments WHERE message_id = :mid"),
                {"mid": message_id}
            )
            attachments = [
                {
                    "id": row.id,
                    "name": row.attachment_name,
                    "type": row.attachment_type,
                    "size": row.attachment_size,
                    "path": row.file_path,
                }
                for row in result
            ]
        return jsonify({"attachments": attachments, "error": None})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/api/attachment/<attachment_id>", methods=["GET"])
def get_attachment(attachment_id):
    try:
        engine = get_db_connection()
        with engine.connect() as conn:
            result = conn.execute(
                text("SELECT attachment_data, attachment_name, attachment_type FROM message_attachments WHERE id = :aid"),
                {"aid": attachment_id}
            )
            row = result.fetchone()
        if row:
            data = row.attachment_data
            name = row.attachment_name
            type = row.attachment_type
            base64_data = base64.b64encode(data).decode("utf-8")
            return jsonify(
                {"data": base64_data, "name": name, "type": type, "error": None}
            )
        return jsonify({"error": "Attachment not found"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/api/capture_screenshot", methods=["GET"])
def capture():
    screenshot = capture_screenshot(full=True)
    if not screenshot:
        print("Screenshot capture failed")
        return None
    return jsonify({"screenshot": screenshot})
@app.route("/api/jinxes/available", methods=["GET"])
def get_available_jinxes():
    try:
        current_path = request.args.get('currentPath')
        jinx_names = set()
        dirs_to_scan = []
        if current_path:
            dirs_to_scan.append(os.path.join(current_path, 'agents', 'jinxes'))
            dirs_to_scan.append(os.path.join(current_path, 'npc_team', 'jinxes'))
        user_npc_dir = app.config.get('user_npc_directory')
        if user_npc_dir:
            dirs_to_scan.append(os.path.join(os.path.dirname(user_npc_dir), 'agents', 'jinxes'))
            dirs_to_scan.append(os.path.join(user_npc_dir, 'jinxes'))
        package_dir = app.config.get('PACKAGE_NPC_TEAM_DIR')
        if package_dir:
            dirs_to_scan.append(os.path.join(package_dir, 'jinxes'))
        for d in dirs_to_scan:
            for jinx in load_jinxes_from_directory(d):
                jinx_names.add(jinx.jinx_name)
        return jsonify({'jinxes': sorted(list(jinx_names)), 'error': None})
    except Exception as e:
        print(f"Error getting available jinxes: {str(e)}")
        traceback.print_exc()
        return jsonify({'jinxes': [], 'error': str(e)}), 500
@app.route("/api/jinx/execute", methods=["POST"])
def execute_jinx():
    """
    Execute a specific jinx with provided arguments.
    Returns the output as a JSON response.
    """
    data = request.json
    stream_id = _setup_stream(data)
    print(f"--- Jinx Execution Request for streamId: {stream_id} ---", file=sys.stderr)
    print(f"Request Data: {json.dumps(data, indent=2)}", file=sys.stderr)
    jinx_name = data.get("jinxName")
    jinx_args = data.get("jinxArgs", [])
    print(f"Jinx Name: {jinx_name}, Jinx Args: {jinx_args}", file=sys.stderr)
    conversation_id = data.get("conversationId")
    model = data.get("model")
    provider = data.get("provider")
    if not conversation_id:
        print("ERROR: conversationId is required for Jinx execution with persistent variables", file=sys.stderr)
        return jsonify({"error": "conversationId is required for Jinx execution with persistent variables"}), 400
    npc_name = data.get("npc")
    npc_source = data.get("npcSource", "global")
    current_path = data.get("currentPath")
    if not jinx_name:
        print("ERROR: jinxName is required", file=sys.stderr)
        return jsonify({"error": "jinxName is required"}), 400
    if current_path:
        load_project_env(current_path)
    jinx = None
    if npc_name:
        npc_object = load_npc_by_name_and_source(npc_name, npc_source, current_path)
        if not npc_object and npc_source == 'project':
            npc_object = load_npc_by_name_and_source(npc_name, 'global')
    else:
        npc_object = None
    if npc_object and hasattr(npc_object, 'jinxes_dict') and jinx_name in npc_object.jinxes_dict:
        jinx = npc_object.jinxes_dict[jinx_name]
        print(f"Found jinx in NPC's jinxes_dict", file=sys.stderr)
    if not jinx and current_path:
        project_jinxes_base = os.path.join(current_path, 'npc_team', 'jinxes')
        for j in load_jinxes_from_directory(project_jinxes_base):
            if j.jinx_name == jinx_name:
                jinx = j
                print(f"Found jinx in project jinxes", file=sys.stderr)
                break
    if not jinx:
        print(f"ERROR: Jinx '{jinx_name}' not found", file=sys.stderr)
        searched_paths = []
        if npc_object:
            searched_paths.append(f"NPC {npc_name} jinxes_dict")
        if current_path:
            searched_paths.append(f"Project jinxes at {os.path.join(current_path, 'npc_team', 'jinxes')}")
        print(f"Searched in: {', '.join(searched_paths)}", file=sys.stderr)
        return jsonify({"error": f"Jinx '{jinx_name}' not found"}), 404
    from npcpy.npc_compiler import extract_jinx_inputs
    fixed_args = []
    i = 0
    cleaned_jinx_args = [arg for arg in jinx_args if arg is not None]
    while i < len(cleaned_jinx_args):
        arg = cleaned_jinx_args[i]
        if arg.startswith('-'):
            fixed_args.append(arg)
            value_parts = []
            i += 1
            while i < len(cleaned_jinx_args) and not cleaned_jinx_args[i].startswith('-'):
                value_parts.append(cleaned_jinx_args[i])
                i += 1
            if value_parts:
                full_value = " ".join(value_parts)
                if full_value.startswith("'") and full_value.endswith("'"):
                    full_value = full_value[1:-1]
                elif full_value.startswith('"') and full_value.endswith('"'):
                    full_value = full_value[1:-1]
                fixed_args.append(full_value)
        else:
            fixed_args.append(arg)
            i += 1
    input_values = extract_jinx_inputs(fixed_args, jinx)
    print(f'Executing jinx with input_values: {input_values}', file=sys.stderr)
    if npc_object and hasattr(npc_object, 'jinxes_dict'):
        all_jinxes.update(npc_object.jinxes_dict)
    if conversation_id not in app.jinx_conversation_contexts:
        app.jinx_conversation_contexts[conversation_id] = {}
    jinx_local_context = app.jinx_conversation_contexts[conversation_id]
    print(f"--- CONTEXT STATE (conversationId: {conversation_id}) ---", file=sys.stderr)
    print(f"jinx_local_context BEFORE Jinx execution: {jinx_local_context}", file=sys.stderr)
    state = ServeState(
        npc=npc_object,
        team=None,
        conversation_id=conversation_id,
        chat_model=model,
        chat_provider=provider,
        current_path=current_path or os.getcwd(),
    )
    extra_globals_for_jinx = {
        **jinx_local_context,
        'state': state,
    }
    jinx_execution_result = jinx.execute(
        input_values=input_values,
        jinja_env=npc_object.jinja_env if npc_object else None,
        npc=npc_object,
        messages=messages,
        extra_globals=extra_globals_for_jinx
    )
    output_from_jinx_result = jinx_execution_result.get('output')
    final_output_string = str(output_from_jinx_result) if output_from_jinx_result is not None else ""
    if isinstance(jinx_execution_result, dict):
        for key, value in jinx_execution_result.items():
            jinx_local_context[key] = value
    print(f"jinx_local_context AFTER Jinx execution (final state): {jinx_local_context}", file=sys.stderr)
    print(f"Jinx execution result output: {output_from_jinx_result}", file=sys.stderr)
    user_command_log = f"/{jinx_name} {' '.join(cleaned_jinx_args)}"
    if is_html:
        return Response(final_output_string, mimetype="text/html")
    else:
        return Response(final_output_string, mimetype="text/plain")
@app.route("/api/models", methods=["GET"])
def get_models():
    """
    Return models configured via the `providers` field in team/NPC config.
    If `providers` is set, dynamically scan APIs (based on env vars) for
    the listed providers.  If not set, fall back to the explicit
    model/provider fields.
    """
    current_path = request.args.get("currentPath") or os.path.expanduser('~')
    registered_teams = _parse_registered_teams()
    seen = set()
    formatted_models = []
    _scan_cache: dict = {}
    def _add_model(m, p):
        if not m or (m, p) in seen:
            return
        seen.add((m, p))
        formatted_models.append({
            "value": m,
            "provider": p,
            "display_name": f"{m} | {p}",
        })
    def _resolve_providers(providers_list, scan_path):
        if not providers_list:
            return
        if scan_path not in _scan_cache:
            try:
                _scan_cache[scan_path] = get_locally_available_models(
                    scan_path, airplane_mode=False
                )
            except Exception as e:
                print(f"[models] Failed to scan local models for {scan_path}: {e}")
                _scan_cache[scan_path] = {}
        available = _scan_cache[scan_path]
        for entry in providers_list:
            if isinstance(entry, str):
                provider_name = entry
                if provider_name == "models":
                    continue
                for model_name, model_provider in available.items():
                    if model_provider == provider_name:
                        _add_model(model_name, provider_name)
            elif isinstance(entry, dict):
                provider_name = entry.get("provider_type") or entry.get("name")
                if not provider_name or provider_name == "models":
                    continue
                base_model = entry.get("model")
                if base_model:
                    _add_model(base_model, provider_name)
                model_list = entry.get("models")
                if isinstance(model_list, list):
                    for model_name in model_list:
                        _add_model(model_name, provider_name)
                if not base_model and not model_list:
                    for model_name, model_provider in available.items():
                        if model_provider == provider_name:
                            _add_model(model_name, provider_name)
    def _collect_team_models(team, scan_path):
        team_providers = getattr(team, 'providers', None)
        if isinstance(team_providers, list) and team_providers:
            _resolve_providers(team_providers, scan_path)
        for npc in team.npcs.values():
            npc_providers = getattr(npc, '_extra_fields', {}).get('providers')
            if isinstance(npc_providers, list) and npc_providers:
                _resolve_providers(npc_providers, scan_path)
        if team.model and team.provider:
            _add_model(team.model, team.provider)
        for npc in team.npcs.values():
            if npc.model and npc.provider:
                _add_model(npc.model, npc.provider)
    project_team_path = os.path.join(current_path, 'npc_team')
    if os.path.isdir(project_team_path):
        try:
            team = Team(team_path=project_team_path)
            _collect_team_models(team, current_path)
        except Exception as e:
            print(f"[models] Failed to load project team: {e}")
    for team_path in registered_teams:
        if not os.path.isdir(team_path):
            continue
        try:
            team = Team(team_path=team_path)
            _collect_team_models(team, team_path)
        except Exception as e:
            print(f"[models] Failed to load registered team {team_path}: {e}")
    print(f"[models] Returning {len(formatted_models)} team-configured models")
    return jsonify({"models": formatted_models, "error": None})
@app.route("/api/available_models", methods=["GET"])
def get_available_models():
    current_path = request.args.get("currentPath") or os.path.expanduser('~')
    try:
        available = get_locally_available_models(current_path, airplane_mode=False)
        models = []
        seen = set()
        for model_name, model_provider in available.items():
            key = (model_name, model_provider)
            if key in seen:
                continue
            seen.add(key)
            models.append({
                "value": model_name,
                "provider": model_provider,
                "display_name": f"{model_name} | {model_provider}",
            })
        return jsonify({"models": models, "error": None})
    except Exception as e:
        print(f"[available_models] Failed: {e}")
        return jsonify({"models": [], "error": str(e)})
@app.route('/api/<command>', methods=['POST'])
def api_command(command):
    data = request.json or {}
    handler = router.get_route(command)
    if not handler:
        return jsonify({"error": f"Unknown command: {command}"})
    if router.shell_only.get(command, False):
        return jsonify({"error": f"Command {command} is only available in shell mode"})
    try:
        args = data.get('args', [])
        kwargs = data.get('kwargs', {})
        command_str = command
        if args:
            command_str += " " + " ".join(str(arg) for arg in args)
        result = handler(command_str, **kwargs)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})
@app.route("/api/jinxes/save", methods=["POST"])
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
            user_npc_dir = app.config.get('user_npc_directory')
            if not user_npc_dir:
                return jsonify({"error": "user_npc_directory not configured"}), 500
            jinxes_dir = os.path.join(user_npc_dir, "jinxes")
        else:
            if not current_path.endswith("npc_team"):
                current_path = os.path.join(current_path, "npc_team")
            jinxes_dir = os.path.join(current_path, "jinxes")
        jinx = Jinx(jinx_data=jinx_data)
        jinx_rel_path = jinx_data.get("path", "")
        if jinx_rel_path and "/" in jinx_rel_path:
            save_dir = os.path.join(jinxes_dir, os.path.dirname(jinx_rel_path))
        else:
            save_dir = jinxes_dir
        jinx.save(save_dir)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/api/jinxes/delete", methods=["POST"])
def delete_jinx():
    """Delete a jinx file from the filesystem."""
    try:
        data = request.json or {}
        jinx_path = data.get("jinxPath", "")
        scope = data.get("scope", "global")
        current_path = data.get("currentPath", "")
        source_path = data.get("sourcePath", "")
        if source_path and os.path.exists(source_path):
            file_path = source_path
        elif jinx_path:
            if scope == "global":
                jinxes_dir = os.path.join(app.config.get('user_npc_directory') or os.path.expanduser('~/npc_team'), 'jinxes')
            else:
                base = current_path
                if not base.endswith("npc_team"):
                    base = os.path.join(base, "npc_team")
                jinxes_dir = os.path.join(base, "jinxes")
            file_path = os.path.join(jinxes_dir, f"{jinx_path}.jinx")
        else:
            return jsonify({"error": "jinxPath or sourcePath required"}), 400
        if not os.path.exists(file_path):
            return jsonify({"error": f"File not found: {file_path}"}), 404
        os.unlink(file_path)
        parent = os.path.dirname(file_path)
        while parent and parent != jinxes_dir if not source_path else False:
            try:
                if not os.listdir(parent):
                    os.rmdir(parent)
                    parent = os.path.dirname(parent)
                else:
                    break
            except OSError:
                break
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/api/jinxes/ingest", methods=["POST"])
def ingest_jinx_from_url():
    """
    Ingest a jinx or skill from a URL. Supports:
    - .jinx files (YAML) → saved directly
    - SKILL.md files → saved as skill directory
    - Raw text/markdown → wrapped as a skill with sections
    - GitHub URLs → auto-resolved to raw content
    """
    try:
        import requests as req_lib
        data = request.json
        url = data.get("url", "").strip()
        name = data.get("name", "").strip()
        scope = data.get("scope", "project")
        current_path = data.get("currentPath", "")
        skill_type = data.get("type", "auto")
        if not url:
            return jsonify({"error": "URL is required"}), 400
        if "github.com" in url and "/blob/" in url:
            url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        resp = req_lib.get(url, timeout=30)
        resp.raise_for_status()
        content = resp.text
        if scope == "global":
            jinxes_dir = os.path.join(app.config.get('user_npc_directory') or os.path.expanduser('~/npc_team'), 'jinxes')
        else:
            base = current_path if current_path else (app.config.get('user_npc_directory') or os.path.expanduser('~/npc_team'))
            if not base.endswith("npc_team"):
                base = os.path.join(base, "npc_team")
            jinxes_dir = os.path.join(base, "jinxes")
        os.makedirs(jinxes_dir, exist_ok=True)
        url_lower = url.lower()
        if skill_type == "auto":
            if url_lower.endswith(".jinx") or url_lower.endswith(".yaml") or url_lower.endswith(".yml"):
                skill_type = "jinx"
            elif "SKILL.md" in url or url_lower.endswith("skill.md"):
                skill_type = "skill"
            elif content.strip().startswith("---") and "jinx_name" in content[:500]:
                skill_type = "jinx"
            elif content.strip().startswith("---") and ("name:" in content[:500] or "description:" in content[:500]):
                skill_type = "skill"
            else:
                skill_type = "skill"
        if not name:
            path_parts = url.rstrip("/").split("/")
            raw_name = path_parts[-1] if path_parts else "imported_skill"
            for ext in [".jinx", ".yaml", ".yml", ".md"]:
                if raw_name.lower().endswith(ext):
                    raw_name = raw_name[: -len(ext)]
            name = raw_name.replace(" ", "_").replace("-", "_").lower()
        if skill_type == "jinx":
            file_path = os.path.join(jinxes_dir, f"{name}.jinx")
            with open(file_path, "w") as f:
                f.write(content)
            return jsonify({
                "status": "success",
                "type": "jinx",
                "name": name,
                "path": file_path,
                "message": f"Jinx '{name}' saved to {file_path}"
            })
        else:
            skill_dir = os.path.join(jinxes_dir, "skills", name)
            os.makedirs(skill_dir, exist_ok=True)
            skill_path = os.path.join(skill_dir, "SKILL.md")
            if content.strip().startswith("---"):
                with open(skill_path, "w") as f:
                    f.write(content)
            else:
                frontmatter = f"---\nname: {name}\ndescription: Skill ingested from {url}\n---\n"
                with open(skill_path, "w") as f:
                    f.write(frontmatter + "\n" + content)
            return jsonify({
                "status": "success",
                "type": "skill",
                "name": name,
                "path": skill_path,
                "message": f"Skill '{name}' saved to {skill_path}"
            })
    except req_lib.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to fetch URL: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/api/jinx/test", methods=["POST"])
def test_jinx():
    data = request.json
    jinx_data = data.get("jinx")
    test_inputs = data.get("inputs", {})
    current_path = data.get("currentPath")
    if current_path:
        load_project_env(current_path)
    jinx = Jinx(jinx_data=jinx_data)
    from jinja2.sandbox import SandboxedEnvironment
    temp_env = SandboxedEnvironment()
    jinx.render_first_pass(temp_env, {})
    conversation_id = f"jinx_test_{uuid.uuid4().hex[:8]}"
    try:
        result = jinx.execute(
            input_values=test_inputs,
            npc=None,
            messages=[],
            extra_globals={},
            jinja_env=temp_env
        )
        output = result.get('output', str(result))
        if result.get('error'):
            jinx_execution_status = "failed"
            jinx_error_message = str(result.get('error'))
    except Exception as e:
        jinx_execution_status = "failed"
        jinx_error_message = str(e)
        output = f"Jinx execution failed: {e}"
    return jsonify({
        "output": output,
        "conversation_id": conversation_id,
        "execution_id": user_message_id,
        "error": jinx_error_message
    })
from npcpy.ft.diff import train_diffusion, DiffusionConfig
import threading
from collections import defaultdict
finetune_jobs = {}
def extract_and_store_memories(
    conversation_text,
    conversation_id,
    npc_name,
    team_name,
    current_path,
    model,
    provider,
    npc_object=None
):
    from npcpy.llm_funcs import get_facts
    from npcpy.memory.knowledge_store import get_store_for_path
    memory_context = ""
    if current_path:
        try:
            store = get_store_for_path(current_path)
            memory_context = store.build_context(max_memories=10)
        except Exception:
            pass
    from npcpy.llm_funcs import resolve_model_provider
    resolved_model, resolved_provider, _, _ = resolve_model_provider(
        npc=npc_object,
        team=npc_object.team if npc_object else None,
        model=model,
        provider=provider,
    )
    from npcpy.llm_funcs import CONVERSATION_RULES
    facts = get_facts(
        conversation_text,
        model=resolved_model,
        provider=resolved_provider,
        npc=npc_object,
        context=memory_context,
        rules=CONVERSATION_RULES,
    )
    memories_for_approval = []
    from npcpy.memory.knowledge_store import get_store_for_path
    if facts and current_path:
        store = get_store_for_path(current_path)
        for i, fact in enumerate(facts):
            message_id = f"{conversation_id}_{datetime.datetime.now().strftime('%H%M%S')}_{i}"
            mem_id = store.append_memory(
                message_id=message_id,
                conversation_id=conversation_id,
                npc=npc_name or "default",
                team=team_name or "default",
                directory_path=current_path or "/",
                initial_memory=fact.get('statement', str(fact)),
                status="pending_approval",
                model=resolved_model,
                provider=resolved_provider,
                source_type="conversation",
                source_id=conversation_id,
            )
            memories_for_approval.append({
                "memory_id": mem_id,
                "content": fact.get('statement', str(fact)),
                "type": fact.get('type', 'unknown'),
                "context": fact.get('source_text', ''),
                "npc": npc_name or "default"
            })
    return memories_for_approval
@app.route('/api/finetuned_models', methods=['GET'])
def get_finetuned_models():
    current_path = request.args.get("currentPath")
    potential_root_paths = [
        get_models_dir(),
        get_images_dir(),
    ]
    if current_path:
        project_models_path = os.path.join(current_path, 'models')
        project_images_path = os.path.join(current_path, 'images')
        potential_root_paths.extend([project_models_path, project_images_path])
    finetuned_models = []
    print(f"🌋 Searching for fine-tuned models in potential root paths: {set(potential_root_paths)}")
    for root_path in set(potential_root_paths):
        if not os.path.exists(root_path) or not os.path.isdir(root_path):
            print(f"🌋 Skipping non-existent or non-directory root path: {root_path}")
            continue
        print(f"🌋 Scanning root path: {root_path}")
        for model_dir_name in os.listdir(root_path):
            full_model_path = os.path.join(root_path, model_dir_name)
            if not os.path.isdir(full_model_path):
                print(f"🌋 Skipping {full_model_path}: Not a directory.")
                continue
            has_model_final_pt = os.path.exists(os.path.join(full_model_path, 'model_final.pt'))
            has_checkpoints_dir = os.path.isdir(os.path.join(full_model_path, 'checkpoints'))
            if has_model_final_pt or has_checkpoints_dir:
                print(f"🌋 Identified fine-tuned model: {model_dir_name} at {full_model_path} (found model_final.pt or checkpoints dir)")
                finetuned_models.append({
                    "value": full_model_path,
                    "provider": "diffusers",
                    "display_name": f"{model_dir_name} | Fine-tuned Diffuser"
                })
                continue
            print(f"🌋 Skipping {full_model_path}: No model_final.pt or checkpoints directory found at root.")
    print(f"🌋 Finished scanning. Found {len(finetuned_models)} fine-tuned models.")
    return jsonify({"models": finetuned_models, "error": None})
@app.route('/api/finetune_diffusers', methods=['POST'])
def finetune_diffusers():
    data = request.json
    images = data.get('images', [])
    captions = data.get('captions', [])
    output_name = data.get('outputName', 'my_diffusion_model')
    num_epochs = data.get('epochs', 100)
    batch_size = data.get('batchSize', 4)
    learning_rate = data.get('learningRate', 1e-4)
    output_path = data.get('outputPath', get_models_dir())
    print(f"🌋 Finetune Diffusers Request Received!")
    print(f"  Images: {len(images)} files")
    print(f"  Output Name: {output_name}")
    print(f"  Epochs: {num_epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}")
    if not images:
        print("🌋 Error: No images provided for finetuning.")
        return jsonify({'error': 'No images provided'}), 400
    if not captions or len(captions) != len(images):
        print("🌋 Warning: Captions not provided or mismatching image count. Using empty captions.")
        captions = [''] * len(images)
    expanded_images = [os.path.expanduser(p) for p in images]
    output_dir = os.path.expanduser(
        os.path.join(output_path, output_name)
    )
    job_id = f"ft_{int(time.time())}"
    finetune_jobs[job_id] = {
        'status': 'running',
        'output_dir': output_dir,
        'epochs': num_epochs,
        'current_epoch': 0,
        'current_batch': 0,
        'total_batches': 0,
        'current_loss': None,
        'loss_history': [],
        'step': 0,
        'start_time': datetime.datetime.now().isoformat()
    }
    print(f"🌋 Finetuning job {job_id} initialized. Output directory: {output_dir}")
    def progress_callback(progress_data):
        """Callback to update job progress from training loop."""
        finetune_jobs[job_id]['current_epoch'] = progress_data.get('epoch', 0)
        finetune_jobs[job_id]['epochs'] = progress_data.get('total_epochs', num_epochs)
        finetune_jobs[job_id]['current_batch'] = progress_data.get('batch', 0)
        finetune_jobs[job_id]['total_batches'] = progress_data.get('total_batches', 0)
        finetune_jobs[job_id]['step'] = progress_data.get('step', 0)
        finetune_jobs[job_id]['current_loss'] = progress_data.get('loss')
        if progress_data.get('loss_history'):
            finetune_jobs[job_id]['loss_history'] = progress_data['loss_history']
    def run_training_async():
        print(f"🌋 Finetuning job {job_id}: Starting asynchronous training thread...")
        try:
            config = DiffusionConfig(
                num_epochs=num_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                output_model_path=output_dir
            )
            print(f"🌋 Finetuning job {job_id}: Calling train_diffusion with config: {config}")
            model_path = train_diffusion(
                expanded_images,
                captions,
                config=config,
                progress_callback=progress_callback
            )
            finetune_jobs[job_id]['status'] = 'complete'
            finetune_jobs[job_id]['model_path'] = model_path
            finetune_jobs[job_id]['end_time'] = datetime.datetime.now().isoformat()
            print(f"🌋 Finetuning job {job_id}: Training complete! Model saved to: {model_path}")
        except Exception as e:
            finetune_jobs[job_id]['status'] = 'error'
            finetune_jobs[job_id]['error_msg'] = str(e)
            finetune_jobs[job_id]['end_time'] = datetime.datetime.now().isoformat()
            print(f"🌋 Finetuning job {job_id}: ERROR during training: {e}")
            traceback.print_exc()
        print(f"🌋 Finetuning job {job_id}: Asynchronous training thread finished.")
    thread = threading.Thread(target=run_training_async)
    thread.daemon = True
    thread.start()
    print(f"🌋 Finetuning job {job_id} successfully launched in background. Returning initial status.")
    return jsonify({
        'status': 'started',
        'jobId': job_id,
        'message': f"Finetuning job '{job_id}' started. Check /api/finetune_status/{job_id} for updates."
    })
@app.route('/api/finetune_status/<job_id>', methods=['GET'])
def finetune_status(job_id):
    if job_id not in finetune_jobs:
        return jsonify({'error': 'Job not found'}), 404
    job = finetune_jobs[job_id]
    if job['status'] == 'complete':
        return jsonify({
            'status': 'complete',
            'complete': True,
            'outputPath': job.get('model_path', job['output_dir']),
            'loss_history': job.get('loss_history', [])
        })
    elif job['status'] == 'error':
        return jsonify({
            'status': 'error',
            'error': job.get('error_msg', 'Unknown error')
        })
    return jsonify({
        'status': 'running',
        'epoch': job.get('current_epoch', 0),
        'total_epochs': job.get('epochs', 0),
        'batch': job.get('current_batch', 0),
        'total_batches': job.get('total_batches', 0),
        'step': job.get('step', 0),
        'loss': job.get('current_loss'),
        'loss_history': job.get('loss_history', []),
        'start_time': job.get('start_time')
    })
instruction_finetune_jobs = {}
@app.route('/api/finetune_instruction', methods=['POST'])
def finetune_instruction():
    """
    Fine-tune an LLM on instruction/conversation data.
    Request body:
    {
        "trainingData": [
            {"input": "user prompt", "output": "assistant response"},
            // For DPO: include "reward" or "quality" score (0-1)
            // For memory_classifier: include "status" as "approved"/"rejected"
            ...
        ],
        "outputName": "my_instruction_model",
        "baseModel": "google/gemma-3-270m-it",
        "strategy": "sft",  // "sft", "usft", "dpo", or "memory_classifier"
        "epochs": 20,
        "learningRate": 3e-5,
        "batchSize": 2,
        "loraR": 8,
        "loraAlpha": 16,
        "outputPath": "<data_dir>/npc_team/models",
        "systemPrompt": "optional system prompt to prepend",
        "npc": "optional npc name",
        "formatStyle": "gemma"  // "gemma", "llama", or "default"
    }
    Strategies:
    - sft: Supervised Fine-Tuning with input/output pairs
    - usft: Unsupervised Fine-Tuning on raw text (domain adaptation)
    - dpo: Direct Preference Optimization using quality/reward scores
    - memory_classifier: Train memory approval classifier
    """
    from npcpy.ft.sft import run_sft, SFTConfig
    from npcpy.ft.usft import run_usft, USFTConfig
    from npcpy.ft.rl import train_with_dpo, RLConfig
    data = request.json
    training_data = data.get('trainingData', [])
    output_name = data.get('outputName', 'my_instruction_model')
    base_model = data.get('baseModel', 'google/gemma-3-270m-it')
    strategy = data.get('strategy', 'sft')
    num_epochs = data.get('epochs', 20)
    learning_rate = data.get('learningRate', 3e-5)
    batch_size = data.get('batchSize', 2)
    lora_r = data.get('loraR', 8)
    lora_alpha = data.get('loraAlpha', 16)
    output_path = data.get('outputPath', get_models_dir())
    system_prompt = data.get('systemPrompt', '')
    format_style = data.get('formatStyle', 'gemma')
    npc_name = data.get('npc', None)
    print(f"🎓 Instruction Fine-tune Request Received!")
    print(f"  Training examples: {len(training_data)}")
    print(f"  Strategy: {strategy}")
    print(f"  Base model: {base_model}")
    print(f"  Output name: {output_name}")
    print(f"  Epochs: {num_epochs}, LR: {learning_rate}, Batch: {batch_size}")
    if not training_data:
        print("🎓 Error: No training data provided.")
        return jsonify({'error': 'No training data provided'}), 400
    min_examples = 10 if strategy == 'memory_classifier' else 3
    if len(training_data) < min_examples:
        print(f"🎓 Error: Need at least {min_examples} training examples for {strategy}.")
        return jsonify({'error': f'Need at least {min_examples} training examples for {strategy}'}), 400
    expanded_output_dir = os.path.expanduser(os.path.join(output_path, output_name))
    job_id = f"ift_{int(time.time())}"
    instruction_finetune_jobs[job_id] = {
        'status': 'running',
        'strategy': strategy,
        'output_dir': expanded_output_dir,
        'base_model': base_model,
        'epochs': num_epochs,
        'current_epoch': 0,
        'current_step': 0,
        'total_steps': 0,
        'current_loss': None,
        'loss_history': [],
        'start_time': datetime.datetime.now().isoformat(),
        'npc': npc_name,
        'num_examples': len(training_data)
    }
    print(f"🎓 Instruction fine-tuning job {job_id} initialized. Output: {expanded_output_dir}")
    def run_training_async():
        print(f"🎓 Job {job_id}: Starting {strategy.upper()} training thread...")
        try:
            if strategy == 'sft':
                X = []
                y = []
                for example in training_data:
                    inp = example.get('input', example.get('prompt', ''))
                    out = example.get('output', example.get('response', example.get('completion', '')))
                    if system_prompt:
                        inp = f"{system_prompt}\n\n{inp}"
                    X.append(inp)
                    y.append(out)
                config = SFTConfig(
                    base_model_name=base_model,
                    output_model_path=expanded_output_dir,
                    num_train_epochs=num_epochs,
                    learning_rate=learning_rate,
                    per_device_train_batch_size=batch_size,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha
                )
                print(f"🎓 Job {job_id}: Running SFT with config: {config}")
                model_path = run_sft(
                    X=X,
                    y=y,
                    config=config,
                    format_style=format_style
                )
                instruction_finetune_jobs[job_id]['status'] = 'complete'
                instruction_finetune_jobs[job_id]['model_path'] = model_path
                instruction_finetune_jobs[job_id]['end_time'] = datetime.datetime.now().isoformat()
                print(f"🎓 Job {job_id}: SFT complete! Model saved to: {model_path}")
            elif strategy == 'usft':
                texts = []
                for example in training_data:
                    if 'text' in example:
                        texts.append(example['text'])
                    else:
                        inp = example.get('input', example.get('prompt', ''))
                        out = example.get('output', example.get('response', ''))
                        if inp and out:
                            texts.append(f"{inp}\n{out}")
                        elif inp:
                            texts.append(inp)
                        elif out:
                            texts.append(out)
                config = USFTConfig(
                    base_model_name=base_model,
                    output_model_path=expanded_output_dir,
                    num_train_epochs=num_epochs,
                    learning_rate=learning_rate,
                    per_device_train_batch_size=batch_size,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha
                )
                print(f"🎓 Job {job_id}: Running USFT with {len(texts)} texts")
                model_path = run_usft(texts=texts, config=config)
                instruction_finetune_jobs[job_id]['status'] = 'complete'
                instruction_finetune_jobs[job_id]['model_path'] = model_path
                instruction_finetune_jobs[job_id]['end_time'] = datetime.datetime.now().isoformat()
                print(f"🎓 Job {job_id}: USFT complete! Model saved to: {model_path}")
            elif strategy == 'dpo':
                traces = []
                for example in training_data:
                    traces.append({
                        'task_prompt': example.get('input', example.get('prompt', '')),
                        'final_output': example.get('output', example.get('response', '')),
                        'reward': example.get('reward', example.get('quality', 0.5))
                    })
                config = RLConfig(
                    base_model_name=base_model,
                    adapter_path=expanded_output_dir,
                    num_train_epochs=num_epochs,
                    learning_rate=learning_rate,
                    per_device_train_batch_size=batch_size,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha
                )
                print(f"🎓 Job {job_id}: Running DPO with {len(traces)} traces")
                adapter_path = train_with_dpo(traces, config)
                if adapter_path:
                    instruction_finetune_jobs[job_id]['status'] = 'complete'
                    instruction_finetune_jobs[job_id]['model_path'] = adapter_path
                else:
                    instruction_finetune_jobs[job_id]['status'] = 'error'
                    instruction_finetune_jobs[job_id]['error_msg'] = 'Not enough valid preference pairs for DPO training'
                instruction_finetune_jobs[job_id]['end_time'] = datetime.datetime.now().isoformat()
                print(f"🎓 Job {job_id}: DPO complete! Adapter saved to: {adapter_path}")
            elif strategy == 'memory_classifier':
                from npcpy.ft.memory_trainer import MemoryTrainer
                approved_memories = []
                rejected_memories = []
                for example in training_data:
                    status = example.get('status', 'approved')
                    memory_data = {
                        'initial_memory': example.get('input', example.get('memory', '')),
                        'final_memory': example.get('output', example.get('final_memory', '')),
                        'context': example.get('context', '')
                    }
                    if status in ['approved', 'model-approved']:
                        approved_memories.append(memory_data)
                    else:
                        rejected_memories.append(memory_data)
                if len(approved_memories) < 10 or len(rejected_memories) < 10:
                    instruction_finetune_jobs[job_id]['status'] = 'error'
                    instruction_finetune_jobs[job_id]['error_msg'] = 'Need at least 10 approved and 10 rejected memories'
                    instruction_finetune_jobs[job_id]['end_time'] = datetime.datetime.now().isoformat()
                    return
                trainer = MemoryTrainer(model_name=base_model)
                success = trainer.train(
                    approved_memories=approved_memories,
                    rejected_memories=rejected_memories,
                    output_dir=expanded_output_dir,
                    epochs=num_epochs
                )
                if success:
                    instruction_finetune_jobs[job_id]['status'] = 'complete'
                    instruction_finetune_jobs[job_id]['model_path'] = expanded_output_dir
                else:
                    instruction_finetune_jobs[job_id]['status'] = 'error'
                    instruction_finetune_jobs[job_id]['error_msg'] = 'Memory classifier training failed'
                instruction_finetune_jobs[job_id]['end_time'] = datetime.datetime.now().isoformat()
                print(f"🎓 Job {job_id}: Memory classifier complete!")
            else:
                raise ValueError(f"Unknown strategy: {strategy}. Supported: sft, usft, dpo, memory_classifier")
        except Exception as e:
            instruction_finetune_jobs[job_id]['status'] = 'error'
            instruction_finetune_jobs[job_id]['error_msg'] = str(e)
            instruction_finetune_jobs[job_id]['end_time'] = datetime.datetime.now().isoformat()
            print(f"🎓 Job {job_id}: ERROR during training: {e}")
            traceback.print_exc()
        print(f"🎓 Job {job_id}: Training thread finished.")
    thread = threading.Thread(target=run_training_async)
    thread.daemon = True
    thread.start()
    print(f"🎓 Job {job_id} launched in background.")
    return jsonify({
        'status': 'started',
        'jobId': job_id,
        'strategy': strategy,
        'message': f"Instruction fine-tuning job '{job_id}' started. Check /api/finetune_instruction_status/{job_id} for updates."
    })
@app.route('/api/finetune_instruction_status/<job_id>', methods=['GET'])
def finetune_instruction_status(job_id):
    """Get the status of an instruction fine-tuning job."""
    if job_id not in instruction_finetune_jobs:
        return jsonify({'error': 'Job not found'}), 404
    job = instruction_finetune_jobs[job_id]
    if job['status'] == 'complete':
        return jsonify({
            'status': 'complete',
            'complete': True,
            'outputPath': job.get('model_path', job['output_dir']),
            'strategy': job.get('strategy'),
            'loss_history': job.get('loss_history', []),
            'start_time': job.get('start_time'),
            'end_time': job.get('end_time')
        })
    elif job['status'] == 'error':
        return jsonify({
            'status': 'error',
            'error': job.get('error_msg', 'Unknown error'),
            'start_time': job.get('start_time'),
            'end_time': job.get('end_time')
        })
    return jsonify({
        'status': 'running',
        'strategy': job.get('strategy'),
        'epoch': job.get('current_epoch', 0),
        'total_epochs': job.get('epochs', 0),
        'step': job.get('current_step', 0),
        'total_steps': job.get('total_steps', 0),
        'loss': job.get('current_loss'),
        'loss_history': job.get('loss_history', []),
        'start_time': job.get('start_time'),
        'num_examples': job.get('num_examples', 0)
    })
@app.route('/api/instruction_models', methods=['GET'])
def get_instruction_models():
    """Get list of available instruction-tuned models."""
    current_path = request.args.get("currentPath")
    potential_root_paths = [
        get_models_dir(),
    ]
    if current_path:
        project_models_path = os.path.join(current_path, 'models')
        potential_root_paths.append(project_models_path)
    instruction_models = []
    print(f"🎓 Searching for instruction models in: {set(potential_root_paths)}")
    for root_path in set(potential_root_paths):
        if not os.path.exists(root_path) or not os.path.isdir(root_path):
            continue
        for model_dir_name in os.listdir(root_path):
            full_model_path = os.path.join(root_path, model_dir_name)
            if not os.path.isdir(full_model_path):
                continue
            has_adapter_config = os.path.exists(os.path.join(full_model_path, 'adapter_config.json'))
            has_config = os.path.exists(os.path.join(full_model_path, 'config.json'))
            has_tokenizer = os.path.exists(os.path.join(full_model_path, 'tokenizer_config.json'))
            if has_adapter_config or (has_config and has_tokenizer):
                model_type = 'lora_adapter' if has_adapter_config else 'full_model'
                print(f"🎓 Found instruction model: {model_dir_name} ({model_type})")
                instruction_models.append({
                    "value": full_model_path,
                    "name": model_dir_name,
                    "type": model_type,
                    "display_name": f"{model_dir_name} | Instruction Model"
                })
    print(f"🎓 Found {len(instruction_models)} instruction models.")
    return jsonify({"models": instruction_models, "error": None})
ge_jobs = {}
ge_populations = {}
@app.route('/api/genetic/create_population', methods=['POST'])
def create_genetic_population():
    """
    Create a new genetic evolution population.
    Request body:
    {
        "populationId": "optional_id",
        "populationType": "prompt" | "npc_config" | "model_ensemble" | "custom",
        "populationSize": 20,
        "config": {
            "mutationRate": 0.15,
            "crossoverRate": 0.7,
            "tournamentSize": 3,
            "elitismCount": 2
        },
        "initialPopulation": [...],  // Optional initial individuals
        "fitnessEndpoint": "/api/evaluate_fitness"  // Optional custom fitness endpoint
    }
    """
    from npcpy.ft.ge import GeneticEvolver, GAConfig
    data = request.json
    population_id = data.get('populationId', f"pop_{int(time.time())}")
    population_type = data.get('populationType', 'prompt')
    population_size = data.get('populationSize', 20)
    config_data = data.get('config', {})
    initial_population = data.get('initialPopulation', [])
    npc_name = data.get('npc', None)
    config = GAConfig(
        population_size=population_size,
        mutation_rate=config_data.get('mutationRate', 0.15),
        crossover_rate=config_data.get('crossoverRate', 0.7),
        tournament_size=config_data.get('tournamentSize', 3),
        elitism_count=config_data.get('elitismCount', 2),
        generations=config_data.get('generations', 50)
    )
    print(f"🧬 Creating genetic population {population_id} (type: {population_type})")
    if population_type == 'prompt':
        import random
        def initialize_fn():
            if initial_population:
                return random.choice(initial_population)
            return f"You are a helpful assistant. {random.choice(['Be concise.', 'Be detailed.', 'Be creative.', 'Be precise.'])}"
        def mutate_fn(individual):
            mutations = [
                lambda s: s + " Think step by step.",
                lambda s: s + " Be specific.",
                lambda s: s.replace("helpful", "expert"),
                lambda s: s.replace("assistant", "specialist"),
                lambda s: s + " Provide examples.",
            ]
            return random.choice(mutations)(individual)
        def crossover_fn(p1, p2):
            words1 = p1.split()
            words2 = p2.split()
            mid = len(words1) // 2
            return ' '.join(words1[:mid] + words2[mid:])
        def fitness_fn(individual):
            return len(individual) / 100.0
    elif population_type == 'npc_config':
        import random
        def initialize_fn():
            if initial_population:
                return random.choice(initial_population)
            return {
                'temperature': random.uniform(0.1, 1.0),
                'top_p': random.uniform(0.7, 1.0),
                'system_prompt_modifier': random.choice(['detailed', 'concise', 'creative']),
            }
        def mutate_fn(individual):
            mutated = individual.copy()
            key = random.choice(list(mutated.keys()))
            if key == 'temperature':
                mutated[key] = max(0.1, min(2.0, mutated[key] + random.gauss(0, 0.1)))
            elif key == 'top_p':
                mutated[key] = max(0.5, min(1.0, mutated[key] + random.gauss(0, 0.05)))
            return mutated
        def crossover_fn(p1, p2):
            child = {}
            for key in p1:
                child[key] = random.choice([p1.get(key), p2.get(key)])
            return child
        def fitness_fn(individual):
            return 0.5
    else:
        import random
        def initialize_fn():
            if initial_population:
                return random.choice(initial_population)
            return {"value": random.random()}
        def mutate_fn(individual):
            if isinstance(individual, dict):
                mutated = individual.copy()
                mutated['value'] = individual.get('value', 0) + random.gauss(0, 0.1)
                return mutated
            return individual
        def crossover_fn(p1, p2):
            if isinstance(p1, dict) and isinstance(p2, dict):
                return {'value': (p1.get('value', 0) + p2.get('value', 0)) / 2}
            return p1
        def fitness_fn(individual):
            if isinstance(individual, dict):
                return 1.0 - abs(individual.get('value', 0) - 0.5)
            return 0.5
    evolver = GeneticEvolver(
        fitness_fn=fitness_fn,
        mutate_fn=mutate_fn,
        crossover_fn=crossover_fn,
        initialize_fn=initialize_fn,
        config=config
    )
    evolver.initialize_population()
    ge_populations[population_id] = {
        'evolver': evolver,
        'type': population_type,
        'config': config,
        'generation': 0,
        'history': [],
        'npc': npc_name,
        'created_at': datetime.datetime.now().isoformat()
    }
    return jsonify({
        'populationId': population_id,
        'populationType': population_type,
        'populationSize': population_size,
        'generation': 0,
        'message': f"Population '{population_id}' created with {population_size} individuals"
    })
@app.route('/api/genetic/evolve', methods=['POST'])
def evolve_population():
    """
    Run evolution for N generations.
    Request body:
    {
        "populationId": "pop_123",
        "generations": 10,
        "fitnessScores": [...]  // Optional: external fitness scores for current population
    }
    """
    data = request.json
    population_id = data.get('populationId')
    generations = data.get('generations', 1)
    fitness_scores = data.get('fitnessScores', None)
    if population_id not in ge_populations:
        return jsonify({'error': f"Population '{population_id}' not found"}), 404
    pop_data = ge_populations[population_id]
    evolver = pop_data['evolver']
    print(f"🧬 Evolving population {population_id} for {generations} generations")
    if fitness_scores and len(fitness_scores) == len(evolver.population):
        original_fitness = evolver.fitness_fn
        score_iter = iter(fitness_scores)
        evolver.fitness_fn = lambda x: next(score_iter, 0.5)
    results = []
    for gen in range(generations):
        gen_stats = evolver.evolve_generation()
        pop_data['generation'] += 1
        pop_data['history'].append(gen_stats)
        results.append({
            'generation': pop_data['generation'],
            'bestFitness': gen_stats['best_fitness'],
            'avgFitness': gen_stats['avg_fitness'],
            'bestIndividual': gen_stats['best_individual']
        })
    if fitness_scores:
        evolver.fitness_fn = original_fitness
    return jsonify({
        'populationId': population_id,
        'generationsRun': generations,
        'currentGeneration': pop_data['generation'],
        'results': results,
        'bestIndividual': results[-1]['bestIndividual'] if results else None,
        'population': evolver.population[:5]
    })
@app.route('/api/genetic/population/<population_id>', methods=['GET'])
def get_population(population_id):
    """Get current state of a population."""
    if population_id not in ge_populations:
        return jsonify({'error': f"Population '{population_id}' not found"}), 404
    pop_data = ge_populations[population_id]
    evolver = pop_data['evolver']
    return jsonify({
        'populationId': population_id,
        'type': pop_data['type'],
        'generation': pop_data['generation'],
        'populationSize': len(evolver.population),
        'population': evolver.population,
        'history': pop_data['history'][-50:],
        'createdAt': pop_data['created_at'],
        'npc': pop_data.get('npc')
    })
@app.route('/api/genetic/populations', methods=['GET'])
def list_populations():
    """List all active populations."""
    populations = []
    for pop_id, pop_data in ge_populations.items():
        populations.append({
            'populationId': pop_id,
            'type': pop_data['type'],
            'generation': pop_data['generation'],
            'populationSize': len(pop_data['evolver'].population),
            'createdAt': pop_data['created_at'],
            'npc': pop_data.get('npc')
        })
    return jsonify({'populations': populations})
@app.route('/api/genetic/population/<population_id>', methods=['DELETE'])
def delete_population(population_id):
    """Delete a population."""
    if population_id not in ge_populations:
        return jsonify({'error': f"Population '{population_id}' not found"}), 404
    del ge_populations[population_id]
    print(f"🧬 Deleted population {population_id}")
    return jsonify({'message': f"Population '{population_id}' deleted"})
@app.route('/api/genetic/inject', methods=['POST'])
def inject_individuals():
    """
    Inject new individuals into a population.
    Request body:
    {
        "populationId": "pop_123",
        "individuals": [...],
        "replaceWorst": true  // Replace worst individuals or append
    }
    """
    data = request.json
    population_id = data.get('populationId')
    individuals = data.get('individuals', [])
    replace_worst = data.get('replaceWorst', True)
    if population_id not in ge_populations:
        return jsonify({'error': f"Population '{population_id}' not found"}), 404
    pop_data = ge_populations[population_id]
    evolver = pop_data['evolver']
    if replace_worst:
        fitness_scores = evolver.evaluate_population()
        sorted_pop = sorted(zip(evolver.population, fitness_scores), key=lambda x: x[1], reverse=True)
        keep_count = len(sorted_pop) - len(individuals)
        evolver.population = [ind for ind, _ in sorted_pop[:keep_count]] + individuals
    else:
        evolver.population.extend(individuals)
    print(f"🧬 Injected {len(individuals)} individuals into {population_id}")
    return jsonify({
        'populationId': population_id,
        'injectedCount': len(individuals),
        'newPopulationSize': len(evolver.population)
    })
@app.route("/api/ml/train", methods=["POST"])
def train_ml_model():
    import joblib
    import numpy as np
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
    data = request.json
    model_name = data.get("name")
    model_type = data.get("type")
    target = data.get("target")
    features = data.get("features")
    training_data = data.get("data")
    hyperparams = data.get("hyperparameters", {})
    df = pd.DataFrame(training_data)
    X = df[features].values
    metrics = {}
    model = None
    if model_type == "linear_regression":
        y = df[target].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = {
            "r2_score": r2_score(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred))
        }
    elif model_type == "logistic_regression":
        y = df[target].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = {"accuracy": accuracy_score(y_test, y_pred)}
    elif model_type == "random_forest":
        y = df[target].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = {
            "r2_score": r2_score(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred))
        }
    elif model_type == "clustering":
        n_clusters = hyperparams.get("n_clusters", 3)
        model = KMeans(n_clusters=n_clusters)
        labels = model.fit_predict(X)
        metrics = {"inertia": model.inertia_, "n_clusters": n_clusters}
    elif model_type == "gradient_boost":
        y = df[target].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        model = GradientBoostingRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = {
            "r2_score": r2_score(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred))
        }
    model_id = f"{model_name}_{int(time.time())}"
    model_path = os.path.join(get_models_dir(), f"{model_id}.joblib")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump({
        "model": model,
        "features": features,
        "target": target,
        "type": model_type
    }, model_path)
    return jsonify({
        "model_id": model_id,
        "metrics": metrics,
        "error": None
    })
@app.route("/api/ml/predict", methods=["POST"])
def ml_predict():
    import joblib
    data = request.json
    model_name = data.get("model_name")
    input_data = data.get("input_data")
    model_dir = get_models_dir()
    model_files = [f for f in os.listdir(model_dir) if f.startswith(model_name)]
    if not model_files:
        return jsonify({"error": f"Model {model_name} not found"})
    model_path = os.path.join(model_dir, model_files[0])
    model_data = joblib.load(model_path)
    model = model_data["model"]
    prediction = model.predict([input_data])
    return jsonify({
        "prediction": prediction.tolist(),
        "error": None
    })
@app.route("/api/jinx/executions/label", methods=["POST"])
def label_jinx_execution():
    data = request.json
    execution_id = data.get("executionId")
    label = data.get("label")
    try:
        engine = get_db_connection()
        with engine.begin() as conn:
            conn.execute(
                text("INSERT INTO labels (entity_type, entity_id, label) VALUES (:et, :eid, :lbl)"),
                {"et": "message", "eid": execution_id, "lbl": label}
            )
        return jsonify({"success": True, "error": None})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/api/npc/executions", methods=["GET"])
def get_npc_executions():
    npc_name = request.args.get("npcName")
    try:
        engine = get_db_connection()
        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS npc_executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    npc TEXT,
                    tool_name TEXT,
                    parameters TEXT,
                    result TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """))
            if npc_name:
                result = conn.execute(
                    text("SELECT * FROM npc_executions WHERE npc = :npc ORDER BY timestamp DESC LIMIT 1000"),
                    {"npc": npc_name}
                )
            else:
                result = conn.execute(
                    text("SELECT * FROM npc_executions ORDER BY timestamp DESC LIMIT 1000")
                )
            executions = [dict(row._mapping) for row in result]
        return jsonify({"executions": executions, "error": None})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/api/save_npc", methods=["POST"])
def save_npc():
    try:
        data = request.json
        npc_data = data.get("npc")
        source_path = data.get("sourcePath")
        if not npc_data or "name" not in npc_data:
            return jsonify({"error": "Invalid NPC data"}), 400
        npc_directory = os.path.dirname(source_path) if source_path else None
        if not npc_directory:
            return jsonify({"error": "sourcePath required"}), 400
        existing_npc_path = os.path.join(npc_directory, f"{npc_data['name']}.npc")
        existing_model = npc_data.get("model", "")
        existing_provider = npc_data.get("provider", "")
        if os.path.exists(existing_npc_path):
            try:
                existing_data = load_yaml_file(existing_npc_path)
                if existing_model in (None, "", "null") and existing_data.get("model"):
                    existing_model = existing_data["model"]
                if existing_provider in (None, "", "null") and existing_data.get("provider"):
                    existing_provider = existing_data["provider"]
            except Exception:
                pass
        known_keys = {"name", "primary_directive", "model", "provider", "api_url", "jinxes"}
        extra = {k: v for k, v in npc_data.items() if k not in known_keys}
        npc = NPC(
            name=npc_data["name"],
            primary_directive=npc_data.get("primary_directive", ""),
            model=existing_model,
            provider=existing_provider,
            api_url=npc_data.get("api_url", ""),
            jinxes=npc_data.get("jinxes"),
            **extra,
        )
        npc.save(npc_directory)
        return jsonify({"message": "NPC saved successfully", "error": None})
    except Exception as e:
        print(f"Error saving NPC: {str(e)}")
        return jsonify({"error": str(e)}), 500
@app.route("/api/jinxes/project", methods=["GET"])
def get_jinxes_project():
    project_dir = request.args.get("currentPath")
    if not project_dir:
        return jsonify({"jinxes": [], "error": "currentPath required"}), 400
    if not project_dir.endswith("jinxes"):
        project_dir = os.path.join(project_dir, "jinxes")
    if not os.path.exists(project_dir):
        return jsonify({"jinxes": [], "error": None})
    return jsonify({"jinxes": _serialize_jinxes_from_dir(project_dir), "error": None})
@app.route("/api/npcsql/run_model", methods=["POST"])
def run_npcsql_model():
    """Execute a single SQL model using ModelCompiler"""
    try:
        from npcpy.sql.npcsql import ModelCompiler
        data = request.json
        models_dir = data.get("modelsDir")
        model_name = data.get("modelName")
        npc_directory = data.get("npcDirectory")
        if not npc_directory:
            return jsonify({"success": False, "error": "npcDirectory is required"}), 400
        target_db = data.get("targetDb", app.config.get('DB_PATH'))
        if not models_dir or not model_name:
            return jsonify({"success": False, "error": "modelsDir and modelName are required"}), 400
        if not os.path.exists(models_dir):
            return jsonify({"success": False, "error": f"Models directory not found: {models_dir}"}), 404
        compiler = ModelCompiler(
            models_dir=models_dir,
            target_engine=target_db,
            npc_directory=npc_directory
        )
        compiler.discover_models()
        if model_name not in compiler.models:
            available = list(compiler.models.keys())
            return jsonify({
                "success": False,
                "error": f"Model '{model_name}' not found. Available: {available}"
            }), 404
        result_df = compiler.execute_model(model_name)
        row_count = len(result_df) if result_df is not None else 0
        return jsonify({
            "success": True,
            "rows": row_count,
            "message": f"Model '{model_name}' executed successfully. {row_count} rows materialized."
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500
@app.route("/api/npcsql/run_all", methods=["POST"])
def run_all_npcsql_models():
    """Execute all SQL models in dependency order using ModelCompiler"""
    try:
        from npcpy.sql.npcsql import ModelCompiler
        data = request.json
        models_dir = data.get("modelsDir")
        npc_directory = data.get("npcDirectory")
        if not npc_directory:
            return jsonify({"success": False, "error": "npcDirectory is required"}), 400
        target_db = data.get("targetDb", app.config.get('DB_PATH'))
        if not models_dir:
            return jsonify({"success": False, "error": "modelsDir is required"}), 400
        if not os.path.exists(models_dir):
            return jsonify({"success": False, "error": f"Models directory not found: {models_dir}"}), 404
        compiler = ModelCompiler(
            models_dir=models_dir,
            target_engine=target_db,
            npc_directory=npc_directory
        )
        results = compiler.run_all_models()
        summary = {
            name: len(df) if df is not None else 0
            for name, df in results.items()
        }
        return jsonify({
            "success": True,
            "models_executed": list(results.keys()),
            "row_counts": summary,
            "message": f"Executed {len(results)} models successfully."
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500
@app.route("/api/npcsql/models", methods=["GET"])
def list_npcsql_models():
    """List available SQL models in a directory"""
    try:
        from npcpy.sql.npcsql import ModelCompiler
        models_dir = request.args.get("modelsDir")
        if not models_dir:
            return jsonify({"success": False, "error": "modelsDir query param required"}), 400
        if not os.path.exists(models_dir):
            return jsonify({"models": [], "error": None})
        compiler = ModelCompiler(
            models_dir=models_dir,
            target_engine=app.config.get('DB_PATH'),
            npc_directory=app.config.get('user_npc_directory')
        )
        compiler.discover_models()
        models_info = []
        for name, model in compiler.models.items():
            models_info.append({
                "name": name,
                "path": model.path,
                "has_ai_function": model.has_ai_function,
                "dependencies": list(model.dependencies),
                "config": model.config
            })
        return jsonify({"models": models_info, "error": None})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"models": [], "error": str(e)}), 500
@app.route("/api/cron/crontab", methods=["GET"])
def get_crontab():
    system = platform.system()
    result = {"platform": system.lower(), "user_crontab": "", "system_crontab": "", "cron_d": [], "timers": [], "services": []}
    if system == "Windows":
        r = subprocess.run(["schtasks", "/query", "/fo", "CSV", "/nh"], capture_output=True, text=True)
        result["user_crontab"] = r.stdout if r.returncode == 0 else ""
    else:
        r = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
        result["user_crontab"] = r.stdout if r.returncode == 0 else ""
        try:
            with open("/etc/crontab", "r") as f:
                result["system_crontab"] = f.read()
        except Exception:
            pass
        cron_d_dir = "/etc/cron.d"
        if os.path.isdir(cron_d_dir):
            for fname in sorted(os.listdir(cron_d_dir)):
                fpath = os.path.join(cron_d_dir, fname)
                if os.path.isfile(fpath):
                    try:
                        with open(fpath, "r") as f:
                            result["cron_d"].append({"name": fname, "content": f.read()})
                    except Exception:
                        result["cron_d"].append({"name": fname, "content": "(unreadable)"})
        if system == "Linux":
            r = subprocess.run(["systemctl", "list-timers", "--all", "--no-pager"], capture_output=True, text=True)
            if r.returncode == 0:
                result["timers"] = r.stdout
            r = subprocess.run(["systemctl", "--user", "list-units", "--type=service", "--all", "--no-pager", "--plain"], capture_output=True, text=True)
            if r.returncode == 0:
                result["services"] = r.stdout
    return jsonify(result)
@app.route("/api/cron/daemons", methods=["GET"])
def list_system_daemons():
    system = platform.system()
    result = {"services": "", "platform": system.lower()}
    if system == "Linux":
        r = subprocess.run(["systemctl", "list-units", "--type=service", "--state=running", "--no-pager", "--plain"], capture_output=True, text=True)
        result["services"] = r.stdout if r.returncode == 0 else ""
        r2 = subprocess.run(["systemctl", "--user", "list-units", "--type=service", "--state=running", "--no-pager", "--plain"], capture_output=True, text=True)
        if r2.returncode == 0:
            result["user_services"] = r2.stdout
    elif system == "Darwin":
        r = subprocess.run(["launchctl", "list"], capture_output=True, text=True)
        result["services"] = r.stdout if r.returncode == 0 else ""
    elif system == "Windows":
        r = subprocess.run(["tasklist", "/fo", "CSV", "/nh"], capture_output=True, text=True)
        result["services"] = r.stdout if r.returncode == 0 else ""
    return jsonify(result)
@app.route("/api/cron/service-info/<unit>", methods=["GET"])
def get_service_info(unit):
    """Return unit file contents and recent journal logs for a systemd service."""
    system = platform.system()
    if system != "Linux":
        return jsonify({"error": "Only supported on Linux"})
    result = {"unit": unit, "unit_file": "", "journal": ""}
    r = subprocess.run(["systemctl", "cat", unit], capture_output=True, text=True)
    if r.returncode == 0:
        result["unit_file"] = r.stdout
    else:
        r2 = subprocess.run(["systemctl", "--user", "cat", unit], capture_output=True, text=True)
        if r2.returncode == 0:
            result["unit_file"] = r2.stdout
    r = subprocess.run(["journalctl", "-u", unit, "-n", "100", "--no-pager", "--output=short-iso"], capture_output=True, text=True)
    if r.returncode == 0:
        result["journal"] = r.stdout
    else:
        r2 = subprocess.run(["journalctl", "--user-unit", unit, "-n", "100", "--no-pager", "--output=short-iso"], capture_output=True, text=True)
        if r2.returncode == 0:
            result["journal"] = r2.stdout
    return jsonify(result)
@app.route("/api/npc_team_global")
def get_npc_team_global():
    npc_data = []
    seen_names = set()
    registered_teams = _parse_registered_teams()
    search_dirs = []
    user_dir = app.config.get('user_npc_directory')
    if user_dir and os.path.exists(user_dir):
        search_dirs.append(user_dir)
    for p in registered_teams:
        if p and os.path.isdir(p):
            search_dirs.append(p)
    for team_dir in search_dirs:
        try:
            team = Team(team_path=team_dir)
            for name, npc in team.npcs.items():
                if name not in seen_names:
                    seen_names.add(name)
                    d = npc.to_dict()
                    npc_data.append(d)
        except Exception as e:
            print(f"Error loading team from {team_dir}: {e}")
    return jsonify({"npcs": npc_data, "error": None})
@app.route("/api/npc_team_project", methods=["GET"])
def get_npc_team_project():
    project_npc_directory = request.args.get("currentPath")
    if not project_npc_directory:
        return jsonify({"npcs": [], "error": "currentPath required"}), 400
    if not project_npc_directory.endswith("npc_team"):
        project_npc_directory = os.path.join(
            project_npc_directory,
            "npc_team"
        )
    if not os.path.exists(project_npc_directory):
        return jsonify({"npcs": [], "error": None})
    try:
        team = Team(team_path=project_npc_directory)
        npc_data = []
        for npc in team.npcs.values():
            d = npc.to_dict()
            d["team"] = "project"
            npc_data.append(d)
    except Exception as e:
        print(f"Error loading project team from {project_npc_directory}: {e}")
        npc_data = []
    return jsonify({"npcs": npc_data, "error": None})
@app.route("/api/npc_team_from_path", methods=["GET"])
def get_npc_team_from_path():
    team_path = request.args.get("path")
    if not team_path or not os.path.isdir(team_path):
        return jsonify({"npcs": [], "error": "invalid path"})
    try:
        team = Team(team_path=team_path)
        npc_data = []
        for npc in team.npcs.values():
            d = npc.to_dict()
            d["team"] = os.path.basename(team_path)
            npc_data.append(d)
        return jsonify({"npcs": npc_data, "error": None})
    except Exception as e:
        print(f"Error loading team from {team_path}: {e}")
        return jsonify({"npcs": [], "error": str(e)})
@app.route("/api/npc-team/import", methods=["POST"])
def import_npc_team():
    """
    Import an npc_team from a git repository URL.
    Clones the repo, finds npc_team/ directory, copies contents to target.
    """
    import tempfile
    import shutil as _shutil
    data = request.json or {}
    repo_url = data.get("repoUrl", "").strip()
    scope = data.get("scope", "global")
    current_path = data.get("currentPath", "")
    branch = data.get("branch", "")
    if not repo_url:
        return jsonify({"error": "repoUrl is required"}), 400
    if scope == "global":
        target = app.config.get('user_npc_directory')
        if not target:
            return jsonify({"error": "user_npc_directory not configured"}), 500
    else:
        if not current_path:
            return jsonify({"error": "currentPath required for project scope"}), 400
        target = os.path.join(current_path, "npc_team")
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            clone_cmd = ["git", "clone", "--depth", "1"]
            if branch:
                clone_cmd += ["-b", branch]
            clone_cmd += [repo_url, tmp_dir]
            result = subprocess.run(
                clone_cmd, capture_output=True, text=True, timeout=120
            )
            if result.returncode != 0:
                return jsonify({"error": f"Git clone failed: {result.stderr.strip()}"}), 400
            npc_team_src = os.path.join(tmp_dir, "npc_team")
            if not os.path.isdir(npc_team_src):
                for item in os.listdir(tmp_dir):
                    candidate = os.path.join(tmp_dir, item, "npc_team")
                    if os.path.isdir(candidate):
                        npc_team_src = candidate
                        break
            if not os.path.isdir(npc_team_src):
                return jsonify({"error": "No npc_team/ directory found in repository"}), 404
            imported = {"jinxes": 0, "npcs": 0, "contexts": 0, "other": 0}
            for root, dirs, files in os.walk(npc_team_src):
                dirs[:] = [d for d in dirs if d != '.git']
                rel = os.path.relpath(root, npc_team_src)
                dest_dir = os.path.join(target, rel) if rel != '.' else target
                os.makedirs(dest_dir, exist_ok=True)
                for f in files:
                    src_file = os.path.join(root, f)
                    dst_file = os.path.join(dest_dir, f)
                    _shutil.copy2(src_file, dst_file)
                    if f.endswith(".jinx"):
                        imported["jinxes"] += 1
                    elif f.endswith(".npc"):
                        imported["npcs"] += 1
                    elif f.endswith(".ctx"):
                        imported["contexts"] += 1
                    else:
                        imported["other"] += 1
        return jsonify({"status": "success", "imported": imported, "target": target, "error": None})
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Git clone timed out (120s limit)"}), 504
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
def get_ctx_path(is_global, current_path=None, create_default=False):
    """Determines the path to the .ctx file."""
    if is_global:
        user_npc_dir = app.config.get('user_npc_directory')
        if not user_npc_dir:
            return None
        ctx_dir = os.path.join(user_npc_dir)
        ctx_files = glob.glob(os.path.join(ctx_dir, "*.ctx"))
        if ctx_files:
            return ctx_files[0]
        elif create_default:
            return os.path.join(ctx_dir, "team.ctx")
        return None
    else:
        if not current_path:
            return None
        ctx_dir = os.path.join(current_path, "npc_team")
        ctx_files = glob.glob(os.path.join(ctx_dir, "*.ctx"))
        if ctx_files:
            return ctx_files[0]
        elif create_default:
            return os.path.join(ctx_dir, "team.ctx")
        return None
def read_ctx_file(file_path):
    """Reads and parses a YAML .ctx file, normalizing list of strings to list of objects."""
    if file_path and os.path.exists(file_path):
        with open(file_path, 'r') as f:
            try:
                data = yaml.safe_load(f) or {}
                if 'databases' in data and isinstance(data['databases'], list):
                    normalized = []
                    for item in data['databases']:
                        if isinstance(item, dict):
                            normalized.append(item)
                        else:
                            normalized.append({"path": str(item)})
                    data['databases'] = normalized
                if 'mcp_servers' in data and isinstance(data['mcp_servers'], list):
                    data['mcp_servers'] = [
                        item if isinstance(item, dict) and 'value' in item
                        else {"value": item}
                        for item in data['mcp_servers']
                    ]
                if 'websites' in data and isinstance(data['websites'], list):
                    data['websites'] = [{"value": item} for item in data['websites']]
                return data
            except yaml.YAMLError as e:
                print(f"YAML parsing error in {file_path}: {e}")
                return {"error": "Failed to parse YAML."}
    return {} 
def write_ctx_file(file_path, data):
    """Writes a dictionary to a YAML .ctx file, denormalizing list of objects back to strings."""
    if not file_path:
        return False
    data_to_save = json.loads(json.dumps(data)) 
    if 'databases' in data_to_save and isinstance(data_to_save['databases'], list):
        normalized = []
        for item in data_to_save['databases']:
            if isinstance(item, dict):
                if set(item.keys()) == {"value"}:
                    normalized.append(item["value"])
                else:
                    normalized.append(item)
            elif isinstance(item, str):
                normalized.append(item)
        data_to_save['databases'] = normalized
    if 'mcp_servers' in data_to_save and isinstance(data_to_save['mcp_servers'], list):
        normalized = []
        for item in data_to_save['mcp_servers']:
            if isinstance(item, dict):
                has_extras = any(k in item for k in ('env', 'name', 'id'))
                if has_extras:
                    normalized.append({k: v for k, v in item.items() if v})
                else:
                    normalized.append(item.get("value", ""))
            elif isinstance(item, str):
                normalized.append(item)
        data_to_save['mcp_servers'] = normalized
    if 'websites' in data_to_save and isinstance(data_to_save['websites'], list):
        data_to_save['websites'] = [item.get("value", "") for item in data_to_save['websites'] if isinstance(item, dict)]
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        yaml.dump(data_to_save, f, default_flow_style=False, sort_keys=False)
    return True
@app.route("/api/context/global", methods=["GET"])
def get_global_context():
    """Gets the global team.ctx content."""
    try:
        ctx_path = get_ctx_path(is_global=True)
        data = read_ctx_file(ctx_path)
        return jsonify({"context": data, "path": ctx_path, "error": None})
    except Exception as e:
        print(f"Error getting global context: {e}")
        return jsonify({"error": str(e)}), 500
@app.route("/api/context/global", methods=["POST"])
def save_global_context():
    """Saves the global team.ctx content."""
    try:
        data = request.json.get("context", {})
        ctx_path = get_ctx_path(is_global=True)
        if write_ctx_file(ctx_path, data):
            return jsonify({"message": "Global context saved.", "error": None})
        else:
            return jsonify({"error": "Failed to write global context file."}), 500
    except Exception as e:
        print(f"Error saving global context: {e}")
        return jsonify({"error": str(e)}), 500
@app.route("/api/context/project", methods=["GET"])
def get_project_context():
    """Gets the project-specific team.ctx content."""
    try:
        current_path = request.args.get("path")
        if not current_path:
            return jsonify({"error": "Project path is required."}), 400
        ctx_path = get_ctx_path(is_global=False, current_path=current_path)
        data = read_ctx_file(ctx_path)
        return jsonify({"context": data, "path": ctx_path, "error": None})
    except Exception as e:
        print(f"Error getting project context: {e}")
        return jsonify({"error": str(e)}), 500
@app.route("/api/context/project", methods=["POST"])
def save_project_context():
    """Saves the project-specific team.ctx content."""
    try:
        data = request.json
        current_path = data.get("path")
        context_data = data.get("context", {})
        if not current_path:
            return jsonify({"error": "Project path is required."}), 400
        ctx_path = get_ctx_path(is_global=False, current_path=current_path)
        if write_ctx_file(ctx_path, context_data):
            return jsonify({"message": "Project context saved.", "error": None})
        else:
            return jsonify({"error": "Failed to write project context file."}), 500
    except Exception as e:
        print(f"Error saving project context: {e}")
        return jsonify({"error": str(e)}), 500
@app.route("/api/context/project/init", methods=["POST"])
def init_project_team():
    """Initialize a new npc_team folder in the project directory."""
    try:
        data = request.json
        project_path = data.get("path")
        if not project_path:
            return jsonify({"error": "Project path is required."}), 400
        result = initialize_npc_project(directory=project_path)
        return jsonify({"message": "Project team initialized.", "path": result, "error": None})
    except Exception as e:
        print(f"Error initializing project team: {e}")
        return jsonify({"error": str(e)}), 500
@app.route("/api/npc-team/status", methods=["GET"])
def npc_team_sync_status_endpoint():
    try:
        return jsonify(team_sync_status(request.args.get("team_path")))
    except Exception as e:
        return jsonify({"status": "unavailable", "error": str(e)})
@app.route("/api/npc-team/init", methods=["POST"])
def npc_team_sync_init_endpoint():
    try:
        data = request.json or {}
        return jsonify(team_sync_init(data.get("team_path")))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/api/npc-team/sync", methods=["POST"])
def npc_team_sync_pull_endpoint():
    try:
        data = request.json or {}
        result = team_sync_pull(data.get("team_path"))
        if "error" in result and result["error"] and "conflicts" not in result:
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/api/npc-team/resolve", methods=["POST"])
def npc_team_sync_resolve_endpoint():
    try:
        data = request.json or {}
        result = team_sync_resolve(
            team_path=data.get("team_path"),
            file_path=data.get("file"),
            resolution=data.get("resolution", "ours"),
            content=data.get("content"),
        )
        if "error" in result and result["error"]:
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/api/npc-team/commit", methods=["POST"])
def npc_team_sync_commit_endpoint():
    try:
        data = request.json or {}
        return jsonify(team_sync_commit(data.get("team_path"), data.get("message", "Update NPC team")))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/api/npc-team/diff", methods=["GET"])
def npc_team_sync_diff_endpoint():
    try:
        return jsonify(team_sync_diff(request.args.get("team_path"), request.args.get("file")))
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/api/context/websites", methods=["GET"])
def get_context_websites():
    """Gets the websites list from a .ctx file."""
    try:
        current_path = request.args.get("path")
        is_global = request.args.get("global", "false").lower() == "true"
        ctx_path = get_ctx_path(is_global=is_global, current_path=current_path)
        data = read_ctx_file(ctx_path)
        websites = data.get("websites", [])
        if isinstance(websites, list):
            normalized = []
            for item in websites:
                if isinstance(item, str):
                    normalized.append({"value": item})
                elif isinstance(item, dict):
                    normalized.append(item)
            websites = normalized
        return jsonify({
            "websites": websites,
            "path": ctx_path,
            "error": None
        })
    except Exception as e:
        print(f"Error getting websites from context: {e}")
        return jsonify({"error": str(e)}), 500
@app.route("/api/context/websites", methods=["POST"])
def save_context_websites():
    """Saves the websites list to a .ctx file."""
    try:
        data = request.json
        websites = data.get("websites", [])
        current_path = data.get("path")
        is_global = data.get("global", False)
        ctx_path = get_ctx_path(is_global=is_global, current_path=current_path, create_default=True)
        if not ctx_path:
            return jsonify({"error": "Could not determine ctx file path. Provide a path or use global=true."}), 400
        existing_data = read_ctx_file(ctx_path) or {}
        normalized_websites = []
        for item in websites:
            if isinstance(item, dict) and "value" in item:
                normalized_websites.append(item["value"])
            elif isinstance(item, str):
                normalized_websites.append(item)
        existing_data["websites"] = normalized_websites
        if write_ctx_file(ctx_path, existing_data):
            return jsonify({
                "message": "Websites saved to context.",
                "websites": [{"value": w} for w in normalized_websites],
                "path": ctx_path,
                "error": None
            })
        else:
            return jsonify({"error": "Failed to write context file."}), 500
    except Exception as e:
        print(f"Error saving websites to context: {e}")
        return jsonify({"error": str(e)}), 500
@app.route("/api/get_attachment_response", methods=["POST"])
def get_attachment_response():
    data = request.json
    attachments = data.get("attachments", [])
    messages = data.get("messages")
    conversation_id = data.get("conversationId")
    current_path = data.get("currentPath")
    if current_path:
        loaded_vars = load_project_env(current_path)
        print(f"Loaded project env variables for attachment response: {list(loaded_vars.keys())}")
    npc_object = None
    if npc_name:
        npc_object = load_npc_by_name_and_source(npc_name, npc_source, current_path)
        if not npc_object and npc_source == 'project':
            print(f"NPC {npc_name} not found in project directory, trying global...")
            npc_object = load_npc_by_name_and_source(npc_name, 'global')
        if npc_object:
            print(f"Successfully loaded NPC {npc_name} from {npc_source} directory")
        else:
            print(f"Warning: Could not load NPC {npc_name}")
    images = []
    attachments_loaded = []
    for attachment in attachments:
        extension = attachment["name"].split(".")[-1]
        extension_mapped = extension_map.get(extension.upper(), "others")
        _type_dir_map = {
            "images": get_images_dir(),
            "videos": get_videos_dir(),
            "models": get_models_dir(),
        }
        _type_base = _type_dir_map.get(extension_mapped, os.path.join(get_attachments_dir(), extension_mapped))
        os.makedirs(_type_base, exist_ok=True)
        file_path = os.path.join(_type_base, attachment["name"])
        if extension_mapped == "images":
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            img = Image.open(attachment["path"])
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)
            img.save(file_path, optimize=True, quality=50)
            images.append(file_path)
            attachments_loaded.append({
                "name": attachment["name"], "type": extension_mapped,
                "data": img_byte_arr.read(), "size": os.path.getsize(file_path)
            })
    message_to_send = messages[-1]["content"]
    if isinstance(message_to_send, list):
        message_to_send = message_to_send[0]
    response = get_llm_response(
        message_to_send,
        images=images,
        messages=messages,
        model=model,
        provider=provider,
        npc=npc_object,
    )
    messages = response["messages"]
    response = response["response"]
    return jsonify({
        "status": "success",
        "message": response,
        "conversationId": conversation_id,
        "messages": messages,
    })
# Minimal fallback for providers litellm doesn't cover with image model lists.
# These are suggestions, not gates — users can always pass a custom model ID.
_IMAGE_MODELS_FALLBACK = {
    "ollama": [
        {"value": "x/z-image-turbo", "display_name": "Z-Image Turbo (6B)"},
        {"value": "x/flux2-klein", "display_name": "FLUX.2 Klein (4B)"},
    ],
    "bfl": [
        {"value": "flux-pro-1.1", "display_name": "FLUX Pro 1.1"},
        {"value": "flux-pro", "display_name": "FLUX Pro"},
        {"value": "flux-dev", "display_name": "FLUX Dev"},
    ],
    "bagel": [
        {"value": "bagel-image-v1", "display_name": "Bagel Image v1"},
    ],
    "leonardo": [
        {"value": "leonardo-diffusion-xl", "display_name": "Leonardo Diffusion XL"},
        {"value": "leonardo-vision-xl", "display_name": "Leonardo Vision XL"},
    ],
    "ideogram": [
        {"value": "ideogram-v2", "display_name": "Ideogram v2"},
        {"value": "ideogram-v2-turbo", "display_name": "Ideogram v2 Turbo"},
    ],
}

# Map provider key -> (litellm_attr_name, filter_func)
# If attr_name is None, the provider has no litellm image coverage and falls back to hardcoded.
_LITELLM_IMAGE_PROVIDER_ATTRS = {
    "gemini": ("gemini_models", lambda m: any(k in m.lower() for k in ["image", "imagen", "nano", "banana"]) and "veo" not in m.lower()),
    "openai": ("openai_image_generation_models", None),
    "stability": ("stability_models", None),
    "fal": ("fal_ai_models", None),
    "replicate": (None, None),
    "together": (None, None),
    "fireworks": (None, None),
    "deepinfra": (None, None),
    "ollama": (None, None),
}

_LOCAL_DIFFUSERS_CACHE_PATH = Path.home() / ".npcsh" / "image_models_cache.json"


def _scan_hf_diffusers_cache():
    """Scan HuggingFace cache for diffusers models (identified by model_index.json)."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    if not cache_dir.exists():
        return []
    models = []
    for entry in cache_dir.iterdir():
        if not entry.is_dir() or not entry.name.startswith("models--"):
            continue
        model_id = entry.name.replace("models--", "").replace("--", "/")
        found = False
        snaps = entry / "snapshots"
        if snaps.exists():
            for snap in snaps.iterdir():
                if snap.is_dir() and (snap / "model_index.json").exists():
                    found = True
                    break
        if found:
            models.append({
                "value": model_id,
                "provider": "diffusers",
                "display_name": f"{model_id} | diffusers (HF cache)",
            })
    return models


def _get_cached_local_diffusers_models(force_refresh=False):
    """Return cached local diffusers models; refresh if stale or forced."""
    if not force_refresh and _LOCAL_DIFFUSERS_CACHE_PATH.exists():
        try:
            data = json.loads(_LOCAL_DIFFUSERS_CACHE_PATH.read_text())
            cached_at = datetime.datetime.fromisoformat(data.get("cached_at", "1970-01-01"))
            if (datetime.datetime.now() - cached_at).days < 1:
                return data.get("models", [])
        except Exception:
            pass
    models = _scan_hf_diffusers_cache()
    try:
        _LOCAL_DIFFUSERS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _LOCAL_DIFFUSERS_CACHE_PATH.write_text(
            json.dumps({
                "cached_at": datetime.datetime.now().isoformat(),
                "models": models,
            }, indent=2)
        )
    except Exception as e:
        print(f"Warning: could not write local diffusers cache: {e}")
    return models


IMAGE_PROVIDER_API_KEYS = {
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "stability": "STABILITY_API_KEY",
    "replicate": "REPLICATE_API_TOKEN",
    "fal": "FAL_KEY",
    "together": "TOGETHER_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "deepinfra": "DEEPINFRA_API_KEY",
    "bfl": "BFL_API_KEY",
    "bagel": "BAGEL_API_KEY",
    "leonardo": "LEONARDO_API_KEY",
    "ideogram": "IDEOGRAM_API_KEY",
}
def _get_finetuned_models_internal(current_path=None):
    potential_root_paths = [
        get_models_dir(),
        get_images_dir(),
    ]
    if current_path:
        project_models_path = os.path.join(current_path, 'models')
        project_images_path = os.path.join(current_path, 'images')
        potential_root_paths.extend([project_models_path, project_images_path])
    finetuned_models = []
    print(f"🌋 (Internal) Searching for fine-tuned models in potential root paths: {set(potential_root_paths)}")
    for root_path in set(potential_root_paths):
        if not os.path.exists(root_path) or not os.path.isdir(root_path):
            print(f"🌋 (Internal) Skipping non-existent or non-directory root path: {root_path}")
            continue
        print(f"🌋 (Internal) Scanning root path: {root_path}")
        for model_dir_name in os.listdir(root_path):
            full_model_path = os.path.join(root_path, model_dir_name)
            if not os.path.isdir(full_model_path):
                print(f"🌋 (Internal) Skipping {full_model_path}: Not a directory.")
                continue
            has_model_final_pt = os.path.exists(os.path.join(full_model_path, 'model_final.pt'))
            has_checkpoints_dir = os.path.isdir(os.path.join(full_model_path, 'checkpoints'))
            if has_model_final_pt or has_checkpoints_dir:
                print(f"🌋 (Internal) Identified fine-tuned model: {model_dir_name} at {full_model_path} (found model_final.pt or checkpoints dir)")
                finetuned_models.append({
                    "value": full_model_path,
                    "provider": "diffusers",
                    "display_name": f"{model_dir_name} | Fine-tuned Diffuser"
                })
                continue
            print(f"🌋 (Internal) Skipping {full_model_path}: No model_final.pt or checkpoints directory found at root.")
    print(f"🌋 (Internal) Finished scanning. Found {len(finetuned_models)} fine-tuned models.")
    return {"models": finetuned_models, "error": None}
def get_available_image_models(current_path=None):
    """
    Retrieves available image generation models from litellm, local HF cache,
    fine-tuned models, and minimal fallbacks.  No hardcoded allowlist.
    """
    if current_path:
        load_project_env(current_path)
    all_image_models = []

    # 1) Configured custom model
    cfg_image_model = app.config.get('IMAGE_MODEL')
    cfg_image_provider = app.config.get('IMAGE_PROVIDER')
    if cfg_image_model and cfg_image_provider:
        all_image_models.append({
            "value": cfg_image_model,
            "provider": cfg_image_provider,
            "display_name": f"{cfg_image_model} | {cfg_image_provider} (Configured)",
        })

    # 2) litellm-driven providers + fallbacks + custom passthrough
    try:
        import litellm
    except Exception:
        litellm = None

    for provider_key, api_key_env in IMAGE_PROVIDER_API_KEYS.items():
        if not os.environ.get(api_key_env):
            continue

        # litellm lookup
        attr_name, filter_fn = _LITELLM_IMAGE_PROVIDER_ATTRS.get(provider_key, (None, None))
        if litellm and attr_name:
            try:
                model_set = getattr(litellm, attr_name, None) or set()
                for model_id in sorted(model_set):
                    if filter_fn and not filter_fn(model_id):
                        continue
                    # Strip provider prefix (e.g. gemini/gemini-3-pro-image -> gemini-3-pro-image)
                    clean_id = model_id.split("/")[-1] if "/" in model_id else model_id
                    all_image_models.append({
                        "value": clean_id,
                        "provider": provider_key,
                        "display_name": f"{clean_id} | {provider_key}",
                    })
            except Exception as e:
                print(f"Warning: litellm lookup failed for {provider_key}: {e}")

        # Minimal hardcoded fallback for providers litellm doesn't cover
        fallback = _IMAGE_MODELS_FALLBACK.get(provider_key, [])
        for model in fallback:
            all_image_models.append({
                **model,
                "provider": provider_key,
                "display_name": f"{model['display_name']} | {provider_key}",
            })

        # Always expose a custom passthrough so users aren't gated by discovery
        all_image_models.append({
            "value": "__custom__",
            "provider": provider_key,
            "display_name": f"Custom {provider_key} model",
        })

    # 3) Local diffusers from HF cache (cached in ~/.npcsh)
    try:
        all_image_models.extend(_get_cached_local_diffusers_models())
    except Exception as e:
        print(f"Warning: local diffusers cache lookup failed: {e}")

    # 4) Fine-tuned project models
    try:
        finetuned_data_result = _get_finetuned_models_internal(current_path)
        if finetuned_data_result and finetuned_data_result.get("models"):
            all_image_models.extend(finetuned_data_result["models"])
        elif finetuned_data_result.get("error"):
            print(f"Internal error in _get_finetuned_models_internal: {finetuned_data_result['error']}")
    except Exception as e:
        print(f"Error calling _get_finetuned_models_internal: {e}")

    # 5) Deduplicate
    seen_models = set()
    unique_models = []
    for model_entry in all_image_models:
        key = (model_entry["value"], model_entry["provider"])
        if key not in seen_models:
            seen_models.add(key)
            unique_models.append(model_entry)
    return unique_models
@app.route('/api/generative_fill', methods=['POST'])
def generative_fill():
    data = request.get_json()
    image_path = data.get('imagePath')
    mask_data = data.get('mask')
    prompt = data.get('prompt')
    model = data.get('model')
    provider = data.get('provider')
    if not all([image_path, mask_data, prompt, model, provider]):
        return jsonify({"error": "Missing required fields"}), 400
    try:
        image_path = os.path.expanduser(image_path)
        mask_b64 = mask_data.split(',')[1] if ',' in mask_data else mask_data
        mask_bytes = base64.b64decode(mask_b64)
        mask_image = Image.open(BytesIO(mask_bytes))
        original_image = Image.open(image_path)
        if provider == 'openai':
            result = inpaint_openai(original_image, mask_image, prompt, model)
        elif provider == 'gemini':
            result = inpaint_gemini(original_image, mask_image, prompt, model)
        elif provider == 'diffusers':
            result = inpaint_diffusers(original_image, mask_image, prompt, model)
        else:
            return jsonify({"error": f"Provider {provider} not supported"}), 400
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"inpaint_{timestamp}.png"
        save_dir = os.path.dirname(image_path)
        result_path = os.path.join(save_dir, filename)
        result.save(result_path)
        return jsonify({"resultPath": result_path, "error": None})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
def inpaint_openai(image, mask, prompt, model):
    import io
    from openai import OpenAI
    from PIL import Image
    import base64
    client = OpenAI()
    original_size = image.size
    if model == 'dall-e-2':
        max_dim = max(image.width, image.height)
        if max_dim <= 256:
            target_size = (256, 256); size_str = '256x256'
        elif max_dim <= 512:
            target_size = (512, 512); size_str = '512x512'
        else:
            target_size = (1024, 1024); size_str = '1024x1024'
    else:
        valid_sizes = {
            (1024, 1024): "1024x1024",
            (1024, 1536): "1024x1536",
            (1536, 1024): "1536x1024",
        }
        target_size = (1024, 1024)
        for size in valid_sizes.keys():
            if image.width > image.height and size == (1536, 1024):
                target_size = size; break
            elif image.height > image.width and size == (1024, 1536):
                target_size = size; break
        size_str = valid_sizes[target_size]
    resized_image = image.resize(target_size, Image.Resampling.LANCZOS)
    mask_l = mask.convert('L').resize(target_size, Image.Resampling.NEAREST)
    edit_rgba = Image.new('RGBA', target_size, (255, 255, 255, 255))
    alpha = mask_l.point(lambda v: 0 if v > 128 else 255)
    edit_rgba.putalpha(alpha)
    rgba_image = resized_image.convert('RGBA')
    rgba_image.putalpha(alpha)
    img_bytes = io.BytesIO()
    rgba_image.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    img_bytes.name = 'image.png'
    mask_bytes = io.BytesIO()
    edit_rgba.save(mask_bytes, format='PNG')
    mask_bytes.seek(0)
    mask_bytes.name = 'mask.png'
    response = client.images.edit(
        model=model,
        image=img_bytes,
        mask=mask_bytes,
        prompt=prompt,
        n=1,
        size=size_str,
    )
    if response.data[0].url:
        import requests
        img_data = requests.get(response.data[0].url).content
    elif hasattr(response.data[0], 'b64_json'):
        img_data = base64.b64decode(response.data[0].b64_json)
    else:
        raise Exception("No image data in response")
    result_image = Image.open(io.BytesIO(img_data))
    result_image = result_image.resize(original_size, Image.Resampling.LANCZOS)
    full_mask_l = mask.convert('L').resize(original_size, Image.Resampling.NEAREST)
    base = image.convert('RGBA')
    over = result_image.convert('RGBA')
    composed = Image.composite(over, base, full_mask_l)
    return composed.convert(image.mode if image.mode in ('RGB', 'RGBA') else 'RGB')
def inpaint_diffusers(image, mask, prompt, model):
    from diffusers import StableDiffusionInpaintPipeline
    import torch
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model,
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask
    ).images[0]
    return result
def inpaint_gemini(image, mask, prompt, model):
    from npcpy.gen.image_gen import generate_image
    import io
    import numpy as np
    from PIL import Image as PILImage
    mask_l = mask.convert('L').resize(image.size, PILImage.NEAREST)
    mask_np = np.array(mask_l)
    ys, xs = np.where(mask_np > 128)
    if len(xs) == 0:
        return image
    x_center = int(np.mean(xs))
    y_center = int(np.mean(ys))
    width_pct = (xs.max() - xs.min()) / image.width * 100
    height_pct = (ys.max() - ys.min()) / image.height * 100
    position = "center"
    if y_center < image.height / 3:
        position = "top"
    elif y_center > 2 * image.height / 3:
        position = "bottom"
    if x_center < image.width / 3:
        position += " left"
    elif x_center > 2 * image.width / 3:
        position += " right"
    img_bytes = io.BytesIO()
    image.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    full_prompt = (
        f"Using the provided image, change only the region in the {position} "
        f"(approximately {int(width_pct)}% wide by {int(height_pct)}% tall) to: {prompt}.\n\n"
        "Keep everything else exactly the same, matching the original lighting and style. "
        "You are in-painting the image. Do NOT alter anything outside that region. "
        "Return an image with the same dimensions as the input."
    )
    results = generate_image(
        prompt=full_prompt,
        model=model,
        provider='gemini',
        attachments=[img_bytes],
        n_images=1,
    )
    if not results:
        return None
    gen = results[0]
    if gen.size != image.size:
        gen = gen.resize(image.size, PILImage.LANCZOS)
    base = image.convert('RGBA')
    over = gen.convert('RGBA')
    alpha = mask_l
    composed = PILImage.composite(over, base, alpha)
    return composed.convert(image.mode if image.mode in ('RGB', 'RGBA') else 'RGB')
@app.route('/api/generate_images', methods=['POST'])
def generate_images():
    data = request.get_json()
    prompt = data.get('prompt')
    n = data.get('n', 1)
    model_name = data.get('model')
    provider_name = data.get('provider')
    attachments = data.get('attachments', [])
    base_filename = data.get('base_filename', 'vixynt_gen')  
    save_dir = data.get('currentPath', get_images_dir())     
    if not prompt:
        return jsonify({"error": "Prompt is required."}), 400
    if not model_name or not provider_name:
        return jsonify({"error": "Image model and provider are required."}), 400
    save_dir = os.path.expanduser(save_dir)
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename_with_time = f"{base_filename}_{timestamp}"
    generated_images_base64 = []
    generated_filenames = []
    try:
        input_images = []
        attachments_loaded = []
        if attachments:
            for attachment in attachments:
                print(attachment)
                if isinstance(attachment, dict) and 'path' in attachment:
                    image_path = attachment['path']
                    if os.path.exists(image_path):
                        try:
                            pil_img = Image.open(image_path)
                            pil_img = pil_img.convert("RGB")
                            pil_img.thumbnail((1024, 1024))
                            input_images.append(pil_img)
                            compressed_bytes = BytesIO()
                            pil_img.save(compressed_bytes, format="JPEG", quality=85, optimize=True)
                            img_data = compressed_bytes.getvalue()
                            attachments_loaded.append({
                                "name": os.path.basename(image_path),
                                "type": "images",
                                "data": img_data,
                                "size": len(img_data)
                            })
                        except Exception as e:
                            print(f"Warning: Could not load attachment image {image_path}: {e}")
        images_list = gen_image(
            prompt, 
            model=model_name, 
            provider=provider_name, 
            n_images=n,
            input_images=input_images if input_images else None
        )
        print(images_list)
        if not isinstance(images_list, list):
            images_list = [images_list] if images_list is not None else []
        generated_attachments = []
        for i, pil_image in enumerate(images_list):
            if isinstance(pil_image, Image.Image):
                filename = f"{base_filename_with_time}_{i+1:03d}.png" if n > 1 else f"{base_filename_with_time}.png"
                filepath = os.path.join(save_dir, filename)
                print(f'saved file to {filepath}')
                pil_image.save(filepath, format="PNG")
                generated_filenames.append(filepath)
                buffered = BytesIO()
                pil_image.save(buffered, format="PNG")
                img_data = buffered.getvalue()
                generated_attachments.append({
                    "name": filename,
                    "type": "images", 
                    "data": img_data,
                    "size": len(img_data)
                })
                img_str = base64.b64encode(img_data).decode("utf-8")
                generated_images_base64.append(f"data:image/png;base64,{img_str}")
            else:
                converted = None
                try:
                    b64 = getattr(pil_image, "b64_json", None)
                    url = getattr(pil_image, "url", None)
                    if b64:
                        converted = Image.open(BytesIO(base64.b64decode(b64)))
                    elif url:
                        import requests as _req
                        resp = _req.get(url, timeout=30)
                        resp.raise_for_status()
                        converted = Image.open(BytesIO(resp.content))
                except Exception as _e:
                    print(f"Warning: failed to unwrap image object ({type(pil_image)}): {_e}")
                if converted is not None:
                    filename = f"{base_filename_with_time}_{i+1:03d}.png" if n > 1 else f"{base_filename_with_time}.png"
                    filepath = os.path.join(save_dir, filename)
                    converted.save(filepath, format="PNG")
                    generated_filenames.append(filepath)
                    buffered = BytesIO()
                    converted.save(buffered, format="PNG")
                    img_data = buffered.getvalue()
                    generated_attachments.append({
                        "name": filename,
                        "type": "images",
                        "data": img_data,
                        "size": len(img_data),
                    })
                    img_str = base64.b64encode(img_data).decode("utf-8")
                    generated_images_base64.append(f"data:image/png;base64,{img_str}")
                    print(f"saved file to {filepath} (unwrapped from {type(pil_image).__name__})")
                else:
                    print(f"Warning: gen_image returned non-PIL object ({type(pil_image)}). Skipping image conversion.")
        return jsonify({
            "images": generated_images_base64, 
            "filenames": generated_filenames,
            "error": None
        })
    except Exception as e:
        print(f"Image generation error: {str(e)}")
        traceback.print_exc()
        return jsonify({"images": [], "filenames": [], "error": str(e)}), 500
@app.route("/api/mcp_tools", methods=["GET"])
def get_mcp_tools():
    """
    API endpoint to retrieve the list of tools available from a given MCP server script.
    It will try to use an existing client from corca_states if available and matching,
    otherwise it creates a temporary client.
    """
    raw_server_path = request.args.get("mcpServerPath")
    current_path_arg = request.args.get("currentPath")
    conversation_id = request.args.get("conversationId")
    npc_name = request.args.get("npc")
    selected_filter = request.args.get("selected", "")
    selected_names = [s.strip() for s in selected_filter.split(",") if s.strip()]
    if not raw_server_path:
        return jsonify({"error": "mcpServerPath parameter is required."}), 400
    resolved_path = resolve_mcp_server_path(
        current_path=current_path_arg,
        explicit_path=raw_server_path,
        force_global=False
    )
    if _is_command_string(resolved_path):
        server_path = resolved_path.strip()
    else:
        server_path = os.path.abspath(os.path.expanduser(resolved_path))
    temp_mcp_client = None
    jinx_tools = []
    try:
        if conversation_id and npc_name and hasattr(app, 'corca_states'):
            state_key = f"{conversation_id}_{npc_name or 'default'}"
            if state_key in app.corca_states:
                existing_corca_state = app.corca_states[state_key]
                if hasattr(existing_corca_state, 'mcp_client') and existing_corca_state.mcp_client \
                   and existing_corca_state.mcp_client.server_script_path == server_path:
                    print(f"Using existing MCP client for {state_key} to fetch tools.")
                    temp_mcp_client = existing_corca_state.mcp_client
                    if temp_mcp_client.is_connected():
                        tools = temp_mcp_client.available_tools_llm
                        if selected_names:
                            tools = [t for t in tools if t.get("function", {}).get("name") in selected_names]
                        return jsonify({"tools": tools, "error": None})
                    else:
                        temp_mcp_client.disconnect_sync()
                        existing_corca_state.mcp_client = None
        print(f"Creating a temporary MCP client to fetch tools for {server_path}.")
        temp_mcp_client = MCPClientNPC()
        if temp_mcp_client.connect_sync(server_path):
            server_label = os.path.basename(server_path).replace('.py', '') if server_path else 'mcp'
            mcp_tools = []
            for t in temp_mcp_client.available_tools_llm:
                tagged = dict(t)
                tagged["_source"] = f"mcp:{server_label}"
                mcp_tools.append(tagged)
            tools = mcp_tools
            if selected_names:
                tools = [t for t in tools if t.get("function", {}).get("name") in selected_names]
            try:
                temp_mcp_client.disconnect_sync()
            except Exception:
                pass
            return jsonify({"tools": tools, "error": None})
        else:
            try:
                temp_mcp_client.disconnect_sync()
            except Exception:
                pass
            return jsonify({"error": f"Failed to connect to MCP server at {server_path}."}), 500
    except FileNotFoundError as e:
        return jsonify({"error": f"MCP Server script not found: {e}"}), 404
    except ValueError as e:
        return jsonify({"error": f"Invalid MCP Server script: {e}"}), 400
    except Exception as e:
        print(f"Error getting MCP tools for {server_path}: {traceback.format_exc()}")
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500
    finally:
        if temp_mcp_client and temp_mcp_client.session and (
            not (conversation_id and npc_name and hasattr(app, 'corca_states') and state_key in app.corca_states and getattr(app.corca_states[state_key], 'mcp_client', None) == temp_mcp_client)
        ):
            print(f"Disconnecting temporary MCP client for {server_path}.")
            temp_mcp_client.disconnect_sync()
def _parse_registered_teams():
    """Parse registered_teams from request query params (comma-separated paths)."""
    raw = request.args.get('registered_teams', '')
    if raw:
        return [p.strip() for p in raw.split(',') if p.strip()]
    teams_dict = getattr(app, 'registered_teams', None)
    if teams_dict:
        return [p for p in teams_dict.values() if isinstance(p, str) and p.strip()]
    return []
@app.route("/api/npc_tools", methods=["GET"])
def get_npc_tools():
    """
    Returns the resolved tool set for an NPC, sourced from its config
    (jinxes + mcp_servers + python tools). Also returns available team servers.
    """
    npc_name_param = request.args.get("npc")
    team_path_param = request.args.get("team_path")
    current_path_arg = request.args.get("currentPath")
    registered_teams = _parse_registered_teams()
    try:
        from npcpy.npc_compiler import NPC, Team, build_jinx_tool_catalog
        team_obj = None
        if team_path_param and os.path.isdir(team_path_param):
            try:
                team_obj = Team(team_path=team_path_param)
            except Exception as e:
                print(f"[npc_tools] Failed to load team from {team_path_param}: {e}")
        npc_obj = None
        if npc_name_param and team_obj and npc_name_param in team_obj.npcs:
            npc_obj = team_obj.npcs[npc_name_param]
        elif npc_name_param:
            search_dirs = []
            if current_path_arg:
                search_dirs.append(os.path.join(os.path.abspath(current_path_arg), "npc_team"))
            for team_path in registered_teams:
                search_dirs.append(team_path)
            for d in search_dirs:
                npc_file = os.path.join(d, f"{npc_name_param}.npc")
                if os.path.exists(npc_file):
                    try:
                        team_obj = Team(team_path=d)
                        npc_obj = team_obj.npcs.get(npc_name_param)
                    except Exception as e:
                        print(f"[npc_tools] Failed to load team/NPC from {d}: {e}")
                    break
        npc_tools = []
        if npc_obj:
            if not hasattr(app, 'mcp_clients_cache'):
                app.mcp_clients_cache = {}
            try:
                tools_for_llm, tool_executors = npc_obj.resolve_tools(
                    mcp_clients_cache=app.mcp_clients_cache
                )
                for tool_def in tools_for_llm:
                    name = tool_def["function"]["name"]
                    executor = tool_executors.get(name, {})
                    source = executor.get("type", "unknown")
                    if source == "mcp" and executor.get("client"):
                        source = f"mcp:{executor['client'].server_script_path or 'unknown'}"
                    npc_tools.append({
                        "name": name,
                        "description": tool_def["function"].get("description", ""),
                        "source": source,
                        "enabled": True,
                    })
            except Exception as e:
                print(f"[npc_tools] Error resolving tools: {e}")
                traceback.print_exc()
        team_servers = []
        if team_obj and hasattr(team_obj, "mcp_servers"):
            for srv in (team_obj.mcp_servers or []):
                if isinstance(srv, str):
                    team_servers.append({"path": srv, "enabled": False})
                elif isinstance(srv, dict):
                    label = srv.get("path") or srv.get("url") or f"{srv.get('command', '')} {' '.join(srv.get('args', []))}"
                    team_servers.append({**srv, "label": label, "enabled": False})
        return jsonify({
            "npc_tools": npc_tools,
            "team_servers": team_servers,
            "error": None,
        })
    except Exception as e:
        print(f"[npc_tools] Error: {traceback.format_exc()}")
        return jsonify({"error": str(e), "npc_tools": [], "team_servers": []}), 500
@app.route("/api/mcp/server/resolve", methods=["GET"])
def api_mcp_resolve():
    current_path = request.args.get("currentPath")
    explicit = request.args.get("serverPath")
    try:
        resolved = resolve_mcp_server_path(current_path=current_path, explicit_path=explicit)
        return jsonify({"serverPath": resolved, "error": None})
    except Exception as e:
        return jsonify({"serverPath": None, "error": str(e)}), 500
@app.route("/api/mcp/server/start", methods=["POST"])
def api_mcp_start():
    data = request.get_json() or {}
    current_path = data.get("currentPath")
    explicit = data.get("serverPath")
    env_vars = data.get("envVars")
    try:
        if _is_command_string(explicit or ""):
            server_path = (explicit or "").strip()
        else:
            server_path = resolve_mcp_server_path(current_path=current_path, explicit_path=explicit)
        result = mcp_server_manager.start(server_path, env_vars=env_vars)
        return jsonify({**result, "error": None})
    except Exception as e:
        print(f"Error starting MCP server: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route("/api/mcp/server/stop", methods=["POST"])
def api_mcp_stop():
    data = request.get_json() or {}
    explicit = data.get("serverPath")
    if not explicit:
        return jsonify({"error": "serverPath is required to stop a server."}), 400
    try:
        result = mcp_server_manager.stop(explicit)
        return jsonify({**result, "error": None})
    except Exception as e:
        print(f"Error stopping MCP server: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route("/api/mcp/server/status", methods=["GET"])
def api_mcp_status():
    explicit = request.args.get("serverPath")
    current_path = request.args.get("currentPath")
    try:
        if explicit:
            if _is_command_string(explicit):
                result = mcp_server_manager.status(explicit.strip())
            else:
                result = mcp_server_manager.status(explicit)
        else:
            resolved = resolve_mcp_server_path(current_path=current_path, explicit_path=explicit)
            result = mcp_server_manager.status(resolved)
        return jsonify({**result, "running": result.get("status") == "running", "all": mcp_server_manager.running(), "error": None})
    except Exception as e:
        print(f"Error checking MCP server status: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route("/api/image_models", methods=["GET"]) 
def get_image_models_api():
    """
    API endpoint to retrieve available image generation models.
    """
    current_path = request.args.get("currentPath")
    try:
        image_models = get_available_image_models(current_path)
        print('image models', image_models)
        return jsonify({"models": image_models, "error": None})
    except Exception as e:
        print(f"Error getting available image models: {str(e)}")
        traceback.print_exc()
        return jsonify({"models": [], "error": str(e)}), 500
@app.route("/api/generate_video", methods=["POST"])
def generate_video_api():
    """
    API endpoint for video generation.
    """
    try:
        data = request.get_json()
        prompt = data.get("prompt", "")
        model = data.get("model", "veo-3.1-generate-preview")
        provider = data.get("provider", "gemini")
        duration = data.get("duration", 5)
        output_dir = data.get("output_dir")
        negative_prompt = data.get("negative_prompt", "")
        reference_image = data.get("reference_image")
        if not prompt:
            return jsonify({"error": "Prompt is required"}), 400
        if output_dir:
            save_dir = os.path.expanduser(output_dir)
        else:
            save_dir = get_videos_dir()
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"video_{timestamp}.mp4"
        output_path = os.path.join(save_dir, output_filename)
        num_frames = int(duration * 25) if provider == "diffusers" else 25
        print(f"Generating video with model={model}, provider={provider}, duration={duration}s")
        result = gen_video(
            prompt=prompt,
            model=model,
            provider=provider,
            output_path=output_path,
            num_frames=num_frames,
            negative_prompt=negative_prompt,
        )
        if result and "output" in result:
            video_path = output_path
            if os.path.exists(video_path):
                with open(video_path, "rb") as f:
                    video_data = f.read()
                video_base64 = base64.b64encode(video_data).decode("utf-8")
                return jsonify({
                    "success": True,
                    "video_path": video_path,
                    "video_base64": f"data:video/mp4;base64,{video_base64}",
                    "message": result.get("output", "Video generated successfully")
                })
            else:
                return jsonify({"error": "Video file was not created"}), 500
        else:
            return jsonify({"error": result.get("output", "Video generation failed")}), 500
    except Exception as e:
        print(f"Error generating video: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route("/api/video_models", methods=["GET"])
def get_video_models_api():
    """
    API endpoint to retrieve available video generation models.
    """
    video_models = [
        {"value": "veo-3.1-generate-preview", "display_name": "Veo 3.1 | gemini", "provider": "gemini", "max_duration": 8},
        {"value": "veo-3.1-fast-generate-preview", "display_name": "Veo 3.1 Fast | gemini", "provider": "gemini", "max_duration": 8},
        {"value": "veo-2.0-generate-001", "display_name": "Veo 2 | gemini", "provider": "gemini", "max_duration": 8},
        {"value": "damo-vilab/text-to-video-ms-1.7b", "display_name": "ModelScope 1.7B (Local) | diffusers", "provider": "diffusers", "max_duration": 4},
    ]
    return jsonify({"models": video_models, "error": None})
@app.route("/api/text_predict", methods=["POST"])
def text_predict():
    data = request.json
    stream_id = _setup_stream(data)
    print(f"Starting text prediction stream with ID: {stream_id}")
    text_content = data.get("text_content", "")
    cursor_position = data.get("cursor_position", len(text_content))
    current_path = data.get("currentPath")
    model = data.get("model")
    provider = data.get("provider")
    context_type = data.get("context_type", "general")
    file_path = data.get("file_path")
    if current_path:
        load_project_env(current_path)
    text_before_cursor = text_content[:cursor_position]
    if context_type == 'code':
        prompt_for_llm = f"You are an AI code completion assistant. Your task is to complete the provided code snippet.\nYou MUST ONLY output the code that directly completes the snippet.\nDO NOT include any explanations, comments, or additional text.\nDO NOT wrap the completion in markdown code blocks.\n\nHere is the code context where the completion should occur (file: {file_path or 'unknown'}):\n\n{text_before_cursor}\n\nPlease provide the completion starting from the end of the last line shown.\n"
        system_prompt = "You are an AI code completion assistant. Only provide code. Do not add explanations or any other text."
    elif context_type == 'chat':
        prompt_for_llm = f"You are an AI chat assistant. Your task is to provide a natural and helpful completion to the user's ongoing message.\nYou MUST ONLY output the text that directly completes the message.\nDO NOT include any explanations or additional text.\n\nHere is the message context where the completion should occur:\n\n{text_before_cursor}\n\nPlease provide the completion starting from the end of the last line shown.\n"
        system_prompt = "You are an AI chat assistant. Only provide natural language completion. Do not add explanations or any other text."
    else:
        prompt_for_llm = f"You are an AI text completion assistant. Your task is to provide a natural and helpful completion to the user's ongoing text.\nYou MUST ONLY output the text that directly completes the snippet.\nDO NOT include any explanations or additional text.\n\nHere is the text context where the completion should occur:\n\n{text_before_cursor}\n\nPlease provide the completion starting from the end of the last line shown.\n"
        system_prompt = "You are an AI text completion assistant. Only provide natural language completion. Do not add explanations or any other text."
    npc_object = None
    messages_for_llm = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_for_llm}
    ]
    def event_stream_text_predict(current_stream_id):
        complete_prediction = []
        try:
            stream_response_generator = get_llm_response(
                prompt_for_llm,
                messages=messages_for_llm,
                model=model,
                provider=provider,
                npc=npc_object,
                stream=True,
                think=False,
            )
            if isinstance(stream_response_generator, dict) and 'response' in stream_response_generator:
                stream_generator = stream_response_generator['response']
            else:
                output_content = ""
                if isinstance(stream_response_generator, dict) and 'output' in stream_response_generator:
                    output_content = stream_response_generator['output']
                elif isinstance(stream_response_generator, str):
                    output_content = stream_response_generator
                yield f"data: {json.dumps({'choices': [{'delta': {'content': output_content}}]})}\n\n"
                yield f"data: [DONE]\n\n"
                return
            for response_chunk in stream_generator:
                with cancellation_lock:
                    if cancellation_flags.get(current_stream_id, False):
                        print(f"Cancellation flag triggered for {current_stream_id}. Breaking loop.")
                        break
                chunk_content = ""
                if "hf.co" in model or provider == 'ollama':
                    msg = response_chunk["message"] if "message" in response_chunk else None
                    if msg:
                        chunk_content = msg.get("content", "") or msg.get("thinking", "")
                else:
                    chunk_content = "".join(choice.delta.content for choice in response_chunk.choices if choice.delta.content is not None)
                print(chunk_content, end='')
                if chunk_content:
                    complete_prediction.append(chunk_content)
                    yield f"data: {json.dumps({'choices': [{'delta': {'content': chunk_content}}]})}\n\n"
        except Exception as e:
            print(f"\nAn exception occurred during text prediction streaming for {current_stream_id}: {e}")
            traceback.print_exc()
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            print(f"\nText prediction stream {current_stream_id} finished.")
            yield f"data: [DONE]\n\n"
            _cleanup_stream(current_stream_id)
    return Response(event_stream_text_predict(stream_id), mimetype="text/event-stream", headers={
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no'
    })
@app.route("/api/stream", methods=["POST"])
def stream():
    data = request.json
    stream_id = _setup_stream(data)
    print(f"Starting stream with ID: {stream_id}")
    commandstr = data.get("commandstr")
    conversation_id = data.get("conversationId")
    if not conversation_id:
        return jsonify({"error": "conversationId is required"}), 400
    model = data.get("model", None)
    provider = data.get("provider", None)
    print(f"🔍 Stream request - model: {model}, provider from request: {provider}")
    if provider is None and model:
        resolved_provider = available_models.get(model) or lookup_provider(model)
        if resolved_provider:
            provider = resolved_provider
            print(f"🔍 Provider looked up from available_models/lookup_provider: {provider}")
    npc_name = data.get("npc", None)
    npc_source = data.get("npcSource", "global")
    current_path = data.get("currentPath")
    registered_teams = data.get("registered_teams", [])
    print(f"[STREAM] registered_teams received: {registered_teams}")
    is_resend = data.get("isResend", False)
    parent_message_id = data.get("parentMessageId", None)
    frontend_user_message_id = data.get("userMessageId", None)
    frontend_assistant_message_id = data.get("assistantMessageId", None)
    user_parent_message_id = data.get("userParentMessageId", None)
    extract_memories = bool(data.get("extractMemories", True))
    params = {}
    if data.get("temperature") is not None:
        params["temperature"] = data.get("temperature")
    if data.get("top_p") is not None:
        params["top_p"] = data.get("top_p")
    if data.get("top_k") is not None:
        params["top_k"] = data.get("top_k")
    if data.get("max_tokens") is not None:
        params["max_tokens"] = data.get("max_tokens")
    disable_thinking = data.get("disableThinking", False)
    params = params if params else None
    if current_path:
        loaded_vars = load_project_env(current_path)
        print(f"Loaded project env variables for stream request: {list(loaded_vars.keys())}")
    npc_object = None
    team_object = None
    team = None
    tool_results_for_db = []
    stream_response = {"output": "", "messages": []}
    if npc_name:
        if hasattr(app, 'registered_teams'):
            for team_name, team_object in app.registered_teams.items():
                if hasattr(team_object, 'npcs'):
                    team_npcs = team_object.npcs
                    if isinstance(team_npcs, dict):
                        if npc_name in team_npcs:
                            npc_object = team_npcs[npc_name]
                            team = team_name 
                            npc_object.team = team_object
                            print(f"Found NPC {npc_name} in registered team {team_name}")
                            break
                    elif isinstance(team_npcs, list):
                        for npc in team_npcs:
                            if hasattr(npc, 'name') and npc.name == npc_name:
                                npc_object = npc
                                team = team_name  
                                npc_object.team = team_object
                                print(f"Found NPC {npc_name} in registered team {team_name}")
                                break
                if not npc_object and hasattr(team_object, 'forenpc') and hasattr(team_object.forenpc, 'name'):
                    if team_object.forenpc.name == npc_name:
                        npc_object = team_object.forenpc
                        npc_object.team = team_object
                        team = team_name
                        print(f"Found NPC {npc_name} as forenpc in team {team_name}")
                        break
                if npc_object:
                    break
        if not npc_object and hasattr(app, 'registered_npcs') and npc_name in app.registered_npcs:
            npc_object = app.registered_npcs[npc_name]
            print(f"Found NPC {npc_name} in registered NPCs (no specific team)")
            team_object = Team(team_path=npc_object.npc_directory)
            npc_object.team = team_object
        if not npc_object and registered_teams:
            print(f"[STREAM] Searching for {npc_name} in {len(registered_teams)} registered teams")
            for team_path in registered_teams:
                if not team_path or not os.path.isdir(team_path):
                    print(f"[STREAM] Skipping invalid team path: {team_path}")
                    continue
                try:
                    team_obj = Team(team_path=team_path)
                    print(f"[STREAM] Loaded team {team_obj.name} from {team_path} with {len(team_obj.jinxes_dict)} jinxes, npcs: {list(team_obj.npcs.keys())}")
                    if npc_name in team_obj.npcs:
                        npc_object = team_obj.npcs[npc_name]
                        team_object = team_obj
                        print(f"[STREAM] Found NPC {npc_name} in registered team {team_path}")
                        print(f"[STREAM] NPC {npc_name} jinxes_spec: {getattr(npc_object, 'jinxes_spec', None)}")
                        print(f"[STREAM] NPC {npc_name} jinxes_dict keys: {list(npc_object.jinxes_dict.keys())}")
                        break
                    elif hasattr(team_obj, 'forenpc') and team_obj.forenpc and team_obj.forenpc.name == npc_name:
                        npc_object = team_obj.forenpc
                        team_object = team_obj
                        print(f"[STREAM] Found NPC {npc_name} as forenpc in registered team {team_path}")
                        break
                except Exception as e:
                    print(f"[STREAM] Error loading registered team {team_path}: {e}")
                    traceback.print_exc()
                    continue
        if not npc_object:
            npc_object = load_npc_by_name_and_source(npc_name, npc_source, current_path)
            if not npc_object and npc_source == 'project':
                print(f"NPC {npc_name} not found in project directory, trying global...")
                npc_object = load_npc_by_name_and_source(npc_name, 'global')
            if npc_object and hasattr(npc_object, 'npc_directory') and npc_object.npc_directory:
                team_directory = npc_object.npc_directory
                if os.path.exists(team_directory):
                    team_object = Team(team_path=team_directory)
                    print('team', team_object)
                else:
                    team_object = Team(npcs=[npc_object])
                    team_object.name = os.path.basename(team_directory) if team_directory else f"{npc_name}_team"
                    npc_object.team = team_object
                    print('team', team_object)
                team_name = team_object.name
                if not hasattr(app, 'registered_teams'):
                    app.registered_teams = {}
                app.registered_teams[team_name] = team_object
                team = team_name
                print(f"Created and registered team '{team_name}' with NPC {npc_name}")
            if npc_object:
                npc_object.team = team_object
                print(f"Successfully loaded NPC {npc_name} from {npc_source} directory")
            else:
                print(f"Warning: Could not load NPC {npc_name}")
            if npc_object:
                print(f"Successfully loaded NPC {npc_name} from {npc_source} directory")
            else:
                print(f"Warning: Could not load NPC {npc_name}")
    attachments = data.get("attachments", [])
    images: list = []
    attachment_paths_for_llm: list = []
    attachments_for_db: list = []
    print(f"[DEBUG] Received attachments: {attachments}")
    if attachments:
        print(f"[DEBUG] Processing {len(attachments)} attachments")
        for attachment in attachments:
            try:
                file_name = attachment["name"]
                extension = file_name.split(".")[-1].upper() if "." in file_name else ""
                extension_mapped = extension_map.get(extension, "others")
                file_path = None
                file_content_bytes = None
                if "path" in attachment and attachment["path"]:
                    file_path = attachment["path"]
                    if os.path.exists(file_path):
                        with open(file_path, "rb") as f:
                            file_content_bytes = f.read()
                    else:
                        print(f"Warning: Attachment file does not exist: {file_path}")
                        if "data" in attachment and attachment["data"]:
                            file_content_bytes = base64.b64decode(attachment["data"])
                            import tempfile
                            temp_dir = tempfile.mkdtemp()
                            file_path = os.path.join(temp_dir, file_name)
                            with open(file_path, "wb") as f:
                                f.write(file_content_bytes)
                elif "data" in attachment and attachment["data"]:
                    file_content_bytes = base64.b64decode(attachment["data"])
                    import tempfile
                    temp_dir = tempfile.mkdtemp()
                    file_path = os.path.join(temp_dir, file_name)
                    with open(file_path, "wb") as f:
                        f.write(file_content_bytes)
                if not file_path or file_content_bytes is None:
                    print(f"Warning: Skipping attachment {file_name} - no valid path or data")
                    continue
                attachment_paths_for_llm.append(file_path)
                if extension_mapped == "images":
                    images.append(file_path)
                attachments_for_db.append({
                    "name": file_name,
                    "path": file_path,
                    "type": extension_mapped,
                    "data": file_content_bytes,
                    "size": len(file_content_bytes) if file_content_bytes else 0
                })
            except Exception as e:
                print(f"Error processing attachment {attachment.get('name', 'N/A')}: {e}")
                traceback.print_exc()
    print(f"[DEBUG] After processing - images: {images}, attachment_paths_for_llm: {attachment_paths_for_llm}")
    explicit_messages = data.get("messages")
    if explicit_messages:
        messages = explicit_messages
        print(f"[DEBUG] Using explicit messages from frontend ({len(messages)} messages)")
    else:
        messages = []
    messages = clean_messages_for_llm(messages)
    exe_mode = data.get('executionMode','chat')
    if exe_mode == 'chat' and npc_object is not None and hasattr(npc_object, 'jinxes_dict'):
        npc_object.jinxes_dict = {}
    messages = ensure_system_prompt(messages, npc=npc_object, tool_capable=(exe_mode == 'tool_agent'))
    stream_response = {"output": "", "messages": messages}
    tool_args = {}
    if exe_mode == 'tool_agent' and npc_object is not None:
        if hasattr(npc_object, 'tools') and npc_object.tools:
            if isinstance(npc_object.tools, list) and callable(npc_object.tools[0]):
                tools_schema, tool_map = auto_tools(npc_object.tools)
                tool_args['tools'] = tools_schema
                tool_args['tool_map'] = tool_map
            else:
                tool_args['tools'] = npc_object.tools
                if hasattr(npc_object, 'tool_map') and npc_object.tool_map:
                    tool_args['tool_map'] = npc_object.tool_map
        elif hasattr(npc_object, 'tool_map') and npc_object.tool_map:
            tool_args['tool_map'] = npc_object.tool_map
        if 'tools' in tool_args and tool_args['tools']:
            tool_args['tool_choice'] = {"type": "auto"}
    api_url = None
    if npc_object is not None:
        try:
            api_url = npc_object.api_url if npc_object.api_url else None
        except AttributeError:
            api_url = None
    thinking_kwargs = {}
    if disable_thinking:
        if provider in ('ollama',):
            thinking_kwargs['think'] = False
        else:
            thinking_kwargs['reasoning_effort'] = 'none'
    elif provider in ('anthropic',):
        thinking_kwargs['thinking'] = {"type": "enabled", "budget_tokens": 10000}
        if params and 'temperature' in params:
            del params['temperature']
    if exe_mode == 'chat':
        print(f"[DEBUG] Calling get_llm_response with images={images}, attachments={attachment_paths_for_llm}")
        stream_response = get_llm_response(
            commandstr,
            messages=messages,
            images=images if images else None,
            model=model,
            provider=provider,
            npc=npc_object,
            api_url = api_url,
            team=team_object,
            stream=True,
            attachments=attachment_paths_for_llm if attachment_paths_for_llm else None,
            include_usage=True,
            **(params or {}),
            **thinking_kwargs,
        )
        messages = stream_response.get('messages', messages)
    elif exe_mode == 'tool_agent':
        selected_mcp_tools_from_request = data.get("selectedMcpTools", [])
        if not hasattr(app, 'mcp_clients_cache'):
            app.mcp_clients_cache = {}
        tools_for_llm = []
        tool_executors = {}
        if npc_object and hasattr(npc_object, 'resolve_tools'):
            tools_for_llm, tool_executors = npc_object.resolve_tools(
                mcp_clients_cache=app.mcp_clients_cache
            )
        extra_paths = []
        if "mcpServerPaths" in data and isinstance(data["mcpServerPaths"], list):
            extra_paths = [p for p in data["mcpServerPaths"] if p]
        elif data.get("mcpServerPath"):
            extra_paths = [data.get("mcpServerPath")]
        for extra_mcp_path in extra_paths:
            resolved_path = resolve_mcp_server_path(current_path, extra_mcp_path, False)
            if not resolved_path:
                continue
            client = app.mcp_clients_cache.get(resolved_path)
            if client and not client.is_connected():
                client.disconnect_sync()
                app.mcp_clients_cache.pop(resolved_path, None)
                client = None
            if not client:
                client = MCPClientNPC()
                if client.connect_sync(resolved_path):
                    app.mcp_clients_cache[resolved_path] = client
                else:
                    client = None
            if client:
                existing_names = {td["function"]["name"] for td in tools_for_llm}
                for t in client.available_tools_llm:
                    name = t["function"]["name"]
                    if name not in existing_names:
                        tools_for_llm.append(t)
                        tool_executors[name] = {
                            "type": "mcp",
                            "client": client,
                            "tool_func": client.tool_map.get(name),
                        }
                        existing_names.add(name)
        if npc_object and hasattr(npc_object, "jinx_tool_catalog"):
            jinx_tool_catalog = npc_object.jinx_tool_catalog or {}
            existing_names = {td["function"]["name"] for td in tools_for_llm}
            for t in jinx_tool_catalog.values():
                name = t["function"]["name"]
                if name not in existing_names:
                    tools_for_llm.append(t)
                    tool_executors[name] = {
                        "type": "jinx",
                        "jinx": npc_object.jinxes_dict.get(name),
                    }
                    existing_names.add(name)
        if selected_mcp_tools_from_request:
            tools_for_llm = [t for t in tools_for_llm if t["function"]["name"] in selected_mcp_tools_from_request]
            allowed = set(selected_mcp_tools_from_request)
            tool_executors = {k: v for k, v in tool_executors.items() if k in allowed}
        print(f"[MCP] resolved {len(tools_for_llm)} tools: {[t['function']['name'] for t in tools_for_llm]}")
        if not hasattr(app, 'mcp_clients'):
            app.mcp_clients = {}
        state_key = f"{conversation_id}_{npc_name or 'default'}"
        app._last_mcp_state_key = state_key
        if state_key not in app.mcp_clients:
            app.mcp_clients[state_key] = {"client": None, "server_path": None, "messages": messages}
        app.mcp_clients[state_key].setdefault("messages", messages)
        request_messages = clean_messages_for_llm(messages)
        if not request_messages:
            request_messages = []
        if not any(m.get('role') == 'system' for m in request_messages):
            system_prompt = npc_object.get_system_prompt(tool_capable=True) if npc_object else "You are a helpful assistant with access to tools."
            request_messages.insert(0, {'role': 'system', 'content': system_prompt})
        messages = request_messages
        def stream_mcp_sse():
            nonlocal messages
            iteration = 0
            prompt = commandstr
            total_input_tokens = 0
            total_output_tokens = 0
            while iteration < 10:
                iteration += 1
                print(f"[MCP] iteration {iteration} prompt len={len(prompt)}")
                print(f"[MCP] tools_for_llm: {[t['function']['name'] for t in tools_for_llm]}")
                agent_context = f'''The user's working directory is {current_path}
IMPORTANT AGENT BEHAVIOR:
- If a tool call fails or returns an error, DO NOT give up. Try alternative approaches.
- If a file is not found, search for it using different paths or patterns.
- If one method doesn't work, try another method to accomplish the task.
- Keep working on the task until it is complete or you have exhausted all reasonable options.
- When you encounter errors, explain what went wrong and what you're trying next.'''
                print(f"[MCP DEBUG] Messages for LLM (iteration {iteration}): {json.dumps(messages, indent=2, default=str)[:3000]}")
                call_prompt = prompt if iteration == 1 else ""
                llm_response = get_llm_response_with_handling(
                    prompt=call_prompt,
                    npc=npc_object,
                    model=model,
                    provider=provider,
                    messages=messages,
                    tools=tools_for_llm,
                    stream=True,
                    team=team_object,
                    context=agent_context if iteration == 1 else None,
                    **(params or {}),
                    **thinking_kwargs,
                )
                print('RESPONSE', llm_response)
                stream = llm_response.get("response", [])
                usage = llm_response.get("usage", {})
                total_input_tokens += usage.get("input_tokens", 0) or 0
                total_output_tokens += usage.get("output_tokens", 0) or 0
                collected_content = ""
                collected_tool_calls = []
                agent_tool_call_data = {"id": None, "function_name": None, "arguments": ""}
                last_response_chunk = None
                for response_chunk in stream:
                    last_response_chunk = response_chunk
                    with cancellation_lock:
                        if cancellation_flags.get(stream_id, False):
                            yield {"type": "interrupt"}
                            return
                    if "hf.co" in model or provider == 'ollama':
                        msg = getattr(response_chunk, "message", None) or (response_chunk.get("message", {}) if hasattr(response_chunk, "get") else {})
                        chunk_content = getattr(msg, "content", None) or (msg.get("content") if hasattr(msg, "get") else "") or ""
                        reasoning_content = getattr(msg, "thinking", None) or (msg.get("thinking") if hasattr(msg, "get") else None)
                        tool_calls = getattr(msg, "tool_calls", None) or (msg.get("tool_calls") if hasattr(msg, "get") else None)
                        if tool_calls:
                            for tool_call in tool_calls:
                                tc_id = getattr(tool_call, "id", None) or (tool_call.get("id") if hasattr(tool_call, "get") else None)
                                tc_func = getattr(tool_call, "function", None) or (tool_call.get("function") if hasattr(tool_call, "get") else None)
                                if tc_func:
                                    tc_name = getattr(tc_func, "name", None) or (tc_func.get("name") if hasattr(tc_func, "get") else None)
                                    tc_args = getattr(tc_func, "arguments", None) or (tc_func.get("arguments") if hasattr(tc_func, "get") else None)
                                    if tc_name:
                                        arg_str = tc_args
                                        if isinstance(arg_str, dict):
                                            arg_str = json.dumps(arg_str)
                                        elif arg_str is None:
                                            arg_str = "{}"
                                        collected_tool_calls.append({
                                            "id": tc_id or f"call_{len(collected_tool_calls)}",
                                            "type": "function",
                                            "function": {"name": tc_name, "arguments": arg_str}
                                        })
                        if chunk_content:
                            collected_content += chunk_content
                        created_at = getattr(response_chunk, "created_at", None) or (response_chunk.get("created_at") if hasattr(response_chunk, "get") else None)
                        model_name = getattr(response_chunk, "model", None) or (response_chunk.get("model") if hasattr(response_chunk, "get") else model)
                        msg_role = getattr(msg, "role", None) or (msg.get("role") if hasattr(msg, "get") else "assistant")
                        done_reason = getattr(response_chunk, "done_reason", None) or (response_chunk.get("done_reason") if hasattr(response_chunk, "get") else None)
                        chunk_data = {
                            "id": None,
                            "object": None,
                            "created": str(created_at) if created_at else datetime.datetime.now().isoformat(),
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "content": chunk_content,
                                        "role": msg_role,
                                        "reasoning_content": reasoning_content
                                    },
                                    "finish_reason": done_reason
                                }
                            ]
                        }
                        yield chunk_data
                    elif hasattr(response_chunk, "choices") and response_chunk.choices:
                        delta = response_chunk.choices[0].delta
                        if hasattr(delta, "content") and delta.content:
                            collected_content += delta.content
                            chunk_data = {
                                "id": getattr(response_chunk, "id", None),
                                "object": getattr(response_chunk, "object", None),
                                "created": getattr(response_chunk, "created", datetime.datetime.now().strftime('YYYY-DD-MM-HHMMSS')),
                                "model": getattr(response_chunk, "model", model),
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {
                                            "content": delta.content,
                                            "role": "assistant"
                                        },
                                        "finish_reason": None
                                    }
                                ]
                            }
                            yield chunk_data
                        if hasattr(delta, "tool_calls") and delta.tool_calls:
                            for tool_call_delta in delta.tool_calls:
                                idx = getattr(tool_call_delta, "index", 0)
                                while len(collected_tool_calls) <= idx:
                                    collected_tool_calls.append({
                                        "id": "",
                                        "type": "function",
                                        "function": {"name": "", "arguments": ""}
                                    })
                                if getattr(tool_call_delta, "id", None):
                                    collected_tool_calls[idx]["id"] = tool_call_delta.id
                                if hasattr(tool_call_delta, "function"):
                                    fn = tool_call_delta.function
                                    if getattr(fn, "name", None):
                                        collected_tool_calls[idx]["function"]["name"] = fn.name
                                    if getattr(fn, "arguments", None):
                                        collected_tool_calls[idx]["function"]["arguments"] += fn.arguments
                if not collected_tool_calls:
                    print("[MCP] no tool calls, finishing streaming loop")
                    if last_response_chunk is not None:
                        chunk_usage = getattr(last_response_chunk, 'usage', None)
                        if chunk_usage is None and isinstance(last_response_chunk, dict):
                            chunk_usage = last_response_chunk.get('usage')
                        if chunk_usage:
                            inp = getattr(chunk_usage, 'prompt_tokens', None) or (chunk_usage.get('prompt_tokens', 0) if isinstance(chunk_usage, dict) else 0)
                            out = getattr(chunk_usage, 'completion_tokens', None) or (chunk_usage.get('completion_tokens', 0) if isinstance(chunk_usage, dict) else 0)
                            if inp: total_input_tokens += inp
                            if out: total_output_tokens += out
                        prompt_eval = getattr(last_response_chunk, 'prompt_eval_count', None)
                        eval_count = getattr(last_response_chunk, 'eval_count', None)
                        if prompt_eval:
                            total_input_tokens += prompt_eval
                        if eval_count:
                            total_output_tokens += eval_count
                    break
                print(f"[MCP] collected tool calls: {[tc['function']['name'] for tc in collected_tool_calls]}")
                serialized_tool_calls = []
                for tc in collected_tool_calls:
                    parsed_args = tc["function"]["arguments"]
                    if isinstance(parsed_args, dict):
                        args_for_message = json.dumps(parsed_args)
                    else:
                        args_for_message = str(parsed_args)
                    serialized_tool_calls.append({
                        "id": tc["id"],
                        "type": tc["type"],
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": args_for_message
                        }
                    })
                messages.append({
                    "role": "assistant",
                    "content": collected_content,
                    "tool_calls": serialized_tool_calls
                })
                yield {
                    "type": "tool_execution_start",
                    "tool_calls": [
                        {
                            "name": tc["function"]["name"],
                            "id": tc["id"],
                            "function": {
                                "name": tc["function"]["name"],
                                "arguments": tc["function"].get("arguments", "")
                            }
                        } for tc in collected_tool_calls
                    ]
                }
                tool_results = []
                session_grants = {}
                for tc in collected_tool_calls:
                    with cancellation_lock:
                        if cancellation_flags.get(stream_id, False):
                            yield {"type": "interrupt"}
                            return
                    tool_name = tc["function"]["name"]
                    tool_args = tc["function"]["arguments"]
                    tool_id = tc["id"]
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args) if tool_args.strip() else {}
                        except json.JSONDecodeError:
                            tool_args = {}
                    executor = tool_executors.get(tool_name)
                    cmd_key = _build_command_key(tool_name, tool_args)
                    perm = _check_tool_permission(tool_name, tool_args, executor, session_grants, team_object)
                    if perm == "deny":
                        tool_content = f"EPERM: Tool '{tool_name}' is denied by permission settings."
                        messages.append({"role": "tool", "tool_call_id": tool_id, "name": tool_name, "content": tool_content})
                        tool_results.append({"name": tool_name, "tool_call_id": tool_id, "content": tool_content})
                        yield {"type": "tool_error", "name": tool_name, "id": tool_id, "error": tool_content}
                        continue
                    if perm == "ask":
                        request_id = f"perm_{stream_id}_{tool_id}_{uuid.uuid4().hex[:8]}"
                        preview = json.dumps(tool_args, default=str)
                        if len(preview) > 500:
                            preview = preview[:500] + "..."
                        yield {
                            "type": "permission_request",
                            "request_id": request_id,
                            "tool_name": tool_name,
                            "command_key": cmd_key,
                            "args_preview": preview,
                        }
                        decision = _wait_for_permission_response(request_id, timeout=120)
                        if decision is None:
                            decision = "No"
                        allowed = _apply_permission_decision(decision, tool_name, tool_args, executor, session_grants, team_object)
                        if not allowed:
                            tool_content = f"EPERM: User denied execution of '{tool_name}'"
                            messages.append({"role": "tool", "tool_call_id": tool_id, "name": tool_name, "content": tool_content})
                            tool_results.append({"name": tool_name, "tool_call_id": tool_id, "content": tool_content})
                            yield {"type": "tool_error", "name": tool_name, "id": tool_id, "error": tool_content}
                            continue
                    print(f"[MCP] tool_start {tool_name} args={tool_args}")
                    yield {"type": "tool_start", "name": tool_name, "id": tool_id, "args": tool_args}
                    try:
                        tool_content = ""
                        if executor:
                            if executor["type"] == "jinx":
                                jinx_obj = executor["jinx"]
                                try:
                                    jinx_ctx = jinx_obj.execute(
                                        input_values=tool_args if isinstance(tool_args, dict) else {},
                                        npc=npc_object
                                    )
                                    tool_content = str(jinx_ctx.get('output', '')) if isinstance(jinx_ctx, dict) else str(jinx_ctx)
                                except Exception as e:
                                    tool_content = f"Jinx execution error: {str(e)}"
                            elif executor["type"] == "mcp":
                                try:
                                    tool_func = executor["tool_func"]
                                    print(f"[MCP] Calling tool_func for {tool_name}")
                                    result = tool_func(**(tool_args if isinstance(tool_args, dict) else {}))
                                    print(f"[MCP] Raw result type: {type(result)}, value: {result}")
                                    if hasattr(result, 'content'):
                                        if result.content and len(result.content) > 0:
                                            tool_content = str(result.content[0].text)
                                        else:
                                            tool_content = str(result)
                                    else:
                                        tool_content = str(result) if result is not None else "Tool returned no result"
                                    print(f"[MCP] Final tool_content: {tool_content}")
                                except Exception as mcp_e:
                                    print(f"[MCP] Tool exception: {mcp_e}")
                                    traceback.print_exc()
                                    tool_content = f"MCP tool error: {str(mcp_e)}"
                            elif executor["type"] == "python":
                                try:
                                    tool_content = str(executor["func"](**(tool_args if isinstance(tool_args, dict) else {})))
                                except Exception as py_e:
                                    tool_content = f"Python tool error: {str(py_e)}"
                        else:
                            tool_content = f"Tool '{tool_name}' not found in resolved tools"
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "name": tool_name,
                            "content": tool_content
                        })
                        tool_results.append({
                            "name": tool_name,
                            "tool_call_id": tool_id,
                            "content": tool_content
                        })
                        print(f"[MCP] tool_result {tool_name}: {tool_content}")
                        yield {"type": "tool_result", "name": tool_name, "id": tool_id, "result": tool_content}
                    except Exception as e:
                        error_msg = f"Tool execution error: {str(e)}"
                        print(f"[MCP] tool_error {tool_name}: {error_msg}")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_id,
                            "name": tool_name,
                            "content": error_msg
                        })
                        tool_results.append({
                            "name": tool_name,
                            "tool_call_id": tool_id,
                            "content": error_msg
                        })
                        yield {"type": "tool_error", "name": tool_name, "id": tool_id, "error": error_msg}
                tool_results_for_db = tool_results
                prompt = ""
            app.mcp_clients[state_key]["messages"] = messages
            mcp_cost = calculate_cost(model, total_input_tokens, total_output_tokens) if total_input_tokens or total_output_tokens else 0
            if total_input_tokens or total_output_tokens:
                yield {"type": "usage", "input_tokens": total_input_tokens, "output_tokens": total_output_tokens, "cost": mcp_cost or 0}
            return
        stream_response = stream_mcp_sse()
    else:
        stream_response = {"output": f"Unsupported execution mode: {exe_mode}", "messages": messages}
    user_message_filled = ''
    if len(messages) >0:
      if isinstance(messages[-1] .get('content'), list):
          for cont in messages[-1].get('content'):
              txt = cont.get('text')
              if txt is not None:
                  user_message_filled += txt
    def event_stream(current_stream_id):
        complete_response = []
        complete_reasoning = []
        accumulated_tool_calls = []
        dot_count = 0
        interrupted = False
        tool_call_data = {"id": None, "function_name": None, "arguments": ""}
        total_input_tokens = 0
        total_output_tokens = 0
        try:
            if hasattr(stream_response, "__iter__") and not isinstance(stream_response, (dict, str)):
                for chunk in stream_response:
                    with cancellation_lock:
                        if cancellation_flags.get(current_stream_id, False):
                            interrupted = True
                            break
                    if chunk is None:
                        continue
                    if isinstance(chunk, dict):
                        if chunk.get("type") == "interrupt":
                            interrupted = True
                            break
                        yield f"data: {json.dumps(chunk)}\n\n"
                        if chunk.get("choices"):
                            for choice in chunk["choices"]:
                                delta = choice.get("delta", {})
                                content_piece = delta.get("content")
                                if content_piece:
                                    complete_response.append(content_piece)
                                reasoning_piece = delta.get("reasoning_content")
                                if reasoning_piece:
                                    complete_reasoning.append(reasoning_piece)
                        if chunk.get("type") == "tool_call":
                            tc = chunk.get("tool_call", {})
                            if tc.get("id") and tc.get("name"):
                                accumulated_tool_calls.append({
                                    "id": tc.get("id"),
                                    "function_name": tc.get("name"),
                                    "arguments": tc.get("arguments", "")
                                })
                        if chunk.get("type") == "tool_execution_start":
                            for tc in chunk.get("tool_calls", []):
                                accumulated_tool_calls.append({
                                    "id": tc.get("id", ""),
                                    "function_name": tc.get("name", ""),
                                    "arguments": tc.get("function", {}).get("arguments", "") if isinstance(tc.get("function"), dict) else ""
                                })
                        if chunk.get("type") == "tool_result":
                            tool_results_for_db.append({
                                "name": chunk.get("name"),
                                "tool_call_id": chunk.get("id"),
                                "content": chunk.get("result", "")
                            })
                        if chunk.get("type") == "usage":
                            total_input_tokens += chunk.get("input_tokens", 0) or 0
                            total_output_tokens += chunk.get("output_tokens", 0) or 0
                        continue
                    yield f"data: {json.dumps({'choices':[{'delta':{'content': str(chunk), 'role': 'assistant'},'finish_reason':None}]})}\n\n"
            elif isinstance(stream_response, str) :
                print('stream a str and not a gen')
                chunk_data = {
                        "id": None,
                        "object": None,
                        "created": datetime.datetime.now().strftime('YYYY-DD-MM-HHMMSS'),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta":
                                    {
                                        "content": stream_response,
                                        "role": "assistant"
                                  },
                                "finish_reason": 'done'
                            }
                        ]
                    }
                yield f"data: {json.dumps(chunk_data)}\n\n"
            elif isinstance(stream_response, dict) and 'output' in stream_response and isinstance(stream_response.get('output'), str):
                print('stream a str and not a gen')
                chunk_data = {
                        "id": None,
                        "object": None,
                        "created": datetime.datetime.now().strftime('YYYY-DD-MM-HHMMSS'),
                        "model": model,
                        "choices": [
                            {
                                "index": 0,
                                "delta":
                                    {
                                        "content": stream_response.get('output') ,
                                        "role": "assistant"
                                  },
                                "finish_reason": 'done'
                            }
                        ]
                    }
                yield f"data: {json.dumps(chunk_data)}\n\n"
            elif isinstance(stream_response, dict):
                if provider == 'lora':
                    lora_text = stream_response.get('response', stream_response.get('output', ''))
                    if lora_text:
                        complete_response.append(lora_text)
                        chunk_data = {
                            "id": None,
                            "object": None,
                            "created": datetime.datetime.now().strftime('YYYY-DD-MM-HHMMSS'),
                            "model": model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "content": lora_text,
                                        "role": "assistant"
                                    },
                                    "finish_reason": "stop"
                                }
                            ]
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"
                else:
                  if isinstance(stream_response, dict) and stream_response.get('error'):
                      error_msg = stream_response['error']
                      yield f"data: {json.dumps({'choices': [{'delta': {'content': f'Error: {error_msg}', 'role': 'assistant'}, 'finish_reason': 'stop'}]})}\n\n"
                      return
                  last_response_chunk = None
                  for response_chunk in stream_response.get('response', stream_response.get('output')):
                    last_response_chunk = response_chunk
                    with cancellation_lock:
                        if cancellation_flags.get(current_stream_id, False):
                            print(f"Cancellation flag triggered for {current_stream_id}. Breaking loop.")
                            interrupted = True
                            break
                    print('.', end="", flush=True)
                    dot_count += 1
                    if provider == 'llamacpp':
                        chunk_content = ""
                        reasoning_content = None
                        if isinstance(response_chunk, dict) and response_chunk.get("choices"):
                            delta = response_chunk["choices"][0].get("delta", {})
                            chunk_content = delta.get("content", "") or ""
                            reasoning_content = delta.get("reasoning_content")
                        if chunk_content:
                            complete_response.append(chunk_content)
                        if reasoning_content:
                            complete_reasoning.append(reasoning_content)
                        chunk_data = {
                            "id": response_chunk.get("id"),
                            "object": response_chunk.get("object"),
                            "created": response_chunk.get("created"),
                            "model": response_chunk.get("model", model),
                            "choices": [{"index": 0, "delta": {"content": chunk_content, "role": "assistant", "reasoning_content": reasoning_content}, "finish_reason": response_chunk.get("choices", [{}])[0].get("finish_reason")}]
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"
                    elif "hf.co" in model or provider == 'ollama':
                        msg = getattr(response_chunk, "message", None) or response_chunk.get("message", {}) if hasattr(response_chunk, "get") else {}
                        chunk_content = getattr(msg, "content", None) or (msg.get("content") if hasattr(msg, "get") else "") or ""
                        reasoning_content = getattr(msg, "thinking", None) or (msg.get("thinking") if hasattr(msg, "get") else None)
                        tool_calls = getattr(msg, "tool_calls", None) or (msg.get("tool_calls") if hasattr(msg, "get") else None)
                        if tool_calls:
                            for tool_call in tool_calls:
                                tc_id = getattr(tool_call, "id", None) or (tool_call.get("id") if hasattr(tool_call, "get") else None)
                                if tc_id:
                                    tool_call_data["id"] = tc_id
                                tc_func = getattr(tool_call, "function", None) or (tool_call.get("function") if hasattr(tool_call, "get") else None)
                                if tc_func:
                                    tc_name = getattr(tc_func, "name", None) or (tc_func.get("name") if hasattr(tc_func, "get") else None)
                                    if tc_name:
                                        tool_call_data["function_name"] = tc_name
                                    tc_args = getattr(tc_func, "arguments", None) or (tc_func.get("arguments") if hasattr(tc_func, "get") else None)
                                    if tc_args:
                                        arg_val = tc_args
                                        if isinstance(arg_val, dict):
                                            arg_val = json.dumps(arg_val)
                                        tool_call_data["arguments"] += arg_val
                                if tc_id and tc_func and tc_name:
                                    accumulated_tool_calls.append({
                                        "id": tc_id,
                                        "function_name": tc_name,
                                        "arguments": arg_val if tc_args else ""
                                    })
                        if reasoning_content:
                            complete_reasoning.append(reasoning_content)
                        if chunk_content:
                            complete_response.append(chunk_content)
                        created_at = getattr(response_chunk, "created_at", None) or (response_chunk.get("created_at") if hasattr(response_chunk, "get") else None)
                        model_name = getattr(response_chunk, "model", None) or (response_chunk.get("model") if hasattr(response_chunk, "get") else model)
                        msg_role = getattr(msg, "role", None) or (msg.get("role") if hasattr(msg, "get") else "assistant")
                        done_reason = getattr(response_chunk, "done_reason", None) or (response_chunk.get("done_reason") if hasattr(response_chunk, "get") else None)
                        chunk_data = {
                            "id": None, "object": None,
                            "created": str(created_at) if created_at else datetime.datetime.now().isoformat(),
                            "model": model_name,
                            "choices": [{"index": 0, "delta": {"content": chunk_content, "role": msg_role, "reasoning_content": reasoning_content}, "finish_reason": done_reason}]
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"
                    else:
                        chunk_content = ""
                        reasoning_content = ""
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
                                    if tool_call.id and tool_call.function and tool_call.function.name:
                                        accumulated_tool_calls.append({
                                            "id": tool_call.id,
                                            "function_name": tool_call.function.name,
                                            "arguments": tool_call.function.arguments or ""
                                        })
                        for choice in response_chunk.choices:
                            if hasattr(choice.delta, "reasoning_content") and choice.delta.reasoning_content:
                                reasoning_content += choice.delta.reasoning_content
                                complete_reasoning.append(choice.delta.reasoning_content)
                        chunk_content = "".join(choice.delta.content for choice in response_chunk.choices if choice.delta.content is not None)
                        if chunk_content:
                            complete_response.append(chunk_content)
                        chunk_data = {
                            "id": response_chunk.id, "object": response_chunk.object, "created": response_chunk.created, "model": response_chunk.model,
                            "choices": [{"index": choice.index, "delta": {"content": choice.delta.content, "role": choice.delta.role, "reasoning_content": reasoning_content if hasattr(choice.delta, "reasoning_content") else None}, "finish_reason": choice.finish_reason} for choice in response_chunk.choices]
                        }
                        yield f"data: {json.dumps(chunk_data)}\n\n"
                    chunk_usage = getattr(response_chunk, 'usage', None)
                    if chunk_usage is None and isinstance(response_chunk, dict):
                        chunk_usage = response_chunk.get('usage')
                    if chunk_usage:
                        inp = getattr(chunk_usage, 'prompt_tokens', None) or (chunk_usage.get('prompt_tokens', 0) if isinstance(chunk_usage, dict) else 0)
                        out = getattr(chunk_usage, 'completion_tokens', None) or (chunk_usage.get('completion_tokens', 0) if isinstance(chunk_usage, dict) else 0)
                        if inp: total_input_tokens = inp
                        if out: total_output_tokens = out
                    prompt_eval = getattr(response_chunk, 'prompt_eval_count', None)
                    eval_count = getattr(response_chunk, 'eval_count', None)
                    if prompt_eval:
                        total_input_tokens = prompt_eval
                    if eval_count:
                        total_output_tokens = eval_count
                  if last_response_chunk is not None:
                      final_usage = getattr(last_response_chunk, 'usage', None)
                      if final_usage is None and isinstance(last_response_chunk, dict):
                          final_usage = last_response_chunk.get('usage')
                      if final_usage:
                          inp = getattr(final_usage, 'prompt_tokens', None) or (final_usage.get('prompt_tokens', 0) if isinstance(final_usage, dict) else 0)
                          out = getattr(final_usage, 'completion_tokens', None) or (final_usage.get('completion_tokens', 0) if isinstance(final_usage, dict) else 0)
                          if inp: total_input_tokens = inp
                          if out: total_output_tokens = out
                      final_prompt_eval = getattr(last_response_chunk, 'prompt_eval_count', None)
                      final_eval_count = getattr(last_response_chunk, 'eval_count', None)
                      if final_prompt_eval:
                          total_input_tokens = final_prompt_eval
                      if final_eval_count:
                          total_output_tokens = final_eval_count
        except Exception as e:
            print(f"\nAn exception occurred during streaming for {current_stream_id}: {e}")
            traceback.print_exc()
            interrupted = True
        finally:
            print(f"\nStream {current_stream_id} finished. Interrupted: {interrupted}")
            print('\r' + ' ' * dot_count*2 + '\r', end="", flush=True)
            final_response_text = ''.join(complete_response)
            if total_input_tokens or total_output_tokens:
                stream_cost = calculate_cost(model, total_input_tokens, total_output_tokens) if total_input_tokens or total_output_tokens else 0
                yield f"data: {json.dumps({'type': 'usage', 'input_tokens': total_input_tokens, 'output_tokens': total_output_tokens, 'cost': stream_cost or 0})}\n\n"
            yield f"data: {json.dumps({'type': 'message_stop'})}\n\n"
            _cleanup_stream(current_stream_id, getattr(app, '_last_mcp_state_key', None))
    return Response(event_stream(stream_id), mimetype="text/event-stream", headers={
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no'
    })
@app.route("/api/memory/approve", methods=["POST"])
def approve_memories():
    """Approve or reject memories in the local .knowledge.yaml."""
    try:
        data = request.json
        approvals = data.get("approvals", [])
        directory_path = data.get("currentPath", os.getcwd())
        from npcpy.memory.knowledge_store import get_store_for_path
        store = get_store_for_path(directory_path)
        for approval in approvals:
            store.update_memory(
                mem_id=str(approval['memory_id']),
                status=approval['decision'],
                final_memory=approval.get('final_memory'),
            )
        return jsonify({"success": True, "processed": len(approvals)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/api/memory/search", methods=["GET"])
def search_memories():
    """Search memories in the local .knowledge.yaml."""
    try:
        q = request.args.get("q", "")
        npc = request.args.get("npc")
        team = request.args.get("team")
        directory_path = request.args.get("directory_path", os.getcwd())
        status = request.args.get("status")
        limit = int(request.args.get("limit", 50))
        if not q:
            return jsonify({"error": "Query parameter 'q' is required"}), 400
        from npcpy.memory.knowledge_store import KnowledgeStore
        store = KnowledgeStore(directory_path)
        results = store.search_memories(q, limit=limit)
        if npc or team or status:
            filtered = []
            for mem in results:
                if npc and mem.get('npc') != npc:
                    continue
                if team and mem.get('team') != team:
                    continue
                if status and mem.get('status') != status:
                    continue
                filtered.append(mem)
            results = filtered
        return jsonify({"memories": results, "count": len(results)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route("/api/memory/pending", methods=["GET"])
def get_pending_memories():
    """Get memories awaiting approval from the local .knowledge.yaml."""
    try:
        limit = int(request.args.get("limit", 50))
        npc = request.args.get("npc")
        team = request.args.get("team")
        directory_path = request.args.get("directory_path", os.getcwd())
        from npcpy.memory.knowledge_store import KnowledgeStore
        store = KnowledgeStore(directory_path)
        results = store.get_memories(status="pending_approval", limit=limit)
        if npc or team:
            filtered = []
            for mem in results:
                if npc and mem.get('npc') != npc:
                    continue
                if team and mem.get('team') != team:
                    continue
                filtered.append(mem)
            results = filtered
        return jsonify({"memories": results, "count": len(results)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route("/api/memory/scope", methods=["GET"])
def get_memories_by_scope():
    """Get memories for a specific scope from the local .knowledge.yaml."""
    try:
        npc = request.args.get("npc", "")
        team = request.args.get("team", "")
        directory_path = request.args.get("directory_path", os.getcwd())
        status = request.args.get("status")
        from npcpy.memory.knowledge_store import KnowledgeStore
        store = KnowledgeStore(directory_path)
        results = store.get_memories(status=status)
        filtered = []
        for mem in results:
            if npc and mem.get('npc') != npc:
                continue
            if team and mem.get('team') != team:
                continue
            filtered.append(mem)
        return jsonify({"memories": filtered, "count": len(filtered)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route("/api/knowledge/extract", methods=["POST"])
def extract_facts_preview():
    """Extract facts from arbitrary conversation text without storing. Returns facts for review."""
    try:
        data = request.json or {}
        conversation_text = data.get("conversation_text", "")
        conversation_id = data.get("conversation_id", "")
        model = data.get("model", "")
        provider = data.get("provider", "")
        npc_name = data.get("npc", "")
        team_name = data.get("team", "")
        current_path = data.get("currentPath", "")
        if not conversation_text:
            return jsonify({"facts": [], "error": "No conversation_text provided"}), 400
        npc_object = None
        if npc_name and current_path:
            try:
                from npcpy.npc_compiler import load_npcs
                npcs = load_npcs(current_path)
                npc_object = npcs.get(npc_name)
            except Exception:
                pass
        from npcpy.llm_funcs import get_facts, resolve_model_provider
        resolved_model, resolved_provider, _, _ = resolve_model_provider(
            npc=npc_object,
            team=npc_object.team if npc_object else None,
            model=model,
            provider=provider,
        )
        memory_context = ""
        if current_path:
            try:
                from npcpy.memory.knowledge_store import get_store_for_path
                store = get_store_for_path(current_path)
                memory_context = store.build_context(max_memories=10)
            except Exception:
                pass
        from npcpy.llm_funcs import CONVERSATION_RULES
        facts = get_facts(
            conversation_text,
            model=resolved_model,
            provider=resolved_provider,
            npc=npc_object,
            context=memory_context,
            rules=CONVERSATION_RULES,
        )
        return jsonify({
            "facts": facts or [],
            "count": len(facts) if facts else 0,
            "model": resolved_model,
            "provider": resolved_provider,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route("/api/knowledge/extract-and-store", methods=["POST"])
def extract_and_store_facts():
    """Extract facts from conversation text and store as pending memories."""
    try:
        data = request.json or {}
        conversation_text = data.get("conversation_text", "")
        conversation_id = data.get("conversation_id", "")
        model = data.get("model", "")
        provider = data.get("provider", "")
        npc_name = data.get("npc", "")
        team_name = data.get("team", "")
        current_path = data.get("currentPath", "")
        if not conversation_text:
            return jsonify({"facts": [], "error": "No conversation_text provided"}), 400
        npc_object = None
        if npc_name and current_path:
            try:
                from npcpy.npc_compiler import load_npcs
                npcs = load_npcs(current_path)
                npc_object = npcs.get(npc_name)
            except Exception:
                pass
        memories = extract_and_store_memories(
            conversation_text=conversation_text,
            conversation_id=conversation_id,
            npc_name=npc_name or "default",
            team_name=team_name or "default",
            current_path=current_path,
            model=model,
            provider=provider,
            npc_object=npc_object,
        )
        return jsonify({
            "memories": memories or [],
            "count": len(memories) if memories else 0,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
@app.route("/api/interrupt", methods=["POST"])
def interrupt_stream():
    data = request.json
    stream_id_to_cancel = data.get("streamId")
    if not stream_id_to_cancel:
        return jsonify({"error": "streamId is required"}), 400
    with cancellation_lock:
        print(f"Received interruption request for stream ID: {stream_id_to_cancel}")
        cancellation_flags[stream_id_to_cancel] = True
    mcp_state_key = getattr(app, '_last_mcp_state_key', None)
    if mcp_state_key and hasattr(app, 'mcp_clients') and mcp_state_key in app.mcp_clients:
        print(f"[INTERRUPT] Removing MCP state for {mcp_state_key}")
        del app.mcp_clients[mcp_state_key]
    return jsonify({"success": True, "message": f"Interruption for stream {stream_id_to_cancel} registered."})
def _build_command_key(tool_name: str, arguments: dict) -> str:
    """Build a hierarchical command key for permission matching."""
    cmd_key = tool_name
    if tool_name == "sh" and arguments.get("bash_command"):
        parts = arguments["bash_command"].strip().split()
        if parts:
            cmd_key = f"sh:{parts[0]}"
            if len(parts) > 1 and not parts[1].startswith("-"):
                cmd_key = f"sh:{parts[0]} {parts[1]}"
    elif tool_name == "python" and arguments.get("code"):
        cmd_key = "python"
    elif tool_name == "edit_file" and arguments.get("filepath"):
        cmd_key = f"edit_file:{os.path.basename(arguments['filepath'])}"
    elif tool_name == "delegate" and arguments.get("target"):
        cmd_key = f"delegate:{arguments['target']}"
    return cmd_key
def _match_permission(cmd_key: str, rules: dict) -> Optional[str]:
    """Find the most specific matching permission rule."""
    if cmd_key in rules:
        return rules[cmd_key]
    best_match = None
    best_len = 0
    for rule_key in rules:
        if cmd_key.startswith(rule_key):
            nxt = cmd_key[len(rule_key):len(rule_key)+1]
            if nxt in ("", ":", " ") and len(rule_key) > best_len:
                best_len = len(rule_key)
                best_match = rules[rule_key]
    return best_match
def _load_permission_file(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        rules = data.get("rules", data) if isinstance(data, dict) else {}
        return {k: str(v) for k, v in rules.items()}
    except Exception:
        return {}
def _permission_rules_for_team(team_object):
    rules = _load_permission_file(os.path.expanduser("~/.npcsh/npc_team/permissions.yaml"))
    if team_object and getattr(team_object, "team_path", None):
        global_path = os.path.expanduser("~/.npcsh/npc_team")
        if os.path.abspath(team_object.team_path) != os.path.abspath(global_path):
            workspace_path = os.path.join(team_object.team_path, "permissions.yaml")
            if os.path.exists(workspace_path):
                rules.update(_load_permission_file(workspace_path))
    return rules
def _check_tool_permission(tool_name, arguments, executor, session_grants, team_object):
    """Return 'allow', 'deny', or 'ask' for a tool call."""
    cmd_key = _build_command_key(tool_name, arguments)
    # Session grants first.
    session_decision = _match_permission(cmd_key, session_grants)
    if session_decision:
        return session_decision if session_decision != "session" else "allow"
    # Jinx own metadata.
    if executor and executor.get("type") == "jinx" and executor.get("jinx"):
        jinx_perm = executor["jinx"].check_permission()
        if jinx_perm != "ask":
            return jinx_perm
    # Workspace/global rules.
    rules = _permission_rules_for_team(team_object)
    rule = _match_permission(cmd_key, rules)
    if rule:
        return "allow" if rule == "auto" else rule
    # Safe defaults.
    if tool_name in ("chat", "help", "stop"):
        return "allow"
    return "ask"
def _apply_permission_decision(decision, tool_name, arguments, executor, session_grants, team_object):
    """Apply user's decision and persist if always/never. Returns True if allowed."""
    cmd_key = _build_command_key(tool_name, arguments)
    allowed = str(decision).startswith("Yes")
    if "session" in decision.lower():
        session_grants[cmd_key] = "session"
    elif "always" in decision.lower():
        session_grants[cmd_key] = "auto"
        if executor and executor.get("type") == "jinx" and executor.get("jinx"):
            executor["jinx"].set_permission("allow")
        else:
            _save_permission(cmd_key, "auto", team_object)
    elif "never" in decision.lower():
        if executor and executor.get("type") == "jinx" and executor.get("jinx"):
            executor["jinx"].set_permission("deny")
        else:
            _save_permission(cmd_key, "deny", team_object)
    return allowed
def _save_permission(key: str, level: str, team_object):
    team_dir = getattr(team_object, "team_path", None)
    if team_dir:
        global_path = os.path.expanduser("~/.npcsh/npc_team")
        if os.path.abspath(team_dir) == os.path.abspath(global_path):
            team_dir = None
    dir_path = team_dir if team_dir else os.path.expanduser("~/.npcsh/npc_team")
    os.makedirs(dir_path, exist_ok=True)
    perm_path = os.path.join(dir_path, "permissions.yaml")
    existing = _load_permission_file(perm_path)
    existing[key] = level
    with open(perm_path, "w") as f:
        yaml.dump({"rules": existing}, f, default_flow_style=False)
def _wait_for_permission_response(request_id, timeout=120):
    event = threading.Event()
    with permission_lock:
        permission_requests[request_id] = {"event": event, "decision": None}
    ready = event.wait(timeout=timeout)
    with permission_lock:
        entry = permission_requests.pop(request_id, None)
    if not ready or not entry:
        return None
    return entry.get("decision")
@app.route("/api/permission_response", methods=["POST"])
def permission_response():
    data = request.json or {}
    request_id = data.get("request_id")
    decision = data.get("decision")
    if not request_id:
        return jsonify({"error": "request_id is required"}), 400
    with permission_lock:
        entry = permission_requests.get(request_id)
        if entry is None:
            return jsonify({"error": "unknown request_id"}), 404
        entry["decision"] = decision
        entry["event"].set()
    return jsonify({"success": True})
@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,PUT,POST,DELETE,OPTIONS")
    response.headers.add("Access-Control-Allow-Credentials", "true")
    return response
@app.route("/api/ollama/tool_models", methods=["GET"])
def get_ollama_tool_models():
    """
    Returns all Ollama models. Tool capability detection is unreliable,
    so we don't filter - let the user try and the backend will handle failures.
    """
    try:
        detected = []
        listing = ollama.list()
        for model in listing.get("models", []):
            name = getattr(model, "model", None) or model.get("name") if isinstance(model, dict) else None
            if name:
                detected.append(name)
        return jsonify({"models": detected, "error": None})
    except Exception as e:
        print(f"Error listing Ollama models: {e}")
        return jsonify({"models": [], "error": str(e)}), 500
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
@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "error": None})
@app.route("/v1/chat/completions", methods=["POST"])
def openai_chat_completions():
    """
    OpenAI-compatible chat completions endpoint.
    Allows using NPC team as a drop-in replacement for OpenAI API.
    Extra parameter:
      - agent: NPC name to use (optional, uses team's forenpc if not specified)
    """
    try:
        data = request.get_json()
        messages = data.get("messages", [])
        model = data.get("model", "gpt-4o-mini")
        stream = data.get("stream", False)
        temperature = data.get("temperature", 0.7)
        max_tokens = data.get("max_tokens", 4096)
        agent_name = data.get("agent") or data.get("npc")
        current_path = request.headers.get("X-Current-Path", os.getcwd())
        registered_teams = data.get("registered_teams", [])
        npc = None
        team = None
        project_team_path = os.path.join(current_path, "npc_team")
        search_paths = [project_team_path] + [p for p in registered_teams if p and os.path.isdir(p)]
        for team_path in search_paths:
            if os.path.exists(team_path):
                try:
                    team = Team(team_path)
                    if agent_name and agent_name in team.npcs:
                        npc = team.npcs[agent_name]
                        break
                    elif team.forenpc:
                        npc = team.forenpc
                        break
                except Exception as e:
                    print(f"Error loading team {team_path}: {e}")
                    continue
        if not npc and agent_name:
            for team_path in search_paths:
                npc_file = os.path.join(team_path, f"{agent_name}.npc")
                if os.path.exists(npc_file):
                    try:
                        npc = NPC(npc_file=npc_file)
                        break
                    except Exception as e:
                        print(f"Error loading NPC {npc_file}: {e}")
        prompt = ""
        conversation_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join([c.get("text", "") for c in content if c.get("type") == "text"])
            conversation_messages.append({"role": role, "content": content})
            if role == "user":
                prompt = content
        provider = data.get("provider")
        if not provider:
            if "gpt" in model or "o1" in model or model.startswith("o3"):
                provider = "openai"
            elif "claude" in model:
                provider = "anthropic"
            elif "gemini" in model:
                provider = "gemini"
            else:
                provider = "openai"
        if stream:
            def generate_stream():
                request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
                created = int(time.time())
                try:
                    response = get_llm_response(
                        prompt,
                        model=model,
                        provider=provider,
                        npc=npc,
                        team=team,
                        messages=conversation_messages[:-1],
                        stream=True,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                    for chunk in response:
                        if isinstance(chunk, str):
                            delta_content = chunk
                        elif hasattr(chunk, 'choices') and chunk.choices:
                            delta = chunk.choices[0].delta
                            delta_content = getattr(delta, 'content', '') or ''
                        else:
                            delta_content = str(chunk)
                        if delta_content:
                            chunk_data = {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": delta_content},
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(chunk_data)}\n\n"
                    final_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                except Exception as e:
                    error_chunk = {
                        "error": {
                            "message": str(e),
                            "type": "server_error"
                        }
                    }
                    yield f"data: {json.dumps(error_chunk)}\n\n"
            return Response(
                generate_stream(),
                mimetype='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'X-Accel-Buffering': 'no'
                }
            )
        else:
            response = get_llm_response(
                prompt,
                model=model,
                provider=provider,
                npc=npc,
                team=team,
                messages=conversation_messages[:-1],
                stream=False,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = ""
            if isinstance(response, str):
                content = response
            elif hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content or ""
            elif isinstance(response, dict):
                content = response.get("response") or response.get("output") or str(response)
            else:
                content = str(response)
            return jsonify({
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": content
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": -1,
                    "completion_tokens": -1,
                    "total_tokens": -1
                }
            })
    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "error": {
                "message": str(e),
                "type": "server_error",
                "code": 500
            }
        }), 500
@app.route("/v1/models", methods=["GET"])
def openai_list_models():
    """OpenAI-compatible models listing - returns available NPCs as models."""
    current_path = request.headers.get("X-Current-Path", os.getcwd())
    registered_teams = _parse_registered_teams()
    models = []
    search_paths = [os.path.join(current_path, "npc_team")] + [p for p in registered_teams if p and os.path.isdir(p)]
    for team_path in search_paths:
        if os.path.exists(team_path):
            for npc_file in Path(team_path).glob("*.npc"):
                models.append({
                    "id": npc_file.stem,
                    "object": "model",
                    "created": int(os.path.getmtime(npc_file)),
                    "owned_by": "npc-team"
                })
    return jsonify({
        "object": "list",
        "data": models
    })
@app.route('/api/models/gguf/scan', methods=['GET'])
def scan_gguf_models():
    """Scan for GGUF/GGML model files in specified or default directories."""
    directory = request.args.get('directory')
    models_dir = get_models_dir()
    default_dirs = [
        os.path.join(models_dir, 'gguf'),
        models_dir,
        os.path.expanduser('~/models'),
        os.path.expanduser('~/.cache/huggingface/hub'),
    ]
    cfg_dir = app.config.get('GGUF_DIR')
    if cfg_dir:
        default_dirs.insert(0, os.path.expanduser(cfg_dir))
    dirs_to_scan = [os.path.expanduser(directory)] if directory else default_dirs
    models = []
    seen_paths = set()
    for scan_dir in dirs_to_scan:
        if not os.path.isdir(scan_dir):
            continue
        for root, dirs, files in os.walk(scan_dir):
            for f in files:
                if f.endswith(('.gguf', '.ggml', '.bin')) and not f.startswith('.'):
                    full_path = os.path.join(root, f)
                    if full_path not in seen_paths:
                        seen_paths.add(full_path)
                        try:
                            size = os.path.getsize(full_path)
                            models.append({
                                'name': f,
                                'path': full_path,
                                'size': size,
                                'size_gb': round(size / (1024**3), 2)
                            })
                        except OSError:
                            pass
    return jsonify({'models': models, 'error': None})
@app.route('/api/models/hf/download', methods=['POST'])
def download_hf_model():
    """Download a GGUF model from HuggingFace."""
    data = request.json
    url = data.get('url', '')
    default_target = os.path.join(get_models_dir(), 'gguf')
    target_dir = data.get('target_dir', default_target)
    target_dir = os.path.expanduser(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    try:
        if url.startswith('http'):
            import requests
            filename = url.split('/')[-1].split('?')[0]
            target_path = os.path.join(target_dir, filename)
            print(f"Downloading {url} to {target_path}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return jsonify({'path': target_path, 'error': None})
        else:
            try:
                from huggingface_hub import hf_hub_download, list_repo_files
                files = list_repo_files(url)
                gguf_files = [f for f in files if f.endswith('.gguf')]
                if not gguf_files:
                    return jsonify({'error': 'No GGUF files found in repository'}), 400
                q4_files = [f for f in gguf_files if 'Q4' in f or 'q4' in f]
                file_to_download = q4_files[0] if q4_files else gguf_files[0]
                print(f"Downloading {file_to_download} from {url}")
                path = hf_hub_download(
                    repo_id=url,
                    filename=file_to_download,
                    local_dir=target_dir,
                    local_dir_use_symlinks=False
                )
                return jsonify({'path': path, 'error': None})
            except ImportError:
                return jsonify({'error': 'huggingface_hub not installed. Run: pip install huggingface_hub'}), 500
    except Exception as e:
        print(f"Error downloading HF model: {e}")
        return jsonify({'error': str(e)}), 500
@app.route('/api/models/hf/download_file', methods=['POST'])
def download_hf_file():
    """Download a specific file from a HuggingFace repository."""
    data = request.json
    repo_id = data.get('repo_id', '')
    filename = data.get('filename', '')
    default_target = os.path.join(get_models_dir(), 'gguf')
    target_dir = data.get('target_dir', default_target)
    if not repo_id or not filename:
        return jsonify({'error': 'repo_id and filename are required'}), 400
    target_dir = os.path.expanduser(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download
        print(f"Downloading {filename} from {repo_id} to {target_dir}")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=target_dir,
            local_dir_use_symlinks=False
        )
        return jsonify({'path': path, 'error': None})
    except ImportError:
        return jsonify({'error': 'huggingface_hub not installed. Run: pip install huggingface_hub'}), 500
    except Exception as e:
        print(f"Error downloading HF file: {e}")
        return jsonify({'error': str(e)}), 500
@app.route('/api/generate_music', methods=['POST'])
def generate_music_endpoint():
    """Generate music from a text prompt.
    JSON body: { prompt, provider, model?, duration?, api_key?, currentPath? }
    Providers: 'local' (MusicGen via transformers), 'replicate' (musicgen/stable-audio/riffusion),
    'elevenlabs' (sound-generation, <=22s).
    """
    try:
        import base64 as _b64
        from npcpy.gen.audio_gen import generate_music
        data = request.json or {}
        prompt = data.get('prompt', '').strip()
        if not prompt:
            return jsonify({'success': False, 'error': 'prompt is required'}), 400
        provider = data.get('provider', 'replicate')
        model = data.get('model')
        duration = int(data.get('duration', 10))
        api_key = data.get('api_key')
        current_path = data.get('currentPath') or data.get('current_path')
        if not current_path:
            return jsonify({'success': False, 'error': 'currentPath is required'}), 400
        result = generate_music(
            prompt=prompt,
            provider=provider,
            model=model,
            duration=duration,
            api_key=api_key,
        )
        save_dir = os.path.abspath(os.path.expanduser(current_path))
        os.makedirs(save_dir, exist_ok=True)
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'scherzo_gen_{ts}.{result["format"]}'
        filepath = os.path.join(save_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(result['audio'])
        return jsonify({
            'success': True,
            'filename': filepath,
            'format': result['format'],
            'provider': result['provider'],
            'model': result['model'],
            'audio': _b64.b64encode(result['audio']).decode('utf-8'),
        })
    except Exception as e:
        print(f"Music generation error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500
@app.route('/api/audio/tts', methods=['POST'])
def text_to_speech_endpoint():
    """Convert text to speech and return audio file."""
    try:
        import base64
        from npcpy.gen.audio_gen import (
            text_to_speech, get_available_engines,
            pcm16_to_wav
        )
        data = request.json or {}
        text = data.get('text', '')
        engine = data.get('engine', 'kokoro')
        voice = data.get('voice', 'af_heart')
        if not text:
            return jsonify({'success': False, 'error': 'No text provided'}), 400
        engines = get_available_engines()
        if engine not in engines:
            return jsonify({'success': False, 'error': f'Unknown engine: {engine}'}), 400
        if not engines[engine]['available']:
            if engines.get('kokoro', {}).get('available'):
                engine = 'kokoro'
            elif engines.get('gtts', {}).get('available'):
                engine = 'gtts'
                voice = 'en'
            else:
                return jsonify({
                    'success': False,
                    'error': f'{engine} not available. Install: {engines[engine].get("install", engines[engine].get("requires", ""))}'
                }), 400
        audio_bytes = text_to_speech(text, engine=engine, voice=voice)
        if engine in ['kokoro']:
            audio_format = 'wav'
        elif engine in ['elevenlabs', 'gtts']:
            audio_format = 'mp3'
        elif engine in ['openai', 'gemini']:
            audio_bytes = pcm16_to_wav(audio_bytes, sample_rate=24000)
            audio_format = 'wav'
        else:
            audio_format = 'wav'
        audio_data = base64.b64encode(audio_bytes).decode('utf-8')
        return jsonify({
            'success': True,
            'audio': audio_data,
            'format': audio_format,
            'engine': engine,
            'voice': voice
        })
    except ImportError as e:
        return jsonify({'success': False, 'error': f'TTS dependency not installed: {e}'}), 500
    except Exception as e:
        print(f"TTS error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500
@app.route('/api/audio/stt', methods=['POST'])
def speech_to_text_endpoint():
    """Convert speech audio to text using various STT engines."""
    try:
        import tempfile
        import base64
        from npcpy.data.audio import speech_to_text, get_available_stt_engines
        data = request.json or {}
        audio_data = data.get('audio')
        audio_format = data.get('format', 'webm')
        language = data.get('language')
        engine = data.get('engine', 'whisper')
        model_size = data.get('model', 'base')
        if not audio_data:
            return jsonify({'success': False, 'error': 'No audio data provided'}), 400
        audio_bytes = base64.b64decode(audio_data)
        wav_bytes = audio_bytes
        if audio_format != 'wav':
            with tempfile.NamedTemporaryFile(suffix=f'.{audio_format}', delete=False) as f:
                f.write(audio_bytes)
                temp_path = f.name
            wav_path = temp_path.replace(f'.{audio_format}', '.wav')
            converted = False
            try:
                subprocess.run([
                    'ffmpeg', '-y', '-i', temp_path,
                    '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000',
                    wav_path
                ], check=True, capture_output=True)
                with open(wav_path, 'rb') as f:
                    wav_bytes = f.read()
                converted = True
                os.unlink(wav_path)
            except FileNotFoundError:
                pass
            except subprocess.CalledProcessError:
                pass
            if not converted:
                try:
                    from pydub import AudioSegment
                    audio = AudioSegment.from_file(temp_path, format=audio_format)
                    audio = audio.set_frame_rate(16000).set_channels(1)
                    import io
                    wav_buffer = io.BytesIO()
                    audio.export(wav_buffer, format='wav')
                    wav_bytes = wav_buffer.getvalue()
                    converted = True
                except ImportError:
                    pass
                except Exception as e:
                    print(f"pydub conversion failed: {e}")
            os.unlink(temp_path)
            if not converted:
                return jsonify({
                    'success': False,
                    'error': 'Audio conversion failed. Install ffmpeg: sudo apt-get install ffmpeg'
                }), 500
        result = speech_to_text(
            wav_bytes,
            engine=engine,
            language=language,
            model_size=model_size
        )
        return jsonify({
            'success': True,
            'text': result.get('text', ''),
            'language': result.get('language', language or 'en'),
            'segments': result.get('segments', [])
        })
    except Exception as e:
        print(f"STT error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500
@app.route('/api/audio/stt/engines', methods=['GET'])
def get_stt_engines_endpoint():
    """Get available STT engines."""
    try:
        from npcpy.data.audio import get_available_stt_engines
        engines = get_available_stt_engines()
        return jsonify({'success': True, 'engines': engines})
    except Exception as e:
        print(f"Error getting STT engines: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
@app.route('/api/audio/voices', methods=['GET'])
def get_available_voices_endpoint():
    """Get available TTS voices/engines."""
    try:
        from npcpy.gen.audio_gen import get_available_engines, get_available_voices
        engines_info = get_available_engines()
        result = {}
        for engine_id, info in engines_info.items():
            voices = get_available_voices(engine_id) if info['available'] else []
            result[engine_id] = {
                'name': info['name'],
                'type': info.get('type', 'unknown'),
                'available': info['available'],
                'description': info.get('description', ''),
                'default': engine_id == 'kokoro',
                'voices': voices
            }
            if not info['available']:
                if 'install' in info:
                    result[engine_id]['install'] = info['install']
                if 'requires' in info:
                    result[engine_id]['requires'] = info['requires']
        return jsonify({'success': True, 'engines': result})
    except Exception as e:
        print(f"Error getting voices: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500
@app.route('/api/activity/track', methods=['POST'])
def track_activity():
    """Track user activity for predictive features."""
    try:
        data = request.json or {}
        activity_type = data.get('type', 'unknown')
        return jsonify({'success': True, 'tracked': activity_type})
    except Exception as e:
        print(f"Error tracking activity: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
def start_flask_server(
    port=5337,
    host="0.0.0.0",
    cors_origins=None,
    static_files=None,
    debug=False,
    teams=None,
    npcs=None,
    db_path: str ='',
    user_npc_directory = None,
    data_dir = None,
    kg_registry = None,
    image_model = None,
    image_provider = None,
    gguf_dir = None,
):
    try:
        if teams:
            app.registered_teams = teams
            print(f"Registered {len(teams)} teams: {list(teams.keys())}")
        else:
            app.registered_teams = {}
        if npcs:
            app.registered_npcs = npcs
            print(f"Registered {len(npcs)} NPCs: {list(npcs.keys())}")
        else:
            app.registered_npcs = {}
        app.config['DB_PATH'] = db_path
        app.config['user_npc_directory'] = user_npc_directory
        app.config['DATA_DIR'] = data_dir
        app.config['KG_REGISTRY_PATH'] = kg_registry
        app.config['IMAGE_MODEL'] = image_model
        app.config['IMAGE_PROVIDER'] = image_provider
        app.config['GGUF_DIR'] = gguf_dir
        if cors_origins:
            CORS(
                app,
                origins=cors_origins,
                allow_headers=["Content-Type", "Authorization"],
                methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                supports_credentials=True,
            )
        print(f"Starting Flask server on http://{host}:{port}")
        app.run(host=host, port=port, debug=debug, threaded=True)
    except OSError as e:
        if "Address already in use" in str(e) or "10048" in str(e):
            print(f"Address already in use")
            print(f"Port {port} is in use by another program. Either identify and stop that program, or start the server with a different port.")
        else:
            print(f"Error starting server: {str(e)}")
        raise
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        raise
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generic npcpy API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server on")
    parser.add_argument("--port", type=int, default=5337, help="Port to bind the server on")
    parser.add_argument("--db-path", default=None, help="Path to the SQLite history database")
    parser.add_argument("--dir", default=".", help="Path to the NPC directory (default: current working directory)")
    parser.add_argument("--data-dir", default=None, help="Path to the server data directory")
    parser.add_argument("--kg-registry", default=None, help="Path to a YAML KG registry file")
    parser.add_argument("--image-model", default=None, help="Default image generation model")
    parser.add_argument("--image-provider", default=None, help="Default image generation provider")
    parser.add_argument("--gguf-dir", default=None, help="Directory to scan for GGUF models")
    parser.add_argument("--team", action="append", dest="teams", default=None, help="Path to a team directory to register (can be given multiple times)")
    parser.add_argument("--teams-yaml", default=None, help="Path to a YAML file mapping team names to team directory paths")
    args = parser.parse_args()
    base_dir = os.path.expanduser("~/.npcpy")
    db_path = args.db_path or os.path.join(base_dir, "history.db")
    npc_dir = os.path.abspath(os.path.expanduser(args.dir))
    data_dir = args.data_dir or os.path.join(base_dir, "data")
    kg_registry = args.kg_registry
    try:
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        os.makedirs(npc_dir, exist_ok=True)
        os.makedirs(os.path.join(npc_dir, "jinxes"), exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
    except Exception as dir_err:
        print(f"[SERVE] Warning: Could not create directories: {dir_err}")
    teams = {}
    if args.teams_yaml:
        teams_yaml_path = os.path.abspath(os.path.expanduser(args.teams_yaml))
        if os.path.isfile(teams_yaml_path):
            try:
                with open(teams_yaml_path, 'r') as f:
                    yaml_data = yaml.safe_load(f) or {}
                loaded = yaml_data.get('teams', yaml_data)
                if isinstance(loaded, dict):
                    for team_name, team_path in loaded.items():
                        team_path = os.path.abspath(os.path.expanduser(str(team_path)))
                        if os.path.isdir(team_path):
                            teams[str(team_name)] = team_path
                elif isinstance(loaded, list):
                    for team_path in loaded:
                        team_path = os.path.abspath(os.path.expanduser(str(team_path)))
                        if os.path.isdir(team_path):
                            team_name = os.path.basename(team_path)
                            teams[team_name] = team_path
            except Exception as e:
                print(f"[SERVE] Warning: Could not load teams YAML {teams_yaml_path}: {e}")
    if args.teams:
        for team_path in args.teams:
            team_path = os.path.abspath(os.path.expanduser(team_path))
            if os.path.isdir(team_path):
                team_name = os.path.basename(team_path)
                teams[team_name] = team_path
    try:
        start_flask_server(
            host=args.host,
            port=args.port,
            db_path=db_path,
            user_npc_directory=npc_dir,
            data_dir=data_dir,
            kg_registry=kg_registry,
            image_model=args.image_model,
            image_provider=args.image_provider,
            gguf_dir=args.gguf_dir,
            teams=teams,
        )
    except OSError as e:
        if "Address already in use" in str(e) or "10048" in str(e):
            print(f"Port {args.port} is in use by another program. Either identify and stop that program, or start the server with a different port.")
        else:
            print(f"[SERVE] Failed to start server: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[SERVE] Failed to start server: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
@app.errorhandler(Exception)
def handle_global_exception(e):
    """Handle all unhandled exceptions and return JSON instead of HTML."""
    print(f"Unhandled exception: {e}")
    traceback.print_exc()
    return jsonify({
        'success': False,
        'error': str(e),
        'error_type': type(e).__name__
    }), 500
@app.errorhandler(404)
def handle_404(e):
    """Handle 404 errors and return JSON instead of HTML."""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'path': request.path
    }), 404
@app.errorhandler(500)
def handle_500(e):
    """Handle 500 errors and return JSON instead of HTML."""
    return jsonify({
        'success': False,
        'error': 'Internal server error',
        'details': str(e)
    }), 500
