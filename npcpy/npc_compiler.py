import os
import shutil
from pyexpat.errors import messages
import urllib.parse
import yaml
import json
import sqlite3
import numpy as np
import pandas as pd
import re
import random
from datetime import datetime
import hashlib
import pathlib
import sys 
import fnmatch
import subprocess
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from jinja2 import Environment, FileSystemLoader, Template, Undefined, DictLoader
from jinja2.sandbox import SandboxedEnvironment
from sqlalchemy import create_engine, text
import npcpy as npy 
from npcpy.tools import auto_tools
import math
import random
import base64
from npcpy.npc_sysenv import (
    get_system_message,
    print_and_process_stream_with_markdown,
    )

class SilentUndefined(Undefined):
    """Undefined that silently returns empty string instead of raising errors"""
    def _fail_with_undefined_error(self, *args, **kwargs):
        return ""

    def __str__(self):
        return ""

    def __repr__(self):
        return ""

    def __bool__(self):
        return False

    def __eq__(self, other):
        return other == "" or other is None or isinstance(other, Undefined)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __iter__(self):
        return iter([])

    def __len__(self):
        return 0

import math
from PIL import Image
from jinja2 import Environment, ChainableUndefined

class PreserveUndefined(ChainableUndefined):
    """Undefined that preserves the original {{ variable }} syntax"""
    def __str__(self):
        return f"{{{{ {self._undefined_name} }}}}"

def agent_pass_handler(command, extracted_data, **kwargs):
    """Handler for agent pass action"""
    npc = kwargs.get('npc')
    team = kwargs.get('team')    
    if not team and npc and hasattr(npc, '_current_team'):
        team = npc._current_team
    
    
    if not npc or not team:
        return {"messages": kwargs.get('messages', []), "output": f"Error: No NPC ({npc.name if npc else 'None'}) or team ({team.name if team else 'None'}) available for agent pass"}
    
    target_npc_name = extracted_data.get('target_npc')
    if not target_npc_name:
        return {"messages": kwargs.get('messages', []), "output": "Error: No target NPC specified"}
    
    messages = kwargs.get('messages', [])
    
    
    pass_count = 0
    recent_passes = []
    
    for msg in messages[-10:]:  
        if 'NOTE: THIS COMMAND HAS BEEN PASSED FROM' in msg.get('content', ''):
            pass_count += 1
            
            if 'PASSED FROM' in msg.get('content', ''):
                content = msg.get('content', '')
                if 'PASSED FROM' in content and 'TO YOU' in content:
                    parts = content.split('PASSED FROM')[1].split('TO YOU')[0].strip()
                    recent_passes.append(parts)
    

    
    target_npc = team.get_npc(target_npc_name)
    if not target_npc:
        available_npcs = list(team.npcs.keys()) if hasattr(team, 'npcs') else []
        return {"messages": kwargs.get('messages', []), 
                "output": f"Error: NPC '{target_npc_name}' not found in team. Available: {available_npcs}"}
    
    
    
    result = npc.handle_agent_pass(
        target_npc,
        command,
        messages=kwargs.get('messages'),
        context=kwargs.get('context'),
        shared_context=getattr(team, 'shared_context', None),
        stream=kwargs.get('stream', False),
        team=team
    )
    
    return result

def create_or_replace_table(db_path, table_name, data):
    """Creates or replaces a table in the SQLite database"""
    conn = sqlite3.connect(os.path.expanduser(db_path))
    try:
        data.to_sql(table_name, conn, if_exists="replace", index=False)
        print(f"Table '{table_name}' created/replaced successfully.")
        return True
    except Exception as e:
        print(f"Error creating/replacing table '{table_name}': {e}")
        return False
    finally:
        conn.close()

def find_file_path(filename, search_dirs, suffix=None):
    """Find a file in multiple directories"""
    if suffix and not filename.endswith(suffix):
        filename += suffix
        
    for dir_path in search_dirs:
        file_path = os.path.join(os.path.expanduser(dir_path), filename)
        if os.path.exists(file_path):
            return file_path
            
    return None

def get_log_entries(entity_id, db_path, entry_type=None, limit=10):
    """Get log entries for an NPC or team"""
    db_path = os.path.expanduser(db_path)
    with sqlite3.connect(db_path) as conn:
        query = "SELECT entry_type, content, metadata, timestamp FROM npc_log WHERE entity_id = ?"
        params = [entity_id]

        if entry_type:
            query += " AND entry_type = ?"
            params.append(entry_type)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        results = conn.execute(query, params).fetchall()

        return [
            {
                "entry_type": r[0],
                "content": json.loads(r[1]),
                "metadata": json.loads(r[2]) if r[2] else None,
                "timestamp": r[3]
            }
            for r in results
        ]

def log_entry(entity_id, db_path, entry_type, content, metadata=None):
    """Log an entry for an NPC or team"""
    db_path = os.path.expanduser(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO npc_log (entity_id, entry_type, content, metadata) VALUES (?, ?, ?, ?)",
            (entity_id, entry_type, json.dumps(content), json.dumps(metadata) if metadata else None)
        )
        conn.commit()

def _json_dumps_with_undefined(obj, **kwargs):
    """Custom JSON dumps that handles SilentUndefined objects"""
    def default_handler(o):
        if isinstance(o, Undefined):
            return ""
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
    return json.dumps(obj, default=default_handler, **kwargs)

def load_yaml_file(file_path, jinja_context=None):
    """Load a YAML file with error handling, rendering Jinja2 first.

    Args:
        file_path: Path to the YAML file
        jinja_context: Optional dict of variables to pass to Jinja2 rendering.
            Used by Team to inject jinx name->path mappings so NPC files can
            use {{ jinx_name }} to resolve to the correct relative path.
    """
    try:
        with open(os.path.expanduser(file_path), 'r', encoding="utf-8") as f:
            content = f.read()

        has_jinja = '{%' in content or (jinja_context is not None and '{{' in content and '}}' in content)
        if not has_jinja:
            return yaml.safe_load(content)

        jinja_env = SandboxedEnvironment(undefined=SilentUndefined)
        jinja_env.policies['json.dumps_function'] = _json_dumps_with_undefined
        template = jinja_env.from_string(content)
        rendered_content = template.render(jinja_context or {})

        return yaml.safe_load(rendered_content)
    except Exception as e:
        print(f"Error loading YAML file {file_path}: {e}")
        return None

def log_entry(entity_id, entry_type, content, metadata=None, db_path="~/npcpy_history.db"):
    """Log an entry for an NPC or team"""
    db_path = os.path.expanduser(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO npc_log (entity_id, entry_type, content, metadata) VALUES (?, ?, ?, ?)",
            (entity_id, entry_type, json.dumps(content), json.dumps(metadata) if metadata else None)
        )
        conn.commit()

def initialize_npc_project(
    directory=None,
    context=None,
    model=None,
    provider=None,
) -> str:
    """Initialize an NPC project."""
    if directory is None:
        directory = os.getcwd()
    directory = os.path.expanduser(os.fspath(directory))

    npc_team_dir = os.path.join(directory, "npc_team")
    os.makedirs(npc_team_dir, exist_ok=True)

    for subdir in ["jinxes",
                   "jinxes/skills",
                   "assembly_lines",
                   "sql_models",
                   "jobs",
                   "triggers",
                   "tools",
                   "images",
                   "models",
                   "attachments",
                   "mcp_servers"]:
        os.makedirs(os.path.join(npc_team_dir, subdir), exist_ok=True)

    forenpc_path = os.path.join(npc_team_dir, "forenpc.npc")

    if not os.path.exists(forenpc_path):
        default_npc = {
            "name": "forenpc",
            "primary_directive": "You are the forenpc of an NPC team",
        }
        if model:
            default_npc["model"] = model
        if provider:
            default_npc["provider"] = provider
        with open(forenpc_path, "w", encoding="utf-8") as f:
            yaml.dump(default_npc, f)

    ctx_destination: Optional[str] = None
    preexisting_ctx = [
        os.path.join(npc_team_dir, f)
        for f in os.listdir(npc_team_dir)
        if f.endswith(".ctx")
    ]
    if preexisting_ctx:
        ctx_destination = preexisting_ctx[0]
        if len(preexisting_ctx) > 1:
            print(
                "Warning: Multiple .ctx files already present; using first and ignoring the rest."
            )

    if not ctx_destination:
        default_ctx_path = os.path.join(npc_team_dir, "team.ctx")
        default_ctx = {
            'name': '',
            'context' : context or '',
            'mcp_servers': '',
            'databases':'',
            'forenpc': 'forenpc'
        }
        if model:
            default_ctx['model'] = model
        if provider:
            default_ctx['provider'] = provider
        with open(default_ctx_path, "w", encoding="utf-8") as f:
            yaml.dump(default_ctx, f)
        ctx_destination = default_ctx_path

    return f"NPC project initialized in {npc_team_dir}"

def write_yaml_file(file_path, data):
    """Write data to a YAML file"""
    try:
        with open(os.path.expanduser(file_path), 'w', encoding="utf-8") as f:
            yaml.dump(data, f)
        return True
    except Exception as e:
        print(f"Error writing YAML file {file_path}: {e}")
        return False


def render_jinja_content(content):
    from jinja2 import Environment
    env = Environment()
    env.globals['Jinx'] = lambda name: name
    try:
        return env.from_string(content).render()
    except Exception:
        return content


def _update_field_in_yaml(content, field, new_value):
    lines = content.split('\n')
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        match = re.match(r'^(\s*)(' + re.escape(field) + r'):\s*(.*)$', line)
        if match:
            indent = match.group(1)
            if isinstance(new_value, list):
                result.append(f"{indent}{field}:")
                for item in new_value:
                    result.append(f"{indent}  - {item}")
            elif isinstance(new_value, str) and '\n' in new_value:
                result.append(f"{indent}{field}: |2")
                for sub in new_value.split('\n'):
                    result.append(f"{indent}  {sub}")
            else:
                result.append(f"{indent}{field}: {new_value}")
            i += 1
            target_indent = len(indent)
            while i < len(lines):
                next_line = lines[i]
                if next_line.strip() == '':
                    i += 1
                    continue
                next_indent = len(next_line) - len(next_line.lstrip())
                if next_indent <= target_indent:
                    break
                i += 1
            continue
        result.append(line)
        i += 1
    return '\n'.join(result)


DEFAULT_MD_AGENT_JINXES = [
    'sh', 'python', 'edit_file', 'load_file', 'file_search',
    'web_search', 'chat', 'stop',
]


class Jinx:
    ''' 
    Jinx represents a workflow template with Jinja-rendered steps.
    
    Loads YAML definition containing:
    - jinx_name: identifier
    - inputs: list of input parameters
    - description: what the jinx does
    - npc: optional NPC to execute with
    - steps: list of step definitions with code. This section can now be a Jinja template itself.
    - file_context: optional list of file patterns to include as context
    
    Execution:
    - Renders Jinja templates in step code with input values
    - Executes resulting Python code
    - Returns context with outputs
    '''
    def __init__(self, jinx_data=None, jinx_path=None):
        if jinx_path:
            self._load_from_file(jinx_path)
        elif jinx_data:
            self._load_from_data(jinx_data)
        else:
            raise ValueError("Either jinx_data or jinx_path must be provided")
        
        self._raw_steps = list(self.steps)
        if self.steps and all(isinstance(s, dict) for s in self.steps):
            pass
        else:
            self.steps = []
        self.parsed_files = {}
        if self.file_context:
            self.parsed_files = self._parse_file_patterns(self.file_context)

    def _load_from_file(self, path):
        jinx_data = load_yaml_file(path)
        if not jinx_data:
            raise ValueError(f"Failed to load jinx from {path}")
        jinx_data['_source_path'] = path
        self._load_from_data(jinx_data)
            

    def _load_from_data(self, jinx_data):
        if not jinx_data or not isinstance(jinx_data, dict):
            raise ValueError("Invalid jinx data provided")
            
        if "jinx_name" not in jinx_data:
            raise KeyError("Missing 'jinx_name' in jinx definition")
            
        self.jinx_name = jinx_data.get("jinx_name")
        self.inputs = jinx_data.get("inputs", [])
        self.description = jinx_data.get("description", "")
        self.npc = jinx_data.get("npc")
        self.steps = jinx_data.get("steps", [])
        self.file_context = jinx_data.get("file_context", [])
        self._source_path = jinx_data.get("_source_path", None)
        self.permissions = jinx_data.get("permissions", {})
        if not isinstance(self.permissions, dict):
            self.permissions = {}
        if "default" not in self.permissions:
            self.permissions["default"] = "ask"

    def set_permission(self, level: str):
        """Persist a permission level to this jinx's metadata and source file."""
        self.permissions["default"] = level
        if self._source_path:
            self.save(os.path.dirname(self._source_path))

    def check_permission(self) -> str:
        """Return this jinx's default permission level: allow, deny, or ask."""
        level = self.permissions.get("default", "ask")
        if level not in ("allow", "deny", "ask"):
            return "ask"
        return level

    def to_tool_def(self) -> Dict[str, Any]:
        """Convert this Jinx to an OpenAI-style tool definition."""
        properties = {}
        required = []
        for inp in self.inputs:
            if isinstance(inp, str):
                properties[inp] = {"type": "string", "description": f"Parameter: {inp}"}
                required.append(inp)
            elif isinstance(inp, dict):
                name = list(inp.keys())[0]
                default_val = inp.get(name, "")
                desc = f"Parameter: {name}"
                if default_val != "":
                    desc += f" (default: {default_val})"
                properties[name] = {"type": "string", "description": desc}
        return {
            "type": "function",
            "function": {
                "name": self.jinx_name,
                "description": self.description or f"Jinx: {self.jinx_name}",
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }

    def render_first_pass(
        self,
        jinja_env_for_macros: Environment,
        all_jinx_callables: Dict[str, Callable]
    ):
        """
        Performs the first-pass Jinja rendering on the Jinx's raw steps.
        This expands Jinja control flow (for, if) to generate step structures,
        then expands nested Jinx calls (e.g., {{ sh(...) }} or engine: jinx_name)
        and inline macros.
        """
        if self._raw_steps and isinstance(self._raw_steps[0], dict):
            structurally_expanded_steps = list(self._raw_steps)
        else:
            raw_steps_template_string = "\n".join(self._raw_steps)

            try:
                steps_template = jinja_env_for_macros.from_string(raw_steps_template_string)
                rendered_steps_yaml_string = steps_template.render(**jinja_env_for_macros.globals)
            except Exception as e:
                self.steps = list(self._raw_steps)
                return

            try:
                structurally_expanded_steps = yaml.safe_load(rendered_steps_yaml_string)
                if not isinstance(structurally_expanded_steps, list):
                    if structurally_expanded_steps is None:
                        structurally_expanded_steps = []
                    else:
                        raise ValueError(f"Rendered steps YAML did not result in a list: {type(structurally_expanded_steps)}")
            except Exception as e:
                self.steps = list(self._raw_steps)
                return

        final_rendered_steps = []
        for raw_step in structurally_expanded_steps:
            if not isinstance(raw_step, dict):
                final_rendered_steps.append(raw_step)
                continue

            engine_name = raw_step.get('engine')
            
            if engine_name and engine_name in all_jinx_callables:
                step_name = raw_step.get('name', f'call_{engine_name}')
                jinx_args = {
                    k: v for k, v in raw_step.items() 
                    if k not in ['engine', 'name']
                }
                
                jinx_callable = all_jinx_callables[engine_name]
                try:
                    expanded_yaml_string = jinx_callable(**jinx_args)
                    expanded_steps = yaml.safe_load(expanded_yaml_string)
                    
                    if isinstance(expanded_steps, list):
                        final_rendered_steps.extend(expanded_steps)
                    elif expanded_steps is not None:
                        final_rendered_steps.append(expanded_steps)
                except Exception as e:
                    final_rendered_steps.append(raw_step)
            elif raw_step.get('engine') in ['python', 'bash']:
                processed_step = {}
                for key, value in raw_step.items():
                    if isinstance(value, str):
                        has_template_var = '{{' in value and '}}' in value
                        if has_template_var:
                            processed_step[key] = value
                        else:
                            try:
                                template = jinja_env_for_macros.from_string(value)
                                rendered_value = template.render({})
                                try:
                                    loaded_value = yaml.safe_load(rendered_value)
                                    processed_step[key] = loaded_value
                                except yaml.YAMLError:
                                    processed_step[key] = rendered_value
                            except Exception as e:
                                processed_step[key] = value
                    else:
                        processed_step[key] = value
                final_rendered_steps.append(processed_step)
            else:
                processed_step = {}
                for key, value in raw_step.items():
                    if isinstance(value, str):
                        try:
                            template = jinja_env_for_macros.from_string(value)
                            rendered_value = template.render({})
                            try:
                                loaded_value = yaml.safe_load(rendered_value)
                                processed_step[key] = loaded_value
                            except yaml.YAMLError:
                                processed_step[key] = rendered_value
                        except Exception as e:
                            processed_step[key] = value
                    else:
                        processed_step[key] = value
                final_rendered_steps.append(processed_step)
        
        self.steps = final_rendered_steps

    def execute(self,
                input_values: Dict[str, Any],
                npc: Optional[Any] = None,
                messages: Optional[List[Dict[str, str]]] = None,
                extra_globals: Optional[Dict[str, Any]] = None,
                jinja_env: Optional[Environment] = None):

        if jinja_env is None:
            jinja_env = SandboxedEnvironment(
                loader=DictLoader({}),
                undefined=SilentUndefined,
            )

        active_npc = self.npc if self.npc else npc

        from npcpy.npc_array import NPCArray
        if isinstance(active_npc, (list, NPCArray)):
            arr = NPCArray.from_npcs(active_npc) if isinstance(active_npc, list) else active_npc
            return arr.jinx(self.jinx_name, inputs=input_values).collect()
        
        context = (
            active_npc.shared_context.copy() 
            if active_npc and hasattr(active_npc, 'shared_context') 
            else {}
        )
        context.update(input_values)
        context.update({
            "llm_response": None,
            "output": None,
            "messages": messages,
            "npc": active_npc
        })
        
        if self.parsed_files:
            context['file_context'] = self._format_parsed_files_context(self.parsed_files)
            context['files'] = self.parsed_files

        for i, step in enumerate(self.steps):
            context = self._execute_step(
                step,
                context,
                jinja_env,
                npc=active_npc,
                messages=messages,
                extra_globals=extra_globals
            )
            output_str = str(context.get("output", ""))
            if "error" in output_str.lower():
                break

        return context

    def _execute_step(self,
                  step: Dict[str, Any],
                  context: Dict[str, Any],
                  jinja_env: Environment,
                  npc: Optional[Any] = None,
                  messages: Optional[List[Dict[str, str]]] = None,
                  extra_globals: Optional[Dict[str, Any]] = None):

        step_name = step.get("name", "unnamed_step")
        step_npc = step.get("npc")
        active_npc = step_npc if step_npc else npc

        code_content = step.get("code") or ""

        try:
            template = jinja_env.from_string(code_content)
            rendered_code = template.render(**context)
        except Exception as e:
            error_msg = f"Error rendering template for step '{step_name}' (second pass): {type(e).__name__}: {e}"
            context['output'] = error_msg
            return context

        from npcpy.npc_array import NPCArray, infer_matrix, ensemble_vote

        exec_globals = {
            "__builtins__": __builtins__,
            "npc": active_npc,
            "context": context,
            "math": math,
            "random": random,
            "datetime": datetime,
            "Image": Image,
            "pd": pd,
            "sys": sys,
            "subprocess": subprocess,
            "np": np,
            "os": os,
            're': re,
            "json": json,
            "Path": pathlib.Path,
            "fnmatch": fnmatch,
            "pathlib": pathlib,
            "subprocess": subprocess,
            "get_llm_response": npy.llm_funcs.get_llm_response,
            "print_and_process_stream_with_markdown": print_and_process_stream_with_markdown,
            "NPCArray": NPCArray,
            "infer_matrix": infer_matrix,
            "ensemble_vote": ensemble_vote,
        }
        
        if extra_globals:
            exec_globals.update(extra_globals)

        exec_globals.update(context)

        exec_locals = exec_globals

        try:
            exec(rendered_code, exec_globals, exec_locals)
        except SystemExit as e:
            error_msg = f"Error executing step '{step_name}' in jinx '{self.jinx_name}': code called sys.exit({e.code})"
            print(f"[JINX-ERROR] {error_msg}")
            context['output'] = error_msg
            return context
        except Exception as e:
            error_msg = f"Error executing step '{step_name}' in jinx '{self.jinx_name}': {type(e).__name__}: {e}"
            print(f"[JINX-ERROR] {error_msg}")
            context['output'] = error_msg
            return context

        context_output = context.get("output")
        context.update(exec_locals)

        if context_output is not None:
            context["output"] = context_output
            context[step_name] = context_output

        if "output" in exec_locals and exec_locals["output"] is not None:
            outp = exec_locals["output"]
            context["output"] = outp
            context[step_name] = outp

        final_output = context.get("output")
        if final_output and messages is not None:
            messages.append({
                'role':'assistant',
                'content': f'Jinx {self.jinx_name} step {step_name} executed: {final_output}'
            })
            context['messages'] = messages
        
        return context

    def _parse_file_patterns(self, patterns_config):
        """Parse file patterns configuration and load matching files into KV cache"""
        if not patterns_config:
            return {}
        
        file_cache = {}
        
        for pattern_entry in patterns_config:
            if isinstance(pattern_entry, str):
                pattern_entry = {"pattern": pattern_entry}
            
            pattern = pattern_entry.get("pattern", "")
            recursive = pattern_entry.get("recursive", False)
            base_path = pattern_entry.get("base_path", ".")
            
            if not pattern:
                continue
                
            if self._source_path:
                base_path = os.path.join(os.path.dirname(self._source_path), base_path)
            base_path = os.path.expanduser(base_path)
            
            if not os.path.isabs(base_path):
                base_path = os.path.join(os.getcwd(), base_path)
            
            matching_files = self._find_matching_files(pattern, base_path, recursive)
            
            for file_path in matching_files:
                file_content = self._load_file_content(file_path)
                if file_content:
                    relative_path = os.path.relpath(file_path, base_path)
                    file_cache[relative_path] = file_content
        
        return file_cache

    def _find_matching_files(self, pattern, base_path, recursive=False):
        """Find files matching the given pattern"""
        matching_files = []
        
        if not os.path.exists(base_path):
            return matching_files
        
        if recursive:
            for root, dirs, files in os.walk(base_path):
                for filename in files:
                    if fnmatch.fnmatch(filename, pattern):
                        matching_files.append(os.path.join(root, filename))
        else:
            try:
                for item in os.listdir(base_path):
                    item_path = os.path.join(base_path, item)
                    if os.path.isfile(item_path) and fnmatch.fnmatch(item, pattern):
                        matching_files.append(item_path)
            except PermissionError:
                print(f"Permission denied accessing {base_path}")
        
        return matching_files

    def _load_file_content(self, file_path):
        """Load content from a file with error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

    def _format_parsed_files_context(self, parsed_files):
        """Format parsed files into context string"""
        if not parsed_files:
            return ""
        
        context_parts = ["Additional context from files:"]
        
        for file_path, content in parsed_files.items():
            context_parts.append(f"\n--- {file_path} ---")
            context_parts.append(content)
            context_parts.append("")
        
        return "\n".join(context_parts)

    def to_dict(self):
        result = {
            "jinx_name": self.jinx_name,
            "description": self.description,
            "inputs": self.inputs,
            "steps": self._raw_steps,
            "file_context": self.file_context
        }
        
        if self.npc:
            result["npc"] = self.npc

        result["permissions"] = self.permissions

        return result

    def save(self, directory):
        jinx_path = os.path.join(directory, f"{self.jinx_name}.jinx")
        os.makedirs(os.path.dirname(jinx_path), exist_ok=True)
        return write_yaml_file(jinx_path, self.to_dict())
        
    @classmethod
    def from_mcp(cls, mcp_tool):
        try:
            import inspect

            doc = mcp_tool.__doc__ or ""
            name = mcp_tool.__name__
            signature = inspect.signature(mcp_tool)
            
            inputs = []
            for param_name, param in signature.parameters.items():
                if param_name != 'self':
                    param_type = (
                        param.annotation 
                        if param.annotation != inspect.Parameter.empty 
                        else None
                    )
                    param_default = (
                        None 
                        if param.default == inspect.Parameter.empty 
                        else param.default
                    )
                    
                    inputs.append({
                        "name": param_name,
                        "type": str(param_type),
                        "default": param_default
                    })
            
            jinx_data = {
                "jinx_name": name,
                "description": doc.strip(),
                "inputs": inputs,
                "file_context": [],
                "steps": [
                    {
                        "name": "mcp_function_call",
                        "code": f"""
import {mcp_tool.__module__}
output = {mcp_tool.__module__}.{name}(
    {', '.join([
        f'{inp["name"]}=context.get("{inp["name"]}")' 
        for inp in inputs
    ])}
)
"""
                    }
                ]
            }
            
            return cls(jinx_data=jinx_data)
            
        except: 
            pass

def _parse_skill_md(path):
    """Parse a skill markdown file with YAML frontmatter and ## sections.

    Expected format:
        ---
        name: skill-name
        description: What it does. Use when ...
        ---

        Content for section one...

        Content for section two...

    Returns dict with name, description, sections, frontmatter or None on failure.
    """
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()

    if not content.startswith('---'):
        return None

    parts = content.split('---', 2)
    if len(parts) < 3:
        return None

    frontmatter = yaml.safe_load(parts[1])
    if not frontmatter or not isinstance(frontmatter, dict):
        return None

    body = parts[2].strip()

    sections = {}
    current_section = None
    current_content = []

    for line in body.split('\n'):
        if line.startswith('## '):
            if current_section:
                sections[current_section] = '\n'.join(current_content).strip()
            current_section = line[3:].strip()
            current_content = []
        elif current_section is not None:
            current_content.append(line)

    if current_section:
        sections[current_section] = '\n'.join(current_content).strip()

    basename = os.path.splitext(os.path.basename(path))[0]
    if basename.upper() == 'SKILL':
        default_name = os.path.basename(os.path.dirname(path))
    else:
        default_name = basename

    return {
        'name': frontmatter.get('name', default_name),
        'description': frontmatter.get('description', ''),
        'sections': sections,
        'frontmatter': frontmatter
    }

def _compile_skill_to_jinx(skill_data, source_path=None):
    """Compile skill data into a Jinx whose step is ``engine: skill``.

    Everything about the skill — name, description, sections, scripts,
    references, assets — is passed as structured data into ``skill.jinx``.
    The sub-jinx owns the full representation; the compiler just parses the
    source format and hands it off.

    Sections content is base64-encoded to survive the two-pass Jinja pipeline
    without mangling.  Metadata lists (scripts, references, assets) are passed
    as JSON strings.
    """
    name = skill_data.get('jinx_name', skill_data.get('name', ''))
    description = skill_data.get('description', '')
    sections = skill_data.get('sections', {})

    sections_json = json.dumps(sections, ensure_ascii=False)
    content_b64 = base64.b64encode(sections_json.encode('utf-8')).decode('ascii')

    section_names = list(sections.keys())
    desc_suffix = f" [Sections: {', '.join(section_names)}]" if section_names else ""

    scripts_list = []
    references_list = []
    assets_list = []

    file_context = list(skill_data.get('file_context', []))
    for subdir, collector in (('scripts', scripts_list),
                               ('references', references_list),
                               ('assets', assets_list)):
        entries = skill_data.get(subdir)
        if isinstance(entries, list):
            collector.extend(entries)
            for entry in entries:
                if isinstance(entry, str):
                    file_context.append({
                        'pattern': entry,
                        'base_path': os.path.join('.', subdir) if source_path else '.',
                    })
                elif isinstance(entry, dict):
                    file_context.append(entry)
        elif entries is None and source_path:
            skill_dir = os.path.dirname(source_path)
            subdir_path = os.path.join(skill_dir, subdir)
            if os.path.isdir(subdir_path):
                for r, _d, fnames in os.walk(subdir_path):
                    for fn in fnames:
                        rel = os.path.relpath(os.path.join(r, fn), skill_dir)
                        collector.append(rel)
                file_context.append({
                    'pattern': '*',
                    'base_path': subdir,
                    'recursive': True,
                })

    jinx_data = {
        'jinx_name': name,
        'description': (description + desc_suffix) if description else f"Skill: {name}{desc_suffix}",
        'inputs': [{'section': 'all'}],
        'steps': [{
            'engine': 'skill',
            'skill_name': name,
            'skill_description': description,
            'sections': content_b64,
            'scripts_json': json.dumps(scripts_list),
            'references_json': json.dumps(references_list),
            'assets_json': json.dumps(assets_list),
            'section': '{{section}}'
        }],
        'file_context': file_context,
        '_source_path': source_path
    }

    return Jinx(jinx_data=jinx_data)

def _load_skill_from_md(path):
    """Load a skill from a SKILL.md file with YAML frontmatter and ## sections.

    The skill name defaults to the parent folder name (matching the Anthropic
    skill-folder convention), falling back to the frontmatter ``name`` field.

    Sibling directories (``scripts/``, ``references/``, ``assets/``) are
    auto-discovered and attached as ``file_context``.
    """
    parsed = _parse_skill_md(path)
    if not parsed or not parsed.get('sections'):
        return None

    parent_dir = os.path.basename(os.path.dirname(path))
    name = parsed['name'] or parent_dir

    skill_data = {
        'jinx_name': name,
        'description': parsed['description'],
        'sections': parsed['sections'],
    }

    fm = parsed.get('frontmatter', {})
    for key in ('scripts', 'references', 'assets', 'file_context'):
        if key in fm:
            skill_data[key] = fm[key]

    return _compile_skill_to_jinx(skill_data, source_path=path)

def load_jinxes_from_directory(directory):
    """Load all jinxes from a directory recursively.

    Handles two file types:

    1. **.jinx** — regular jinxes loaded as-is.  Skill-type jinxes are just
       regular ``.jinx`` files whose steps use ``engine: skill`` with the
       full structured args (sections, scripts, references, assets).
    2. **SKILL.md inside a skill folder** — Anthropic-style skill folders::

           jinxes/skills/code-review/SKILL.md
           jinxes/skills/code-review/scripts/
           jinxes/skills/code-review/references/

       The folder name becomes the skill name.  The SKILL.md must have YAML
       frontmatter (``---`` delimiters) and ``##`` section headers.
       These are compiled into jinxes with ``engine: skill`` steps.
    """
    jinxes = []
    directory = os.path.expanduser(directory)

    if not os.path.exists(directory):
        return jinxes

    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".jinx"):
                try:
                    jinx_path = os.path.join(root, filename)
                    jinx = Jinx(jinx_path=jinx_path)
                    jinxes.append(jinx)
                except Exception as e:
                    print(f"Error loading jinx {filename}: {e}")
            elif filename == "SKILL.md":
                try:
                    md_path = os.path.join(root, filename)
                    jinx = _load_skill_from_md(md_path)
                    if jinx:
                        jinxes.append(jinx)
                except Exception as e:
                    skill_folder = os.path.basename(root)
                    print(f"Error loading skill from {skill_folder}/SKILL.md: {e}")

    return jinxes

def jinx_to_tool_def(jinx_obj: 'Jinx') -> Dict[str, Any]:
    """Convert a Jinx instance into an MCP/LLM-compatible tool schema definition."""
    return jinx_obj.to_tool_def()

def build_jinx_tool_catalog(jinxes: Dict[str, 'Jinx']) -> Dict[str, Dict[str, Any]]:
    """Helper to build a name->tool_def catalog from a dict of Jinx objects."""
    return {name: jinx_to_tool_def(jinx_obj) for name, jinx_obj in jinxes.items()}

def match_jinx_spec_to_names(jinx_spec: str, team_jinxes_dict: Dict[str, 'Jinx'], jinxes_base_dir: str, jinx_path_map: dict = None) -> List[str]:
    """
    Match a jinx spec to actual jinx names from the team's jinxes_dict.

    Specs can be:
    - A direct jinx name: 'edit_file', 'sh', 'python'
    - A relative path (resolved via {{ Jinx() }} in .npc files): 'lib/core/files/edit_file'
    - A glob pattern for bulk loading: 'lib/browser/*'

    Args:
        jinx_spec: The spec string
        team_jinxes_dict: Dict mapping jinx_name -> Jinx object
        jinxes_base_dir: Base directory where team jinxes are stored
        jinx_path_map: Optional dict mapping jinx_name -> relative path (for reverse lookup)

    Returns:
        List of jinx names that match the spec
    """
    if jinx_spec in team_jinxes_dict:
        return [jinx_spec]

    if jinx_path_map:
        for name, path in jinx_path_map.items():
            if jinx_spec == path and name in team_jinxes_dict:
                return [name]

    spec_pattern = jinx_spec
    if not spec_pattern.endswith('.jinx') and not spec_pattern.endswith('*'):
        spec_pattern += '.jinx'

    matched_names = []
    for jinx_name, jinx_obj in team_jinxes_dict.items():
        source_path = getattr(jinx_obj, '_source_path', None)
        if not source_path:
            continue

        try:
            rel_path = os.path.relpath(source_path, jinxes_base_dir)
        except ValueError:
            continue

        if fnmatch.fnmatch(rel_path, spec_pattern):
            matched_names.append(jinx_name)

    return matched_names

def extract_jinx_inputs(args: List[str], jinx: Jinx) -> Dict[str, Any]:
    inputs = {}

    flag_mapping = {}
    for input_ in jinx.inputs:
        if isinstance(input_, str):
            flag_mapping[f"-{input_[0]}"] = input_
            flag_mapping[f"--{input_}"] = input_
        elif isinstance(input_, dict):
            key = list(input_.keys())[0]
            flag_mapping[f"-{key[0]}"] = key
            flag_mapping[f"--{key}"] = key

    if len(jinx.inputs) > 1:
        used_args = set()
        for i, arg in enumerate(args):
            if '=' in arg and arg != '=' and not arg.startswith('-'):
                key, value = arg.split('=', 1)
                key = key.strip().strip("'\"")
                value = value.strip().strip("'\"")
                inputs[key] = value
                used_args.add(i)
    else:
        used_args = set()

    for i, arg in enumerate(args):
        if i in used_args:
            continue
            
        if arg in flag_mapping:
            if i + 1 < len(args) and not args[i + 1].startswith('-'):
                input_name = flag_mapping[arg]
                inputs[input_name] = args[i + 1]
                used_args.add(i)
                used_args.add(i + 1)
            else:
                input_name = flag_mapping[arg]
                inputs[input_name] = True
                used_args.add(i)

    unused_args = [arg for i, arg in enumerate(args) if i not in used_args]

    first_required = None
    for input_ in jinx.inputs:
        if isinstance(input_, str):
            first_required = input_
            break

    if first_required and unused_args:
        inputs[first_required] = ' '.join(unused_args).strip()
    else:
        jinx_input_names = []
        for input_ in jinx.inputs:
            if isinstance(input_, str):
                jinx_input_names.append(input_)
            elif isinstance(input_, dict):
                jinx_input_names.append(list(input_.keys())[0])
        
        if len(jinx_input_names) == 1 and unused_args:
            inputs[jinx_input_names[0]] = ' '.join(unused_args).strip()
        else:
            for i, arg in enumerate(unused_args):
                if i < len(jinx_input_names):
                    input_name = jinx_input_names[i]
                    if input_name not in inputs: 
                        inputs[input_name] = arg

    for input_ in jinx.inputs:
        if isinstance(input_, str):
            if input_ not in inputs:
                raise ValueError(f"Missing required input: {input_}")
        elif isinstance(input_, dict):
            key = list(input_.keys())[0]
            default_value = input_[key]
            if key not in inputs:
                inputs[key] = default_value

    return inputs
from npcpy.memory.knowledge_graph import kg_initial, kg_evolve_incremental, kg_sleep_process, kg_dream_process
try:
    from npcpy.memory.knowledge_manager import KnowledgeManager
except Exception:
    KnowledgeManager = None
try:
    from npcpy.memory.knowledge_store import KnowledgeStore
except Exception:
    KnowledgeStore = None
from npcpy.llm_funcs import get_llm_response, breathe
import os
from datetime import datetime
import json

class NPC:
    def __init__(
        self,
        file: str = None,
        name: str = None,
        primary_directive: str = None,
        plain_system_message: bool = False,
        team = None,
        jinxes: list = None,
        tools: list = None,
        model: str = None,
        provider: str = None,
        api_url: str = None,
        api_key: str = None,
        db_conn=None,
        memory = False,
        **kwargs
    ):
        """
        Initialize an NPC from a file path or with explicit parameters
        
        Args:
            file: Path to .npc file or name for the NPC
            primary_directive: System prompt/directive for the NPC
            jinxes: List of jinxes available to the NPC or "*" to load all jinxes
            model: LLM model to use
            provider: LLM provider to use
            api_url: API URL for LLM
            api_key: API key for LLM
            db_conn: Deprecated/no-op. Use initialize_db() to attach a DB explicitly.
        """
        if not file and not name and not primary_directive:
            raise ValueError("Either 'file' or 'name' and 'primary_directive' must be provided")

        self.team = team

        if file:
            if file.endswith(".npc"):
                self._load_from_file(file)
            file_parent = os.path.dirname(file)
            self.jinxes_directory = os.path.join(file_parent, "jinxes")
            self.npc_directory = file_parent
        else:
            self.name = name
            self.primary_directive = primary_directive
            self.model = model
            self.provider = provider
            self.api_url = api_url
            self.api_key = api_key
            self._extra_fields = kwargs

            self.jinxes_directory = None
            self.npc_directory = None

        if not hasattr(self, 'jinxes_spec') or jinxes is not None:
            self.jinxes_spec = jinxes or []

        if tools is not None:
            tools_schema, tool_map = auto_tools(tools)
            self.tools = tools_schema  
            self.tool_map = tool_map   
            self.tools_schema = tools_schema  
        else:
            self.tools = []
            self.tool_map = {}
            self.tools_schema = []
        self.plain_system_message = plain_system_message
        self.jinx_tool_catalog: Dict[str, Dict[str, Any]] = {}
        self.mcp_servers = []
        
        self.memory_length = 20
        self.memory_strategy = 'recent'
        dirs = []
        if self.npc_directory:
            dirs.append(self.npc_directory)
        if self.jinxes_directory:
            dirs.append(self.jinxes_directory)
            
        self.jinja_env = SandboxedEnvironment(
            loader=FileSystemLoader([
                os.path.expanduser(d) for d in dirs
            ]),
            undefined=SilentUndefined,
        )
        
        self.db_conn = db_conn
        self.kg_data = None
        self.tables = None
        self.memory = None
        self.knowledge_manager = None
        self.knowledge_scopes = []

        self.jinxes_dict = {}
        if jinxes and jinxes != "*": 
            for jinx_item in jinxes:
                if isinstance(jinx_item, Jinx):
                    self.jinxes_dict[jinx_item.jinx_name] = jinx_item
                elif isinstance(jinx_item, dict):
                    jinx_obj = Jinx(jinx_data=jinx_item)
                    self.jinxes_dict[jinx_obj.jinx_name] = jinx_obj
                elif isinstance(jinx_item, str):
                    jinx_path = find_file_path(jinx_item, [self.npc_jinxes_directory], suffix=".jinx")
                    if jinx_path:
                        jinx_obj = Jinx(jinx_path=jinx_path)
                        self.jinxes_dict[jinx_obj.jinx_name] = jinx_obj
                    else:
                        print(f"Warning: Jinx '{jinx_item}' not found for NPC '{self.name}' during initial load.")
        
        self.shared_context = {
            "dataframes": {},
            "current_data": None,
            "computation_results": [],
            "locals": {},

            "memories": {},

            "mcp_client": None,
            "mcp_tools": [],
            "mcp_tool_map": {},

            "session_input_tokens": 0,
            "session_output_tokens": 0,
            "session_cost_usd": 0.0,
            "turn_count": 0,

            "current_mode": "agent",
            "attachments": [],
        }
        
        for key, value in kwargs.items():
            setattr(self, key, value)
            
    def initialize_jinxes(self, team_raw_jinxes: Optional[List['Jinx']] = None):
        """
        Loads and performs first-pass Jinja rendering for NPC-specific jinxes,
        now that the NPC's team context is fully established.
        """
        print(f"[JINX] initialize_jinxes called for NPC '{self.name}'")
        print(f"[JINX] NPC '{self.name}' jinxes_spec: {self.jinxes_spec}")
        if self.team:
            print(f"[JINX] NPC '{self.name}' team: {self.team.name} team_path: {getattr(self.team, 'team_path', None)}")
            print(f"[JINX] NPC '{self.name}' team jinxes_dict keys: {list(self.team.jinxes_dict.keys())}")
        else:
            print(f"[JINX] NPC '{self.name}' has NO TEAM")
        npc_jinxes_raw_list = []
        
        if self.jinxes_spec == "*":
            if self.team and hasattr(self.team, 'jinxes_dict') and self.team.jinxes_dict:
                self.jinxes_dict.update(self.team.jinxes_dict)
        else:
            if self.team and hasattr(self.team, 'jinxes_dict') and self.team.jinxes_dict:
                jinxes_base_dir = None
                if hasattr(self.team, 'team_path') and self.team.team_path:
                    jinxes_base_dir = os.path.join(self.team.team_path, 'jinxes')

                path_map = getattr(self.team, '_jinx_path_map', None)
                for jinx_spec in self.jinxes_spec:
                    if jinxes_base_dir:
                        matched_names = match_jinx_spec_to_names(jinx_spec, self.team.jinxes_dict, jinxes_base_dir, jinx_path_map=path_map)
                    else:
                        matched_names = [jinx_spec] if jinx_spec in self.team.jinxes_dict else []

                    if not matched_names:
                        print(
                            f"Warning: NPC '{self.name}' references jinx '{jinx_spec}' but no matching jinx was found. "
                            f"Skipping. Available jinxes (first 20): {list(self.team.jinxes_dict.keys())[:20]}",
                            file=sys.stderr,
                        )
                        continue

                    for jinx_name in matched_names:
                        if jinx_name in self.team.jinxes_dict:
                            self.jinxes_dict[jinx_name] = self.team.jinxes_dict[jinx_name]

        team_jinxes_spec = getattr(self.team, 'team_jinxes_spec', None) if self.team else None
        if team_jinxes_spec and hasattr(self.team, 'jinxes_dict') and self.team.jinxes_dict:
            jinxes_base_dir = None
            if hasattr(self.team, 'team_path') and self.team.team_path:
                jinxes_base_dir = os.path.join(self.team.team_path, 'jinxes')
            path_map = getattr(self.team, '_jinx_path_map', None)
            for jinx_spec in team_jinxes_spec:
                if jinxes_base_dir:
                    matched_names = match_jinx_spec_to_names(jinx_spec, self.team.jinxes_dict, jinxes_base_dir, jinx_path_map=path_map)
                else:
                    matched_names = [jinx_spec] if jinx_spec in self.team.jinxes_dict else []
                for jinx_name in matched_names:
                    if jinx_name in self.team.jinxes_dict and jinx_name not in self.jinxes_dict:
                        self.jinxes_dict[jinx_name] = self.team.jinxes_dict[jinx_name]

        should_load_from_directory = False
        if hasattr(self, 'npc_jinxes_directory') and self.npc_jinxes_directory and os.path.exists(self.npc_jinxes_directory):
            if not self.team:
                should_load_from_directory = True
            elif hasattr(self.team, 'team_path') and self.team.team_path:
                team_jinxes_dir = os.path.join(self.team.team_path, 'jinxes')
                if os.path.normpath(self.npc_jinxes_directory) != os.path.normpath(team_jinxes_dir):
                    should_load_from_directory = True

        if should_load_from_directory:
            for jinx_obj in load_jinxes_from_directory(self.npc_jinxes_directory):
                if jinx_obj.jinx_name not in self.jinxes_dict:
                    npc_jinxes_raw_list.append(jinx_obj)
        
        if npc_jinxes_raw_list or team_raw_jinxes:
            all_available_raw_jinxes = list(team_raw_jinxes or [])
            all_available_raw_jinxes.extend(npc_jinxes_raw_list)

            combined_raw_jinxes_dict = {j.jinx_name: j for j in all_available_raw_jinxes}

            npc_first_pass_jinja_env = SandboxedEnvironment(undefined=SilentUndefined)

            jinx_macro_globals = {}
            for raw_jinx in combined_raw_jinxes_dict.values():
                def create_jinx_callable(jinx_obj_in_closure):
                    def callable_jinx(**kwargs):
                        temp_jinja_env = SandboxedEnvironment(undefined=SilentUndefined)
                        rendered_target_steps = []
                        for target_step in jinx_obj_in_closure._raw_steps:
                            temp_rendered_step = {}
                            for k, v in target_step.items():
                                if isinstance(v, str):
                                    try:
                                        temp_rendered_step[k] = temp_jinja_env.from_string(v).render(**kwargs)
                                    except Exception as e:
                                        print(f"Warning: Error in Jinx macro '{jinx_obj_in_closure.jinx_name}' rendering step field '{k}' (NPC first pass): {e}")
                                        temp_rendered_step[k] = v
                                else:
                                    temp_rendered_step[k] = v
                            rendered_target_steps.append(temp_rendered_step)
                        return yaml.dump(rendered_target_steps, default_flow_style=False)
                    return callable_jinx
                
                jinx_macro_globals[raw_jinx.jinx_name] = create_jinx_callable(raw_jinx)
            
            npc_first_pass_jinja_env.globals.update(jinx_macro_globals)

            for raw_npc_jinx in npc_jinxes_raw_list:
                try:
                    raw_npc_jinx.render_first_pass(npc_first_pass_jinja_env, jinx_macro_globals)
                    self.jinxes_dict[raw_npc_jinx.jinx_name] = raw_npc_jinx
                except Exception as e:
                    print(f"Error performing first-pass rendering for NPC Jinx '{raw_npc_jinx.jinx_name}': {e}")
        
        self.jinx_tool_catalog = build_jinx_tool_catalog(self.jinxes_dict)
        print(f"NPC {self.name} loaded {len(self.jinxes_dict)} jinxes and built catalog with {len(self.jinx_tool_catalog)} tools.", file=sys.stderr)


    def get_memory_context(self):
        """Get formatted memory context for system prompt using KnowledgeManager."""
        parts = []

        if hasattr(self, 'knowledge_store') and self.knowledge_store:
            try:
                local_ctx = self.knowledge_store.build_context(max_memories=10)
                if local_ctx:
                    parts.append(local_ctx)
            except Exception as e:
                logger.warning(f".knowledge.yaml context failed for {self.name}: {e}")

        if self.knowledge_manager and self.knowledge_scopes:
            try:
                db_ctx = self.knowledge_manager.get_memory_context(
                    self.knowledge_scopes, max_facts=10, max_memories=5
                )
                if db_ctx:
                    parts.append(db_ctx)
            except Exception as e:
                logger.warning(f"KnowledgeManager context failed for {self.name}: {e}")

        if not parts and self.kg_data:
            recent_facts = self.kg_data.get('facts', [])[-10:]
            if recent_facts:
                parts.append("Recent memories:")
                for fact in recent_facts:
                    parts.append(f"- {fact['statement']}")
            concepts = self.kg_data.get('concepts', [])
            if concepts:
                concept_names = [c['name'] for c in concepts[:5]]
                parts.append(f"Key concepts: {', '.join(concept_names)}")

        return "\n\n".join(parts) if parts else ""

    def enter_tool_use_loop(
        self, 
        prompt: str, 
        tools: list = None, 
        tool_map: dict = None, 
        max_iterations: int = 5,
        stream: bool = False
    ):
        """Enter interactive tool use loop for complex tasks"""
        if not tools:
            tools = self.tools
        if not tool_map:
            tool_map = self.tool_map
            
        messages = self.memory.copy() if self.memory else []
        messages.append({"role": "user", "content": prompt})
        
        for iteration in range(max_iterations):
            response = get_llm_response(
                prompt="",
                model=self.model,
                provider=self.provider,
                npc=self,
                messages=messages,
                tools=tools,
                tool_map=tool_map,
                auto_process_tool_calls=True,
                stream=stream
            )
            
            messages = response.get('messages', messages)
            
            if not response.get('tool_calls'):
                return {
                    "final_response": response.get('response'),
                    "messages": messages,
                    "iterations": iteration + 1
                }
                
        return {
            "final_response": "Max iterations reached",
            "messages": messages,
            "iterations": max_iterations
        }

    def get_code_response(
        self, 
        prompt: str, 
        language: str = "python", 
        execute: bool = False, 
        locals_dict: dict = None
    ):
        """Generate and optionally execute code responses"""
        code_prompt = f"""Generate {language} code for: {prompt}
        
        Provide ONLY executable {language} code without explanations.
        Do not include markdown formatting or code blocks.
        Begin directly with the code."""
        
        response = get_llm_response(
            prompt=code_prompt,
            model=self.model,
            provider=self.provider,
            npc=self,
            stream=False
        )
        
        generated_code = response.get('response', '')
        
        result = {
            "code": generated_code,
            "executed": False,
            "output": None,
            "error": None
        }
        
        if execute and language == "python":
            if locals_dict is None:
                locals_dict = {}
                
            exec_globals = {"__builtins__": __builtins__}
            exec_globals.update(locals_dict)
            
            exec_locals = {}
            try:
                exec(generated_code, exec_globals, exec_locals)
            except SystemExit as e:
                result["error"] = f"Code called sys.exit({e.code})"
                return result

            locals_dict.update(exec_locals)
            result["executed"] = True
            result["output"] = exec_locals.get("output", "Code executed successfully")
        
        return result

    def _load_from_file(self, file):
        """Load NPC configuration from file"""
        if "~" in file:
            file = os.path.expanduser(file)
        if not os.path.isabs(file):
            file = os.path.abspath(file)

        jinja_ctx = None
        if self.team and hasattr(self.team, '_npc_jinja_context'):
            jinja_ctx = self.team._npc_jinja_context

        npc_data = load_yaml_file(file, jinja_context=jinja_ctx)
        if not npc_data:
            raise ValueError(f"Failed to load NPC from {file}")
            
        self.name = npc_data.get("name")
        if not self.name:
            self.name = os.path.splitext(os.path.basename(file))[0]
            
        self.primary_directive = npc_data.get("primary_directive")
        
        jinxes_spec = npc_data.get("jinxes", [])

        if jinxes_spec == "*":
            self.jinxes_spec = "*"
        else:
            self.jinxes_spec = jinxes_spec

        self.model = npc_data.get("model")
        self.provider = npc_data.get("provider")
        self.api_url = npc_data.get("api_url")
        self.api_key = npc_data.get("api_key")
        self.mcp_servers = npc_data.get("mcp_servers", [])

        if self.team:
            if not self.model and hasattr(self.team, 'model'):
                self.model = self.team.model
            if not self.provider and hasattr(self.team, 'provider'):
                self.provider = self.team.provider
            if not self.api_url and hasattr(self.team, 'api_url'):
                self.api_url = self.team.api_url
            if not self.api_key and hasattr(self.team, 'api_key'):
                self.api_key = self.team.api_key

            if self.provider and hasattr(self.team, 'providers') and isinstance(self.team.providers, list):
                for prov in self.team.providers:
                    if isinstance(prov, dict) and prov.get('name') == self.provider:
                        if not self.api_url and prov.get('api_url'):
                            self.api_url = os.path.expandvars(prov['api_url'])
                        if not self.api_key and prov.get('api_key'):
                            self.api_key = os.path.expandvars(prov['api_key'])
                        if not self.model and prov.get('model'):
                            self.model = prov['model']
                        break


        self.name = npc_data.get("name", self.name)

        known_keys = {
            "name", "primary_directive", "jinxes", "model", "provider",
            "api_url", "api_key", "mcp_servers", "plain_system_message",
            "tools", "team", "memory",
        }
        self._extra_fields = {k: v for k, v in npc_data.items() if k not in known_keys}

        self.npc_path = file
        self.npc_jinxes_directory = os.path.join(os.path.dirname(file), "jinxes")

    def resolve_tools(self, mcp_clients_cache: dict = None) -> tuple:
        """
        Returns (tools_for_llm, tool_executors) where:
          - tools_for_llm: list of OpenAI-style tool defs for the LLM
          - tool_executors: dict mapping tool_name -> {"type": ..., ...} for execution

        Assembles from: jinx_tool_catalog + mcp_servers + python tools
        """
        from npcpy.serve import MCPClientNPC

        tools_for_llm = []
        tool_executors = {}
        seen = set()

        print(f"[TOOLS] resolve_tools called for NPC '{self.name}'")
        print(f"[TOOLS] NPC '{self.name}' jinxes_dict has {len(self.jinxes_dict)} entries: {list(self.jinxes_dict.keys())}")
        print(f"[TOOLS] NPC '{self.name}' jinx_tool_catalog has {len(self.jinx_tool_catalog or {})} entries")
        print(f"[TOOLS] NPC '{self.name}' mcp_servers: {self.mcp_servers}")

        catalog = self.jinx_tool_catalog or build_jinx_tool_catalog(self.jinxes_dict)
        print(f"[TOOLS] Built catalog with {len(catalog)} jinx entries")
        for name, tool_def in catalog.items():
            if name not in seen:
                tools_for_llm.append(tool_def)
                tool_executors[name] = {"type": "jinx", "jinx": self.jinxes_dict.get(name)}
                seen.add(name)

        if mcp_clients_cache is None:
            mcp_clients_cache = {}

        connectable_specs = []
        nameonly_tools = []
        for server_spec in (self.mcp_servers or []):
            if isinstance(server_spec, str):
                connectable_specs.append({"path": server_spec})
            elif isinstance(server_spec, dict):
                if "path" in server_spec or "command" in server_spec or "url" in server_spec:
                    connectable_specs.append(server_spec)
                elif "tools" in server_spec:
                    nameonly_tools.extend(server_spec["tools"])

        def spec_cache_key(spec):
            if isinstance(spec, str):
                return os.path.expanduser(spec)
            if "path" in spec:
                return os.path.expanduser(spec["path"])
            if "url" in spec:
                return spec["url"]
            if "command" in spec:
                return f"{spec['command']}:{' '.join(spec.get('args', []))}"
            return str(spec)

        for spec in connectable_specs:
            key = spec_cache_key(spec)
            whitelist = spec.get("tools")
            client = mcp_clients_cache.get(key)
            if client and not client.is_connected():
                client.disconnect_sync()
                mcp_clients_cache.pop(key, None)
                client = None
            if not client:
                client = MCPClientNPC()
                if client.connect_sync(spec):
                    mcp_clients_cache[key] = client
                else:
                    continue
            for tool_def in client.available_tools_llm:
                name = tool_def["function"]["name"]
                if name in seen:
                    continue
                if whitelist and name not in whitelist:
                    continue
                tools_for_llm.append(tool_def)
                tool_executors[name] = {
                    "type": "mcp",
                    "client": client,
                    "tool_func": client.tool_map.get(name),
                }
                seen.add(name)

        if nameonly_tools and self.team and hasattr(self.team, "mcp_servers"):
            for team_server in (self.team.mcp_servers or []):
                ts = team_server if isinstance(team_server, dict) else {"path": team_server}
                key = spec_cache_key(ts)
                client = mcp_clients_cache.get(key)
                if not client:
                    client = MCPClientNPC()
                    if client.connect_sync(ts):
                        mcp_clients_cache[key] = client
                    else:
                        continue
                for tool_def in client.available_tools_llm:
                    name = tool_def["function"]["name"]
                    if name in nameonly_tools and name not in seen:
                        tools_for_llm.append(tool_def)
                        tool_executors[name] = {
                            "type": "mcp",
                            "client": client,
                            "tool_func": client.tool_map.get(name),
                        }
                        seen.add(name)

        for tool_def in (self.tools_schema or []):
            name = tool_def["function"]["name"]
            if name not in seen:
                tools_for_llm.append(tool_def)
                tool_executors[name] = {"type": "python", "func": self.tool_map.get(name)}
                seen.add(name)

        return tools_for_llm, tool_executors

    def get_system_prompt(self, simple=False, tool_capable=False):
        """Get system prompt for the NPC"""
        if simple or self.plain_system_message:
            return self.primary_directive
        else:
            return get_system_message(self, team=self.team, tool_capable=tool_capable)

    def initialize_db(self, db_conn):
        """Explicitly attach a database connection for DB-dependent features."""
        if db_conn is None:
            return
        self.db_conn = db_conn
        self._setup_db()

    def _setup_db(self):
        """Set up database tables and determine type, and initialize knowledge manager."""
        dialect = self.db_conn.dialect.name

        with self.db_conn.connect() as conn:
            if dialect == "postgresql":
                result = conn.execute(text("""
                    SELECT table_name, obj_description((quote_ident(table_name))::regclass, 'pg_class')
                    FROM information_schema.tables
                    WHERE table_schema='public';
                """))
                self.tables = result.fetchall()
                self.db_type = "postgres"

            elif dialect == "sqlite":
                result = conn.execute(text(
                    "SELECT name, sql FROM sqlite_master WHERE type='table';"
                ))
                self.tables = result.fetchall()
                self.db_type = "sqlite"

            else:
                print(f"Unsupported DB dialect: {dialect}")
                self.tables = None
                self.db_type = None

        if KnowledgeManager and self.db_conn:
            try:
                self.knowledge_manager = KnowledgeManager(self.db_conn)
                scopes = ["global"]
                if self.name:
                    scopes.append(f"npc:{self.name}")
                if self.team and hasattr(self.team, 'name') and self.team.name:
                    scopes.append(f"team:{self.team.name}")
                self.knowledge_scopes = scopes
                primary_scope = scopes[-1] if len(scopes) > 1 else "global"
                self.kg_data = self.knowledge_manager.load_kg(primary_scope)
            except Exception as e:
                logger.warning(f"Failed to initialize KnowledgeManager for NPC {self.name}: {e}")

        if KnowledgeStore:
            try:
                search_dirs = []
                if self.npc_directory:
                    search_dirs.append(self.npc_directory)
                if self.team and hasattr(self.team, 'team_path') and self.team.team_path:
                    search_dirs.append(self.team.team_path)
                for d in search_dirs:
                    if os.path.exists(os.path.join(d, ".knowledge.yaml")):
                        self.knowledge_store = KnowledgeStore(d)
                        break
            except Exception as e:
                logger.warning(f"Failed to load .knowledge.yaml for NPC {self.name}: {e}")

    def get_llm_response(self, 
                        request,
                        jinxes=None,
                        tools: Optional[list] = None,
                        tool_map: Optional[dict] = None,
                        tool_choice=None, 
                        messages=None,
                        auto_process_tool_calls=True,
                        use_core_tools: bool = False,
                        **kwargs):
        all_candidate_functions = []

        if tools is not None and tool_map is not None:
            all_candidate_functions.extend([func for func in tool_map.values() if callable(func)])
        elif hasattr(self, 'tool_map') and self.tool_map:
            all_candidate_functions.extend([func for func in self.tool_map.values() if callable(func)])

        if use_core_tools:
            dynamic_core_tools_list = [
                self.think_step_by_step,
                self.write_code,
            ]

            if self.db_conn:
                dynamic_core_tools_list.append(self.query_database)

            all_candidate_functions.extend(dynamic_core_tools_list)

        unique_functions = []
        seen_names = set()
        for func in all_candidate_functions:
            if func.__name__ not in seen_names:
                unique_functions.append(func)
                seen_names.add(func.__name__)

        final_tools_schema = None
        final_tool_map_dict = None

        if unique_functions:
            final_tools_schema, final_tool_map_dict = auto_tools(unique_functions)

        if tool_choice is None:
            if final_tools_schema:
                tool_choice = "auto"
            else:
                tool_choice = None

        response = npy.llm_funcs.get_llm_response(
            request, 
            npc=self, 
            jinxes=jinxes,
            tools=final_tools_schema,
            tool_map=final_tool_map_dict,
            tool_choice=tool_choice,           
            auto_process_tool_calls=auto_process_tool_calls,
            messages=self.memory if messages is None else messages,
            **kwargs
        )        

        return response
    

    def search_my_memories(self, query: str, limit: int = 10) -> str:
        """Search through this NPC's knowledge graph memories for relevant facts and concepts"""
        if self.knowledge_manager and self.knowledge_scopes:
            try:
                result = self.knowledge_manager.search_across_scopes(
                    query, self.knowledge_scopes, top_k=limit
                )
                facts = result.get("facts", [])
                memories = result.get("memories", [])
                parts = []
                if facts:
                    parts.append(f"Relevant facts: {'; '.join(f['statement'] for f in facts[:limit])}")
                if memories:
                    parts.append(f"Approved memories: {'; '.join(m['text'] for m in memories[:limit])}")
                return "\n".join(parts) if parts else f"No memories found matching '{query}'"
            except Exception as e:
                logger.warning(f"KnowledgeManager search failed for {self.name}: {e}")

        if not self.kg_data:
            return "No memories available"
        query_lower = query.lower()
        relevant_facts = []
        relevant_concepts = []
        for fact in self.kg_data.get('facts', []):
            if query_lower in fact.get('statement', '').lower():
                relevant_facts.append(fact['statement'])
        for concept in self.kg_data.get('concepts', []):
            if query_lower in concept.get('name', '').lower():
                relevant_concepts.append(concept['name'])
        result_parts = []
        if relevant_facts:
            result_parts.append(f"Relevant memories: {'; '.join(relevant_facts[:limit])}")
        if relevant_concepts:
            result_parts.append(f"Related concepts: {', '.join(relevant_concepts[:limit])}")
        return "\n".join(result_parts) if result_parts else f"No memories found matching '{query}'"

    def query_database(self, sql_query: str) -> str:
        """Execute a SQL query against the available database"""
        if not self.db_conn:
            return "No database connection available"
        
        try:
            with self.db_conn.connect() as conn:
                result = conn.execute(text(sql_query))
                rows = result.fetchall()
                
                if not rows:
                    return "Query executed successfully but returned no results"
                
                columns = result.keys()
                formatted_rows = []
                for row in rows[:20]:  
                    row_dict = dict(zip(columns, row))
                    formatted_rows.append(str(row_dict))
                
                return f"Query results ({len(rows)} total rows, showing first 20):\n" + "\n".join(formatted_rows)
        
        except Exception as e:
            return f"Database query error: {str(e)}"

    def think_step_by_step(self, problem: str) -> str:
        """Think through a problem step by step using chain of thought reasoning"""
        thinking_prompt = f"""Think through this problem step by step:

    {problem}

    Break down your reasoning into clear steps:
    1. First, I need to understand...
    2. Then, I should consider...
    3. Next, I need to...
    4. Finally, I can conclude...

    Provide your step-by-step analysis.
    Do not under any circumstances ask for feedback from a user. These thoughts are part of an agentic tool that is letting the agent
    break down a problem by thinking it through. they will review the results and use them accordingly. 

    
    """
        
        response = self.get_llm_response(thinking_prompt, tool_choice = False)
        return response.get('response', 'Unable to process thinking request')

    def write_code(self, task: str, language: str = "python") -> str:
        """Write code to accomplish a task.

        Args:
            task: Description of what the code should do
            language: Programming language to use (default: python)

        Returns:
            The generated code as a string
        """
        code_prompt = f"""Write {language} code to accomplish the following task:

{task}

Requirements:
- Write clean, well-commented code
- Include error handling where appropriate
- Make sure the code is complete and runnable
- Only output the code, no explanations before or after

```{language}
"""

        response = self.get_llm_response(code_prompt, tool_choice=False)
        code = response.get('response', '')

        if f'```{language}' in code:
            code = code.split(f'```{language}')[-1]
        if '```' in code:
            code = code.split('```')[0]

        return code.strip()

    def create_planning_state(self, goal: str) -> Dict[str, Any]:
        """Create initial planning state for a goal"""
        return {
            "goal": goal,
            "todos": [],
            "constraints": [],
            "facts": [],
            "mistakes": [],
            "successes": [],
            "current_todo_index": 0,
            "current_subtodo_index": 0,
            "context_summary": ""
        }

    def generate_todos(self, user_goal: str, planning_state: Dict[str, Any], additional_context: str = "") -> List[Dict[str, Any]]:
        """Generate high-level todos for a goal"""
        prompt = f"""
        You are a high-level project planner. Structure tasks logically:
        1. Understand current state
        2. Make required changes 
        3. Verify changes work

        User goal: {user_goal}
        {additional_context}
        
        Generate 3-5 todos to accomplish this goal. Use specific actionable language.
        Each todo should be independent where possible and focused on a single component.
        
        Return JSON:
        {{
            "todos": [
                {{"description": "todo description", "estimated_complexity": "simple|medium|complex"}},
                ...
            ]
        }}
        """
        
        response = self.get_llm_response(prompt, format="json", tool_choice=False)
        todos_data = response.get("response", {}).get("todos", [])
        return todos_data

    def should_break_down_todo(self, todo: Dict[str, Any]) -> bool:
        """Ask LLM if a todo needs breakdown"""
        prompt = f"""
        Todo: {todo['description']}
        Complexity: {todo.get('estimated_complexity', 'unknown')}
        
        Should this be broken into smaller steps? Consider:
        - Is it complex enough to warrant breakdown?
        - Would breakdown make execution clearer?
        - Are there multiple distinct steps?
        
        Return JSON: {{"should_break_down": true/false, "reason": "explanation"}}
        """
        
        response = self.get_llm_response(prompt, format="json", tool_choice=False)
        result = response.get("response", {})
        return result.get("should_break_down", False)

    def generate_subtodos(self, todo: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate atomic subtodos for a complex todo"""
        prompt = f"""
        Parent todo: {todo['description']}
        
        Break this into atomic, executable subtodos. Each should be:
        - A single, concrete action
        - Executable in one step
        - Clear and unambiguous
        
        Return JSON:
        {{
            "subtodos": [
                {{"description": "subtodo description", "type": "action|verification|analysis"}},
                ...
            ]
        }}
        """
        
        response = self.get_llm_response(prompt, format="json")
        return response.get("response", {}).get("subtodos", [])

    def execute_planning_item(self, item: Dict[str, Any], planning_state: Dict[str, Any], context: str = "") -> Dict[str, Any]:
        """Execute a single planning item (todo or subtodo)"""
        context_summary = self.get_planning_context_summary(planning_state)
        
        command = f"""
        Current context:
        {context_summary}
        {context}
        
        Execute this task: {item['description']}
        
        Constraints to follow:
        {chr(10).join([f"- {c}" for c in planning_state.get('constraints', [])])}
        """
        
        result = self.check_llm_command(
            command,
            context=self.shared_context,
            stream=False
        )
        
        return result

    def get_planning_context_summary(self, planning_state: Dict[str, Any]) -> str:
        """Get lightweight context for planning prompts"""
        context = []
        facts = planning_state.get('facts', [])
        mistakes = planning_state.get('mistakes', [])
        successes = planning_state.get('successes', [])
        
        if facts:
            context.append(f"Facts: {'; '.join(facts[:5])}")
        if mistakes:
            context.append(f"Recent mistakes: {'; '.join(mistakes[-3:])}")
        if successes:
            context.append(f"Recent successes: {'; '.join(successes[-3:])}")
        return "\n".join(context)

    def compress_planning_state(self, messages):
        if isinstance(messages, list):
            from npcpy.llm_funcs import breathe, get_facts
            
            conversation_summary = breathe(messages=messages, npc=self)
            summary_data = conversation_summary.get('output', '')
            
            conversation_text = "\n".join([msg['content'] for msg in messages])
            from npcpy.llm_funcs import CONVERSATION_RULES
            extracted_facts = get_facts(conversation_text, model=self.model, provider=self.provider, npc=self, rules=CONVERSATION_RULES)
            
            user_inputs = [msg['content'] for msg in messages if msg.get('role') == 'user']
            assistant_outputs = [msg['content'] for msg in messages if msg.get('role') == 'assistant']
            
            planning_state = {
                "goal": summary_data,
                "facts": [fact['statement'] if isinstance(fact, dict) else str(fact) for fact in extracted_facts[-10:]],
                "successes": [output[:100] for output in assistant_outputs[-5:]],
                "mistakes": [],
                "todos": user_inputs[-3:],
                "constraints": []
            }
        else:
            planning_state = messages
        
        todos = planning_state.get('todos', [])
        current_index = planning_state.get('current_todo_index', 0)
        
        if todos and current_index < len(todos):
            current_focus = todos[current_index].get('description', todos[current_index]) if isinstance(todos[current_index], dict) else str(todos[current_index])
        else:
            current_focus = 'No current task'
        
        compressed = {
            "goal": planning_state.get("goal", ""),
            "progress": f"{len(planning_state.get('successes', []))}/{len(todos)} todos completed",
            "context": self.get_planning_context_summary(planning_state),
            "current_focus": current_focus
        }
        return json.dumps(compressed, indent=2)

    def decompress_planning_state(self, compressed_state: str) -> Dict[str, Any]:
        """Restore planning state from compressed string"""
        try:
            data = json.loads(compressed_state)
            return {
                "goal": data.get("goal", ""),
                "todos": [],
                "constraints": [],
                "facts": [],
                "mistakes": [],
                "successes": [],
                "current_todo_index": 0,
                "current_subtodo_index": 0,
                "compressed_context": data.get("context", "")
            }
        except json.JSONDecodeError:
            return self.create_planning_state("")

    def run_planning_loop(self, user_goal: str, interactive: bool = True) -> Dict[str, Any]:
        """Run the full planning loop for a goal"""
        planning_state = self.create_planning_state(user_goal)
        
        todos = self.generate_todos(user_goal, planning_state)
        planning_state["todos"] = todos
        
        for i, todo in enumerate(todos):
            planning_state["current_todo_index"] = i
            
            if self.should_break_down_todo(todo):
                subtodos = self.generate_subtodos(todo)
                
                for j, subtodo in enumerate(subtodos):
                    planning_state["current_subtodo_index"] = j
                    result = self.execute_planning_item(subtodo, planning_state)
                    
                    if result.get("output"):
                        planning_state["successes"].append(f"Completed: {subtodo['description']}")
                    else:
                        planning_state["mistakes"].append(f"Failed: {subtodo['description']}")
            else:
                result = self.execute_planning_item(todo, planning_state)
                
                if result.get("output"):
                    planning_state["successes"].append(f"Completed: {todo['description']}")
                else:
                    planning_state["mistakes"].append(f"Failed: {todo['description']}")
        
        return {
            "planning_state": planning_state,
            "compressed_state": self.compress_planning_state(planning_state),
            "summary": f"Completed {len(planning_state['successes'])} tasks for goal: {user_goal}"
        }
        
    def execute_jinx(
        self,
        jinx_name,
        inputs,
        conversation_id=None,
        message_id=None,
        team_name=None,
        extra_globals=None
    ):
        if jinx_name in self.jinxes_dict:
            jinx = self.jinxes_dict[jinx_name]
        else:
            return {"error": f"jinx '{jinx_name}' not found"}

        import time as _time
        _start = _time.monotonic()
        _status = "success"
        _error = None

        try:
            result = jinx.execute(
                input_values=inputs,
                npc=self,
                extra_globals=extra_globals,
                jinja_env=self.jinja_env
            )
        except Exception as e:
            _status = "error"
            _error = str(e)
            result = {"error": str(e)}

        _duration_ms = int((_time.monotonic() - _start) * 1000)
        return result
    def check_llm_command(self,
                            command,
                            messages=None,
                            context=None,
                            team=None,
                            stream=False,
                            jinxes=None,
                            use_jinxes=True,
                            **kwargs):
        """Check if a command is for the LLM"""
        if context is None:
            context = self.shared_context

        if team:
            self._current_team = team
        if jinxes is None and use_jinxes:
            jinxes_to_use = self.jinxes_dict
        elif jinxes is not None and use_jinxes:
            jinxes_to_use = jinxes

        return npy.llm_funcs.check_llm_command(
            command,
            model=self.model,
            provider=self.provider,
            npc=self,
            team=team,
            messages=self.memory if messages is None else messages,
            context=context,
            stream=stream,
            jinxes=jinxes_to_use,
            **kwargs,
        )

    def run(self, input_text: str, **kwargs):
        return self.check_llm_command(input_text, **kwargs)

    def handle_agent_pass(self, 
                            npc_to_pass,
                            command, 
                            messages=None, 
                            context=None, 
                            shared_context=None, 
                            stream=False,
                            team=None):  
        """Pass a command to another NPC"""
        print('handling agent pass')
        if isinstance(npc_to_pass, NPC):
            target_npc = npc_to_pass
        else:
            return {"error": "Invalid NPC to pass command to"}
        
        if shared_context is not None:
            self.shared_context.update(shared_context)
            target_npc.shared_context.update(shared_context)
            
        updated_command = (
            command
            + "\n\n"
            + f"NOTE: THIS COMMAND HAS BEEN PASSED FROM {self.name} TO YOU, {target_npc.name}.\n"
            + "PLEASE CHOOSE ONE OF THE OTHER OPTIONS WHEN RESPONDING."
        )

        result = target_npc.check_llm_command(
            updated_command,
            messages=messages,
            context=target_npc.shared_context,
            team=team, 
            stream=stream
        )
        if isinstance(result, dict):
            result['npc_name'] = target_npc.name
            result['passed_from'] = self.name
        
        return result    

    def to_dict(self):
        """Convert NPC to dictionary representation"""
        jinx_rep = []
        if self.jinxes_dict:
            jinx_rep = [ jinx.to_dict() for jinx in self.jinxes_dict.values()]
        source_path = getattr(self, 'npc_path', None) or getattr(self, 'source_path', None) or ''
        source_ext = ''
        if source_path:
            lower = source_path.lower()
            if lower.endswith('.npc'):
                source_ext = '.npc'
            elif lower.endswith('.md'):
                source_ext = '.md'
        result = dict(getattr(self, '_extra_fields', {}))
        result.update({
            "name": self.name,
            "primary_directive": self.primary_directive,
            "model": self.model,
            "provider": self.provider,
            "api_url": self.api_url,
            "api_key": self.api_key,
            "jinxes": self.jinxes_spec,
        })
        result["source_path"] = source_path
        result["source_ext"] = source_ext
        return result
        
    def save(self, directory=None):
        """Save NPC to file, preserving original {{ Jinx('...') }} syntax."""
        if directory is None:
            directory = self.npc_directory
        os.makedirs(directory, exist_ok=True)
        npc_path = os.path.join(directory, f"{self.name}.npc")

        if os.path.exists(npc_path):
            with open(npc_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            rendered = render_jinja_content(original_content)
            try:
                current = yaml.safe_load(rendered) or {}
            except Exception:
                current = {}
            content = original_content
            updates = self.to_dict()
            for field in ['name', 'model', 'provider', 'api_url', 'api_key', 'primary_directive']:
                if field in updates and updates[field] != current.get(field):
                    content = _update_field_in_yaml(content, field, updates[field])
            jinxes = updates.get('jinxes')
            if isinstance(jinxes, list):
                jinx_values = []
                for j in jinxes:
                    if isinstance(j, str) and re.match(r'^[a-zA-Z0-9_]+$', j):
                        jinx_values.append(f"{{{{ Jinx('{j}') }}}}")
                    else:
                        jinx_values.append(str(j))
                content = _update_field_in_yaml(content, 'jinxes', jinx_values)
            with open(npc_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True

        return write_yaml_file(npc_path, self.to_dict())
    
    def __str__(self):
        """String representation of NPC"""
        str_rep = f"NPC: {self.name}\nDirective: {self.primary_directive}\nModel: {self.model}\nProvider: {self.provider}\nAPI URL: {self.api_url}\n"
        if self.jinxes_dict:
            str_rep += "Jinxs:\n"
            for jinx_name in self.jinxes_dict.keys():
                str_rep += f"  - {jinx_name}\n"
        else:
            str_rep += "No jinxes available.\n"
        return str_rep

    def execute_jinx_command(self, 
        jinx: Jinx,
        args: List[str],
        messages=None,
    ) -> Dict[str, Any]:
        """
        Execute a jinx command with the given arguments.
        """
        
        input_values = extract_jinx_inputs(args, jinx)

        jinx_output = jinx.execute(
            input_values,
            npc=self,
            messages=messages,
            jinja_env=self.jinja_env
        )

        return {"messages": messages, "output": jinx_output}

class Team:
    def __init__(self,
                    team_path=None,
                    npcs: Optional[List['NPC']] = None,
                    forenpc: Optional[Union[str, 'NPC']] = None,
                    jinxes: Optional[List[Union['Jinx', Dict[str, Any]]]] = None,
                    db_conn=None,
                    model = None,
                    provider = None,
                    api_url = None,
                    api_key = None,
                    team_jinxes: Optional[List['Jinx']] = None):
        """
        Initialize an NPC team from directory or list of NPCs

        Args:
            team_path: Path to team directory
            npcs: List of NPC objects
            db_conn: Deprecated/no-op. DB features are opt-in via NPC.initialize_db().
            team_jinxes: Pre-loaded jinxes (sub-teams use the same jinxes as the team)
        """
        self._team_jinxes = team_jinxes
        self.model = model
        self.provider = provider
        self.api_url = api_url
        self.api_key = api_key
        
        self.npcs: Dict[str, 'NPC'] = {}
        self.sub_teams: Dict[str, 'Team'] = {}
        self.jinxes_dict: Dict[str, 'Jinx'] = {}
        self._raw_jinxes_list: List['Jinx'] = []
        self.jinx_tool_catalog: Dict[str, Dict[str, Any]] = {}
        
        self.jinja_env_for_first_pass = SandboxedEnvironment(undefined=SilentUndefined)

        self.db_conn = db_conn
        self.team_path = os.path.expanduser(team_path) if team_path else None
        self.databases = []
        self.mcp_servers = []
        
        self.forenpc: Optional['NPC'] = None
        self.forenpc_name: Optional[str] = None
        self.skills_directory: Optional[str] = None

        if team_path:
            self.name = os.path.basename(os.path.abspath(team_path))
            self._load_from_directory_and_initialize_forenpc() 
        elif npcs:
            self.name = "custom_team"
            for npc_obj in npcs:
                self.npcs[npc_obj.name] = npc_obj
                npc_obj.team = self
            
            if jinxes:
                for jinx_item in jinxes:
                    if isinstance(jinx_item, Jinx):
                        self._raw_jinxes_list.append(jinx_item)
                    elif isinstance(jinx_item, dict):
                        self._raw_jinxes_list.append(Jinx(jinx_data=jinx_item))
        
            self._determine_forenpc_from_provided_npcs(npcs, forenpc)

        else:
            self.name = "custom_team"
            self._create_default_forenpc()

        self.context = ''
        self.shared_context = {
            "intermediate_results": {},
            "dataframes": {},
            "memories": {},          
            "execution_history": [],   
            "context":''       
            }
        
        if team_path:
            self._load_team_context_into_shared_context()
        elif self.forenpc:
            if not self.context:
                self.context = f"Team '{self.name}' with forenpc '{self.forenpc.name}'"
                self.shared_context['context'] = self.context

        self._perform_first_pass_jinx_rendering()
        self.jinx_tool_catalog = build_jinx_tool_catalog(self.jinxes_dict)
        print(f"[TEAM] Built Jinx tool catalog with {len(self.jinx_tool_catalog)} entries for team {self.name}", file=sys.stderr)

        for npc_obj in self.npcs.values():
            npc_obj.initialize_jinxes(team_raw_jinxes=self._raw_jinxes_list) 
        
    def _load_from_directory_and_initialize_forenpc(self):
        """
        Consolidated method to load NPCs, team context, and resolve the forenpc.
        Ensures self.npcs is populated and self.forenpc is an NPC object.

        Load order: context → jinxes → NPCs (so NPC files can use {{ jinx_name }}
        Jinja references that resolve to the jinx's relative path).
        """
        if not os.path.exists(self.team_path):
            raise ValueError(f"Team directory not found: {self.team_path}")

        self._load_team_context_file()

        if self._team_jinxes:
            self._raw_jinxes_list.extend(self._team_jinxes)

        jinxes_dir = os.path.join(self.team_path, "jinxes")
        if os.path.exists(jinxes_dir):
            for jinx_obj in load_jinxes_from_directory(jinxes_dir):
                self._raw_jinxes_list.append(jinx_obj)

        if hasattr(self, 'skills_directory') and self.skills_directory:
            skills_path = os.path.expanduser(self.skills_directory)
            if not os.path.isabs(skills_path):
                skills_path = os.path.join(self.team_path, skills_path)
            if os.path.exists(skills_path):
                for jinx_obj in load_jinxes_from_directory(skills_path):
                    self._raw_jinxes_list.append(jinx_obj)
                print(f"[TEAM] Loaded skills from SKILLS_DIRECTORY: {skills_path}")
            else:
                print(f"[TEAM] Warning: SKILLS_DIRECTORY not found: {skills_path}")

        self._jinx_path_map = {}
        for jinx_obj in self._raw_jinxes_list:
            if jinx_obj.jinx_name in self._jinx_path_map:
                continue
            source = getattr(jinx_obj, '_source_path', None)
            if source:
                base_dir = None
                parts = source.split(os.sep)
                for i, p in enumerate(parts):
                    if p == 'jinxes':
                        base_dir = os.sep.join(parts[:i+1])
                if not base_dir:
                    continue
                try:
                    rel = os.path.relpath(source, base_dir)
                    if rel.endswith('.jinx'):
                        rel = rel[:-5]
                    self._jinx_path_map[jinx_obj.jinx_name] = rel
                except ValueError:
                    pass

        def _Jinx(name, path=None, *, repo=None, ref=None):
            """Resolve a jinx by name, optionally from a foreign team.

            Usage:
              {{ Jinx('edit_file') }}                                 — own team (or already-loaded external)
              {{ Jinx('kg_search_keyword', '/abs/team/root') }}       — positional path to foreign team
              {{ Jinx('kg_search_keyword', path='/abs/team/root') }}  — same, keyword
              {{ Jinx('kg_search_keyword', repo='owner/repo') }}
                  — foreign team in a GitHub repo; cached to
                    ~/.cache/npcpy/jinxes/<owner>_<repo>[@<ref>]/ on first use
              {{ Jinx('x', repo='owner/repo', ref='v1.2.0') }}        — pin a branch/tag

            Foreign jinxes are loaded into this team's pool on first resolve,
            so subsequent renders hit the normal in-memory path.
            """
            if name in self._jinx_path_map and not (repo or path):
                return self._jinx_path_map[name]

            if repo or path:
                try:
                    team_root = _resolve_external_team_root(repo=repo, path=path, ref=ref)
                except Exception as _e:
                    print(f"Warning: Jinx('{name}', repo={repo!r}, path={path!r}) failed: {_e}", file=sys.stderr)
                    return name

                if team_root:
                    _hydrate_jinx_from_team(team_root, name)

                if name in self._jinx_path_map:
                    return self._jinx_path_map[name]

            print(f"Warning: Jinx('{name}') not found. Available: {list(self._jinx_path_map.keys())[:15]}...", file=sys.stderr)
            return name

        def _resolve_external_team_root(repo=None, path=None, ref=None):
            """Return a local filesystem path to a team root (directory containing
            a jinxes/ folder), fetching from GitHub if needed. Caches under
            ~/.cache/npcpy/jinxes/. Looks recursively for a directory named
            npc_team inside the cloned repo."""
            if path:
                expanded = os.path.expanduser(path)
                return expanded if os.path.isdir(expanded) else None

            if not repo:
                return None

            cache_root = os.path.expanduser('~/.cache/npcpy/jinxes')
            os.makedirs(cache_root, exist_ok=True)
            slug = repo.replace('/', '_')
            if ref:
                slug = f"{slug}@{ref}"
            clone_dir = os.path.join(cache_root, slug)

            if not os.path.isdir(clone_dir):
                import subprocess as _sp
                url = f"https://github.com/{repo}.git"
                cmd = ['git', 'clone', '--depth', '1']
                if ref:
                    cmd += ['--branch', ref]
                cmd += [url, clone_dir]
                _sp.run(cmd, check=True, capture_output=True)

            for root, dirs, _ in os.walk(clone_dir):
                dirs[:] = [d for d in dirs if d not in ('.git', 'node_modules', '.venv', 'venv', 'dist')]
                if os.path.basename(root) == 'npc_team':
                    return root
            if os.path.isdir(os.path.join(clone_dir, 'jinxes')):
                return clone_dir
            return None

        def _hydrate_jinx_from_team(team_root, jinx_name):
            """Load a single named jinx from a foreign team root into this team's
            pool, mutating _raw_jinxes_list and _jinx_path_map in place."""
            jinxes_dir = os.path.join(team_root, 'jinxes')
            if not os.path.isdir(jinxes_dir):
                return
            for root, _, files in os.walk(jinxes_dir):
                target = f"{jinx_name}.jinx"
                if target in files:
                    jinx_path = os.path.join(root, target)
                    try:
                        jinx_obj = Jinx(jinx_path=jinx_path)
                    except Exception as _e:
                        print(f"Warning: failed to load foreign jinx {jinx_path}: {_e}", file=sys.stderr)
                        return
                    for existing in self._raw_jinxes_list:
                        if existing.jinx_name == jinx_obj.jinx_name:
                            return
                    self._raw_jinxes_list.append(jinx_obj)
                    rel = os.path.relpath(jinx_path, jinxes_dir)
                    if rel.endswith('.jinx'):
                        rel = rel[:-5]
                    self._jinx_path_map[jinx_obj.jinx_name] = rel
                    return

        def _NPC(name):
            """Reference an NPC by name.
            Usage: {{ NPC('corca') }} → 'corca'
            Returns the name for use in directives and jinx configs.
            Validation happens at runtime, not compile time.
            """
            return name

        def _ref(model_name):
            """Reference a SQL model by name (dbt-style).
            Usage: FROM {{ ref('customer_feedback') }}
            At compile time in SQL models, resolves to the actual table name.
            In non-SQL contexts, returns the model name as-is.
            """
            return model_name

        def _jinxes_list(pattern):
            """Glob-expand a jinx path pattern to a list of paths.
            Usage: {% for j in jinxes_list('lib/browser/*') %}
              - {{ j }}
            {% endfor %}
            """
            import fnmatch as _fn
            matched = []
            for name, rel_path in self._jinx_path_map.items():
                spec_pattern = pattern
                if not spec_pattern.endswith('.jinx') and not spec_pattern.endswith('*'):
                    spec_pattern += '.jinx'
                rel_with_ext = rel_path + '.jinx'
                if _fn.fnmatch(rel_with_ext, spec_pattern):
                    matched.append(rel_path)
            return matched

        self._npc_jinja_context = {
            'Jinx': _Jinx,
            'NPC': _NPC,
            'ref': _ref,
            'jinxes_list': _jinxes_list,
            **self._jinx_path_map,
        }

        self._resolve_team_jinxes_spec()

        for filename in os.listdir(self.team_path):
            if filename.endswith(".npc"):
                npc_path = os.path.join(self.team_path, filename)
                npc = NPC(npc_path, team=self)
                if _is_cli_provider(npc.provider):
                    npc = CLIAgent(
                        cli_provider=npc.provider,
                        name=npc.name,
                        primary_directive=npc.primary_directive,
                        model=npc.model,
                        provider=npc.provider,
                        api_url=npc.api_url,
                        api_key=npc.api_key,
                        team=self,
                    )
                self.npcs[npc.name] = npc

        project_root = os.path.dirname(os.path.abspath(self.team_path))
        for md_name in ("agents.md", "AGENTS.md", "CLAUDE.md"):
            md_path = os.path.join(project_root, md_name)
            if os.path.exists(md_path):
                self._load_agents_from_md(md_path)

        agents_dir = os.path.join(project_root, "agents")
        if os.path.isdir(agents_dir):
            self._load_agents_from_dir(agents_dir)

        if self.forenpc_name and self.forenpc_name in self.npcs:
            self.forenpc = self.npcs[self.forenpc_name]
        elif self.npcs:
            self.forenpc = list(self.npcs.values())[0]
            self.forenpc_name = self.forenpc.name
        else:
            self._create_default_forenpc()

        self._load_sub_teams()

    def _load_team_context_file(self) -> Dict[str, Any]:
        """Loads team context from .ctx file and updates team attributes.

        Runs before the full Jinja context (Jinx()/NPC()/jinxes_list()) is built,
        so we pass an empty context with SilentUndefined — any `{{ Jinx('x') }}`
        inside jinxes: renders to empty on this pass. The real resolution happens
        in _resolve_team_jinxes_spec after jinx discovery.
        """
        ctx_data = {}
        for fname in os.listdir(self.team_path):
            if fname.endswith('.ctx'):
                ctx_data = load_yaml_file(os.path.join(self.team_path, fname), jinja_context={})
                if ctx_data is not None:
                    self.model = ctx_data.get('model', self.model)
                    self.provider = ctx_data.get('provider', self.provider)
                    self.api_url = ctx_data.get('api_url', self.api_url)
                    self.env = ctx_data.get('env', self.env if hasattr(self, 'env') else None)
                    self.mcp_servers = ctx_data.get('mcp_servers', [])
                    self.databases = ctx_data.get('databases', [])
                    self.providers = ctx_data.get('providers', [])
                    self.forenpc_name = ctx_data.get('forenpc', self.forenpc_name)
                    self.skills_directory = ctx_data.get('SKILLS_DIRECTORY', None)
                return ctx_data
        return {}

    def _load_team_context_into_shared_context(self):
        """Loads team context into shared_context after forenpc is determined."""
        ctx_data = {}
        jinja_ctx = getattr(self, '_npc_jinja_context', None)
        for fname in os.listdir(self.team_path):
            if fname.endswith('.ctx'):
                ctx_data = load_yaml_file(os.path.join(self.team_path, fname), jinja_context=jinja_ctx if jinja_ctx is not None else {})
                if ctx_data is not None:
                    self.context = ctx_data.get('context', '')
                    self.shared_context['context'] = self.context
                    if 'file_patterns' in ctx_data:
                        file_cache = self._parse_file_patterns(ctx_data['file_patterns'])
                        self.shared_context['files'] = file_cache
                    for key, item in ctx_data.items():
                        if key not in ['name', 'mcp_servers', 'databases', 'context', 'file_patterns', 'forenpc', 'model', 'provider', 'api_url', 'env', 'SKILLS_DIRECTORY']:
                            self.shared_context[key] = item
                return
        
    def _determine_forenpc_from_provided_npcs(self, npcs_list: List['NPC'], forenpc_arg: Optional[Union[str, 'NPC']]):
        """Determines self.forenpc when NPCs are provided directly to Team.__init__."""
        if forenpc_arg:
            if isinstance(forenpc_arg, NPC):
                self.forenpc = forenpc_arg
                self.forenpc_name = forenpc_arg.name
            elif isinstance(forenpc_arg, str) and forenpc_arg in self.npcs:
                self.forenpc = self.npcs[forenpc_arg]
                self.forenpc_name = forenpc_arg
            else:
                print(f"Warning: Specified forenpc '{forenpc_arg}' not found among provided NPCs. Falling back to first NPC.")
                if npcs_list:
                    self.forenpc = npcs_list[0]
                    self.forenpc_name = npcs_list[0].name
                else:
                    self._create_default_forenpc()
        elif npcs_list:
            self.forenpc = npcs_list[0]
            self.forenpc_name = npcs_list[0].name
        else:
            self._create_default_forenpc()

    def _create_default_forenpc(self):
        """Creates a default forenpc if none can be determined."""
        forenpc_model = self.model
        forenpc_provider = self.provider or 'ollama'
        if not forenpc_model:
            raise ValueError("No model specified for default forenpc.")
        forenpc_api_key = self.api_key
        forenpc_api_url = self.api_url
        
        default_forenpc = NPC(name='forenpc', 
                                primary_directive="""You are the forenpc of the team, coordinating activities 
                                                    between NPCs on the team, verifying that results from 
                                                    NPCs are high quality and can help to adequately answer 
                                                    user requests.""", 
                                model=forenpc_model,
                                provider=forenpc_provider,
                                api_key=forenpc_api_key,
                                api_url=forenpc_api_url,                            
                                team=self
                                                    )
        self.forenpc = default_forenpc
        self.forenpc_name = default_forenpc.name
        self.npcs[default_forenpc.name] = default_forenpc

    def _perform_first_pass_jinx_rendering(self):
        """
        Performs the first-pass Jinja rendering on all loaded raw Jinxs.
        This expands nested Jinx calls but preserves runtime variables.

        Also injects team-level Jinja helpers into the rendering context:
        - NPC('name') — validates an NPC exists and returns its name
        - jinx name variables — e.g., {{ edit_file }} resolves to 'lib/core/files/edit_file'
        - jinxes_list('pattern') — glob-expands a jinx path pattern to a list
        """
        jinx_macro_globals = {}
        for raw_jinx in self._raw_jinxes_list:
            def create_jinx_callable(jinx_obj_in_closure):
                def callable_jinx(**kwargs):
                    temp_jinja_env = SandboxedEnvironment(undefined=SilentUndefined)

                    rendered_target_steps = []
                    for target_step in jinx_obj_in_closure._raw_steps:
                        temp_rendered_step = {}
                        for k, v in target_step.items():
                            if isinstance(v, str):
                                try:
                                    temp_rendered_step[k] = temp_jinja_env.from_string(v).render(**kwargs)
                                except Exception as e:
                                    print(f"Warning: Error in Jinx macro '{jinx_obj_in_closure.jinx_name}' rendering step field '{k}' (Team first pass): {e}")
                                    temp_rendered_step[k] = v
                            else:
                                temp_rendered_step[k] = v
                        rendered_target_steps.append(temp_rendered_step)

                    return yaml.dump(rendered_target_steps, default_flow_style=False)
                return callable_jinx

            jinx_macro_globals[raw_jinx.jinx_name] = create_jinx_callable(raw_jinx)

        self.jinja_env_for_first_pass.globals['jinxes'] = jinx_macro_globals
        self.jinja_env_for_first_pass.globals['ctx'] = self.shared_context
        if hasattr(self, '_npc_jinja_context'):
            self.jinja_env_for_first_pass.globals.update(self._npc_jinja_context)
        self.jinja_env_for_first_pass.globals.update(jinx_macro_globals)

        for raw_jinx in self._raw_jinxes_list:
            try:
                if hasattr(raw_jinx, 'npc') and isinstance(raw_jinx.npc, str):
                    if '{{' in raw_jinx.npc and '}}' in raw_jinx.npc:
                        try:
                            template = self.jinja_env_for_first_pass.from_string(raw_jinx.npc)
                            raw_jinx.npc = template.render(**self.jinja_env_for_first_pass.globals)
                        except Exception as e:
                            print(f"Warning: Error rendering npc field for jinx '{raw_jinx.jinx_name}': {e}")

                raw_jinx.render_first_pass(self.jinja_env_for_first_pass, jinx_macro_globals)
                self.jinxes_dict[raw_jinx.jinx_name] = raw_jinx
            except Exception as e:
                print(f"Error performing first-pass rendering for Jinx '{raw_jinx.jinx_name}': {e}")

    def update_context(self, messages: list):
        """Update team context based on recent conversation patterns"""
        if len(messages) < 10:
            return
            
        summary = breathe(
            messages=messages[-10:], 
            npc=self.forenpc
        )
        characterization = summary.get('output')
        
        if characterization:
            team_ctx_path = os.path.join(self.team_path, "team.ctx")
            
            if os.path.exists(team_ctx_path):
                with open(team_ctx_path, 'r', encoding="utf-8") as f:
                    ctx_data = yaml.safe_load(f) or {}
            else:
                ctx_data = {}
                
            current_context = ctx_data.get('context', '')
            
            prompt = f"""Based on this characterization: {characterization},
            suggest changes to the team's context.
            Current Context: "{current_context}".
            Respond with JSON: {{"suggestion": "Your sentence."}}"""
            
            response = get_llm_response(
                prompt=prompt,
                npc=self.forenpc,
                format="json"
            )
            suggestion = response.get("response", {}).get("suggestion")
            
            if suggestion:
                new_context = (current_context + " " + suggestion).strip()
                user_approval = input(f"Update context to: {new_context}? [y/N]: ").strip().lower()
                if user_approval == 'y':
                    ctx_data['context'] = new_context
                    self.context = new_context
                    with open(team_ctx_path, 'w', encoding="utf-8") as f:
                        yaml.dump(ctx_data, f)
            
    def _load_sub_teams(self):
        """Load sub-teams from subdirectories.

        A subdirectory becomes a sub-team if it contains either:
          - one or more .npc files (.npc file team), or
          - agents.md / AGENTS.md / CLAUDE.md / an agents/ directory
            (markdown-declared team — inherits this team's jinxes).

        Also scans an agents/ directory inside the team root for sub-teams
        (supports mixed markdown + npc_team style nesting).
        """
        candidates = []

        for item in os.listdir(self.team_path):
            item_path = os.path.join(self.team_path, item)
            if os.path.isdir(item_path) and not item.startswith('.') and item != "jinxes":
                candidates.append((item, item_path))

        agents_dir = os.path.join(self.team_path, "agents")
        if os.path.isdir(agents_dir):
            for item in os.listdir(agents_dir):
                item_path = os.path.join(agents_dir, item)
                if os.path.isdir(item_path) and not item.startswith('.'):
                    candidates.append((item, item_path))

        for item, item_path in candidates:
            entries = os.listdir(item_path)
            has_npc = any(f.endswith(".npc") for f in entries
                          if os.path.isfile(os.path.join(item_path, f)))
            has_md_team = (
                any(f in ("agents.md", "AGENTS.md", "CLAUDE.md") for f in entries) or
                os.path.isdir(os.path.join(item_path, "agents"))
            )
            if has_npc or has_md_team:
                sub_team = Team(team_path=item_path, team_jinxes=self._raw_jinxes_list)
                self.sub_teams[item] = sub_team
        
    def _resolve_team_jinxes_spec(self):
        """Jinja-render the `jinxes:` field from the team .ctx file.

        The first pass of .ctx parsing runs before the Jinx()/NPC()/jinxes_list()
        macros exist, so jinxes: can't be resolved there. This second pass reads
        the .ctx again with the full npc_jinja_context and stores the resolved
        spec on self.team_jinxes_spec — a list of rendered strings (jinx names or
        relative paths) that every NPC on the team will pick up via
        initialize_jinxes.
        """
        self.team_jinxes_spec = []
        if not hasattr(self, 'team_path') or not self.team_path:
            return
        try:
            for fname in os.listdir(self.team_path):
                if not fname.endswith('.ctx'):
                    continue
                ctx_path = os.path.join(self.team_path, fname)
                rendered = load_yaml_file(ctx_path, jinja_context=self._npc_jinja_context)
                if isinstance(rendered, dict):
                    spec = rendered.get('jinxes', [])
                    if isinstance(spec, list):
                        self.team_jinxes_spec = [s for s in spec if s]
                return
        except Exception as e:
            print(f"Warning: failed to resolve team-level jinxes from .ctx: {e}")

    def _load_agents_from_md(self, path: str):
        """Load agents from an agents.md / AGENTS.md / CLAUDE.md file.

        Format:
            directive body text...

        Each H2 heading becomes an NPC with that name and the body as its directive.
        Agents loaded this way pick up the team's .ctx jinxes; if none are defined
        they fall back to DEFAULT_MD_AGENT_JINXES.
        """
        with open(path, 'r') as f:
            content = f.read()

        current_name = None
        current_body = []

        for line in content.split('\n'):
            if line.startswith('## '):
                if current_name:
                    self._register_or_prompt_agent(current_name, '\n'.join(current_body).strip(), path)
                current_name = line[3:].strip()
                current_body = []
            elif current_name is not None:
                current_body.append(line)

        if current_name:
            self._register_or_prompt_agent(current_name, '\n'.join(current_body).strip(), path)

    def _load_agents_from_dir(self, agents_dir: str):
        """Load agents from an agents/ directory.

        Each .md file defines an agent:
        - Filename (without extension) = agent name
        - File content = directive (optionally with YAML frontmatter for
          model/provider/name and a jinxes: or tools: list for per-agent tool spec)
        """
        for fname in os.listdir(agents_dir):
            if not fname.endswith('.md'):
                continue
            name = fname[:-3]
            if name in self.npcs:
                continue
            fpath = os.path.join(agents_dir, fname)
            with open(fpath, 'r') as f:
                content = f.read()

            model = self.model
            provider = self.provider
            jinxes_spec = None
            directive = content
            if content.startswith('---'):
                parts = content.split('---', 2)
                if len(parts) >= 3:
                    try:
                        fm = yaml.safe_load(parts[1])
                        if isinstance(fm, dict):
                            model = fm.get('model', model)
                            provider = fm.get('provider', provider)
                            name = fm.get('name', name)
                            fm_jinxes = fm.get('jinxes', fm.get('tools'))
                            if isinstance(fm_jinxes, list):
                                jinxes_spec = [str(s) for s in fm_jinxes if s]
                        directive = parts[2].strip()
                    except Exception:
                        directive = content

            self._register_md_agent(name, directive, model=model, provider=provider, jinxes_spec=jinxes_spec, source_path=fpath)

    def _register_or_prompt_agent(self, name: str, directive: str, source_path: str):
        """Register a markdown-declared agent, skipping names that collide with an
        existing .npc-declared NPC so curated definitions always win.

        ``source_path`` is accepted for logging/traceability; agents.md / AGENTS.md /
        CLAUDE.md all feed through here.
        """
        if not name or name in self.npcs:
            return
        self._register_md_agent(name, directive, source_path=source_path)

    def _register_md_agent(self, name: str, directive: str, model=None, provider=None, jinxes_spec=None, source_path: str = None):
        """Create an NPC from a markdown agent definition and add to team.

        Markdown agents don't have a strict .npc jinxes: spec. Resolution order:
          1. Explicit frontmatter `jinxes:`/`tools:` list, if provided.
          2. Team-level `jinxes:` from .ctx, if any.
          3. DEFAULT_MD_AGENT_JINXES fallback — a generic agent toolkit.
        Team-level jinxes are then additively merged on top in initialize_jinxes.
        """
        if jinxes_spec is None:
            if getattr(self, 'team_jinxes_spec', None):
                jinxes_spec = list(self.team_jinxes_spec)
            else:
                jinxes_spec = [
                    name for name in DEFAULT_MD_AGENT_JINXES
                    if name in self.jinxes_dict
                ]
        npc = NPC(
            name=name,
            primary_directive=directive,
            model=model or self.model,
            provider=provider or self.provider,
            jinxes=jinxes_spec,
        )
        npc.team = self
        if source_path:
            npc.source_path = source_path
        self.npcs[name] = npc
        try:
            npc.initialize_jinxes(team_raw_jinxes=self._raw_jinxes_list)
        except Exception as e:
            print(f"Warning: failed to initialize jinxes for markdown agent '{name}': {e}")

    def get_forenpc(self) -> Optional['NPC']:
        """
        Returns the forenpc (coordinator) for this team.
        This method is now primarily for external access, as self.forenpc is set in __init__.
        """
        return self.forenpc

    def get_npc(self, npc_ref: Union[str, 'NPC']) -> Optional['NPC']:
        """Get NPC by name or reference with hierarchical lookup capability"""
        if isinstance(npc_ref, NPC):
            return npc_ref
        elif isinstance(npc_ref, str):
            if npc_ref in self.npcs:
                return self.npcs[npc_ref]
            
            for sub_team_name, sub_team in self.sub_teams.items():
                if npc_ref in sub_team.npcs:
                    return sub_team.npcs[npc_ref]
                
                result = sub_team.get_npc(npc_ref)
                if result:
                    return result
            
            return None
        else:
            return None

    def orchestrate(self, request, max_iterations=3):
        """Orchestrate a request through the team"""
        import re
        from termcolor import colored

        forenpc = self.get_forenpc()
        if not forenpc:
            return {"error": "No forenpc available to coordinate the team"}

        print(colored(f"[orchestrate] Starting with forenpc={forenpc.name}, team={self.name}", "cyan"))
        print(colored(f"[orchestrate] Request: {request[:100]}...", "cyan"))

        jinxes_for_orchestration = {k: v for k, v in forenpc.jinxes_dict.items() if k != 'orchestrate'}

        try:
            result = forenpc.check_llm_command(
                request,
                context=getattr(self, 'context', {}),
                team=self,
                jinxes=jinxes_for_orchestration,
            )
            print(colored(f"[orchestrate] Initial result type={type(result)}", "cyan"))
            if isinstance(result, dict):
                print(colored(f"[orchestrate] Result keys={list(result.keys())}", "cyan"))
                if 'error' in result:
                    print(colored(f"[orchestrate] Error in result: {result['error']}", "red"))
                    return result
        except Exception as e:
            print(colored(f"[orchestrate] Exception in check_llm_command: {e}", "red"))
            return {"error": str(e), "output": f"Orchestration failed: {e}"}

        output = ""
        if isinstance(result, dict):
            output = result.get('output') or result.get('response') or ""

        print(colored(f"[orchestrate] Output preview: {output[:200] if output else 'EMPTY'}...", "cyan"))

        if output and self.npcs:
            at_pattern = r'@(\w+)'
            mentions = re.findall(at_pattern, output)

            if not mentions:
                for npc_name in self.npcs.keys():
                    if npc_name.lower() != forenpc.name.lower():
                        if npc_name.lower() in output.lower():
                            mentions.append(npc_name)
                            break

            print(colored(f"[orchestrate] Found mentions: {mentions}", "cyan"))

            for mentioned in mentions:
                mentioned_lower = mentioned.lower()
                if mentioned_lower in self.npcs and mentioned_lower != forenpc.name:
                    target_npc = self.npcs[mentioned_lower]
                    print(colored(f"[orchestrate] Delegating to @{mentioned_lower}", "yellow"))

                    try:
                        target_jinxes = {k: v for k, v in target_npc.jinxes_dict.items() if k != 'orchestrate'}
                        delegate_result = target_npc.check_llm_command(
                            request,
                            context=getattr(self, 'context', {}),
                            team=self,
                            jinxes=target_jinxes,
                        )

                        if isinstance(delegate_result, dict):
                            delegate_output = delegate_result.get('output') or delegate_result.get('response') or ""
                            if delegate_output:
                                output = f"[{mentioned_lower}]: {delegate_output}"
                                result = delegate_result
                                print(colored(f"[orchestrate] Got response from {mentioned_lower}", "green"))
                    except Exception as e:
                        print(colored(f"[orchestrate] Delegation to {mentioned_lower} failed: {e}", "red"))

                    break

        if isinstance(result, dict):
            final_output = output if output else str(result)
            return {
                "output": final_output,
                "result": result,
            }
        else:
            return {
                "output": str(result),
                "result": result,
            }
                
    def to_dict(self):
        """Convert team to dictionary representation"""
        return {
            "name": self.name,
            "npcs": {name: npc.to_dict() for name, npc in self.npcs.items()},
            "sub_teams": {name: team.to_dict() for name, team in self.sub_teams.items()},
            "jinxes": {name: jinx.to_dict() for name, jinx in self.jinxes_dict.items()},
            "context": getattr(self, 'context', {})
        }
    
    def save(self, directory=None):
        """Save team to directory"""
        if directory is None:
            directory = self.team_path
            
        if not directory:
            raise ValueError("No directory specified for saving team")
            
        os.makedirs(directory, exist_ok=True)
        
        if hasattr(self, 'context') and self.context:
            ctx_path = os.path.join(directory, "team.ctx")
            write_yaml_file(ctx_path, self.context)
            
        for npc in self.npcs.values():
            npc.save(directory)
            
        jinxes_dir = os.path.join(directory, "jinxes")
        os.makedirs(jinxes_dir, exist_ok=True)
        
        for jinx in self.jinxes_dict.values():
            jinx.save(jinxes_dir)
            
        for team_name, team in self.sub_teams.items():
            team_dir = os.path.join(directory, team_name)
            team.save(team_dir)
            
        return True
    def _parse_file_patterns(self, patterns_config):
        """Parse file patterns configuration and load matching files into KV cache"""
        if not patterns_config:
            return {}
        
        file_cache = {}
        
        for pattern_entry in patterns_config:
            if isinstance(pattern_entry, str):
                pattern_entry = {"pattern": pattern_entry}
            
            pattern = pattern_entry.get("pattern", "")
            recursive = pattern_entry.get("recursive", False)
            base_path = pattern_entry.get("base_path", ".")
            
            if not pattern:
                continue
                
            base_path = os.path.expanduser(base_path)
            if not os.path.isabs(base_path):
                base_path = os.path.join(self.team_path or os.getcwd(), base_path)
            
            matching_files = self._find_matching_files(pattern, base_path, recursive)
            
            for file_path in matching_files:
                file_content = self._load_file_content(file_path)
                if file_content:
                    relative_path = os.path.relpath(file_path, base_path)
                    file_cache[relative_path] = file_content
        
        return file_cache

    def _find_matching_files(self, pattern, base_path, recursive=False):
        """Find files matching the given pattern"""
        matching_files = []
        
        if not os.path.exists(base_path):
            return matching_files
        
        if recursive:
            for root, dirs, files in os.walk(base_path):
                for filename in files:
                    if fnmatch.fnmatch(filename, pattern):
                        matching_files.append(os.path.join(root, filename))
        else:
            try:
                for item in os.listdir(base_path):
                    item_path = os.path.join(base_path, item)
                    if os.path.isfile(item_path) and fnmatch.fnmatch(item, pattern):
                        matching_files.append(item_path)
            except PermissionError:
                print(f"Permission denied accessing {base_path}")
        
        return matching_files

    def _load_file_content(self, file_path):
        """Load content from a file with error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None

    def _format_parsed_files_context(self, parsed_files):
        """Format parsed files into context string"""
        if not parsed_files:
            return ""
        
        context_parts = ["Additional context from files:"]
        
        for file_path, content in parsed_files.items():
            context_parts.append(f"\n--- {file_path} ---")
            context_parts.append(content)
            context_parts.append("")
        
        return "\n".join(context_parts)


NPC_SHEBANG = "#!/usr/bin/env npc"
JINX_SHEBANG = "#!/usr/bin/env npc-jinx"


def ensure_shebang(file_path: str, shebang: str = None) -> str:
    """Add a shebang to a .npc/.jinx file so it can run as an executable.

    Returns the file path.
    """
    if shebang is None:
        if file_path.endswith('.npc'):
            shebang = NPC_SHEBANG
        elif file_path.endswith('.jinx'):
            shebang = JINX_SHEBANG
        else:
            return file_path

    with open(file_path, 'r') as f:
        content = f.read()

    if content.startswith('#!'):
        return file_path

    with open(file_path, 'w') as f:
        f.write(shebang + '\n' + content)

    current = os.stat(file_path).st_mode
    os.chmod(file_path, current | 0o111)
    return file_path


def strip_shebang(content: str) -> str:
    """Strip shebang line from file content before YAML parsing."""
    if content.startswith('#!'):
        newline = content.find('\n')
        if newline >= 0:
            return content[newline + 1:]
    return content



def _tool_sh(bash_command: str) -> str:
    """Execute a bash/shell command and return stdout+stderr."""
    try:
        result = subprocess.run(
            bash_command, shell=True, capture_output=True, text=True, timeout=120
        )
        out = result.stdout
        if result.returncode != 0 and result.stderr:
            out += f"\nSTDERR:\n{result.stderr}"
        return out or "(no output)"
    except subprocess.TimeoutExpired:
        return "Command timed out after 120s"
    except Exception as e:
        return f"Error: {e}"


def _tool_python(code: str) -> str:
    """Execute Python code and return stdout+stderr."""
    try:
        result = subprocess.run(
            ["python3", "-c", code], capture_output=True, text=True, timeout=120
        )
        out = result.stdout
        if result.returncode != 0 and result.stderr:
            out += f"\nSTDERR:\n{result.stderr}"
        return out or "(no output)"
    except subprocess.TimeoutExpired:
        return "Code execution timed out after 120s"
    except Exception as e:
        return f"Error: {e}"


def _tool_edit_file(path: str, action: str = "create", new_text: str = "", old_text: str = "") -> str:
    """Edit a file. Actions: create/write, append, replace.

    Args:
        path: File path to edit.
        action: One of 'create', 'write', 'append', 'replace'.
        new_text: Text to write/append, or replacement text.
        old_text: Text to find (for replace action).
    """
    path = os.path.expanduser(path)
    try:
        if action in ("create", "write"):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                f.write(new_text)
            return f"Created/wrote {path} ({len(new_text)} bytes)"
        elif action == "append":
            with open(path, "a") as f:
                f.write(new_text)
            return f"Appended to {path}"
        elif action == "replace":
            with open(path, "r") as f:
                content = f.read()
            updated = content.replace(old_text, new_text)
            with open(path, "w") as f:
                f.write(updated)
            return f"Replaced text in {path}"
        else:
            return f"Unknown action: {action}"
    except Exception as e:
        return f"Error: {e}"


def _tool_load_file(path: str) -> str:
    """Read and return the contents of a file.

    Args:
        path: File path to read.
    """
    path = os.path.expanduser(path)
    try:
        with open(path, "r") as f:
            content = f.read()
        lines = content.count("\n") + 1
        if len(content) > 10000:
            return f"File: {path} ({lines} lines, {len(content)} bytes)\n---\n{content[:10000]}...\n[truncated]"
        return f"File: {path} ({lines} lines, {len(content)} bytes)\n---\n{content}"
    except Exception as e:
        return f"Error reading {path}: {e}"


def _tool_web_search(query: str) -> str:
    """Search the web using DuckDuckGo and return results.

    Args:
        query: Search query string.
    """
    try:
        from npcpy.data.web import search_web
        results = search_web(query)
        if isinstance(results, list):
            return "\n".join(str(r) for r in results[:5])
        return str(results)
    except ImportError:
        url = f"https://lite.duckduckgo.com/lite/?q={urllib.parse.quote_plus(query)}"
        try:
            result = subprocess.run(["curl", "-sL", url], capture_output=True, text=True, timeout=15)
            output = "\n".join(result.stdout.splitlines()[:100])
            return output or "No results"
        except Exception as e:
            return f"Search failed: {e}"


def _tool_file_search(query: str, path: str = ".") -> str:
    """Search for files containing a query string.

    Args:
        query: Text pattern to search for.
        path: Directory to search in.
    """
    path = os.path.expanduser(path)
    try:
        result = subprocess.run(
            ["grep", "-rn",
             "--include=*.py", "--include=*.rs", "--include=*.js", "--include=*.ts",
             "--include=*.md", "--include=*.txt", "--include=*.yaml", "--include=*.yml",
             "--include=*.toml", "--include=*.json", "--include=*.sh",
             "-l", query, path],
            capture_output=True, text=True, timeout=30)
        output = "\n".join(result.stdout.splitlines()[:20])
        return output or f"No files found matching '{query}' in {path}"
    except Exception as e:
        return f"Search error: {e}"


def _tool_stop(reason: str = "") -> str:
    """Signal that the task is complete.

    Args:
        reason: Optional reason for stopping.
    """
    return f"STOP: {reason}" if reason else "STOP"


def _tool_chat(message: str) -> str:
    """Respond directly to the user without taking any action.

    Args:
        message: The message to send.
    """
    return message


_DEFAULT_AGENT_TOOLS = [
    _tool_sh, _tool_python, _tool_edit_file, _tool_load_file,
    _tool_web_search, _tool_file_search, _tool_stop, _tool_chat,
]


class Agent(NPC):
    """NPC with a default tool set (sh, python, edit_file, load_file, web_search, file_search, stop, chat).

    Can also load agent definitions from:
    - agents.md files (markdown with agent specs)
    - skills directories (each skill is a jinx)
    - MCP servers (external tool providers)
    """

    def __init__(
        self,
        name: str = "agent",
        primary_directive: str = "You are a helpful AI agent with access to tools.",
        model: str = None,
        provider: str = None,
        tools: list = None,
        extra_tools: list = None,
        agents_md: str = None,
        skills_dir: str = None,
        mcp_servers: list = None,
        safe_tools: bool = False,
        **kwargs,
    ):
        _EXEC_TOOLS = {_tool_sh, _tool_python}
        all_tools = [t for t in _DEFAULT_AGENT_TOOLS if not safe_tools or t not in _EXEC_TOOLS]
        if extra_tools:
            all_tools.extend(extra_tools)
        if tools is not None:
            all_tools = tools

        super().__init__(
            name=name,
            primary_directive=primary_directive,
            model=model,
            provider=provider,
            tools=all_tools,
            **kwargs,
        )

        if mcp_servers:
            self.mcp_servers = mcp_servers

        if agents_md:
            self._load_agents_md(agents_md)

        if skills_dir:
            self._load_skills_dir(skills_dir)

    def _load_agents_md(self, path: str):
        """Load agent definitions from an agents.md file.

        Format: markdown with H2 headings as agent names, body as directives.
        """
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            return

        with open(path, 'r') as f:
            content = f.read()

        current_name = None
        current_body = []
        agents = {}

        for line in content.split('\n'):
            if line.startswith('## '):
                if current_name:
                    agents[current_name] = '\n'.join(current_body).strip()
                current_name = line[3:].strip()
                current_body = []
            elif current_name is not None:
                current_body.append(line)

        if current_name:
            agents[current_name] = '\n'.join(current_body).strip()

        self._sub_agents = agents

    def _load_skills_dir(self, path: str):
        """Load skills (jinxes) from a directory and add them as tools."""
        path = os.path.expanduser(path)
        if not os.path.isdir(path):
            return

        for fname in os.listdir(path):
            if fname.endswith('.jinx'):
                fpath = os.path.join(path, fname)
                try:
                    jinx = Jinx(jinx_path=fpath)
                    self.jinxes_dict[jinx.jinx_name] = jinx
                except Exception as e:
                    print(f"Warning: Failed to load skill {fname}: {e}")

    def run(
        self,
        input_text: str,
        max_iterations: int = 10,
        verbose: bool = False,
        require_permission: bool = True,
        allow_tools: list = None,
        **kwargs,
    ):
        """Run the agent in a tool-calling loop until it stops calling tools.

        Each iteration: call the model with tools; if it emits tool_calls, execute
        them and feed the results back; if it emits plain content, return it.

        Args:
            input_text: initial user prompt.
            max_iterations: hard cap on tool-calling turns.
            verbose: print each turn's tool calls and results.
            require_permission: if True (default), prompt the user before each
                tool call. Answer [y]es / [n]o / [a]ll (allow the rest of this
                run). Non-TTY environments auto-allow so scripts don't hang.
            allow_tools: optional list of tool names that bypass the prompt.
        """
        import sys
        import json as _json
        import uuid as _uuid
        from npcpy.llm_funcs import get_llm_response

        messages = list(kwargs.get("messages", []) or [])
        stream = kwargs.get("stream", False)
        prompt = input_text
        last_content = ""
        allow_tools = set(allow_tools or [])
        approve_all = [False]
        is_tty = sys.stdin.isatty()

        def _log(msg):
            if verbose:
                print(msg, flush=True)

        def _ask(tool_name, args):
            if not require_permission or tool_name in allow_tools or approve_all[0]:
                return True
            if not is_tty:
                _log(f"[agent:{self.name}] non-TTY: auto-allow {tool_name}")
                return True
            args_str = str(args)
            if len(args_str) > 300:
                args_str = args_str[:300] + "…"
            print(f"\n[agent:{self.name}] allow tool: {tool_name}({args_str})?", flush=True)
            resp = input("  [y]es / [n]o / [a]ll: ").strip().lower()
            if resp == "a":
                approve_all[0] = True
                return True
            return resp.startswith("y")

        def _extract_tc(tc):
            if isinstance(tc, dict):
                tc_id = tc.get("id") or str(_uuid.uuid4())
                fn = tc.get("function", {}) or {}
                name = fn.get("name", "")
                args_raw = fn.get("arguments", "{}")
            else:
                tc_id = getattr(tc, "id", None) or str(_uuid.uuid4())
                fn_obj = getattr(tc, "function", None)
                name = getattr(fn_obj, "name", "") if fn_obj else ""
                args_raw = getattr(fn_obj, "arguments", "{}") if fn_obj else "{}"
            if isinstance(args_raw, str):
                try:
                    args = _json.loads(args_raw)
                except (_json.JSONDecodeError, TypeError):
                    args = {"raw_arguments": args_raw}
            else:
                args = args_raw or {}
            return tc_id, name, args

        _log(f"[agent:{self.name}] run start | model={self.model} provider={self.provider} | prompt={input_text!r}")

        for iteration in range(max_iterations):
            _log(f"[agent:{self.name}] iter {iteration+1}/{max_iterations} → calling model")
            result = get_llm_response(
                prompt,
                npc=self,
                model=self.model,
                provider=self.provider,
                tools=self.tools,
                tool_map=self.tool_map,
                messages=messages,
                stream=stream,
            )
            if not isinstance(result, dict):
                return str(result)

            tool_calls = result.get("tool_calls")
            content = result.get("output", result.get("response", "")) or ""
            if content:
                last_content = content

            if not tool_calls:
                _log(f"[agent:{self.name}] no tool_calls → returning content ({len(last_content)} chars)")
                return last_content

            messages = result.get("messages", messages)
            if (messages and messages[-1].get("role") == "assistant"
                    and not messages[-1].get("content")
                    and "tool_calls" not in messages[-1]):
                messages = messages[:-1]

            extracted = [_extract_tc(tc) for tc in tool_calls]
            _log(f"[agent:{self.name}] tool_calls: {[name for _, name, _ in extracted]}")

            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc_id,
                        "type": "function",
                        "function": {"name": name, "arguments": args},
                    }
                    for tc_id, name, args in extracted
                ],
            })

            for tc_id, name, args in extracted:
                if not _ask(name, args):
                    result_str = "[permission denied by user]"
                elif name in self.tool_map:
                    try:
                        tool_result = self.tool_map[name](**args)
                        result_str = _json.dumps(tool_result, default=str) if not isinstance(tool_result, str) else tool_result
                    except Exception as e:
                        result_str = f"Error executing {name}: {e}"
                else:
                    result_str = f"Unknown tool: {name}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc_id,
                    "content": result_str,
                })

                disp_args = str(args)
                disp_res = result_str
                if len(disp_args) > 200:
                    disp_args = disp_args[:200] + "…"
                if len(disp_res) > 400:
                    disp_res = disp_res[:400] + "…"
                _log(f"[agent:{self.name}]   → {name}({disp_args}) = {disp_res}")

            prompt = None

        _log(f"[agent:{self.name}] hit max_iterations={max_iterations}")
        return last_content or "Max iterations reached without a final answer."


class ToolAgent(Agent):
    """Agent with user-provided tool functions and/or MCP servers."""

    def __init__(
        self,
        name: str = "tool_agent",
        primary_directive: str = "You are an AI agent with specialized tools.",
        tools: list = None,
        mcp_servers: list = None,
        include_defaults: bool = True,
        model: str = None,
        provider: str = None,
        **kwargs,
    ):
        all_tools = []
        if include_defaults:
            all_tools.extend(_DEFAULT_AGENT_TOOLS)
        if tools:
            all_tools.extend(tools)

        super().__init__(
            name=name,
            primary_directive=primary_directive,
            model=model,
            provider=provider,
            tools=all_tools,
            mcp_servers=mcp_servers,
            **kwargs,
        )


class CodingAgent(Agent):
    """Agent that auto-detects + executes code blocks in LLM responses."""

    def __init__(
        self,
        name: str = "coding_agent",
        primary_directive: str = None,
        language: str = "python",
        auto_execute: bool = True,
        model: str = None,
        provider: str = None,
        **kwargs,
    ):
        if primary_directive is None:
            primary_directive = (
                f"You are a coding agent specialized in {language}. "
                f"Write {language} code in fenced code blocks (```{language}). "
                "The code will be automatically executed and the output fed back to you."
            )

        super().__init__(
            name=name,
            primary_directive=primary_directive,
            model=model,
            provider=provider,
            **kwargs,
        )

        self.language = language
        self.auto_execute = auto_execute

    def extract_code_blocks(self, text: str) -> list:
        """Extract code blocks matching this agent's language."""
        pattern = re.compile(r"```(\w+)?\s*\n(.*?)```", re.DOTALL)
        blocks = []
        for match in pattern.finditer(text):
            lang = (match.group(1) or "").lower()
            code = match.group(2).strip()
            if lang == self.language.lower() or (not lang and self.language == "python"):
                blocks.append(code)
        return blocks

    def execute_code(self, code: str) -> str:
        """Execute a code block and return the output."""
        lang = self.language.lower()
        if lang == "python":
            return _tool_python(code)
        elif lang in ("bash", "sh", "shell"):
            return _tool_sh(code)
        elif lang in ("javascript", "js", "node"):
            try:
                result = subprocess.run(
                    ["node", "-e", code], capture_output=True, text=True, timeout=120
                )
                out = result.stdout
                if result.returncode != 0 and result.stderr:
                    out += f"\nSTDERR:\n{result.stderr}"
                return out or "(no output)"
            except Exception as e:
                return f"Error: {e}"
        else:
            return f"Execution not supported for language: {lang}"

    def run(self, input_text: str, **kwargs):
        """Run with auto-execution of code blocks."""
        from npcpy.llm_funcs import get_llm_response

        messages = kwargs.get("messages", [])
        max_rounds = kwargs.get("max_rounds", 5)
        current_input = input_text
        response_text = ""

        for _ in range(max_rounds):
            result = get_llm_response(
                current_input,
                npc=self,
                model=self.model,
                provider=self.provider,
                tools=self.tools,
                tool_map=self.tool_map,
                messages=messages,
                stream=kwargs.get("stream", False),
            )

            if isinstance(result, dict):
                response_text = result.get("output", result.get("response", str(result)))
                messages = result.get("messages", messages)
            else:
                response_text = str(result)

            if not self.auto_execute:
                return response_text

            code_blocks = self.extract_code_blocks(response_text)
            if not code_blocks:
                return response_text

            execution_results = []
            for i, code in enumerate(code_blocks, 1):
                output = self.execute_code(code)
                execution_results.append(f"[Block {i} output]:\n{output}")

            current_input = "Code execution results:\n" + "\n\n".join(execution_results)

        return response_text


def _is_cli_provider(provider: str) -> bool:
    """Check if provider is a CLI-based agent."""
    return provider in ("claude_code", "claude", "opencode", "kimi_code", "kimi", "kilo_code", "kilo", "gemini-cli", "codex", "nanocoder", "aider", "amp")


class CLIAgent(Agent):
    """Agent that runs CLI tools (claude, opencode, kimi, kilo) as subprocesses.

    Session context is managed via temp files tied to conversation_id.
    """

    CLI_COMMANDS = {
        "claude_code": ["claude"],
        "claude": ["claude"],
        "opencode": ["opencode", "run"],
        "kimi_code": ["kimi"],
        "kimi": ["kimi"],
        "kilo_code": ["kilo", "run"],
        "kilo": ["kilo", "run"],
        "gemini": ["gemini"],
        "codex": ["codex"],
        "nanocoder": ["nanocoder"],
    }

    def __init__(
        self,
        cli_provider: str,
        name: str = "cli_agent",
        primary_directive: str = None,
        model: str = None,
        session_file: str = None,
        **kwargs,
    ):
        if primary_directive is None:
            primary_directive = f"CLI agent using {cli_provider} for code tasks."

        kwargs.pop("provider", None)
        super().__init__(
            name=name,
            primary_directive=primary_directive,
            model=model,
            provider=cli_provider,
            **kwargs,
        )

        self.cli_provider = cli_provider
        self.session_file = session_file

    def run(self, input_text: str, verbose: bool = False, session_context: str = None, **kwargs):
        from npcpy.gen.cli_agent import run_cli_agent

        full_prompt = f"{session_context}\n\n{input_text}" if session_context else input_text
        return run_cli_agent(
            provider=self.cli_provider,
            prompt=full_prompt,
            model=self.model,
            system_prompt=self.primary_directive,
            session_id=kwargs.get("session_id"),
            history=kwargs.get("messages"),
            images=kwargs.get("images"),
            think=kwargs.get("think"),
            n_samples=kwargs.get("n_samples", 1),
            verbose=verbose,
        )
