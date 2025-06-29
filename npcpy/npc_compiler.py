from email import message
import os
from pyexpat.errors import messages
import yaml
import json
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import random
from datetime import datetime
import hashlib
import pathlib
import fnmatch
import traceback
import subprocess
from typing import Any, Dict, List, Optional, Union
from jinja2 import Environment, FileSystemLoader, Template, Undefined
from sqlalchemy import create_engine, text
import npcpy as npy 


from npcpy.npc_sysenv import (
    ensure_dirs_exist, 
    get_npc_path,
    init_db_tables,
    print_and_process_stream_with_markdown, 
    get_system_message
    )
from npcpy.memory.command_history import CommandHistory

class SilentUndefined(Undefined):
    def _fail_with_undefined_error(self, *args, **kwargs):
        return ""

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------





import math
from PIL import Image


class Jinx:
    ''' 
    
    Jinx is a class that provides methods for rendering jinja templates to execute
    natural language commands within the NPC ecosystem, python, and eventually
    other code languages.
    '''
    def __init__(self, jinx_data=None, jinx_path=None):
        """Initialize a jinx from data or file path"""
        if jinx_path:
            self._load_from_file(jinx_path)
        elif jinx_data:
            self._load_from_data(jinx_data)
        else:
            raise ValueError("Either jinx_data or jinx_path must be provided")
            
    def _load_from_file(self, path):
        """Load jinx from file"""
        jinx_data = load_yaml_file(path)
        if not jinx_data:
            raise ValueError(f"Failed to load jinx from {path}")
        self._load_from_data(jinx_data)
            
    def _load_from_data(self, jinx_data):
        """Load jinx from data dictionary"""
        if not jinx_data or not isinstance(jinx_data, dict):
            raise ValueError("Invalid jinx data provided")
            
        if "jinx_name" not in jinx_data:
            raise KeyError("Missing 'jinx_name' in jinx definition")
            
        self.jinx_name = jinx_data.get("jinx_name")
        self.inputs = jinx_data.get("inputs", [])
        self.description = jinx_data.get("description", "")
        self.steps = self._parse_steps(jinx_data.get("steps", []))
            
    def _parse_steps(self, steps):
        """Parse steps from jinx definition"""
        parsed_steps = []
        for i, step in enumerate(steps):
            if isinstance(step, dict):
                parsed_step = {
                    "name": step.get("name", f"step_{i}"),
                    "engine": step.get("engine", "natural"),
                    "code": step.get("code", "")
                }
                parsed_steps.append(parsed_step)
            else:
                raise ValueError(f"Invalid step format: {step}")
        return parsed_steps
        
    def execute(self,
                input_values, 
                jinxs_dict, 
                jinja_env = None,
                npc = None,
                messages=None):
        """Execute the jinx with given inputs"""
        if jinja_env is None:
            jinja_env = Environment(
                loader=FileSystemLoader([npc.npc_directory, npc.jinxs_directory]),
                undefined=SilentUndefined,
            )
        # Create context with input values and jinxs
        context = (npc.shared_context.copy() if npc else {})
        context.update(input_values)
        context.update({
            "jinxs": jinxs_dict,
            "llm_response": None,
            "output": None
        })
        
        # Process each step in sequence
        for i, step in enumerate(self.steps):
            context = self._execute_step(
                step, 
                context,
                jinja_env, 
                npc=npc, 
                messages=messages, 

            )            

        return context
            
    def _execute_step(self,
                      step, 
                      context, 
                      jinja_env,
                      npc=None,
                      messages=None, 
):
        """Execute a single step of the jinx"""
        engine = step.get("engine", "natural")
        code = step.get("code", "")
        step_name = step.get("name", "unnamed_step")

        
        

        try:
            #print(code)
            template = jinja_env.from_string(code)
            rendered_code = template.render(**context)
            
            engine_template = jinja_env.from_string(engine)
            rendered_engine = engine_template.render(**context)
        
        except Exception as e:
            print(f"Error rendering templates for step {step_name}: {e}")
            rendered_code = code
            rendered_engine = engine
                
        # Execute based on engine type
        if rendered_engine == "natural":
            if rendered_code.strip():
                # Handle streaming case
                response =  npc.get_llm_response(
                    rendered_code,
                    context=context,
                    messages=messages,
                )
               # print(response)
                response_text = response.get("response", "")
                context['output'] = response_text
                context["llm_response"] = response_text
                context["results"] = response_text
                context[step_name] = response_text
                context['messages'] = response.get('messages')
        elif rendered_engine == "python":
            # Setup execution environment
            exec_globals = {
                "__builtins__": __builtins__,
                "npc": npc,
                "context": context,
                "pd": pd,
                "plt": plt,
                "np": np,
                "os": os,
                're': re, 
                "json": json,
                "Path": pathlib.Path,
                "fnmatch": fnmatch,
                "pathlib": pathlib,
                "subprocess": subprocess,
                "get_llm_response": npy.llm_funcs.get_llm_response, 
                
                }
            
            
            # Execute the code
            exec_locals = {}
            exec(rendered_code, exec_globals, exec_locals)
            
            # Update context with results
            context.update(exec_locals)
            
            # Handle explicit output
            if "output" in exec_locals:
                context["output"] = exec_locals["output"]
                context[step_name] = exec_locals["output"]
        else:
            # Handle unknown engine
            context[step_name] = {"error": f"Unsupported engine: {rendered_engine}"}
            
        return context
        
    def to_dict(self):
        """Convert to dictionary representation"""
        return {
            "jinx_name": self.jinx_name,
            "description": self.description,
            "inputs": self.inputs,
            "steps": [
                {
                    "name": step.get("name", f"step_{i}"),
                    "engine": step.get("engine"),
                    "code": step.get("code")
                }
                for i, step in enumerate(self.steps)
            ]
        }
        
    def save(self, directory):
        """Save jinx to file"""
        jinx_path = os.path.join(directory, f"{self.jinx_name}.jinx")
        ensure_dirs_exist(os.path.dirname(jinx_path))
        return write_yaml_file(jinx_path, self.to_dict())
        
    @classmethod
    def from_mcp(cls, mcp_tool):
        """Convert an MCP tool to NPC jinx format"""
        # Extract function info from MCP tool
        try:
            import inspect

            # Get basic info
            doc = mcp_tool.__doc__ or ""
            name = mcp_tool.__name__
            signature = inspect.signature(mcp_tool)
            
            # Extract inputs from signature
            inputs = []
            for param_name, param in signature.parameters.items():
                if param_name != 'self':  # Skip self for methods
                    param_type = param.annotation if param.annotation != inspect.Parameter.empty else None
                    param_default = None if param.default == inspect.Parameter.empty else param.default
                    
                    inputs.append({
                        "name": param_name,
                        "type": str(param_type),
                        "default": param_default
                    })
            
            # Create tool data
            jinx_data = {
                "jinx_name": name,
                "description": doc.strip(),
                "inputs": inputs,
                "steps": [
                    {
                        "name": "mcp_function_call",
                        "engine": "python",
                        "code": f"""
# Call the MCP function
import {mcp_tool.__module__}
output = {mcp_tool.__module__}.{name}(
    {', '.join([f'{inp["name"]}=context.get("{inp["name"]}")' for inp in inputs])}
)
"""
                    }
                ]
            }
            
            return cls(jinx_data=jinx_data)
            
        except: 
            pass    
def load_jinxs_from_directory(directory):
    """Load all jinxs from a directory"""
    jinxs = []
    directory = os.path.expanduser(directory)
    
    if not os.path.exists(directory):
        return jinxs
        
    for filename in os.listdir(directory):
        if filename.endswith(".jinx"):
            try:
                jinx_path = os.path.join(directory, filename)
                jinx = Jinx(jinx_path=jinx_path)
                jinxs.append(jinx)
            except Exception as e:
                print(f"Error loading jinx {filename}: {e}")
                
    return jinxs

# ---------------------------------------------------------------------------
# NPC Class
# ---------------------------------------------------------------------------

class NPC:
    def __init__(
        self,
        file: str = None,
        name: str = None,
        primary_directive: str = None,
        jinxs: list = None,
        tools: list = None,
        model: str = None,
        provider: str = None,
        api_url: str = None,
        api_key: str = None,
        db_conn=None,
        use_global_jinxs=False,
        **kwargs
    ):
        """
        Initialize an NPC from a file path or with explicit parameters
        
        Args:
            file: Path to .npc file or name for the NPC
            primary_directive: System prompt/directive for the NPC
            jinxs: List of jinxs available to the NPC or "*" to load all jinxs
            model: LLM model to use
            provider: LLM provider to use
            api_url: API URL for LLM
            api_key: API key for LLM
            db_conn: Database connection
        """
        if not file and not name and not primary_directive:
            raise ValueError("Either 'file' or 'name' and 'primary_directive' must be provided") 
        if file:
            if file.endswith(".npc"):
                self._load_from_file(file)
            file_parent = os.path.dirname(file)
            self.jinxs_directory = os.path.join(file_parent, "jinxs")
            self.npc_directory = file_parent
        else:
            self.name = name            
            self.primary_directive = primary_directive
            self.model = model 
            self.provider = provider 
            self.api_url = api_url 
            self.api_key = api_key
            #for these cases
            # if npcsh is initialized, use the ~/.npcsh/npc_team
            # otherwise imply
            self.jinxs_directory = os.path.expanduser('~/.npcsh/npc_team/jinxs/')

            self.npc_directory = None # only makes sense when the input is also a file 
            # keep the jinxs tho to enable easieros.path.abspath('./npc_team/')

        self.use_global_jinxs = use_global_jinxs
        
        self.memory_length = 20
        self.memory_strategy = 'recent'
        dirs = []
        if self.npc_directory:
            dirs.append(self.npc_directory)
        if self.jinxs_directory:
            dirs.append(self.jinxs_directory)
            
        self.jinja_env = Environment(
            loader=FileSystemLoader([
                os.path.expanduser(d) for d in dirs
            ]),
            undefined=SilentUndefined,
        )
        
        # Set up database connection
        self.db_conn = db_conn
        if self.db_conn:
            self._setup_db()
            self.command_history = CommandHistory(db=self.db_conn)
            self.memory = self._load_npc_memory()
        else:   
            self.command_history = None
            self.memory = None
            self.tables = None
            
            
        # Load jinxs
        self.jinxs = self._load_npc_jinxs(jinxs or "*")
        
        # Set up shared context for NPC
        self.shared_context = {
            "dataframes": {},
            "current_data": None,
            "computation_results": [],
            "memories":{}
        }
        
        # Add any additional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        if db_conn is not None:
            init_db_tables()
    def _load_npc_memory(self):
        memory = self.command_history.get_messages_by_npc(self.name, n_last=self.memory_length)
        #import pdb 
        #pdb.set_trace()
        memory = [{'role':mem['role'], 'content':mem['content']} for mem in memory]
        
        return memory 
    def _load_from_file(self, file):
        """Load NPC configuration from file"""
        if "~" in file:
            file = os.path.expanduser(file)
        if not os.path.isabs(file):
            file = os.path.abspath(file)
            
        npc_data = load_yaml_file(file)
        if not npc_data:
            raise ValueError(f"Failed to load NPC from {file}")
            
        # Extract core fields
        self.name = npc_data.get("name")
        if not self.name:
            # Fall back to filename if name not in file
            self.name = os.path.splitext(os.path.basename(file))[0]
            
        self.primary_directive = npc_data.get("primary_directive")
        
        # Handle wildcard jinxs specification
        jinxs_spec = npc_data.get("jinxs", "*")
        #print(jinxs_spec)
        if jinxs_spec == "*":
            # Will be loaded in _load_npc_jinxs
            self.jinxs_spec = "*" 
        else:
            self.jinxs_spec = jinxs_spec

        self.model = npc_data.get("model")
        self.provider = npc_data.get("provider")
        self.api_url = npc_data.get("api_url")
        self.api_key = npc_data.get("api_key")
        self.name = npc_data.get("name", self.name)

        # Store path for future reference
        self.npc_path = file
        
        # Set NPC-specific jinxs directory path
        self.npc_jinxs_directory = os.path.join(os.path.dirname(file), "jinxs")
    def get_system_prompt(self, simple=False):
        if simple:
            return self.primary_directive
        else:
            return get_system_message(self)
    def _setup_db(self):
        """Set up database tables and determine type"""
        try:

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

        except Exception as e:
            print(f"Error setting up database: {e}")
            self.tables = None
            self.db_type = None    
    def _load_npc_jinxs(self, jinxs):
        """Load and process NPC-specific jinxs"""
        npc_jinxs = []
        
        # Handle wildcard case - load all jinxs from the jinxs directory
        if jinxs == "*":
            #print(f'loading all jinxs for {self.name}')
            # Try to find jinxs in NPC-specific jinxs dir first
            #print(self.npc_jinxs_directory)
            npc_jinxs.extend(load_jinxs_from_directory(self.jinxs_directory))
            #print(npc_jinxs)               
            if os.path.exists(self.jinxs_directory):
                npc_jinxs.extend(load_jinxs_from_directory(self.jinxs_directory))                
            # Return all loaded jinxs
            self.jinxs_dict = {jinx.jinx_name: jinx for jinx in npc_jinxs}
            #print(npc_jinxs)
            return npc_jinxs
            

        for jinx in jinxs:
            #need to add a block here for mcp jinxs.
                
            if isinstance(jinx, Jinx):
                npc_jinxs.append(jinx)
            elif isinstance(jinx, dict):
                npc_jinxs.append(Jinx(jinx_data=jinx))
            
                # Try to load from file
                jinx_path = None
                jinx_name = jinx
                if not jinx_name.endswith(".jinx"):
                    jinx_name += ".jinx"
                
                # Check NPC-specific jinxs directory first
                if hasattr(self, 'jinxs_directory') and os.path.exists(self.jinxs_directory):
                    candidate_path = os.path.join(self.jinxs_directory, jinx_name)
                    if os.path.exists(candidate_path):
                        jinx_path = candidate_path
                        
                if jinx_path:
                    try:
                        jinx_obj = Jinx(jinx_path=jinx_path)
                        npc_jinxs.append(jinx_obj)
                    except Exception as e:
                        print(f"Error loading jinx {jinx_path}: {e}")
        
        # Update jinxs dictionary
        self.jinxs_dict = {jinx.jinx_name: jinx for jinx in npc_jinxs}
        return npc_jinxs
    
    def get_llm_response(self, 
                         request,
                         jinxs= None,
                         tools=None,
                         messages: Optional[List[Dict[str, str]]] = None,
                         **kwargs):
        """Get a response from the LLM"""
        
        # Call the LLM
        response = npy.llm_funcs.get_llm_response(
            request, 
            model=self.model, 
            provider=self.provider, 
            npc=self, 
            jinxs=jinxs,
            tools = tools, 
            messages=self.memory if messages is None else messages,
            **kwargs
        )        
        
        return response
    
    def execute_jinx(self, jinx_name, inputs, conversation_id=None, message_id=None, team_name=None):
        """Execute a jinx by name"""
        # Find the jinx
        if jinx_name in self.jinxs_dict:
            jinx = self.jinxs_dict[jinx_name]
        elif jinx_name in self.jinxs_dict:
            jinx = self.jinxs_dict[jinx_name]
        else:
            return {"error": f"jinx '{jinx_name}' not found"}
        
        result = jinx.execute(
            input_values=inputs,
            context=self.shared_context,
            jinja_env=self.jinja_env,
            npc=self
        )
        if self.db_conn is not None:
            self.db_conn.add_jinx_call(
                triggering_message_id=message_id,
                conversation_id=conversation_id,
                jinx_name=jinx_name,
                jinx_inputs=inputs,
                jinx_output=result,
                status="success",
                error_message=None,
                duration_ms=None,
                npc_name=self.name,
                team_name=team_name,
            )
        return result
    
    def check_llm_command(self,
                          command, 
                          messages=None,
                          context=None,
                          team=None,
                          stream=False):
        """Check if a command is for the LLM"""
        if context is None:
            context = self.shared_context            
        # Call the LLM command checker
        return npy.llm_funcs.check_llm_command(
            command,
            model=self.model,
            provider=self.provider,
            team=team,
            messages=messages,
            context=context,
            stream=stream            
        )
    
    def handle_agent_pass(self, 
                          npc_to_pass,
                          command, 
                          messages=None, 
                          context=None, 
                          shared_context=None, 
                          stream = False):
        """Pass a command to another NPC"""
        print('handling agent pass')
        if isinstance(npc_to_pass, NPC):
            target_npc = npc_to_pass
        else:
            # Try to find the NPC by name
            try:
                npc_path = get_npc_path(npc_to_pass, "~/npcsh_history.db")
                if not npc_path:
                    return {"error": f"NPC '{npc_to_pass}' not found"}
                    
                target_npc = NPC(npc_path, db_conn=self.db_conn)
            except Exception as e:
                return {"error": f"Error loading NPC '{npc_to_pass}': {e}"}
        
        # Update shared context
        if shared_context is not None:
            self.shared_context = shared_context
            
        # Add a note that this command was passed from another NPC
        updated_command = (
            command
            + "\n\n"
            + f"NOTE: THIS COMMAND HAS BEEN PASSED FROM {self.name} TO YOU, {target_npc.name}.\n"
            + "PLEASE CHOOSE ONE OF THE OTHER OPTIONS WHEN RESPONDING."
        )
        
        # Pass to the target NPC
        return target_npc.check_llm_command(
            updated_command,
            messages=messages,
            context=self.shared_context,
            stream = stream
            
            
        )
    
    def to_dict(self):
        """Convert NPC to dictionary representation"""
        return {
            "name": self.name,
            "primary_directive": self.primary_directive,
            "model": self.model,
            "provider": self.provider,
            "api_url": self.api_url,
            "api_key": self.api_key,
            "jinxs": [jinx.to_dict() if isinstance(jinx, Jinx) else jinx for jinx in self.jinxs],
            "use_global_jinxs": self.use_global_jinxs
        }
        
    def save(self, directory=None):
        """Save NPC to file"""
        if directory is None:
            directory = self.npc_directory
            
        ensure_dirs_exist(directory)
        npc_path = os.path.join(directory, f"{self.name}.npc")
        
        return write_yaml_file(npc_path, self.to_dict())
    
    def __str__(self):
        """String representation of NPC"""
        return f"NPC: {self.name}\nDirective: {self.primary_directive}\nModel: {self.model}\nProvider: {self.provider}\nAPI URL: {self.api_url}\njinxs: {', '.join([jinx.jinx_name for jinx in self.jinxs])}"

class Team:
    def __init__(self, 
                 team_path=None, 
                 npcs=None, 
                 forenpc=None,
                 jinxs=None,                   
                 db_conn=None):
        """
        Initialize an NPC team from directory or list of NPCs
        
        Args:
            team_path: Path to team directory
            npcs: List of NPC objects
            db_conn: Database connection
        """
        self.npcs = {}
        self.sub_teams = {}
        self.jinxs_dict = jinxs or {}
        self.db_conn = db_conn
        self.team_path = os.path.expanduser(team_path) if team_path else None
        self.databases = {}
        self.mcp_servers = {}
        if forenpc is not None:
            self.forenpc = forenpc
        else:
            self.forenpc  = npcs[0] if npcs else None
        
        
        if team_path:
            self.name = os.path.basename(os.path.abspath(team_path))
        else:
            self.name = "custom_team"
        self.context = {}
        self.shared_context = {
            "intermediate_results": {},
            "dataframes": {},
            "memories": {},          
            "execution_history": [],   
            "npc_messages": {}                 
            }
                
        if team_path:
            print('loading npc team from directory')
            self._load_from_directory()
            
        elif npcs:
            for npc in npcs:
                self.npcs[npc.name] = npc
            

        
        self.jinja_env = Environment(undefined=SilentUndefined)
        
            
        if db_conn is not None:
            init_db_tables()
            
        
    def _load_from_directory(self):
        """Load team from directory"""
        if not os.path.exists(self.team_path):
            raise ValueError(f"Team directory not found: {self.team_path}")
        
        # Load team context if available
        self.context = self._load_team_context()
        # Load NPCs
        for filename in os.listdir(self.team_path):
            print('filename: ', filename)
            if filename.endswith(".npc"):
                try:
                    npc_path = os.path.join(self.team_path, filename)
                    npc = NPC(npc_path, db_conn=self.db_conn)
                    self.npcs[npc.name] = npc
                    
                except Exception as e:
                    print(f"Error loading NPC {filename}: {e}")
        
        # Load jinxs from jinxs directory
        jinxs_dir = os.path.join(self.team_path, "jinxs")
        if os.path.exists(jinxs_dir):
            for jinx in load_jinxs_from_directory(jinxs_dir):
                self.jinxs_dict[jinx.jinx_name] = jinx
        
        # Load sub-teams (subfolders)
        self._load_sub_teams()
        print(self.jinxs_dict)
    def _load_team_context(self):
        """Load team context from .ctx file"""

                                
        #check if any .ctx file exists 
        for fname in os.listdir(self.team_path):
            if fname.endswith('.ctx'):
                # do stuff on the file
                
                ctx_data = load_yaml_file(os.path.join(self.team_path, fname))
                
                if ctx_data is not None:
                    if 'mcp_servers' in ctx_data:
                        self.mcp_servers = ctx_data['mcp_servers']
                    if 'databases' in ctx_data:
                        self.databases = ctx_data['databases']

                    # check other potential keys
                    for key, item in ctx_data.items():
                        if key not in ['name', 'mcp_servers', 'databases', 'context']:
                            self.shared_context[key] = item
                return ctx_data
        return {}
        
    def _load_sub_teams(self):
        """Load sub-teams from subdirectories"""
        for item in os.listdir(self.team_path):
            item_path = os.path.join(self.team_path, item)
            if (os.path.isdir(item_path) and 
                not item.startswith('.') and 
                item != "jinxs"):
                
                # Check if directory contains NPCs
                if any(f.endswith(".npc") for f in os.listdir(item_path) 
                      if os.path.isfile(os.path.join(item_path, f))):
                    try:
                        sub_team = Team(team_path=item_path, db_conn=self.db_conn)
                        self.sub_teams[item] = sub_team
                    except Exception as e:
                        print(f"Error loading sub-team {item}: {e}")
        
    def get_forenpc(self):
        """
        Get the forenpc (coordinator) for this team.
        The forenpc is set only if explicitly defined in the context.
                
        """
        if isinstance(self.forenpc, NPC):
            return self.forenpc
        if hasattr(self, 'context') and self.context and 'forenpc' in self.context:
            forenpc_ref = self.context['forenpc']
            
            # Handle Jinja template references
            if '{{ref(' in forenpc_ref:
                # Extract NPC name from {{ref('npc_name')}}
                match = re.search(r"{{\s*ref\('([^']+)'\)\s*}}", forenpc_ref)
                if match:
                    forenpc_name = match.group(1)
                    if forenpc_name in self.npcs:
                        return self.npcs[forenpc_name]
            elif forenpc_ref in self.npcs:
                return self.npcs[forenpc_ref]
        else:
            forenpc_model=self.context.get('model', 'llama3.2'),
            forenpc_provider=self.context.get('provider', 'ollama'),
            forenpc_api_key=self.context.get('api_key', None),
            forenpc_api_url=self.context.get('api_url', None)
            
            forenpc = NPC(name='forenpc', 
                          primary_directive="""You are the forenpc of the team, coordinating activities 
                                                between NPCs on the team, verifying that results from 
                                                NPCs are high quality and can help to adequately answer 
                                                user requests.""", 
                            model=forenpc_model,
                            provider=forenpc_provider,
                            api_key=forenpc_api_key,
                            api_url=forenpc_api_url,                            
                                                )
        return None
    
    def get_npc(self, npc_ref):
        """
        Get an NPC by reference, handling hierarchical references
        Example: "analysis_team.data_scientist" gets the data_scientist 
        from the analysis_team sub-team
        """
        if '.' in npc_ref:
            # Handle hierarchical reference
            name, npc_name = npc_ref.split('.', 1)
            if name in self.sub_teams:
                return self.sub_teams[name].get_npc(npc_name)
        elif npc_ref in self.npcs:
            return self.npcs[npc_ref]
        
        return None
    
    def orchestrate(self, request):
        """Orchestrate a request through the team"""
        forenpc = self.get_forenpc()
        if not forenpc:
            return {"error": "No forenpc available to coordinate the team"}
        
        # Log the orchestration start
        log_entry(
            self.name,
            "orchestration_start",
            {"request": request}
        )
        
        # Initial request goes to forenpc
        result = forenpc.check_llm_command(request,
            context=getattr(self, 'context', {}),
            #shared_context=self.shared_context,
            team = self, 
        )
        
        # Track execution until complete
        while True:
            # Save the result
            completion_prompt= "" 
            if isinstance(result, dict):
                self.shared_context["execution_history"].append(result)
                
                # Track messages by NPC
                if result.get("messages") and result.get("npc_name"):
                    if result["npc_name"] not in self.shared_context["npc_messages"]:
                        self.shared_context["npc_messages"][result["npc_name"]] = []
                    self.shared_context["npc_messages"][result["npc_name"]].extend(
                        result["messages"]
                    )
                
                completion_prompt += f"""Context:
                    User request '{request}', previous agent
                    
                    previous agent returned:
                    {result.get('output')}

                    
                Instructions:

                    Check whether the response is relevant to the user's request.

                """
                if self.npcs is None or len(self.npcs) == 0:
                    completion_prompt += f"""
                    The team has no members, so the forenpc must handle the request alone.
                    """
                else:
                    completion_prompt += f"""
                    
                    These are all the members of the team: {', '.join(self.npcs.keys())}

                    Therefore, if you are trying to evaluate whether a request was fulfilled relevantly,
                    consider that requests are made to the forenpc: {forenpc.name}
                    and that the forenpc must pass those along to the other npcs. 
                    """
                completion_prompt += f"""

                Mainly concern yourself with ensuring there are no
                glaring errors nor fundamental mishaps in the response.
                Do not consider stylistic hiccups as the answers being
                irrelevant. By providing responses back to for the user to
                comment on, they can can more efficiently iterate and resolve any issues by 
                prompting more clearly.
                natural language itself is very fuzzy so there will always be some level
                of misunderstanding, but as long as the response is clearly relevant 
                to the input request and along the user's intended direction,
                it is considered relevant.
                               

                If there is enough information to begin a fruitful conversation with the user, 
                please consider the request relevant so that we do not
                arbritarily stall business logic which is more efficiently
                determined by iterations than through unnecessary pedantry.

                It is more important to get a response to the user
                than to account for all edge cases, so as long as the response more or less tackles the
                initial problem to first order, consider it relevant.

                Return a JSON object with:
                    -'relevant' with boolean value
                    -'explanation' for irrelevance with quoted citations in your explanation noting why it is irrelevant to user input must be a single string.
                Return only the JSON object."""
            # Check if the result is complete
            completion_check = npy.llm_funcs.get_llm_response(
                completion_prompt, 
                model=forenpc.model,
                provider=forenpc.provider,
                api_key=forenpc.api_key,
                api_url=forenpc.api_url,
                npc=forenpc,
                format="json"
            )
            # Extract completion status
            if isinstance(completion_check.get("response"), dict):
                complete = completion_check["response"].get("relevant", False)
                explanation = completion_check["response"].get("explanation", "")
            else:
                # Default to incomplete if format is wrong
                complete = False
                explanation = "Could not determine completion status"
            
            #import pdb 
            #pdb.set_trace()
            if complete:
                
                debrief = npy.llm_funcs.get_llm_response(
                    f"""Context:
                    Original request: {request}
                    Execution history: {self.shared_context['execution_history']}

                    Instructions:
                    Provide summary of actions taken and recommendations.
                    Return a JSON object with:
                    - 'summary': Overview of what was accomplished
                    - 'recommendations': Suggested next steps
                    Return only the JSON object.""",
                    model=forenpc.model,
                    provider=forenpc.provider,
                    api_key=forenpc.api_key,
                    api_url=forenpc.api_url,
                    npc=forenpc,
                    format="json"
                )
                
                
                return {
                    "debrief": debrief.get("response"),
                    "output": result.get("output"),
                    "execution_history": self.shared_context["execution_history"],
                }
            else:
                # Continue with updated request
                updated_request = (
                    request
                    + "\n\nThe request has not yet been fully completed. "
                    + explanation
                    + "\nPlease address only the remaining parts of the request."
                )
                print('updating request', updated_request)

                
                # Call forenpc again
                result = forenpc.check_llm_command(
                    updated_request,
                    context=getattr(self, 'context', {}),
                    stream = False,
                    team = self
                    
                )
                
    def to_dict(self):
        """Convert team to dictionary representation"""
        return {
            "name": self.name,
            "npcs": {name: npc.to_dict() for name, npc in self.npcs.items()},
            "sub_teams": {name: team.to_dict() for name, team in self.sub_teams.items()},
            "jinxs": {name: jinx.to_dict() for name, jinx in self.jinxs.items()},
            "context": getattr(self, 'context', {})
        }
    
    def save(self, directory=None):
        """Save team to directory"""
        if directory is None:
            directory = self.team_path
            
        if not directory:
            raise ValueError("No directory specified for saving team")
            
        # Create team directory
        ensure_dirs_exist(directory)
        
        # Save context
        if hasattr(self, 'context') and self.context:
            ctx_path = os.path.join(directory, "team.ctx")
            write_yaml_file(ctx_path, self.context)
            
        # Save NPCs
        for npc in self.npcs.values():
            npc.save(directory)
            
        # Create jinxs directory
        jinxs_dir = os.path.join(directory, "jinxs")
        ensure_dirs_exist(jinxs_dir)
        
        # Save jinxs
        for jinx in self.jinxs.values():
            jinx.save(jinxs_dir)
            
        # Save sub-teams
        for team_name, team in self.sub_teams.items():
            team_dir = os.path.join(directory, team_name)
            team.save(team_dir)
            
        return True

# ---------------------------------------------------------------------------
# Pipeline Runner
# ---------------------------------------------------------------------------

class Pipeline:
    def __init__(self, pipeline_data=None, pipeline_path=None, npc_team=None):
        """Initialize a pipeline from data or file path"""
        self.npc_team = npc_team
        self.steps = []
        
        if pipeline_path:
            self._load_from_path(pipeline_path)
        elif pipeline_data:
            self.name = pipeline_data.get("name", "unnamed_pipeline")
            self.steps = pipeline_data.get("steps", [])
        else:
            raise ValueError("Either pipeline_data or pipeline_path must be provided")
            
    def _load_from_path(self, path):
        """Load pipeline from file"""
        pipeline_data = load_yaml_file(path)
        if not pipeline_data:
            raise ValueError(f"Failed to load pipeline from {path}")
            
        self.name = os.path.splitext(os.path.basename(path))[0]
        self.steps = pipeline_data.get("steps", [])
        self.pipeline_path = path
        
    def execute(self, initial_context=None):
        """Execute the pipeline with given context"""
        context = initial_context or {}
        results = {}
        
        # Initialize database tables
        init_db_tables()
        
        # Generate pipeline hash for tracking
        pipeline_hash = self._generate_hash()
        
        # Create results table specific to this pipeline
        results_table = f"{self.name}_results"
        self._ensure_results_table(results_table)
        
        # Create run entry
        run_id = self._create_run_entry(pipeline_hash)
        
        # Add utility functions to context
        context.update({
            "ref": lambda step_name: results.get(step_name),
            "source": self._fetch_data_from_source,
        })
        
        # Execute each step
        for step in self.steps:
            step_name = step.get("step_name")
            if not step_name:
                raise ValueError(f"Missing step_name in step: {step}")
                
            # Get NPC for this step
            npc_name = self._render_template(step.get("npc", ""), context)
            npc = self._get_npc(npc_name)
            if not npc:
                raise ValueError(f"NPC {npc_name} not found for step {step_name}")
                
            # Render task template
            task = self._render_template(step.get("task", ""), context)
            
            # Execute with appropriate NPC
            model = step.get("model", npc.model)
            provider = step.get("provider", npc.provider)
            
            # Check for special mixa (mixture of agents) mode
            mixa = step.get("mixa", False)
            if mixa:
                response = self._execute_mixa_step(step, context, npc, model, provider)
            else:
                # Check for data source
                source_matches = re.findall(r"{{\s*source\('([^']+)'\)\s*}}", task)
                if source_matches:
                    response = self._execute_data_source_step(step, context, source_matches, npc, model, provider)
                else:
                    # Standard LLM execution
                    llm_response = npy.llm_funcs.get_llm_response(task, model=model, provider=provider, npc=npc)
                    response = llm_response.get("response", "")
            
            # Store result
            results[step_name] = response
            context[step_name] = response
            
            # Save to database
            self._store_step_result(run_id, step_name, npc_name, model, provider, 
                                   {"task": task}, response, results_table)
            
        # Return all results
        return {
            "results": results,
            "run_id": run_id
        }
        
    def _render_template(self, template_str, context):
        """Render a template with the given context"""
        if not template_str:
            return ""
            
        try:
            template = Template(template_str)
            return template.render(**context)
        except Exception as e:
            print(f"Error rendering template: {e}")
            return template_str
            
    def _get_npc(self, npc_name):
        """Get NPC by name from team"""
        if not self.npc_team:
            raise ValueError("No NPC team available")
            
        return self.npc_team.get_npc(npc_name)
        
    def _generate_hash(self):
        """Generate a hash for the pipeline"""
        if hasattr(self, 'pipeline_path') and self.pipeline_path:
            with open(self.pipeline_path, 'r') as f:
                content = f.read()
            return hashlib.sha256(content.encode()).hexdigest()
        else:
            # Generate hash from steps
            content = json.dumps(self.steps)
            return hashlib.sha256(content.encode()).hexdigest()
            
    def _ensure_results_table(self, table_name):
        """Ensure results table exists"""
        db_path = "~/npcsh_history.db"
        with sqlite3.connect(os.path.expanduser(db_path)) as conn:
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    step_name TEXT,
                    npc_name TEXT,
                    model TEXT,
                    provider TEXT,
                    inputs TEXT,
                    outputs TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(run_id) REFERENCES pipeline_runs(run_id)
                )
            """)
            conn.commit()
            
    def _create_run_entry(self, pipeline_hash):
        """Create run entry in pipeline_runs table"""
        db_path = "~/npcsh_history.db"
        with sqlite3.connect(os.path.expanduser(db_path)) as conn:
            cursor = conn.execute(
                "INSERT INTO pipeline_runs (pipeline_name, pipeline_hash, timestamp) VALUES (?, ?, ?)",
                (self.name, pipeline_hash, datetime.now())
            )
            conn.commit()
            return cursor.lastrowid
            
    def _store_step_result(self, run_id, step_name, npc_name, model, provider, inputs, outputs, table_name):
        """Store step result in database"""
        db_path = "~/npcsh_history.db"
        with sqlite3.connect(os.path.expanduser(db_path)) as conn:
            conn.execute(
                f"""
                INSERT INTO {table_name} 
                (run_id, step_name, npc_name, model, provider, inputs, outputs)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    step_name,
                    npc_name,
                    model,
                    provider,
                    json.dumps(self._clean_for_json(inputs)),
                    json.dumps(self._clean_for_json(outputs))
                )
            )
            conn.commit()
            
    def _clean_for_json(self, obj):
        """Clean an object for JSON serialization"""
        if isinstance(obj, dict):
            return {
                k: self._clean_for_json(v)
                for k, v in obj.items()
                if not k.startswith("_") and not callable(v)
            }
        elif isinstance(obj, list):
            return [self._clean_for_json(i) for i in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)
            
    def _fetch_data_from_source(self, table_name):
        """Fetch data from a database table"""
        db_path = "~/npcsh_history.db"
        try:
            engine = create_engine(f"sqlite:///{os.path.expanduser(db_path)}")
            df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
            return df.to_json(orient="records")
        except Exception as e:
            print(f"Error fetching data from {table_name}: {e}")
            return "[]"
            
    def _execute_mixa_step(self, step, context, npc, model, provider):
        """Execute a mixture of agents step"""
        # Get task template
        task = self._render_template(step.get("task", ""), context)
        
        # Get configuration
        mixa_turns = step.get("mixa_turns", 5)
        num_generating_agents = len(step.get("mixa_agents", []))
        if num_generating_agents == 0:
            num_generating_agents = 3  # Default
            
        num_voting_agents = len(step.get("mixa_voters", []))
        if num_voting_agents == 0:
            num_voting_agents = 3  # Default
            
        # Step 1: Initial Response Generation
        round_responses = []
        for _ in range(num_generating_agents):
            response = npy.llm_funcs.get_llm_response(task, model=model, provider=provider, npc=npc)
            round_responses.append(response.get("response", ""))
            
        # Loop for each round of voting and refining
        for turn in range(1, mixa_turns + 1):
            # Step 2: Voting by agents
            votes = [0] * len(round_responses)
            for _ in range(num_voting_agents):
                voted_index = random.choice(range(len(round_responses)))
                votes[voted_index] += 1
                
            # Step 3: Refinement feedback
            refined_responses = []
            for i, resp in enumerate(round_responses):
                feedback = (
                    f"Current responses and their votes:\n" + 
                    "\n".join([f"Response {j+1}: {r[:100]}... - Votes: {votes[j]}" 
                             for j, r in enumerate(round_responses)]) +
                    f"\n\nRefine your response #{i+1}: {resp}"
                )
                
                response = npy.llm_funcs.get_llm_response(feedback, model=model, provider=provider, npc=npc)
                refined_responses.append(response.get("response", ""))
                
            # Update responses for next round
            round_responses = refined_responses
            
        # Final synthesis
        synthesis_prompt = (
            "Synthesize these responses into a coherent answer:\n" +
            "\n".join(round_responses)
        )
        final_response = npy.llm_funcs.get_llm_response(synthesis_prompt, model=model, provider=provider, npc=npc)
        
        return final_response.get("response", "")
        
    def _execute_data_source_step(self, step, context, source_matches, npc, model, provider):
        """Execute a step with data source"""
        task_template = step.get("task", "")
        table_name = source_matches[0]
        
        try:
            # Fetch data
            db_path = "~/npcsh_history.db"
            engine = create_engine(f"sqlite:///{os.path.expanduser(db_path)}")
            df = pd.read_sql(f"SELECT * FROM {table_name}", engine)
            
            # Handle batch mode vs. individual processing
            if step.get("batch_mode", False):
                # Replace source reference with all data
                data_str = df.to_json(orient="records")
                task = task_template.replace(f"{{{{ source('{table_name}') }}}}", data_str)
                task = self._render_template(task, context)
                
                # Process all at once
                response = npy.llm_funcs.get_llm_response(task, model=model, provider=provider, npc=npc)
                return response.get("response", "")
            else:
                # Process each row individually
                results = []
                for _, row in df.iterrows():
                    # Replace source reference with row data
                    row_data = json.dumps(row.to_dict())
                    row_task = task_template.replace(f"{{{{ source('{table_name}') }}}}", row_data)
                    row_task = self._render_template(row_task, context)
                    
                    # Process row
                    response = npy.llm_funcs.get_llm_response(row_task, model=model, provider=provider, npc=npc)
                    results.append(response.get("response", ""))
                    
                return results
        except Exception as e:
            print(f"Error processing data source {table_name}: {e}")
            return f"Error: {str(e)}"



def log_entry(entity_id, entry_type, content, metadata=None, db_path="~/npcsh_history.db"):
    """Log an entry for an NPC or team"""
    db_path = os.path.expanduser(db_path)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO npc_log (entity_id, entry_type, content, metadata) VALUES (?, ?, ?, ?)",
            (entity_id, entry_type, json.dumps(content), json.dumps(metadata) if metadata else None)
        )
        conn.commit()

def get_log_entries(entity_id, entry_type=None, limit=10, db_path="~/npcsh_history.db"):
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


def load_yaml_file(file_path):
    """Load a YAML file with error handling"""
    try:
        with open(os.path.expanduser(file_path), 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading YAML file {file_path}: {e}")
        return None

def write_yaml_file(file_path, data):
    """Write data to a YAML file"""
    try:
        with open(os.path.expanduser(file_path), 'w') as f:
            yaml.dump(data, f)
        return True
    except Exception as e:
        print(f"Error writing YAML file {file_path}: {e}")
        return False

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



def initialize_npc_project(
    directory=None,
    templates=None,
    context=None,
    model=None,
    provider=None,
) -> str:
    """Initialize an NPC project"""
    if directory is None:
        directory = os.getcwd()

    npc_team_dir = os.path.join(directory, "npc_team")
    os.makedirs(npc_team_dir, exist_ok=True)
    
    for subdir in ["jinxs", "assembly_lines", "sql_models", "jobs"]:
        os.makedirs(os.path.join(npc_team_dir, subdir), exist_ok=True)
    
    forenpc_path = os.path.join(npc_team_dir, "sibiji.npc")
    

    # Always ensure default NPC exists
    if not os.path.exists(forenpc_path):
        # Use your new NPC class to create sibiji
        default_npc = {
            "name": "sibiji",
            "primary_directive": "You are sibiji, the forenpc of an NPC team...",
            "model": model or "llama3.2",
            "provider": provider or "ollama"
        }
        with open(forenpc_path, "w") as f:
            yaml.dump(default_npc, f)
            
    return f"NPC project initialized in {npc_team_dir}"





def execute_jinx_command(
    jinx: Jinx,
    args: List[str],
    messages=None,
    npc: NPC = None,
) -> Dict[str, Any]:
    """
    Execute a jinx command with the given arguments.
    """
    # Extract inputs for the current jinx
    input_values = extract_jinx_inputs(args, jinx)

    # print(f"Input values: {input_values}")
    # Execute the jinx with the extracted inputs

    jinx_output = jinx.execute(
        input_values,
        jinx.jinx_name,
        npc=npc,
    )

    return {"messages": messages, "output": jinx_output}



def extract_jinx_inputs(args: List[str], jinx: Jinx) -> Dict[str, Any]:
    inputs = {}

    # Create flag mapping
    flag_mapping = {}
    for input_ in jinx.inputs:
        if isinstance(input_, str):
            flag_mapping[f"-{input_[0]}"] = input_
            flag_mapping[f"--{input_}"] = input_
        elif isinstance(input_, dict):
            key = list(input_.keys())[0]
            flag_mapping[f"-{key[0]}"] = key
            flag_mapping[f"--{key}"] = key

    # Process arguments
    used_args = set()
    for i, arg in enumerate(args):
        if arg in flag_mapping:
            # If flag is found, next argument is its value
            if i + 1 < len(args):
                input_name = flag_mapping[arg]
                inputs[input_name] = args[i + 1]
                used_args.add(i)
                used_args.add(i + 1)
            else:
                print(f"Warning: {arg} flag is missing a value.")

    # If no flags used, combine remaining args for first input
    unused_args = [arg for i, arg in enumerate(args) if i not in used_args]
    if unused_args and jinx.inputs:
        first_input = jinx.inputs[0]
        if isinstance(first_input, str):
            inputs[first_input] = " ".join(unused_args)
        elif isinstance(first_input, dict):
            key = list(first_input.keys())[0]
            inputs[key] = " ".join(unused_args)

    # Add default values for inputs not provided
    for input_ in jinx.inputs:
        if isinstance(input_, str):
            if input_ not in inputs:
                if any(args):  # If we have any arguments at all
                    raise ValueError(f"Missing required input: {input_}")
                else:
                    inputs[input_] = None  # Allow None for completely empty calls
        elif isinstance(input_, dict):
            key = list(input_.keys())[0]
            if key not in inputs:
                inputs[key] = input_[key]

    return inputs

