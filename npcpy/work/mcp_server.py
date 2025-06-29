#!/usr/bin/env python
"""
Enhanced MCP server that incorporates functionality from npcpy.routes, 
npcpy.llm_funcs, and npcpy.npc_compiler as tools.
"""

import os
import subprocess
import json
import asyncio

from typing import Optional, Dict, Any, List, Union, Callable
# MCP imports
from mcp.server.fastmcp import FastMCP
import importlib
# npcpy imports
from npcpy.gen.response import get_litellm_response
from npcpy.npc_sysenv import (
    NPCSH_CHAT_MODEL,
    NPCSH_CHAT_PROVIDER,
    NPCSH_API_URL,
    NPCSH_IMAGE_GEN_MODEL,
    NPCSH_IMAGE_GEN_PROVIDER,
    NPCSH_VIDEO_GEN_MODEL,
    NPCSH_VIDEO_GEN_PROVIDER,
    get_model_and_provider,
    lookup_provider,
)

import os
import subprocess
import json
import asyncio
try:
    import inspect
except: 
    pass
from typing import Optional, Dict, Any, List, Union, Callable, get_type_hints
# Add these imports to the top of your file
from functools import wraps
# Initialize the MCP server
mcp = FastMCP("npcpy_enhanced")

# Define the default workspace
DEFAULT_WORKSPACE = os.path.join(os.getcwd(), "workspace")
os.makedirs(DEFAULT_WORKSPACE, exist_ok=True)

# ==================== SYSTEM TOOLS ====================

@mcp.tool()
async def run_server_command(command: str) -> str:
    """
    Run a terminal command in the workspace.
    
    Args:
        command: The shell command to run
        
    Returns:
        The command output or an error message.
    """
    try:
        result = subprocess.run(
            command, 
            cwd=DEFAULT_WORKSPACE, 
            shell=True, 
            capture_output=True, 
            text=True
        )
        return result.stdout or result.stderr
    except Exception as e:
        return str(e)
def make_async_wrapper(func: Callable) -> Callable:
    """Create an async wrapper for sync functions that fixes schema validation issues."""
    
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        # Direct parameter dict case (most common failure)
        if len(args) == 1 and isinstance(args[0], dict):
            params = args[0]
            
            # Fix for search_web - add required kwargs parameter
            if "kwargs" not in params:
                # Create a new dict with the kwargs parameter added
                params = {**params, "kwargs": ""}
            
            # Call the function with the parameters
            if asyncio.iscoroutinefunction(func):
                return await func(**params)
            else:
                return await asyncio.to_thread(func, **params)
        
        # Normal function call or other cases
        if asyncio.iscoroutinefunction(func):
            return await func(*args, **kwargs)
        else:
            return await asyncio.to_thread(func, *args, **kwargs)
    
    # Preserve function metadata
    async_wrapper.__name__ = func.__name__
    async_wrapper.__doc__ = func.__doc__
    async_wrapper.__annotations__ = func.__annotations__
    
    return async_wrapper

# Update your register_module_tools function to use this improved wrapper
def register_module_tools(module_name: str) -> None:
    """
    Register all suitable functions from a module as MCP tools with improved argument handling.
    """
    functions = load_module_functions(module_name)
    for func in functions:
        # Skip functions that don't have docstrings
        if not func.__doc__:
            print(f"Skipping function without docstring: {func.__name__}")
            continue
            
        # Create async wrapper with improved argument handling
        async_func = make_async_wrapper(func)
        
        # Register as MCP tool
        try:
            mcp.tool()(async_func)
            print(f"Registered tool: {func.__name__}")
        except Exception as e:
            print(f"Failed to register {func.__name__}: {e}")
def load_module_functions(module_name: str) -> List[Callable]:
    """
    Dynamically load functions from a module.
    """
    try:
        module = importlib.import_module(module_name)
        # Get all callables from the module that don't start with underscore
        functions = []
        for name, func in inspect.getmembers(module, callable):
            if not name.startswith('_'):
                # Check if it's a function, not a class
                if inspect.isfunction(func) or inspect.ismethod(func):
                    functions.append(func)
        return functions
    except ImportError as e:
        print(f"Warning: Could not import module {module_name}: {e}")
        return []

print("Loading tools from npcpy modules...")

# Load modules from npcpy.routes
try:
    from npcpy.routes import routes
    for route_name, route_func in routes.items():
        if callable(route_func):
            async_func = make_async_wrapper(route_func)
            try:
                mcp.tool()(async_func)
                print(f"Registered route: {route_name}")
            except Exception as e:
                print(f"Failed to register route {route_name}: {e}")
except ImportError as e:
    print(f"Warning: Could not import routes: {e}")


# Load npc_compiler functions
print("Loading functions from npcpy.npc_compiler...")
try:
    import importlib.util
    if importlib.util.find_spec("npcpy.npc_compiler"):
        register_module_tools("npcpy.npc_compiler")
except ImportError:
    print("npcpy.npc_compiler not found, skipping...")

# Load npc_sysenv functions
#print("Loading functions from npcpy.npc_sysenv...")
#register_module_tools("npcpy.npc_sysenv")
register_module_tools("npcpy.memory.search")

register_module_tools("npcpy.work.plan")
register_module_tools("npcpy.work.trigger")
register_module_tools("npcpy.work.desktop")

#print("Loading functions from npcpy.command_history...")
#register_module_tools("npcpy.memory.command_history")


#print("Loading functions from npcpy.npc_sysenv...")
#register_module_tools("npcpy.llm_funcs")


# ==================== MAIN ENTRY POINT ====================

if __name__ == "__main__":
    print(f"Starting enhanced NPCPY MCP server...")
    print(f"Workspace: {DEFAULT_WORKSPACE}")
    
    # Run the server
    mcp.run(transport="stdio")