"""
Memory integration module for connecting breathe/sleep functionality to REPL loops.
This file provides helper functions to integrate knowledge graph memory with
the various REPL loops (npcsh, npc, wander, spool, pti, yap, guac, alicanto).
"""

import os
from typing import List, Dict, Any, Optional
from npcpy.memory.knowledge_graph import breathe
from npcpy.memory.sleep import sleep, forget
from npcpy.memory.command_history import CommandHistory

def execute_breathe(messages: List[Dict[str, Any]], model: str, provider: str, npc: Any = None) -> str:
    """
    Execute the breathe function to extract facts, lessons, and mistakes from conversation history.
    
    Args:
        messages: List of conversation messages
        model: LLM model to use
        provider: LLM provider to use
        npc: Optional NPC object
        
    Returns:
        String summarizing what was learned
    """
    result = breathe(messages, model, provider, npc)
    return f"📝 **Knowledge Integrated**\n\n{result}"

def execute_sleep(messages: List[Dict[str, Any]], model: str, provider: str, npc: Any = None) -> str:
    """
    Execute the sleep function to consolidate memory and extract important information.
    
    Args:
        messages: List of conversation messages
        model: LLM model to use
        provider: LLM provider to use
        npc: Optional NPC object
        
    Returns:
        String summarizing what was learned and forgotten
    """
    result = sleep(messages, model, provider, npc)
    return f"💤 **Memory Consolidated**\n\n{result}"

def execute_forget(entity: str, model: str, provider: str, npc: Any = None) -> str:
    """
    Execute the forget function to remove specific information from memory.
    
    Args:
        entity: Entity or concept to forget
        model: LLM model to use
        provider: LLM provider to use
        npc: Optional NPC object
        
    Returns:
        String confirming what was forgotten
    """
    result = forget(entity, model, provider, npc)
    return f"🗑️ **Memory Updated**\n\n{result}"

def enhance_prompt_with_memory(prompt: str, npc: Any = None) -> str:
    """
    Enhance a prompt with relevant memory context.
    
    Args:
        prompt: Original prompt
        npc: Optional NPC object
        
    Returns:
        Enhanced prompt with memory context
    """
    # Implementation depends on how memory is stored for the NPC
    if npc and hasattr(npc, "memory"):
        # Retrieve relevant memories based on the prompt
        relevant_memories = npc.memory.retrieve_relevant(prompt)
        if relevant_memories:
            memory_context = "\n\n".join(relevant_memories)
            return f"{prompt}\n\nRelevant context from memory:\n{memory_context}"
    return prompt

def register_memory_commands(router_or_command_dict: Any) -> None:
    """
    Register memory-related commands with a router or command dictionary.
    
    Args:
        router_or_command_dict: Either a CommandRouter object or a dictionary mapping command names to handler functions
    """
    if hasattr(router_or_command_dict, 'route'):
        # It's a router
        router = router_or_command_dict
        if not hasattr(router, 'routes') or 'breathe' not in router.routes:
            @router.route("breathe", "Extract facts, lessons, and mistakes from conversation history")
            def breathe_handler(command: str, **kwargs):
                messages = kwargs.get("messages", [])
                model = kwargs.get("model", "gpt-4")
                provider = kwargs.get("provider", "openai")
                npc = kwargs.get("npc")
                result = execute_breathe(messages, model, provider, npc)
                return {"output": result, "messages": messages}
                
        if not hasattr(router, 'routes') or 'sleep' not in router.routes:
            @router.route("sleep", "Consolidate memory and extract important information")
            def sleep_handler(command: str, **kwargs):
                messages = kwargs.get("messages", [])
                model = kwargs.get("model", "gpt-4")
                provider = kwargs.get("provider", "openai")
                npc = kwargs.get("npc")
                result = execute_sleep(messages, model, provider, npc)
                return {"output": result, "messages": messages}
                
        if not hasattr(router, 'routes') or 'forget' not in router.routes:
            @router.route("forget", "Remove specific information from memory")
            def forget_handler(command: str, **kwargs):
                command_parts = command.split()
                if len(command_parts) < 2:
                    return {"output": "Usage: /forget <entity or concept>", "messages": kwargs.get("messages", [])}
                entity = " ".join(command_parts[1:])
                model = kwargs.get("model", "gpt-4")
                provider = kwargs.get("provider", "openai")
                npc = kwargs.get("npc")
                result = execute_forget(entity, model, provider, npc)
                return {"output": result, "messages": kwargs.get("messages", [])}
    else:
        # Assume it's a command dictionary
        command_dict = router_or_command_dict
        
        if 'breathe' not in command_dict:
            command_dict['breathe'] = lambda cmd, **kwargs: {
                "output": execute_breathe(
                    kwargs.get("messages", []), 
                    kwargs.get("model", "gpt-4"), 
                    kwargs.get("provider", "openai"),
                    kwargs.get("npc")
                ),
                "messages": kwargs.get("messages", [])
            }
            
        if 'sleep' not in command_dict:
            command_dict['sleep'] = lambda cmd, **kwargs: {
                "output": execute_sleep(
                    kwargs.get("messages", []), 
                    kwargs.get("model", "gpt-4"), 
                    kwargs.get("provider", "openai"),
                    kwargs.get("npc")
                ),
                "messages": kwargs.get("messages", [])
            }
            
        if 'forget' not in command_dict:
            command_dict['forget'] = lambda cmd, **kwargs: {
                "output": execute_forget(
                    " ".join(cmd.split()[1:]) if len(cmd.split()) > 1 else "",
                    kwargs.get("model", "gpt-4"), 
                    kwargs.get("provider", "openai"),
                    kwargs.get("npc")
                ) if len(cmd.split()) > 1 else "Usage: /forget <entity or concept>",
                "messages": kwargs.get("messages", [])
            }