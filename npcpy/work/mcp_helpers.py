#!/usr/bin/env python
"""
Raw MCP client with no exception handling and full visibility.
"""

import asyncio
import os
import sys
import json
try:
    import inspect
except: 
    pass
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack

# MCP imports
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Local imports from npcpy
from npcpy.gen.response import get_litellm_response
from npcpy.npc_sysenv import (
    NPCSH_CHAT_MODEL,
    NPCSH_CHAT_PROVIDER,
    NPCSH_API_URL,
)

class MCPClient:
    """
    Raw MCP Client with no exception handling.
    """
    
    def __init__(
        self,
        model: str = NPCSH_CHAT_MODEL,
        provider: str = NPCSH_CHAT_PROVIDER,
        api_url: str = NPCSH_API_URL,
        api_key: Optional[str] = None,
        debug: bool = True,
    ):
        self.model = model
        self.provider = provider
        self.api_url = api_url
        self.api_key = api_key
        self.debug = debug
        self.session = None
        self.exit_stack = AsyncExitStack()
        self.tools = []
        self.available_tools = []
        
    def _log(self, message: str) -> None:
        """Log debug messages."""
        if self.debug:
            print(f"[MCP Client] {message}")

    async def connect_to_server(self, server_script_path: str) -> None:
        """
        Connect to an MCP server.
        
        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        self._log(f"Connecting to server: {server_script_path}")
        
        # Configure server parameters
        command = "python" if server_script_path.endswith('.py') else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )
        
        # Set up the connection
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        read, write = stdio_transport
        
        # Create the session
        self.session = await self.exit_stack.enter_async_context(ClientSession(read, write))
        
        # Initialize the session
        await self.session.initialize()
        
        # List available tools
        response = await self.session.list_tools()
        self.tools = response.tools
        
        # Display tool details for debugging
        for tool in self.tools:
            print(f"\nJinx: {tool.name}")
            print(f"Description: {tool.description}")
            
            # Print all attributes
            for attribute_name in dir(tool):
                if not attribute_name.startswith('_'):
                    attribute = getattr(tool, attribute_name)
                    if not callable(attribute):
                        print(f"  {attribute_name}: {attribute}")
            
            # Check if the tool has source or function definition
            if hasattr(tool, 'source'):
                print(f"Source: {tool.source}")
                
            # Try to inspect the tool function
            try:
                tool_module = inspect.getmodule(tool)
                if tool_module:
                    print(f"Module: {tool_module.__name__}")
                    if hasattr(tool_module, tool.name):
                        tool_func = getattr(tool_module, tool.name)
                        if callable(tool_func):
                            print(f"Function signature: {inspect.signature(tool_func)}")
            except:
                pass
        
        # Convert tools to the format expected by the LLM
        self.available_tools = []
        for tool in self.tools:
            # Use inputSchema if available, otherwise create a default schema
            schema = getattr(tool, "inputSchema", {})
            
            # Create tool definition for LLM
            tool_info = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": schema
                }
            }
            self.available_tools.append(tool_info)
            
            # Print the schema for debugging
            print(f"\nJinx schema for {tool.name}:")
            print(json.dumps(schema, indent=2))
        
        tool_names = [tool.name for tool in self.tools]
        self._log(f"Available tools: {', '.join(tool_names)}")

    async def process_query(
        self,
        query: str,
        messages: Optional[List[Dict[str, str]]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Process a query using the LLM and available tools.
        
        Args:
            query: User query
            messages: Optional conversation history
            stream: Whether to stream the response
            
        Returns:
            Dict with response text and updated messages
        """
        self._log(f"Processing query: {query}")
        
        # Initialize or update messages
        if messages is None:
            messages = []
            
        current_messages = messages.copy()
        if not current_messages or current_messages[-1]["role"] != "user":
            current_messages.append({"role": "user", "content": query})
        elif current_messages[-1]["role"] == "user":
            current_messages[-1]["content"] = query
            
        # Initial LLM call with tools
        self._log("Making initial LLM call with tools")
        response = get_litellm_response(
            model=self.model,
            provider=self.provider,
            api_url=self.api_url,
            api_key=self.api_key,
            messages=current_messages,
            tools=self.available_tools,
            stream=False  # Don't stream the initial call
        )
        
        # Print full response for debugging
        print("\nLLM Response:")
        print(json.dumps(response, indent=2, default=str))
        
        # Extract response content and tool calls
        response_content = response.get("response", "")
        tool_calls = response.get("tool_calls", [])
        
        # Print tool calls for debugging
        print("\nJinx Calls:")
        print(json.dumps(tool_calls, indent=2, default=str))
        
        # Create final text buffer
        final_text = []
        
        # If we have plain text response with no tool calls
        if response_content and not tool_calls:
            final_text.append(response_content)
            
            # Update messages with assistant response
            current_messages.append({
                "role": "assistant",
                "content": response_content
            })
        
        # Process tool calls if any
        if tool_calls:
            self._log(f"Processing {len(tool_calls)} tool calls")
            
            # Get the assistant message with tool calls
            assistant_message = {
                "role": "assistant",
                "content": response_content if response_content else None,
                "tool_calls": []
            }
            
            # Process each tool call
            for tool_call in tool_calls:
                # Extract tool info based on format
                if isinstance(tool_call, dict):
                    tool_id = tool_call.get("id", "")
                    tool_name = tool_call.get("function", {}).get("name", "")
                    tool_args = tool_call.get("function", {}).get("arguments", {})
                else:
                    # Assume object with attributes
                    tool_id = getattr(tool_call, "id", "")
                    tool_name = getattr(tool_call.function, "name", "")
                    tool_args = getattr(tool_call.function, "arguments", {})
                
                # Parse arguments if it's a string
                if isinstance(tool_args, str):
                    print(f"\nJinx args is string: {tool_args}")
                    tool_args = json.loads(tool_args)
                    print(f"Parsed to: {tool_args}")
                
                # Add tool call to assistant message
                assistant_message["tool_calls"].append({
                    "id": tool_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_args) if isinstance(tool_args, dict) else tool_args
                    }
                })
                
                # Execute tool call
                self._log(f"Executing tool: {tool_name} with args: {tool_args}")
                print(f"\nExecuting tool call:")
                print(f"  Jinx name: {tool_name}")
                print(f"  Jinx args: {tool_args}")
                print(f"  Jinx args type: {type(tool_args)}")
                
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
                
                # Call the tool with the arguments exactly as received
                result = await self.session.call_tool(tool_name, tool_args)
                
                # Print full result for debugging
                print("\nJinx Result:")
                print(f"  Result: {result}")
                print(f"  Content: {result.content}")
                print(f"  Content type: {type(result.content)}")
                
                tool_result = result.content
                
                # Handle TextContent objects
                if hasattr(tool_result, 'text'):
                    print(f"  TextContent detected, text: {tool_result.text}")
                    tool_result = tool_result.text
                elif isinstance(tool_result, list) and all(hasattr(item, 'text') for item in tool_result):
                    print(f"  List of TextContent detected")
                    tool_result = [item.text for item in tool_result]
                
                # Add tool result to messages
                current_messages.append(assistant_message)
                current_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": json.dumps(tool_result) if not isinstance(tool_result, str) else str(tool_result)
                })
            
            # Print updated messages for debugging
            print("\nUpdated Messages:")
            print(json.dumps(current_messages, indent=2, default=str))
            
            # Get final response with tool results
            self._log("Getting final response after tool calls")
            final_response = get_litellm_response(
                model=self.model,
                provider=self.provider,
                api_url=self.api_url,
                api_key=self.api_key,
                messages=current_messages,
                stream=stream
            )
            
            final_text.append(final_response.get("response", ""))
            
            # Update messages with final assistant response
            current_messages.append({
                "role": "assistant",
                "content": final_response.get("response", "")
            })
        
        return {
            "response": "\n".join(final_text),
            "messages": current_messages
        }

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")
        
        messages = []
        
        while True:
            query = input("\nQuery: ").strip()
            
            if query.lower() == 'quit':
                break
            
            # Process the query
            result = await self.process_query(query, messages)
            messages = result.get("messages", [])
            
            # Display the response
            print("\nResponse:")
            print(result.get("response", ""))

    async def cleanup(self):
        """Clean up resources"""
        self._log("Cleaning up resources")
        await self.exit_stack.aclose()

async def main():
    """Entry point for the MCP client."""
    if len(sys.argv) < 2:
        print("Usage: python raw_mcp_client.py <path_to_server_script>")
        sys.exit(1)
        
    server_script = sys.argv[1]
    
    # Create and configure the client
    client = MCPClient()
    
    # Connect to the server
    await client.connect_to_server(server_script)
    
    # Run the interactive chat loop
    await client.chat_loop()
    
    # Clean up resources
    await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())