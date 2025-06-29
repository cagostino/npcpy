from typing import Any, Dict, List, Union
from pydantic import BaseModel
from npcpy.data.image import compress_image
from npcpy.npc_sysenv import get_system_message, lookup_provider, render_markdown
import base64
import json
import uuid
import os 
try: 
    import ollama
except ImportError:
    
    pass
except OSError:
    # Handle case where ollama is not installed or not available
    print("Ollama is not installed or not available. Please install it to use this feature.")
try:
    from litellm import completion
except ImportError:
    pass
except OSError:
    # Handle case where litellm is not installed or not available
    pass
def handle_streaming_json(api_params):
    """
    Handles streaming responses when JSON format is requested.
    
    Args:
        api_params (dict): API parameters for the completion call.
        
    Yields:
        Processed chunks of the JSON response.
    """
    json_buffer = ""
    stream = completion(**api_params)
    
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            json_buffer += content
            # Try to parse as valid JSON but only yield once we have complete JSON
            try:
                # Check if we have a complete JSON object
                json.loads(json_buffer)
                # If successful, yield the chunk
                yield chunk
            except json.JSONDecodeError:
                # Not complete JSON yet, continue buffering
                pass
    
    # After the stream ends, try to ensure we have valid JSON
    try:
        final_json = json.loads(json_buffer)
        # Could yield a special "completion" chunk here if needed
    except json.JSONDecodeError:
        # Handle case where stream ended but JSON is invalid
        print(f"Warning: Complete stream did not produce valid JSON: {json_buffer}")
        
        
def get_ollama_response(
    prompt: str,
    model: str,
    images: List[str] = None,
    tools: list = None,
    tool_choice: Dict = None,
    format: Union[str, BaseModel] = None,
    messages: List[Dict[str, str]] = None,
    stream: bool = False,
    attachments: List[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Generates a response using the Ollama API, supporting both streaming and non-streaming.
    """
    import ollama
    image_paths = []
    if images:
        image_paths.extend(images)
    
    # Handle attachments - simply add them to images if they exist
    if attachments:
        for attachment in attachments:
            # Check if file exists
            if os.path.exists(attachment):
                # Extract extension to determine file type
                _, ext = os.path.splitext(attachment)
                ext = ext.lower()
                
                # Handle image attachments
                if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                    image_paths.append(attachment)
                # Handle PDF attachments
                elif ext == '.pdf':
                    try:
                        from npcpy.data.load import load_pdf
                        pdf_data = load_pdf(attachment)
                        if pdf_data is not None:
                            # Extract text and add to prompt
                            texts = json.loads(pdf_data['texts'].iloc[0])
                            pdf_text = "\n\n".join([item.get('content', '') for item in texts])
                            
                            if prompt:
                                prompt += f"\n\nContent from PDF: {os.path.basename(attachment)}\n{pdf_text[:2000]}..."
                            else:
                                prompt = f"Content from PDF: {os.path.basename(attachment)}\n{pdf_text[:2000]}..."
                            
                            # Add images from PDF if needed
                            try:
                                images_data = json.loads(pdf_data['images'].iloc[0])
                                # We would need to save these images temporarily and add paths to image_paths
                                # This would require more complex implementation
                            except Exception as e:
                                print(f"Error processing PDF images: {e}")
                    except Exception as e:
                        print(f"Error processing PDF attachment: {e}")
                # Handle CSV attachments
                elif ext == '.csv':
                    try:
                        from npcpy.data.load import load_csv
                        csv_data = load_csv(attachment)
                        if csv_data is not None:
                            csv_sample = csv_data.head(10).to_string()
                            if prompt:
                                prompt += f"\n\nContent from CSV: {os.path.basename(attachment)} (first 10 rows):\n{csv_sample}"
                            else:
                                prompt = f"Content from CSV: {os.path.basename(attachment)} (first 10 rows):\n{csv_sample}"
                    except Exception as e:
                        print(f"Error processing CSV attachment: {e}")
    
    # Update the user message with processed prompt content
    if prompt:
        if messages and messages[-1]["role"] == "user":
            if isinstance(messages[-1]["content"], str):
                messages[-1]["content"] = prompt
            elif isinstance(messages[-1]["content"], list):
                for i, item in enumerate(messages[-1]["content"]):
                    if item.get("type") == "text":
                        messages[-1]["content"][i]["text"] = prompt
                        break
                else:
                    messages[-1]["content"].append({"type": "text", "text": prompt})
        else:
            messages.append({"role": "user", "content": prompt})
    
    # Add images to the last user message for Ollama
    if image_paths:
        # Find the last user message or create one
        last_user_idx = None
        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                last_user_idx = i
        
        if last_user_idx is None:
            messages.append({"role": "user", "content": ""})
            last_user_idx = len(messages) - 1
            
        # For Ollama, we directly attach the images to the message
        messages[last_user_idx]["images"] = image_paths
    
    # Prepare API parameters
    api_params = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    
    # Add tools if provided
    if tools:
        api_params["tools"] = tools
    
    # Add tool choice if specified
    if tool_choice:
        api_params["tool_choice"] = tool_choice
    
    # Add any additional parameters
    for key, value in kwargs.items():
        if key in [
            "stop",
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "tools",
            "tool_choice",
            "extra_headers",
            "parallel_tool_calls",
            "response_format",
            "user",
        ]:
            api_params[key] = value
    
    # Handle formatting
    if isinstance(format, type) and not stream:
        schema = format.model_json_schema()
        api_params["format"] = schema
    elif isinstance(format, str) and format == "json" and not stream:
        api_params["format"] = "json"
    
    # Create standardized response structure
    result = {
        "response": None,
        "messages": messages.copy(),
        "raw_response": None,
        "tool_calls": []
    }
    
    # Handle streaming
    if stream:
        result["response"] = ollama.chat(**api_params)
        return result
    
    # Non-streaming case
    res = ollama.chat(**api_params)
    result["raw_response"] = res
    
    # Extract the response content
    response_content = res.get("message", {}).get("content")
    result["response"] = response_content
    
    # Handle tool calls if tools were provided
    if tools and "tool_calls" in res.get("message", {}):
        result["tool_calls"] = res["message"]["tool_calls"]
    
    # Append response to messages
    result["messages"].append({"role": "assistant", "content": response_content})
    
    # Handle JSON format if specified
    if format == "json":
        try:
            if isinstance(response_content, str):
                if response_content.startswith("```json"):
                    response_content = (
                        response_content.replace("```json", "")
                        .replace("```", "")
                        .strip()
                    )
                parsed_response = json.loads(response_content)
                result["response"] = parsed_response
        except json.JSONDecodeError:
            result["error"] = f"Invalid JSON response: {response_content}"
    
    return result

def get_litellm_response(
    prompt: str = None,
    model: str = None,
    provider: str = None,
    images: List[str] = None,
    tools: list = None,
    tool_choice: Dict = None,
    tool_map: Dict = None,
    format: Union[str, BaseModel] = None,
    messages: List[Dict[str, str]] = None,
    api_key: str = None,
    api_url: str = None,
    stream: bool = False,
    attachments: List[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Unified function for generating responses using litellm, supporting both streaming and non-streaming.
    """
    # Create standardized response structure
    result = {
        "response": None,
        "messages": messages.copy() if messages else [],
        "raw_response": None,
        "tool_calls": []
    }
    
    # Handle Ollama separately
    if provider == "ollama":
        return get_ollama_response(
            prompt, 
            model, 
            images=images,
            tools=tools, 
            tool_choice=tool_choice,
            format=format, 
            messages=messages, 
            stream=stream, 
            attachments=attachments,
            **kwargs
        )
    
    # Handle JSON format instructions
    if format == "json" and not stream:
        json_instruction = """If you are a returning a json object, begin directly with the opening {.
            If you are returning a json array, begin directly with the opening [.
            Do not include any additional markdown formatting or leading
            ```json tags in your response. The item keys should be based on the ones provided
            by the user. Do not invent new ones."""
            
        if result["messages"] and result["messages"][-1]["role"] == "user":
            if isinstance(result["messages"][-1]["content"], list):
                result["messages"][-1]["content"].append({
                    "type": "text", 
                    "text": json_instruction
                })
            elif isinstance(result["messages"][-1]["content"], str):
                result["messages"][-1]["content"] += "\n" + json_instruction
    
    # Handle images
    if images:
        last_user_idx = None
        for i, msg in enumerate(result["messages"]):
            if msg["role"] == "user":
                last_user_idx = i
        
        if last_user_idx is None:
            result["messages"].append({"role": "user", "content": []})
            last_user_idx = len(result["messages"]) - 1
        
        if isinstance(result["messages"][last_user_idx]["content"], str):
            result["messages"][last_user_idx]["content"] = [
                {"type": "text", "text": result["messages"][last_user_idx]["content"]}
            ]
        elif not isinstance(result["messages"][last_user_idx]["content"], list):
            result["messages"][last_user_idx]["content"] = []
        
        for image_path in images:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(compress_image(image_file.read())).decode("utf-8")
                result["messages"][last_user_idx]["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    }
                )
    
    # Prepare API parameters
    api_params = {
        "messages": result["messages"],
    }
    
    # Handle provider, model, and API settings
    if api_url is not None and provider == "openai-like":
        api_params["api_base"] = api_url
        provider = "openai"
    
    if format == "json" and not stream:
        api_params["response_format"] = {"type": "json_object"}
    elif isinstance(format, BaseModel):
        api_params["response_format"] = format
    if model is None:
        print('model not provided, using defaults')
        model = os.environ.get("NPCSH_CHAT_MODEL", "llama3.2")
    if provider is None:
        provider = os.environ.get("NPCSH_CHAT_PROVIDER", "openai")

    if "/" not in model:
        model_str = f"{provider}/{model}"
    else:
        model_str = model
    
    api_params["model"] = model_str
    
    if api_key is not None:
        api_params["api_key"] = api_key
    
    # Add tools if provided
    if tools:
        api_params["tools"] = tools
    
    # Add tool choice if specified
    if tool_choice:
        api_params["tool_choice"] = tool_choice
    
    # Add additional parameters
    if kwargs:
        for key, value in kwargs.items():
            if key in [
                "stop",
                "temperature",
                "top_p",
                "max_tokens",
                "max_completion_tokens",
                "tools",
                "tool_choice",
                "extra_headers",
                "parallel_tool_calls",
                "response_format",
                "user",
            ]:
                api_params[key] = value
    

    # Handle streaming
    if stream:
        #print('streaming response')
        if format == "json":
            print('streaming json output')
            result["response"] = handle_streaming_json(api_params)
        elif tools:
            # do a call to get tool choice
            result["response"] = completion(**api_params, stream=False)
            result = process_tool_calls(result, tool_map, model, provider, messages, stream=True)
            
        else:
            result["response"] = completion(**api_params, stream=True)
            
        return result
    
    # Non-streaming case
    resp = completion(**api_params)
    result["raw_response"] = resp
    
    # Extract response content
    llm_response = resp.choices[0].message.content
    result["response"] = llm_response
    
    # Extract tool calls if any
    if hasattr(resp.choices[0].message, 'tool_calls') and resp.choices[0].message.tool_calls:
        result["tool_calls"] = resp.choices[0].message.tool_calls
    
    # Handle JSON format requests
    if format == "json":
        try:
            if isinstance(llm_response, str):
                # Clean up JSON response if needed
                if llm_response.startswith("```json"):
                    llm_response = llm_response.replace("```json", "").replace("```", "").strip()
                parsed_json = json.loads(llm_response)
                
                if "json" in parsed_json:
                    result["response"] = parsed_json["json"]
                else:
                    result["response"] = parsed_json
                
        except (json.JSONDecodeError, TypeError) as e:
            print(f"JSON parsing error: {str(e)}")
            print(f"Raw response: {llm_response}")
            result["error"] = "Invalid JSON response"
    
    # Add assistant response to message history
    result["messages"].append(
        {
            "role": "assistant",
            "content": (
                llm_response if isinstance(llm_response, str) else str(llm_response)
            ),
        }
    )
    
    return result


def handle_streaming_json(api_params):
    """
    Handles streaming responses when JSON format is requested.
    """
    json_buffer = ""
    stream = completion(**api_params)
    
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            json_buffer += content
            # Try to parse as valid JSON but only yield once we have complete JSON
            try:
                # Check if we have a complete JSON object
                json.loads(json_buffer)
                # If successful, yield the chunk
                yield chunk
            except json.JSONDecodeError:
                # Not complete JSON yet, continue buffering
                pass
    
    # After the stream ends, try to ensure we have valid JSON
    try:
        final_json = json.loads(json_buffer)
        # Could yield a special "completion" chunk here if needed
    except json.JSONDecodeError:
        # Handle case where stream ended but JSON is invalid
        print(f"Warning: Complete stream did not produce valid JSON: {json_buffer}")



def process_tool_calls(response_dict, tool_map, model, provider, messages, stream=False):
    """
    Process tool calls from a response and execute corresponding tools.
    
    Args:
        response_dict (dict): The response dictionary from get_litellm_response or get_ollama_response
        tool_map (dict): Mapping of tool names to their implementation functions
        model (str): The model to use for follow-up responses
        provider (str): The provider to use for follow-up responses
        messages (list): The current message history
        stream (bool): Whether to stream the response
        
    Returns:
        dict: Updated response dictionary with tool results and final response
    """
    
    result = response_dict.copy()
    result["tool_results"] = []
    #print(tool_map)
    
    # Make sure messages is initialized
    if "messages" not in result:
        result["messages"] = messages if messages else []
    
    # Extract tool calls from the response
    if "response" in result:
        if hasattr(result["response"], "choices") and hasattr(result["response"].choices[0], "message"):
            tool_calls = result["response"].choices[0].message.tool_calls
        elif isinstance(result["response"], dict) and "tool_calls" in result["response"]:
            tool_calls = result["response"]["tool_calls"]
        else:
            tool_calls = None
    else:
        tool_calls = None
    
    if tool_calls is not None:
        for tool_call in tool_calls:
            
            # Extract tool details - handle both Ollama and LiteLLM formats
            if isinstance(tool_call, dict):  # Ollama format
                tool_id = tool_call.get("id", str(uuid.uuid4()))
                tool_name = tool_call.get("function", {}).get("name")
                arguments_str = tool_call.get("function", {}).get("arguments", "{}")
            else:  # LiteLLM format - expect object with attributes
                tool_id = getattr(tool_call, "id", str(uuid.uuid4()))
                # Handle function as either attribute or dict
                if hasattr(tool_call, "function"):
                    if isinstance(tool_call.function, dict):
                        tool_name = tool_call.function.get("name")
                        arguments_str = tool_call.function.get("arguments", "{}")
                    else:
                        tool_name = getattr(tool_call.function, "name", None)
                        arguments_str = getattr(tool_call.function, "arguments", "{}")
                else:
                    raise ValueError("Jinx call missing function attribute or property")
            
            # Parse arguments
            if not arguments_str:
                arguments = {}
            else:
                try:
                    arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
                except json.JSONDecodeError:
                    arguments = {"raw_arguments": arguments_str}

            render_markdown('# tool_call \n - '+ tool_name + '\n - ' + str(arguments))

            
            # Execute the tool if it exists in the tool map
            if tool_name in tool_map:
                try:
                    # Try calling with keyword arguments first
                    tool_result = tool_map[tool_name](**arguments)
                except TypeError:
                    # If that fails, try calling with the arguments as a single parameter
                    tool_result = tool_map[tool_name](arguments)
                # Convert tool result to a serializable format
                serializable_result = None
                tool_result_str = ""
                
                try:
                    # Check if it's TextContent or similar with .text attribute
                    if hasattr(tool_result, 'content') and isinstance(tool_result.content, list):
                        content_list = tool_result.content
                        text_parts = []
                        for item in content_list:
                            if hasattr(item, 'text'):
                                text_parts.append(item.text)
                        tool_result_str = " ".join(text_parts)
                        serializable_result = {"text": tool_result_str}
                    # Handle other common types
                    elif hasattr(tool_result, 'model_dump'):
                        serializable_result = tool_result.model_dump()
                        tool_result_str = str(serializable_result)
                    elif hasattr(tool_result, 'to_dict'):
                        serializable_result = tool_result.to_dict()
                        tool_result_str = str(serializable_result)
                    elif hasattr(tool_result, '__dict__'):
                        serializable_result = tool_result.__dict__
                        tool_result_str = str(serializable_result)
                    elif isinstance(tool_result, (dict, list)):
                        serializable_result = tool_result
                        tool_result_str = json.dumps(tool_result)
                    else:
                        # Fall back to string representation
                        tool_result_str = str(tool_result)
                        serializable_result = {"text": tool_result_str}
                except Exception as e:
                    tool_result_str = f"Error serializing result: {str(e)}"
                    serializable_result = {"error": tool_result_str}
                
                # Store the serializable result
                result["tool_results"].append({
                    "tool_call_id": tool_id,
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "result": serializable_result
                })
                
                # Add the tool call message
                result["messages"].append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tool_id,
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(arguments)
                            }
                        }
                    ]
                })
                
                # Add the tool response message with string content
                result["messages"].append({
                    "role": "tool",
                    "tool_call_id": tool_id,
                    "content": tool_result_str
                })
        
        # Follow up with a request to the LLM
        follow_up_prompt = "Based on the tool results, please provide a helpful response."
        
        # Get follow-up response
        follow_up_response = get_litellm_response(
            prompt=follow_up_prompt,
            model=model,
            provider=provider,
            messages=result["messages"],
            stream=stream,
        )
        
        # Update the result
        if isinstance(follow_up_response, dict):
            if "response" in follow_up_response:
                result["response"] = follow_up_response["response"]
            if "messages" in follow_up_response:
                result["messages"] = follow_up_response["messages"]
    
    return result