from typing import Any, Dict, List, Union
from pydantic import BaseModel
from npcpy.data.image import compress_image
from npcpy.npc_sysenv import get_system_message, lookup_provider, render_markdown
import base64
import json
import os
import uuid
import yaml
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = nn = F = None

try:
    import ollama
    HAS_OLLAMA = True
except ImportError:
    ollama = None
    HAS_OLLAMA = False
except OSError:
    ollama = None
    HAS_OLLAMA = False
    logger.warning("Ollama is not installed or not available.")


def _require_ollama() -> None:
    """Raise a clear ImportError when the optional ``ollama`` package is absent.

    Called at each entry point into the ollama provider so that users get a
    descriptive error instead of a bare ``NameError: name 'ollama' is not
    defined`` at an arbitrary call site.
    """
    if not HAS_OLLAMA:
        raise ImportError(
            "The 'ollama' package is required for the ollama provider. "
            "Install it with: pip install ollama"
        )

try:
    import litellm
    from litellm import completion
    litellm.suppress_debug_info = True
except ImportError:
    pass
except OSError:
    pass

def sanitize_messages(messages: list) -> list:
    """Remove orphaned tool_use and tool_result blocks from message history.

    Checks EVERY assistant message with tool_calls (not just the last one)
    to ensure Anthropic never sees a tool_use without a matching tool_result.
    For mid-history orphans, the tool_calls key is removed (keeping text content).
    For tail orphans, the assistant message is stripped entirely.
    Also merges consecutive same-role messages and ensures the conversation
    doesn't end with an assistant message (Anthropic rejects that).
    """
    if not messages:
        return messages

    def _extract_tc_ids(tool_calls_list):
        ids = set()
        for tc in tool_calls_list:
            if isinstance(tc, dict):
                tc_id = tc.get('id') or (tc.get('function') or {}).get('id', '')
            else:
                tc_id = getattr(tc, 'id', '')
            if tc_id:
                ids.add(tc_id)
        return ids

    cleaned = []
    i = 0
    while i < len(messages):
        msg = messages[i]

        if msg.get('role') == 'assistant' and msg.get('tool_calls'):
            expected_ids = _extract_tc_ids(msg['tool_calls'])

            fulfilled_ids = set()
            j = i + 1
            while j < len(messages) and messages[j].get('role') == 'tool':
                tid = messages[j].get('tool_call_id', '')
                if tid:
                    fulfilled_ids.add(tid)
                j += 1

            if expected_ids and expected_ids.issubset(fulfilled_ids):
                cleaned.append(msg)
                for k in range(i + 1, j):
                    cleaned.append(messages[k])
                i = j
            elif not expected_ids and j > i + 1:
                cleaned.append(msg)
                for k in range(i + 1, j):
                    cleaned.append(messages[k])
                i = j
            else:
                text_content = msg.get('content')
                if text_content:
                    cleaned.append({'role': 'assistant', 'content': text_content})
                i = j
        elif msg.get('role') == 'tool':
            content = msg.get('content', '')
            name = msg.get('name', 'tool')
            cleaned.append({
                'role': 'assistant',
                'content': f"[{name} result]: {content}" if name != 'tool' else content
            })
            i += 1
        else:
            cleaned.append(msg)
            i += 1

    merged = []
    for msg in cleaned:
        role = msg.get('role', '')
        if (merged
                and role == merged[-1].get('role')
                and role in ('user', 'assistant')
                and not msg.get('tool_calls')
                and not merged[-1].get('tool_calls')):
            prev_content = merged[-1].get('content', '') or ''
            new_content = msg.get('content', '') or ''
            merged[-1]['content'] = (prev_content + '\n' + new_content).strip()
        else:
            merged.append(msg)

    while merged and merged[-1].get('role') == 'assistant' and not merged[-1].get('tool_calls'):
        merged.pop()

    return merged


def calculate_cost(model: str, input_tokens: int, output_tokens: int, provider: str = None) -> float:
    """Calculate cost in USD for a response using litellm's model cost database."""
    if not model or input_tokens < 0 or output_tokens < 0:
        return 0.0

    def _cost_from_info(info: dict) -> float | None:
        in_cost = info.get("input_cost_per_token") or 0
        out_cost = info.get("output_cost_per_token") or 0
        if not in_cost and not out_cost:
            return None
        return (input_tokens * float(in_cost)) + (output_tokens * float(out_cost))

    try:
        info = litellm.get_model_info(model)
        cost = _cost_from_info(info)
        if cost is not None:
            return cost
    except Exception:
        pass

    # Try with provider prefix (e.g. openai/gpt-4o)
    try:
        resolved_provider = provider or lookup_provider(model)
        if resolved_provider:
            info = litellm.get_model_info(f"{resolved_provider}/{model}")
            cost = _cost_from_info(info)
            if cost is not None:
                return cost
    except Exception:
        pass

    return 0.0

def get_model_context_window(model: str, provider: str = None) -> int:
    """Get the context window size (max input tokens) for a model.

    Uses litellm's model info database. Falls back to provider-specific
    queries (e.g. ollama show) when litellm doesn't have the model.

    Returns 0 if the context window cannot be determined.
    """
    if not model:
        return 0

    try:
        info = litellm.get_model_info(model)
        ctx = info.get("max_input_tokens") or info.get("max_tokens") or 0
        if ctx > 0:
            return ctx
    except Exception:
        pass

    resolved_provider = provider
    if not resolved_provider:
        try:
            resolved_provider = lookup_provider(model)
        except Exception:
            pass

    if resolved_provider:
        try:
            prefixed = f"{resolved_provider}/{model}"
            info = litellm.get_model_info(prefixed)
            ctx = info.get("max_input_tokens") or info.get("max_tokens") or 0
            if ctx > 0:
                return ctx
        except Exception:
            pass

    if resolved_provider == "ollama":
        try:
            client = ollama.Client()
            info = client.show(model)
            params = info.get("model_info", {})
            for key, val in params.items():
                if "context_length" in key:
                    return int(val)
        except Exception:
            pass
        return int(os.environ.get("OLLAMA_NUM_CTX", 32768))

    return 0


def handle_streaming_json(api_params):
    """
    Handles streaming responses when JSON format is requested from LiteLLM.
    """
    json_buffer = ""
    stream = completion(**api_params)
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            json_buffer += content
            try:
                json.loads(json_buffer)
                yield chunk
            except json.JSONDecodeError:
                pass

def get_transformers_response(
   prompt: str = None,
   model=None,
   tokenizer=None, 
   tools: list = None,
   tool_map: Dict = None,
   format: str = None,
   messages: List[Dict[str, str]] = None,
   auto_process_tool_calls: bool = False,
   **kwargs,
) -> Dict[str, Any]:
   import torch
   import json
   import uuid
   from transformers import AutoTokenizer, AutoModelForCausalLM
   
   result = {
       "response": None,
       "messages": messages.copy() if messages else [],
       "raw_response": None,
       "tool_calls": [], 
       "tool_results": []
   }
   
   if model is None or tokenizer is None:
       model_name = model if isinstance(model, str) else "Qwen/Qwen3-1.7b"
       tokenizer = AutoTokenizer.from_pretrained(model_name)
       model = AutoModelForCausalLM.from_pretrained(model_name)
       
       if tokenizer.pad_token is None:
           tokenizer.pad_token = tokenizer.eos_token
   
   if prompt:
       if result['messages'] and result['messages'][-1]["role"] == "user":
           result['messages'][-1]["content"] = prompt
       else:
           result['messages'].append({"role": "user", "content": prompt})
   
   if format == "json":
       json_instruction = """If you are returning a json object, begin directly with the opening {.
Do not include any additional markdown formatting or leading ```json tags in your response."""
       if result["messages"] and result["messages"][-1]["role"] == "user":
           result["messages"][-1]["content"] += "\n" + json_instruction

   chat_text = tokenizer.apply_chat_template(result["messages"], tokenize=False, add_generation_prompt=True)
   device = next(model.parameters()).device
   inputs = tokenizer(chat_text, return_tensors="pt", padding=True, truncation=True)
   inputs = {k: v.to(device) for k, v in inputs.items()}
   
       
   with torch.no_grad():
       outputs = model.generate(
           **inputs,
           max_new_tokens=256,
           temperature=0.7,
           do_sample=True,
           pad_token_id=tokenizer.eos_token_id,
       )
   
   response_content = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
   result["response"] = response_content
   result["raw_response"] = response_content
   result["messages"].append({"role": "assistant", "content": response_content})

   if auto_process_tool_calls and tools and tool_map:
       detected_tools = []
       for tool in tools:
           tool_name = tool.get("function", {}).get("name", "")
           if tool_name in response_content:
               detected_tools.append({
                   "id": str(uuid.uuid4()),
                   "function": {
                       "name": tool_name,
                       "arguments": "{}"
                   }
               })
       
       if detected_tools:
           result["tool_calls"] = detected_tools
           result = process_tool_calls(result, tool_map, "local", "transformers", result["messages"], tools=tools)
   
   if format == "json":
       try:
           if response_content.startswith("```json"):
               response_content = response_content.replace("```json", "").replace("```", "").strip()
           parsed_response = json.loads(response_content)
           result["response"] = parsed_response
       except json.JSONDecodeError:
           result["error"] = f"Invalid JSON response: {response_content}"
   
   return result

        
def get_ollama_response(
    prompt: str,
    model: str,
    images: List[str] = None,
    tools: list = None,
    tool_choice: Dict = None,
    tool_map: Dict = None,
    think= None ,
    format: Union[str, BaseModel] = None,
    messages: List[Dict[str, str]] = None,
    stream: bool = False,
    attachments: List[str] = None,
    auto_process_tool_calls: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Generates a response using the Ollama API, supporting both streaming and non-streaming.
    """
    _require_ollama()

    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except Exception:
        pass

    api_key = kwargs.pop("api_key", None) or os.environ.get("OLLAMA_API_KEY")
    api_url = kwargs.pop("api_url", None) or os.environ.get("OLLAMA_HOST")
    client_kwargs = {}
    if api_url:
        client_kwargs["host"] = api_url
    if api_key:
        client_kwargs["headers"] = {"Authorization": f"Bearer {api_key}"}

    client = ollama.Client(**client_kwargs)

    options = {}

    num_ctx = int(os.environ.get("OLLAMA_NUM_CTX", 0)) or kwargs.pop("num_ctx", 32768)
    options["num_ctx"] = num_ctx

    image_paths = []
    if images:
        image_paths.extend(images)
    
    if attachments:
        for attachment in attachments:
            if os.path.exists(attachment):
                _, ext = os.path.splitext(attachment)
                ext = ext.lower()
                
                if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                    image_paths.append(attachment)
                elif ext == '.pdf':
                    try:
                        from npcpy.data.load import load_pdf
                        pdf_data = load_pdf(attachment)
                        if pdf_data is not None:
                            if prompt:
                                prompt += f"\n\nContent from PDF: {os.path.basename(attachment)}\n{pdf_data[:5000]}..."
                            else:
                                prompt = f"Content from PDF: {os.path.basename(attachment)}\n{pdf_data[:5000]}..."
                    except Exception:
                        pass
                elif ext == '.csv':
                    try:
                        from npcpy.data.load import load_csv
                        csv_data = load_csv(attachment)
                        if csv_data is not None:
                            csv_sample = csv_data.head(100).to_string()
                            if prompt:
                                prompt += f"\n\nContent from CSV: {os.path.basename(attachment)} (first 100 rows):\n{csv_sample} \n csv description: {csv_data.describe()}"
                            else:
                                prompt = f"Content from CSV: {os.path.basename(attachment)} (first 100 rows):\n{csv_sample} \n csv description: {csv_data.describe()}"
                    except Exception:
                        pass
                else:
                    text_extensions = {'.txt', '.text', '.log', '.md', '.markdown', '.rst', '.json', '.yaml', '.yml', '.toml', '.ini', '.conf', '.cfg', '.xml', '.html', '.htm', '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.h', '.cpp', '.hpp', '.go', '.rs', '.rb', '.php', '.sh', '.bash', '.sql', '.css', '.scss'}
                    filename = os.path.basename(attachment)
                    if ext in text_extensions or ext == '':
                        try:
                            with open(attachment, 'r', encoding='utf-8', errors='replace') as f:
                                text_content = f.read()
                            max_chars = 50000
                            if len(text_content) > max_chars:
                                text_content = text_content[:max_chars] + f"\n\n... [truncated]"
                            if text_content.strip():
                                if prompt:
                                    prompt += f"\n\nContent from {filename}:\n```\n{text_content}\n```"
                                else:
                                    prompt = f"Content from {filename}:\n```\n{text_content}\n```"
                        except Exception:
                            pass

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
            if not messages:
                messages = []
            messages.append({"role": "user", "content": prompt})
    if format == "json" and not stream:
        json_instruction = """If you are a returning a json object, begin directly with the opening {.
            If you are returning a json array, begin directly with the opening [.
            Do not include any additional markdown formatting or leading
            ```json tags in your response. The item keys should be based on the ones provided
            by the user. Do not invent new ones."""

        if messages and messages[-1]["role"] == "user":
            if isinstance(messages[-1]["content"], list):
                messages[-1]["content"].append({
                    "type": "text",
                    "text": json_instruction
                })
            elif isinstance(messages[-1]["content"], str):
                messages[-1]["content"] += "\n" + json_instruction

    if format == "yaml" and not stream:
        yaml_instruction = """Return your response as valid YAML. Do not include ```yaml markdown tags.
            For multi-line strings like code, use the literal block scalar (|) syntax:
            code: |
              your code here
              more lines here
            The keys should be based on the ones requested by the user. Do not invent new ones."""

        if messages and messages[-1]["role"] == "user":
            if isinstance(messages[-1]["content"], list):
                messages[-1]["content"].append({
                    "type": "text",
                    "text": yaml_instruction
                })
            elif isinstance(messages[-1]["content"], str):
                messages[-1]["content"] += "\n" + yaml_instruction

    if image_paths:
        last_user_idx = -1
        for i, msg in enumerate(messages):
            if msg["role"] == "user":
                last_user_idx = i
        if last_user_idx == -1:
            messages.append({"role": "user", "content": ""})
            last_user_idx = len(messages) - 1
        messages[last_user_idx]["images"] = image_paths

    for msg in messages:
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if tc.get("function") and isinstance(tc["function"].get("arguments"), str):
                    try:
                        tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
                    except (json.JSONDecodeError, TypeError):
                        tc["function"]["arguments"] = {}

    api_params = {
        "model": model,
        "messages": messages,
        "stream": stream if not (tools and tool_map and auto_process_tool_calls) else False,
    }

    if tools:
        api_params["tools"] = tools
        if tool_choice:
            options["tool_choice"] = tool_choice

    if think is not None:
        api_params['think'] = think

    if isinstance(format, type) and not stream:
        api_params["format"] = format.model_json_schema()
    elif isinstance(format, str) and format == "json" and not stream:
        api_params["format"] = "json"

    for key, value in kwargs.items():
        if key in [
            "stop",
            "temperature",
            "top_p",
            "top_k",
            "max_tokens",
            "max_completion_tokens",
            "extra_headers",
            "parallel_tool_calls",
            "response_format",
            "user",
        ]:
            options[key] = value

    result = {
        "response": None,
        "messages": messages.copy(),
        "raw_response": None,
        "tool_calls": [], 
        "tool_results": []
    }

    

    
    api_params["messages"] = sanitize_messages(api_params["messages"])
    result["messages"] = api_params["messages"]

    if not auto_process_tool_calls or not (tools and tool_map):
        res = client.chat(**api_params, options=options)
        result["raw_response"] = res

        if stream:
            result["response"] = res
            return result

        if hasattr(res, 'prompt_eval_count') or 'prompt_eval_count' in res:
            input_tokens = getattr(res, 'prompt_eval_count', None) or res.get('prompt_eval_count', 0) or 0
            output_tokens = getattr(res, 'eval_count', None) or res.get('eval_count', 0) or 0
            result["usage"] = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }

        message = res.get("message", {})
        response_content = message.get("content", "")
        result["response"] = response_content

        assistant_msg = {"role": "assistant", "content": response_content}
        if message.get('tool_calls'):
            result["tool_calls"] = message['tool_calls']
            assistant_msg["tool_calls"] = message['tool_calls']
        result["messages"].append(assistant_msg)

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

        if format == "yaml":
            try:
                if isinstance(response_content, str):
                    if response_content.startswith("```yaml"):
                        response_content = (
                            response_content.replace("```yaml", "")
                            .replace("```", "")
                            .strip()
                        )
                    parsed_response = yaml.safe_load(response_content)
                    result["response"] = parsed_response
            except yaml.YAMLError:
                result["error"] = f"Invalid YAML response: {response_content}"

        return result

    logger.debug(f"ollama api_params: {api_params}")
    res = client.chat(**api_params, options=options)
    result["raw_response"] = res
    
    
    
    message = res.get("message", {})
    response_content = message.get("content", "")
    
    
    if message.get('tool_calls'):

        
        result["tool_calls"] = message['tool_calls']
        
        response_for_processing = {
            "response": response_content,
            "raw_response": res,
            "messages": messages,
            "tool_calls": message['tool_calls']
        }
        
        
        processed_result = process_tool_calls(response_for_processing,
                                              tool_map, model,
                                              'ollama',
                                              messages,
                                              stream=False,
                                              tools=tools)
        
        
        clean_messages = []
        tool_results_summary = []

        for msg in processed_result["messages"]:
            role = msg.get('role', '')
            if role == 'assistant' and 'tool_calls' in msg:
                continue
            elif role == 'tool':
                content = msg.get('content', '')
                if len(content) > 2000:
                    content = content[:2000] + "... (truncated)"
                tool_results_summary.append(content)
            else:
                clean_messages.append(msg)

        if tool_results_summary:
            clean_messages.append({
                "role": "assistant",
                "content": "I executed the requested tools. Here are the results:\n\n" + "\n\n".join(tool_results_summary)
            })

        clean_messages.append({
            "role": "user",
            "content": "Based on the tool results above, provide a brief summary of what happened. Do NOT output any code - the tool has already executed. Just describe the results concisely."
        })

        final_api_params = {
            "model": model,
            "messages": clean_messages,
            "stream": stream,
        }

        if stream:
            final_stream = client.chat(**final_api_params, options=options)
            processed_result["response"] = final_stream
        else:
            final_resp = client.chat(**final_api_params, options=options)
            final_message = final_resp.get("message", {})
            final_content = final_message.get("content", "")
            if final_content:
                processed_result["response"] = final_content
                processed_result["messages"].append({"role": "assistant", "content": final_content})
            elif tool_results_summary:
                processed_result["response"] = "\n\n".join(tool_results_summary)
            else:
                processed_result["response"] = "Tool executed successfully."

        return processed_result
    
    
    else:
        result["response"] = response_content
        result["messages"].append({"role": "assistant", "content": response_content})
        
        if stream:
            
            stream_api_params = {
                "model": model,
                "messages": messages,
                "stream": True,
            }
            if tools:
                stream_api_params["tools"] = tools
            
            result["response"] = client.chat(**stream_api_params, options=options)
        else:

            if format == "json":
                try:
                    llm_response = response_content
                    if isinstance(llm_response, str):
                        llm_response = llm_response.strip()
                        
                        if '```json' in llm_response:
                            start = llm_response.find('```json') + 7
                            end = llm_response.rfind('```')
                            if end > start:
                                llm_response = llm_response[start:end].strip()
                        
                        first_brace = llm_response.find('{')
                        first_bracket = llm_response.find('[')
                        
                        if first_brace == -1 and first_bracket == -1:
                            result["response"] = {}
                            result["error"] = "No JSON found in response"
                            return result
                        
                        if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
                            llm_response = llm_response[first_brace:]
                            last_brace = llm_response.rfind('}')
                            if last_brace != -1:
                                llm_response = llm_response[:last_brace+1]
                        else:
                            llm_response = llm_response[first_bracket:]
                            last_bracket = llm_response.rfind(']')
                            if last_bracket != -1:
                                llm_response = llm_response[:last_bracket+1]
                        
                        parsed_json = json.loads(llm_response, strict=False)
                        
                        if "json" in parsed_json:
                            result["response"] = parsed_json["json"]
                        else:
                            result["response"] = parsed_json
                        
                except (json.JSONDecodeError, TypeError) as e:
                    logger.debug(f"JSON parsing error: {str(e)}, raw response: {llm_response[:500]}")
                    result["response"] = {}
                    result["error"] = "Invalid JSON response"

            if format == "yaml":
                try:
                    if isinstance(llm_response, str):
                        llm_response = llm_response.strip()

                        if '```yaml' in llm_response:
                            start = llm_response.find('```yaml') + 7
                            end = llm_response.rfind('```')
                            if end > start:
                                llm_response = llm_response[start:end].strip()

                        parsed_yaml = yaml.safe_load(llm_response)
                        result["response"] = parsed_yaml

                except (yaml.YAMLError, TypeError) as e:
                    logger.debug(f"YAML parsing error: {str(e)}, raw response: {llm_response[:500]}")
                    result["response"] = {}
                    result["error"] = "Invalid YAML response"

        return result

import time

def get_lora_response(
    prompt: str = None,
    model: str = None,
    tools: list = None,
    tool_map: Dict = None,
    format: str = None,
    messages: List[Dict[str, str]] = None,
    stream: bool = False,
    auto_process_tool_calls: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Generate response using a LoRA adapter on top of a base model.
    The adapter path should contain adapter_config.json with base_model_name_or_path.
    """
    print(f"🎯 get_lora_response called with model={model}, prompt={prompt[:50] if prompt else 'None'}...")

    result = {
        "response": None,
        "messages": messages.copy() if messages else [],
        "raw_response": None,
        "tool_calls": [],
        "tool_results": []
    }

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        print("🎯 Successfully imported torch, transformers, peft")
    except ImportError as e:
        print(f"🎯 Import error: {e}")
        return {
            "response": "",
            "messages": messages or [],
            "error": f"Missing dependencies for LoRA. Install with: pip install transformers peft torch. Error: {e}"
        }

    adapter_path = os.path.expanduser(model)
    adapter_config_path = os.path.join(adapter_path, 'adapter_config.json')

    if not os.path.exists(adapter_config_path):
        return {
            "response": "",
            "messages": messages or [],
            "error": f"No adapter_config.json found at {adapter_path}"
        }

    try:
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
        base_model_id = adapter_config.get('base_model_name_or_path')
        if not base_model_id:
            return {
                "response": "",
                "messages": messages or [],
                "error": "adapter_config.json missing base_model_name_or_path"
            }
    except Exception as e:
        return {
            "response": "",
            "messages": messages or [],
            "error": f"Failed to read adapter config: {e}"
        }

    if prompt:
        if result['messages'] and result['messages'][-1]["role"] == "user":
            result['messages'][-1]["content"] = prompt
        else:
            result['messages'].append({"role": "user", "content": prompt})

    if format == "json":
        json_instruction = """If you are returning a json object, begin directly with the opening {.
Do not include any additional markdown formatting or leading ```json tags in your response."""
        if result["messages"] and result["messages"][-1]["role"] == "user":
            result["messages"][-1]["content"] += "\n" + json_instruction

    try:
        logger.info(f"Loading base model: {base_model_id}")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True
        )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"Loading LoRA adapter: {adapter_path}")
        model_with_adapter = PeftModel.from_pretrained(base_model, adapter_path)

        chat_text = tokenizer.apply_chat_template(
            result["messages"],
            tokenize=False,
            add_generation_prompt=True
        )
        device = next(model_with_adapter.parameters()).device
        inputs = tokenizer(chat_text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        max_new_tokens = kwargs.get("max_tokens", 512)
        temperature = kwargs.get("temperature", 0.7)

        with torch.no_grad():
            outputs = model_with_adapter.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        response_content = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        ).strip()

        result["response"] = response_content
        result["raw_response"] = response_content
        result["messages"].append({"role": "assistant", "content": response_content})

        if format == "json":
            try:
                if response_content.startswith("```json"):
                    response_content = response_content.replace("```json", "").replace("```", "").strip()
                parsed_response = json.loads(response_content)
                result["response"] = parsed_response
            except json.JSONDecodeError:
                result["error"] = f"Invalid JSON response: {response_content}"

    except Exception as e:
        logger.error(f"LoRA inference error: {e}")
        result["error"] = f"LoRA inference error: {str(e)}"
        result["response"] = ""

    return result

def _parse_mlx_tool_calls(text, tools=None):
    """Parse structured tool-call blocks a local MLX model (Qwen3 / Qwen2.5
    chat templates) emits, into the call-dict shape process_tool_calls expects.

    Tokens are built from chr() so this source never hardcodes the delimiters.
    Recognizes tool_call and function=NAME blocks plus fenced json; falls back
    to a verbatim tool-name scan (empty args) so a small model that only
    mentions a tool still yields a tool_call for the agent loop to execute.
    """
    import re
    LT, GT = chr(60), chr(62)
    tc_open = LT + "tool_call" + GT
    tc_close = LT + "/tool_call" + GT
    fn_close = LT + "/function" + GT

    def _make(name, args):
        if not name:
            return None
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw_arguments": args}
        if not isinstance(args, dict):
            args = {"value": args}
        func = {}
        func["name"] = name
        func["arguments"] = json.dumps(args)
        call = {"id": str(uuid.uuid4()), "type": "function", "function": func}
        return call

    calls = []
    pat = re.escape(tc_open) + r"\s*(\{.*?\})\s*" + re.escape(tc_close)
    for m in re.finditer(pat, text, re.DOTALL):
        body = m.group(1).strip()
        body = re.sub(r"^```(?:json|tool_call)?\s*", "", body).strip()
        body = re.sub(r"\s*```$", "", body).strip()
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            continue
        calls.append(_make(obj.get("name"), obj.get("arguments", {})))

    if not calls:
        pat2 = re.escape(LT) + r"function=([^>" + re.escape(GT) + r"]+)>\s*(.*?)\s*" + re.escape(fn_close)
        param_open = LT + "parameter="
        param_close = LT + "/parameter" + GT
        pat_param = re.escape(param_open) + r"([^>" + re.escape(GT) + r"]+)>(.*?)" + re.escape(param_close)
        for m in re.finditer(pat2, text, re.DOTALL):
            body = m.group(2).strip()
            args = {}
            params = list(re.finditer(pat_param, body, re.DOTALL))
            if params:
                for pm in params:
                    key = pm.group(1).strip()
                    val = pm.group(2).strip()
                    try:
                        args[key] = json.loads(val)
                    except json.JSONDecodeError:
                        args[key] = val
            else:
                try:
                    args = json.loads(body)
                except json.JSONDecodeError:
                    args = {}
            calls.append(_make(m.group(1).strip(), args))

    if not calls:
        for m in re.finditer(r"```(?:tool_call|json)?\s*(\{.*?\})\s*```", text, re.DOTALL):
            try:
                obj = json.loads(m.group(1))
            except json.JSONDecodeError:
                continue
            if "name" in obj:
                calls.append(_make(obj["name"], obj.get("arguments", {})))

    if not calls and tools:
        for tool in tools:
            tool_name = tool.get("function", {}).get("name", "")
            if tool_name and tool_name in text:
                calls.append(_make(tool_name, {}))

    return [c for c in calls if c]


def get_mlx_response(
    prompt: str = None,
    model: str = None,
    tools: list = None,
    tool_map: Dict = None,
    format: Union[str, BaseModel] = None,
    messages: List[Dict[str, str]] = None,
    stream: bool = False,
    auto_process_tool_calls: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Generate a response with a local MLX model, optionally with a trained
    LoRA adapter. npcy owns the mlx_lm import here — callers just pass
    provider="mlx" and model=<mlx-community model name | adapter dir>.

    An adapter dir must contain adapter_config.json with a "model" field naming
    the mlx-community base and "fine_tune_type": "lora" (the format written by
    npcpy.ft's MLX trainers).
    """
    result = {
        "response": None,
        "messages": messages.copy() if messages else [],
        "raw_response": None,
        "tool_calls": [],
        "tool_results": [],
    }

    try:
        from mlx_lm import load as mlx_load, generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler
    except ImportError as e:
        return {
            "response": "",
            "messages": messages or [],
            "error": f"MLX backend not installed: pip install mlx mlx-lm. Error: {e}",
        }

    model_arg = os.path.expanduser(str(model)) if model else ""
    adapter_config_path = os.path.join(model_arg, "adapter_config.json")
    base_model_id = model
    adapter_path = None
    if os.path.isdir(model_arg) and os.path.exists(adapter_config_path):
        try:
            with open(adapter_config_path, "r") as f:
                adapter_config = json.load(f)
            base_model_id = adapter_config.get("model") or adapter_config.get("base_model_name_or_path")
            if adapter_config.get("fine_tune_type") == "lora" or "lora_parameters" in adapter_config:
                adapter_path = model_arg
        except Exception as e:
            return {"response": "", "messages": messages or [],
                    "error": f"Failed to read adapter config: {e}"}
        if not base_model_id:
            return {"response": "", "messages": messages or [],
                    "error": "adapter_config.json missing base model ('model'/'base_model_name_or_path')"}

    if prompt:
        if result["messages"] and result["messages"][-1].get("role") == "user":
            result["messages"][-1]["content"] = prompt
        else:
            result["messages"].append({"role": "user", "content": prompt})

    if format == "json":
        json_instruction = """If you are returning a json object, begin directly with the opening {.
Do not include any additional markdown formatting or leading ```json tags in your response."""
        if result["messages"] and result["messages"][-1]["role"] == "user":
            result["messages"][-1]["content"] += "\n" + json_instruction

    try:
        logger.info(f"Loading MLX model: {base_model_id}"
                    + (f" + adapter {adapter_path}" if adapter_path else ""))
        if adapter_path:
            mlx_model, tokenizer = mlx_load(base_model_id, adapter_path=adapter_path)
        else:
            mlx_model, tokenizer = mlx_load(base_model_id)

        chat_kwargs = {"tokenize": False, "add_generation_prompt": True}
        if tools:
            chat_kwargs["tools"] = tools
        try:
            chat_text = tokenizer.apply_chat_template(result["messages"], **chat_kwargs)
        except (TypeError, ValueError):
            chat_text = tokenizer.apply_chat_template(
                result["messages"], tokenize=False, add_generation_prompt=True
            )

        max_tokens = kwargs.get("max_tokens", kwargs.get("max_new_tokens", 512))
        temperature = kwargs.get("temperature", 0.7)
        top_p = kwargs.get("top_p", 0.0)
        sampler = make_sampler(temp=temperature, top_p=top_p)
        out = mlx_generate(
            mlx_model, tokenizer, prompt=chat_text,
            max_tokens=max_tokens, sampler=sampler, verbose=False,
        )
        text = out if isinstance(out, str) else getattr(out, "text", str(out))
        if text.startswith(chat_text):
            text = text[len(chat_text):]
        response_content = text.strip()

        tool_calls = _parse_mlx_tool_calls(response_content, tools) if tools else []

        if tool_calls:
            result["tool_calls"] = tool_calls
            result["response"] = response_content
            result["raw_response"] = response_content
            result["messages"].append({"role": "assistant", "content": response_content})
            if auto_process_tool_calls and tool_map:
                result = process_tool_calls(
                    result, tool_map, model, "mlx",
                    result["messages"], tools=tools,
                )
        else:
            result["response"] = response_content
            result["raw_response"] = response_content
            result["messages"].append({"role": "assistant", "content": response_content})
            if format == "json":
                cleaned = response_content
                if cleaned.startswith("```json"):
                    cleaned = cleaned.replace("```json", "").replace("```", "").strip()
                try:
                    result["response"] = json.loads(cleaned)
                except json.JSONDecodeError:
                    result["error"] = f"Invalid JSON response: {response_content}"

    except Exception as e:
        logger.error(f"MLX inference error: {e}")
        result["error"] = f"MLX inference error: {str(e)}"
        result["response"] = ""

    return result


def get_llamacpp_response(
    prompt: str = None,
    model: str = None,
    images: List[str] = None,
    tools: list = None,
    tool_choice: Dict = None,
    tool_map: Dict = None,
    think=None,
    format: Union[str, BaseModel] = None,
    messages: List[Dict[str, str]] = None,
    stream: bool = False,
    attachments: List[str] = None,
    auto_process_tool_calls: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Generate response using llama-cpp-python for local GGUF/GGML files.
    """
    try:
        from llama_cpp import Llama
    except ImportError:
        return {
            "response": "",
            "messages": messages or [],
            "error": "llama-cpp-python not installed. Install with: pip install llama-cpp-python"
        }

    result = {
        "response": None,
        "messages": messages.copy() if messages else [],
        "raw_response": None,
        "tool_calls": [],
        "tool_results": []
    }

    if prompt:
        if messages and messages[-1]["role"] == "user":
            messages[-1]["content"] = prompt
        else:
            if not messages:
                messages = []
            messages.append({"role": "user", "content": prompt})

    try:
        n_ctx = kwargs.get("n_ctx", 32768)
        n_gpu_layers = kwargs.get("n_gpu_layers", -1)

        llm = Llama(
            model_path=model,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            verbose=False
        )

        params = {
            "messages": messages,
            "stream": stream,
        }
        if kwargs.get("temperature") is not None:
            params["temperature"] = kwargs["temperature"]
        if kwargs.get("max_tokens"):
            params["max_tokens"] = kwargs["max_tokens"]
        if kwargs.get("top_p") is not None:
            params["top_p"] = kwargs["top_p"]
        if kwargs.get("top_k") is not None:
            params["top_k"] = kwargs["top_k"]
        if kwargs.get("stop"):
            params["stop"] = kwargs["stop"]

        if stream:
            response = llm.create_chat_completion(**params)

            def generate():
                for chunk in response:
                    yield chunk

            result["response"] = generate()
        else:
            response = llm.create_chat_completion(**params)
            result["raw_response"] = response

            if response.get("choices"):
                content = response["choices"][0].get("message", {}).get("content", "")
                result["response"] = content
                result["messages"].append({"role": "assistant", "content": content})

            if response.get("usage"):
                result["usage"] = {
                    "input_tokens": response["usage"].get("prompt_tokens", 0),
                    "output_tokens": response["usage"].get("completion_tokens", 0),
                }

    except Exception as e:
        result["error"] = f"llama.cpp error: {str(e)}"
        result["response"] = ""

    return result

_AIRLLM_MODEL_CACHE = {}
_AIRLLM_MLX_PATCHED = False

def _patch_airllm_mlx_bias():
    """
    Monkey-patch airllm's MLX Attention/FeedForward to use bias=True.
    AirLLM hardcodes bias=False which fails for non-Llama architectures (e.g. Qwen2).
    Using bias=True is safe: MLX nn.Linear(bias=True) accepts weight-only updates,
    so Llama models (no bias in weights) still work correctly.
    """
    global _AIRLLM_MLX_PATCHED
    if _AIRLLM_MLX_PATCHED:
        return
    try:
        import airllm.airllm_llama_mlx as mlx_mod
        import mlx.core as mx
        from mlx import nn

        class PatchedAttention(nn.Module):
            def __init__(self, args):
                super().__init__()
                self.args = args
                self.n_heads = args.n_heads
                self.n_kv_heads = args.n_kv_heads
                self.repeats = self.n_heads // self.n_kv_heads
                self.scale = args.head_dim ** -0.5
                self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=True)
                self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=True)
                self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=True)
                self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=True)
                self.rope = nn.RoPE(
                    args.head_dim, traditional=args.rope_traditional, base=args.rope_theta
                )

            def __call__(self, x, mask=None, cache=None):
                B, L, D = x.shape
                queries, keys, values = self.wq(x), self.wk(x), self.wv(x)
                queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
                keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
                values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

                def repeat(a):
                    a = mx.concatenate([mx.expand_dims(a, 2)] * self.repeats, axis=2)
                    return a.reshape([B, self.n_heads, L, -1])
                keys, values = map(repeat, (keys, values))

                if cache is not None:
                    key_cache, value_cache = cache
                    queries = self.rope(queries, offset=key_cache.shape[2])
                    keys = self.rope(keys, offset=key_cache.shape[2])
                    keys = mx.concatenate([key_cache, keys], axis=2)
                    values = mx.concatenate([value_cache, values], axis=2)
                else:
                    queries = self.rope(queries)
                    keys = self.rope(keys)

                scores = (queries * self.scale) @ keys.transpose(0, 1, 3, 2)
                if mask is not None:
                    scores += mask
                weights = mx.softmax(scores.astype(mx.float32), axis=-1).astype(scores.dtype)
                output = (weights @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)
                return self.wo(output), (keys, values)

        class PatchedFeedForward(nn.Module):
            def __init__(self, args):
                super().__init__()
                self.w1 = nn.Linear(args.dim, args.hidden_dim, bias=True)
                self.w2 = nn.Linear(args.hidden_dim, args.dim, bias=True)
                self.w3 = nn.Linear(args.dim, args.hidden_dim, bias=True)

            def __call__(self, x):
                return self.w2(nn.silu(self.w1(x)) * self.w3(x))

        mlx_mod.Attention = PatchedAttention
        mlx_mod.FeedForward = PatchedFeedForward
        _AIRLLM_MLX_PATCHED = True
        logger.debug("Patched airllm MLX classes for bias support")
    except Exception as e:
        logger.warning(f"Failed to patch airllm MLX bias support: {e}")

def get_airllm_response(
    prompt: str = None,
    model: str = None,
    tools: list = None,
    tool_map: Dict = None,
    format: str = None,
    messages: List[Dict[str, str]] = None,
    auto_process_tool_calls: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Generate response using AirLLM for 70B+ model inference.
    Supports macOS (MLX backend) and Linux (CUDA backend with 4-bit compression).
    """
    import platform
    is_macos = platform.system() == "Darwin"

    result = {
        "response": None,
        "messages": messages.copy() if messages else [],
        "raw_response": None,
        "tool_calls": [],
        "tool_results": []
    }

    try:
        from airllm import AutoModel
    except ImportError:
        result["response"] = ""
        result["error"] = "airllm not installed. Install with: pip install airllm"
        return result

    if is_macos:
        _patch_airllm_mlx_bias()

    if prompt:
        if result['messages'] and result['messages'][-1]["role"] == "user":
            result['messages'][-1]["content"] = prompt
        else:
            result['messages'].append({"role": "user", "content": prompt})

    if format == "json":
        json_instruction = """If you are returning a json object, begin directly with the opening {.
Do not include any additional markdown formatting or leading ```json tags in your response."""
        if result["messages"] and result["messages"][-1]["role"] == "user":
            result["messages"][-1]["content"] += "\n" + json_instruction

    model_name = model or "meta-llama/Meta-Llama-3.1-70B-Instruct"
    default_compression = None if is_macos else "4bit"
    compression = kwargs.get("compression", default_compression)
    max_tokens = kwargs.get("max_tokens", 256)
    temperature = kwargs.get("temperature", 0.7)

    hf_token = kwargs.get("hf_token")
    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not hf_token:
        try:
            from huggingface_hub import HfFolder
            hf_token = HfFolder.get_token()
        except Exception:
            pass

    cache_key = f"{model_name}:{compression}"
    if cache_key not in _AIRLLM_MODEL_CACHE:
        load_kwargs = {"pretrained_model_name_or_path": model_name}
        if compression:
            load_kwargs["compression"] = compression
        if hf_token:
            load_kwargs["hf_token"] = hf_token
        for k in ["delete_original", "max_seq_len", "prefetching"]:
            if k in kwargs:
                load_kwargs[k] = kwargs[k]
        _AIRLLM_MODEL_CACHE[cache_key] = AutoModel.from_pretrained(**load_kwargs)

    air_model = _AIRLLM_MODEL_CACHE[cache_key]

    try:
        chat_text = air_model.tokenizer.apply_chat_template(
            result["messages"], tokenize=False, add_generation_prompt=True
        )
    except Exception:
        chat_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in result["messages"]
        )
        chat_text += "\nassistant:"

    try:
        if is_macos:
            import mlx.core as mx
            tokens = air_model.tokenizer(
                chat_text, return_tensors="np", truncation=True, max_length=2048
            )
            output = air_model.generate(
                mx.array(tokens['input_ids']),
                max_new_tokens=max_tokens,
            )
            response_content = output if isinstance(output, str) else str(output)
        else:
            tokens = air_model.tokenizer(
                chat_text, return_tensors="pt", truncation=True, max_length=2048
            )
            gen_out = air_model.generate(
                tokens['input_ids'].cuda(),
                max_new_tokens=max_tokens,
            )
            output_ids = gen_out.sequences[0] if hasattr(gen_out, 'sequences') else gen_out[0]
            response_content = air_model.tokenizer.decode(output_ids, skip_special_tokens=True)
            input_text = air_model.tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)
            if response_content.startswith(input_text):
                response_content = response_content[len(input_text):]

        response_content = response_content.strip()
        for stop_tok in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>", "</s>"]:
            if stop_tok in response_content:
                response_content = response_content[:response_content.index(stop_tok)].strip()
    except Exception as e:
        logger.error(f"AirLLM inference error: {e}")
        result["error"] = f"AirLLM inference error: {str(e)}"
        result["response"] = ""
        return result

    result["response"] = response_content
    result["raw_response"] = response_content
    result["messages"].append({"role": "assistant", "content": response_content})

    if format == "json":
        try:
            if response_content.startswith("```json"):
                response_content = response_content.replace("```json", "").replace("```", "").strip()
            parsed_response = json.loads(response_content)
            result["response"] = parsed_response
        except json.JSONDecodeError:
            result["error"] = f"Invalid JSON response: {response_content}"

    return result


# ── QLLM-PAM local provider (self-contained) ──────────────────────────────────

_QLLM_IM_START = "<|im_start|>"
_QLLM_IM_END = "<|im_end|>"
_QLLM_THINK_START = " " + "<think>"
_QLLM_THINK_END = " " + "</think>"


def _download_qllm_from_hf(repo_id: str) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError(
            "huggingface_hub is required to download QLLM-PAM models by repo id. "
            "Install it with: pip install huggingface_hub"
        )
    return snapshot_download(repo_id, repo_type="model")


def _resolve_qllm_checkpoint(model: str) -> tuple:
    # Hugging Face repo id (e.g. owner/name)
    if "/" in model and not os.path.exists(model):
        model_dir = _download_qllm_from_hf(model)
        pt_files = sorted(f for f in os.listdir(model_dir) if f.endswith(".pt"))
        if not pt_files:
            raise ValueError(f"No .pt checkpoint found in HF repo: {model}")
        chat_files = [f for f in pt_files if "chat" in f.lower()]
        checkpoint_file = chat_files[0] if chat_files else pt_files[0]
        return os.path.join(model_dir, checkpoint_file), model_dir

    if os.path.isfile(model):
        if not model.endswith(".pt"):
            raise ValueError(f"QLLM model file must be a .pt checkpoint: {model}")
        checkpoint_path = os.path.abspath(model)
        return checkpoint_path, os.path.dirname(checkpoint_path)

    if os.path.isdir(model):
        model_dir = os.path.abspath(model)
        pt_files = sorted(f for f in os.listdir(model_dir) if f.endswith(".pt"))
        if not pt_files:
            raise ValueError(f"No .pt checkpoint found in QLLM model directory: {model}")
        chat_files = [f for f in pt_files if "chat" in f.lower()]
        checkpoint_file = chat_files[0] if chat_files else pt_files[0]
        return os.path.join(model_dir, checkpoint_file), model_dir

    raise ValueError(
        f"QLLM model must be a HF repo id, a directory containing a .pt checkpoint, or a .pt file: {model}"
    )


def _get_qllm_device(device_pref: str = None):
    import torch
    if device_pref:
        return torch.device(device_pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Self-contained QLLM-PAM V11 model ──────────────────────────────────────────

class _QllmConfig:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _qllm_cmul(a, b):
    return torch.stack([
        a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1],
        a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0],
    ], dim=-1)


def _qllm_cconj(x):
    return torch.stack([x[..., 0], -x[..., 1]], dim=-1)


def _qllm_cabs(x):
    return torch.sqrt(x[..., 0].square() + x[..., 1].square() + 1e-8)


def _qllm_cnormalize(x):
    return x / _qllm_cabs(x).unsqueeze(-1)


def _qllm_to_real_concat(x):
    return torch.cat([x[..., 0], x[..., 1]], dim=-1)


def _qllm_complex_norm(z, scale, eps=1e-6):
    mag = _qllm_cabs(z)
    rms = torch.sqrt(mag.square().mean(dim=-1, keepdim=True) + eps)
    scaled = (mag / rms) * scale
    phase = z / (mag.unsqueeze(-1) + 1e-8)
    return phase * scaled.unsqueeze(-1)


def _qllm_mod_relu(z, bias):
    mag = _qllm_cabs(z)
    activated = F.relu(mag + bias)
    phase = z / (mag.unsqueeze(-1) + 1e-8)
    return phase * activated.unsqueeze(-1)


def _qllm_mod_swish(z, bias, beta):
    mag = _qllm_cabs(z)
    activated = mag * torch.sigmoid(beta * mag + bias)
    phase = z / (mag.unsqueeze(-1) + 1e-8)
    return phase * activated.unsqueeze(-1)


def _qllm_cgu_gate(gate, up):
    gmag = _qllm_cabs(gate)
    gate_mag = torch.sigmoid(gmag)
    phase = gate / (gmag.unsqueeze(-1) + 1e-8)
    pr, pi = phase[..., 0], phase[..., 1]
    ur, ui = up[..., 0], up[..., 1]
    out_r = (pr * ur - pi * ui) * gate_mag
    out_i = (pr * ui + pi * ur) * gate_mag
    return torch.stack([out_r, out_i], dim=-1)


def _qllm_fused_decay_matrix(gamma, T):
    log_gamma = torch.log(gamma + 1e-6)
    C = torch.cumsum(-log_gamma, dim=-1)
    log_D = (C.unsqueeze(-1) - C.unsqueeze(-2)).transpose(-1, -2)
    causal = torch.tril(torch.ones(T, T, device=gamma.device, dtype=gamma.dtype))
    log_D = log_D * causal + (1 - causal) * (-1e4)
    return torch.exp(log_D.clamp(max=0.0))


if torch is not None:
    class _QllmComplexLinear(nn.Module):
        def __init__(self, in_dim, out_dim, bias=True):
            super().__init__()
            scale = (2 / (in_dim + out_dim)) ** 0.5
            self.weight_real = nn.Parameter(torch.empty(out_dim, in_dim))
            self.weight_imag = nn.Parameter(torch.empty(out_dim, in_dim))
            nn.init.orthogonal_(self.weight_real, gain=scale)
            nn.init.orthogonal_(self.weight_imag, gain=scale)
            if bias:
                self.bias_real = nn.Parameter(torch.zeros(out_dim))
                self.bias_imag = nn.Parameter(torch.zeros(out_dim))
            else:
                self.bias_real = self.bias_imag = None
    
        def forward(self, x):
            xr, xi = x[..., 0], x[..., 1]
            yr = F.linear(xr, self.weight_real) - F.linear(xi, self.weight_imag)
            yi = F.linear(xr, self.weight_imag) + F.linear(xi, self.weight_real)
            if self.bias_real is not None:
                yr = yr + self.bias_real
                yi = yi + self.bias_imag
            return torch.stack([yr, yi], dim=-1)
    
    
    class _QllmComplexNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.scale = nn.Parameter(torch.ones(dim))
            self.eps = eps
    
        def forward(self, z):
            return _qllm_complex_norm(z, self.scale, self.eps)
    
    
    class _QllmModReLU(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.bias = nn.Parameter(torch.full((dim,), -0.1))
    
        def forward(self, z):
            return _qllm_mod_relu(z, self.bias)
    
    
    class _QllmModSwish(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(dim))
            self.beta = nn.Parameter(torch.ones(dim))
    
        def forward(self, z):
            return _qllm_mod_swish(z, self.bias, self.beta)
    
    
    class _QllmPhaseModulatedActivation(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(dim))
            self.beta = nn.Parameter(torch.ones(dim))
            self.phase_alpha = nn.Parameter(torch.zeros(dim))
            self.phase_beta = nn.Parameter(torch.zeros(dim))
    
        def forward(self, z):
            mag = _qllm_cabs(z)
            activated = mag * torch.sigmoid(self.beta * mag + self.bias)
            phase = z / (mag.unsqueeze(-1) + 1e-8)
            theta = self.phase_alpha * mag + self.phase_beta
            rot = torch.stack([theta.cos(), theta.sin()], dim=-1)
            phase = _qllm_cmul(phase, rot)
            return phase * activated.unsqueeze(-1)
    
    
    def _qllm_build_activation(name, dim):
        if name == 'swish':
            return _QllmModSwish(dim)
        if name == 'phase_mod':
            return _QllmPhaseModulatedActivation(dim)
        return _QllmModReLU(dim)
    
    
    class _QllmComplexGatedUnit(nn.Module):
        def __init__(self, dim, expand=3, activation='modrelu'):
            super().__init__()
            hidden = dim * expand
            self.gate_proj = _QllmComplexLinear(dim, hidden, bias=False)
            self.up_proj = _QllmComplexLinear(dim, hidden, bias=False)
            self.down_proj = _QllmComplexLinear(hidden, dim, bias=False)
            self.act = _qllm_build_activation(activation, hidden)
    
        def forward(self, z):
            gate = self.gate_proj(z)
            up = self.act(self.up_proj(z))
            gated = _qllm_cgu_gate(gate, up)
            return self.down_proj(gated)
    
    
    class _QllmComplexEmbed(nn.Module):
        def __init__(self, vocab_size, dim):
            super().__init__()
            self.dim = dim
            self.embed_real = nn.Embedding(vocab_size, dim)
            self.embed_imag = nn.Embedding(vocab_size, dim)
            nn.init.normal_(self.embed_real.weight, std=0.02)
            nn.init.normal_(self.embed_imag.weight, std=0.02)
    
        def forward(self, ids):
            return torch.stack([self.embed_real(ids), self.embed_imag(ids)], dim=-1)
    
    
    class _QllmComplexPosEmbed(nn.Module):
        def __init__(self, max_seq_len, dim):
            super().__init__()
            self.max_seq_len = max_seq_len
            self.pos_embed = nn.Embedding(max_seq_len, dim)
            nn.init.normal_(self.pos_embed.weight, std=0.02)
    
        def forward(self, z, step_offset=0):
            T = z.shape[1]
            end = step_offset + T
            if end > self.max_seq_len:
                raise ValueError(f'Position range exceeds max_seq_len {self.max_seq_len}')
            pos = torch.arange(step_offset, end, device=z.device)
            p = self.pos_embed(pos)
            return z + p.unsqueeze(0).unsqueeze(-1)
    
    
    def _qllm_build_rope_cache(max_len, head_dim):
        freqs = 1.0 / (10000.0 ** (torch.arange(head_dim).float() / head_dim))
        positions = torch.arange(max_len).float()
        angles = positions.unsqueeze(1) * freqs.unsqueeze(0)
        return torch.stack([angles.cos(), angles.sin()], dim=-1)
    
    
    class _QllmPAMLayer(nn.Module):
        def __init__(self, cfg, layer_idx=0):
            super().__init__()
            self.num_heads = cfg.n_heads
            self.head_dim = cfg.head_dim
            inner = cfg.n_heads * cfg.head_dim
            self.inner_dim = inner
            self.dim = cfg.dim
            self.fused_qkv = cfg.fused_qkv
            self.use_rope = cfg.use_rope
            self.use_gsp = cfg.use_gsp
            self.qk_norm = cfg.qk_norm
            self.decay_mode = cfg.decay_mode
            self.write_mode = cfg.write_mode
            self.n_states = cfg.n_states
            self.delta_chunk = cfg.delta_chunk
            self.gate_content_aware = getattr(cfg, 'gate_content_aware', False)
            self.protect_gate_bias = getattr(cfg, 'protect_gate_bias', -3.0)
    
            if cfg.fused_qkv:
                self.qkv_proj = _QllmComplexLinear(cfg.dim, 3 * inner, bias=False)
            else:
                self.q_proj = _QllmComplexLinear(cfg.dim, inner, bias=False)
                self.k_proj = _QllmComplexLinear(cfg.dim, inner, bias=False)
                self.v_proj = _QllmComplexLinear(cfg.dim, inner, bias=False)
            self.o_proj = _QllmComplexLinear(inner, cfg.dim, bias=False)
    
            decay_out = cfg.n_heads * (cfg.head_dim if cfg.decay_mode == 'per_channel' else 1)
            self.dt_proj = nn.Linear(cfg.dim * 2, decay_out)
            if cfg.decay_mode == 'per_channel':
                self.dt_bias = nn.Parameter(torch.zeros(cfg.n_heads, cfg.head_dim) + cfg.base_dt_bias)
            else:
                self.dt_bias = nn.Parameter(torch.zeros(cfg.n_heads) + cfg.base_dt_bias)
    
            if cfg.use_gsp:
                gate_in = cfg.dim * 2 if self.gate_content_aware else cfg.dim
                self.protect_gate = nn.Linear(gate_in, cfg.n_heads)
                nn.init.constant_(self.protect_gate.bias, self.protect_gate_bias)
    
            if cfg.n_states > 1:
                offs = torch.linspace(-cfg.state_dt_spread, cfg.state_dt_spread, cfg.n_states)
                self.state_dt_offset = nn.Parameter(offs.clone())
                self.phase_proj = nn.Linear(cfg.dim, cfg.n_heads * cfg.n_states)
                nn.init.zeros_(self.phase_proj.weight)
                nn.init.zeros_(self.phase_proj.bias)
    
            if cfg.use_rope:
                self.register_buffer(
                    'rope_cache',
                    _qllm_build_rope_cache(cfg.max_seq_len, cfg.head_dim),
                    persistent=False,
                )
    
            self.dropout = nn.Dropout(cfg.dropout)
            self.chunk_size = cfg.chunk_size
            _causal_size = cfg.chunk_size if cfg.chunk_size > 0 else cfg.max_seq_len
            self.register_buffer(
                '_causal',
                torch.tril(torch.ones(_causal_size, _causal_size)),
                persistent=False,
            )
    
        def _project(self, x, step_offset):
            B, T, _, _ = x.shape
            H, d = self.num_heads, self.head_dim
            if self.fused_qkv:
                qkv = self.qkv_proj(x).view(B, T, 3, H, d, 2)
                q = qkv[:, :, 0].transpose(1, 2).contiguous()
                k = qkv[:, :, 1].transpose(1, 2).contiguous()
                v = qkv[:, :, 2].transpose(1, 2).contiguous()
            else:
                q = self.q_proj(x).view(B, T, H, d, 2).transpose(1, 2).contiguous()
                k = self.k_proj(x).view(B, T, H, d, 2).transpose(1, 2).contiguous()
                v = self.v_proj(x).view(B, T, H, d, 2).transpose(1, 2).contiguous()
    
            if self.use_rope:
                end = step_offset + T
                if end > self.rope_cache.shape[0]:
                    self.register_buffer(
                        'rope_cache',
                        _qllm_build_rope_cache(end * 2, d).to(x.device),
                        persistent=False,
                    )
                pos = self.rope_cache[step_offset:end].to(dtype=x.dtype)
                q = _qllm_cmul(q, pos)
                k = _qllm_cmul(k, pos)
    
            if self.qk_norm:
                q = _qllm_cnormalize(q)
                k = _qllm_cnormalize(k)
            return q, k, v
    
        def _gamma_and_vprime(self, x, v, state_offset=0.0):
            B, T = x.shape[0], x.shape[1]
            H, d = self.num_heads, self.head_dim
            x_flat = _qllm_to_real_concat(x)
            if self.decay_mode == 'per_channel':
                dt = self.dt_proj(x_flat).view(B, T, H, d)
                dt = F.softplus(dt + self.dt_bias + state_offset)
                dt = dt.permute(0, 2, 1, 3).contiguous()
            else:
                dt = self.dt_proj(x_flat)
                dt = F.softplus(dt + self.dt_bias + state_offset)
                dt = dt.transpose(1, 2).contiguous()
    
            if self.use_gsp:
                gate_in = _qllm_to_real_concat(x) if self.gate_content_aware else _qllm_cabs(x)
                p = torch.sigmoid(self.protect_gate(gate_in)).transpose(1, 2)
                if self.decay_mode == 'per_channel':
                    p_e = p.unsqueeze(-1)
                    gamma = torch.exp(-dt) * (1 - p_e) + p_e
                else:
                    gamma = torch.exp(-dt) * (1 - p) + p
                v_prime = v * (1 - p).unsqueeze(-1).unsqueeze(-1)
            else:
                gamma = torch.exp(-dt)
                v_prime = v
            return gamma, v_prime
    
        @staticmethod
        def _dual_form_block(q_s, k, v_prime, gamma, causal_mask):
            B, H, T = gamma.shape
            gamma_flat = gamma.reshape(B * H, T)
            D = _qllm_fused_decay_matrix(gamma_flat, T).reshape(B, H, T, T)
            qr, qi = q_s[..., 0], q_s[..., 1]
            kr, ki = k[..., 0], k[..., 1]
            wr = qr @ kr.transpose(-1, -2) + qi @ ki.transpose(-1, -2)
            wi = qi @ kr.transpose(-1, -2) - qr @ ki.transpose(-1, -2)
            ar, ai = wr * D, wi * D
            vpr, vpi = v_prime[..., 0], v_prime[..., 1]
            yr = ar @ vpr - ai @ vpi
            yi = ar @ vpi + ai @ vpr
            y = torch.stack([yr, yi], dim=-1)
            D_last = D[:, :, -1, :]
            wv_r = vpr * D_last.unsqueeze(-1)
            wv_i = vpi * D_last.unsqueeze(-1)
            sr = wv_r.transpose(-1, -2) @ kr + wv_i.transpose(-1, -2) @ ki
            si = wv_i.transpose(-1, -2) @ kr - wv_r.transpose(-1, -2) @ ki
            S_block = torch.stack([sr, si], dim=-1)
            return y, S_block
    
        def _forward_chunked_head(self, q, k, v_prime, gamma, d):
            B, H, T = q.shape[:3]
            C = self.chunk_size
            scale = d ** -0.5
            q_s = q * scale
            S = q.new_zeros(B, H, d, d, 2)
            outputs = []
            for start in range(0, T, C):
                end = min(start + C, T)
                Tc = end - start
                q_c, k_c = q_s[:, :, start:end], k[:, :, start:end]
                v_c, g_c = v_prime[:, :, start:end], gamma[:, :, start:end]
                causal = self._causal[:Tc, :Tc]
                y_c, S_chunk = self._dual_form_block(q_c, k_c, v_c, g_c, causal)
                log_g = torch.log(g_c + 1e-6)
                cum_decay = torch.exp(torch.cumsum(log_g, dim=-1))
                if start > 0:
                    Sr, Si = S[..., 0], S[..., 1]
                    qr_c, qi_c = q_c[..., 0], q_c[..., 1]
                    Sq_r = (Sr @ qr_c.transpose(-1, -2) - Si @ qi_c.transpose(-1, -2)).transpose(-1, -2)
                    Sq_i = (Sr @ qi_c.transpose(-1, -2) + Si @ qr_c.transpose(-1, -2)).transpose(-1, -2)
                    cd = cum_decay.unsqueeze(-1)
                    y_c = y_c + torch.stack([Sq_r * cd, Sq_i * cd], dim=-1)
                outputs.append(y_c)
                total_decay = cum_decay[:, :, -1]
                S = S * total_decay[..., None, None, None] + S_chunk
            return torch.cat(outputs, dim=2), S
    
        def _forward_multistate(self, x, q, k, v_prime, d):
            B, T = x.shape[0], x.shape[1]
            H, K = self.num_heads, self.n_states
            scale = d ** -0.5
            phi = self.phase_proj(_qllm_cabs(x)).view(B, T, H, K).permute(0, 2, 3, 1)
            y_sum = None
            S_list = []
            for kdx in range(K):
                gamma_k, vp_k = self._gamma_and_vprime(
                    x, v_prime, state_offset=self.state_dt_offset[kdx]
                )
                if self.decay_mode == 'per_channel':
                    raise NotImplementedError("per_channel + multistate not implemented in qllm provider")
                elif self.chunk_size > 0 and T > self.chunk_size:
                    y_k, S_k = self._forward_chunked_head(q, k, vp_k, gamma_k, d)
                else:
                    q_s = q * scale
                    y_k, S_k = self._dual_form_block(q_s, k, vp_k, gamma_k, self._causal[:T, :T])
                rot = torch.stack([
                    torch.cos(phi[:, :, kdx]),
                    torch.sin(phi[:, :, kdx])
                ], dim=-1)
                y_k = _qllm_cmul(y_k, rot.unsqueeze(-2))
                y_sum = y_k if y_sum is None else y_sum + y_k
                S_list.append(S_k)
            return y_sum, torch.stack(S_list, dim=0)
    
        def forward(self, x, state=None, step_offset=0):
            B, T, _, _ = x.shape
            H, d = self.num_heads, self.head_dim
            q, k, v = self._project(x, step_offset)
    
            if state is None and T > 1:
                if self.n_states > 1:
                    y, new_state = self._forward_multistate(x, q, k, v, d)
                elif self.decay_mode == 'per_channel':
                    raise NotImplementedError("per_channel decay not implemented in qllm provider")
                elif self.write_mode == 'delta':
                    raise NotImplementedError("delta write not implemented in qllm provider")
                else:
                    gamma, v_prime = self._gamma_and_vprime(x, v)
                    if self.chunk_size > 0 and T > self.chunk_size:
                        y, new_state = self._forward_chunked_head(q, k, v_prime, gamma, d)
                    else:
                        q_s = q * (d ** -0.5)
                        y, new_state = self._dual_form_block(q_s, k, v_prime, gamma, self._causal[:T, :T])
            else:
                y, new_state = self._recurrent(x, q, k, v, state, d)
    
            y = y.transpose(1, 2).contiguous().view(B, T, self.inner_dim, 2)
            out = self.o_proj(y)
            if self.training:
                mask = self.dropout(torch.ones(B, T, self.dim, device=x.device))
                out = out * mask.unsqueeze(-1)
            return out, new_state
    
        def _recurrent(self, x, q, k, v, state, d):
            B, T = x.shape[0], x.shape[1]
            H, K = self.num_heads, self.n_states
            scale = d ** -0.5
            if state is None:
                if self.n_states > 1:
                    S = torch.zeros(K, B, H, d, d, 2, device=x.device, dtype=x.dtype)
                else:
                    S = torch.zeros(B, H, d, d, 2, device=x.device, dtype=x.dtype)
            else:
                S = state
    
            y_list = []
            phi = None
            if self.n_states > 1:
                phi = self.phase_proj(_qllm_cabs(x)).view(B, T, H, K).permute(0, 2, 3, 1)
            for t in range(T):
                xt = x[:, t:t+1]
                k_t = k[:, :, t]
                q_t = q[:, :, t] * scale
                v_t = v[:, :, t]
                if self.n_states > 1:
                    y_acc = None
                    S_new = []
                    for kdx in range(K):
                        gamma_k, vp_k = self._gamma_and_vprime(
                            xt, v[:, :, t:t+1], state_offset=self.state_dt_offset[kdx]
                        )
                        g = gamma_k[:, :, 0]
                        yk, Sk = self._recur_step_additive(S[kdx], g, vp_k[:, :, 0], k_t, q_t)
                        rot = torch.stack([
                            torch.cos(phi[:, :, kdx, t]),
                            torch.sin(phi[:, :, kdx, t])
                        ], dim=-1)
                        yk = _qllm_cmul(yk, rot.unsqueeze(-2))
                        y_acc = yk if y_acc is None else y_acc + yk
                        S_new.append(Sk)
                    y_list.append(y_acc)
                    S = torch.stack(S_new, dim=0)
                    continue
    
                gamma, v_prime = self._gamma_and_vprime(xt, v[:, :, t:t+1])
                g = gamma[:, :, 0]
                vp_t = v_prime[:, :, 0]
                yk, S = self._recur_step_additive(S, g, vp_t, k_t, q_t)
                y_list.append(yk)
    
            y = torch.stack(y_list, dim=2)
            return y, S
    
        def _recur_step_additive(self, S, g, v_t, k_t, q_t):
            k_conj = torch.stack([k_t[..., 0], -k_t[..., 1]], dim=-1).unsqueeze(-3)
            outer_r = v_t[..., 0].unsqueeze(-1) * k_conj[..., 0] - v_t[..., 1].unsqueeze(-1) * k_conj[..., 1]
            outer_i = v_t[..., 0].unsqueeze(-1) * k_conj[..., 1] + v_t[..., 1].unsqueeze(-1) * k_conj[..., 0]
            outer = torch.stack([outer_r, outer_i], dim=-1)
            if g.dim() == S.dim() - 3:
                gg = g.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            else:
                gg = g.unsqueeze(-2).unsqueeze(-1)
            S = S * gg + outer
            sq_r = S[..., 0] * q_t[..., 0].unsqueeze(-2) - S[..., 1] * q_t[..., 1].unsqueeze(-2)
            sq_i = S[..., 0] * q_t[..., 1].unsqueeze(-2) + S[..., 1] * q_t[..., 0].unsqueeze(-2)
            y = torch.stack([sq_r.sum(dim=-1), sq_i.sum(dim=-1)], dim=-1)
            return y, S
    
    
    class _QllmBlock(nn.Module):
        def __init__(self, cfg, layer_idx=0):
            super().__init__()
            self.norm1 = _QllmComplexNorm(cfg.dim)
            self.cgu = _QllmComplexGatedUnit(cfg.dim, cfg.expand, activation=cfg.activation)
            self.cgu_scale = nn.Parameter(torch.tensor(1.0))
            self.cgu_dropout = nn.Dropout(cfg.dropout)
            self.norm2 = _QllmComplexNorm(cfg.dim)
            self.pam = _QllmPAMLayer(cfg, layer_idx=layer_idx)
            self.pam_scale = nn.Parameter(torch.tensor(0.1))
    
        def forward(self, x, pam_state=None, step_offset=0):
            cgu_out = self.cgu(self.norm1(x))
            if self.training:
                drop = self.cgu_dropout(torch.ones(cgu_out.shape[:-1], device=cgu_out.device))
                cgu_out = cgu_out * drop.unsqueeze(-1)
            x = x + cgu_out * self.cgu_scale
            pam_out, new_state = self.pam(self.norm2(x), state=pam_state, step_offset=step_offset)
            x = x + pam_out * self.pam_scale
            return x, new_state
    
    
    class _QllmLM(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.config = cfg
            self.embed = _QllmComplexEmbed(cfg.vocab_size, cfg.dim)
            self.pos_embed = _QllmComplexPosEmbed(cfg.max_seq_len, cfg.dim) if cfg.use_learned_pos else None
            self.embed_norm = _QllmComplexNorm(cfg.dim)
            self.blocks = nn.ModuleList([_QllmBlock(cfg, layer_idx=i) for i in range(cfg.n_layers)])
            self.output_norm = _QllmComplexNorm(cfg.dim)
            self.lm_head_proj = _QllmComplexLinear(cfg.dim, cfg.dim)
            self.lm_head_norm = _QllmComplexNorm(cfg.dim)
            self._init_weights()
    
        def _init_weights(self):
            embed_embeddings = {self.embed.embed_real, self.embed.embed_imag}
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding) and module not in embed_embeddings:
                    nn.init.normal_(module.weight, std=0.02)
            for _, module in self.named_modules():
                if hasattr(module, 'protect_gate') and isinstance(module.protect_gate, nn.Linear):
                    nn.init.constant_(module.protect_gate.bias, getattr(module, 'protect_gate_bias', -3.0))
    
        def forward(self, input_ids, states=None, step_offset=0, labels=None):
            z = self.embed(input_ids)
            if self.pos_embed is not None:
                z = self.pos_embed(z, step_offset=step_offset)
            z = self.embed_norm(z)
            new_states = []
            for i, block in enumerate(self.blocks):
                s = states[i] if states is not None else None
                z, new_s = block(z, pam_state=s, step_offset=step_offset)
                new_states.append(new_s)
            z = self.output_norm(z)
            lm = self.lm_head_norm(self.lm_head_proj(z))
            logits = (
                lm[..., 0] @ self.embed.embed_real.weight.T
                + lm[..., 1] @ self.embed.embed_imag.weight.T
            )
            return logits, new_states, torch.tensor(0.0, device=input_ids.device)
    
        @torch.no_grad()
        def generate(self, input_ids, max_new_tokens=100, temperature=1.0,
                     top_k=50, top_p=0.0, repetition_penalty=1.0, eos_token_id=None):
            self.eval()
            generated = input_ids.clone()
            logits, states, _ = self.forward(generated)
            step = generated.shape[1]
            finished = torch.zeros(generated.shape[0], dtype=torch.bool, device=generated.device)
            for _ in range(max_new_tokens):
                next_logits = logits[:, -1]
                if temperature > 0:
                    next_logits = next_logits / temperature
                if repetition_penalty != 1.0:
                    score = torch.gather(next_logits, 1, generated)
                    score = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
                    next_logits.scatter_(1, generated, score)
                if top_k > 0 and temperature > 0:
                    v, _ = next_logits.topk(min(top_k, next_logits.size(-1)))
                    next_logits[next_logits < v[:, -1:]] = float('-inf')
                if top_p > 0 and temperature > 0:
                    sl, si = next_logits.sort(descending=True)
                    cum = sl.softmax(dim=-1).cumsum(dim=-1)
                    rm = cum - sl.softmax(dim=-1) >= top_p
                    sl[rm] = float('-inf')
                    next_logits = sl.scatter(1, si, sl)
                if temperature <= 0:
                    nxt = next_logits.argmax(dim=-1, keepdim=True)
                else:
                    nxt = torch.multinomial(next_logits.softmax(dim=-1), 1)
                generated = torch.cat([generated, nxt], dim=1)
                if eos_token_id is not None:
                    finished |= nxt.squeeze(1) == eos_token_id
                    if bool(finished.all()):
                        break
                logits, states, _ = self.forward(nxt, states=states, step_offset=step)
                step += 1
            return generated


def _qllm_config_from_dict(d: dict) -> _QllmConfig:
    fields = {
        'vocab_size', 'dim', 'n_heads', 'head_dim', 'n_layers', 'expand',
        'dropout', 'max_seq_len', 'use_learned_pos', 'use_rope', 'use_gsp',
        'fused_qkv', 'qk_norm', 'tie_weights', 'gradient_checkpointing',
        'activation', 'chunk_size', 'decay_mode', 'write_mode', 'n_states',
        'delta_chunk', 'state_dt_spread', 'base_dt_bias', 'gate_content_aware',
        'protect_gate_bias',
    }
    return _QllmConfig(**{k: v for k, v in d.items() if k in fields})


_QLLM_MODEL_CACHE: Dict[str, Any] = {}


def _load_qllm_model_native(checkpoint_path: str, device):
    import torch
    cache_key = f"{checkpoint_path}:{str(device)}"
    if cache_key in _QLLM_MODEL_CACHE:
        return _QLLM_MODEL_CACHE[cache_key]

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if 'config' in ckpt:
        cfg = _qllm_config_from_dict(ckpt['config'])
    else:
        model_dir = os.path.dirname(checkpoint_path)
        config_path = os.path.join(model_dir, 'config.json')
        with open(config_path, 'r') as f:
            cfg = _qllm_config_from_dict(json.load(f))

    model = _QllmLM(cfg)
    model.load_state_dict(ckpt['model_state_dict'])
    if device is not None:
        model.to(device)
    model.eval()
    _QLLM_MODEL_CACHE[cache_key] = model
    return model


def _get_qllm_tokenizer_native(model):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    extras = [_QLLM_IM_START, _QLLM_IM_END]
    if getattr(model.config, "vocab_size", 50257) >= 50261:
        extras.extend([_QLLM_THINK_START, _QLLM_THINK_END])
    tok.add_special_tokens({"additional_special_tokens": extras})
    tok.pad_token = tok.eos_token
    if len(tok) != model.config.vocab_size:
        raise ValueError(
            f"Tokenizer vocab ({len(tok)}) does not match model vocab_size "
            f"({model.config.vocab_size})"
        )
    return tok


def _format_qllm_chat_messages(messages, default_system="You are a helpful assistant."):
    system_prompt = default_system
    for msg in messages:
        if msg.get("role") == "system" and str(msg.get("content", "")).strip():
            system_prompt = str(msg.get("content", "")).strip()
            break
    parts = [f"{_QLLM_IM_START}system\n{system_prompt}{_QLLM_IM_END}\n"]
    for msg in messages:
        role = str(msg.get("role", "")).strip().lower()
        content = str(msg.get("content", "")).strip()
        if role in ("system", "") or not content:
            continue
        parts.append(f"{_QLLM_IM_START}{role}\n{content}{_QLLM_IM_END}\n")
    return "".join(parts) + f"{_QLLM_IM_START}assistant\n"


def _stream_qllm_native_tokens(
    model, tokenizer, input_ids, im_end_id, max_new_tokens,
    temperature, top_k, top_p, repetition_penalty,
):
    generated = input_ids.clone()
    logits, states, _ = model.forward(generated)
    step = generated.shape[1]
    for _ in range(max_new_tokens):
        next_logits = logits[:, -1].clone()
        if temperature > 0:
            next_logits = next_logits / temperature
        if repetition_penalty != 1.0:
            score = torch.gather(next_logits, 1, generated)
            score = torch.where(score > 0, score / repetition_penalty, score * repetition_penalty)
            next_logits.scatter_(1, generated, score)
        if top_k > 0 and temperature > 0:
            v, _ = next_logits.topk(min(top_k, next_logits.size(-1)))
            next_logits[next_logits < v[:, -1:]] = float('-inf')
        if top_p > 0 and temperature > 0:
            sl, si = next_logits.sort(descending=True)
            cum = sl.softmax(dim=-1).cumsum(dim=-1)
            rm = cum - sl.softmax(dim=-1) >= top_p
            sl[rm] = float('-inf')
            next_logits = sl.scatter(1, si, sl)
        if temperature <= 0:
            nxt = next_logits.argmax(dim=-1, keepdim=True)
        else:
            nxt = torch.multinomial(next_logits.softmax(dim=-1), 1)
        token_id = int(nxt[0, 0].item())
        if token_id == im_end_id:
            break
        generated = torch.cat([generated, nxt], dim=1)
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        if token_text:
            yield {"content": token_text}
        logits, states, _ = model.forward(nxt, states=states, step_offset=step)
        step += 1


def get_qllm_response(
    prompt: str = None,
    model: str = None,
    messages: List[Dict[str, str]] = None,
    tools: list = None,
    tool_map: Dict = None,
    format: str = None,
    stream: bool = False,
    attachments: List[str] = None,
    auto_process_tool_calls: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    import torch
    result = {
        "response": None,
        "messages": messages.copy() if messages else [],
        "raw_response": None,
        "tool_calls": [],
        "tool_results": [],
    }

    if isinstance(model, str):
        if (model.startswith("orcarouter/") or model.startswith("orca/")) and model.count("/") > 1:
            model = model.split("/", 1)[1]
    if model is None:
        raise ValueError("No QLLM model specified. Pass a directory or .pt checkpoint path.")

    checkpoint_path, _ = _resolve_qllm_checkpoint(model)

    if attachments:
        logger.warning("QLLM provider does not support attachments yet; ignoring.")

    if prompt:
        if result["messages"] and result["messages"][-1].get("role") == "user":
            result["messages"][-1]["content"] = prompt
        else:
            result["messages"].append({"role": "user", "content": prompt})

    if format == "json":
        json_instruction = """If you are returning a json object, begin directly with the opening {.
Do not include any additional markdown formatting or leading ```json tags in your response."""
        if result["messages"] and result["messages"][-1].get("role") == "user":
            result["messages"][-1]["content"] += "\n" + json_instruction

    device = _get_qllm_device(kwargs.pop("device", None))
    qllm_model = _load_qllm_model_native(checkpoint_path, device)
    tokenizer = _get_qllm_tokenizer_native(qllm_model)

    chat_text = _format_qllm_chat_messages(result["messages"])
    input_ids = tokenizer.encode(chat_text, return_tensors="pt").to(device)
    im_end_id = tokenizer.convert_tokens_to_ids(_QLLM_IM_END)

    max_new_tokens = kwargs.pop("max_tokens", kwargs.pop("max_new_tokens", 256))
    temperature = kwargs.pop("temperature", 0.7)
    top_k = kwargs.pop("top_k", 50)
    top_p = kwargs.pop("top_p", 0.0)
    repetition_penalty = kwargs.pop("repetition_penalty", 1.15)

    if stream:
        result["response"] = _stream_qllm_native_tokens(
            qllm_model, tokenizer, input_ids, im_end_id,
            max_new_tokens, temperature, top_k, top_p, repetition_penalty,
        )
        return result

    with torch.no_grad():
        output_ids = qllm_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=im_end_id,
        )

    generated_ids = output_ids[0, input_ids.shape[1]:].tolist()
    response_content = tokenizer.decode(generated_ids, skip_special_tokens=False)
    if _QLLM_IM_END in response_content:
        response_content = response_content.split(_QLLM_IM_END, 1)[0].strip()

    result["response"] = response_content
    result["raw_response"] = response_content
    result["messages"].append({"role": "assistant", "content": response_content})

    if auto_process_tool_calls and tools and tool_map:
        detected_tools = []
        for tool in tools:
            tool_name = tool.get("function", {}).get("name", "")
            if tool_name in response_content:
                detected_tools.append({
                    "id": str(uuid.uuid4()),
                    "function": {"name": tool_name, "arguments": "{}"},
                })
        if detected_tools:
            result["tool_calls"] = detected_tools
            result = process_tool_calls(result, tool_map, model, "qllm", result["messages"], tools=tools)

    if format == "json":
        try:
            cleaned = response_content
            if cleaned.startswith("```json"):
                cleaned = cleaned.replace("```json", "").replace("```", "").strip()
            parsed_response = json.loads(cleaned)
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
    think= None,
    format: Union[str, BaseModel] = None,
    messages: List[Dict[str, str]] = None,
    api_key: str = None,
    api_url: str = None,
    stream: bool = False,
    attachments: List[str] = None,
    auto_process_tool_calls: bool = False, 
    include_usage: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    if not model:
        raise ValueError("No model specified. Please set a model in your NPC configuration or team settings.")
    result = {
        "response": None,
        "messages": messages.copy() if messages else [],
        "raw_response": None,
        "tool_calls": [],
        "tool_results":[],
    }
    if provider == "ollama":
        return get_ollama_response(
            prompt, 
            model, 
            images=images, 
            tools=tools, 
            tool_choice=tool_choice, 
            tool_map=tool_map,
            think=think,
            format=format, 
            messages=messages, 
            stream=stream, 
            attachments=attachments, 
            auto_process_tool_calls=auto_process_tool_calls, 
            **kwargs
        )
    elif provider == 'transformers':
        return get_transformers_response(
            prompt,
            model,
            images=images,
            tools=tools,
            tool_choice=tool_choice,
            tool_map=tool_map,
            think=think,
            format=format,
            messages=messages,
            stream=stream,
            attachments=attachments,
            auto_process_tool_calls=auto_process_tool_calls,
            **kwargs
        )
    elif provider == 'lora':
        print(f"🔧 LoRA provider detected, calling get_lora_response with model: {model}")
        result = get_lora_response(
            prompt=prompt,
            model=model,
            tools=tools,
            tool_map=tool_map,
            format=format,
            messages=messages,
            stream=stream,
            auto_process_tool_calls=auto_process_tool_calls,
            **kwargs
        )
        print(f"🔧 LoRA response: {result.get('response', 'NO RESPONSE')[:200] if result.get('response') else 'EMPTY'}")
        if result.get('error'):
            print(f"🔧 LoRA error: {result.get('error')}")
        return result
    elif provider == 'mlx':
        return get_mlx_response(
            prompt=prompt,
            model=model,
            tools=tools,
            tool_map=tool_map,
            format=format,
            messages=messages,
            stream=stream,
            auto_process_tool_calls=auto_process_tool_calls,
            **kwargs
        )
    elif provider == 'llamacpp':
        return get_llamacpp_response(
            prompt,
            model,
            images=images,
            tools=tools,
            tool_choice=tool_choice,
            tool_map=tool_map,
            think=think,
            format=format,
            messages=messages,
            stream=stream,
            attachments=attachments,
            auto_process_tool_calls=auto_process_tool_calls,
            **kwargs
        )
    elif provider == 'airllm':
        return get_airllm_response(
            prompt=prompt,
            model=model,
            tools=tools,
            tool_map=tool_map,
            format=format,
            messages=messages,
            auto_process_tool_calls=auto_process_tool_calls,
            **kwargs
        )
    elif provider == 'qllm':
        return get_qllm_response(
            prompt,
            model,
            messages=messages,
            tools=tools,
            tool_map=tool_map,
            format=format,
            stream=stream,
            attachments=attachments,
            auto_process_tool_calls=auto_process_tool_calls,
            **kwargs
        )
    elif provider == 'lmstudio' or (model and '.lmstudio' in str(model)):
        api_url = api_url or "http://127.0.0.1:1234/v1"
        provider = "openai"
        api_key = api_key or "lm-studio"
        if 'timeout' not in kwargs:
            kwargs['timeout'] = 300
    elif provider == 'llamacpp-server':
        api_url = api_url or "http://127.0.0.1:8080/v1"
        provider = "openai"
        api_key = api_key or "llamacpp"
        if 'timeout' not in kwargs:
            kwargs['timeout'] = 300
    elif provider == 'omlx':
        api_url = api_url or "http://127.0.0.1:8000/v1"
        provider = "openai"
        api_key = api_key or "omlx"
        if 'timeout' not in kwargs:
            kwargs['timeout'] = 300
    elif provider in ('orcarouter', 'orca'):
        api_url = api_url or os.environ.get("ORCAROUTER_API_URL") or "https://api.orcarouter.ai/v1"
        api_key = api_key or os.environ.get("ORCAROUTER_API_KEY")
        provider = "openai"
        if 'timeout' not in kwargs:
            kwargs['timeout'] = 300

    if attachments:
        for attachment in attachments:
            if os.path.exists(attachment):
                _, ext = os.path.splitext(attachment)
                ext = ext.lower()
                
                if ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
                    if not images:
                        images = []
                    images.append(attachment)
                elif ext == '.pdf':
                    try:
                        from npcpy.data.load import load_pdf
                        pdf_data = load_pdf(attachment)
                        if pdf_data is not None:
                            if prompt:
                                prompt += f"\n\nContent from PDF: {os.path.basename(attachment)}\n{pdf_data}..."
                            else:
                                prompt = f"Content from PDF: {os.path.basename(attachment)}\n{pdf_data}..."

                    except Exception:
                        pass
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
                    except Exception:
                        pass
                else:
                    text_extensions = {'.txt', '.text', '.log', '.md', '.markdown', '.rst', '.json', '.yaml', '.yml', '.toml', '.ini', '.conf', '.cfg', '.xml', '.html', '.htm', '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.c', '.h', '.cpp', '.hpp', '.go', '.rs', '.rb', '.php', '.sh', '.bash', '.sql', '.css', '.scss'}
                    filename = os.path.basename(attachment)
                    if ext in text_extensions or ext == '':
                        try:
                            with open(attachment, 'r', encoding='utf-8', errors='replace') as f:
                                text_content = f.read()
                            max_chars = 50000
                            if len(text_content) > max_chars:
                                text_content = text_content[:max_chars] + f"\n\n... [truncated]"
                            if text_content.strip():
                                if prompt:
                                    prompt += f"\n\nContent from {filename}:\n```\n{text_content}\n```"
                                else:
                                    prompt = f"Content from {filename}:\n```\n{text_content}\n```"
                        except Exception:
                            pass

    if prompt:
        if result['messages'] and result['messages'][-1]["role"] == "user":
            if isinstance(messages[-1]["content"], str):
                result['messages'][-1]["content"] = prompt
            elif isinstance(result['messages'][-1]["content"], list):
                for i, item in enumerate(result['messages'][-1]["content"]):
                    if item.get("type") == "text":
                        result['messages'][-1]["content"][i]["text"] = prompt
                        break
                else:
                    result['messages'][-1]["content"].append({"type": "text", "text": prompt})
        else:
            result['messages'].append({"role": "user", "content": prompt})

    if format == "json" and not stream:
        json_instruction = """If you are a returning a json object, begin directly with the opening {.
            If you are returning a json array, begin directly with the opening [.
            Do not include any additional markdown formatting or leading
            ```json tags in your response. The item keys should be based on the ones provided
            by the user. Do not invent new ones."""

        if result["messages"] and result["messages"][-1]["role"] == "user":
            if isinstance(result["messages"][-1]["content"], list):
                result["messages"][-1]["content"].append({"type": "text", "text": json_instruction})
            elif isinstance(result["messages"][-1]["content"], str):
                result["messages"][-1]["content"] += "\n" + json_instruction

    if format == "yaml" and not stream:
        yaml_instruction = """Return your response as valid YAML. Do not include ```yaml markdown tags.
            For multi-line strings like code, use the literal block scalar (|) syntax:
            code: |
              your code here
              more lines here
            The keys should be based on the ones requested by the user. Do not invent new ones."""

        if result["messages"] and result["messages"][-1]["role"] == "user":
            if isinstance(result["messages"][-1]["content"], list):
                result["messages"][-1]["content"].append({"type": "text", "text": yaml_instruction})
            elif isinstance(result["messages"][-1]["content"], str):
                result["messages"][-1]["content"] += "\n" + yaml_instruction

    if images:
        last_user_idx = -1
        for i, msg in enumerate(result["messages"]):
            if msg["role"] == "user":
                last_user_idx = i
        if last_user_idx == -1:
            result["messages"].append({"role": "user", "content": []})
            last_user_idx = len(result["messages"]) - 1
        if isinstance(result["messages"][last_user_idx]["content"], str):
            
            result["messages"][last_user_idx]["content"] = [{"type": "text", 
                                                             "text": result["messages"][last_user_idx]["content"]
                                                             }]

        elif not isinstance(result["messages"][last_user_idx]["content"], list):
            result["messages"][last_user_idx]["content"] = []
        for image_path in images:
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(compress_image(image_file.read())).decode("utf-8")
                result["messages"][last_user_idx]["content"].append(
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                )

    

    result["messages"] = sanitize_messages(result["messages"])
    api_params = {"messages": result["messages"]}

    if include_usage:
      litellm.include_cost_in_streaming_usage = True
      api_params['stream_options'] = {"include_usage": True}

    if api_url is not None and (
        provider in ("openai", "openai-like")
        or 'openai-like' in provider
        or provider == "anthropic"
    ):
        api_params["api_base"] = api_url
        if provider != "anthropic":
            provider = "openai"

    if isinstance(format, type) and issubclass(format, BaseModel):
        api_params["response_format"] = format
    if isinstance(model, str):
        if (model.startswith("orcarouter/") or model.startswith("orca/")) and model.count("/") > 1:
            model = model.split("/", 1)[1]
    if model is None:
        raise ValueError("No model specified. Please set a model in your NPC configuration or team settings.")
    if provider is None:
        raise ValueError("No provider specified. Please set a provider in your NPC configuration or team settings.")

    # LiteLLM routes many providers (e.g. OpenRouter) by the model prefix.
    # e.g. `moonshotai/kimi-k3` is unresolvable, but `openrouter/moonshotai/kimi-k3` works.
    # Use a lowercase provider slug for the prefix because LiteLLM expects that.
    normalized_model = model.lower()
    normalized_provider = provider.lower().replace(" ", "")
    if "api_base" in api_params and normalized_provider == "openai":
        api_params["model"] = f"openai/{model}"
    elif "/" not in model or model.startswith("/"):
        api_params["model"] = f"{normalized_provider}/{model}"
    elif not normalized_model.startswith(normalized_provider + "/"):
        api_params["model"] = f"{normalized_provider}/{model}"
    else:
        api_params["model"] = model
    if api_key is not None: 
        api_params["api_key"] = api_key
    if tools: 
        api_params["tools"] = tools
    if tool_choice: 
        api_params["tool_choice"] = tool_choice
    
    if kwargs:
        for key, value in kwargs.items():
            if key in [
                "stop", "temperature", "top_p", "max_tokens", "max_completion_tokens",
                 "extra_headers", "parallel_tool_calls",
                "response_format", "user", "timeout", "think", "thinking", "reasoning_effort",
            ]:
                if key == "temperature" and "claude" in str(api_params.get("model", "")).lower():
                    api_params[key] = value
                elif key == "top_p" and "claude" in str(api_params.get("model", "")).lower():
                    if "temperature" not in kwargs:
                        api_params[key] = value
                else:
                    api_params[key] = value

    if not auto_process_tool_calls or not (tools and tool_map):
        api_params["stream"] = stream
        resp = completion(**api_params)
        result["raw_response"] = resp

        if hasattr(resp, 'usage') and resp.usage:
            result["usage"] = {
                "input_tokens": getattr(resp.usage, 'prompt_tokens', 0) or 0,
                "output_tokens": getattr(resp.usage, 'completion_tokens', 0) or 0,
            }
        elif hasattr(resp, 'prompt_eval_count'):
            result["usage"] = {
                "input_tokens": getattr(resp, 'prompt_eval_count', 0) or 0,
                "output_tokens": getattr(resp, 'eval_count', 0) or 0,
            }

        if stream:
            result["response"] = resp
            return result
        else:
            
            llm_response = resp.choices[0].message.content
            result["response"] = llm_response
            assistant_msg = {"role": "assistant", "content": llm_response}
            if hasattr(resp.choices[0].message, 'tool_calls') and resp.choices[0].message.tool_calls:
                raw_tcs = resp.choices[0].message.tool_calls
                result["tool_calls"] = raw_tcs
                tc_dicts = []
                for tc in raw_tcs:
                    if isinstance(tc, dict):
                        tc_dicts.append(tc)
                    else:
                        tc_dicts.append({
                            "id": getattr(tc, "id", str(uuid.uuid4())),
                            "type": "function",
                            "function": {
                                "name": getattr(tc.function, "name", "") if hasattr(tc, "function") else "",
                                "arguments": getattr(tc.function, "arguments", "{}") if hasattr(tc, "function") else "{}"
                            }
                        })
                assistant_msg["tool_calls"] = tc_dicts
            result["messages"].append(assistant_msg)
            if format == "json":
                try:
                    if isinstance(llm_response, str):
                        llm_response = llm_response.strip()
                        
                        if '```json' in llm_response:
                            start = llm_response.find('```json') + 7
                            end = llm_response.rfind('```')
                            if end > start:
                                llm_response = llm_response[start:end].strip()
                        
                        first_brace = llm_response.find('{')
                        first_bracket = llm_response.find('[')
                        
                        if first_brace == -1 and first_bracket == -1:
                            result["response"] = {}
                            result["error"] = "No JSON found in response"
                            return result
                        
                        if first_brace != -1 and (first_bracket == -1 or first_brace < first_bracket):
                            llm_response = llm_response[first_brace:]
                            last_brace = llm_response.rfind('}')
                            if last_brace != -1:
                                llm_response = llm_response[:last_brace+1]
                        else:
                            llm_response = llm_response[first_bracket:]
                            last_bracket = llm_response.rfind(']')
                            if last_bracket != -1:
                                llm_response = llm_response[:last_bracket+1]
                        
                        parsed_json = json.loads(llm_response, strict=False)
                        
                        if "json" in parsed_json:
                            result["response"] = parsed_json["json"]
                        else:
                            result["response"] = parsed_json
                        
                except (json.JSONDecodeError, TypeError) as e:
                    logger.debug(f"JSON parsing error: {str(e)}, raw response: {llm_response[:500]}")
                    result["response"] = {}
                    result["error"] = "Invalid JSON response"

            if format == "yaml":
                try:
                    if isinstance(llm_response, str):
                        llm_response = llm_response.strip()

                        if '```yaml' in llm_response:
                            start = llm_response.find('```yaml') + 7
                            end = llm_response.rfind('```')
                            if end > start:
                                llm_response = llm_response[start:end].strip()
                        elif '```' in llm_response:
                            start = llm_response.find('```') + 3
                            newline = llm_response.find('\n', start)
                            if newline != -1:
                                start = newline + 1
                            end = llm_response.rfind('```')
                            if end > start:
                                llm_response = llm_response[start:end].strip()

                        parsed_yaml = yaml.safe_load(llm_response)
                        result["response"] = parsed_yaml

                except (yaml.YAMLError, TypeError) as e:
                    logger.debug(f"YAML parsing error: {str(e)}, raw response: {llm_response[:500]}")
                    result["response"] = {}
                    result["error"] = "Invalid YAML response"

            return result

    
    
    initial_api_params = api_params.copy()
    initial_api_params["stream"] = False

    try:
        resp = completion(**initial_api_params)
    except Exception as e:
        logger.error(f"litellm completion() failed: {type(e).__name__}: {e}")
        result["error"] = str(e)
        result["response"] = f"LLM call failed: {e}"
        return result

    result["raw_response"] = resp

    if hasattr(resp, 'usage') and resp.usage:
        result["usage"] = {
            "input_tokens": getattr(resp.usage, 'prompt_tokens', 0) or 0,
            "output_tokens": getattr(resp.usage, 'completion_tokens', 0) or 0,
        }

    if not resp.choices:
        result["response"] = "No response from model"
        return result

    has_tool_calls = hasattr(resp.choices[0].message, 'tool_calls') and resp.choices[0].message.tool_calls
    
    if has_tool_calls:
        result["tool_calls"] = resp.choices[0].message.tool_calls

        processed_result = process_tool_calls(result,
                                              tool_map,
                                              model,
                                              provider,
                                              result["messages"],
                                              stream=False,
                                              tools=tools)

        clean_messages = []
        tool_results_summary = []

        for msg in processed_result["messages"]:
            role = msg.get('role', '')
            if role == 'assistant' and 'tool_calls' in msg:
                continue
            elif role == 'tool':
                content = msg.get('content', '')
                if len(content) > 2000:
                    content = content[:2000] + "... (truncated)"
                tool_results_summary.append(content)
            else:
                clean_messages.append(msg)

        if tool_results_summary:
            clean_messages.append({
                "role": "assistant",
                "content": "I executed the requested tools. Here are the results:\n\n" + "\n\n".join(tool_results_summary)
            })

        clean_messages.append({
            "role": "user",
            "content": "Based on the tool results above, provide a brief summary of what happened. Do NOT output any code - the tool has already executed. Just describe the results concisely."
        })

        final_api_params = api_params.copy()
        final_api_params["messages"] = clean_messages
        final_api_params["stream"] = stream
        if "tools" in final_api_params:
            del final_api_params["tools"]
        if "tool_choice" in final_api_params:
            del final_api_params["tool_choice"]

        final_resp = completion(**final_api_params)

        if stream:
            processed_result["response"] = final_resp
        else:
            if final_resp.choices:
                final_content = final_resp.choices[0].message.content
                processed_result["response"] = final_content
                processed_result["messages"].append({"role": "assistant", "content": final_content})
            else:
                if tool_results_summary:
                    fallback_content = "\n\n".join(tool_results_summary)
                else:
                    fallback_content = "Tool executed successfully."
                processed_result["response"] = fallback_content
                processed_result["messages"].append({"role": "assistant", "content": fallback_content})

        return processed_result
        
        
    else:
        llm_response = resp.choices[0].message.content
        result["messages"].append({"role": "assistant", "content": llm_response})
        
        if stream:
            def string_chunk_generator():
                chunk_size = 1
                for i, char in enumerate(llm_response):
                    yield type('MockChunk', (), {
                        'id': f'mock-chunk-{i}',
                        'object': 'chat.completion.chunk',
                        'created': int(time.time()),
                        'model': model or 'unknown',
                        'choices': [type('Choice', (), {
                            'index': 0,
                            'delta': type('Delta', (), {
                                'content': char,
                                'role': 'assistant' if i == 0 else None
                            })(),
                            'finish_reason': 'stop' if i == len(llm_response) - 1 else None
                        })()]
                    })()
            
            result["response"] = string_chunk_generator()
        else:
            result["response"] = llm_response
    return result            
def process_tool_calls(response_dict, tool_map, model, provider, messages, stream=False, tools=None):
    result = response_dict.copy()
    result["tool_results"] = []

    if "messages" not in result:
        result["messages"] = messages if messages else []

    tool_calls = result.get("tool_calls", [])

    if not tool_calls:
        return result

    tool_calls_for_message = []
    for tc in tool_calls:
        if isinstance(tc, dict):
            tool_calls_for_message.append(tc)
        else:
            tool_calls_for_message.append({
                "id": getattr(tc, "id", str(uuid.uuid4())),
                "type": "function",
                "function": {
                    "name": getattr(tc.function, "name", "") if hasattr(tc, "function") else "",
                    "arguments": getattr(tc.function, "arguments", "{}") if hasattr(tc, "function") else "{}"
                }
            })

    result["messages"].append({
        "role": "assistant",
        "content": None,
        "tool_calls": tool_calls_for_message
    })

    for tool_call in tool_calls:
        tool_id = str(uuid.uuid4())
        tool_name = None
        arguments = {}
        

        if isinstance(tool_call, dict):
            tool_id = tool_call.get("id", str(uuid.uuid4()))
            tool_name = tool_call.get("function", {}).get("name")
            arguments_str = tool_call.get("function", {}).get("arguments", "{}")
        else:
            tool_id = getattr(tool_call, "id", str(uuid.uuid4()))
            if hasattr(tool_call, "function"):
                func_obj = tool_call.function
                tool_name = getattr(func_obj, "name", None)
                arguments_str = getattr(func_obj, "arguments", "{}")
            else:
                continue

        try:
            arguments = json.loads(arguments_str) if isinstance(arguments_str, str) else arguments_str
        except json.JSONDecodeError:
            arguments = {"raw_arguments": arguments_str}
        
        
        if tool_name in tool_map:
            tool_result = None
            tool_result_str = ""
            serializable_result = None

            try:
                tool_result = tool_map[tool_name](**arguments)
            except Exception as e:
                tool_result = f"Error executing tool '{tool_name}': {str(e)}"

            try:
                tool_result_str = json.dumps(tool_result, default=str)
                try:
                    serializable_result = json.loads(tool_result_str)
                except json.JSONDecodeError:
                    serializable_result = {"result": tool_result_str}
            except Exception as e_serialize:
                tool_result_str = f"Error serializing result for {tool_name}: {str(e_serialize)}"
                serializable_result = {"error": tool_result_str}

            result["tool_results"].append({
                "tool_call_id": tool_id,
                "tool_name": tool_name,
                "arguments": arguments,
                "result": serializable_result
            })

            result["messages"].append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": tool_result_str
            })
    
    return result
