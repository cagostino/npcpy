from jinja2 import Environment, FileSystemLoader, Undefined
import json
import os
import PIL
import random
import subprocess
import copy
import itertools
import logging
from typing import List, Dict, Any, Optional, Union

logger = logging.getLogger("npcpy.llm_funcs")
from npcpy.npc_sysenv import (
    print_and_process_stream_with_markdown,
    render_markdown,
    lookup_provider,
    request_user_input, 
    get_system_message
)

from npcpy.gen.response import get_litellm_response
from npcpy.gen.image_gen import generate_image
from npcpy.gen.video_gen import generate_video_diffusers, generate_video_veo3

from datetime import datetime 

def gen_image(
    prompt: str,
    model: str = None,
    provider: str = None,
    npc: Any = None,
    height: int = 1024,
    width: int = 1024,
    n_images: int=1, 
    input_images: List[Union[str, bytes, PIL.Image.Image]] = None,
    save = False, 
    filename = '',
    api_key: str = None,
):
    """This function generates an image using the specified provider and model.
    Args:
        prompt (str): The prompt for generating the image.
    Keyword Args:
        model (str): The model to use for generating the image.
        provider (str): The provider to use for generating the image.
        filename (str): The filename to save the image to.
        npc (Any): The NPC object.
        api_key (str): The API key for the image generation service.
    Returns:
        List[PIL.Image.Image]: A list of generated PIL Image objects.
    """
    if model is not None and provider is not None:
        pass
    elif model is not None and provider is None:
        provider = lookup_provider(model)
    elif npc is not None:
        if npc.provider is not None:
            provider = npc.provider
        if npc.model is not None:
            model = npc.model
        if npc.api_url is not None:
            api_url = npc.api_url

    images = generate_image(
        prompt=prompt,
        model=model,
        provider=provider,
        height=height,
        width=width, 
        attachments=input_images,
        n_images=n_images, 
        api_key=api_key,
        
    )
    if save:
        if len(filename) == 0 :
            todays_date = datetime.now().strftime("%Y-%m-%d")
            filename = 'vixynt_gen'
        for i, image in enumerate(images):
            
            image.save(filename+'_'+str(i)+'.png')
    return images

def gen_video(
    prompt,
    model: str = None,
    provider: str = None,
    npc: Any = None,
    device: str = "cpu",
    output_path="",
    num_inference_steps=10,
    num_frames=25,
    height=256,
    width=256,
    negative_prompt="",
    messages: list = None,
):
    """
    Function Description:
        This function generates a video using either Diffusers or Veo 3 via Gemini API.
    Args:
        prompt (str): The prompt for generating the video.
    Keyword Args:
        model (str): The model to use for generating the video.
        provider (str): The provider to use for generating the video (gemini for Veo 3).
        device (str): The device to run the model on ('cpu' or 'cuda').
        negative_prompt (str): What to avoid in the video (Veo 3 only).
    Returns:
        dict: Response with output path and messages.
    """
    
    if provider == "gemini":
        
        try:
            output_path = generate_video_veo3(
                prompt=prompt,
                model=model,
                negative_prompt=negative_prompt,
                output_path=output_path,
            )
            return {
                "output": f"High-fidelity video with synchronized audio generated at {output_path}",
                "messages": messages
            }
        except Exception as e:
            print(f"Veo 3 generation failed: {e}")
            print("Falling back to diffusers...")
            provider = "diffusers"
    
    if provider == "diffusers" or provider is None:
      
        output_path = generate_video_diffusers(
            prompt,
            model,
            npc=npc,
            device=device,
            output_path=output_path,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            height=height,
            width=width,
        )
        return {
            "output": f"Video generated at {output_path}",
            "messages": messages
        }
    
    return {
        "output": f"Unsupported provider: {provider}",
        "messages": messages
    }

def resolve_model_provider(
    npc=None, team=None, model=None, provider=None,
    api_url=None, api_key=None, images=None, attachments=None,
):
    """Resolve model, provider, api_url, and api_key from npc/team/explicit overrides."""
    m, p, a_url, a_key = model, provider, api_url, api_key
    if m is not None and p is not None:
        pass
    elif p is None and m is not None:
        p = lookup_provider(m)
    elif npc is not None:
        if npc.provider is not None:
            p = npc.provider
        if npc.model is not None:
            m = npc.model
        elif team is not None and team.model is not None:
            m = team.model
        if p is None and team is not None and team.provider is not None:
            p = team.provider
    elif team is not None:
        if team.model is not None:
            m = team.model
        if team.provider is not None:
            p = team.provider
    else:
        pass
    if a_url is None and npc is not None and getattr(npc, 'api_url', None) is not None:
        a_url = npc.api_url
    if a_url is None and team is not None and getattr(team, 'api_url', None) is not None:
        a_url = team.api_url
    if a_key is None and npc is not None and getattr(npc, 'api_key', None) is not None:
        a_key = npc.api_key
    if a_key is None and team is not None and getattr(team, 'api_key', None) is not None:
        a_key = team.api_key
    if p == "minimax":
        a_url = a_url or os.environ.get("MINIMAX_API_URL") or "https://api.minimax.io/v1"
        p = "anthropic" if a_url.rstrip("/").endswith("/anthropic") else "openai"
        a_key = a_key or os.environ.get("MINIMAX_API_KEY")
    if p in ("orcarouter", "orca"):
        a_url = a_url or os.environ.get("ORCAROUTER_API_URL") or "https://api.orcarouter.ai/v1"
        a_key = a_key or os.environ.get("ORCAROUTER_API_KEY")
    return m, p, a_url, a_key


def get_llm_response(
    prompt: str,
    model: str = None,
    provider: str = None,
    images: List[str] = None,
    npc: Any = None,
    team: Any = None,
    messages: List[Dict[str, str]] = None,
    api_url: str = None,
    api_key: str = None,
    context=None,
    stream: bool = False,
    attachments: List[str] = None,
    include_usage: bool = False,
    n_samples: int = 1,
    matrix: Optional[Dict[str, List[Any]]] = None,
    **kwargs,
):
    """Generate a response using the specified provider and model."""
    logger.debug(
        f"[get_llm_response] {len(messages) if messages else 0} messages, "
        f"prompt_len={len(prompt) if prompt else 0}, "
        f"context={'yes' if context else 'no'}"
    )

    base_model, base_provider, base_api_url, base_api_key = resolve_model_provider(
        npc=npc, team=team, model=model, provider=provider,
        api_url=api_url, api_key=api_key, images=images, attachments=attachments,
    )
    if api_key is None:
        api_key = base_api_key

    use_matrix = matrix is not None and len(matrix) > 0
    multi_sample = n_samples and n_samples > 1

    if not use_matrix and not multi_sample:
        run_model, run_provider, run_api_url, run_api_key = resolve_model_provider(
            npc=npc, team=team, model=base_model, provider=base_provider,
            api_url=api_url, api_key=api_key, images=images, attachments=attachments,
        )
        tool_capable = bool(kwargs.get("tools"))
        system_message = get_system_message(npc, team, tool_capable=tool_capable) if npc is not None else "You are a helpful assistant."

        run_messages = copy.deepcopy(messages) if messages else []
        if not run_messages:
            run_messages = [{"role": "system", "content": system_message}]

        full_text = ""
        if prompt and context:
            full_text = f"{prompt}\n\n\nUser Provided Context: {context}"
        elif prompt:
            full_text = prompt
        elif context:
            full_text = f"User Provided Context: {context}"

        if full_text:
            if run_messages[-1]["role"] == "user" and isinstance(run_messages[-1]["content"], str):
                run_messages[-1]["content"] += "\n" + full_text
            else:
                run_messages.append({"role": "user", "content": full_text})

        if not run_model:
            raise ValueError("No model specified. Please set a model in your NPC configuration or team settings.")
        return get_litellm_response(
            full_text or None,
            messages=run_messages,
            model=run_model,
            provider=run_provider,
            api_url=run_api_url,
            api_key=api_key,
            images=images,
            attachments=attachments,
            stream=stream,
            include_usage=include_usage,
            **kwargs,
        )

    combos = []
    if use_matrix:
        keys = list(matrix.keys())
        values = []
        for key in keys:
            val = matrix[key]
            if isinstance(val, (list, tuple, set)):
                values.append(list(val))
            else:
                values.append([val])
        for combo_values in itertools.product(*values):
            combos.append(dict(zip(keys, combo_values)))
    else:
        combos.append({})

    runs = []
    for combo in combos:
        run_npc = combo.get("npc", npc)
        run_team = combo.get("team", team)
        run_context = combo.get("context", context)
        extra_kwargs = dict(kwargs)
        for k, v in combo.items():
            if k not in {"model", "provider", "npc", "context", "team"}:
                extra_kwargs[k] = v

        run_model, run_provider, run_api_url, run_api_key = resolve_model_provider(
            npc=run_npc, team=run_team,
            model=combo.get("model", base_model),
            provider=combo.get("provider", base_provider),
            api_url=api_url, api_key=api_key, images=images, attachments=attachments,
        )

        tool_capable = bool(extra_kwargs.get("tools"))
        system_message = get_system_message(run_npc, run_team, tool_capable=tool_capable) if run_npc is not None else "You are a helpful assistant."

        run_messages = copy.deepcopy(messages) if messages else []
        if not run_messages:
            run_messages = [{"role": "system", "content": system_message}]

        full_text = ""
        if prompt and run_context:
            full_text = f"{prompt}\n\n\nUser Provided Context: {run_context}"
        elif prompt:
            full_text = prompt
        elif run_context:
            full_text = f"User Provided Context: {run_context}"

        if full_text:
            if run_messages[-1]["role"] == "user" and isinstance(run_messages[-1]["content"], str):
                run_messages[-1]["content"] += "\n" + full_text
            else:
                run_messages.append({"role": "user", "content": full_text})

        for sample_idx in range(max(1, n_samples or 1)):
            resp = get_litellm_response(
                full_text or None,
                messages=run_messages,
                model=run_model,
                provider=run_provider,
                api_url=run_api_url,
                api_key=api_key,
                images=images,
                attachments=attachments,
                stream=stream,
                include_usage=include_usage,
                **extra_kwargs,
            )
            runs.append({
                "response": resp.get("response") if isinstance(resp, dict) else resp,
                "raw": resp,
                "combo": combo,
                "sample_index": sample_idx,
            })

    aggregated = {
        "response": runs[0]["response"] if runs else None,
        "runs": runs,
    }
    if runs and isinstance(runs[0]["raw"], dict) and "messages" in runs[0]["raw"]:
        aggregated["messages"] = runs[0]["raw"].get("messages")
    return aggregated

def execute_llm_command(
    command: str,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    api_url: str = None,
    api_key: str = None,
    npc: Optional[Any] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    stream=False,
    context=None,
) -> str:
    """This function executes an LLM command.
    Args:
        command (str): The command to execute.

    Keyword Args:
        model (Optional[str]): The model to use for executing the command.
        provider (Optional[str]): The provider to use for executing the command.
        npc (Optional[Any]): The NPC object.
        messages (Optional[List[Dict[str, str]]): The list of messages.
    Returns:
        str: The result of the LLM command.
    """
    if messages is None:
        messages = []
    max_attempts = 5
    attempt = 0
    subcommands = []

   
    context = ""
    while attempt < max_attempts:
        prompt = f"""
        A user submitted this query: {command}.
        You need to generate a bash command that will accomplish the user's intent.
        Respond ONLY with the bash command that should be executed. 
        Do not include markdown formatting
        """
        response = get_llm_response(
            prompt,
            model=model,
            provider=provider,
            api_url=api_url,
            api_key=api_key,
            messages=messages,
            npc=npc,
            context=context,
        )

        bash_command = response.get("response", {})
 
        print(f"LLM suggests the following bash command: {bash_command}")
        subcommands.append(bash_command)

        try:
            print(f"Running command: {bash_command}")
            result = subprocess.run(
                bash_command, shell=True, text=True, capture_output=True, check=True
            )
            print(f"Command executed with output: {result.stdout}")

            prompt = f"""
                Here was the output of the result for the {command} inquiry
                which ran this bash command {bash_command}:

                {result.stdout}

                Provide a simple response to the user that explains to them
                what you did and how it accomplishes what they asked for.
                """

            messages.append({"role": "user", "content": prompt})
           
            response = get_llm_response(
                prompt,
                model=model,
                provider=provider,
                api_url=api_url,
                api_key=api_key,
                npc=npc,
                messages=messages,
                context=context,
                stream =stream
            )

            return response
        except subprocess.CalledProcessError as e:
            print(f"Command failed with error:")
            print(e.stderr)

            error_prompt = f"""
            The command '{bash_command}' failed with the following error:
            {e.stderr}
            Please suggest a fix or an alternative command.
            Respond with a JSON object containing the key "bash_command" with the suggested command.
            Do not include any additional markdown formatting.

            """

            fix_suggestion = get_llm_response(
                error_prompt,
                model=model,
                provider=provider,
                npc=npc,
                api_url=api_url,
                api_key=api_key,
                format="json",
                messages=messages,
                context=context,
            )

            fix_suggestion_response = fix_suggestion.get("response", {})

            try:
                if isinstance(fix_suggestion_response, str):
                    fix_suggestion_response = json.loads(fix_suggestion_response)

                if (
                    isinstance(fix_suggestion_response, dict)
                    and "bash_command" in fix_suggestion_response
                ):
                    print(
                        f"LLM suggests fix: {fix_suggestion_response['bash_command']}"
                    )
                    command = fix_suggestion_response["bash_command"]
                else:
                    raise ValueError(
                        "Invalid response format from LLM for fix suggestion"
                    )
            except (json.JSONDecodeError, ValueError) as e:
                print(f"Error parsing LLM fix suggestion: {e}")

        attempt += 1

    return {
        "messages": messages,
        "output": "Max attempts reached. Unable to execute the command successfully.",
    }

def handle_request_input(
    context: str,
    model: str ,
    provider: str 
):
    """
    Analyze text and decide what to request from the user
    """
    json_format = """Return a JSON object with:
    {
        "input_needed": boolean,
        "request_reason": string explaining why input is needed,
        "request_prompt": string to show user if input needed
    }

    Do not include any additional markdown formatting or leading ```json tags. Your response
    must be a valid JSON object."""

    prompt = f"""
    Analyze the text:
    {context}
    and determine what additional input is needed.
    {json_format}
    """

    response = get_llm_response(
        prompt,
        model=model,
        provider=provider,
        messages=[],
        format="json",
    )

    result = response.get("response", {})
    if isinstance(result, str):
        result = json.loads(result)

    user_input = request_user_input(
        {"reason": result["request_reason"], "prompt": result["request_prompt"]},
    )
    return user_input

def _get_jinxes(npc, team):
    """Get available jinxes from npc (already filtered by jinxes_spec)."""
    jinxes = {}
    if npc and hasattr(npc, 'jinxes_dict'):
        jinxes.update(npc.jinxes_dict)
    return jinxes

def _jinxes_to_tools(jinxes):
    """Convert jinxes to OpenAI-style tool definitions."""
    return [jinx.to_tool_def() for jinx in jinxes.values()]

def _execute_jinx(jinx, inputs, npc, team, messages, extra_globals):
    """Execute a jinx and return output."""
    try:
        jinja_env = None
        if npc and hasattr(npc, 'jinja_env'):
            jinja_env = npc.jinja_env
        elif team and hasattr(team, 'forenpc') and team.forenpc:
            jinja_env = getattr(team.forenpc, 'jinja_env', None)

        full_inputs = {}
        for inp in getattr(jinx, 'inputs', []):
            if isinstance(inp, dict):
                key = list(inp.keys())[0]
                default_val = inp[key]
                if isinstance(default_val, str) and default_val.startswith("~"):
                    default_val = os.path.expanduser(default_val)
                full_inputs[key] = default_val
            else:
                full_inputs[inp] = ""
        full_inputs.update(inputs or {})

        result = jinx.execute(
            input_values=full_inputs,
            npc=npc,
            messages=messages,
            extra_globals=extra_globals,
            jinja_env=jinja_env
        )
        if result is None:
            return "Executed with no output."
        if not isinstance(result, dict):
            return str(result)
        return result.get("output", str(result))
    except Exception as e:
        return f"Error: {e}"

def _build_jinx_schema(jinx_obj):
    """Build a self-describing schema string from a jinx's own metadata.

    Only shows params the model must fill: the primary param (first with
    empty default) plus any param with a non-empty default.  Optional
    params with empty defaults are skipped — _execute_jinx fills those
    from the jinx's own defaults so the model can't hallucinate values.
    """
    desc = getattr(jinx_obj, 'description', '') or ''
    inp_list = getattr(jinx_obj, 'inputs', [])
    params = []
    has_primary = False
    for inp in inp_list:
        if isinstance(inp, str):
            params.append(f'"{inp}": "..."')
            has_primary = True
        elif isinstance(inp, dict):
            for k, v in inp.items():
                if isinstance(v, dict) and 'description' in v:
                    params.append(f'"{k}": "...({v["description"]})"')
                    has_primary = True
                elif v:
                    params.append(f'"{k}": "...(default: {v})"')
                else:
                    if not has_primary:
                        params.append(f'"{k}": "..."')
                        has_primary = True
    schema_str = '{' + ', '.join(params) + '}'
    return desc, schema_str


def handle_jinx_call(
    command,
    jinx_name,
    jinxes,
    model=None,
    provider=None,
    api_url=None,
    api_key=None,
    npc=None,
    team=None,
    messages=None,
    stream=False,
    context=None,
    extra_globals=None,
    previous_output=None,
    n_attempts=3,
    attempt=0,
):
    """Resolve inputs for a single jinx and execute it. Retries on failure."""
    if messages is None:
        messages = []

    jinx = jinxes.get(jinx_name)
    if not jinx:
        if attempt < n_attempts:
            available = ", ".join(jinxes.keys())
            return check_llm_command(
                f"""In the previous attempt, the jinx name was: {jinx_name}.
That jinx was not available. Only select from: {available}.
Original request: {command}""",
                model=model, 
                provider=provider, 
                api_url=api_url, 
                api_key=api_key,
                npc=npc, 
                team=team,
                messages=messages, 
                stream=stream,
                context=context, 
                extra_globals=extra_globals,
            )
        return {"output": f"Jinx '{jinx_name}' not found after {n_attempts} attempts.", "messages": messages}

    print(f"[JINX] {jinx.jinx_name}", flush=True)

    example_format = {}
    for inp in jinx.inputs:
        if isinstance(inp, str):
            example_format[inp] = "..."
        elif isinstance(inp, dict):
            key = list(inp.keys())[0]
            example_format[key] = "..."
    json_format_str = json.dumps(example_format, indent=4)

    prompt = f"""The user wants to use the jinx '{jinx_name}' with the following request:
'{command}'

Here were the previous 5 messages in the conversation: {messages[-5:]}

Here is the jinx file:
```
{jinx.to_dict()}
```

Please determine the required inputs for the jinx as a JSON object.
They must be exactly as they are named in the jinx.
If the jinx requires a file path, you must include an absolute path with extension.
If the jinx requires code, generate it exactly according to the instructions.

Return only the JSON object without any markdown formatting.
The format of the JSON object is:
{json_format_str}"""

    if npc and hasattr(npc, "shared_context"):
        if npc.shared_context.get("dataframes"):
            context_info = "\nAvailable dataframes:\n"
            for df_name in npc.shared_context["dataframes"].keys():
                context_info += f"- {df_name}\n"
            prompt += f"\nContextual info: {context_info}"

    response = get_llm_response(
        prompt,
        format="json",
        model=model,
        provider=provider,
        api_url=api_url,
        api_key=api_key,
        messages=messages[-10:],
        npc=npc,
        team=team,
        context=context,
    )

    response_text = response.get("response", "{}")
    if isinstance(response_text, str):
        response_text = response_text.replace("```json", "").replace("```", "").strip()
        try:
            input_values = json.loads(response_text)
        except json.JSONDecodeError as e:
            if attempt < n_attempts:
                return handle_jinx_call(
                    command, 
                    jinx_name, 
                    jinxes,
                    model=model, 
                    provider=provider, 
                    api_url=api_url, 
                    api_key=api_key,
                    npc=npc, 
                    team=team, 
                    messages=messages, 
                    stream=stream,
                    context=f"Previous attempt failed to parse JSON: {e}. Raw: {response_text}",
                    extra_globals=extra_globals, previous_output=previous_output,
                    n_attempts=n_attempts, 
                    attempt=attempt + 1,
                )
            return {"output": f"Error extracting inputs for jinx '{jinx_name}'", "messages": messages}
    elif isinstance(response_text, dict):
        input_values = response_text
    else:
        input_values = {}

    missing = []
    for inp in jinx.inputs:
        if not isinstance(inp, dict):
            if inp not in input_values or input_values[inp] == "":
                missing.append(inp)
    if missing and attempt < n_attempts:
        return handle_jinx_call(
            command + f". Previous attempt missing inputs: {missing}. Values were: {input_values}.",
            jinx_name,
            jinxes,
            model=model,
            provider=provider,
            api_url=api_url,
            api_key=api_key,
            npc=npc, team=team,
            messages=messages,
            stream=stream,
            context=context,
            extra_globals=extra_globals,
            previous_output=previous_output,
            n_attempts=n_attempts,
            attempt=attempt + 1,
        )
    elif missing:
        return {"output": f"Missing inputs for jinx '{jinx_name}': {missing}", "messages": messages}

    print(f"[INPUTS] {json.dumps({k: str(v)[:200] for k, v in input_values.items()})}", flush=True)

    if previous_output is not None:
        input_values['previous_output'] = previous_output

    output = _execute_jinx(jinx, input_values, npc, team, messages, extra_globals)

    print(f"[RESULT] {str(output)[:300]}", flush=True)

    if isinstance(output, str) and output.startswith("Error:") and attempt < n_attempts:
        return handle_jinx_call(
            command,
            jinx_name,
            jinxes,
            model=model,
            provider=provider,
            api_url=api_url,
            api_key=api_key,
            npc=npc, team=team,
            messages=messages,
            stream=stream,
            context=f"Jinx failed: {output}. Previous inputs: {input_values}",
            extra_globals=extra_globals,
            previous_output=previous_output,
            n_attempts=n_attempts,
            attempt=attempt + 1,
        )

    stream_events = []
    if npc and hasattr(npc, 'shared_context') and npc.shared_context.get('sub_events'):
        stream_events = npc.shared_context.pop('sub_events')

    return {
        "output": output,
        "messages": messages,
        "jinx_calls": [{
            "name": jinx_name,
            "arguments": input_values,
            "result": str(output) if output else "",
        }],
        "stream_events": stream_events,
    }

def handle_action_choice(command: str,
                         action_data: dict,
                         jinxes: list,
                         model : str = None,
                         provider: str = None,
                         api_url:str = None,
                         api_key: str = None ,
                         npc: Any = None,
                         team: Any = None,
                         messages: list = None,
                         images: list = None,
                         stream: bool = False,
                         extra_globals: dict = None,
                         last_jinx_output = None,
                         step_outputs: list = None,
                         context: str = '',
                         **kwargs,
                        ):
    action_name = action_data.get("action", "answer")
    if action_name == "invoke_jinx" or 'jinx_name' in action_data:
        jname = action_data.get("jinx_name", "")
        step_context = context or ""
        if step_outputs:
            step_context += f"\nContext from previous steps: {json.dumps(step_outputs)}"

        result = handle_jinx_call(
            command,
            jname,
            jinxes,
            model=model,
            provider=provider,
            api_url=api_url,
            api_key=api_key,
            npc=npc,
            team=team,
            messages=messages,
            stream=stream,
            context=step_context,
            extra_globals=extra_globals,
            previous_output=last_jinx_output,
        )
        output = result.get("output", "")
        current_messages = result.get("messages", messages)
        jinx_calls = result.get("jinx_calls", [])
        stream_events = result.get("stream_events", [])

        if output and str(output).strip():
            content = str(output).replace('\\n', '\n').replace('\\t', '\t')
            render_markdown(f"\n⚡ {jname}:")
            lines = content.split('\n')
            if len(lines) > 50:
                render_markdown('\n'.join(lines[:25]))
                print(f"\n... ({len(lines) - 50} lines hidden) ...\n")
                render_markdown('\n'.join(lines[-25:]))
            else:
                render_markdown(content)
    elif action_name != 'answer':
        output = 'INVALID_ACTION'
        jinx_calls = []
        return {'output':output, 'messages':messages, 'jinx_calls': jinx_calls}
    else:
        response = get_llm_response(
            f"""The user asked: {command}

              Provide a direct answer. Do not reference tools or jinxes.""",
            model=model,
            provider=provider,
            api_url=api_url,
            api_key=api_key,
            messages=[],
            npc=npc,
            team=team,
            images=images,
            stream=stream,
            context=context,
            **kwargs,
        )
        output = response.get("response", "")
        current_messages = response.get("messages", messages)
        jinx_calls = []

    return {'output':output, 'messages':current_messages, 'jinx_calls': jinx_calls, 'stream_events': stream_events if 'stream_events' in locals() else []}


def check_llm_command(
    command: str,
    model: str = None,
    provider: str = None,
    api_url: str = None,
    api_key: str = None,
    npc: Any = None,
    team: Any = None,
    messages: List[Dict[str, str]] = None,
    images: list = None,
    stream=False,
    context=None,
    actions: Dict[str, Dict] = None,
    extra_globals=None,
    max_iterations: int = 5,
    jinxes: Dict = None,
    tool_capable: bool = None,
    **kwargs,
):
    """Plan and execute: decide whether to answer directly or use jinxes, then do it.

    When stream=True, returns a generator yielding (event_type, data) tuples.
    When stream=False, returns a dict.
    """
    if messages is None:
        messages = []

    if jinxes is None:
        jinxes = _get_jinxes(npc, team)

    if not jinxes:
        print('no jinxes detected')

        response = get_llm_response(
            command,
            model=model,
            provider=provider,
            api_url=api_url,
            api_key=api_key,
            messages=messages[-10:],
            npc=npc,
            team=team,
            images=images,
            stream=False,
            context=context,
            **kwargs,
        )
        messages.append({"role": "user", "content": command})
        out = response.get("response", "")
        if out and isinstance(out, str):
            messages.append({"role": "assistant", "content": out})
        return {"messages": messages, "output": out, "usage": response.get("usage", {})}


    prompt = f"""

    
          A user submitted this request: {command}
    
      
          Determine the nature of the user's request:
      
          1. Should a jinx be invoked to fulfill the request? A jinx is a jinja-template execution script.
      
          2. Is it a general question that requires an informative answer or a highly specific question that
              requires information on the web?
      
                        
              Use jinxes when it is obvious that the answer needs to be as up-to-date as possible. For example,
                  a question about where mount everest is does not necessarily need to be answered by a jinx call or an agent pass.
          
              If a user asks to explain the plot of the aeneid, this can be answered without a jinx call or agent pass.
              
              If a user were to ask for the current weather in tokyo or the current price of bitcoin or who the mayor of a city is,
                  then a jinx call is appropriate.
          
              If the user wants you to read a file, it must use a jinx to read the file.
          
              If the user asks you to edit a file, you must use a jinx to edit the file.
          
              If the user asks you to take a screenshot, you must to use a jinx to take the screenshot if available
              
              If a user asks you to search or to take a screenshot or to open a program or to write a program most likely it is
              appropriate to use a jinx. 


              remember, in your output, return only the action sequence. do not include and leading ```json or other markdown tags.
              
              """

    if messages:
        prompt += f"\nRecent conversation: {messages[-5:]}"

    response = get_llm_response(
        prompt,
        model=model,
        provider=provider,
        api_url=api_url,
        api_key=api_key,
        npc=npc,
        team=team,
        format="json",
        messages=[],
        context=context,
        **kwargs,
    )

    actions = response.get("response", {})
    print(actions)

    if not isinstance(actions,list) and isinstance(actions,dict):
        actions = [actions]

    def _execute():
        step_outputs = []
        all_jinx_calls = []
        current_messages = messages.copy()
        last_jinx_output = None

        for i, action_data in enumerate(actions):
            render_markdown(f"- {action_data}")
            action_name = action_data.get("action", "")
            is_jinx = action_name == "invoke_jinx" or "jinx_name" in action_data
            jname = action_data.get("jinx_name", "")

            if stream and is_jinx and jname:
                yield ('tool_start', {'name': jname, 'args': action_data.get('inputs', {})})

            action_result = handle_action_choice(
                         command,
                         action_data,
                         jinxes,
                         model = model,
                         provider = provider,
                         api_url = api_url,
                         api_key = api_key,
                         npc = npc,
                         team = team,
                         messages = current_messages,
                         stream = stream,
                         extra_globals = extra_globals,
                         last_jinx_output = last_jinx_output,
                         step_outputs = step_outputs,
                         context = context,
                         **kwargs,
            )
            current_messages = action_result.get('messages', [])
            output = action_result.get('output', [])
            all_jinx_calls.extend(action_result.get('jinx_calls', []))
            if output == 'INVALID_ACTION':
                for evt in check_llm_command(
                    f"""In the previous attempt, the correct action name was not provided and a jinx could not be deciphered. only select from available jinxes.
                Original request: {command}""",
                    model=model, provider=provider, api_url=api_url, api_key=api_key,
                    npc=npc, team=team, messages=messages, stream=stream,
                    context=context, extra_globals=extra_globals, **kwargs,
                ):
                    yield evt
                return

            if stream:
                for evt in action_result.get('stream_events', []):
                    yield evt
                if is_jinx and jname:
                    yield ('tool_result', {'name': jname, 'result': output, 'args': action_data.get('inputs', {})})
                else:
                    yield ('content', output)

            step_outputs.append(output)
            last_jinx_output = output

        final_output = step_outputs[0] if len(step_outputs) == 1 else "\n".join(str(o) for o in step_outputs)
        yield ('result', {
            "messages": current_messages,
            "output": final_output,
            "usage": response.get("usage", {}),
            "jinx_calls": all_jinx_calls,
        })

    if stream:
        return _execute()

    result = None
    for event_type, data in _execute():
        if event_type == 'result':
            result = data
    return result or {"messages": messages, "output": "", "jinx_calls": []}


def identify_groups(
    facts: List[str],
    model,
    provider,
    npc =  None,
    context: str = None,
    **kwargs
) -> List[str]:
    """Identify natural groups from a list of facts"""

        
    prompt = """What are the main groups these facts could be organized into?
    Express these groups in plain, natural language.

    For example, given:
        - User enjoys programming in Python
        - User works on machine learning projects
        - User likes to play piano
        - User practices meditation daily

    You might identify groups like:
        - Programming
        - Machine Learning
        - Musical Interests
        - Daily Practices

    Return a JSON object with the following structure:
        `{
            "groups": ["list of group names"]
        }`

    Return only the JSON object. Do not include any additional markdown formatting or
    leading json characters.
    """

    response = get_llm_response(
        prompt + f"\n\nFacts: {json.dumps(facts)}",
        model=model,
        provider=provider,
        format="json",
        npc=npc,
        context=context,
        
        **kwargs
    )
    return response["response"]["groups"]

def get_related_concepts_multi(node_name: str, 
                               node_type: str, 
                               all_concept_names, 
                               model: str = None,
                               provider: str = None,
                               npc=None,
                               context : str = None, 
                               **kwargs):
    """Links any node (fact or concept) to ALL relevant concepts in the entire ontology."""
    json_format = 'Respond with JSON: {"related_concepts": ["Concept A", "Concept B", ...]}'
    prompt = f"""
    Which of the following concepts from the entire ontology relate to the given {node_type}?
    Select all that apply, from the most specific to the most abstract.

    {node_type.capitalize()}: "{node_name}"

    Available Concepts:
    {json.dumps(all_concept_names, indent=2)}

    {json_format}
    """
    response = get_llm_response(prompt, 
                                model=model, 
                                provider=provider, 
                                format="json", 
                                npc=npc,
                                context=context, 
                                **kwargs)
    return response["response"].get("related_concepts", [])

def assign_groups_to_fact(
    fact: str,
    groups: List[str],
    model = None,
    provider = None,
    npc = None, 
    context: str = None,
    **kwargs
) -> Dict[str, List[str]]:
    """Assign facts to the identified groups"""
    json_format = """Return a JSON object with the following structure:
        {
            "groups": ["list of group names"]
        }

    Do not include any additional markdown formatting or leading json characters."""

    prompt = f"""Given this fact, assign it to any relevant groups.

    A fact can belong to multiple groups if it fits.

    Here is the fact: {fact}

    Here are the groups: {groups}

    {json_format}
    """

    response = get_llm_response(
        prompt,
        model=model,
        provider=provider,
        format="json",
        npc=npc,
        context=context,
        **kwargs
    )
    return response["response"]

def generate_group_candidates(
    items: List[str],
    item_type: str,
    model: str = None,
    provider: str =None,
    npc = None,
    context: str = None,
    n_passes: int = 3,
    subset_size: int = 10, 
    **kwargs
) -> List[str]:
    """Generate candidate groups for items (facts or groups) based on core semantic meaning."""
    all_candidates = []
    
    for pass_num in range(n_passes):
        if len(items) > subset_size:
            item_subset = random.sample(items, min(subset_size, len(items)))
        else:
            item_subset = items
        
      
        prompt = f"""From the following {item_type}, identify specific and relevant conceptual groups.
        Think about the core subject or entity being discussed.
        
        GUIDELINES FOR GROUP NAMES:
        1.  **Prioritize Specificity:** Names should be precise and directly reflect the content.
        2.  **Favor Nouns and Noun Phrases:** Use descriptive nouns or noun phrases.
        3.  **AVOID:**
            *   Gerunds (words ending in -ing when used as nouns, like "Understanding", "Analyzing", "Processing"). If a gerund is unavoidable, try to make it a specific action (e.g., "User Authentication Module" is better than "Authenticating Users").
            *   Adverbs or descriptive adjectives that don't form a core part of the subject's identity (e.g., "Quickly calculating", "Effectively managing").
            *   Overly generic terms (e.g., "Concepts", "Processes", "Dynamics", "Mechanics", "Analysis", "Understanding", "Interactions", "Relationships", "Properties", "Structures", "Systems", "Frameworks", "Predictions", "Outcomes", "Effects", "Considerations", "Methods", "Techniques", "Data", "Theoretical", "Physical", "Spatial", "Temporal").
        4.  **Direct Naming:** If an item is a specific entity or action, it can be a group name itself (e.g., "Earth", "Lamb Shank Braising", "World War I").
        
        EXAMPLE:
        Input {item_type.capitalize()}: ["Self-intersection shocks drive accretion disk formation.", "Gravity stretches star into stream.", "Energy dissipation in shocks influences capture fraction."]
        Desired Output Groups: ["Accretion Disk Formation (Self-Intersection Shocks)", "Stellar Tidal Stretching", "Energy Dissipation from Shocks"]
        
        ---
        
        Now, analyze the following {item_type}:
        {item_type.capitalize()}: {json.dumps(item_subset)}
        
        Return a JSON object:
        """ + '{"groups": ["list of specific, precise, and relevant group names"]}'
      
        
        response = get_llm_response(
            prompt,
            model=model,
            provider=provider,
            format="json",
            npc=npc,
            context=context,
            **kwargs
        )
        
        candidates = response["response"].get("groups", [])
        all_candidates.extend(candidates)

    return list(set(all_candidates))

def remove_idempotent_groups(
    group_candidates: List[str],
    model: str = None,
    provider: str =None,
    npc = None, 
    context : str = None,
    **kwargs: Any
) -> List[str]:
    """Remove groups that are essentially identical in meaning, favoring specificity and direct naming, and avoiding generic structures."""
    
    prompt = f"""Compare these group names. Identify and list ONLY the groups that are conceptually distinct and specific.
    
    GUIDELINES FOR SELECTING DISTINCT GROUPS:
    1.  **Prioritize Specificity and Direct Naming:** Favor precise nouns or noun phrases that directly name the subject.
    2.  **Prefer Concrete Entities/Actions:** If a name refers to a specific entity or action (e.g., "Earth", "Sun", "Water", "France", "User Authentication Module", "Lamb Shank Braising", "World War I"), keep it if it's distinct.
    3.  **Rephrase Gerunds:** If a name uses a gerund (e.g., "Understanding TDEs"), rephrase it to a noun or noun phrase (e.g., "Tidal Disruption Events").
    4.  **AVOID OVERLY GENERIC TERMS:** Do NOT use very broad or abstract terms that don't add specific meaning. Examples to avoid: "Concepts", "Processes", "Dynamics", "Mechanics", "Analysis", "Understanding", "Interactions", "Relationships", "Properties", "Structures", "Systems", "Frameworks", "Predictions", "Outcomes", "Effects", "Considerations", "Methods", "Techniques", "Data", "Theoretical", "Physical", "Spatial", "Temporal". If a group name seems overly generic or abstract, it should likely be removed or refined.
    5.  **Similarity Check:** If two groups are very similar, keep the one that is more descriptive or specific to the domain.

    EXAMPLE 1:
    Groups: ["Accretion Disk Formation", "Accretion Disk Dynamics", "Formation of Accretion Disks"]
    Distinct Groups: ["Accretion Disk Formation", "Accretion Disk Dynamics"] 

    EXAMPLE 2:
    Groups: ["Causes of Events", "Event Mechanisms", "Event Drivers"]
    Distinct Groups: ["Event Causation", "Event Mechanisms"] 

    EXAMPLE 3:
    Groups: ["Astrophysics Basics", "Fundamental Physics", "General Science Concepts"]
    Distinct Groups: ["Fundamental Physics"] 

    EXAMPLE 4:
    Groups: ["Earth", "The Planet Earth", "Sun", "Our Star"]
    Distinct Groups: ["Earth", "Sun"]
    
    EXAMPLE 5:
    Groups: ["User Authentication Module", "Authentication System", "Login Process"]
    Distinct Groups: ["User Authentication Module", "Login Process"]
    
    ---
    
    Now, analyze the following groups:
    Groups: {json.dumps(group_candidates)}
    
    Return JSON:
    """ + '{"distinct_groups": ["list of specific, precise, and distinct group names to keep"]}'
    
    response = get_llm_response(
        prompt,
        model=model,
        provider=provider,
        format="json",
        npc=npc,
        context=context,
        **kwargs
    )
    
    return response["response"]["distinct_groups"]

def breathe(
    messages: List[Dict[str, str]],
    model: str = None,
    provider: str = None, 
    npc =  None,
    context: str = None,
    **kwargs: Any
) -> Dict[str, Any]:
    """Condense the conversation context into a small set of key extractions."""
    if not messages:
        return {"output": {}, "messages": []}

    if 'stream' in kwargs:
        kwargs['stream'] = False
    conversation_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

    prompt = f'''
    Read the following conversation:

    {conversation_text}

    ''' +'''

    Now identify the following items:

    1. The high level objective
    2. The most recent task
    3. The accomplishments thus far
    4. The failures thus far

    Return a JSON like so:

    {
        "high_level_objective": "the overall goal so far for the user", 
        "most_recent_task": "The currently ongoing task", 
        "accomplishments": ["accomplishment1", "accomplishment2"], 
        "failures": ["falures1", "failures2"], 
    }

    '''

    
    result = get_llm_response(prompt, 
                           model=model, 
                           provider=provider, 
                           npc=npc, 
                           context=context, 
                           format='json', 
                           **kwargs)

    res = result.get('response', {})
    if isinstance(res, str):
        raise Exception
    format_output = f"""Here is a summary of the previous session. 
    The high level objective was: {res.get('high_level_objective')} \n The accomplishments were: {res.get('accomplishments')}, 
    the failures were: {res.get('failures')} and the most recent task was: {res.get('most_recent_task')}   """
    return {'output': format_output,
            'summary': res,
            'messages': [
                         {
                           'content': format_output,
                           'role': 'assistant'}
                           ]
                          }
def abstract(groups, 
             model, 
             provider, 
             npc=None,
             context: str = None, 
             **kwargs):
    """
    Create more abstract terms from groups.
    """
    sample_groups = random.sample(groups, min(len(groups), max(3, len(groups) // 2)))
    
    groups_text_for_prompt = "\n".join([f'- "{g["name"]}"' for g in sample_groups])

    prompt = f"""
        Create more abstract categories from this list of groups.

        Groups:
        {groups_text_for_prompt}

        You will create higher-level concepts that interrelate between the given groups. 

        Create abstract categories that encompass multiple related facts, but do not unnecessarily combine facts with conjunctions. For example, do not try to combine "characters", "settings", and "physical reactions" into a
        compound group like "Characters, Setting, and Physical Reactions". This kind of grouping is not productive and only obfuscates true abstractions. 
        For example, a group that might encompass the three aforermentioned names might be "Literary Themes" or "Video Editing Functionis", depending on the context.
        Your aim is to abstract, not to just arbitrarily generate associations. 

        Group names should never be more than two words. They should not contain gerunds. They should never contain conjunctions like "AND" or "OR".
        Generate no more than 5 new concepts and no fewer than 2. 

        Respond with JSON:
        """ + '{"groups": [{"name": "abstract category name"}]}'

    response = get_llm_response(prompt,
                                model=model,
                                provider=provider,
                                format="json",
                                npc=npc,
                                context=context,
                                **kwargs)

    return response["response"].get("groups", [])

CONVERSATION_RULES = """
ABSOLUTE RULES — violations return EMPTY array:
1. NEVER attribute statements to speakers. No "the user", "the assistant",
   "I asked", "they said". Strip all speaker/agent attribution completely.
2. NEVER describe interaction mechanics: opening panes, clicking buttons,
   invoking functions, loading pages, API calls.
3. NEVER extract greetings, pleasantries, apologies, or meta-chat.
4. NEVER output generic truisms that lack specific context (e.g. "testing is important").
5. If the text is purely commands, greetings, or operational chatter, return EMPTY.

EXTRACT ONLY facts containing specific technical substance:
- Concrete implementation decisions with rationale (WHAT was changed, WHY, in WHAT system)
- Quantified observations with metrics or thresholds
- Causal relationships and their conditions
- Architectural constraints or requirements
- Error patterns and their specific resolutions
- Domain-specific methodologies with their applicability conditions
"""

FILE_RULES = """
ABSOLUTE RULES — violations return EMPTY array:
1. NEVER output generic truisms ("code should be tested", "imports are necessary").
2. NEVER describe trivial syntax ("this file uses Python functions").
3. NEVER restate the filename as a fact.
4. If the file is purely boilerplate, auto-generated, or contains no
   substantive logic, return EMPTY.

EXTRACT facts containing specific technical substance:
- Purpose and responsibilities of modules/classes/functions (WHAT it does)
- Dependencies and integrations with other systems (WHAT it connects to)
- Configuration values and their semantic meaning (WHAT each setting controls)
- Algorithms, data structures, or protocols implemented (HOW it works)
- Design constraints, performance characteristics, or limitations (WHY it works this way)
- API surfaces: endpoints, arguments, return types, error conditions
- Security or correctness considerations
"""


def get_facts(content_text,
              model=None,
              provider=None,
              npc=None,
              context: str = None,
              attempt_number=1,
              n_attempts=3,
              rules=None,
              **kwargs):
    """Extract facts from content text.

    ``rules`` is injected directly into the prompt after the base
    instructions. Callers pass ``CONVERSATION_RULES`` or
    ``FILE_RULES`` (or any custom rules string).
    """

    rules = rules or FILE_RULES

    full_context = ""
    if npc and hasattr(npc, "context") and npc.context:
        full_context = str(npc.context)
    if context:
        if full_context:
            full_context += "\n\n" + str(context)
        else:
            full_context = str(context)

    instruction = f"""
    Extract substantive, self-contained domain facts from the following content.

    A valid fact is a rich statement that captures non-obvious technical knowledge,
    including the context needed to understand what system or domain it pertains to.
    Facts should preserve nuance: uncertainty, conditions, trade-offs, and causal
    relationships. They should NOT be shallow one-liners like "X is Y".

    {rules}

    YOUR TASK — extract up to 5 facts from:
    "{content_text}"

    If no facts meet the criteria, return an EMPTY facts array.
    Respond with JSON:
    """

    examples = """

    EXAMPLES:

    Input: "We were seeing OOMs during training on the 40GB A100s when batch size hit 64.
    Turns out the gradient checkpointing wasn't being applied to the cross-attention layers.
    Once we wrapped those in checkpoint() too, we could push to batch size 96 without issues."

    Output:
    {
      "facts": [
        {
          "statement": "In the transformer model being trained, gradient checkpointing was initially missing from cross-attention layers, causing out-of-memory errors on 40GB A100 GPUs at batch size 64",
          "source_text": "the gradient checkpointing wasn't being applied to the cross-attention layers",
          "type": "explicit"
        },
        {
          "statement": "Applying gradient checkpointing to cross-attention layers increased feasible batch size from 64 to 96 on the same 40GB A100 hardware without OOM errors",
          "source_text": "Once we wrapped those in checkpoint() too, we could push to batch size 96 without issues",
          "type": "inferred"
        }
      ]
    }

    Input: "Can you open the knowledge graph editor? I want to see if the nodes show up."

    Output:
    {
      "facts": []
    }

    Input: "Hello! How are you today? I hope the weather is nice."

    Output:
    {
      "facts": []
    }
    """

    json_fmt = '{"facts": [{"statement": "rich self-contained domain claim with full contextual specificity", "source_text": "relevant excerpt", "type": "explicit or inferred"}]}'

    prompt = instruction + examples + json_fmt

    response = get_llm_response(prompt,
                                model=model,
                                provider=provider,
                                npc=npc,
                                format="json",
                                context=full_context,
                                **kwargs)
    try:
    
        if len(response.get("response", {}).get("facts", [])) == 0 and attempt_number < n_attempts:
            print(f"  Attempt {attempt_number} to extract facts yielded no results. Retrying...")
            return get_facts(content_text,
                             model=model,
                             provider=provider,
                             npc=npc,
                             context=full_context,
                             attempt_number=attempt_number+1,
                             n_attempts=n_attempts,
                             rules=rules,
                             **kwargs)
    except AttributeError:
        return get_facts(content_text,
                 model=model,
                 provider=provider,
                 npc=npc,
                 context=full_context,
                 attempt_number=attempt_number+1,
                 n_attempts=n_attempts,
                 rules=rules,
                 **kwargs)
    return response["response"].get("facts", [])

        

def zoom_in(facts, 
            model= None,
            provider=None, 
            npc=None,
            context: str = None, 
            attempt_number: int = 1,
            n_attempts=3,            
            **kwargs):
    """Infer new implied facts from existing facts"""
    valid_facts = []
    for fact in facts:
        if isinstance(fact, dict) and 'statement' in fact:
            valid_facts.append(fact)
    if not valid_facts:
        return []     

    fact_lines = []
    for fact in valid_facts:
        fact_lines.append(f"- {fact['statement']}")
    facts_text = "\n".join(fact_lines)
    
    prompt = f"""
    Look at these facts and infer new implied facts:

    {facts_text}

    What other facts can be reasonably inferred from these?
    """ +"""
    Respond with JSON:
    {
        "implied_facts": [
            {
                "statement": "new implied fact",
                "inferred_from": ["which facts this comes from"]
            }
        ]
    }
    """
    
    response = get_llm_response(prompt, 
                                model=model, 
                                provider=provider, 
                                format="json", 
                                context=context,
                                npc=npc,
                                **kwargs)

    facts =  response.get("response", {}).get("implied_facts", [])
    if len(facts) == 0:
        return zoom_in(valid_facts, 
                       model=model, 
                       provider=provider, 
                       npc=npc,
                       context=context,
                       attempt_number=attempt_number+1,
                       n_tries=n_attempts,
                       **kwargs)
    return facts
def generate_groups(facts, 
                    model=None,
                    provider=None,
                    npc=None,
                    context: str =None, 
                    **kwargs):
    """Generate conceptual groups for facts"""
    
    facts_text = "\n".join([f"- {fact['statement']}" for fact in facts])
    
    prompt = f"""
    Generate conceptual groups for this group off facts:

    {facts_text}

    Create categories that encompass multiple related facts, but do not unnecessarily combine facts with conjunctions. 
    
    Your aim is to generalize commonly occurring ideas into groups, not to just arbitrarily generate associations. 
    Focus on the key commonly occurring items and expresions.     

    Group names should never be more than two words. They should not contain gerunds. They should never contain conjunctions like "AND" or "OR".
    Respond with JSON:
    """ + '{"groups": [{"name": "group name"}]}'

    response = get_llm_response(prompt,
                                model=model,
                                provider=provider,
                                format="json",
                                context=context,
                                npc=npc,
                                **kwargs)

    return response["response"].get("groups", [])

def remove_redundant_groups(groups, 
                            model=None,
                            provider=None,
                            npc=None,
                            context: str = None,
                            **kwargs):
    """Remove redundant groups"""
    
    groups_text = "\n".join([f"- {g['name']}" for g in groups])
    
    prompt = f"""
    Remove redundant groups from this list:

    {groups_text}

    Merge similar groups and keep only distinct concepts.
    Create abstract categories that encompass multiple related facts, but do not unnecessarily combine facts with conjunctions. For example, do not try to combine "characters", "settings", and "physical reactions" into a
    compound group like "Characters, Setting, and Physical Reactions". This kind of grouping is not productive and only obfuscates true abstractions. 
    For example, a group that might encompass the three aforermentioned names might be "Literary Themes" or "Video Editing Functionis", depending on the context.
    Your aim is to abstract, not to just arbitrarily generate associations. 

    Group names should never be more than two words. They should not contain gerunds. They should never contain conjunctions like "AND" or "OR".

    Respond with JSON:
    """ + '{"groups": [{"name": "final group name"}]}'

    response = get_llm_response(prompt,
                                model=model,
                                provider=provider,
                                npc=npc,
                                context=context,
                                **kwargs)

    return response["response"].get("groups", [])

def prune_fact_subset_llm(fact_subset, 
                          concept_name, 
                          model=None,
                          provider=None,
                          npc=None,
                          context : str = None,
                          **kwargs):
    """Identifies redundancies WITHIN a small, topically related subset of facts."""
    print(f"  Step Sleep-A: Pruning fact subset for concept '{concept_name}'...")
    

    prompt = f"""
    The following facts are all related to the concept "{concept_name}".
    Review ONLY this subset and identify groups of facts that are semantically identical.
    Return only the set of facts that are semantically distinct, and archive the rest.

    Fact Subset: {json.dumps(fact_subset, indent=2)}

    Return a json list of groups
    """ + '{"refined_facts": [fact1, fact2, fact3, ...]}'
    response = get_llm_response(prompt, 
                                model=model, 
                                provider=provider, 
                                npc=None,
                                format="json", 
                                context=context)
    return response['response'].get('refined_facts', [])

def consolidate_facts_llm(new_fact, 
                          existing_facts, 
                          model, 
                          provider, 
                          npc=None,
                          context: str =None,
                          **kwargs):
    """
    Uses an LLM to decide if a new fact is novel or redundant.
    """
    prompt = f"""
        Analyze the "New Fact" in the context of the "Existing Facts" list.
        Your task is to determine if the new fact provides genuinely new information or if it is essentially a repeat or minor rephrasing of information already present.

        New Fact:
        "{new_fact['statement']}"

        Existing Facts:
        {json.dumps([f['statement'] for f in existing_facts], indent=2)}

        Possible decisions:
        - 'novel': The fact introduces new, distinct information not covered by the existing facts.
        - 'redundant': The fact repeats information already present in the existing facts.

        Respond with a JSON object:
        """ + '{"decision": "novel or redundant", "reason": "A brief explanation for your decision."}'
    response = get_llm_response(prompt,
                                model=model, 
                                provider=provider, 
                                format="json", 
                                npc=npc,
                                context=context,
                                **kwargs)
    return response['response']

def get_related_facts_llm(new_fact_statement, 
                          existing_fact_statements, 
                          model = None, 
                          provider = None,
                          npc = None, 
                          attempt_number = 1,
                          n_attempts = 3,
                          context='', 
                          **kwargs):
    """Identifies which existing facts are causally or thematically related to a new fact."""
    prompt = f"""
    A new fact has been learned: "{new_fact_statement}"

    Which of the following existing facts are directly related to it (causally, sequentially, or thematically)?
    Select only the most direct and meaningful connections.

    Existing Facts:
    {json.dumps(existing_fact_statements, indent=2)}

    Respond with JSON:
    """ + '{"related_facts": ["statement of a related fact", ...]}'
    response = get_llm_response(prompt,
                                model=model, 
                                provider=provider, 
                                format="json", 
                                npc=npc,
                                context=context,
                                **kwargs)   
    if attempt_number <= n_attempts:
        if not response["response"].get("related_facts", []):
            print(f"  Attempt {attempt_number} to find related facts yielded no results. Retrying...")
            return get_related_facts_llm(new_fact_statement, 
                                           existing_fact_statements, 
                                           model=model, 
                                           provider=provider, 
                                           npc=npc,
                                           attempt_number=attempt_number+1,
                                           n_attempts=n_attempts,
                                           context=context,
                                           **kwargs)    

    return response["response"].get("related_facts", [])

def find_best_link_concept_llm(candidate_concept_name, 
                               existing_concept_names, 
                               model = None,
                               provider = None,
                               npc = None,
                               context: str = None,
                               **kwargs   ):
    """
    Finds the best existing concept to link a new candidate concept to.
    This prompt now uses neutral "association" language.
    """
    prompt = f"""
    Here is a new candidate concept: "{candidate_concept_name}"
    
    Which of the following existing concepts is it most closely related to? The relationship could be as a sub-category, a similar idea, or a related domain.

    Existing Concepts:
    {json.dumps(existing_concept_names, indent=2)}

    Respond with the single best-fit concept to link to from the list, or respond with "none" if it is a genuinely new root idea.
    """ + '{"best_link_concept": "The single best concept name OR none"}'
    response = get_llm_response(prompt, 
                                model=model, 
                                provider=provider, 
                                format="json", 
                                npc=npc,
                                context=context,
                                **kwargs)
    return response['response'].get('best_link_concept')

def asymptotic_freedom(parent_concept, 
                       supporting_facts, 
                       model=None, 
                       provider=None, 
                       npc = None,
                       context: str = None, 
                       **kwargs):
    """Given a concept and its facts, proposes an intermediate layer of sub-concepts."""
    print(f"  Step Sleep-B: Attempting to deepen concept '{parent_concept['name']}'...")
    fact_statements = []
    for f in supporting_facts:
        fact_statements.append(f['statement'])
        
    prompt = f"""
    The concept "{parent_concept['name']}" is supported by many diverse facts.
    Propose a layer of 2-4 more specific sub-concepts to better organize these facts.
    These new concepts will exist as nodes that link to "{parent_concept['name']}".

    Supporting Facts: {json.dumps(fact_statements, indent=2)}
    Respond with JSON:
    """ + '{"new_sub_concepts": ["sub_layer1", "sub_layer2"]}'
    response = get_llm_response(prompt, 
                                model=model, 
                                provider=provider,
                                format="json", 
                                context=context, npc=npc,
                                **kwargs)
    return response['response'].get('new_sub_concepts', [])

def bootstrap(
    prompt: str,
    model: str = None,
    provider: str = None,
    npc: Any = None,
    team: Any = None,
    sample_params: Dict[str, Any] = None,
    sync_strategy: str = "consensus",
    context: str = None,
    n_samples: int = 3,
    **kwargs
) -> Dict[str, Any]:
    """Bootstrap by sampling multiple agents from team or varying parameters"""
    
    if team and hasattr(team, 'npcs') and len(team.npcs) >= n_samples:
      
        sampled_npcs = list(team.npcs.values())[:n_samples]
        results = []
        
        for i, agent in enumerate(sampled_npcs):
            response = get_llm_response(
                f"Sample {i+1}: {prompt}\nContext: {context}",
                npc=agent,
                context=context,
                **kwargs
            )
            results.append({
                'agent': agent.name,
                'response': response.get("response", "")
            })
    else:
      
        if sample_params is None:
            sample_params = {"temperature": [0.3, 0.7, 1.0]}
        
        results = []
        for i in range(n_samples):
            temp = sample_params.get('temperature', [0.7])[i % len(sample_params.get('temperature', [0.7]))]
            response = get_llm_response(
                f"Sample {i+1}: {prompt}\nContext: {context}",
                model=model,
                provider=provider,
                npc=npc,
                temperature=temp,
                context=context,
                **kwargs
            )
            results.append({
                'variation': f'temp_{temp}',
                'response': response.get("response", "")
            })
    
  
    response_texts = [r['response'] for r in results]
    return synthesize(response_texts, sync_strategy, model, provider, npc or (team.forenpc if team else None), context)

def harmonize(
    prompt: str,
    items: List[str],
    model: str = None,
    provider: str = None,
    npc: Any = None,
    team: Any = None,
    harmony_rules: List[str] = None,
    context: str = None,
    agent_roles: List[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Harmonize using multiple specialized agents"""
    
    if team and hasattr(team, 'npcs'):
      
        available_agents = list(team.npcs.values())
        
        if agent_roles:
          
            selected_agents = []
            for role in agent_roles:
                matching_agent = next((a for a in available_agents if role.lower() in a.name.lower() or role.lower() in a.primary_directive.lower()), None)
                if matching_agent:
                    selected_agents.append(matching_agent)
            agents_to_use = selected_agents or available_agents[:len(items)]
        else:
          
            agents_to_use = available_agents[:min(len(items), len(available_agents))]
        
        harmonized_results = []
        for i, (item, agent) in enumerate(zip(items, agents_to_use)):
            harmony_prompt = f"""Harmonize this element: {item}
Task: {prompt}
Rules: {', '.join(harmony_rules or ['maintain_consistency'])}
Context: {context}
Your role in harmony: {agent.primary_directive}"""
            
            response = get_llm_response(
                harmony_prompt,
                npc=agent,
                context=context,
                **kwargs
            )
            harmonized_results.append({
                'agent': agent.name,
                'item': item,
                'harmonized': response.get("response", "")
            })
        
      
        coordinator = team.get_forenpc() if team else npc
        synthesis_prompt = f"""Synthesize these harmonized elements:
{chr(10).join([f"{r['agent']}: {r['harmonized']}" for r in harmonized_results])}
Create unified harmonious result."""
        
        return get_llm_response(synthesis_prompt, npc=coordinator, context=context, **kwargs)
    
    else:
      
        items_text = chr(10).join([f"{i+1}. {item}" for i, item in enumerate(items)])
        harmony_prompt = f"""Harmonize these items: {items_text}
Task: {prompt}
Rules: {', '.join(harmony_rules or ['maintain_consistency'])}
Context: {context}"""
        
        return get_llm_response(harmony_prompt, model=model, provider=provider, npc=npc, context=context, **kwargs)

def orchestrate(
    prompt: str,
    items: List[str],
    model: str = None,
    provider: str = None,
    npc: Any = None,
    team: Any = None,
    workflow: str = "sequential_coordination",
    context: str = None,
    **kwargs
) -> Dict[str, Any]:
    """Orchestrate using team.orchestrate method"""
    
    if team and hasattr(team, 'orchestrate'):
      
        orchestration_request = f"""Orchestrate workflow: {workflow}
Task: {prompt}
Items: {chr(10).join([f'- {item}' for item in items])}
Context: {context}"""
        
        return team.orchestrate(orchestration_request)
    
    else:
      
        items_text = chr(10).join([f"{i+1}. {item}" for i, item in enumerate(items)])
        orchestrate_prompt = f"""Orchestrate using {workflow}:
Task: {prompt}
Items: {items_text}
Context: {context}"""
        
        return get_llm_response(orchestrate_prompt, model=model, provider=provider, npc=npc, context=context, **kwargs)

def spread_and_sync(
    prompt: str,
    variations: List[str],
    model: str = None,
    provider: str = None,
    npc: Any = None,
    team: Any = None,
    sync_strategy: str = "consensus",
    context: str = None,
    **kwargs
) -> Dict[str, Any]:
    """Spread across agents/variations then sync with distribution analysis"""
    
    if team and hasattr(team, 'npcs') and len(team.npcs) >= len(variations):
      
        agents = list(team.npcs.values())[:len(variations)]
        results = []
        
        for variation, agent in zip(variations, agents):
            variation_prompt = f"""Analyze from {variation} perspective:
Task: {prompt}
Context: {context}
Apply your expertise with {variation} approach."""
            
            response = get_llm_response(variation_prompt, npc=agent, context=context, **kwargs)
            results.append({
                'agent': agent.name,
                'variation': variation,
                'response': response.get("response", "")
            })
    else:
      
        results = []
        agent = npc or (team.get_forenpc() if team else None)
        
        for variation in variations:
            variation_prompt = f"""Analyze from {variation} perspective:
Task: {prompt}
Context: {context}"""
            
            response = get_llm_response(variation_prompt, model=model, provider=provider, npc=agent, context=context, **kwargs)
            results.append({
                'variation': variation,
                'response': response.get("response", "")
            })
    
  
    response_texts = [r['response'] for r in results]
    return synthesize(response_texts, sync_strategy, model, provider, npc or (team.get_forenpc() if team else None), context)

def criticize(
    prompt: str,
    model: str = None,
    provider: str = None,
    npc: Any = None,
    team: Any = None,
    context: str = None,
    **kwargs
) -> Dict[str, Any]:
    """Provide critical analysis and constructive criticism"""
    critique_prompt = f"""
    Provide a critical analysis and constructive criticism of the following:
    {prompt}
    
    Focus on identifying weaknesses, potential improvements, and alternative approaches.
    Be specific and provide actionable feedback.
    """
    
    return get_llm_response(
        critique_prompt,
        model=model,
        provider=provider,
        npc=npc,
        team=team,
        context=context,
        **kwargs
    )
def synthesize(
    prompt: str,
    model: str = None,
    provider: str = None,
    npc: Any = None,
    team: Any = None,
    context: str = None,
    **kwargs
) -> Dict[str, Any]:
    """Synthesize information from multiple sources or perspectives"""
    
    responses = kwargs.get('responses', [prompt])
    sync_strategy = kwargs.get('sync_strategy', 'consensus')
    
    if len(responses) > 1:
        synthesis_prompt = f"""Synthesize these multiple perspectives:
        
        {chr(10).join([f'Response {i+1}: {r}' for i, r in enumerate(responses)])}
        
        Synthesis strategy: {sync_strategy}
        Context: {context}
        
        Create a coherent synthesis that incorporates key insights from all perspectives."""
    else:
        synthesis_prompt = f"""Refine and synthesize this content:
        
        {responses[0]}
        
        Context: {context}
        
        Create a clear, concise synthesis that captures the essence of the content."""
    
    return get_llm_response(
        synthesis_prompt,
        model=model,
        provider=provider,
        npc=npc,
        team=team,
        context=context,
        **kwargs
    )
