"""
npcpy.streaming — Reusable streaming core for NPC chat and tool-agent flows.

Extracts the duplicated SSE streaming, chunk parsing, message cleaning,
tool resolution, and tool execution logic that was copy-pasted between
npcpy/serve.py and app-specific server wrappers (e.g. Lavanzaro).

Usage:
    from npcpy.streaming import (
        StreamConfig, StreamEvent,
        clean_messages_for_llm,
        ensure_system_prompt,
        parse_stream_chunk,
        format_sse_event,
        resolve_npc_tools,
        execute_tool,
        create_chat_stream,
        create_tool_agent_stream,
    )
"""

import json
import time
import traceback
import datetime
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

from npcpy.llm_funcs import get_llm_response, check_llm_command
from npcpy.npc_compiler import NPC, Team


@dataclass
class StreamConfig:
    """Configuration for a streaming request."""
    npc: Optional[Any] = None
    team: Optional[Any] = None
    model: str = ""
    provider: str = ""
    messages: List[dict] = field(default_factory=list)
    commandstr: str = ""
    temperature: float = 0.7
    params: Optional[dict] = None
    attachments: List = field(default_factory=list)
    images: List = field(default_factory=list)
    disable_thinking: bool = False
    max_tool_iterations: int = 10
    current_path: Optional[str] = None
    context: Optional[str] = None
    stream: bool = True
    api_url: Optional[str] = None


@dataclass
class StreamEvent:
    """A typed event emitted by the streaming generators.

    Types:
        content_delta   — a chunk of assistant text
        reasoning_delta — a chunk of reasoning/thinking text
        tool_execution_start — signals tool calls are about to run
        tool_start      — a single tool is starting
        tool_result     — a single tool finished successfully
        tool_error      — a single tool errored
        usage           — token usage data
        message_stop    — stream is done
        interrupt       — stream was cancelled
        thinking        — status message for the frontend
    """
    type: str
    data: dict = field(default_factory=dict)


def clean_messages_for_llm(messages: List[dict]) -> List[dict]:
    """Remove orphaned tool_calls and tool results from message history.

    Tool call messages without matching tool results (and vice versa) cause
    API errors with OpenAI-compatible providers.  This function strips them.
    """
    if messages is None:
        return []
    tool_call_ids_with_results = set()
    all_tool_call_ids = set()

    for msg in messages:
        if msg.get('role') == 'assistant' and msg.get('tool_calls'):
            for tc in msg['tool_calls']:
                if isinstance(tc, dict) and tc.get('id'):
                    all_tool_call_ids.add(tc['id'])
        if msg.get('role') == 'tool' and msg.get('tool_call_id'):
            tool_call_ids_with_results.add(msg['tool_call_id'])

    valid_tool_call_ids = all_tool_call_ids & tool_call_ids_with_results

    cleaned = []
    for msg in messages:
        if msg.get('role') == 'tool':
            tool_call_id = msg.get('tool_call_id')
            if not tool_call_id or tool_call_id not in valid_tool_call_ids:
                continue
        if msg.get('role') == 'assistant':
            clean_msg = dict(msg)
            if 'tool_calls' in msg and msg['tool_calls']:
                valid_tcs = [
                    tc for tc in msg['tool_calls']
                    if isinstance(tc, dict) and tc.get('id') in valid_tool_call_ids
                ]
                if valid_tcs:
                    clean_msg['tool_calls'] = valid_tcs
                else:
                    clean_msg.pop('tool_calls', None)
            cleaned.append(clean_msg)
            continue
        cleaned.append(msg)
    return cleaned


def ensure_system_prompt(messages: List[dict], npc=None, system_prompt: str = None, tool_capable=False) -> List[dict]:
    """Ensure messages start with a system prompt.

    If *system_prompt* is given it takes precedence; otherwise falls back to
    ``npc.get_system_prompt()``.  Returns the (possibly modified) list.
    """
    prompt = system_prompt
    if prompt is None and npc is not None and hasattr(npc, 'get_system_prompt'):
        prompt = npc.get_system_prompt(tool_capable=tool_capable)

    if not prompt:
        return messages

    if not messages:
        messages.insert(0, {'role': 'system', 'content': prompt})
    elif messages[0]['role'] != 'system':
        messages.insert(0, {'role': 'system', 'content': prompt})
    else:
        messages[0]['content'] = prompt

    return messages


def parse_stream_chunk(response_chunk, model: str = "", provider: str = "") -> Tuple[str, str, list]:
    """Normalize a streaming chunk from any provider into (content, reasoning, tool_call_deltas).

    Handles:
      - OpenAI / litellm SDK objects (choices[].delta.content / .tool_calls)
      - Ollama / HuggingFace dicts     (message.content / .tool_calls)
      - llamacpp dicts                  (choices[].delta.content)
      - Plain dicts with 'content' key
    """
    content = ""
    reasoning = ""
    tool_call_deltas = []

    if provider == 'ollama' or (isinstance(model, str) and 'hf.co' in model):
        msg = getattr(response_chunk, 'message', None)
        if msg is None and hasattr(response_chunk, 'get'):
            msg = response_chunk.get('message', {})
        if msg is None:
            msg = {}

        content = getattr(msg, 'content', None) or (msg.get('content') if isinstance(msg, dict) else '') or ''
        reasoning = getattr(msg, 'thinking', None) or (msg.get('thinking') if isinstance(msg, dict) else None) or ''

        tool_calls = getattr(msg, 'tool_calls', None) or (msg.get('tool_calls') if isinstance(msg, dict) else None)
        if tool_calls:
            for tc in tool_calls:
                tc_id = getattr(tc, 'id', None) or (tc.get('id') if isinstance(tc, dict) else None)
                tc_func = getattr(tc, 'function', None) or (tc.get('function') if isinstance(tc, dict) else None)
                if tc_func:
                    tc_name = getattr(tc_func, 'name', None) or (tc_func.get('name') if isinstance(tc_func, dict) else None)
                    tc_args = getattr(tc_func, 'arguments', None) or (tc_func.get('arguments') if isinstance(tc_func, dict) else None)
                    if tc_name:
                        arg_str = tc_args
                        if isinstance(arg_str, dict):
                            arg_str = json.dumps(arg_str)
                        elif arg_str is None:
                            arg_str = "{}"
                        tool_call_deltas.append({
                            'id': tc_id or '',
                            'type': 'function',
                            'function': {'name': tc_name, 'arguments': arg_str}
                        })
        return content, reasoning, tool_call_deltas

    if provider == 'llamacpp':
        if isinstance(response_chunk, dict) and response_chunk.get('choices'):
            delta = response_chunk['choices'][0].get('delta', {})
            content = delta.get('content', '') or ''
            reasoning = delta.get('reasoning_content', '') or ''
        return content, reasoning, tool_call_deltas

    if hasattr(response_chunk, 'choices') and response_chunk.choices:
        for choice in response_chunk.choices:
            delta = getattr(choice, 'delta', None)
            if delta is None:
                continue
            if hasattr(delta, 'content') and delta.content:
                content += delta.content
            if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                reasoning += delta.reasoning_content
            if hasattr(delta, 'tool_calls') and delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    tool_call_deltas.append(tc_delta)
        return content, reasoning, tool_call_deltas

    if isinstance(response_chunk, dict):
        content = response_chunk.get('content', '') or response_chunk.get('response', '') or ''

    return content, reasoning, tool_call_deltas


def _build_sse_chunk(content: str, model: str, reasoning: str = None,
                     chunk_id=None, chunk_obj=None, created=None, finish_reason=None) -> dict:
    """Build a normalized SSE chunk dict in OpenAI format."""
    return {
        "id": chunk_id,
        "object": chunk_obj or "chat.completion.chunk",
        "created": created or int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {
                "content": content,
                "role": "assistant",
                "reasoning_content": reasoning,
            },
            "finish_reason": finish_reason,
        }]
    }


_SSE_FLUSH_PAD = ": {}\n\n".format(" " * 4096)


def format_sse_event(event: StreamEvent) -> str:
    """Convert a StreamEvent to an SSE ``data: ...`` line."""
    if event.type == 'content_delta':
        chunk = _build_sse_chunk(
            event.data.get('content', ''),
            event.data.get('model', ''),
            reasoning=event.data.get('reasoning'),
        )
        return "data: {}\n\n".format(json.dumps(chunk)) + _SSE_FLUSH_PAD

    if event.type == 'reasoning_delta':
        chunk = _build_sse_chunk(
            '',
            event.data.get('model', ''),
            reasoning=event.data.get('reasoning', ''),
        )
        return "data: {}\n\n".format(json.dumps(chunk)) + _SSE_FLUSH_PAD

    payload = dict(event.data)
    payload['type'] = event.type
    return "data: {}\n\n".format(json.dumps(payload)) + _SSE_FLUSH_PAD


def format_sse_raw(data: dict) -> str:
    """Format a raw dict as an SSE data line."""
    return "data: {}\n\n".format(json.dumps(data)) + _SSE_FLUSH_PAD


def resolve_npc_tools(npc, mcp_clients_cache: dict = None,
                      selected_tools: List[str] = None) -> Tuple[list, dict]:
    """Resolve all tools available to an NPC: jinx catalog + MCP + python funcs.

    Returns (tools_for_llm, tool_executors) where:
      - tools_for_llm: list of OpenAI-format tool schemas
      - tool_executors: dict mapping tool_name -> executor info
    """
    from npcpy.tools import auto_tools

    tools_for_llm = []
    tool_executors = {}

    if npc is None:
        return tools_for_llm, tool_executors

    if hasattr(npc, 'resolve_tools'):
        tools_for_llm, tool_executors = npc.resolve_tools(
            mcp_clients_cache=mcp_clients_cache or {}
        )

    if hasattr(npc, 'jinx_tool_catalog') and npc.jinx_tool_catalog:
        existing_names = {td['function']['name'] for td in tools_for_llm}
        jinxes_dict = getattr(npc, 'jinxes_dict', {}) or getattr(npc, 'jinxs_dict', {}) or {}
        for t in npc.jinx_tool_catalog.values():
            name = t['function']['name']
            if name not in existing_names:
                tools_for_llm.append(t)
                tool_executors[name] = {
                    'type': 'jinx',
                    'jinx': jinxes_dict.get(name),
                }
                existing_names.add(name)

    if not tools_for_llm and hasattr(npc, 'tools') and npc.tools:
        if isinstance(npc.tools, list) and npc.tools and callable(npc.tools[0]):
            tools_schema, tool_map = auto_tools(npc.tools)
            tools_for_llm = tools_schema
            for name, func in tool_map.items():
                tool_executors[name] = {'type': 'python', 'func': func}

    if selected_tools:
        allowed = set(selected_tools)
        tools_for_llm = [t for t in tools_for_llm if t['function']['name'] in allowed]
        tool_executors = {k: v for k, v in tool_executors.items() if k in allowed}

    return tools_for_llm, tool_executors


def execute_tool(tool_name: str, tool_args: dict, tool_id: str,
                 tool_executors: dict, npc=None) -> StreamEvent:
    """Execute a single tool and return a StreamEvent (tool_result or tool_error)."""
    try:
        executor = tool_executors.get(tool_name)
        if not executor:
            return StreamEvent('tool_error', {
                'name': tool_name, 'id': tool_id,
                'error': "Tool '{}' not found in resolved tools".format(tool_name)
            })

        tool_content = ""
        stop_requested = False
        if executor['type'] == 'jinx':
            jinx_obj = executor['jinx']
            try:
                jinx_ctx = jinx_obj.execute(
                    input_values=tool_args if isinstance(tool_args, dict) else {},
                    npc=npc,
                )
                tool_content = str(jinx_ctx.get('output', '')) if isinstance(jinx_ctx, dict) else str(jinx_ctx)
                stop_requested = bool(isinstance(jinx_ctx, dict) and jinx_ctx.get('stop_requested'))
            except Exception as e:
                tool_content = "Jinx execution error: {}".format(str(e))

        elif executor['type'] == 'mcp':
            tool_func = executor.get('tool_func')
            result = tool_func(**(tool_args if isinstance(tool_args, dict) else {}))
            if hasattr(result, 'content') and result.content and len(result.content) > 0:
                tool_content = str(result.content[0].text)
            else:
                tool_content = str(result) if result is not None else "Tool returned no result"

        elif executor['type'] == 'python':
            func = executor.get('func')
            tool_content = str(func(**(tool_args if isinstance(tool_args, dict) else {})))

        else:
            tool_content = "Unknown executor type: {}".format(executor['type'])

        return StreamEvent('tool_result', {
            'name': tool_name, 'id': tool_id, 'result': tool_content, 'args': tool_args,
            'stop_requested': stop_requested
        })

    except Exception as e:
        return StreamEvent('tool_error', {
            'name': tool_name, 'id': tool_id,
            'error': "Tool execution error: {}".format(str(e))
        })


def _accumulate_tool_call_deltas(collected: list, deltas) -> list:
    """Merge streaming tool_call deltas into collected list."""
    for tc_delta in deltas:
        idx = getattr(tc_delta, 'index', len(collected))
        while len(collected) <= idx:
            collected.append({
                'id': '',
                'type': 'function',
                'function': {'name': '', 'arguments': ''}
            })
        if getattr(tc_delta, 'id', None):
            collected[idx]['id'] = tc_delta.id
        fn = getattr(tc_delta, 'function', None)
        if fn:
            if getattr(fn, 'name', None):
                collected[idx]['function']['name'] = fn.name
            if getattr(fn, 'arguments', None):
                collected[idx]['function']['arguments'] += fn.arguments
    return collected


def create_chat_stream(config: StreamConfig,
                       cancellation_check: Callable[[], bool] = None,
                       ) -> Generator[StreamEvent, None, None]:
    """Stream an LLM response without tool execution.

    Yields StreamEvent objects.  The caller is responsible for SSE formatting
    and persistence.
    """
    messages = config.messages
    npc = config.npc
    model = config.model
    provider = config.provider

    kwargs = {}
    if config.params:
        kwargs.update(config.params)
    if config.temperature and 'temperature' not in kwargs:
        kwargs['temperature'] = config.temperature

    thinking_kwargs = {}
    if not config.disable_thinking and provider in ('anthropic',):
        thinking_kwargs['thinking'] = {"type": "enabled", "budget_tokens": 10000}
        kwargs.pop('temperature', None)

    try:
        if npc and hasattr(npc, 'get_llm_response'):
            response = npc.get_llm_response(
                config.commandstr,
                messages=messages,
                stream=True,
                attachments=config.attachments if config.attachments else None,
                auto_process_tool_calls=False,
                **kwargs,
                **thinking_kwargs,
            )
        else:
            response = get_llm_response(
                config.commandstr,
                messages=messages,
                model=model,
                provider=provider,
                npc=npc,
                team=config.team,
                stream=True,
                images=config.images if config.images else None,
                attachments=config.attachments if config.attachments else None,
                api_url=config.api_url,
                include_usage=True,
                **kwargs,
                **thinking_kwargs,
            )
    except Exception as e:
        yield StreamEvent('tool_error', {'error': "LLM call failed: {}".format(str(e))})
        return

    stream_iter = response
    if isinstance(response, dict):
        for key in ('content', 'text', 'message', 'output'):
            if key in response:
                val = response[key]
                if isinstance(val, dict) and 'content' in val:
                    val = val['content']
                yield StreamEvent('content_delta', {'content': str(val), 'model': model})
                yield StreamEvent('message_stop', {})
                return
        stream_iter = response.get('response', response)
        if isinstance(stream_iter, str):
            yield StreamEvent('content_delta', {'content': stream_iter, 'model': model})
            yield StreamEvent('message_stop', {})
            return
        if isinstance(stream_iter, dict):
            content = stream_iter.get('content', stream_iter.get('output', str(stream_iter)))
            yield StreamEvent('content_delta', {'content': str(content), 'model': model})
            yield StreamEvent('message_stop', {})
            return

    try:
        for chunk in stream_iter:
            if cancellation_check and cancellation_check():
                yield StreamEvent('interrupt', {})
                return

            content, reasoning, tool_call_deltas = parse_stream_chunk(chunk, model, provider)

            if content:
                yield StreamEvent('content_delta', {'content': content, 'model': model})
            if reasoning:
                yield StreamEvent('reasoning_delta', {'reasoning': reasoning, 'model': model})
    except Exception as e:
        print("Error during chat stream: {}".format(e))
        traceback.print_exc()

    yield StreamEvent('message_stop', {})


def create_tool_agent_stream(config: StreamConfig,
                             tools_for_llm: list,
                             tool_executors: dict,
                             cancellation_check: Callable[[], bool] = None,
                             ) -> Generator[StreamEvent, None, None]:
    """Agentic streaming loop: call LLM → stream content → execute tool calls → repeat.

    Uses the OpenAI-style tool_calls mechanism (not check_llm_command).
    Mutates config.messages in place so the caller can read final state.
    """
    messages = config.messages
    npc = config.npc
    model = config.model
    provider = config.provider
    prompt = config.commandstr
    total_input_tokens = 0
    total_output_tokens = 0

    kwargs = {}
    if config.params:
        kwargs.update(config.params)
    if config.temperature and 'temperature' not in kwargs:
        kwargs['temperature'] = config.temperature

    thinking_kwargs = {}
    if not config.disable_thinking and provider in ('anthropic',):
        thinking_kwargs['thinking'] = {"type": "enabled", "budget_tokens": 10000}
        kwargs.pop('temperature', None)

    agent_context = ''
    if config.current_path:
        agent_context = "The user's working directory is {}".format(config.current_path)

    iteration = 0
    stop_requested = False
    while iteration < config.max_tool_iterations:
        iteration += 1

        if stop_requested:
            break

        try:
            llm_response = get_llm_response(
                prompt=prompt,
                npc=npc,
                model=model,
                provider=provider,
                messages=messages,
                tools=tools_for_llm,
                stream=True,
                team=config.team,
                context=agent_context if agent_context else config.context,
                api_url=config.api_url,
                include_usage=True,
                **kwargs,
                **thinking_kwargs,
            )
        except Exception as e:
            yield StreamEvent('tool_error', {'error': "LLM call failed: {}".format(str(e))})
            break

        stream_iter = llm_response.get('response', llm_response) if isinstance(llm_response, dict) else llm_response
        usage = llm_response.get('usage', {}) if isinstance(llm_response, dict) else {}
        total_input_tokens += usage.get('input_tokens', 0) or 0
        total_output_tokens += usage.get('output_tokens', 0) or 0

        collected_content = ""
        collected_tool_calls = []

        if hasattr(stream_iter, '__iter__') and not isinstance(stream_iter, (str, dict)):
            for chunk in stream_iter:
                if cancellation_check and cancellation_check():
                    yield StreamEvent('interrupt', {})
                    return

                content, reasoning, tc_deltas = parse_stream_chunk(chunk, model, provider)

                if content:
                    collected_content += content
                    yield StreamEvent('content_delta', {'content': content, 'model': model})
                if reasoning:
                    yield StreamEvent('reasoning_delta', {'reasoning': reasoning, 'model': model})

                if tc_deltas:
                    _accumulate_tool_call_deltas(collected_tool_calls, tc_deltas)

                chunk_usage = getattr(chunk, 'usage', None)
                if chunk_usage is None and isinstance(chunk, dict):
                    chunk_usage = chunk.get('usage')
                if chunk_usage:
                    inp = getattr(chunk_usage, 'prompt_tokens', None) or (chunk_usage.get('prompt_tokens', 0) if isinstance(chunk_usage, dict) else 0)
                    out = getattr(chunk_usage, 'completion_tokens', None) or (chunk_usage.get('completion_tokens', 0) if isinstance(chunk_usage, dict) else 0)
                    if inp:
                        total_input_tokens = inp
                    if out:
                        total_output_tokens = out
                prompt_eval = getattr(chunk, 'prompt_eval_count', None)
                eval_count = getattr(chunk, 'eval_count', None)
                if prompt_eval:
                    total_input_tokens = prompt_eval
                if eval_count:
                    total_output_tokens = eval_count

        elif isinstance(stream_iter, str):
            collected_content = stream_iter
            yield StreamEvent('content_delta', {'content': stream_iter, 'model': model})
        elif isinstance(stream_iter, dict):
            val = stream_iter.get('content', stream_iter.get('output', str(stream_iter)))
            collected_content = str(val)
            yield StreamEvent('content_delta', {'content': collected_content, 'model': model})

        if not collected_tool_calls:
            break

        serialized_tool_calls = []
        for tc in collected_tool_calls:
            parsed_args = tc['function']['arguments']
            if isinstance(parsed_args, dict):
                args_for_message = json.dumps(parsed_args)
            else:
                args_for_message = str(parsed_args)
            serialized_tool_calls.append({
                'id': tc['id'],
                'type': tc.get('type', 'function'),
                'function': {
                    'name': tc['function']['name'],
                    'arguments': args_for_message,
                }
            })

        messages.append({
            'role': 'assistant',
            'content': collected_content,
            'tool_calls': serialized_tool_calls,
        })

        yield StreamEvent('tool_execution_start', {
            'tool_calls': [
                {'name': tc['function']['name'], 'id': tc['id'],
                 'function': {'name': tc['function']['name'], 'arguments': tc['function'].get('arguments', '')}}
                for tc in collected_tool_calls
            ]
        })

        for tc in collected_tool_calls:
            tool_name = tc['function']['name']
            tool_args = tc['function']['arguments']
            tool_id = tc['id']

            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args) if tool_args.strip() else {}
                except json.JSONDecodeError:
                    tool_args = {}

            yield StreamEvent('tool_start', {'name': tool_name, 'id': tool_id, 'args': tool_args})

            result_event = execute_tool(tool_name, tool_args, tool_id, tool_executors, npc=npc)
            yield result_event

            if result_event.data.get('stop_requested'):
                stop_requested = True
                prompt = ""
                break

            tool_content = result_event.data.get('result', result_event.data.get('error', ''))
            messages.append({
                'role': 'tool',
                'tool_call_id': tool_id,
                'name': tool_name,
                'content': tool_content,
            })

        prompt = ""

    if total_input_tokens or total_output_tokens:
        try:
            from npcpy.gen.response import calculate_cost
            cost = calculate_cost(model, total_input_tokens, total_output_tokens)
        except Exception:
            cost = 0
        yield StreamEvent('usage', {
            'input_tokens': total_input_tokens,
            'output_tokens': total_output_tokens,
            'cost': cost or 0,
        })

    yield StreamEvent('message_stop', {})


def create_jinx_stream(npc,
                       command: str,
                       cancellation_check: Callable[[], bool] = None,
                       max_followups: int = 10,
                       **kwargs,
                       ) -> Generator[StreamEvent, None, None]:
    """Agentic jinx streaming loop.

    Args:
        npc: NPC instance (model, provider, memory, jinxes_dict, api_url, api_key)
        command: The user's message
        cancellation_check: Optional cancellation callback
        max_followups: Max agentic iterations
    """
    model = npc.model
    provider = npc.provider
    messages = npc.memory

    jinxes_dict = None
    if npc:
        jinxes_dict = getattr(npc, 'jinxes_dict', None) or getattr(npc, 'jinxs_dict', None)

    if not jinxes_dict:
        response = get_llm_response(
            command, model=model, provider=provider, npc=npc,
            messages=messages, stream=True, **kwargs,
        )
        out = response.get("response", "")
        if hasattr(out, '__iter__') and not isinstance(out, (str, bytes, dict)):
            for chunk in out:
                content, reasoning, tcd = parse_stream_chunk(chunk, model, provider)
                if content:
                    yield StreamEvent('content_delta', {'content': content, 'model': model})
        elif isinstance(out, str) and out:
            yield StreamEvent('content_delta', {'content': out, 'model': model})
        yield StreamEvent('message_stop', {})
        return

    for iteration in range(max_followups):
        if cancellation_check and cancellation_check():
            yield StreamEvent('interrupt', {})
            return

        if iteration > 0:
            yield StreamEvent('thinking', {'message': 'Processing...'})

        cmd = command if iteration == 0 else "Continue. If results are ready, present them to the user using chat, then call stop. Do not call stop without first responding to the user."

        try:
            gen = npc.check_llm_command(cmd, messages=messages, stream=True, **kwargs)
        except Exception as e:
            print("check_llm_command error (iter {}): {}".format(iteration, e))
            break

        result = None
        stop_called = False
        has_jinx_calls = False

        for event_type, data in gen:
            if event_type == 'tool_start':
                name = data.get('name', '')
                if name not in ('chat', 'stop'):
                    has_jinx_calls = True
                    yield StreamEvent('tool_start', data)
            elif event_type == 'tool_result':
                name = data.get('name', '')
                if name in ('stop', 'done', 'finish'):
                    stop_called = True
                output = data.get('result', '')

                if name == 'chat':
                    if hasattr(output, '__iter__') and not isinstance(output, (str, bytes, dict)):
                        for chunk in output:
                            content, reasoning, tcd = parse_stream_chunk(chunk, model, provider)
                            if content:
                                yield StreamEvent('content_delta', {'content': content, 'model': model})
                    elif isinstance(output, str):
                        if output.startswith('[Response delivered to user] '):
                            output = output[len('[Response delivered to user] '):]
                        if output:
                            yield StreamEvent('content_delta', {'content': output, 'model': model})
                elif name != 'stop':
                    yield StreamEvent('tool_result', data)
            elif event_type == 'content':
                output = data
                if hasattr(output, '__iter__') and not isinstance(output, (str, bytes, dict)):
                    for chunk in output:
                        content, reasoning, tcd = parse_stream_chunk(chunk, model, provider)
                        if content:
                            yield StreamEvent('content_delta', {'content': content, 'model': model})
                elif isinstance(output, str) and output:
                    yield StreamEvent('content_delta', {'content': output, 'model': model})
            elif event_type == 'content':
                output = data
                if hasattr(output, '__iter__') and not isinstance(output, (str, bytes, dict)):
                    for chunk in output:
                        content, reasoning, tcd = parse_stream_chunk(chunk, model, provider)
                        if content:
                            yield StreamEvent('content_delta', {'content': content, 'model': model})
                elif isinstance(output, str) and output:
                    yield StreamEvent('content_delta', {'content': output, 'model': model})
            elif event_type == 'result':
                result = data

        if stop_called:
            break
        if not has_jinx_calls:
            break

    yield StreamEvent('message_stop', {})
