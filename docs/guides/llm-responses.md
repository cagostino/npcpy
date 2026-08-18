# Working with LLM Responses

The `get_llm_response` function is the primary interface for calling language models in npcpy. It supports multiple providers, streaming, structured output, conversation history, and image attachments. This guide covers the function's capabilities and the shape of the data it returns.

## Basic Usage

At minimum, provide a prompt, model, and provider.

```python
from npcpy.llm_funcs import get_llm_response

response = get_llm_response(
    "Who was the Celtic messenger god?",
    model='qwen3:4b',
    provider='ollama'
)
print(response['response'])
```

```
The Celtic messenger god is often associated with the figure of Lugh,
a deity of many skills who served as a messenger and champion among
the Tuatha De Danann in Irish mythology.
```

## The Response Dict

Every call to `get_llm_response` returns a dict with a consistent structure:

```python
{
    'response': str | None,       # The text reply, or None if tools were called
    'raw_response': object,       # The unprocessed provider response
    'messages': list,             # Full conversation history with the new exchange appended
    'tool_calls': list,           # Tool calls the LLM made (empty if none)
    'tool_results': list,         # Results from executing tools (empty if none)
}
```

The `messages` list is particularly useful for multi-turn conversations, as it includes the system message, all user prompts, and all assistant replies.

## Streaming Responses

Set `stream=True` to get a generator instead of a complete string in `response['response']`. Use `print_and_process_stream` to consume it.

```python
from npcpy.npc_sysenv import print_and_process_stream
from npcpy.llm_funcs import get_llm_response

response = get_llm_response(
    "When did the United States government begin sending advisors to Vietnam?",
    model='qwen3:latest',
    provider='ollama',
    stream=True
)

# response['response'] is a generator here -- print_and_process_stream
# iterates it, prints tokens as they arrive, and returns the full text
full_response = print_and_process_stream(
    response['response'], 'qwen3:latest', 'ollama'
)
```

When streaming, the `messages` list in the response does not include the assistant reply automatically. You must append the completed text yourself if you want to continue the conversation.

## Structured Output with format='json'

Request JSON output by setting `format='json'`. npcpy parses the string into a Python dict for you.

```python
from npcpy.llm_funcs import get_llm_response

response = get_llm_response(
    "What is the sentiment of the American people towards the repeal of "
    "Roe v Wade? Return a JSON object with `sentiment` as the key and "
    "a float value from -1 to 1 as the value.",
    model='deepseek-chat',
    provider='deepseek',
    format='json'
)
print(response['response'])
```

```
{'sentiment': -0.7}
```

This works with local models too:

```python
response = get_llm_response(
    "List the three largest cities in Colombia as a JSON array of objects "
    "with 'name' and 'population' keys.",
    model='llama3.2',
    provider='ollama',
    format='json'
)
print(response['response'])
```

```
[
    {"name": "Bogota", "population": 7181469},
    {"name": "Medellin", "population": 2569007},
    {"name": "Cali", "population": 2227642}
]
```

## Message History

Pass a `messages` list to continue a conversation. The function appends the new user prompt and assistant reply, then returns the updated list in the response.

```python
from npcpy.llm_funcs import get_llm_response

messages = [{'role': 'system', 'content': 'You are a terse historian.'}]

# First turn
response = get_llm_response(
    "When was the Battle of Boyaca?",
    model='llama3.2',
    provider='ollama',
    messages=messages
)
print(response['response'])
# "August 7, 1819."

# Continue the conversation using the returned messages
messages = response['messages']

response = get_llm_response(
    "Who commanded the republican forces?",
    model='llama3.2',
    provider='ollama',
    messages=messages
)
print(response['response'])
# "Simon Bolivar."
```

Each call builds on the prior context. The `messages` list follows the standard format with `role` and `content` keys.

## Attachments and Images

Pass image paths via the `images` parameter. Models with vision capabilities (like `llava:7b`) will process them.

```python
from npcpy.llm_funcs import get_llm_response

messages = [{'role': 'system', 'content': 'You are an annoyed assistant.'}]

response = get_llm_response(
    "What is in this image?",
    model='llava:7b',
    provider='ollama',
    images=['./screenshot.png'],
    messages=messages
)
print(response['response'])
```

The `attachments` parameter handles mixed file types -- images, PDFs, and CSVs. npcpy automatically detects the file type and processes it appropriately:

```python
response = get_llm_response(
    "Summarize this document and describe the chart.",
    model='gemini-2.0-flash',
    provider='gemini',
    attachments=['./report.pdf', './chart.png']
)
```

## AirLLM Provider

npcpy supports AirLLM for running large models with limited memory through layer-by-layer inference.

```python
from npcpy.llm_funcs import get_llm_response

response = get_llm_response(
    "Explain quantum entanglement in simple terms.",
    model='Qwen/Qwen2.5-7B-Instruct',
    provider='airllm'
)
print(response['response'])
```

AirLLM loads and processes model layers sequentially, allowing you to run models that would not fit entirely in memory. The trade-off is slower inference compared to fully-loaded models.

## Provider Summary

`get_llm_response` routes to the appropriate backend based on the `provider` parameter:

| Provider | Example Model | Notes |
|----------|--------------|-------|
| `ollama` | `llama3.2`, `gemma3:4b` | Local models, default provider |
| `openai` | `gpt-4o`, `gpt-4o-mini` | Requires `OPENAI_API_KEY` |
| `openai` | `orcarouter/auto` | [OrcaRouter](https://www.orcarouter.ai) gateway: set `api_url` to `https://api.orcarouter.ai/v1` with a `sk-orca-` key |
| `anthropic` | `claude-sonnet-4` | Requires `ANTHROPIC_API_KEY` |
| `gemini` | `gemini-2.0-flash` | Requires `GEMINI_API_KEY` |
| `deepseek` | `deepseek-chat`, `deepseek-reasoner` | Requires `DEEPSEEK_API_KEY` |
| `airllm` | `Qwen/Qwen2.5-7B-Instruct` | Layer-by-layer local inference |
| `lmstudio` | local model name | OpenAI-compatible on port 1234 |
| `llamacpp` | path to GGUF file | Direct llama.cpp binding |
| `transformers` | HuggingFace model ID | Full HuggingFace transformers |
| `lora` | path to adapter | LoRA fine-tuned adapters |

If no provider is specified and no NPC is given, npcpy defaults to `ollama` with `llama3.2` (or `llava:7b` if images are present).
