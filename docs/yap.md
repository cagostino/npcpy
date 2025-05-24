# yap

`yap` is an agentic voice control interface that allows users to interact with NPCs and LLMs using voice commands. This tool represents the evolution of voice interaction in the npcpy framework, replacing the older `/whisper` functionality.

## Overview

The `yap` command launches a voice interaction session where you can speak your queries and commands to your NPCs, and they'll respond in text format. This creates a natural conversational flow while maintaining text-based responses for clarity and reference.

## Usage

### As a standalone command

You can launch `yap` directly from your terminal:

```bash
yap
```

This starts a voice interaction session with the default NPC.

### With a specific NPC

```bash
yap -n <npc_name>
```

This starts a voice interaction session with the specified NPC.

### From within npcsh

When using the NPC shell, you can switch to voice control mode:

```
/yap
```

## How it Works

1. When launched, `yap` calibrates to your environment's ambient noise level
2. It listens for your voice input (indicated by a "Listening..." message)
3. When you finish speaking, it processes your audio and converts it to text
4. Your transcribed message is sent to the active NPC/LLM
5. The response is displayed as text
6. The system automatically begins listening for your next query

## Features

- **Automatic silence detection**: Stops listening when you stop speaking
- **Maximum duration limit**: Prevents unnecessarily long recordings
- **Adjustable sensitivity**: Can be calibrated for different environments
- **Full NPC integration**: Works with all NPC capabilities, including jinxs and tools

## Requirements

To use `yap`, you need:

- A working microphone
- The required audio dependencies installed (see the installation guide for your OS)
- Python speech recognition libraries (installed with `pip install npcpy[yap]`)

## Example Session

```
$ yap -n simon_bolivar

Calibrating silence level. Please remain quiet...
Silence threshold set to: 5234.21
Listening... (speak now)
...............

Processing...
You said: What territories should we focus on liberating first?

As the liberator of South America, I believe our primary focus should be on securing New Granada (modern-day Colombia), particularly the areas around Bogotá and Cartagena. These territories are strategically essential for several reasons:

1. Controlling New Granada provides access to both the Caribbean Sea and Pacific Ocean
2. The region has valuable resources to fund our revolutionary efforts
3. From this position, we can establish a base to launch campaigns southward toward Quito and Peru

Once New Granada is secured, we should immediately press southward to liberate Quito (Ecuador), which forms a critical geographic link between the northern and southern territories.

Venezuela, though my homeland, has proven difficult to hold permanently due to Spanish naval power. It will be easier to secure once we control the interior territories.

Remember, our ultimate goal is not just military victory, but the establishment of Gran Colombia as a unified republic free from Spanish tyranny.

Listening... (speak now)
...............
```

## Troubleshooting

- **If your microphone isn't detected**: Ensure it's properly connected and your system recognizes it
- **If speech recognition is inaccurate**: Try speaking more clearly or adjusting your distance from the microphone
- **If you're in a noisy environment**: Consider using the `-t` flag to adjust the noise threshold manually

## See Also

- [npcsh - The NPC Shell](npcsh.md)
- [spool - Interactive NPC Chat](spool.md)