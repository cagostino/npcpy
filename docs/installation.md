## Installation
`npcpy` is available on PyPI and can be installed using pip. Before installing, make sure you have the necessary dependencies installed on your system. Below are the instructions for installing such dependencies on Linux, Mac, and Windows. If you find any other dependencies that are needed, please let us know so we can update the installation instructions to be more accommodating.

### Linux install
```bash

# for audio primarily
sudo apt-get install espeak
sudo apt-get install portaudio19-dev python3-pyaudio
sudo apt-get install alsa-base alsa-utils
sudo apt-get install libcairo2-dev
sudo apt-get install libgirepository1.0-dev
sudo apt-get install ffmpeg

# for triggers
sudo apt install inotify-tools


#And if you don't have ollama installed, use this:
curl -fsSL https://ollama.com/install.sh | sh

ollama pull llama3.2
ollama pull llava:7b
ollama pull nomic-embed-text
pip install npcpy
# if you want to install with the API libraries
pip install npcpy[lite]
# if you want the full local package set up (ollama, diffusers, transformers, cuda etc.)
pip install npcpy[local]
# if you want to use tts/stt
pip install npcpy[yap]

# if you want everything:
pip install npcpy[all]
```


### Mac install
```bash
#mainly for audio
brew install portaudio
brew install ffmpeg
brew install pygobject3

# for triggers
brew install inotify-tools


brew install ollama
brew services start ollama
ollama pull llama3.2
ollama pull llava:7b
ollama pull nomic-embed-text
pip install npcpy
# if you want to install with the API libraries
pip install npcpy[lite]
# if you want the full local package set up (ollama, diffusers, transformers, cuda etc.)
pip install npcpy[local]
# if you want to use tts/stt
pip install npcpy[yap]

# if you want everything:
pip install npcpy[all]

```
### Windows Install

Download and install ollama exe.

Then, in a powershell. Download and install ffmpeg.

```
ollama pull llama3.2
ollama pull llava:7b
ollama pull nomic-embed-text
pip install npcpy
# if you want to install with the API libraries
pip install npcpy[lite]
# if you want the full local package set up (ollama, diffusers, transformers, cuda etc.)
pip install npcpy[local]
# if you want to use tts/stt
pip install npcpy[yap]

# if you want everything:
pip install npcpy[all]

```



### Fedora Install 
- python3-dev (fixes hnswlib issues with chroma db)
- xhost +  (pyautogui)
- python-tkinter (pyautogui)

## Startup Configuration and Project Structure
After it has been pip installed, `npcsh` can be used as a command line tool. Start it by typing:
```bash
npcsh
```
When initialized, `npcsh` will generate a .npcshrc file in your home directory that stores your npcsh settings, like your default chat model/provider, image generation model/provider, embedding model/provider, database path, etc.

On startup, `npcsh` comes with a set of jinxs and NPCs that are used in processing. It will generate a folder at ~/.npcsh/ that contains the jinxs and NPCs that are used by the shell by default if there is no `npc_team` within the current directory. Additionally, `npcsh` records interactions and compiled information about npcs within a local SQLite database at the path specified in the .npcshrc file. This will default to ~/npcsh_history.db if not specified. 

The installer will automatically add this file to your shell config so that it initialize these variables whenever a shell is activated, but if it does not do so successfully for whatever reason (i.e. if you use an alternative rc type) you can add the following to your .bashrc or .zshrc:

```bash
# Source NPCSH configuration
if [ -f ~/.npcshrc ]; then
    . ~/.npcshrc
fi
```

We support inference via all major providers through our litellm integration, including but not limited to: `openai`, `anthropic`, `ollama`,`gemini`, `deepseek`,  and `openai-like` APIs. The default provider must be one of `['openai','anthropic','ollama', 'gemini', 'deepseek', 'openai-like']` or other litellm compatible ones. `openai-like` is `npcsh`-specific in how it works but is intended forr custom servers/locally hosted ones (like those from LM Studio or Llama CPP). The model must be one available from those providers.

To use models that require API keys, create an `.env` file up in the folder where you are working or place relevant API keys as env variables in your `~/.npcshrc`. If you already have these API keys set in a `~/.bashrc` or a `~/.zshrc` or similar files, you need not additionally add them to `~/.npcshrc` or to an `.env` file, but `npcsh` will always check the current folder's `.env` should you want to have projects use separate api keys without manually switching them.
Here is an example of what an `.env` file might look like:

```bash
export OPENAI_API_KEY="your_openai_key"
export ANTHROPIC_API_KEY="your_anthropic_key"
export DEEPSEEK_API_KEY='your_deepseek_key'
export GEMINI_API_KEY='your_gemini_key'
export PERPLEXITY_API_KEY='your_perplexity_key'
```


Individual npcs can also be set to use different models and providers by setting the `model` and `provider` keys in the npc files.

Once initialized and set up, you will find the following in your ~/.npcsh directory:

```bash
~/.npcsh/

└── images/ # images created or uploaded during conversations
└── jobs/ # scheduled jobs 
└── logs/ # logs for triggers and jobs
├── npc_team/           # Global NPCs
│   ├── jinxs/          # Global jinxs
│   └── assembly_lines/ # Workflow pipelines
└── screenshots/ # taken with the screenshot jinx or /ots macro
└── triggers/ # jobs that trigger on certain conditions
```

For cases where you wish to set up a project specific set of NPCs, jinxs, and assembly lines, add a `npc_team` directory to your project and `npcsh` should be able to pick up on its presence, like so:
```bash
./npc_team/            # Project-specific NPCs
├── jinxs/             # Project jinxs #example jinx next
│   └── example.jinx
└── assembly_lines/    # Project workflows
    └── example.pipe
└── models/    # Project workflows
    └── example.model
└── example1.npc        # Example NPC
└── example2.npc        # Example NPC
└── team.ctx            # Example ctx
```

