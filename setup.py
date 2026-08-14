from setuptools import setup, find_packages

from pathlib import Path
import os


def package_files(directory):
    paths = []
    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join(path, filename))
    return paths




base_requirements = [
    "jinja2",
    "litellm==1.91.1",
    "scipy", 
    "numpy",
    "requests",
    "docx", 
    "exa-py", 
    "elevenlabs",
    "markdown",
    "networkx", 
    "PyYAML",
    "pypdf",
    "pyautogui",
    "pydantic", 
    "pygments>=2.20.0",
    "sqlalchemy",
    "termcolor",
    "rich",
    "colorama",
    "docstring_parser",
    "Pillow>=12.1.1",
    "python-dotenv",
    "pandas",
    "polars",
    "beautifulsoup4",
    "ddgs>=9.0.0",
    "flask",
    "flask_cors",
    "redis",
    "psycopg2-binary",
    "flask_sse",
    "mcp",
    "httpx>=0.28.0,<1.0",
]


api_requirements = [
    "anthropic",
    "openai",
    "ollama", 
    "google-generativeai",
    "google-genai",
]


local_requirements = [
    "sentence_transformers",
    "opencv-python",
    "ollama",
    "chromadb",
    "diffusers",
    "torch",
    "datasets",
    "airllm",
]


voice_requirements = [
    "pyaudio",
    "gtts",
    "playsound==1.2.2",
    "pygame", 
    "faster_whisper",
    "pyttsx3",
]

extra_files = package_files("npcpy/npc_team/")

setup(
    name="npcpy",
    version="2.1.13",
    packages=find_packages(exclude=["tests*"]),
    install_requires=base_requirements,  
    extras_require={
        "lite": api_requirements,
        "local": local_requirements,
        "yap": voice_requirements,
        "all": api_requirements + local_requirements + voice_requirements,
    },
    entry_points={
        "console_scripts": [
            "npc-init=npcpy.init:main",
            "npc-claude=npcpy.launcher:main",
            "npc-codex=npcpy.launcher:main",
            "npc-gemini=npcpy.launcher:main",
            "npc-opencode=npcpy.launcher:main",
            "npc-aider=npcpy.launcher:main",
            "npc-amp=npcpy.launcher:main",
            "npc-plugin=npcpy.plugin_setup:main",
            "jinx2skill=npcpy.convert:_cli_jinx_to_skill",
            "skill2jinx=npcpy.convert:_cli_skill_to_jinx",
            "agents2npc=npcpy.convert:_cli_agents_to_npc",
            "npc2agents=npcpy.convert:_cli_npc_to_agents",
            "npc2mcp=npcpy.npc2mcp:main",
        ],
    },
    author="Christopher Agostino",
    author_email="info@npcworldwi.de",
    description="npcpy is the premier open-source library for integrating LLMs and Agents into python systems.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/NPC-Worldwide/npcpy",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    include_package_data=True,
    python_requires=">=3.10",
)

