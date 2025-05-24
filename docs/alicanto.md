# Alicanto: Deep Thematic Research

Alicanto is a deep research method inspired by the Chilean mythological bird that can lead miners to gold or to death. In NPC Shell, Alicanto conducts extensive multi-perspective research on any topic by exploring with both breadth (across different expert perspectives) and depth (diving deeper into promising directions).

## Overview

Alicanto creates a team of diverse AI researchers who explore your research question from multiple angles. It then analyzes their findings, identifies thematic clusters, evaluates the quality and risk of each insight, and produces a comprehensive report with "gold insights" (high quality, low risk) and "cliff warnings" (high risk).

## Usage

```bash
# Basic usage
npc alicanto "What are the implications of quantum computing for cybersecurity?"

# With more researchers and deeper exploration
npc alicanto "How might climate change impact global food security?" --num-npcs 8 --depth 5

# Control exploration vs. exploitation balance
npc alicanto "What ethical considerations should guide AI development?" --exploration 0.5

# Different output formats
npc alicanto "What is the future of remote work?" --format report
```

## Options

- `--num-npcs N`: Number of researcher NPCs to use (default: 5)
- `--depth N`: Depth of research chains for each NPC (default: 3)
- `--exploration FLOAT`: Balance between exploration and exploitation (0.0-1.0, default: 0.3)
- `--format FORMAT`: Output format: "report" (default), "summary", or "full"

## How it Works

1. **Expert Generation**: Creates a diverse team of AI researchers with different expertise and perspectives
2. **Research Chains**: Each researcher conducts a series of research steps, going deeper with each iteration
3. **Insight Extraction**: Key insights are extracted from each research chain
4. **Thematic Grouping**: Insights are grouped into thematic clusters across researchers
5. **Quality Evaluation**: Each theme is evaluated for quality (novelty, depth, practicality, evidence) and risk
6. **Gold and Cliff Identification**: High-quality insights with low risk are marked as "gold insights," while high-risk insights are flagged as "cliff warnings"
7. **Integration**: A comprehensive research report synthesizes findings and provides guidance

## Example Output

The output includes:
- An integrated overview connecting all themes
- The most significant overall findings
- Specific recommendations for further research
- Particular cautions and limitations
- Clearly marked gold insights and cliff warnings
- Detailed thematic findings with quality and risk scores

## Use Cases

- Exploring complex, multifaceted topics
- Getting diverse perspectives on controversial issues
- Identifying promising research directions
- Evaluating the quality and risk of different approaches
- Discovering connections between seemingly unrelated ideas