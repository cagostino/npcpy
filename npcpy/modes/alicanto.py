import os
import random
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from collections import defaultdict, Counter
import itertools
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from io import BytesIO
import base64
import datetime
import tempfile
import subprocess
import networkx as nx

from npcpy.npc_compiler import NPC
from npcpy.llm_funcs import get_llm_response
from npcpy.npc_sysenv import NPCSH_CHAT_MODEL, NPCSH_CHAT_PROVIDER, print_and_process_stream_with_markdown
from npcpy.memory.deep_research import consolidate_research
from npcpy.memory.knowledge_graph import extract_facts, identify_groups, assign_groups_to_fact

def generate_random_npcs(num_npcs: int, model: str, provider: str, request: str) -> List[NPC]:
    """
    Generate a diverse set of NPCs with different expertise and perspectives
    related to the research request.
    """
    # For single NPC, use a simpler approach to avoid unnecessary LLM calls
    if num_npcs == 1:
        # Generate directly without complex JSON parsing
        name = f"Expert Researcher on {request}"
        expertise = "Interdisciplinary semantic theory researcher"
        background = "Extensive experience in linguistics, cognitive science, and NLP"
        perspective = "Combines formal logic with empirical linguistic evidence"
        quirk = "Uses mathematical metaphors to explain language phenomena"
        biases = "May favor formal approaches over descriptive linguistics"
        
        system_prompt = f"""
        You are {name}, {expertise}.
        
        Background: {background}
        
        Your perspective: {perspective}
        
        Your methodological quirk: {quirk}
        
        Note: Be aware that you may have these biases: {biases}
        
        Your task is to research the given topic thoroughly, focusing on your unique perspective.
        Challenge conventional thinking and identify unexpected connections.
        Your insights should be provocative and novel.
        
        IMPORTANT: You must be extremely concise. Limit responses to 50-75 words maximum.
        """
        
        npc = NPC(name=name, primary_directive=f"Research expert on {request}")
        npc.system_prompt = system_prompt
        return [npc]
    
    # Generate diverse expert personas based on the research topic
    prompt = f"""
    For the research topic: "{request}"
    
    Generate {num_npcs} diverse expert personas who would have different valuable perspectives on this topic.
    I need truly diverse and unusual viewpoints that can lead to innovative insights.
    
    For each expert, provide:
    1. A name
    2. Their field of expertise (be creative - include unconventional and interdisciplinary fields)
    3. Their background/experience (include unusual career paths and perspectives)
    4. Their unique perspective or approach to the topic (emphasize contrarian, minority, or unexpected viewpoints)
    5. A methodological quirk that makes their research approach unusual
    6. Any potential biases they might have
    """
    
    response = get_llm_response(
        prompt=prompt, 
        model=model, 
        provider=provider,
        format="json"  # Directly request JSON format
    )
    
    # Response will be properly structured JSON from get_llm_response
    experts_data = response.get('response', [])
    
    # Create NPC instances from expert data
    npcs = []
    
    # Handle experts_data safely whether it's a list or not
    if isinstance(experts_data, list):
        experts_to_process = experts_data[:num_npcs]
    else:
        # If not a list, try to convert or use as a single item
        if isinstance(experts_data, dict):
            experts_to_process = [experts_data]
        else:
            # Create a basic expert as fallback
            experts_to_process = [{
                "name": f"Expert_1",
                "expertise": "Interdisciplinary researcher",
                "background": "Diverse academic and practical experience",
                "perspective": "Balanced analysis with focus on innovative connections",
                "methodological_quirk": "Uses unconventional conceptual frameworks",
                "biases": "Tends toward theoretical rather than practical solutions"
            }]
    
    for expert in experts_to_process:
        name = expert.get("name", f"Expert_{len(npcs)}")
        
        # Create a system prompt that defines this NPC's expertise and perspective
        system_prompt = f"""
        You are {name}, {expert.get('expertise', 'an expert researcher')}.
        
        Background: {expert.get('background', 'You have extensive knowledge in your field.')}
        
        Your perspective: {expert.get('perspective', 'You provide detailed, balanced analysis.')}
        
        Your methodological quirk: {expert.get('methodological_quirk', 'You approach problems in unconventional ways.')}
        
        Note: Be aware that you may have these biases: {expert.get('biases', 'None specifically noted.')}
        
        Your task is to research the given topic thoroughly, focusing on your unique perspective and methodological approach.
        Challenge conventional thinking, explore neglected angles, and identify unexpected connections or contradictions.
        Your insights should be provocative and novel, not just rehashing mainstream views.
        
        IMPORTANT: You must be extremely concise. Limit responses to 50-75 words maximum. Focus on substance over verbosity.
        Prioritize precision, clarity, and insight density. Eliminate unnecessary words and focus on communicating 
        the essence of your insights in the most efficient way possible.
        """
        
        # Create NPC with name and primary_directive (required parameters)
        npc = NPC(name=name, primary_directive=f"Research expert on {request}")
        npc.system_prompt = system_prompt
        npcs.append(npc)
    
    return npcs

def generate_research_chain(request: str, npc: NPC, depth: int, memory: int = 3, 
                           context: str = None, model: str = None, provider: str = None,
                           exploration_factor: float = 0.3,
                           creativity_factor: float = 0.5) -> List[str]:
    """
    Generate a chain of research thoughts from a single NPC, diving deeper with each step.
    
    Args:
        request: The research question/topic
        npc: The NPC generating the research
        depth: How many steps of research to perform
        memory: How many previous steps to include in context
        context: Additional context to include
        model: LLM model to use
        provider: LLM provider to use
        exploration_factor: Probability (0-1) of exploring a tangential direction
        creativity_factor: Probability (0-1) of pursuing highly creative or unusual ideas
    
    Returns:
        List of research findings/thoughts from this chain
    """
    chain = []
    
    # Initial research prompt
    initial_prompt = f"""
    Research request: {request}
    
    {f"Additional context: {context}" if context else ""}
    
    As {npc.name}, begin your research process by:
    1. Analyzing what you know about this topic
    2. Identifying key questions that need to be explored
    3. Providing initial insights based on your expertise and unique perspective
    
    BE EXTREMELY CONCISE. Focus on substance over wordiness. Provide clear, high-value insights in 50-75 words maximum.
    """
    
    response = get_llm_response(prompt=initial_prompt, model=model, provider=provider, npc=npc, temperature=0.7)
    initial_findings = response.get('response', '')
    if isinstance(initial_findings, (list, dict)) or hasattr(initial_findings, '__iter__') and not isinstance(initial_findings, (str, bytes)):
        initial_findings = ''.join([str(chunk) for chunk in initial_findings])
    
    chain.append(initial_findings)
    
    # For each level of depth, continue the research
    for i in range(1, depth):
        # Get recent memory to include as context
        memory_context = "\n\n".join(chain[-memory:]) if len(chain) > 0 else ""
        
        # Simple follow-up prompt without specific research modes
        next_prompt = f"""
        Research request: {request}
        
        Recent research findings:
        {memory_context}
        
        As {npc.name}, continue your research on this topic. Build on previous insights and explore new aspects.
        
        BE EXTREMELY CONCISE. Keep your response to 50-75 words maximum.
        """
        
        response = get_llm_response(prompt=next_prompt, model=model, provider=provider, npc=npc, temperature=0.7)
        next_findings = response.get('response', '')
        if isinstance(next_findings, (list, dict)) or hasattr(next_findings, '__iter__') and not isinstance(next_findings, (str, bytes)):
            next_findings = ''.join([str(chunk) for chunk in next_findings])
        
        chain.append(next_findings)
    
    return chain

def format_facts_list(facts: List[str]) -> str:
    """Format a list of facts for display in a report"""
    return "\n".join([f"• {fact}" for fact in facts])

def simulate_experiments(research: Dict[str, Any], request: str, model: str = None, provider: str = None, max_experiments: int = None) -> Dict[str, Dict[str, Any]]:
    """
    Simulate thought experiments based on research findings
    
    Args:
        research: Consolidated research data
        request: Original research question
        model: LLM model to use
        provider: LLM provider to use
        max_experiments: Maximum number of experiments to generate
        
    Returns:
        Dictionary mapping experiment titles to experiment data
    """
    # Prepare context with key facts
    facts_context = ""
    
    # Add facts from thematic groups
    if "fact_groups" in research:
        for group, facts in list(research["fact_groups"].items())[:5]:  # Use top 5 groups
            facts_context += f"\n\nThematic Group: {group}\n"
            facts_context += format_facts_list(facts)
    
    # Add insights from combinations
    if "combination_insights" in research:
        facts_context += "\n\nEmergent Insights:\n"
        for combo in research["combination_insights"][:3]:  # Use top 3 insights
            facts_context += f"• {combo.get('emergent_insight', '')}\n"
    
    # Create prompt to design experiments
    prompt = f"""
    You are a creative research scientist exploring the topic: "{request}"
    
    Based on the following research findings:
    
    {facts_context}
    
    Design {max_experiments if max_experiments else "3-5"} thought experiments that could test, validate, or extend these insights.
    
    For each experiment:
    1. Create a descriptive title that captures the experiment's focus
    2. Describe the experimental design/methodology (be specific and detailed)
    3. Predict the potential results and their implications
    4. Explain how these results would advance our understanding of {request}
    
    Format your response as JSON with this structure:
    {{
      "experiment_title_1": {{
        "design": "detailed description of experimental design",
        "results": "predicted results and implications"
      }},
      "experiment_title_2": {{
        ...
      }}
    }}
    
    Be bold and imaginative in your experimental designs. Consider unconventional approaches,
    simulations, thought experiments, and interdisciplinary methods.
    """
    
    response = get_llm_response(prompt=prompt, model=model, provider=provider, temperature=0.8, format="json")
    experiments = response.get("response", {})
    
    # Limit experiments if needed
    if max_experiments and isinstance(experiments, dict) and len(experiments) > max_experiments:
        # Sort by title length (approximating complexity/interestingness)
        sorted_exps = sorted(experiments.items(), key=lambda x: len(x[0]), reverse=True)
        experiments = dict(sorted_exps[:max_experiments])
    
    return experiments

def alicanto(request: str,
             num_npcs: int = 5,
             depth: int = 3, memory: int = 3, 
             context: str = None, 
             model: str = None, 
             provider: str = None,
             exploration_factor: float = 0.3,
             creativity_factor: float = 0.5,
             output_format: str = "report",
             max_facts_per_chain: int = None,
             max_thematic_groups: int = None, 
             max_criticisms_per_group: int = None,
             max_conceptual_combinations: int = None,
             max_experiments: int = None,
             generate_pdf: bool = True) -> Dict[str, Any]:
    """
    Alicanto: Generate diverse research insights by coordinating multiple NPCs with different expertise.
    
    Args:
        request: The research question/topic
        num_npcs: Number of NPCs to generate (with different expertise)
        depth: Depth of research for each NPC
        memory: How many previous steps to include in context
        context: Additional context to include
        model: LLM model to use
        provider: LLM provider to use
        exploration_factor: Probability (0-1) of exploring a tangential direction
        creativity_factor: Probability (0-1) of pursuing highly creative or unusual ideas
        output_format: Format of the output ("report", "json", "markdown")
        max_facts_per_chain: Maximum number of facts to extract per research chain
        max_thematic_groups: Maximum number of thematic groups to identify
        max_criticisms_per_group: Maximum number of criticisms per thematic group
        max_conceptual_combinations: Maximum number of conceptual combinations to generate
        max_experiments: Maximum number of experiments to generate
        generate_pdf: Whether to generate a PDF report
        
    Returns:
        Dictionary with research results
    """
    # Use default model/provider if not specified
    if model is None:
        model = NPCSH_CHAT_MODEL
    if provider is None:
        provider = NPCSH_CHAT_PROVIDER
    
    # Generate researcher NPCs with diverse expertise
    print(f"Generating {num_npcs} diverse researcher NPCs...")
    researchers = generate_random_npcs(num_npcs, model, provider, request)
    
    # Generate research chains for each NPC
    print(f"Generating research chains (depth={depth})...")
    research_chains = {}
    facts_by_researcher = {}
    
    for npc in researchers:
        print(f"  Research chain from {npc.name}...")
        chain = generate_research_chain(
            request=request,
            npc=npc,
            depth=depth,
            memory=memory,
            context=context,
            model=model,
            provider=provider,
            exploration_factor=exploration_factor,
            creativity_factor=creativity_factor
        )
        research_chains[npc.name] = chain
        
        # Extract facts from chain
        print(f"  Extracting facts from {npc.name}'s research...")
        facts = extract_facts("\n\n".join(chain), model=model, provider=provider, npc=npc, context=request)
        
        # Limit facts if specified
        if max_facts_per_chain is not None and len(facts) > max_facts_per_chain:
            facts = facts[:max_facts_per_chain]
            
        facts_by_researcher[npc.name] = facts
        print({"fact_list": facts})
    
    # Identify thematic groups across all research
    print("Identifying thematic groups across all research insights...")
    all_facts = []
    for researcher_facts in facts_by_researcher.values():
        all_facts.extend(researcher_facts)
    
    groups = identify_groups(all_facts, model=model, provider=provider)
    
    # Limit number of groups if specified
    if max_thematic_groups is not None and len(groups) > max_thematic_groups:
        groups = groups[:max_thematic_groups]
    
    # Assign facts to groups
    fact_groups = {group: [] for group in groups}
    for fact in all_facts:
        group_assignments = assign_groups_to_fact(fact, groups, model=model, provider=provider)
        assigned_groups = group_assignments.get("groups", [])
        for group in assigned_groups:
            if group in fact_groups:
                fact_groups[group].append(fact)
    
    # Evaluate thematic groups
    print("Evaluating thematic groups for quality and risk...")
    group_evaluations = evaluate_thematic_groups(
        fact_groups, 
        request,
        model=model,
        provider=provider,
        max_criticisms=max_criticisms_per_group
    )
    
    # Generate group summaries
    group_summaries = {}
    for group_name, facts in fact_groups.items():
        if not facts:
            continue
            
        prompt = f"""
        Summarize the key insights from this thematic group of research findings on the topic:
        "{request}"
        
        Thematic Group: {group_name}
        
        Findings:
        {format_facts_list(facts)}
        
        Provide a concise, coherent synthesis that captures the core ideas, 
        emphasizes what's most novel or significant, and suggests potential implications.
        Keep your response to 200-300 words.
        """
        
        response = get_llm_response(prompt=prompt, model=model, provider=provider)
        summary = response.get('response', '')
        if isinstance(summary, (list, dict)) or hasattr(summary, '__iter__') and not isinstance(summary, (str, bytes)):
            summary = ''.join([str(chunk) for chunk in summary])
        
        group_summaries[group_name] = summary
    
    # Generate conceptual combinations to spark novel ideas
    print("Generating conceptual combinations to spark novel insights...")
    fact_lists = list(facts_by_researcher.values())
    combinations = generate_conceptual_combinations(
        fact_lists,
        sample_size=min(3, len(all_facts)),
        num_combinations=max_conceptual_combinations if max_conceptual_combinations is not None else 5
    )
    
    # Analyze combinations for emergent insights
    print("Analyzing conceptual combinations for emergent insights...")
    combination_insights = analyze_conceptual_combinations(
        combinations,
        request,
        model=model,
        provider=provider
    )
    
    # Identify meta-patterns
    print("Identifying meta-patterns across research approaches...")
    meta_patterns = identify_patterns_across_chains(research_chains, model=model, provider=provider)
    
    # Generate consolidated research summary
    print("Consolidating research into comprehensive synthesis...")
    
    # Extract key points for integration
    integration_points = []
    
    # Add top facts from each thematic group
    for group, facts in fact_groups.items():
        if facts:
            integration_points.append(f"From thematic group '{group}':")
            for fact in facts[:3]:  # Top 3 facts per group
                integration_points.append(f"- {fact}")
    
    # Add insights from combinations
    for insight in combination_insights[:3]:  # Top 3 insights
        integration_points.append(f"Emergent insight: {insight.get('emergent_insight', '')}")
    
    # Add key points from meta-analysis
    integration_points.append(f"Meta-analysis insight: {meta_patterns.get('meta_analysis', '')[:300]}...")
    
    # Generate integration
    integration_prompt = f"""
    Consolidate these diverse research findings into a comprehensive, integrative analysis of the topic:
    "{request}"
    
    Key points from the research:
    {format_facts_list(integration_points)}
    
    Your consolidation should:
    1. Provide a coherent synthesis of the diverse perspectives
    2. Identify the most significant findings and patterns
    3. Note any tensions, contradictions, or complementary insights
    4. Suggest an integrated framework for understanding the topic
    5. Briefly outline implications and future directions
    
    Aim for a comprehensive, balanced, and insightful analysis (300-500 words).
    """
    
    integration_response = get_llm_response(integration_prompt, model=model, provider=provider)
    integration = integration_response.get('response', '')
    if isinstance(integration, (list, dict)) or hasattr(integration, '__iter__') and not isinstance(integration, (str, bytes)):
        integration = ''.join([str(chunk) for chunk in integration])
    
    # Create concise summary
    summary_prompt = f"""
    Create a concise executive summary (150 words max) of this research on:
    "{request}"
    
    Integration:
    {integration}
    
    Focus on the most significant findings and implications. This should be suitable for someone who only has time to read a brief overview.
    """
    
    summary_response = get_llm_response(summary_prompt, model=model, provider=provider)
    ideas_summarized = summary_response.get('response', '')
    if isinstance(ideas_summarized, (list, dict)) or hasattr(ideas_summarized, '__iter__') and not isinstance(ideas_summarized, (str, bytes)):
        ideas_summarized = ''.join([str(chunk) for chunk in ideas_summarized])
    
    # Simulate experiments
    print("Generating simulated experiments...")
    research_results = {
        "research_request": request,
        "research_chains": research_chains,
        "fact_groups": fact_groups,
        "group_evaluations": group_evaluations,
        "group_summaries": group_summaries,
        "combination_insights": combination_insights,
        "meta_patterns": meta_patterns,
        "integration": integration,
        "ideas_summarized": ideas_summarized
    }
    
    experiments = simulate_experiments(
        research_results,
        request,
        model=model,
        provider=provider,
        max_experiments=max_experiments
    )
    
    # Generate PDF report if requested
    pdf_path = None
    if generate_pdf:
        pdf_path = generate_pdf_report(request, model, provider, research_results, experiments)
    
    # Final research results
    research_results["experiments"] = experiments
    research_results["pdf_path"] = pdf_path
    
    return research_results

def evaluate_thematic_groups(fact_groups: Dict[str, List[str]], request: str, model: str = None, provider: str = None, max_criticisms: int = None) -> Dict[str, Dict[str, int]]:
    """
    Evaluate each thematic group for quality, potential risks, and biases.
    
    Args:
        fact_groups: Dictionary mapping group names to lists of facts
        request: The original research question
        model: LLM model to use
        provider: LLM provider to use
        max_criticisms: Maximum number of criticisms to generate per group
        
    Returns:
        Dictionary mapping group names to evaluation metrics
    """
    evaluations = {}
    
    for group_name, facts in fact_groups.items():
        facts_text = format_facts_list(facts)
        
        prompt = f"""
        Evaluate this thematic group of research insights on the topic:
        "{request}"
        
        Thematic Group: {group_name}
        
        Insights:
        {facts_text}
        
        Evaluate this group of insights on a scale of 1-10 (where 10 is highest) for:
        1. Novelty: How original and non-obvious are these insights?
        2. Depth: How deeply do they explore the underlying concepts?
        3. Practicality: How useful are these insights for further research or application?
        4. Evidence: How well-supported do these claims appear to be?
        5. Risk: What is the chance that these insights lead to problematic directions or dead ends?
        
        Then identify potential weaknesses, biases, or limitations in these insights.
        {f"Provide exactly {max_criticisms} criticisms." if max_criticisms is not None else ""}
        
        Format your response as:
        Novelty: [score]
        Depth: [score]
        Practicality: [score]
        Evidence: [score]
        Risk: [score]
        
        Criticisms:
        1. [First criticism]
        2. [Second criticism]
        ...
        """
        
        response = get_llm_response(prompt=prompt, model=model, provider=provider)
        eval_text = response.get('response', '')
        if isinstance(eval_text, (list, dict)) or hasattr(eval_text, '__iter__') and not isinstance(eval_text, (str, bytes)):
            eval_text = ''.join([str(chunk) for chunk in eval_text])
        
        # Parse scores
        scores = {}
        criticisms = []
        in_criticisms = False
        
        for line in eval_text.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            if line.lower() == "criticisms:":
                in_criticisms = True
                continue
            
            if in_criticisms:
                # Parse criticisms
                if line[0].isdigit() and line[1:].startswith('. '):
                    criticism = line[line.find(' ')+1:].strip()
                    criticisms.append(criticism)
            else:
                # Parse scores
                if ':' in line:
                    metric, score_str = line.split(':', 1)
                    metric = metric.strip()
                    try:
                        score = int(score_str.strip())
                        scores[metric] = score
                    except ValueError:
                        pass
        
        # Apply criticism limit if specified
        if max_criticisms is not None and len(criticisms) > max_criticisms:
            criticisms = criticisms[:max_criticisms]
        
        evaluations[group_name] = {
            **scores,
            "criticisms": criticisms
        }
    
    return evaluations

def generate_conceptual_combinations(fact_lists: List[List[str]], sample_size: int = 3, num_combinations: int = 5) -> List[Dict]:
    """
    Generate interesting combinations of facts from different researchers to spark novel ideas.
    
    Args:
        fact_lists: List of fact lists from different NPCs
        sample_size: Number of facts to include in each combination
        num_combinations: Number of combinations to generate
        
    Returns:
        List of dictionaries containing the combinations and generated insights
    """
    # Flatten facts with researcher ID
    all_facts_with_source = []
    for i, facts in enumerate(fact_lists):
        for fact in facts:
            all_facts_with_source.append((i, fact))
    
    # Generate random combinations
    combinations = []
    for _ in range(num_combinations):
        if len(all_facts_with_source) <= sample_size:
            sample = all_facts_with_source
        else:
            sample = random.sample(all_facts_with_source, sample_size)
        
        combinations.append({
            "facts": [fact for _, fact in sample],
            "sources": [source for source, _ in sample]
        })
    
    return combinations

def analyze_conceptual_combinations(combinations: List[Dict], request: str, model: str = None, provider: str = None) -> List[Dict]:
    """
    Analyze combinations of facts to identify emergent patterns and generate novel hypotheses.
    
    Args:
        combinations: List of fact combinations
        request: The original research question
        model: LLM model to use
        provider: LLM provider to use
        
    Returns:
        List of dictionaries with analysis results
    """
    results = []
    
    for i, combo in enumerate(combinations):
        facts_formatted = format_facts_list(combo["facts"])
        
        prompt = f"""
        Consider these seemingly unrelated insights from different researchers exploring the topic:
        "{request}"
        
        {facts_formatted}
        
        Your task is to identify a non-obvious connection, pattern, or insight that emerges when these ideas are juxtaposed.
        Focus on discovering something truly novel that none of the individual researchers may have recognized.
        
        1. Identify a surprising emergent pattern or connection
        2. Develop a novel hypothesis or research question based on this pattern
        3. Explain how this insight challenges or extends conventional thinking on the topic
        4. Suggest an unconventional methodology or approach to explore this new direction
        
        Be bold, imaginative, and interdisciplinary in your thinking.
        """
        
        response = get_llm_response(prompt=prompt, model=model, provider=provider, temperature=0.9)
        insight = response.get('response', '')
        if isinstance(insight, (list, dict)) or hasattr(insight, '__iter__') and not isinstance(insight, (str, bytes)):
            insight = ''.join([str(chunk) for chunk in insight])
        
        results.append({
            "combination_id": i+1,
            "facts": combo["facts"],
            "sources": combo["sources"],
            "emergent_insight": insight
        })
    
    return results

def identify_patterns_across_chains(chains: Dict[str, List[str]], model: str = None, provider: str = None) -> Dict:
    """
    Identify meta-patterns across research chains, searching for higher-order insights.
    
    Args:
        chains: Dictionary mapping NPC names to their research chains
        model: LLM model to use
        provider: LLM provider to use
        
    Returns:
        Dictionary with meta-analysis results
    """
    # Prepare a summary of each research chain
    chain_summaries = {}
    for name, chain in chains.items():
        full_text = "\n\n".join(chain)
        
        summary_prompt = f"""
        Summarize the key themes, methodologies, and unusual perspectives in this research chain:
        
        {full_text[:2000]}...
        
        Focus on what makes this researcher's approach unique or valuable. Identify their core assumptions,
        methodological innovations, and blindspots (150-200 words).
        """
        
        response = get_llm_response(prompt=summary_prompt, model=model, provider=provider)
        summary = response.get('response', '')
        if isinstance(summary, (list, dict)) or hasattr(summary, '__iter__') and not isinstance(summary, (str, bytes)):
            summary = ''.join([str(chunk) for chunk in summary])
        
        chain_summaries[name] = summary
    
    # Generate meta-analysis across all chains
    all_summaries = "\n\n".join([f"[{name}]\n{summary}" for name, summary in chain_summaries.items()])
    
    meta_analysis_prompt = f"""
    Analyze these research approaches on the topic:
    
    {all_summaries}
    
    Identify:
    1. Surprising methodological patterns - how are researchers approaching this problem in innovative ways?
    2. Conceptual blindspots - what aspects seem to be collectively overlooked?
    3. Emerging paradigms - are there new frameworks or models taking shape across multiple perspectives?
    4. Productive tensions - where do disagreements or contradictions suggest valuable new research directions?
    5. The topology of the problem space - how might we map the conceptual territory in a novel way?
    
    Focus on identifying higher-order insights that emerge from comparing these different approaches.
    Your analysis should challenge conventions and suggest new ways of framing the entire research domain.
    """
    
    response = get_llm_response(prompt=meta_analysis_prompt, model=model, provider=provider, temperature=0.8)
    meta_analysis = response.get('response', '')
    if isinstance(meta_analysis, (list, dict)) or hasattr(meta_analysis, '__iter__') and not isinstance(meta_analysis, (str, bytes)):
        meta_analysis = ''.join([str(chunk) for chunk in meta_analysis])
    
    # Generate innovative research directions
    directions_prompt = f"""
    Based on this meta-analysis of research approaches to the topic:
    
    {meta_analysis}
    
    Propose 5 highly innovative research directions that could transform this field.
    For each direction:
    1. Frame a provocative research question
    2. Explain why it's both important and neglected
    3. Suggest an unconventional methodology to explore it
    4. Describe what a breakthrough in this direction might look like
    
    Your suggestions should be bold, interdisciplinary, and challenge fundamental assumptions.
    Aim for directions that most researchers haven't considered but that could lead to significant advances.
    """
    
    response = get_llm_response(prompt=directions_prompt, model=model, provider=provider, temperature=0.9)
    new_directions = response.get('response', '')
    if isinstance(new_directions, (list, dict)) or hasattr(new_directions, '__iter__') and not isinstance(new_directions, (str, bytes)):
        new_directions = ''.join([str(chunk) for chunk in new_directions])
    
    return {
        "chain_summaries": chain_summaries,
        "meta_analysis": meta_analysis,
        "innovative_directions": new_directions
    }

def preprocess_content_for_pdf(content: str, model: str = None, provider: str = None, max_words: int = 2000, concise_mode: bool = False) -> str:
    """
    Quick and lightweight preprocessing for PDF generation.
    
    Args:
        content: Raw content to preprocess
        model: LLM model to use (optional)
        provider: LLM provider to use (optional)
        max_words: Maximum word count (default 2000)
        concise_mode: If True, creates a very short summary instead of full formatting
        
    Returns:
        Formatted content ready for PDF generation
    """
    # Handle non-string content
    if not isinstance(content, str):
        content = str(content)
    
    # If in concise mode, create a drastically shortened version
    if concise_mode:
        from npcpy.llm_funcs import get_llm_response
        from npcpy.npc_sysenv import NPCSH_CHAT_MODEL, NPCSH_CHAT_PROVIDER
        
        if model is None:
            model = NPCSH_CHAT_MODEL
        if provider is None:
            provider = NPCSH_CHAT_PROVIDER
            
        concise_prompt = f"""
        Summarize the following content into an extremely concise, no-bullshit format with maximum 500 words:
        {content}
        
        - Use clear section headings
        - Use bullet points for key ideas
        - Focus only on essential insights
        - No verbose academic language
        - No padding or fillers
        - Just the core ideas in simple language
        """
        
        response = get_llm_response(prompt=concise_prompt, model=model, provider=provider)
        content = response.get('response', '')
    
    # Basic cleanup for any problematic characters that cause PDF issues
    for char, replacement in {
        '%': '',
        '#': '-',
        '_': '-',
        '~': '-',
        '^': '',
        '\\': '/',
        '{': '(',
        '}': ')'
    }.items():
        content = content.replace(char, replacement)
    
    # Apply word count limit if the content is too long
    words = content.split()
    if len(words) > max_words:
        content = ' '.join(words[:max_words]) + '... [truncated]'
    
    return content.strip()

def generate_pdf_report(request: str, 
                        model, 
                        provider, 
                        research: Dict[str, Any], 
                        experiments: Dict[str, Dict[str, Any]], 
                        output_path: str = None, 
                        max_pages: int = 5) -> str:
    """
    Generate a professional PDF report using LaTeX for superior formatting, typesetting, and layout.
    
    Args:
        request: The original research question
        research: The consolidated research results
        experiments: The simulated experiments and their results
        output_path: Path to save the PDF report (default: current directory)
        fast_mode: If True, uses simpler formatting
        concise_mode: If True, drastically reduces content length
        max_pages: Maximum number of pages to generate (approximate)
        
    Returns:
        Path to the generated PDF file
    """
    if output_path is None:
        output_path = os.getcwd()
    
    # Create filename
    sanitized_request = "".join(c for c in request if c.isalnum() or c.isspace()).strip()
    sanitized_request = sanitized_request.replace(" ", "_")[:50]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{sanitized_request}_{timestamp}"
    
    # Check for LaTeX installation
    try:
        subprocess.run(["which", "pdflatex"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        print("LaTeX not installed. Attempting to install...")
        try:
            subprocess.run(["apt-get", "update"], check=True)
            subprocess.run(["apt-get", "install", "-y", "texlive-latex-base", "texlive-fonts-recommended", 
                            "texlive-fonts-extra", "texlive-latex-extra"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error installing LaTeX: {str(e)}")
            return None
    # Create chart for thematic groups using matplotlib
    chart_path = None
    try:
        if "group_evaluations" in research and research["group_evaluations"]:
            # Create basic folder for figures
            figures_dir = os.path.join(output_path, "figures")
            os.makedirs(figures_dir, exist_ok=True)
            
            fig, ax = plt.subplots(figsize=(7.5, 4))
            plt.style.use('ggplot')  # Clean style without seaborn
            
            groups = []
            scores = []
            
            for group_name, eval_data in research["group_evaluations"].items():
                groups.append(group_name[:30])  # Truncate long names
                quality_score = (eval_data.get("Novelty", 5) + eval_data.get("Depth", 5) + 
                               eval_data.get("Practicality", 5) + eval_data.get("Evidence", 5)) / 4
                scores.append(quality_score)
            
            # Sort by score
            sorted_data = sorted(zip(groups, scores), key=lambda x: x[1], reverse=True)
            groups = [x[0] for x in sorted_data]
            scores = [x[1] for x in sorted_data]
            
            # Create horizontal bar chart
            y_pos = range(len(groups))
            ax.barh(y_pos, scores, color='steelblue')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(groups)
            ax.set_xlabel('Quality Score (1-10)')
            ax.set_title('Thematic Groups by Quality Score')
            plt.tight_layout()
            
            # Save chart
            chart_path = os.path.join(figures_dir, f"thematic_groups.pdf")
            plt.savefig(chart_path, dpi=300, bbox_inches='tight', format='pdf')
            plt.close()
    except Exception as e:
        print(f"Warning: Could not generate chart: {str(e)}")
    
    # Create LaTeX document
    latex_content = generate_latex_document(request, model, provider,  research, experiments, chart_path, max_pages)
    
    # Write LaTeX to file
    tex_path = os.path.join(output_path, f"{filename}.tex")
    with open(tex_path, "w") as f:
        f.write(latex_content)
    
    # Use subprocess to run pdflatex without check=True to prevent exceptions
    try:
        # First run
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", output_path, tex_path],
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        
        if result.returncode != 0:
            print(f"Warning: First LaTeX run had issues (exit code {result.returncode})")
            # Still continue - sometimes the second run fixes things
        
        # Second run for references
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", output_path, tex_path],
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )
        
        if result.returncode != 0:
            print(f"Warning: Second LaTeX run had issues (exit code {result.returncode})")
            # Write LaTeX log for debugging
            log_path = os.path.join(output_path, f"{filename}.log")
            if os.path.exists(log_path):
                print(f"Check LaTeX log for details: {log_path}")
    except Exception as e:
        print(f"Error during LaTeX compilation: {str(e)}")
        return None
    
    # Clean up temporary files
    for ext in [".aux", ".out", ".toc"]:
        try:
            os.remove(os.path.join(output_path, f"{filename}{ext}"))
        except OSError:
            pass
    
    # Check if PDF was generated successfully
    pdf_path = os.path.join(output_path, f"{filename}.pdf")
    if os.path.exists(pdf_path):
        print(f"PDF report successfully generated using LaTeX: {pdf_path}")
        return pdf_path
    else:
        print(f"PDF generation failed. Check the LaTeX log for details.")
        return None

def generate_latex_document(request: str, model, provider, research: Dict[str, Any], experiments: Dict[str, Dict[str, Any]],
                            chart_path: str = None, max_pages: int = 5) -> str:
    """
    Generate LaTeX document content.
    
    Args:
        request: The research topic
        research: Research results
        experiments: Experiments data
        chart_path: Path to the thematic groups chart
        max_pages: Maximum number of pages (approximate)
    
    Returns:
        LaTeX document content as a string
    """
    # Collect experiment images that might be available
    figure_paths = {}
    if chart_path:
        # Use relative path instead of absolute path for figure
        figure_paths["thematic_groups"] = os.path.basename(chart_path)
    
    # Check for experiment images in the current directory
    # Ensure experiments is a dictionary before trying to get keys
    if isinstance(experiments, dict):
        for title in experiments.keys():
            sanitized_title = title.replace(" ", "_")
            potential_image = f"{sanitized_title}_experiment.png"
            if os.path.exists(potential_image):
                figure_paths[sanitized_title] = potential_image
    
    # Describe available figures to the LLM
    figure_path_description_dict = {}
    for name, path in figure_paths.items():
        figure_path_description_dict[name] = path
    
    # Create the prompt for generating LaTeX content
    prompt = f'''
    Generate a LaTeX document for a research report on the topic: "{request}"
    Here is the summary of the research: {research}
    
    Here is the summary of the experiments: {experiments}''' +"""
    Write your response in a way that academically details the research, its motivation, and experiments
    and ensure any place where a citation may be needed is indicated by including an empty '\\cite{citation_needed}' 
    
    IMPORTANT INSTRUCTIONS FOR DOCUMENT PREPARATION:
    1. DO NOT include \\bibliography{references} or any bibliography commands, as we don't have a references file
    2. Instead, create a \\begin{thebibliography}{99} ... \\end{thebibliography} section with example references
    3. For figures, use relative paths like 'figures/thematic_groups.pdf' rather than absolute paths
    4. Make sure all LaTeX commands are properly formatted and do not use undefined packages
    5. Keep the document structure simple and robust to avoid compilation errors
    """+f"""    
    The figures are located at the following paths: {figure_path_description_dict}
    """
    
    
    latex_response = get_llm_response(prompt=prompt, model=model, provider=provider )
    latex_content = latex_response.get('response', '')
    
    # Post-process the LaTeX content to fix common issues
    latex_content = latex_content.replace('\\bibliography{references}', '')
    latex_content = latex_content.replace('\\bibliographystyle{plain}', '')
    
    # Replace absolute figure paths with relative paths
    latex_content = latex_content.replace('/home/caug/npcww/npcsh/figures/', 'figures/')
    
    # Add a simple bibliography if none exists
    if '\\begin{thebibliography}' not in latex_content and '\\end{document}' in latex_content:
        bibliography = """
\\begin{thebibliography}{9}
\\bibitem{citation1} Author, A. (2023). Title of the work. Journal Name, 10(2), 123-456.
\\bibitem{citation2} Researcher, B. (2022). Another relevant publication. Conference Proceedings, 789-012.
\\end{thebibliography}
"""
        latex_content = latex_content.replace('\\end{document}', f'{bibliography}\n\\end{{document}}')
    
    return latex_content

