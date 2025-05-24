import os 
from sqlalchemy import create_engine
from npcpy.npc_compiler import NPC
from npcpy.llm_funcs import get_llm_response 
from npcpy.npc_sysenv import NPCSH_CHAT_MODEL, NPCSH_CHAT_PROVIDER, NPCSH_STREAM_OUTPUT, print_and_process_stream_with_markdown
import numpy as np
import random
from typing import List, Dict, Any, Optional, Union

import litellm 

def generate_random_events(
    npc, 
    model, 
    provider, 
    problem: str, 
    environment: Optional[str] = None,
    num_events: int = 3,
    **api_kwargs
) -> List[Dict[str, Any]]:
    """
    Generate random events that can occur in the wanderer's environment.
    
    Args:
        npc: The NPC object
        model: The LLM model to use
        provider: The provider to use
        problem: The current problem being explored
        environment: Optional description of the wandering environment. If None, one will be generated.
        num_events: Number of events to generate
        
    Returns:
        List of event dictionaries, each containing:
            - type: The type of event (encounter, discovery, obstacle, etc.)
            - description: Full description of the event
            - impact: How this might impact the problem-solving process
            - location: Where in the environment this occurs
    """
    # If no environment is provided, generate one based on the problem
    if not environment:
        env_prompt = f"""
        I need to create an imaginative environment for an AI to wander through while thinking about this problem:
        
        {problem}
        
        Please create a rich, metaphorical environment that could represent the conceptual space of this problem.
        The environment should:
        1. Have distinct regions or areas
        2. Include various elements, objects, and features
        3. Be metaphorically related to the problem domain
        4. Be described in 3-5 sentences
        
        Do not frame this as a response. Only provide the environment description directly.
        """
        
        env_response = get_llm_response(
            prompt=env_prompt,
            model=model,
            provider=provider,
            npc=npc,
            temperature=0.8,
            **api_kwargs
        )
        
        environment = env_response.get('response', '')
        if isinstance(environment, (list, dict)) or hasattr(environment, '__iter__') and not isinstance(environment, (str, bytes)):
            # Handle streaming response
            environment = ''.join([str(chunk) for chunk in environment])
            
        print(f"\nGenerated wandering environment:\n{environment}\n")
    
    # Define event types with their probability weights
    event_types = [
        {"type": "encounter", "weight": 0.25},  # Meeting someone/something
        {"type": "discovery", "weight": 0.2},   # Finding something unexpected
        {"type": "obstacle", "weight": 0.15},   # Something blocking progress
        {"type": "insight", "weight": 0.2},     # Sudden realization
        {"type": "shift", "weight": 0.1},       # Environment changing
        {"type": "memory", "weight": 0.1}       # Recalling something relevant
    ]
    
    # Calculate cumulative weights for weighted random selection
    cumulative_weights = []
    current_sum = 0
    for event in event_types:
        current_sum += event["weight"]
        cumulative_weights.append(current_sum)
    
    # Select event types based on their weights
    selected_event_types = []
    for _ in range(num_events):
        r = random.random() * current_sum
        for i, weight in enumerate(cumulative_weights):
            if r <= weight:
                selected_event_types.append(event_types[i]["type"])
                break
    
    # Generate the actual events based on selected types
    events_prompt = f"""
    I'm wandering through this environment while thinking about a problem:
    
    Environment: {environment}
    
    Problem: {problem}
    
    Please generate {num_events} detailed events that could occur during my wandering. For each event, provide:
    1. A detailed description of what happens (2-3 sentences)
    2. The specific location in the environment where it occurs
    3. How this event might impact my thinking about the problem
    
    The events should be of these types: {', '.join(selected_event_types)}
    
    Format each event as a dictionary with keys: "type", "description", "location", "impact"
    Return only the JSON list of events, not any other text.
    """
    
    events_response = get_llm_response(
        prompt=events_prompt,
        model=model,
        provider=provider,
        npc=npc,
        temperature=0.7,
        **api_kwargs
    )
    
    events_text = events_response.get('response', '')
    if isinstance(events_text, (list, dict)) or hasattr(events_text, '__iter__') and not isinstance(events_text, (str, bytes)):
        # Handle streaming response
        events_text = ''.join([str(chunk) for chunk in events_text])
    
    # Try to parse JSON, but have a fallback mechanism
    try:
        import json
        events = json.loads(events_text)
        if not isinstance(events, list):
            # Handle case where response isn't a list
            events = [{"type": "fallback", "description": events_text, "location": "unknown", "impact": "unknown"}]
    except:
        # If JSON parsing fails, create structured events from the text
        events = []
        event_chunks = events_text.split("\n\n")
        for i, chunk in enumerate(event_chunks[:num_events]):
            event_type = selected_event_types[i] if i < len(selected_event_types) else "unknown"
            events.append({
                "type": event_type,
                "description": chunk,
                "location": "Extracted from text",
                "impact": "See description"
            })
    
    # Ensure we have exactly num_events
    while len(events) < num_events:
        i = len(events)
        event_type = selected_event_types[i] if i < len(selected_event_types) else "unknown"
        events.append({
            "type": event_type,
            "description": f"An unexpected {event_type} occurred.",
            "location": "Unknown location",
            "impact": "The impact is unclear."
        })
    
    return events[:num_events]

def perform_single_wandering(problem, 
                            npc, 
                            model,
                            provider,
                            environment=None,
                            n_min=50,
                            n_max=200,
                            low_temp=0.5,
                            high_temp=1.9,
                            interruption_likelihood=1,
                            sample_rate=0.4,
                            n_high_temp_streams=5,
                            include_events=True,
                            num_events=3,
                            **api_kwargs):
    """
    Perform a single wandering session with high-temperature exploration and insight generation.
    
    Args:
        problem: The problem or question to explore
        npc: The NPC object
        model: LLM model to use
        provider: Provider to use
        environment: Optional description of wandering environment
        n_min, n_max: Min/max word count before switching to high temp
        low_temp, high_temp: Temperature settings for normal/exploratory thinking
        interruption_likelihood: Chance of interrupting a high-temp stream
        sample_rate: Portion of text to sample from high-temp streams
        n_high_temp_streams: Number of high-temperature exploration streams
        include_events: Whether to include random events in the wandering
        num_events: Number of events to generate if include_events is True
        
    Returns:
        tuple: (high_temp_streams, high_temp_samples, assistant_insight, events, environment)
    """
    # Generate environment and events if needed
    events = []
    if include_events:
        events = generate_random_events(
            npc=npc, 
            model=model, 
            provider=provider, 
            problem=problem,
            environment=environment,
            num_events=num_events,
            **api_kwargs
        )
        # Extract the environment if it was generated
        if not environment and events:
            # The environment was generated in the events function
            environment = get_llm_response(
                prompt=f"Summarize the environment described in these events: {events}",
                model=model,
                provider=provider,
                npc=npc,
                temperature=0.3,
                **api_kwargs
            ).get('response', '')
            
    # Initial response with low temperature
    event_context = ""
    if events:
        event_descriptions = [f"• {event['type'].capitalize()} at {event['location']}: {event['description']}" 
                             for event in events]
        event_context = "\n\nAs you wander, you encounter these events:\n" + "\n".join(event_descriptions)
    
    wandering_prompt = f"""
    You are wandering through a space while thinking about a problem.
    
    Environment: {environment or "An abstract conceptual space related to your problem"}
    
    Problem: {problem}{event_context}
    
    Begin exploring this problem in a focused way. Your thinking will later transition to more associative, creative modes.
    """
    
    response = get_llm_response(wandering_prompt, model=model, provider=provider, npc=npc, stream=True, temperature=low_temp, **api_kwargs)
    switch = np.random.randint(n_min, n_max)
    conversation_result = ""
    
    for chunk in response['response']:
        if len(conversation_result.split()) > switch:
            break
            
        if provider == "ollama":
            chunk_content = chunk["message"]["content"]
            if chunk_content:
                conversation_result += chunk_content
                print(chunk_content, end="")
        else:
            chunk_content = "".join(
                choice.delta.content
                for choice in chunk.choices
                if choice.delta.content is not None
            )
            if chunk_content:
                conversation_result += chunk_content
                print(chunk_content, end="")
                
    print('\n\n--- Beginning to wander ---\n')
    high_temp_streams = []
    high_temp_samples = []
    
    # Insert events between high-temp streams
    events_to_use = events.copy() if events else []
    
    for n in range(n_high_temp_streams):
        print(f'Stream #{n+1}')
        
        # Occasionally inject an event
        if events_to_use and random.random() < 0.7:  # 70% chance to inject event
            event = events_to_use.pop(0)
            print(f"\n[EVENT: {event['type']} at {event['location']}]\n{event['description']}\n")
            # Add the event to the prompt for the next stream
            event_prompt = f"\nSuddenly, {event['description']} This happens at {event['location']}. How does this influence your thinking?"
        else:
            event_prompt = ""
            
        stream_result = ''
        high_temp_response = get_llm_response(
            problem + conversation_result + event_prompt, 
            model=model, 
            provider=provider, 
            npc=npc, 
            stream=True, 
            temperature=high_temp, 
            **api_kwargs
        )
        
        for chunk in high_temp_response['response']:
            interruption = np.random.randint(0,100) < interruption_likelihood
            if interruption:
                high_temp_streams.append(stream_result)
                
                stream_result_list = stream_result.split()
                sample_size = int(len(stream_result_list) * sample_rate)
                if stream_result_list and sample_size > 0:
                    sample_indices = np.random.choice(len(stream_result_list), size=min(sample_size, len(stream_result_list)), replace=False)
                    sampled_stream_result = [stream_result_list[i] for i in sample_indices]
                    sampled_stream_result = ' '.join(sampled_stream_result)
                    high_temp_samples.append(sampled_stream_result)
                break
                
            if provider == "ollama" and 'hf.co' in model:
                chunk_content = chunk["message"]["content"]
                if chunk_content:
                    stream_result += chunk_content
                    print(chunk_content, end="")
            else:
                chunk_content = "".join(
                    choice.delta.content
                    for choice in chunk.choices
                    if choice.delta.content is not None
                )
                if chunk_content:
                    stream_result += chunk_content
                    print(chunk_content, end="")
                    
        if stream_result and stream_result not in high_temp_streams:
            high_temp_streams.append(stream_result)
            
    print('\n\n--- Wandering complete ---\n')
            
    # Combine the samples and evaluate with initial problem
    event_insights = ""
    if events:
        event_insights = "\n\nDuring your wandering, you encountered these events:\n" + "\n".join(
            [f"• {event['type']} at {event['location']}: {event['description']}" for event in events]
        )
    
    prompt = f'''
    Here are some random thoughts I had while wandering through {environment or "an abstract space"}:

    {high_temp_samples}{event_insights}

    I want you to evaluate these thoughts with respect to the following problem:
    {problem}

    Use the thoughts and events creatively and explicitly reference them in your response.
    Are there any specific items contained that are unusual that may suggest a new direction?
    Focus on these, and consider how the environment and events influenced your thinking.
    '''
    
    print("Extracted thought samples:")
    for i, sample in enumerate(high_temp_samples):
        print(f"Sample {i+1}: {sample}")
    print("\nGenerating insights from wandering...\n")
    
    response = get_llm_response(prompt, 
                                model=model, 
                                provider=provider,
                                npc=npc,
                                stream=NPCSH_STREAM_OUTPUT,
                                temperature=low_temp, 
                                **api_kwargs)
    assistant_reply = response['response']
    messages = response['messages']
    
    if NPCSH_STREAM_OUTPUT:
        assistant_reply = print_and_process_stream_with_markdown(response['response'],
                                                                model=model, 
                                                                provider=provider)
        messages.append({
            "role": "assistant",
            "content": assistant_reply,
        })
      
    return high_temp_streams, high_temp_samples, assistant_reply, events, environment

def enter_wander_mode(problem, 
                      npc, 
                      model,
                      provider,
                      environment=None,
                      n_min=50,
                      n_max=200,
                      low_temp=0.5,
                      high_temp=1.9,
                      interruption_likelihood=1,
                      sample_rate=0.4,
                      n_high_temp_streams=5,
                      include_events=True,
                      num_events=3,
                      **api_kwargs):
    """
    Wander mode is an exploratory mode where an LLM is given a task and they begin to wander through space.
    As they wander, they drift in between conscious thought and popcorn-like subconscious thought.
    The former is triggered by external stimuli and when these stimuli come we will capture the recent high entropy
    information from the subconscious popcorn thoughts and then consider them with respect to the initial problem at hand.

    The conscious evaluator will attempt to connect them, thus functionalizing the verse-jumping algorithm
    outlined by Everything Everywhere All at Once.
    
    Args:
        problem: The problem or question to explore
        npc: The NPC object
        model: LLM model to use
        provider: Provider to use
        environment: Optional description of wandering environment
        n_min, n_max: Min/max word count before switching to high temp
        low_temp, high_temp: Temperature settings for normal/exploratory thinking
        interruption_likelihood: Chance of interrupting a high-temp stream
        sample_rate: Portion of text to sample from high-temp streams
        n_high_temp_streams: Number of high-temperature exploration streams
        include_events: Whether to include random events in the wandering
        num_events: Number of events to generate in each session
    """
    current_problem = problem
    current_environment = environment
    wandering_history = []
    
    print(f"\n=== Starting Wander Mode with Problem: '{problem}' ===\n")
    if environment:
        print(f"Environment: {environment}\n")
    
    while True:
        print(f"\nCurrent exploration: {current_problem}\n")
        
        # Perform a single wandering session
        high_temp_streams, high_temp_samples, insight, events, env = perform_single_wandering(
            current_problem, 
            npc=npc,
            model=model,
            provider=provider,
            environment=current_environment,
            n_min=n_min,
            n_max=n_max,
            low_temp=low_temp,
            high_temp=high_temp,
            interruption_likelihood=interruption_likelihood,
            sample_rate=sample_rate,
            n_high_temp_streams=n_high_temp_streams,
            include_events=include_events,
            num_events=num_events,
            **api_kwargs
        )
        
        # If environment was generated, save it
        if not current_environment and env:
            current_environment = env
            
        # Save this wandering session
        wandering_history.append({
            "problem": current_problem,
            "environment": current_environment,
            "streams": high_temp_streams,
            "samples": high_temp_samples,
            "events": events,
            "insight": insight
        })
        
        # Ask user if they want to continue wandering
        print("\n\n--- Wandering session complete ---")
        print("Options:")
        print("1. Continue wandering with the same problem and environment")
        print("2. Continue wandering with a new related problem")
        print("3. Continue wandering in a new environment")
        print("4. Continue wandering with both new problem and environment")
        print("5. End wandering")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            # Continue with the same problem and environment
            pass
        elif choice == "2":
            # Continue with a modified problem
            print("\nBased on the insights gained, what new problem would you like to explore?")
            new_problem = input("New problem: ").strip()
            if new_problem:
                current_problem = new_problem
        elif choice == "3":
            # Continue with a new environment
            print("\nDescribe a new environment for your wandering:")
            new_env = input("New environment: ").strip()
            if new_env:
                current_environment = new_env
        elif choice == "4":
            # Change both problem and environment
            print("\nBased on the insights gained, what new problem would you like to explore?")
            new_problem = input("New problem: ").strip()
            print("\nDescribe a new environment for your wandering:")
            new_env = input("New environment: ").strip()
            if new_problem:
                current_problem = new_problem
            if new_env:
                current_environment = new_env
        else:
            # End wandering mode
            print("\n=== Exiting Wander Mode ===\n")
            break
    
    # Return the entire wandering history
    return wandering_history

def main():
    # Example usage
    import argparse    
    parser = argparse.ArgumentParser(description="Enter wander mode for chatting with an LLM")
    parser.add_argument("problem", type=str, help="Problem to solve")
    parser.add_argument("--model", default=NPCSH_CHAT_MODEL, help="Model to use")
    parser.add_argument("--provider", default=NPCSH_CHAT_PROVIDER, help="Provider to use")
    parser.add_argument("--environment", type=str, help="Wandering environment description")
    parser.add_argument("--no-events", action="store_true", help="Disable random events")
    parser.add_argument("--num-events", type=int, default=3, help="Number of events per wandering session")
    parser.add_argument("--files", nargs="*", help="Files to load into context")
    parser.add_argument("--stream", default="true", help="Use streaming mode")
    parser.add_argument("--npc", type=str, default=os.path.expanduser('~/.npcsh/npc_team/sibiji.npc'), help="Path to NPC file")
    
    args = parser.parse_args()
    
    npc = NPC(file=args.npc)
    print('npc: ', args.npc)
    print(args.stream)
    
    # Enter wander mode
    enter_wander_mode(
        args.problem,
        npc=npc,
        model=args.model,
        provider=args.provider,
        environment=args.environment,
        include_events=not args.no_events,
        num_events=args.num_events,
        files=args.files,
    )

if __name__ == "__main__":
    main()