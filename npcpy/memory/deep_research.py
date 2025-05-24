#deep_research 
import numpy as np 
from npcpy.npc_compiler import NPC 

from npcpy.memory.knowledge_graph import *
import os

from npcpy.data import sample_primary_directives 

def generate_random_npcs(num_npcs, model, provider):
    """
    Function Description:
        This function generates a list of random NPCs.
    Args:
        num_npcs (int): The number of NPCs to generate.
    Returns:
        List[NPC]: A list of generated NPCs.
    """
    # Initialize the list of NPCs
    npcs = []
    
    # Generate the NPCs
    for i, primary_directive in np.random.choice(sample_primary_directives, num_npcs):
        npc = NPC(primary_directive=primary_directive, 
                    model=model, 
                    provider=provider,)
        
        npcs.append(npc)
    return npcs

def generate_research_chain(request, npc, depth, memory=5, context=None):
    """
    Function Description:
        This function generates a research chain for the given NPC.
    Args:
        npc (NPC): The NPC for which to generate the research chain.
        depth (int): The depth of the research chain.
        context (str, optional): Additional context for the research chain. Defaults to None.
    Returns:
        List[str]: A list of generated research chains.
    """
    chain = []
    first_message = f'the user has requested that you research the following: {request}. Please begin providing a single specific question to ask.  '
    if context:
        first_message += f'The user also provided this context: {context}' 
    summary, question_raised = npc.search_and_ask(first_message)
    chain.append(first_message)
    chain.append(summary)
    chain.append(question_raised)
    
    
    
    for i in range(depth):
        memories = chain[-memory:]
        next_message = "\n".join(memories) + 'Last Search Summary: ' + summary + '. New Question'
        
        summary, question_raised = npc.search_and_ask(next_message)
        chain.append(next_message)
        chain.append(summary)
        chain.append(question_raised)
    return chain
        
                
def prune_chains():
    return 



# search and ask will have a check llm command more or less. 
def consolidate_research(chains, facts, groups, model=None, provider=None):
    prompt = f''' 
    You are a research advisor reviewing the notes of your research assisitants who have been working on a request.
    The results from their efforts are contained here:
    
    {chains}       
    
    Please identify the 3 most common ideas, the 3 most unusual ideas, and the 3 most important ideas. 
    
    
    Provide your response as a json object with a list of json objects for "most_common_ideas", "most_unusual_ideas" and "most_important_ideas".
    
    Each of those json objects within the sublists should be structured like so: 
        {{
            'idea': 'the idea',
            'source_npc': 'the name of the npc chain that provided this idea',
            'supporting_links': [
                'link1/to/local/file',
                'link2/to/web/site',                 
            ], 
            'supporting_evidence' : [
                'script x was run by npc and verified this idea ',
                'npc found evidence in site x y was run by npc and verified this idea ',
            ]
        }}   
        
    The links should be a list of links to the original sources of the information that were contained within the chains themselves.
    The supporting evidence should be a list of the evidence that was used to support the idea.
    '''
    ideas = get_llm_response(prompt, model=model, provider=provider, format='json')
    # build knowledge graph 
    
    groups = identify_groups(facts, model=model, provider=provider)

    prompt = f''' 
    You are a research advisor reviewing the notes of your research assisitants who have been working on a request.
    The results from their efforts are contained here:
    
    {facts}       
    
    Additionally, we have already found some common ideas and have produced the following groups:
    {groups}
    
    
    Please identify the 3 most common ideas, the 3 most unusual ideas, and the 3 most important ideas. 
    Provide your response as a json object with 3 lists each containing 3 items. 
    
    '''
    ideas_summarized = get_llm_response(prompt, model=model, provider=provider)
    
    return ideas, ideas_summarized
    


## ultimately wwell do the vector store in the main db. so when we eventually starti adding new facts well  do so by checking similar facts
# there and then if were doing the rag search well do a rag and then graph
