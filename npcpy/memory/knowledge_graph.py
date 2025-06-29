import json
import os
import datetime

import numpy as np

try:
    import kuzu
except ModuleNotFoundError:
    print("kuzu not installed")
from typing import Optional, Dict, List, Union, Tuple, Any, Set


from npcpy.llm_funcs import get_llm_response
from npcpy.npc_compiler import NPC
import sqlite3


import random 
def safe_kuzu_execute(conn, query, error_message="Kuzu query failed"):
    """Execute a Kuzu query with proper error handling"""
    try:
        result = conn.execute(query)
        return result, None
    except Exception as e:
        error = f"{error_message}: {str(e)}"
        print(error)
        return None, error


def create_group(conn, name: str, metadata: str = ""):
    """Create a new group in the database with robust error handling"""
    if conn is None:
        print("Cannot create group: database connection is None")
        return False

    try:
        # Properly escape quotes in strings
        escaped_name = name.replace('"', '\\"')
        escaped_metadata = metadata.replace('"', '\\"')

        query = f"""
        CREATE (g:Groups {{
            name: "{escaped_name}",
            metadata: "{escaped_metadata}"
        }});
        """

        result, error = safe_kuzu_execute(
            conn, query, f"Failed to create group: {name}"
        )
        if error:
            return False

        print(f"Created group: {name}")
        return True
    except Exception as e:
        print(f"Error creating group {name}: {str(e)}")
        traceback.print_exc()
        return False


import traceback
def init_db(db_path: str, drop=False):
    """Initialize Kùzu database and create schema with generational tracking."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        db = kuzu.Database(db_path)
        conn = kuzu.Connection(db)
        print("Database connection established successfully")
        
        if drop:
            # Drop tables in reverse order of dependency
            safe_kuzu_execute(conn, "DROP TABLE IF EXISTS Contains")
            safe_kuzu_execute(conn, "DROP TABLE IF EXISTS EvolvedFrom") # New
            safe_kuzu_execute(conn, "DROP TABLE IF EXISTS Fact")
            safe_kuzu_execute(conn, "DROP TABLE IF EXISTS Groups")

        # Fact table remains the same
        safe_kuzu_execute(
            conn,
            """
            CREATE NODE TABLE IF NOT EXISTS Fact(
              content STRING,
              path STRING,
              recorded_at STRING,
              PRIMARY KEY (content)
            );
            """,
            "Failed to create Fact table",
        )

        # UPDATED Groups table with generational properties
        safe_kuzu_execute(
            conn,
            """
            CREATE NODE TABLE IF NOT EXISTS Groups(
              name STRING,
              metadata STRING,
              generation_created INT64,
              is_active BOOLEAN,
              PRIMARY KEY (name)
            );
            """,
            "Failed to create Groups table",
        )
        print("Groups table (with generation tracking) created or already exists.")
        
        # Contains relationship remains the same
        safe_kuzu_execute(
            conn,
            "CREATE REL TABLE IF NOT EXISTS Contains(FROM Groups TO Fact);",
            "Failed to create Contains relationship table",
        )
        
        # NEW EvolvedFrom relationship table
        safe_kuzu_execute(
            conn,
            """
            CREATE REL TABLE IF NOT EXISTS EvolvedFrom(
                FROM Groups TO Groups,
                event_type STRING,
                generation INT64,
                reason STRING
            );
            """,
            "Failed to create EvolvedFrom relationship table",
        )
        print("EvolvedFrom relationship table created or already exists.")

        return conn
    except Exception as e:
        print(f"Fatal error initializing database: {str(e)}")
        traceback.print_exc()
        return None
def extract_facts(
    text: str,
    model: str,
    provider: str,
    npc: NPC = None,
    context: str = ""
) -> List[str]:
    """Extract concise facts from text using LLM (as defined earlier)"""
    # Implementation from your previous code
    prompt = """Extract concise facts from this text.
        A fact is a piece of information that makes a statement about the world.
        A fact is typically a sentence that is true or false.
        Facts may be simple or complex. They can also be conflicting with each other, usually
        because there is some hidden context that is not mentioned in the text.
        In any case, it is simply your job to extract a list of facts that could pertain to
        an individual's personality.
        
        For example, if a message says:
            "since I am a doctor I am often trying to think up new ways to help people.
            Can you help me set up a new kind of software to help with that?"
        You might extract the following facts:
            - The individual is a doctor
            - They are helpful

        Another example:
            "I am a software engineer who loves to play video games. I am also a huge fan of the
            Star Wars franchise and I am a member of the 501st Legion."
        You might extract the following facts:
            - The individual is a software engineer
            - The individual loves to play video games
            - The individual is a huge fan of the Star Wars franchise
            - The individual is a member of the 501st Legion

        Another example:
            "The quantum tunneling effect allows particles to pass through barriers
            that classical physics says they shouldn't be able to cross. This has
            huge implications for semiconductor design."
        You might extract these facts:
            - Quantum tunneling enables particles to pass through barriers that are
              impassable according to classical physics
            - The behavior of quantum tunneling has significant implications for
              how semiconductors must be designed

        Another example:
            "People used to think the Earth was flat. Now we know it's spherical,
            though technically it's an oblate spheroid due to its rotation."
        You might extract these facts:
            - People historically believed the Earth was flat
            - It is now known that the Earth is an oblate spheroid
            - The Earth's oblate spheroid shape is caused by its rotation

        Another example:
            "My research on black holes suggests they emit radiation, but my professor
            says this conflicts with Einstein's work. After reading more papers, I
            learned this is actually Hawking radiation and doesn't conflict at all."
        You might extract the following facts:
            - Black holes emit radiation
            - The professor believes this radiation conflicts with Einstein's work
            - The radiation from black holes is called Hawking radiation
            - Hawking radiation does not conflict with Einstein's work

        Another example:
            "During the pandemic, many developers switched to remote work. I found
            that I'm actually more productive at home, though my company initially
            thought productivity would drop. Now they're keeping remote work permanent."
        You might extract the following facts:
            - The pandemic caused many developers to switch to remote work
            - The individual discovered higher productivity when working from home
            - The company predicted productivity would decrease with remote work
            - The company decided to make remote work a permanent option

        Thus, it is your mission to reliably extract lists of facts.

        Return a JSON object with the following structure:
            {
                "fact_list": "a list containing the facts where each fact is a string",
            }
    """ 
    if len(context) > 0:
        prompt+=f""" Here is some relevant user context: {context}"""

    prompt+="""    
    Return only the JSON object.
    Do not include any additional markdown formatting.
    """

    response = get_llm_response(
        prompt + f"HERE BEGINS THE TEXT TO INVESTIGATE:\n\nText: {text}",
        model=model,
        provider=provider,
        format="json",
    )
    response = response["response"]
    return response.get("fact_list", [])

def extract_mistakes(
    text: str,
    model: str,
    provider: str,
    npc: NPC = None,
    context: str = ""
) -> List[str]:
    """Extract mistakes from a conversation."""
    prompt = f"""
    Based on this text, identify and list any mistakes made.
    A mistake is a decision or action that resulted in an incorrect outcome
    or a misunderstanding.
    
    Text: {text}

    Return a JSON object:
    {{
        "mistakes": ["list of mistakes"]
    }}
    """
    response = get_llm_response(
        prompt,
        model=model,
        provider=provider,
        format="json",
        npc=npc,
    )
    return response["response"].get("mistakes", [])

def extract_lessons_learned(
    text: str,
    model: str,
    provider: str,
    npc: NPC = None,
    context: str = ""
) -> List[str]:
    """Extract lessons learned from the conversation."""
    prompt = f"""
    Based on this conversation, what lessons were learned?
    A lesson learned is a new understanding or insight.
    
    Text: {text}
    
    Return a JSON object:
    {{
        "lessons_learned": ["list of lessons"]
    }}
    """
    response = get_llm_response(
        prompt,
        model=model,
        provider=provider,
        format="json",
        npc=npc,
    )
    return response["response"].get("lessons_learned", [])


# --- Breathe (Context Condensation) ---
def breathe(
    messages: List[Dict[str, str]],
    model: str,
    provider: str,
    npc: NPC = None,
    context: str = None,
) -> Dict[str, Any]:
    """Condense the conversation context into a small set of key extractions."""
    if not messages:
        return {"output": {}, "messages": []}

    conversation_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

    # Extract facts, mistakes, and lessons learned
    facts = extract_facts(conversation_text, model, provider)
    mistakes = extract_mistakes(conversation_text, model, provider)
    lessons = extract_lessons_learned(conversation_text, model, provider)

    # Combine results for brevity
    output = {
        "facts": facts,
        "mistakes": mistakes,
        "lessons_learned": lessons
    }

    return {"output": output, "messages": []}

# --- Semantic Evolution (Sleep) ---
def semantic_evolution(
    facts: List[str],
    existing_leaf_groups: List[str], # These are groups from previous steps, not necessarily facts
    model: str,
    provider: str,
    npc: NPC = None,
    min_top: int = 4,
    max_top: int = 10,
    max_levels: int = 5
) -> Dict:
    """Build hierarchical group structure iteratively from facts and existing groups."""
    
    # Step 1: Generate initial group candidates from the new facts
    new_group_candidates = generate_group_candidates(facts, "facts", model, provider, npc)
    
    # Step 2: Combine with existing leaf groups and remove idempotents to get our starting set
    # These will be the bottom-most groups that we will then try to abstract upwards.
    initial_groups_for_hierarchy = remove_idempotent_groups(
        new_group_candidates + existing_leaf_groups, model, provider, npc
    )
    
    # Step 3: Build the hierarchy iteratively from these initial groups
    # We pass these initial groups, and build_full_hierarchy will abstract them upwards.
    hierarchy_data = build_full_hierarchy(
        initial_groups_for_hierarchy, # Use the cleaned list of groups
        model=model,
        provider=provider,
        npc=npc,
        min_top=min_top,
        max_top=max_top,
        max_levels=max_levels
    )
    
    return {
        "hierarchy": hierarchy_data,
        "leaf_groups": initial_groups_for_hierarchy, # These are the groups that were NOT abstracted further
    }
# --- Helper Functions for Hierarchy (unchanged from before) ---
def generate_group_candidates(
    items: List[str],
    item_type: str,
    model: str,
    provider: str,
    npc: NPC = None,
    n_passes: int = 3,
    subset_size: int = 10
) -> List[str]:
    """Generate candidate groups for items (facts or groups) based on core semantic meaning."""
    all_candidates = []
    
    for pass_num in range(n_passes):
        if len(items) > subset_size:
            item_subset = random.sample(items, min(subset_size, len(items)))
        else:
            item_subset = items
        
        # --- PROMPT MODIFICATION: Focus on semantic essence, avoid gerunds/adverbs, favor subjects ---
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
        {{
            "groups": ["list of specific, precise, and relevant group names"]
        }}
        """
        # --- END PROMPT MODIFICATION ---
        
        response = get_llm_response(
            prompt,
            model=model,
            provider=provider,
            format="json",
            npc=npc,
        )
        
        candidates = response["response"].get("groups", [])
        all_candidates.extend(candidates)
    print(all_candidates)
    return list(set(all_candidates))


def remove_idempotent_groups(
    group_candidates: List[str],
    model: str,
    provider: str,
    npc: NPC = None
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
    {{
        "distinct_groups": ["list of specific, precise, and distinct group names to keep"]
    }}
    """
    
    response = get_llm_response(
        prompt,
        model=model,
        provider=provider,
        format="json",
        npc=npc
    )
    
    print(response['response']['distinct_groups'])
    return response["response"]["distinct_groups"]


def build_hierarchy_dag(
    groups: List[str],
    model: str,
    provider: str,
    npc: NPC = None,
    max_levels: int = 3,
    target_top_count: int = 8,
    n_passes: int = 3,      # This is the number of times we query the LLM per level
    subset_size: int = 10   # This is how many groups we pass to the LLM at once
) -> Dict:
    """Build DAG hierarchy iteratively from bottom up, abstracting groups."""
    
    # Initialize DAG structure for the initial set of groups
    dag = {group: {"parents": set(), "children": set(), "level": 0} for group in groups}
    all_groups = set(groups)
    current_level_items = groups # Start with the provided groups (the bottom layer)
    level_num = 0
    
    # Keep abstracting until we have a manageable number of top-level groups
    # or reach max_levels. The condition checks the number of groups *currently* without parents.
    while len([g for g in all_groups if not dag.get(g, {}).get("parents")]) > target_top_count and level_num < max_levels:
        level_num += 1
        print(f"Too many top groups ({len([g for g in all_groups if not dag.get(g, {}).get('parents')])}), abstracting level {level_num}")
        
        # --- CRITICAL FIX: Re-introduce the multi-pass sampling for parent suggestions ---
        potential_parents = []
        # Multiple passes with resampling to explore different abstraction possibilities
        for pass_num in range(n_passes): # Iterate n_passes times
            # Sample a subset of groups from the current level for the LLM prompt
            if len(current_level_items) > subset_size:
                # Use a seed based on level and pass to ensure different samples each time
                random.seed(level_num * 10 + pass_num) 
                group_subset = random.sample(current_level_items, min(subset_size, len(current_level_items)))
            else:
                group_subset = current_level_items # Use all if subset_size is larger than available groups
                
            # Prompt the LLM to suggest parent categories for this subset of groups
            prompt = f"""
            What are broader parent categories that could contain these groups?
            Suggest 1-3 broader categories. Make them distinct and meaningful.

            Groups: {json.dumps(group_subset)}

            Return JSON:
            {{
                "parents": ["list of parent categories"]
            }}
            """

            response = get_llm_response(
                prompt, model=model, provider=provider, format="json", npc=npc
            )

            parents = response["response"].get("parents", [])
            potential_parents.extend(parents)
        
        distinct_parents = remove_idempotent_groups(potential_parents, model, provider, npc)
        
        if not distinct_parents: # Stop if no new abstract groups were generated
            print("No distinct parent groups generated, stopping abstraction.")
            break

        # Add these distinct parent groups to the DAG and update relationships
        new_groups_for_next_level = set()
        for parent in distinct_parents:
            if parent not in dag: # If this is a completely new abstract group
                dag[parent] = {
                    "parents": set(), # These new parents have no parents yet in this round
                    "children": set(current_level_items), # The groups from the previous level are their children
                    "level": level_num
                }
                all_groups.add(parent)
                new_groups_for_next_level.add(parent)
            else: # If the parent group already exists (e.g., from a different branch)
                # Update its children to include the current level's groups
                dag[parent]["children"].update(current_level_items)
            
            # Update parent relationship for the children from the previous level
            for child in current_level_items:
                dag[child]["parents"].add(parent)
                
        # The newly found parents become the input for the next abstraction level
        current_level_items = list(new_groups_for_next_level) 

    # After the loop, identify the final top groups (those with no parents in the constructed DAG)
    top_groups_final = [g for g in all_groups if not dag.get(g, {}).get("parents")]

    return {
        "dag": dag,
        "top_groups": top_groups_final,
        "leaf_groups": groups, # The initial set of groups passed in, which are the base for the hierarchy
        "max_level": level_num
    }
    
    
    

def build_full_hierarchy(
    leaf_groups: List[str],
    model: str,
    provider: str,
    npc: NPC = None,
    min_top: int = 4,
    max_top: int = 10,
    max_levels: int = 5
) -> Dict:
    """Build full hierarchy from initial leaf groups up to top groups."""
    # Step 1: Get initial distinct groups from facts (already done by caller if passing leaf_groups)
    # If leaf_groups is empty, we might want to generate them from facts first, but for now, assume they are provided.
    
    # Step 2: Build the DAG structure, abstracting upwards until we have <= max_top groups
    hierarchy = build_hierarchy_dag(
        leaf_groups, model, provider, npc, max_levels, max_top, n_passes=3, subset_size=10
    )
    
    return hierarchy

def assign_fact_to_dag(fact: str, dag_data: Dict, model: str, provider: str, npc: NPC = None) -> Dict:
    """Assign fact to DAG starting from top-level abstract concepts, traversing down."""
    
    top_groups = dag_data.get("top_groups", [])
    if not top_groups: # Handle case where no hierarchy was built
        print(f"Warning: No top groups found for fact: {fact}. Assigning to all leaf groups.")
        # Fallback: assign to leaf groups if no hierarchy exists
        leaf_groups = dag_data.get("leaf_groups", [])
        if not leaf_groups: return {'top_level_groups': [], 'all_groups': [], 'hierarchy_paths': []}
        assignments = get_fact_assignments(fact, leaf_groups, model, provider, npc)
        return {'top_level_groups': assignments, 'all_groups': assignments, 'hierarchy_paths': [f"{g}" for g in assignments]}

    print(f"assign_fact_to_dag: Assigning fact: {fact[:50]}...")
    
    # Start assignment process from the top-level groups
    top_level_assignments = get_fact_assignments(fact, top_groups, model, provider, npc)
    
    # Initialize tracking for all relevant groups and paths
    all_assigned_groups = set(top_level_assignments)
    current_level_to_process = top_level_assignments # Groups at the current level we need to check children for
    hierarchy_paths = [] # Stores the path from top-level to the most specific assigned group

    # Store path segments as we go down
    path_segments = {group: [group] for group in top_level_assignments}

    # Traverse down the hierarchy level by level
    # We continue as long as there are groups at the current level that are assigned to the fact
    # and these groups have children defined in the DAG.
    processed_groups_in_level = set() # To avoid infinite loops if DAG has cycles (though should be acyclic)

    while current_level_to_process:
        next_level_to_process = set()
        
        for current_group in current_level_to_process:
            # Prevent reprocessing the same group in the same level traversal
            if current_group in processed_groups_in_level:
                continue
            processed_groups_in_level.add(current_group)

            # Get children of the current group
            children = dag_data["dag"].get(current_group, {}).get("children", set())
            
            if children:
                # Get assignments for children
                child_assignments = get_fact_assignments(fact, list(children), model, provider, npc)
                
                # If the fact belongs to any children, add them to the next level to process
                if child_assignments:
                    next_level_to_process.update(child_assignments)
                    all_assigned_groups.update(child_assignments)
                    
                    # Update path segments for newly assigned children
                    for assigned_child in child_assignments:
                        # Append the child to the path of its parent
                        if current_group in path_segments:
                            path_segments[assigned_child] = path_segments[current_group] + [assigned_child]
                        else: # Should not happen if logic is correct, but as a safeguard
                            path_segments[assigned_child] = [assigned_child]
        
        # Add completed paths to our final list
        for group, path in path_segments.items():
            if group in current_level_to_process and group not in processed_groups_in_level: # If it was processed and assigned
                if path not in hierarchy_paths:
                    hierarchy_paths.append(' → '.join(path))

        current_level_to_process = next_level_to_process
        processed_groups_in_level = set() # Reset for the next level

    # Ensure all paths are captured even if a fact is only assigned to top-level groups
    for group in top_level_assignments:
        if group in path_segments and ' → '.join(path_segments[group]) not in hierarchy_paths:
             hierarchy_paths.append(' → '.join(path_segments[group]))


    return {
        "top_level_groups": top_level_assignments,
        "all_groups": list(all_assigned_groups),
        "hierarchy_paths": hierarchy_paths
    }

def process_text_with_hierarchy(
    text: str,
    model: str,
    provider: str,
    db_path: str,
    npc: NPC = None,
    existing_knowledge_graph: Optional[Dict] = None
) -> Dict:
    """Full processing pipeline with hierarchical grouping"""
    print('process_text_with_hierarchy: Starting processing')
    facts = extract_facts(text, model, provider, npc)
    print(f'process_text_with_hierarchy: Extracted Facts: {facts}')
    
    conn = init_db(db_path, drop=False)
    if conn is None:
        return None

    leaf_groups = existing_knowledge_graph.get("leaf_groups", []) if existing_knowledge_graph else []
    
    # Build the hierarchy from the extracted facts (and any existing leaf groups)
    hierarchy_data = build_full_hierarchy(facts + leaf_groups, model, provider, npc) # Pass facts to generate initial groups

    assignments = {}
    for fact in facts:
        # Assign facts using the top-down traversal logic
        assignment = assign_fact_to_dag(fact, hierarchy_data, model, provider, npc)
        
        # Store fact and its assignments in Kuzu
        store_success = store_fact_and_group(conn, fact, assignment["all_groups"], "")
        if not store_success:
            print(f'process_text_with_hierarchy: Failed to store fact: {fact}')
        
        assignments[fact] = assignment
    
    conn.close()
    
    print('process_text_with_hierarchy: Finished Processing')
    return {
        'facts': facts,
        'leaf_groups': hierarchy_data.get("leaf_groups", []), # This should be the *final* leaf groups after abstraction
        'hierarchy': hierarchy_data,
        'assignments': assignments
    }



def store_fact_and_group(conn: kuzu.Connection, fact: str,
                        groups: List[str], path: str) -> bool:
    """Insert a fact into the database along with its groups"""
    if not conn:
        print("store_fact_and_group: Database connection is None")
        return False
    
    print(f"store_fact_and_group: Storing fact: {fact}, with groups:"
          f" {groups}") # DEBUG
    try:
        # Insert the fact
        insert_success = insert_fact(conn, fact, path) # Capture return
        if not insert_success:
            print(f"store_fact_and_group: Failed to insert fact: {fact}")
            return False
        
        # Assign fact to groups
        for group in groups:
            assign_success = assign_fact_to_group_graph(conn, fact, group)
            if not assign_success:
                print(f"store_fact_and_group: Failed to assign fact"
                      f" {fact} to group {group}")
                return False
        
        return True
    except Exception as e:
        print(f"store_fact_and_group: Error storing fact and group: {e}")
        traceback.print_exc()
        return False

def insert_fact(conn, fact: str, path: str) -> bool:
    """Insert a fact into the database with robust error handling"""
    if conn is None:
        print("insert_fact: Cannot insert fact:"
              " database connection is None")
        return False

    try:
        # Properly escape quotes in strings
        escaped_fact = fact.replace('"', '\\"')
        escaped_path = os.path.expanduser(path).replace('"', '\\"')

        # Generate timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"insert_fact: Attempting to insert fact: {fact}") #DEBUG

        # Begin transaction
        safe_kuzu_execute(conn, "BEGIN TRANSACTION")

        # Check if fact already exists
        check_query = f"""
        MATCH (f:Fact {{content: "{escaped_fact}"}})
        RETURN f
        """

        result, error = safe_kuzu_execute(
            conn, check_query, "insert_fact: Failed to check if fact exists"
        )
        if error:
            safe_kuzu_execute(conn, "ROLLBACK")
            print(f"insert_fact: Error checking if fact exists: {error}")
            return False

        # Insert fact if it doesn't exist
        if not result.has_next():
            insert_query = f"""
            CREATE (f:Fact {{
                content: "{escaped_fact}",
                path: "{escaped_path}",
                recorded_at: "{timestamp}"
            }})
            """

            result, error = safe_kuzu_execute(
                conn, insert_query, "insert_fact: Failed to insert fact"
            )
            if error:
                safe_kuzu_execute(conn, "ROLLBACK")
                print(f"insert_fact: Error inserting fact: {error}")
                return False

        # Commit transaction
        safe_kuzu_execute(conn, "COMMIT")
        print(f"insert_fact: Successfully inserted/found fact: {fact}")
        return True
    except Exception as e:
        print(f"insert_fact: Error inserting fact: {str(e)}")
        traceback.print_exc()
        safe_kuzu_execute(conn, "ROLLBACK")
        return False

def assign_fact_to_group_graph(conn, fact: str, group: str) -> bool:
    """Create a relationship between a fact and a group with robust
       error handling"""
    if conn is None:
        print("assign_fact_to_group_graph: Cannot assign fact to group:"
              " database connection is None")
        return False

    try:
        # Properly escape quotes in strings
        escaped_fact = fact.replace('"', '\\"')
        escaped_group = group.replace('"', '\\"')

        print(f"assign_fact_to_group_graph: Assigning fact: {fact} to group:"
              f" {group}") #DEBUG

        # Check if both fact and group exist before creating relationship
        check_query = f"""
        MATCH (f:Fact {{content: "{escaped_fact}"}})
        RETURN f
        """

        result, error = safe_kuzu_execute(
            conn, check_query, "assign_fact_to_group_graph: Failed to check"
                               " if fact exists"
        )
        if error or not result.has_next():
            print(f"assign_fact_to_group_graph: Fact not found: {fact}")
            return False

        check_query = f"""
        MATCH (g:Groups {{name: "{escaped_group}"}})
        RETURN g
        """

        result, error = safe_kuzu_execute(
            conn, check_query, "assign_fact_to_group_graph: Failed to check"
                               " if group exists"
        )
        if error or not result.has_next():
            print(f"assign_fact_to_group_graph: Group not found: {group}")
            return False

        # Create relationship
        query = f"""
        MATCH (f:Fact), (g:Groups)
        WHERE f.content = "{escaped_fact}" AND g.name = "{escaped_group}"
        CREATE (g)-[:Contains]->(f)
        """

        result, error = safe_kuzu_execute(
            conn, query, "assign_fact_to_group_graph: Failed to create"
                         " relationship: {error}"
        )
        if error:
            print(f"assign_fact_to_group_graph: Failed to create"
                  f" relationship: {error}")
            return False

        print(f"assign_fact_to_group_graph: Assigned fact to group:"
              f" {group}")
        return True
    except Exception as e:
        print(f"assign_fact_to_group_graph: Error assigning fact to group:"
              f" {str(e)}")
        traceback.print_exc()
        return False
    
def get_fact_assignments(
    fact: str,
    groups: List[str],
    model: str,
    provider: str,
    npc: NPC = None
) -> List[str]:
    """Get direct group assignments for a fact"""

    prompt = f"""Which of these groups does this fact belong to?
    Select ALL that apply.
    
    Fact: {fact}
    Groups: {json.dumps(groups)}
    
    Return JSON:
    {{
        "selected_groups": ["list of relevant groups"]
    }}
    """
    response = get_llm_response(prompt, 
                                model=model, 
                                provider=provider,
                                format="json", 
                                npc=npc)
    return response["response"]["selected_groups"]
def get_ancestor_groups(group: str, dag: Dict) -> Set[str]:
    """Get all ancestor groups in the DAG for a given group."""
    ancestors = set()
    queue = [group]
    
    while queue:
        current = queue.pop(0)
        # Ensure current group exists in DAG and has parents
        if current in dag and dag[current].get("parents"):
            for parent in dag[current]["parents"]:
                if parent not in ancestors:
                    ancestors.add(parent)
                    queue.append(parent)
    return ancestors


# --- Main Process Flow ---
def process_text_with_hierarchy(
    text: str,
    model: str,
    provider: str,
    db_path: str,
    npc: NPC = None,
    existing_knowledge_graph: Optional[Dict] = None
) -> Dict:
    """Full processing pipeline with hierarchical grouping"""
    print("process_text_with_hierarchy: Starting processing")
    # Step 1: Extract facts from text
    facts = extract_facts(text, model, provider, npc)
    print(f"process_text_with_hierarchy: Extracted Facts: {facts}")
    
    # Build the DB connection
    conn = init_db(db_path, drop=False)
    if conn is None:
        return None

    # Use the existing leaf_groups for semantic evolution
    if existing_knowledge_graph:
        leaf_groups = existing_knowledge_graph.get("leaf_groups", [])
    else:
        leaf_groups = []
    
    # Build the hierarchy from the database
    hierarchy_data = build_full_hierarchy(leaf_groups, model, provider, npc)

    # Step 3: Assign facts to hierarchy
    assignments = {}
    for fact in facts:
        assignment = assign_fact_to_dag(fact, hierarchy_data, model, provider, npc)
        # Store fact and group in kuzu
        store_success = store_fact_and_group(conn, fact, assignment["all_groups"], "")
        if not store_success:
            print(f"process_text_with_hierarchy: Failed to store fact: {fact}")
        assignments[fact] = assignment
    
    conn.close()
    
    print("process_text_with_hierarchy: Finished Processing")
    return {
        "facts": facts,
        "leaf_groups": leaf_groups,
        "hierarchy": hierarchy_data,
        "assignments": assignments
    }


#--- Kuzu Database integration ---
def store_fact_and_group(conn: kuzu.Connection, fact: str, groups: List[str], path: str) -> bool:
    """Insert a fact into the database along with its groups"""
    if not conn:
        print("store_fact_and_group: Database connection is None")
        return False
    
    print(f"store_fact_and_group: Storing fact: {fact}, with groups: {groups}") # DEBUG
    try:
        # Insert the fact
        insert_success = insert_fact(conn, fact, path) # Capture return value
        if not insert_success:
            print(f"store_fact_and_group: Failed to insert fact: {fact}") #DEBUG
            return False
        
        # Assign fact to groups
        for group in groups:
            assign_success = assign_fact_to_group_graph(conn, fact, group)
            if not assign_success:
                print(f"store_fact_and_group: Failed to assign fact {fact} to group {group}") #DEBUG
                return False
        
        return True
    except Exception as e:
        print(f"store_fact_and_group: Error storing fact and group: {e}")
        traceback.print_exc()
        return False
    
        
# ---Database and other helper methods---
def safe_kuzu_execute(conn, query, error_message="Kuzu query failed"):
    """Execute a Kuzu query with proper error handling"""
    try:
        result = conn.execute(query)
        return result, None
    except Exception as e:
        error = f"{error_message}: {str(e)}"
        print(error)
        return None, error


def test_hierarchical_knowledge_graph():
    """Test the full hierarchical knowledge graph implementation"""
    text = """
    npcsh is a Python-based command-line tool for integrating LLMs into daily workflows.
    It features a smart interpreter that understands natural language commands.
    The tool remembers command history and can reference previous commands.
    It supports creating custom NPCs with specific personalities and directives.
    Advanced customization is possible through configuration files.
    """
    
    # Initialize with model and provider
    model = "gpt-4o-mini"
    provider = "openai"
    
    # Create knowledge graph
    kg = create_knowledge_graph(text, model, provider)
    
    # Print results
    print("FACTS:")
    for i, fact in enumerate(kg["facts"]):
        print(f"{i+1}. {fact}")
    
    print("\nHIERARCHY LEVELS:")
    for level in range(kg["hierarchy"]["top_level"], -1, -1):
        groups = kg["hierarchy"][f"level_{level}"]["groups"]
        print(f"Level {level} ({len(groups)} groups):")
        for group in groups:
            print(f"  - {group}")
    
    print("\nASSIGNMENTS:")
    for fact, assignment in kg["assignments"].items():
        print(f"\nFact: {fact}")
        print("Assignments by level:")
        for level, groups in assignment["all_assignments"].items():
            print(f"  Level {level}: {groups}")

def find_similar_groups(
    conn,
    fact: str,  # Ensure fact is passed as a string
    model: str = "llama3.2",
    provider: str = "ollama",
    npc: NPC = None,
) -> List[str]:
    """Find existing groups that might contain this fact"""
    response = conn.execute(f"MATCH (g:Groups) RETURN g.name;")  # Execute query
    print(response)
    print(type(response))
    print(dir(response))
    groups = response.fetch_as_df()
    print(f"Groups: {groups}")
    if not groups:
        return []

    prompt = """Given a fact and a list of groups, determine which groups this fact belongs to.
        A fact should belong to a group if it is semantically related to the group's theme or purpose.
        For example, if a fact is "The user loves programming" and there's a group called "Technical_Interests",
        that would be a match.

    Return a JSON object with the following structure:
        {
            "group_list": "a list containing the names of matching groups"
        }

    Return only the JSON object.
    Do not include any additional markdown formatting.
    """

    response = get_llm_response(
        prompt + f"\n\nFact: {fact}\nGroups: {json.dumps(groups)}",
        model=model,
        provider=provider,
        format="json",
        npc=npc,
    )
    response = response["response"]
    return response["group_list"]


def identify_groups(
    facts: List[str],
    model: str = "llama3.2",
    provider: str = "ollama",
    npc: NPC = None,
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
    )
    return response["response"]["groups"]


def assign_groups_to_fact(
    fact: str,
    groups: List[str],
    model: str = "llama3.2",
    provider: str = "ollama",
    npc: NPC = None,
) -> Dict[str, List[str]]:
    """Assign facts to the identified groups"""
    prompt = f"""Given this fact, assign it to any relevant groups.

    A fact can belong to multiple groups if it fits.

    Here is the fact: {fact}

    Here are the groups: {groups}

    Return a JSON object with the following structure:
        {{
            "groups": ["list of group names"]
        }}

    Do not include any additional markdown formatting or leading json characters.


    """

    response = get_llm_response(
        prompt,
        model=model,
        provider=provider,
        format="json",
        npc=npc,
    )
    return response["response"]


def save_facts_to_db(
    conn, facts: List[str], path: str, batch_size: int
):
    """Save a list of facts to the database in batches"""
    for i in range(0, len(facts), batch_size):
        batch = facts[i : i + batch_size]
        print(f"\nProcessing batch {i//batch_size + 1} ({len(batch)} facts)")

        # Process each fact in the batch
        for fact in batch:
            try:
                print(f"Inserting fact: {fact}")
                print(f"With path: {path}")
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"With recorded_at: {timestamp}")

                insert_fact(conn, fact, path)
                print("Success!")
            except Exception as e:
                print(f"Failed to insert fact: {fact}")
                print(f"Error: {e}")
                continue

        print(f"Completed batch {i//batch_size + 1}")


def process_text(
    db_path: str,
    text: str,
    path: str,
    model: str = "llama3.2",
    provider: str = "ollama",
    npc: NPC = None,
    batch_size: int = 5,
    conn=None,
):
    """Process text and add extracted facts to the database with robust error handling"""

    try:
        # Initialize database
        if conn is None:
            conn = init_db(db_path, drop=False)

            return []

        # Extract facts
        facts = extract_facts(text, model=model, provider=provider, npc=npc)
        if not facts:
            print("No facts extracted")
            return []

        print(f"Extracted {len(facts)} facts")
        for fact in facts:
            print(f"- {fact}")

        # Process facts in batches
        for i in range(0, len(facts), batch_size):
            batch = facts[i : i + batch_size]
            print(f"\nProcessing batch {i//batch_size + 1} ({len(batch)} facts)")

            for fact in batch:
                try:
                    print(f"Inserting fact: {fact}")
                    success = insert_fact(conn, fact, path)
                    if success:
                        print("Success!")
                    else:
                        print("Failed to insert fact")
                except Exception as e:
                    print(f"Error processing fact: {str(e)}")
                    traceback.print_exc()

            print(f"Completed batch {i//batch_size + 1}")

        return facts
    except Exception as e:
        print(f"Error processing text: {str(e)}")
        traceback.print_exc()
        return []


import networkx as nx
import matplotlib.pyplot as plt


def visualize_graph(conn):
    """Visualize the knowledge graph using networkx"""
    # Create a networkx graph
    G = nx.DiGraph()

    # Get all facts and groups with their relationships
    facts_result = conn.execute("MATCH (f:Fact) RETURN f.content;").get_as_df()
    facts = [row["f.content"] for index, row in facts_result.iterrows()]

    groups_result = conn.execute("MATCH (g:Groups) RETURN g.name;").get_as_df()
    groups = [row["g.name"] for index, row in groups_result.iterrows()]

    relationships_result = conn.execute(
        """
        MATCH (g:Groups)-[r:Contains]->(f:Fact)
        RETURN g.name, f.content;
    """
    ).get_as_df()

    # Add nodes with different colors for facts and groups
    for fact in facts:
        G.add_node(fact, node_type="fact")
    for group in groups:
        G.add_node(group, node_type="group")

    # Add edges from relationships
    for index, row in relationships_result.iterrows():
        G.add_edge(row["g.name"], row["f.content"])  # group name -> fact content

    # Set up the visualization
    plt.figure(figsize=(20, 12))
    pos = nx.spring_layout(G, k=2, iterations=50)

    # Draw groups (larger nodes, distinct color)
    group_nodes = [
        n for n, attr in G.nodes(data=True) if attr.get("node_type") == "group"
    ]
    nx.draw_networkx_nodes(
        G, pos, nodelist=group_nodes, node_color="lightgreen", node_size=3000, alpha=0.7
    )

    # Draw facts (smaller nodes, different color)
    fact_nodes = [
        n for n, attr in G.nodes(data=True) if attr.get("node_type") == "fact"
    ]
    nx.draw_networkx_nodes(
        G, pos, nodelist=fact_nodes, node_color="lightblue", node_size=2000, alpha=0.5
    )

    # Draw edges with arrows
    nx.draw_networkx_edges(G, pos, edge_color="gray", arrows=True, arrowsize=20)

    # Add labels with different sizes for groups and facts
    group_labels = {node: node for node in group_nodes}
    fact_labels = {
        node: node[:50] + "..." if len(node) > 50 else node for node in fact_nodes
    }

    nx.draw_networkx_labels(G, pos, group_labels, font_size=10, font_weight="bold")
    nx.draw_networkx_labels(G, pos, fact_labels, font_size=8)

    plt.title("Knowledge Graph: Groups and Facts", pad=20, fontsize=16)
    plt.axis("off")
    plt.tight_layout()

    # Print statistics
    print("\nKnowledge Graph Statistics:")
    print(f"Number of facts: {len(facts)}")
    print(f"Number of groups: {len(groups)}")
    print(f"Number of relationships: {len(relationships_result)}")

    print("\nGroups:")
    for g in groups:
        related_facts = [
            row["f.content"]
            for index, row in relationships_result.iterrows()
            if row["g.name"] == g
        ]
        print(f"\n{g}:")
        for f in related_facts:
            print(f"  - {f}")

    plt.show()


def store_fact_with_embedding(
    collection, fact: str, metadata: dict, embedding: List[float]
) -> str:
    """Store a fact with its pre-generated embedding in Chroma DB

    Args:
        collection: Chroma collection
        fact: The fact text
        metadata: Dictionary with metadata (path, source, timestamp, etc.)
        embedding: Pre-generated embedding vector from get_embeddings

    Returns:
        ID of the stored fact
    """
    try:
        # Generate a deterministic ID from the fact content
        import hashlib

        fact_id = hashlib.md5(fact.encode()).hexdigest()

        # Store document with pre-generated embedding
        collection.add(
            documents=[fact],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[fact_id],
        )

        return fact_id
    except Exception as e:
        print(f"Error storing fact in Chroma: {e}")
        return None


def find_similar_facts_chroma(
    collection,
    query: str,
    query_embedding: List[float],
    n_results: int = 5,
    metadata_filter: Optional[Dict] = None,
) -> List[Dict]:
    """Find facts similar to the query using pre-generated embedding

    Args:
        collection: Chroma collection
        query: Query text (for reference only)
        query_embedding: Pre-generated embedding from get_embeddings
        n_results: Number of results to return
        metadata_filter: Optional filter for metadata fields

    Returns:
        List of dictionaries with results
    """
    try:
        # Perform query with optional metadata filtering
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=metadata_filter,
        )

        # Format results
        formatted_results = []
        for i, doc in enumerate(results["documents"][0]):
            formatted_results.append(
                {
                    "fact": doc,
                    "metadata": results["metadatas"][0][i],
                    "id": results["ids"][0][i],
                    "distance": (
                        results["distances"][0][i] if "distances" in results else None
                    ),
                }
            )

        return formatted_results
    except Exception as e:
        print(f"Error searching in Chroma: {e}")
        return []


def process_text_with_chroma(
    kuzu_db_path: str,
    chroma_db_path: str,
    text: str,
    path: str,
    model: str ,
    provider: str ,
    embedding_model: str ,
    embedding_provider: str ,
    npc: NPC = None,
    batch_size: int = 5,
):
    """Process text and store facts in both Kuzu and Chroma DB

    Args:
        kuzu_db_path: Path to Kuzu graph database
        chroma_db_path: Path to Chroma vector database
        text: Input text to process
        path: Source path or identifier
        model: LLM model to use
        provider: LLM provider
        embedding_model: Model to use for embeddings
        npc: Optional NPC instance
        batch_size: Batch size for processing

    Returns:
        List of extracted facts
    """
    # Initialize databases
    kuzu_conn = init_db(kuzu_db_path, drop=False)
    chroma_client, chroma_collection = setup_chroma_db( 
        "knowledge_graph",
        "Facts extracted from various sources",
        chroma_db_path
    )

    # Extract facts
    facts = extract_facts(text, model=model, provider=provider, npc=npc)

    # Process extracted facts
    for i in range(0, len(facts), batch_size):
        batch = facts[i : i + batch_size]
        print(f"\nProcessing batch {i//batch_size + 1} ({len(batch)} facts)")

        # Generate embeddings for the batch using npcpy.llm_funcs.get_embeddings
        from npcpy.llm_funcs import get_embeddings

        batch_embeddings = get_embeddings(
            batch,
        )

        for j, fact in enumerate(batch):
            print(f"Processing fact: {fact}")
            embedding = batch_embeddings[j]

            # Check for similar facts in Chroma before inserting
            similar_facts = find_similar_facts_chroma(
                chroma_collection, fact, query_embedding=embedding, n_results=3
            )

            if similar_facts:
                print(f"Similar facts found:")
                for result in similar_facts:
                    print(f"  - {result['fact']} (distance: {result['distance']})")
                # Note: Could implement a similarity threshold here to skip highly similar facts

            # Prepare metadata
            metadata = {
                "path": path,
                "timestamp": datetime.now().isoformat(),
                "source_model": model,
                "source_provider": provider,
            }

            # Insert into Kuzu graph DB
            kuzu_success = insert_fact(kuzu_conn, fact, path)

            # Insert into Chroma vector DB if Kuzu insert was successful
            if kuzu_success:
                chroma_id = store_fact_with_embedding(
                    chroma_collection, fact, metadata, embedding
                )
                if chroma_id:
                    print(f"Successfully saved fact with ID: {chroma_id}")
                else:
                    print(f"Failed to save fact to Chroma")
            else:
                print(f"Failed to save fact to Kuzu graph")

    # Close Kuzu connection
    kuzu_conn.close()

    return facts


def hybrid_search_with_chroma(
    kuzu_conn,
    chroma_collection,
    query: str,
    group_filter: Optional[List[str]] = None,
    top_k: int = 5,
    metadata_filter: Optional[Dict] = None,
) -> List[Dict]:
    """Perform hybrid search using both Chroma vector search and Kuzu graph relationships

    Args:
        kuzu_conn: Connection to Kuzu graph database
        chroma_collection: Chroma collection for vector search
        query: Search query text
        group_filter: Optional list of groups to filter by in graph
        top_k: Number of results to return
        metadata_filter: Optional metadata filter for Chroma search
        embedding_model: Model to use for embeddings
        provider: Provider for embeddings

    Returns:
        List of dictionaries with combined results
    """
    # Get embedding for query using npcpy.llm_funcs.get_embeddings
    from npcpy.llm_funcs import get_embeddings

    query_embedding = get_embeddings([query])[0]

    # Step 1: Find similar facts using Chroma vector search
    vector_results = find_similar_facts_chroma(
        chroma_collection,
        query,
        query_embedding=query_embedding,
        n_results=top_k,
        metadata_filter=metadata_filter,
    )

    # Extract just the fact texts from vector results
    vector_facts = [result["fact"] for result in vector_results]

    # Step 2: Expand context using graph relationships
    expanded_results = []

    # Add vector search results
    for result in vector_results:
        expanded_results.append(
            {
                "fact": result["fact"],
                "source": "vector_search",
                "relevance": "direct_match",
                "distance": result["distance"],
                "metadata": result["metadata"],
            }
        )

    # For each vector-matched fact, find related facts in the graph
    for fact in vector_facts:
        try:
            # Safely escape fact text for Kuzu query
            escaped_fact = fact.replace('"', '\\"')

            # Find groups containing this fact
            group_result = kuzu_conn.execute(
                f"""
                MATCH (g:Groups)-[:Contains]->(f:Fact)
                WHERE f.content = "{escaped_fact}"
                RETURN g.name
                """
            ).get_as_df()

            # Extract group names
            fact_groups = [row["g.name"] for _, row in group_result.iterrows()]

            # Apply group filter if provided
            if group_filter:
                fact_groups = [g for g in fact_groups if g in group_filter]

            # For each group, find other related facts
            for group in fact_groups:
                escaped_group = group.replace('"', '\\"')

                # Find facts in the same group
                related_facts_result = kuzu_conn.execute(
                    f"""
                    MATCH (g:Groups)-[:Contains]->(f:Fact)
                    WHERE g.name = "{escaped_group}" AND f.content <> "{escaped_fact}"
                    RETURN f.content, f.path, f.recorded_at
                    LIMIT 5
                    """
                ).get_as_df()

                # Add these related facts to results
                for _, row in related_facts_result.iterrows():
                    related_fact = {
                        "fact": row["f.content"],
                        "source": f"graph_relation_via_{group}",
                        "relevance": "group_related",
                        "path": row["f.path"],
                        "recorded_at": row["f.recorded_at"],
                    }

                    # Avoid duplicates
                    if not any(
                        r.get("fact") == related_fact["fact"] for r in expanded_results
                    ):
                        expanded_results.append(related_fact)

        except Exception as e:
            print(f"Error expanding results via graph: {e}")

    # Return results, limiting to top_k if needed
    return expanded_results[:top_k]


def get_facts_for_rag(
    kuzu_db_path: str,
    chroma_db_path: str,
    query: str,
    group_filters: Optional[List[str]] = None,
    top_k: int = 10,
) -> str:
    """Get facts for RAG by combining vector and graph search

    Args:
        kuzu_db_path: Path to Kuzu graph database
        chroma_db_path: Path to Chroma vector database
        query: Search query
        group_filters: Optional list of groups to filter by
        top_k: Number of results to return
        embedding_model: Model to use for embeddings
        provider: Provider for embeddings

    Returns:
        Formatted context string with retrieved facts
    """
    # Initialize connections
    kuzu_conn = init_db(kuzu_db_path)
    chroma_client, chroma_collection = setup_chroma_db( 
        "knowledge_graph",
        "Facts extracted from various sources",
        chroma_db_path
    )

    # Perform hybrid search
    results = hybrid_search_with_chroma(
        kuzu_conn=kuzu_conn,
        chroma_collection=chroma_collection,
        query=query,
        group_filter=group_filters,
        top_k=top_k,
    )

    # Format results as context for RAG
    context = "Related facts:\n\n"

    # First include direct vector matches
    context += "Most relevant facts:\n"
    vector_matches = [r for r in results if r["source"] == "vector_search"]
    for i, item in enumerate(vector_matches):
        context += f"{i+1}. {item['fact']}\n"

    # Then include graph-related facts
    context += "\nRelated concepts:\n"
    graph_matches = [r for r in results if r["source"] != "vector_search"]
    for i, item in enumerate(graph_matches):
        group = item["source"].replace("graph_relation_via_", "")
        context += f"{i+1}. {item['fact']} (related via {group})\n"

    # Close connections
    kuzu_conn.close()

    return context


def answer_with_rag(
    query: str,
    kuzu_db_path: str = os.path.expanduser("~/npcsh_graph.db"),
    chroma_db_path: str = os.path.expanduser("~/npcsh_chroma.db"),
    model: str = "ollama",
    provider: str = "llama3.2",
    embedding_model: str = "text-embedding-3-small",
) -> str:
    """Answer a query using RAG with facts from the knowledge base

    Args:
        query: User query
        kuzu_db_path: Path to Kuzu graph database
        chroma_db_path: Path to Chroma vector database
        model: LLM model to use
        provider: LLM provider
        embedding_model: Model to use for embeddings

    Returns:
        Answer from the model
    """
    # Get relevant facts using hybrid search
    context = get_facts_for_rag(
        kuzu_db_path,
        chroma_db_path,
        query,
    )

    # Craft prompt with retrieved context
    prompt = f"""
    Answer this question based on the retrieved information.

    Question: {query}

    {context}

    Please provide a comprehensive answer based on the facts above. If the information
    doesn't contain a direct answer, please indicate that clearly but try to synthesize
    from the available facts.
    """

    # Get response from LLM
    response = get_llm_response(prompt, model=model, provider=provider)

    return response["response"]

