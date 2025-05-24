# NPCPY


## Python Examples
Integrate `npcpy` into your Python projects for additional flexibility. Below are a few examples of how to use the library programmatically.

### Example 1: using npcpy's get_llm_response

```python
from npcpy.llm_funcs import get_llm_response

# ollama's llama3.2
response = get_llm_response("What is the capital of France? Respond with a json object containing 'capital' as the key and the capital as the value.",
                            model='llama3.2',
                            provider='ollama',
                            format='json')
print(response)
# assistant's response is contained in the 'response' key for easier access
assistant_response = response['response']
print(assistant_response)
# access messages too
messages = response['messages']
print(messages)


#openai's gpt-4o-mini
from npcpy.llm_funcs import get_llm_response

response = get_llm_response("What is the capital of France? Respond with a json object containing 'capital' as the key and the capital as the value.",
                            model='gpt-4o-mini',
                            provider='openai',
                            format='json')
print(response)
# anthropic's claude haikue 3.5 latest

response = get_llm_response("What is the capital of France? Respond with a json object containing 'capital' as the key and the capital as the value.",
                            model='claude-3-5-haiku-latest',
                            provider='anthropic',
                            format='json')



# alternatively, if you have NPCSH_CHAT_MODEL / NPCSH_CHAT_PROVIDER set in your ~/.npcshrc, it will use those values
response = get_llm_response("What is the capital of France? Respond with a json object containing 'capital' as the key and the capital as the value.",
                            format='json')


# with stream
# alternatively, if you have NPCSH_CHAT_MODEL / NPCSH_CHAT_PROVIDER set in your ~/.npcshrc, it will use those values
response = get_llm_response("whats going on tonight?",
                            model='gpt-4o-mini',
                            provider='openai',
                            stream=True)

for chunk in response['response']:
    print(chunk)
```

### Example 2: Building a flow with check_llm_command

```python
#first let's demonstrate the capabilities of npcpy's check_llm_command
from npcpy.llm_funcs import check_llm_command

command = 'can you write a description of the idea of semantic degeneracy?'

response = check_llm_command(command,
                             model='gpt-4o-mini',
                             provider='openai')



# now to make the most of check_llm_command, let's add an NPC with a generic code execution jinx


from npcpy.npc_compiler import NPC, Jinx
from npcpy.llm_funcs import check_llm_command

code_execution_jinx = Jinx(
    {
        "jinx_name": "execute_python",
        "description": """Executes a code block in python.
                Final output from script MUST be stored in a variable called `output`.
                          """,
        "inputs": ["script"],
        "steps": [
            {
                "engine": " python",
                "code": """{{ script }}""",
            }
        ],
    }
)


command = """can you write a description of the idea of semantic degeneracy and save it to a file?
             After, can you take that and make various versions of it from the points of
             views of different sub-disciplines of natural lanaguage processing?
             Finally produce a synthesis of the resultant various versions and save it."
            """
npc = NPC(
    name="NLP_Master",
    primary_directive="Provide astute anlayses on topics related to NLP. Carry out relevant tasks for users to aid them in their NLP-based analyses",
    model="gpt-4o-mini",
    provider="openai",
    jinxs=[code_execution_jinx],
)
response = check_llm_command(
    command, model="gpt-4o-mini", provider="openai", npc=npc, stream=False
)

```



### Example 3: Creating and Using an NPC
This example shows how to create and initialize an NPC and use it to answer a question.
```python
import sqlite3
from npcpy.npc_compiler import NPC

# Set up database connection
db_path = '~/npcsh_history.db'
conn = sqlite3.connect(os.path.expanduser(db_path))

# Load NPC from a file
npc = NPC(
          name='Simon Bolivar',
          primary_directive='Liberate South America from the Spanish Royalists.',
          model='gpt-4o-mini',
          provider='openai',
          db_conn=conn,
          )

response = npc.get_llm_response("What is the most important territory to retain in the Andes mountains?")
print(response['response'])
```
```bash
'The most important territory to retain in the Andes mountains for the cause of liberation in South America would be the region of Quito in present-day Ecuador. This area is strategically significant due to its location and access to key trade routes. It also acts as a vital link between the northern and southern parts of the continent, influencing both military movements and the morale of the independence struggle. Retaining control over Quito would bolster efforts to unite various factions in the fight against Spanish colonial rule across the Andean states.'
```


or to stream a response, 
```python
from npcpy.npc_sysenv import print_and_process_stream
response = npc.get_llm_response("What is the most important territory to retain in the Andes mountains?",
                                 stream=True)
accumulated_response = print_and_process_stream(response['response'])
```

### Example 4: Orchestrating a team



```python
import pandas as pd
import numpy as np
import os
from npcpy.npc_compiler import NPC, Team, Jinx


# Create test data and save to CSV
def create_test_data(filepath="sales_data.csv"):
    sales_data = pd.DataFrame(
        {
            "date": pd.date_range(start="2024-01-01", periods=90),
            "revenue": np.random.normal(10000, 2000, 90),
            "customer_count": np.random.poisson(100, 90),
            "avg_ticket": np.random.normal(100, 20, 90),
            "region": np.random.choice(["North", "South", "East", "West"], 90),
            "channel": np.random.choice(["Online", "Store", "Mobile"], 90),
        }
    )

    # Add patterns to make data more realistic
    sales_data["revenue"] *= 1 + 0.3 * np.sin(
        np.pi * np.arange(90) / 30
    )  # Seasonal pattern
    sales_data.loc[sales_data["channel"] == "Mobile", "revenue"] *= 1.1  # Mobile growth
    sales_data.loc[
        sales_data["channel"] == "Online", "customer_count"
    ] *= 1.2  # Online customer growth

    sales_data.to_csv(filepath, index=False)
    return filepath, sales_data


code_execution_jinx = Jinx(
    {
        "jinx_name": "execute_code",
        "description": """Executes a Python code block with access to pandas,
                          numpy, and matplotlib.
                          Results should be stored in the 'results' dict to be returned.
                          The only input should be a single code block with \n characters included.
                          The code block must use only the  libraries or methods contained withen the
                            pandas, numpy, and matplotlib libraries or using builtin methods.
                          do not include any json formatting or markdown formatting.

                          When generating your script, the final output must be encoded in a variable
                          named "output". e.g.

                          output  = some_analysis_function(inputs, derived_data_from_inputs)
                            Adapt accordingly based on the scope of the analysis

                          """,
        "inputs": ["script"],
        "steps": [
            {
                "engine": "python",
                "code": """{{script}}""",
            }
        ],
    }
)

# Analytics team definition
analytics_team = [
    {
        "name": "analyst",
        "primary_directive": "You analyze sales performance data, focusing on revenue trends, customer behavior metrics, and market indicators. Your expertise is in extracting actionable insights from complex datasets.",
        "model": "gpt-4o-mini",
        "provider": "openai",
        "jinxs": [code_execution_jinx],  # Only the code execution jinx
    },
    {
        "name": "researcher",
        "primary_directive": "You specialize in causal analysis and experimental design. Given data insights, you determine what factors drive observed patterns and design tests to validate hypotheses.",
        "model": "gpt-4o-mini",
        "provider": "openai",
        "jinxs": [code_execution_jinx],  # Only the code execution jinx
    },
    {
        "name": "engineer",
        "primary_directive": "You implement data pipelines and optimize data processing. When given analysis requirements, you create efficient workflows to automate insights generation.",
        "model": "gpt-4o-mini",
        "provider": "openai",
        "jinxs": [code_execution_jinx],  # Only the code execution jinx
    },
]


def create_analytics_team():
    # Initialize NPCs with just the code execution jinx
    npcs = []
    for npc_data in analytics_team:
        npc = NPC(
            name=npc_data["name"],
            primary_directive=npc_data["primary_directive"],
            model=npc_data["model"],
            provider=npc_data["provider"],
            jinxs=[code_execution_jinx],  # Only code execution jinx
        )
        npcs.append(npc)

    # Create coordinator with just code execution jinx
    coordinator = NPC(
        name="coordinator",
        primary_directive="You coordinate the analytics team, ensuring each specialist contributes their expertise effectively. You synthesize insights and manage the workflow.",
        model="gpt-4o-mini",
        provider="openai",
        jinxs=[code_execution_jinx],  # Only code execution jinx
    )

    # Create team
    team = Team(npcs=npcs, foreman=coordinator)
    return team


def main():
    # Create and save test data
    data_path, sales_data = create_test_data()

    # Initialize team
    team = create_analytics_team()

    # Run analysis - updated prompt to reflect code execution approach
    results = team.orchestrate(
        f"""
    Analyze the sales data at {data_path} to:
    1. Identify key performance drivers
    2. Determine if mobile channel growth is significant
    3. Recommend tests to validate growth hypotheses

    Here is a header for the data file at {data_path}:
    {sales_data.head()}

    When working with dates, ensure that date columns are converted from raw strings. e.g. use the pd.to_datetime function.


    When working with potentially messy data, handle null values by using nan versions of numpy functions or
    by filtering them with a mask .

    Use Python code execution to perform the analysis - load the data and perform statistical analysis directly.
    """
    )

    print(results)

    # Cleanup
    os.remove(data_path)


if __name__ == "__main__":
    main()

```