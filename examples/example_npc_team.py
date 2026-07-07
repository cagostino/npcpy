import os
import sys

from npcpy.npc_compiler import NPC, Team
import tempfile
import numpy as np 
import yaml

def setup_test_team():
    """Create a temporary team for testing"""
    temp_dir = tempfile.mkdtemp()
    team_dir = os.path.join(temp_dir, "test_team")
    os.makedirs(team_dir, exist_ok=True)
    os.makedirs(os.path.join(team_dir, "jinxes"), exist_ok=True)
    
    
    analyst_config = {
        "name": "data_analyst",
        "primary_directive": """You are a data analyst specializing in financial and business data analysis. 
        You excel at interpreting datasets, creating visualizations, and providing actionable insights from numbers.""",
        "model": "llama3.2",
        "provider": "ollama"
    }
    
    
    writer_config = {
        "name": "content_writer",
        "primary_directive": """You are a technical content writer who excels at creating clear, engaging documentation, 
        reports, and explanations of complex technical concepts for various audiences.""",
        "model": "llama3.2", 
        "provider": "ollama"
    }
    
    
    researcher_config = {
        "name": "researcher",
        "primary_directive": """You are a research specialist who excels at gathering information, 
        fact-checking, analyzing trends, and providing comprehensive research summaries on various topics.""",
        "model": "llama3.2",
        "provider": "ollama"
    }
    
    
    coordinator_config = {
        "name": "coordinator", 
        "primary_directive": """You are a project coordinator who manages team workflows. You analyze requests 
        and determine which team member is best suited for each task, then pass work accordingly.""",
        "model": "llama3.2",
        "provider": "ollama"
    }
    
    
    configs = [analyst_config, writer_config, researcher_config, coordinator_config]
    for config in configs:
        npc_path = os.path.join(team_dir, f"{config['name']}.npc")
        with open(npc_path, 'w') as f:
            yaml.dump(config, f)
    
    
    team_context = {
        "forenpc": "coordinator"
    }
    
    ctx_path = os.path.join(team_dir, "team.ctx")
    with open(ctx_path, 'w') as f:
        yaml.dump(team_context, f)
    
    
    data_jinx = {
        "jinx_name": "analyze_csv_data",
        "description": "Load and analyze CSV data, create basic statistics and insights",
        "inputs": [
            {"file_path": "path/to/csv/file"},
            {"analysis_type": "basic"}
        ],
        "steps": [
            {
                "name": "load_data",
                "engine": "python", 
                "code": """
import pandas as pd
import numpy as np


try:
    df = pd.read_csv(context['file_path'])
    context['dataframe'] = df
    context['rows'] = len(df)
    context['columns'] = list(df.columns)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    output = f"Successfully loaded data with {len(df)} rows and {len(df.columns)} columns"
except Exception as e:
    output = f"Error loading CSV: {str(e)}"
"""
            },
            {
                "name": "generate_summary",
                "engine": "natural",
                "code": """
Based on the loaded data with {{rows}} rows and columns {{columns}}, 
provide a comprehensive analysis including:
1. Data overview and structure
2. Key statistics for numerical columns  
3. Data quality observations
4. Initial insights and patterns
5. Recommendations for further analysis

Focus on actionable insights that would be valuable for business decision making.
"""
            }
        ]
    }
    
    jinx_path = os.path.join(team_dir, "jinxes", "analyze_csv_data.jinx")
    with open(jinx_path, 'w') as f:
        yaml.dump(data_jinx, f)
    
    return team_dir

def test_financial_analysis_request():
    """Test case: Financial data analysis request"""
    print("=== Test Case 1: Financial Analysis Request ===")
    
    team_dir = setup_test_team()
    team = Team(team_path=team_dir)
    
    request = """
    I need help analyzing our Q3 sales performance. We have a CSV file with transaction data 
    that includes date, product_category, sales_amount, and region columns. I need:
    1. Overall sales trends and patterns
    2. Performance by product category and region  
    3. A executive summary report suitable for board presentation
    
    The data is at /tmp/q3_sales_data.csv (you can simulate this analysis)
    """
    
    
    sample_data_path = "/tmp/q3_sales_data.csv"
    import pandas as pd
    sample_df = pd.DataFrame({
        'date': pd.date_range('2024-07-01', '2024-09-30', freq='D'),
        'product_category': ['Electronics', 'Clothing', 'Home', 'Books'] * 23,
        'sales_amount': np.random.uniform(100, 5000, 92),
        'region': ['North', 'South', 'East', 'West'] * 23
    })
    os.makedirs('/tmp', exist_ok=True)
    sample_df.to_csv(sample_data_path, index=False)
    
    result = team.orchestrate(request)
    print("Result:", result)
    print("\n" + "="*50 + "\n")

def test_competitive_research_request():
    """Test case: Competitive research and content creation"""
    print("=== Test Case 2: Competitive Research & Content Creation ===")
    
    team_dir = setup_test_team()  
    team = Team(team_path=team_dir)
    
    request = """
    We're launching a new AI-powered project management tool. I need:
    1. Research on our top 3 competitors (Asana, Monday.com, Notion)
    2. Analysis of their key features, pricing, and market positioning
    3. A competitive analysis document highlighting our differentiation opportunities
    4. Draft marketing copy for our landing page that emphasizes our AI advantages
    
    Focus on how AI integration sets us apart from traditional PM tools.
    """
    
    result = team.orchestrate(request)
    print("Result:", result)
    print("\n" + "="*50 + "\n")

def test_technical_documentation_request():
    """Test case: Technical documentation creation"""
    print("=== Test Case 3: Technical Documentation Request ===")
    
    team_dir = setup_test_team()
    team = Team(team_path=team_dir)
    
    request = """
    I need comprehensive documentation for our new API endpoints. The research team should 
    gather information about REST API best practices and current documentation standards.
    Then the content writer should create:
    
    1. API reference documentation template
    2. Developer onboarding guide  
    3. Code examples in multiple languages
    4. Error handling documentation
    
    Make it developer-friendly but also accessible to less technical stakeholders.
    """
    
    result = team.orchestrate(request)
    print("Result:", result)
    print("\n" + "="*50 + "\n")

def test_direct_npc_interaction():
    """Test case: Direct interaction with specific NPC"""
    print("=== Test Case 4: Direct NPC Interaction ===")
    
    team_dir = setup_test_team()
    team = Team(team_path=team_dir)
    
    
    analyst = team.get_npc("data_analyst")
    
    request = """
    I have customer churn data showing that 23% of our premium subscribers 
    cancelled in the last quarter. The main reasons cited were:
    - Pricing (45% of churned users)  
    - Lack of advanced features (30%)
    - Poor customer support (25%)
    
    What analysis framework should I use to dive deeper into this problem?
    """
    
    result = analyst.check_llm_command(request, team=team)
    print("Direct NPC Result:", result)
    print("\n" + "="*50 + "\n")

def test_agent_passing():
    """Test case: Explicit agent passing"""
    print("=== Test Case 5: Agent Passing Test ===")
    
    team_dir = setup_test_team()
    team = Team(team_path=team_dir)
    
    
    coordinator = team.get_npc("coordinator") 
    
    request = """
    I need someone to research the latest trends in remote work productivity tools,
    then have our content writer create a blog post about it. This is a multi-step 
    project that requires coordination between team members.
    """
    
    result = coordinator.check_llm_command(request, team=team)
    print("Agent Passing Result:", result)
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    print("Testing NPC Orchestration and Agent Passing...\n")
    
    
    test_financial_analysis_request()
    test_competitive_research_request() 
    test_technical_documentation_request()
    test_direct_npc_interaction()
    test_agent_passing()
    
    print("All tests completed!")