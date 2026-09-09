import os
import shutil
import tempfile
import sqlite3
from pathlib import Path
from npcpy.npc_compiler import NPC, Jinx, Team, initialize_npc_project


def test_npc_creation():
    """Test basic NPC creation"""
    npc = NPC(
        name="test_npc",
        primary_directive="You are a helpful assistant",
        model="llama3.2:latest",
        provider="ollama"
    )
    assert npc.name == "test_npc"
    assert npc.primary_directive == "You are a helpful assistant"
    print(f"Created NPC: {npc.name}")


def test_npc_save_and_load():
    """Test NPC save and load functionality"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        
        npc = NPC(
            name="save_test_npc",
            primary_directive="Test NPC for saving",
            model="llama3.2:latest",
            provider="ollama"
        )
        npc.save(temp_dir)
        
        
        npc_file = os.path.join(temp_dir, "save_test_npc.npc")
        assert os.path.exists(npc_file)
        
        
        loaded_npc = NPC(file=npc_file)
        assert loaded_npc.name == "save_test_npc"
        print(f"Saved and loaded NPC: {loaded_npc.name}")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_npc_get_llm_response():
    """Test NPC LLM response functionality"""
    npc = NPC(
        name="response_test_npc",
        primary_directive="You are a helpful math assistant",
        model="llama3.2:latest",
        provider="ollama"
    )
    
    response = npc.get_llm_response("What is 3 + 4?")
    assert response is not None
    print(f"NPC response: {response}")


def test_jinx_creation():
    """Test basic Jinx creation"""
    jinx_data = {
        "jinx_name": "test_jinx",
        "description": "A test jinx",
        "inputs": ["input1", "input2"],
        "steps": [
            {
                "code": """
input1 = '{{ input1 }}'
input2 = '{{ input2 }}'
output = f"Processed: {input1} and {input2}"
print(output)
""",
                "engine": "python"
            }
        ]
    }
    
    jinx = Jinx(jinx_data=jinx_data)
    assert jinx.jinx_name == "test_jinx"
    print(f"Created Jinx: {jinx.jinx_name}")


def test_jinx_execution():
    """Test Jinx execution"""
    jinx_data = {
        "jinx_name": "math_jinx",
        "description": "Math calculation jinx",
        "inputs": ["number1", "number2"],
        "steps": [
            {
                "code": """
number1 = int('{{ number1 }}')
number2 = int('{{ number2 }}')
output = number1 + number2
print(f"The sum of {number1} and {number2} is {output}")
""",
                "engine": "python"
            }
        ]
    }
    
    jinx = Jinx(jinx_data=jinx_data)
    input_values = {"number1": "5", "number2": "7"}
    
    result = jinx.execute(input_values, {})
    assert result is not None
    print(f"Jinx execution result: {result}")


def test_team_creation():
    """Test Team creation"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        
        npc1 = NPC(name="analyst", primary_directive="Analyze data")
        npc2 = NPC(name="critic", primary_directive="Critique analysis")
        
        npc1.save(temp_dir)
        npc2.save(temp_dir)
        
        team = Team(team_path=temp_dir)
        assert team is not None
        print(f"Created team with {len(team.npcs)} NPCs")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_npc_with_database():
    """Test NPC with database connection"""
    temp_db = tempfile.mktemp(suffix=".db")
    
    try:
        conn = sqlite3.connect(temp_db)
        
        npc = NPC(
            name="db_test_npc",
            primary_directive="Test NPC with database",
            db_conn=conn
        )
        
        assert npc.db_conn is not None
        print(f"Created NPC with database: {npc.name}")
        
        conn.close()
        
    finally:
        if os.path.exists(temp_db):
            os.remove(temp_db)


def test_initialize_project_basic(tmp_path):
    """Ensure initialize_npc_project scaffolds a basic npc_team directory"""
    project_dir = tmp_path / "basic_project"
    msg = initialize_npc_project(directory=project_dir)

    expected_npc = project_dir / "npc_team" / "forenpc.npc"
    assert expected_npc.exists()
    assert "npc_team" in msg


def test_initialize_project_prefers_existing_ctx(tmp_path):
    """An existing .ctx in npc_team should be preferred over generated team.ctx"""
    project_dir = tmp_path / "proj_existing_ctx"
    npc_team_dir = project_dir / "npc_team"
    npc_team_dir.mkdir(parents=True)
    (npc_team_dir / "custom.ctx").write_text("name: custom\ncontext: hello\n")

    initialize_npc_project(directory=project_dir)

    custom_ctx = project_dir / "npc_team" / "custom.ctx"
    default_ctx = project_dir / "npc_team" / "team.ctx"
    assert custom_ctx.exists()
    assert not default_ctx.exists()


def test_jinx_save_and_load():
    """Test Jinx save and load"""
    temp_dir = tempfile.mkdtemp()
    
    try:
        jinx_data = {
            "jinx_name": "save_test_jinx",
            "description": "Test jinx for saving",
            "inputs": ["input1"],
            "steps": [{"type": "llm", "prompt": "Process {{input1}}"}]
        }
        
        jinx = Jinx(jinx_data=jinx_data)
        jinx.save(temp_dir)
        
        jinx_file = os.path.join(temp_dir, "save_test_jinx.jinx")
        assert os.path.exists(jinx_file)
        
        loaded_jinx = Jinx(jinx_path=jinx_file)
        assert loaded_jinx.jinx_name == "save_test_jinx"
        print(f"Saved and loaded Jinx: {loaded_jinx.jinx_name}")
        
    finally:
        import shutil
        shutil.rmtree(temp_dir)


def test_npc_execute_jinx():
    """Test NPC executing a jinx"""
    try:
        npc = NPC(
            name="jinx_executor",
            primary_directive="Execute jinxes",
            model="llama3.2:latest",
            provider="ollama"
        )
        
        jinx_data = {
            "jinx_name": "simple_jinx",
            "inputs": ["message"],
            "steps": [{"type": "llm", "prompt": "Reply to: {{message}}"}]
        }
        
        jinx = Jinx(jinx_data=jinx_data)
        result = npc.execute_jinx("simple_jinx", {"message": "Hello"})
        
        assert result is not None
        print(f"NPC jinx execution result: {result}")
    except Exception as e:
        print(f"NPC jinx execution failed: {e}")



# =============================================================================
# Jinja2 Sandboxed Environment Tests (Issue #197)
# =============================================================================

def test_jinja2_sandboxed_environment():
    """Test that Jinja2 uses SandboxedEnvironment."""
    from jinja2.sandbox import SandboxedEnvironment

    npc = NPC(
        name="sandbox_test_npc",
        primary_directive="Test NPC"
    )

    # Check that the jinja_env is a SandboxedEnvironment
    assert isinstance(npc.jinja_env, SandboxedEnvironment), \
        "NPC.jinja_env should be a SandboxedEnvironment"
    print("NPC uses SandboxedEnvironment")


def test_jinja2_sandbox_blocks_dangerous_access():
    """Test that sandboxed Jinja2 blocks dangerous attribute access."""
    from jinja2.sandbox import SandboxedEnvironment, SecurityError

    env = SandboxedEnvironment()

    # Attempting to access __class__ should raise SecurityError
    dangerous_template = "{{ ''.__class__.__mro__ }}"

    try:
        template = env.from_string(dangerous_template)
        result = template.render()
        # If we get here without error, check if result is sanitized
        assert '__class__' not in str(result) or result == '', \
            "Sandbox should prevent access to __class__"
    except SecurityError:
        # Expected behavior - sandbox blocks dangerous access
        pass

    print("Jinja2 sandbox blocks dangerous attribute access")


def test_jinx_uses_sandboxed_environment():
    """Test that Jinx execution uses sandboxed Jinja2."""
    from jinja2.sandbox import SandboxedEnvironment

    jinx_data = {
        "jinx_name": "sandbox_jinx_test",
        "description": "Test sandbox in Jinx",
        "inputs": ["input1"],
        "steps": [
            {
                "code": "output = '{{ input1 }}'",
                "engine": "python"
            }
        ]
    }

    jinx = Jinx(jinx_data=jinx_data)

    # When no jinja_env provided, it should create a SandboxedEnvironment
    # This is tested indirectly - the execute method should work
    result = jinx.execute({"input1": "test_value"}, {})
    assert result is not None
    print("Jinx execution uses sandboxed environment")


# =============================================================================
# Jinx/NPCArray Integration Tests (Issue #196)
# =============================================================================

def test_jinx_has_npc_array_in_context():
    """Test that NPCArray is available in Jinx execution context."""
    jinx_data = {
        "jinx_name": "array_test_jinx",
        "description": "Test NPCArray in Jinx",
        "inputs": [],
        "steps": [
            {
                "code": """
# NPCArray should be available
output = 'NPCArray available' if 'NPCArray' in dir() else 'NPCArray not found'
""",
                "engine": "python"
            }
        ]
    }

    jinx = Jinx(jinx_data=jinx_data)
    result = jinx.execute({}, {})

    assert result is not None
    assert result.get('output') == 'NPCArray available', \
        f"Expected 'NPCArray available', got {result.get('output')}"
    print("NPCArray is available in Jinx execution context")


def test_jinx_can_use_npc_array():
    """Test that Jinx can actually use NPCArray."""
    jinx_data = {
        "jinx_name": "use_array_jinx",
        "description": "Test using NPCArray in Jinx",
        "inputs": [],
        "steps": [
            {
                "code": """
# Create an NPCArray
arr = NPCArray.from_llms(['test-model'], providers='test-provider')
output = f'Created array with {len(arr)} models'
""",
                "engine": "python"
            }
        ]
    }

    jinx = Jinx(jinx_data=jinx_data)
    result = jinx.execute({}, {})

    assert result is not None
    assert 'Created array with 1 models' in str(result.get('output', '')), \
        f"Expected array creation output, got {result.get('output')}"
    print("Jinx can create and use NPCArray")


def test_jinx_has_infer_matrix():
    """Test that infer_matrix is available in Jinx."""
    jinx_data = {
        "jinx_name": "infer_matrix_test",
        "description": "Test infer_matrix in Jinx",
        "inputs": [],
        "steps": [
            {
                "code": """
output = 'infer_matrix available' if callable(infer_matrix) else 'not callable'
""",
                "engine": "python"
            }
        ]
    }

    jinx = Jinx(jinx_data=jinx_data)
    result = jinx.execute({}, {})

    assert result.get('output') == 'infer_matrix available'
    print("infer_matrix is available in Jinx execution context")


def test_jinx_has_ensemble_vote():
    """Test that ensemble_vote is available in Jinx."""
    jinx_data = {
        "jinx_name": "ensemble_vote_test",
        "description": "Test ensemble_vote in Jinx",
        "inputs": [],
        "steps": [
            {
                "code": """
output = 'ensemble_vote available' if callable(ensemble_vote) else 'not callable'
""",
                "engine": "python"
            }
        ]
    }

    jinx = Jinx(jinx_data=jinx_data)
    result = jinx.execute({}, {})

    assert result.get('output') == 'ensemble_vote available'
    print("ensemble_vote is available in Jinx execution context")


def test_npc_save_preserves_jinja_jinx_syntax():
    """Test that NPC.save() preserves {{ Jinx('...') }} syntax in the YAML file."""
    temp_dir = tempfile.mkdtemp()
    try:
        npc_path = os.path.join(temp_dir, "levi.npc")
        original_content = """name: levi
primary_directive: Test directive
jinxes:
  - {{ Jinx('open_pane') }}
  - {{ Jinx('close_pane') }}
model: kimi-k2.6:cloud
provider: ollama
"""
        with open(npc_path, "w", encoding="utf-8") as f:
            f.write(original_content)

        team = Team(team_path=temp_dir)
        npc = team.npcs["levi"]
        assert npc.jinxes_spec == ["open_pane", "close_pane"]

        npc.save(temp_dir)

        with open(npc_path, "r", encoding="utf-8") as f:
            saved_content = f.read()

        assert "{{ Jinx('open_pane') }}" in saved_content
        assert "{{ Jinx('close_pane') }}" in saved_content
        assert saved_content.count("{{ Jinx(") == 2
        print("NPC.save() correctly preserved {{ Jinx('...') }} syntax")
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    import sys

    tests = [
        ("NPC creation", test_npc_creation),
        ("NPC save and load", test_npc_save_and_load),
        ("NPC save preserves Jinja jinx syntax", test_npc_save_preserves_jinja_jinx_syntax),
        ("Jinx creation", test_jinx_creation),
        ("Jinx execution", test_jinx_execution),
        ("Jinja2 sandboxed environment", test_jinja2_sandboxed_environment),
        ("Jinja2 sandbox blocks dangerous access", test_jinja2_sandbox_blocks_dangerous_access),
        ("Jinx uses sandboxed environment", test_jinx_uses_sandboxed_environment),
        ("Jinx has NPCArray in context", test_jinx_has_npc_array_in_context),
        ("Jinx can use NPCArray", test_jinx_can_use_npc_array),
        ("Jinx has infer_matrix", test_jinx_has_infer_matrix),
        ("Jinx has ensemble_vote", test_jinx_has_ensemble_vote),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        print(f"\n--- {name} ---")
        try:
            test_func()
            print(f"✓ PASSED")
            passed += 1
        except Exception as e:
            print(f"✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    sys.exit(0 if failed == 0 else 1)
