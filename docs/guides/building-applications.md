# Building Applications with npcpy

This guide covers the database-powered features of npcpy that enable persistent, production-ready agent systems. It walks through the data layer, conversation tracking, memory, knowledge graphs, NPCSQL, desktop automation, triggers, scheduled jobs, and patterns for building long-running applications.

## Database Setup

npcpy uses SQLAlchemy for persistence. The default backend is SQLite, but PostgreSQL is also supported.

### Connecting to a Database

```python
from sqlalchemy import create_engine
from npcpy.db import ensure_engine

# SQLite (caller-provided path — npcpy does not hardcode defaults)
db_path = "/home/user/.incognide/history.db"
engine = ensure_engine(db_path)

# PostgreSQL
engine = create_engine("postgresql://user:password@localhost:5432/mydb")
```

The `ensure_engine()` helper accepts either a file path string (for SQLite) or a SQLAlchemy `Engine` instance.

### Passing db_conn to NPC and Team

Both `NPC` and `Team` accept a `db_conn` parameter. When you create a `Team` with `db_conn`, the engine is automatically passed to every NPC in the team:

```python
from sqlalchemy import create_engine
from npcpy.npc_compiler import NPC, Team

engine = create_engine("sqlite:///myapp.db")

# Single NPC with database
agent = NPC(
    name="assistant",
    primary_directive="You help users with tasks.",
    model="llama3.2",
    provider="ollama",
    db_conn=engine,
)

# Team with shared database
team = Team(team_path="./npc_team", db_conn=engine)
# All NPCs in the team now share this database connection
```

The flow is: `Team.db_conn` → each `NPC.db_conn` → direct SQLAlchemy operations via `npcpy.db`.

### Platform-Aware Paths

npcpy provides platform-aware directory helpers in `npcpy.npc_sysenv`:

```python
from npcpy.npc_sysenv import get_data_dir, get_config_dir, get_cache_dir

get_data_dir()    # Linux: ~/.local/share/npcsh, macOS: ~/Library/Application Support/npcsh, Windows: %LOCALAPPDATA%/npcsh
get_config_dir()  # Linux: ~/.config/npcsh, macOS: ~/Library/Application Support/npcsh, Windows: %APPDATA%/npcsh
get_cache_dir()   # Linux: ~/.cache/npcsh, macOS: ~/Library/Caches/npcsh, Windows: %LOCALAPPDATA%/npcsh/cache
```

These return platform-appropriate directories for the **npcsh** CLI tool. Applications built on npcpy (such as incognide) should supply their own paths.

---

## Conversation Tracking

Every LLM interaction can be persisted to the `conversation_history` table.

### Table Schema

```sql
CREATE TABLE conversation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id VARCHAR(50) UNIQUE NOT NULL,
    timestamp VARCHAR(50),
    role VARCHAR(20),
    content TEXT,
    conversation_id VARCHAR(100),
    directory_path TEXT,
    model VARCHAR(100),
    provider VARCHAR(100),
    npc VARCHAR(100),
    team VARCHAR(100),
    reasoning_content TEXT,
    tool_calls TEXT,
    tool_results TEXT,
    parent_message_id VARCHAR(50),
    device_id VARCHAR(255),
    device_name VARCHAR(255),
    params TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost VARCHAR(50)
);
```

Key columns:

- `message_id` — unique identifier for each message
- `conversation_id` — groups related messages into a conversation
- `npc`, `team` — which agent and team handled the message
- `reasoning_content` — chain-of-thought / thinking tokens
- `tool_calls`, `tool_results` — JSON arrays of tool invocations and their results
- `input_tokens`, `output_tokens`, `cost` — usage tracking

### Saving Messages

```python
from npcpy.db import generate_message_id, get_db_connection
import json

message_id = generate_message_id()

with get_db_connection(engine) as conn:
    conn.execute(
        """
        INSERT INTO conversation_history
        (message_id, timestamp, role, content, conversation_id, directory_path,
         model, provider, npc, team, input_tokens, output_tokens, cost)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (message_id, "2025-02-02T12:00:00+00:00", "user",
         "What files are in the current directory?", "myapp_20250202",
         "/home/user/projects", "llama3.2", "ollama", "assistant", "my_team",
         25, 150, "0.0"),
    )
    conn.commit()
```

Optional columns include `reasoning_content`, `tool_calls`, `tool_results`, `parent_message_id`, `device_id`, `device_name`, `params`, and `execution_mode`.

### Retrieving Conversations

```python
from npcpy.db import get_db_connection

with get_db_connection(engine) as conn:
    rows = conn.execute(
        "SELECT * FROM conversation_history WHERE conversation_id = ? ORDER BY timestamp",
        ("myapp_20250202",),
    ).fetchall()
    messages = [dict(row) for row in rows]
    for msg in messages:
        print(f"{msg['role']}: {msg['content'][:80]}")
```

### Starting New Conversations

```python
import datetime

conv_id = f"myapp_{datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%d%H%M%S')}"
```

---

## Execution Logging

npcpy automatically tracks jinx and NPC executions via dedicated tables. These are populated by SQLite triggers on `conversation_history` inserts.

### jinx_executions

```sql
CREATE TABLE jinx_executions (
    message_id VARCHAR(50) PRIMARY KEY,
    jinx_name VARCHAR(100),
    input TEXT,
    timestamp VARCHAR(50),
    npc VARCHAR(100),
    team VARCHAR(100),
    conversation_id VARCHAR(100),
    output TEXT,
    status VARCHAR(50),
    error_message TEXT,
    duration_ms INTEGER
);
```

### npc_executions

```sql
CREATE TABLE npc_executions (
    message_id VARCHAR(50) PRIMARY KEY,
    input TEXT,
    timestamp VARCHAR(50),
    npc VARCHAR(100),
    team VARCHAR(100),
    conversation_id VARCHAR(100),
    model VARCHAR(100),
    provider VARCHAR(100)
);
```

These tables let you query execution patterns, track error rates, and measure latency across your agent system.

---

## Memory and Knowledge Graphs

npcpy provides a human-in-the-loop memory system scoped by `(npc, team, directory_path)`, and a knowledge graph for long-term factual recall.

### Memory Lifecycle

Memories flow through a review pipeline:

1. **pending_approval** — LLM proposes a memory from conversation
2. **human-approved** — user confirms the memory
3. **human-rejected** — user rejects the memory
4. **human-edited** — user provides a corrected version

The `memory_lifecycle` table tracks this:

```sql
CREATE TABLE memory_lifecycle (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id VARCHAR(50) NOT NULL,
    conversation_id VARCHAR(100) NOT NULL,
    npc VARCHAR(100) NOT NULL,
    team VARCHAR(100) NOT NULL,
    directory_path TEXT NOT NULL,
    timestamp VARCHAR(50) NOT NULL,
    initial_memory TEXT NOT NULL,
    final_memory TEXT,
    status VARCHAR(50) NOT NULL,
    model VARCHAR(100),
    provider VARCHAR(100),
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

### Memory Approval UI

For interactive review of pending memories:

```python
from npcpy.memory.memory_processor import memory_approval_ui, memory_batch_review_ui
from npcpy.memory.knowledge_store import get_store_for_path

store = get_store_for_path("/path/to/project")

# Review individual memories
pending = store.get_pending_memories()
decisions = memory_approval_ui(pending, show_context=True)
# Interactive: (a)pprove, (r)eject, (e)dit, (s)kip, (A)pprove all, (R)eject all
```

### Memory in System Messages

Approved memories from `.knowledge.yaml` can be loaded for few-shot context:

```python
store = get_store_for_path("/path/to/project")
approved = store.get_memories(status="human-approved", limit=10)
rejected = store.get_memories(status="human-rejected", limit=10)
# Returns lists of memory dicts
```

When an NPC is created with `memory=True`, the system message is enriched with recent facts and key concepts from the knowledge graph.

### Knowledge Graph Tables

Knowledge graphs are stored across four tables, all scoped by `(team_name, npc_name, directory_path)`:

```sql
CREATE TABLE kg_facts (
    statement TEXT NOT NULL,
    team_name VARCHAR(255) NOT NULL,
    npc_name VARCHAR(255) NOT NULL,
    directory_path TEXT NOT NULL,
    source_text TEXT,
    type VARCHAR(100),
    generation INTEGER,
    origin VARCHAR(100)  -- "organic", "dream", "deepen", "manual_add"
);

CREATE TABLE kg_concepts (
    name TEXT NOT NULL,
    team_name VARCHAR(255) NOT NULL,
    npc_name VARCHAR(255) NOT NULL,
    directory_path TEXT NOT NULL,
    generation INTEGER,
    origin VARCHAR(100)
);

CREATE TABLE kg_links (
    source TEXT NOT NULL,
    target TEXT NOT NULL,
    team_name VARCHAR(255) NOT NULL,
    npc_name VARCHAR(255) NOT NULL,
    directory_path TEXT NOT NULL,
    type VARCHAR(100) NOT NULL  -- "fact_to_concept", "concept_to_concept", "fact_to_fact"
);

CREATE TABLE kg_metadata (
    key VARCHAR(255) NOT NULL,
    team_name VARCHAR(255) NOT NULL,
    npc_name VARCHAR(255) NOT NULL,
    directory_path TEXT NOT NULL,
    value TEXT
);
```

### Loading and Saving Knowledge Graphs

```python
from npcpy.memory.knowledge_graph import load_kg_from_db, save_kg_to_db

kg_data = load_kg_from_db(engine, team_name="my_team", npc_name="assistant", directory_path="/project")
# Returns dict with: generation, facts, concepts, concept_links, fact_to_concept_links, fact_to_fact_links

save_kg_to_db(engine, kg_data, team_name="my_team", npc_name="assistant", directory_path="/project")
```

---

## NPC System Architecture

### .npc File Format

NPCs can be defined in YAML files with a `.npc` extension:

```yaml
name: data_analyst
primary_directive: >
  You are a meticulous data analyst who provides
  insights from structured and unstructured data.
model: llama3.2
provider: ollama
jinxes:
  - "*"          # inherit all team jinxes
tools:
  - statistical_analysis
  - data_visualization
```

Key fields:

| Field | Description |
|-------|-------------|
| `name` | NPC identifier |
| `primary_directive` | System prompt defining persona and behavior |
| `model` | LLM model (inherits from team if unset) |
| `provider` | LLM provider (inherits from team if unset) |
| `jinxes` | `"*"` for all team jinxes, or a list of specific names/paths |
| `tools` | List of Python callable names to expose |
| `api_url` | Custom API endpoint |
| `api_key` | API credentials |

### team.ctx Format

Team configuration is defined in a `.ctx` file:

```yaml
name: Data Science Team
context: >
  We are a data-driven team focused on extracting
  actionable insights from customer data.
model: llama3.2
provider: ollama
forenpc: data_analyst

databases:
  - customer_insights
  - sales_performance

file_patterns:
  - pattern: "*.md"
    recursive: true
    base_path: "./docs"

mcp_servers:
  - name: sqlite
    command: python -m mcp.server.sqlite
    args: []
```

Reserved keys (`name`, `context`, `forenpc`, `model`, `provider`, `api_url`, `databases`, `mcp_servers`, `file_patterns`, `env`) are handled specially. All other keys are merged directly into `shared_context`.

### npc_team/ Directory Structure

```
npc_team/
├── team.ctx                  # Team configuration
├── analyst.npc               # NPC definitions
├── writer.npc
├── jinxes/                    # Team-level workflows
│   ├── summarize.jinx
│   └── analyze.jinx
├── assembly_lines/           # Batch processing pipelines
├── sql_models/               # NPCSQL model files
├── jobs/                     # Scheduled task definitions
└── triggers/                 # Event-triggered workflows
```

When `Team(team_path="./npc_team")` is called, all `.npc` files are loaded as NPCs, `team.ctx` provides defaults, and jinxes from `jinxes/` are made available to all team members (or selectively via the `jinxes` field in each `.npc` file).

### db_conn Flow

```
Team(db_conn=engine)
  └─► NPC(db_conn=engine, team=self)      # for each .npc file
        └─► direct SQLAlchemy via npcpy.db  # conversation persistence
        └─► load_kg_from_db(engine, ...)    # if memory=True
```

### System Message Generation

The system message sent to the LLM is assembled from multiple sources:

1. NPC name and primary directive
2. Current working directory and timestamp
3. Memory context (recent facts and key concepts from the knowledge graph)
4. Database information (available tables)
5. Team context (team description, member list with directives)
6. Available tools and jinxes with descriptions

See `npcpy.npc_sysenv.get_system_message()` for the full assembly logic.

### NPC Version History

npcpy tracks versions of NPC configurations for rollback:

```python
from npcpy.memory.npc_version import save_npc_version, rollback_npc_to_version, get_npc_versions

# Save current NPC state
version = save_npc_version(engine, npc_name="analyst", team_path="./npc_team",
                           content=yaml_content, commit_message="Updated directive")

# List versions
versions = get_npc_versions(engine, npc_name="analyst", team_path="./npc_team")

# Rollback (creates a new version from the old content and writes to file)
content = rollback_npc_to_version(engine, npc_name="analyst", team_path="./npc_team", version=2)
```

---

## NPCSQL — AI-Powered SQL

NPCSQL lets you write SQL model files with embedded AI function calls, Jinja templating, and DAG-based execution. Think dbt meets LLM-powered transformations.

Source: `npcpy/sql/npcsql.py`, `npcpy/sql/database_ai_functions.py`

### SQL Model Files

SQL models are `.sql` files in your `npc_team/sql_models/` directory. They support Jinja templating:

```sql
-- models/base_stats.sql
{{ config(materialized='table') }}

SELECT
    customer_id,
    COUNT(*) as order_count,
    SUM(amount) as total_spent
FROM orders
GROUP BY customer_id
```

```sql
-- models/enriched_customers.sql
{{ config(materialized='table') }}

SELECT
    b.customer_id,
    b.order_count,
    b.total_spent,
    nql.generate_text(b.customer_id, 'analyst') as customer_segment
FROM {{ ref('base_stats') }} b
```

### Jinja Functions

| Function | Usage | Description |
|----------|-------|-------------|
| `{{ ref('model_name') }}` | Reference another model | Creates dependency in the DAG |
| `{{ config(materialized='table') }}` | Model configuration | Controls materialization strategy |
| `{{ npc('name').model }}` | Access NPC properties | Name, model, provider, directive |
| `{{ jinx('name').description }}` | Access jinx properties | Name, description, inputs |
| `{{ env('VAR_NAME') }}` | Environment variable | Read environment values |

### NQL Function Syntax

NQL functions embed AI calls directly in SQL:

```sql
nql.generate_text(column, 'agent_name') as alias
```

The `nql.` prefix indicates an AI function. The column value is passed to the specified NPC agent, and the result replaces the function call in the output.

### DAG-Based Execution

The `ModelCompiler` orchestrates execution:

```python
from npcpy.sql.npcsql import ModelCompiler

compiler = ModelCompiler(
    models_dir="./npc_team/sql_models",
    target_engine=engine,
    npc_directory="./npc_team",
)

# Discover → parse → topological sort → execute → materialize
compiler.run_all_models()
```

The pipeline:

1. **Discover** — find all `.sql` files in `models_dir`
2. **Parse** — extract `{{ ref() }}` dependencies and `nql.*` calls
3. **Topological sort** — order models by dependency
4. **Execute** — run each model's SQL, applying AI functions
5. **Materialize** — write results to the target database

### Database-Native AI

For supported databases, NQL functions are translated to native SQL:

- **Snowflake Cortex** — `nql.generate_text()` → `SNOWFLAKE.CORTEX.COMPLETE()`
- **BigQuery ML** — `nql.generate_text()` → `ML.GENERATE_TEXT()`
- **Databricks** — `nql.generate_text()` → `serving.predict()`

For SQLite and PostgreSQL, npcpy falls back to Python-based execution: the SQL runs without the NQL calls, then AI functions are applied row-by-row on the resulting DataFrame.

---

## Desktop Automation

npcpy includes cross-platform desktop and browser automation.

Source: `npcpy/work/desktop.py`, `npcpy/data/image.py`, `npcpy/work/browser.py`

### Screenshot Capture

```python
from npcpy.data.image import capture_screenshot

# Full-screen capture
result = capture_screenshot(full=True)
print(result['file_path'])  # ~/.npcsh/screenshots/screenshot_20250202_143022.png

# Interactive selection
result = capture_screenshot(full=False)
```

Platform implementations:

| Platform | Full Screen | Interactive |
|----------|-------------|-------------|
| macOS | `screencapture` | `screencapture -i` |
| Linux | `grim` (Wayland), `scrot` (X11), `import` (ImageMagick) | `gnome-screenshot -a`, `scrot -s` |
| Windows | Win32 API (BitBlt) | Windows Snipping Tool (Win+Shift+S) |

### Desktop Actions

```python
from npcpy.work.desktop import perform_action

# Click at screen coordinates (0-100 percentage)
perform_action({"type": "click", "x": 50, "y": 50})

# Type text
perform_action({"type": "type", "text": "Hello, world!"})

# Press a key
perform_action({"type": "key", "keys": ["enter"]})

# Keyboard shortcut
perform_action({"type": "hotkey", "keys": ["ctrl", "s"]})

# Scroll
perform_action({"type": "scroll", "direction": "down", "amount": 3})

# Drag
perform_action({"type": "drag", "start_x": 20, "start_y": 20, "end_x": 80, "end_y": 80})

# Run a shell command
perform_action({"type": "shell", "command": "ls -la"})

# Wait
perform_action({"type": "wait", "duration": 2.0})
```

All actions return `{"status": "success|error", "output": "...", "message": "..."}`.

### Browser Automation

Browser automation uses Selenium WebDriver with session management:

```python
from npcpy.work.browser import get_current_driver, set_driver, close_current

# Register a driver
set_driver("session1", my_webdriver)

# Get the active driver
driver = get_current_driver()

# Close the current session
close_current()
```

Multiple concurrent browser sessions are supported via the global session registry.

---

## Triggers and Scheduled Jobs

npcpy can create file-system triggers and scheduled background jobs using LLM-generated scripts. The scripts are platform-specific and registered with the OS scheduler.

Source: `npcpy/work/trigger.py`, `npcpy/work/plan.py`

### Triggers

Triggers monitor the filesystem for events and run scripts in response:

```python
from npcpy.work.trigger import execute_trigger_command

result = execute_trigger_command(
    "Move PDFs from Downloads to Documents/PDFs",
    npc=my_npc,
    model="llama3.2",
    provider="ollama",
)
```

Platform implementations:

| Platform | File Watcher | Service Manager | Script Location |
|----------|-------------|-----------------|-----------------|
| Linux | `inotifywait` | systemd user service | `~/.npcsh/triggers/trigger_{name}.sh` |
| macOS | `fswatch` | launchd plist | `~/.npcsh/triggers/trigger_{name}.sh` |
| Windows | PowerShell `FileSystemWatcher` | Task Scheduler | `~/.npcsh/triggers/trigger_{name}.ps1` |

### Scheduled Jobs

Jobs run on a schedule using the platform's native scheduler:

```python
from npcpy.work.plan import execute_plan_command

result = execute_plan_command(
    "Record CPU usage every 10 minutes",
    npc=my_npc,
    model="llama3.2",
    provider="ollama",
)
```

Platform implementations:

| Platform | Scheduler | Schedule Format | Script Location |
|----------|-----------|-----------------|-----------------|
| Linux | cron | Crontab expression (`*/10 * * * *`) | `~/.npcsh/jobs/job_{name}.sh` |
| macOS | launchd | `StartInterval` (seconds) | `~/.npcsh/jobs/job_{name}.sh` |
| Windows | schtasks | Task Scheduler params (`/sc minute /mo 10`) | `~/.npcsh/jobs/job_{name}.ps1` |

### Directory Structure

```
~/.npcsh/
├── jobs/           # Scheduled job scripts
├── triggers/       # File-system trigger scripts
└── logs/           # Execution logs for both
```

Both triggers and jobs use `get_llm_response` to generate platform-specific scripts from natural language descriptions. The LLM returns a JSON object with the script content, name, description, and schedule parameters.

---

## Building Persistent Agent Systems

This section covers patterns for building long-running applications with database-backed agents.

### Creating a Database-Backed Team

```python
from sqlalchemy import create_engine
from npcpy.npc_compiler import Team

engine = create_engine("sqlite:///myapp.db")
team = Team(team_path="./npc_team", db_conn=engine)

# All conversations, executions, and memories are persisted
result = team.orchestrate("Analyze the Q4 sales data and write a summary.")
```

### Conversation Persistence Patterns

Use `start_new_conversation()` to create conversation IDs that group related messages:

```python
from npcpy.memory.command_history import start_new_conversation

conv_id = start_new_conversation(prepend="support_ticket_123")

# All subsequent messages with this conv_id are grouped together
response = agent.get_llm_response(
    "Customer reports login failures on mobile.",
    conversation_id=conv_id,
)
```

### Shared Context for Team-Wide State

The `shared_context` dictionary provides session state that flows between NPCs during orchestration:

```python
# Team shared_context structure
team.shared_context = {
    "intermediate_results": {},    # Outputs from jinx steps
    "dataframes": {},              # Pandas DataFrames
    "memories": {},                # Named memory objects
    "execution_history": [],       # Record of executed commands
    "context": "",                 # Team context string
}

# NPC shared_context includes session tracking
npc.shared_context["session_input_tokens"]   # 0
npc.shared_context["session_output_tokens"]  # 0
npc.shared_context["session_cost_usd"]       # 0.0
npc.shared_context["turn_count"]             # 0
```

Custom keys from `team.ctx` (anything not a reserved field) are merged directly into `shared_context`, making them available to all NPCs and jinxes.

### Querying Execution Stats

```python
stats = ch.get_npc_conversation_stats(start_date="2025-01-01", end_date="2025-02-01")
# Returns DataFrame with columns:
#   npc, total_messages, avg_message_length, total_conversations,
#   models_used, providers_used, model_list, provider_list,
#   first_conversation, last_conversation
```

### Cost Tracking

Token counts and costs are stored per message in `conversation_history`. Use standard SQL or the stats functions to aggregate:

```python
import pandas as pd

with engine.connect() as conn:
    df = pd.read_sql("""
        SELECT npc, team, model,
               SUM(input_tokens) as total_input,
               SUM(output_tokens) as total_output,
               SUM(CAST(cost AS FLOAT)) as total_cost
        FROM conversation_history
        WHERE timestamp >= '2025-01-01'
        GROUP BY npc, team, model
    """, conn)
```

### Labels and Training Data Curation

Label messages for fine-tuning or quality review:

```python
# Add a label to a message
ch.add_label(entity_type="message", entity_id=message_id, label="training",
             metadata={"quality": "high", "task": "summarization"})

# Retrieve labeled messages for fine-tuning
training_data = ch.get_training_data_by_label(label="training")
# Returns list of message dicts with the specified label
```

---

## Next Steps

- **[Working with LLMs](llm-responses.md)** — Streaming, JSON output, message history
- **[Building Agents](agents.md)** — NPC creation, directives, tool assignment
- **[Multi-Agent Teams](teams.md)** — Team orchestration patterns
- **[Knowledge Graphs](knowledge-graphs.md)** — Building and evolving knowledge graphs
- **[Architecture](../npc_data_layer.md)** — Technical deep-dive into the data layer
