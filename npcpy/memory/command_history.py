import os
import sqlite3
import json
from datetime import datetime
import uuid
from typing import Optional, List, Dict, Any, Tuple, Union
import pandas as pd
import numpy as np

try:
    import sqlalchemy
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import Engine, Connection as SQLAlchemyConnection
    from sqlalchemy.exc import SQLAlchemyError
    _HAS_SQLALCHEMY = True
except ImportError:
    _HAS_SQLALCHEMY = False
    Engine = type(None) # Define dummy types if sqlalchemy not installed
    SQLAlchemyConnection = type(None)
    create_engine = None
    text = None

def flush_messages(n: int, messages: list) -> dict:
    if n <= 0:
        return {
            "messages": messages,
            "output": "Error: 'n' must be a positive integer.",
        }

    removed_count = min(n, len(messages))  # Calculate how many to remove
    del messages[-removed_count:]  # Remove the last n messages

    return {
        "messages": messages,
        "output": f"Flushed {removed_count} message(s). Context count is now {len(messages)} messages.",
    }


def get_db_connection():
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_messages_for_conversation(conversation_id):
    conn = get_db_connection()
    cursor = conn.cursor()

    query = """
        SELECT role, content, timestamp
        FROM conversation_history
        WHERE conversation_id = ?
        ORDER BY timestamp ASC
    """
    cursor.execute(query, (conversation_id,))
    messages = cursor.fetchall()
    conn.close()

    return [
        {
            "role": message["role"],
            "content": message["content"],
            "timestamp": message["timestamp"],
        }
        for message in messages
    ]



def deep_to_dict(obj):
    """
    Recursively convert objects that have a 'to_dict' method to dictionaries,
    otherwise drop them from the output.
    """
    if isinstance(obj, dict):
        return {key: deep_to_dict(val) for key, val in obj.items()}

    if isinstance(obj, list):
        return [deep_to_dict(item) for item in obj]

    if hasattr(obj, "to_dict") and callable(getattr(obj, "to_dict", None)):
        return deep_to_dict(obj.to_dict())

    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj

    return None  # Drop objects that don't have a known conversion


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        try:
            return deep_to_dict(obj)
        except TypeError:
            return super().default(obj)


def show_history(command_history, args):
    if args:
        search_results = command_history.search(args[0])
        if search_results:
            return "\n".join(
                [f"{item[0]}. [{item[1]}] {item[2]}" for item in search_results]
            )
        else:
            return f"No commands found matching '{args[0]}'"
    else:
        all_history = command_history.get_all()
        return "\n".join([f"{item[0]}. [{item[1]}] {item[2]}" for item in all_history])


def query_history_for_llm(command_history, query):
    results = command_history.search(query)
    formatted_results = [
        f"Command: {r[2]}\nOutput: {r[4]}\nLocation: {r[5]}" for r in results
    ]
    return "\n\n".join(formatted_results)


try:
    import chromadb
except ModuleNotFoundError:
    print("chromadb not installed")
except OSError as e:
    print('os error importing chromadb:', e)
except NameError as e:
    print('name error importing chromadb:', e)
    chromadb = None
import numpy as np
import os
from typing import Optional, Dict, List, Union, Tuple


def setup_chroma_db(collection, description='', db_path: str= ''):
    """Initialize Chroma vector database without a default embedding function"""
    if db_path == '':
        db_path = os.path.expanduser('~/npcsh_chroma_db')
        
    try:
        # Create or connect to Chroma client with persistent storage
        client = chromadb.PersistentClient(path=db_path)

        # Check if collection exists, create if not
        try:
            collection = client.get_collection(collection)
            print("Connected to existing facts collection")
        except ValueError:
            # Create new collection without an embedding function
            # We'll provide embeddings manually using get_embeddings
            collection = client.create_collection(
                name=collection,
                metadata={"description": description},
            )
            print("Created new facts collection")

        return client, collection
    except Exception as e:
        print(f"Error setting up Chroma DB: {e}")
        raise


class CommandHistory:
    def __init__(self, db: Union[str, sqlite3.Connection, Engine] = "~/npcsh_history.db"):

        self._is_sqlalchemy = False
        self.cursor = None
        self.db_path = None # Store the determined path if available

        if isinstance(db, str):
            self.db_path = os.path.expanduser(db)
            try:
                self.conn = sqlite3.connect(self.db_path, check_same_thread=False) # Allow multithread access if needed
                self.conn.row_factory = sqlite3.Row
                self.cursor = self.conn.cursor()
                self.cursor.execute("PRAGMA foreign_keys = ON")
                self.conn.commit()

            except sqlite3.Error as e:
                print(f"FATAL: Error connecting to sqlite3 DB at {self.db_path}: {e}")
                raise

        elif isinstance(db, sqlite3.Connection):
            self.conn = db
            if not hasattr(self.conn, 'row_factory') or self.conn.row_factory is None:
                 # Set row_factory if not already set on provided connection
                 try: self.conn.row_factory = sqlite3.Row
                 except Exception as e: print(f"Warning: Could not set row_factory on provided sqlite3 connection: {e}")

            self.cursor = self.conn.cursor()
            try:
                self.cursor.execute("PRAGMA foreign_keys = ON")
                self.conn.commit()
            except sqlite3.Error as e:
                print(f"Warning: Could not set PRAGMA foreign_keys on provided sqlite3 connection: {e}")


        elif _HAS_SQLALCHEMY and isinstance(db, Engine):
            self.db_path = str(db.url)
            self.conn = db
            self._is_sqlalchemy = True


        else:
            raise TypeError(f"Unsupported type for CommandHistory db parameter: {type(db)}")

        self._initialize_schema()

    def _initialize_schema(self):
        """Creates all necessary tables."""
        print("Initializing database schema...")
        self.create_command_table()
        self.create_conversation_table()
        self.create_attachment_table()
        self.create_jinx_call_table()
        print("Database schema initialization complete.")

    def _execute(self, sql: str, params: Optional[Union[tuple, Dict]] = None, script: bool = False, requires_fk: bool = False) -> Optional[int]:
        """Executes SQL, handling transactions and FK pragma for SQLAlchemy. Returns lastrowid for INSERTs."""
        last_row_id = None
        try:
            if self._is_sqlalchemy:
                with self.conn.connect() as connection:
                    with connection.begin():
                        if requires_fk and self.conn.url.drivername == 'sqlite':
                             try: connection.execute(text("PRAGMA foreign_keys=ON"))
                             except SQLAlchemyError as e: print(f"Warning: SQLAlchemy PRAGMA foreign_keys=ON failed: {e}")

                        if script:
                             statements = [s.strip() for s in sql.split(';') if s.strip()]
                             for statement in statements:
                                  result_proxy = connection.execute(text(statement))
                                  if result_proxy.lastrowid is not None: last_row_id = result_proxy.lastrowid
                        else:
                             result_proxy = connection.execute(text(sql), params or {})
                             if result_proxy.lastrowid is not None: last_row_id = result_proxy.lastrowid
            else:
                 # Existing sqlite3 logic
                 if script:
                     self.cursor.executescript(sql)
                 else:
                     self.cursor.execute(sql, params or ())
                 last_row_id = self.cursor.lastrowid # Get lastrowid for sqlite3
                 self.conn.commit()

            return last_row_id

        except (sqlite3.Error, SQLAlchemyError) as e:
             error_type = "SQLAlchemy" if self._is_sqlalchemy else "SQLite"
             print(f"{error_type} Error executing: {sql[:100]}... Error: {e}")
             if not self._is_sqlalchemy:
                 try: self.conn.rollback()
                 except Exception as rb_err: print(f"SQLite rollback failed: {rb_err}")
             # Decide whether to raise the error or just return None/False
             raise # Re-raise the error to indicate failure
        except Exception as e:
            print(f"Unexpected error in _execute: {e}")
            raise # Re-raise unexpected errors


    def _fetch_one(self, sql: str, params: Optional[Union[tuple, Dict]] = None) -> Optional[Dict]:
        """Fetches a single row, adapting to connection type."""
        try:
            if self._is_sqlalchemy:
                 with self.conn.connect() as connection:
                      # No need for transaction for SELECT
                      result = connection.execute(text(sql), params or {})
                      row = result.fetchone()
                      return dict(row._mapping) if row else None
            else:
                 self.cursor.execute(sql, params or ())
                 row = self.cursor.fetchone()
                 return dict(row) if row else None
        except (sqlite3.Error, SQLAlchemyError) as e:
             error_type = "SQLAlchemy" if self._is_sqlalchemy else "SQLite"
             print(f"{error_type} Error fetching one: {sql[:100]}... Error: {e}")
             return None # Return None on error
        except Exception as e:
            print(f"Unexpected error in _fetch_one: {e}")
            return None

    def _fetch_all(self, sql: str, params: Optional[Union[tuple, Dict]] = None) -> List[Dict]:
        """Fetches all rows, adapting to connection type."""
        try:
            if self._is_sqlalchemy:
                with self.conn.connect() as connection:
                    # Convert tuple params to a dictionary format for SQLAlchemy
                    if params and isinstance(params, tuple):
                        # Extract parameter placeholders from SQL
                        placeholders = []
                        for i, char in enumerate(sql):
                            if char == '?':
                                placeholders.append(i)
                        
                        # Convert tuple params to dict format
                        dict_params = {}
                        for i, value in enumerate(params):
                            param_name = f"param_{i}"
                            # Replace ? with :param_name in SQL
                            sql = sql.replace('?', f":{param_name}", 1)
                            dict_params[param_name] = value
                        
                        params = dict_params
                    
                    result = connection.execute(text(sql), params or {})
                    rows = result.fetchall()
                    return [dict(row._mapping) for row in rows]
            else:
                self.cursor.execute(sql, params or ())
                rows = self.cursor.fetchall()
                return [dict(row) for row in rows]
        except (sqlite3.Error, SQLAlchemyError) as e:
            error_type = "SQLAlchemy" if self._is_sqlalchemy else "SQLite"
            print(f"{error_type} Error fetching all: {sql[:100]}... Error: {e}")
            return [] # Return empty list on error
        except Exception as e:
            print(f"Unexpected error in _fetch_all: {e}")
            return []

    def create_command_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS command_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT, command TEXT,
            subcommands TEXT, output TEXT, location TEXT
        )"""
        self._execute(query)

    def create_conversation_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS conversation_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT, message_id TEXT UNIQUE NOT NULL,
            timestamp TEXT, role TEXT, content TEXT, conversation_id TEXT,
            directory_path TEXT, model TEXT, provider TEXT, npc TEXT, team TEXT
        )"""
        self._execute(query)

    def create_attachment_table(self):
        query = """
        CREATE TABLE IF NOT EXISTS message_attachments (
            id INTEGER PRIMARY KEY AUTOINCREMENT, message_id TEXT NOT NULL,
            attachment_name TEXT, attachment_type TEXT, attachment_data BLOB,
            attachment_size INTEGER, upload_timestamp TEXT,
            FOREIGN KEY (message_id) REFERENCES conversation_history(message_id) ON DELETE CASCADE
        )"""
        self._execute(query, requires_fk=True)

    def create_jinx_call_table(self):
        table_query = '''
        CREATE TABLE IF NOT EXISTS jinx_execution_log (
            execution_id INTEGER PRIMARY KEY AUTOINCREMENT, triggering_message_id TEXT NOT NULL,
            response_message_id TEXT, conversation_id TEXT NOT NULL, timestamp TEXT NOT NULL,
            npc_name TEXT, team_name TEXT, jinx_name TEXT NOT NULL, jinx_inputs TEXT,
            jinx_output TEXT, status TEXT NOT NULL, error_message TEXT, duration_ms INTEGER,
            FOREIGN KEY (triggering_message_id) REFERENCES conversation_history(message_id) ON DELETE CASCADE,
            FOREIGN KEY (response_message_id) REFERENCES conversation_history(message_id) ON DELETE SET NULL
        );
        '''
        self._execute(table_query, requires_fk=True)

        index_queries = [
            "CREATE INDEX IF NOT EXISTS idx_jinx_log_trigger_msg ON jinx_execution_log (triggering_message_id);",
            "CREATE INDEX IF NOT EXISTS idx_jinx_log_convo_id ON jinx_execution_log (conversation_id);",
            "CREATE INDEX IF NOT EXISTS idx_jinx_log_jinx_name ON jinx_execution_log (jinx_name);",
            "CREATE INDEX IF NOT EXISTS idx_jinx_log_timestamp ON jinx_execution_log (timestamp);"
        ]
        for idx_query in index_queries:
             self._execute(idx_query)

    def add_command(self, command, subcommands, output, location):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        safe_subcommands = str(subcommands)
        safe_output = str(output)
        sql = """
            INSERT INTO command_history (timestamp, command, subcommands, output, location)
            VALUES (?, ?, ?, ?, ?)
            """
        params = (timestamp, command, safe_subcommands, safe_output, location)
        self._execute(sql, params)

    def generate_message_id(self) -> str:
        return str(uuid.uuid4())

    def add_conversation(
        self, role, content, conversation_id, directory_path,
        model=None, provider=None, npc=None, team=None,
        attachments=None, message_id=None,
    ):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if message_id is None: message_id = self.generate_message_id()
        if isinstance(content, dict): content = json.dumps(content, cls=CustomJSONEncoder)

        existing_row = self._fetch_one(
             "SELECT content FROM conversation_history WHERE message_id = ?", (message_id,)
        )

        if existing_row:
            sql = "UPDATE conversation_history SET content = ?, timestamp = ? WHERE message_id = ?"
            params = (content, timestamp, message_id)
            self._execute(sql, params)
        else:
            sql = """INSERT INTO conversation_history
                (message_id, timestamp, role, content, conversation_id, directory_path, model, provider, npc, team)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
            params = (message_id, timestamp, role, content, conversation_id, directory_path, model, provider, npc, team,)
            self._execute(sql, params)

        if attachments:
            for attachment in attachments:
                self.add_attachment(
                    message_id, attachment["name"], attachment["type"],
                    attachment["data"], attachment_size=attachment.get("size"),
                )
        return message_id

    def add_attachment(
        self, message_id, attachment_name, attachment_type, attachment_data, attachment_size=None,
    ):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if attachment_size is None and attachment_data is not None:
            attachment_size = len(attachment_data)
        sql = """INSERT INTO message_attachments
            (message_id, attachment_name, attachment_type, attachment_data, attachment_size, upload_timestamp)
            VALUES (?, ?, ?, ?, ?, ?)"""
        params = (message_id, attachment_name, attachment_type, attachment_data, attachment_size, timestamp,)
        self._execute(sql, params)

    def save_jinx_execution(
        self, triggering_message_id: str, conversation_id: str, npc_name: Optional[str],
        jinx_name: str, jinx_inputs: Dict, jinx_output: Any, status: str,
        team_name: Optional[str] = None, error_message: Optional[str] = None,
        response_message_id: Optional[str] = None, duration_ms: Optional[int] = None
    ):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try: inputs_json = json.dumps(jinx_inputs, cls=CustomJSONEncoder)
        except TypeError: inputs_json = json.dumps(str(jinx_inputs))
        try:
            if isinstance(jinx_output, (str, int, float, bool, list, dict, type(None))):
                 outputs_json = json.dumps(jinx_output, cls=CustomJSONEncoder)
            else: outputs_json = json.dumps(str(jinx_output))
        except TypeError: outputs_json = json.dumps(f"Non-serializable output: {type(jinx_output)}")

        sql = """INSERT INTO jinx_execution_log
            (triggering_message_id, conversation_id, timestamp, npc_name, team_name,
             jinx_name, jinx_inputs, jinx_output, status, error_message, response_message_id, duration_ms)
             VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
        params = (triggering_message_id, conversation_id, timestamp, npc_name, team_name,
                  jinx_name, inputs_json, outputs_json, status, error_message, response_message_id, duration_ms)
        try:
             return self._execute(sql, params) # Return lastrowid if available
        except Exception as e:
             print(f"CRITICAL: Failed to save tool execution via _execute: {e}")
             return None

    def get_full_message_content(self, message_id):
        sql = "SELECT content FROM conversation_history WHERE message_id = ? ORDER BY timestamp ASC"
        rows = self._fetch_all(sql, (message_id,))
        return "".join(row['content'] for row in rows)

    def update_message_content(self, message_id, full_content):
        sql = "UPDATE conversation_history SET content = ? WHERE message_id = ?"
        params = (full_content, message_id)
        self._execute(sql, params)

    def get_message_attachments(self, message_id) -> List[Dict]:
        sql = """SELECT id, message_id, attachment_name, attachment_type, attachment_size, upload_timestamp
                 FROM message_attachments WHERE message_id = ?"""
        return self._fetch_all(sql, (message_id,))

    def get_attachment_data(self, attachment_id) -> Optional[Tuple[bytes, str, str]]:
        sql = "SELECT attachment_data, attachment_name, attachment_type FROM message_attachments WHERE id = ?"
        row = self._fetch_one(sql, (attachment_id,))
        if row:
            return row['attachment_data'], row['attachment_name'], row['attachment_type']
        return None, None, None

    def delete_attachment(self, attachment_id) -> bool:
        sql = "DELETE FROM message_attachments WHERE id = ?"
        try:
            # _execute might not return rowcount reliably across drivers
            self._execute(sql, (attachment_id,))
            # We assume success if no exception was raised.
            # A more robust check might involve trying to fetch the deleted row.
            return True
        except Exception as e:
            print(f"Error deleting attachment {attachment_id}: {e}")
            return False

    def get_last_command(self) -> Optional[Dict]:
        sql = "SELECT * FROM command_history ORDER BY id DESC LIMIT 1"
        return self._fetch_one(sql)

    def get_most_recent_conversation_id(self) -> Optional[Dict]:
        sql = "SELECT conversation_id FROM conversation_history ORDER BY id DESC LIMIT 1"
        # Returns dict like {'conversation_id': '...'} or None
        return self._fetch_one(sql)

    def get_last_conversation(self, conversation_id) -> Optional[Dict]:
        sql = """SELECT * FROM conversation_history WHERE conversation_id = ? and role = 'user'
                 ORDER BY id DESC LIMIT 1"""
        return self._fetch_one(sql, (conversation_id,))

    def get_messages_by_npc(self, npc, n_last=20) -> List[Dict]:
        sql = """SELECT * FROM conversation_history WHERE npc = ?
                    ORDER BY timestamp DESC LIMIT ?"""
        params = (npc, n_last)

        return self._fetch_all(sql, params)
    def get_messages_by_team(self, team, n_last=20) -> List[Dict]:
        sql = """SELECT * FROM conversation_history WHERE team = ?
                    ORDER BY timestamp DESC LIMIT ?"""
        params = (team, n_last)

        return self._fetch_all(sql, params)
    def get_message_by_id(self, message_id) -> Optional[Dict]:
        sql = "SELECT * FROM conversation_history WHERE message_id = ?"
        return self._fetch_one(sql, (message_id,))

    def get_most_recent_conversation_id_by_path(self, path) -> Optional[Dict]:
        sql = """SELECT conversation_id FROM conversation_history WHERE directory_path = ?
                 ORDER BY timestamp DESC LIMIT 1"""
        # Returns dict like {'conversation_id': '...'} or None
        return self._fetch_one(sql, (path,))

    def get_last_conversation_by_path(self, directory_path) -> Optional[List[Dict]]:
        result_dict = self.get_most_recent_conversation_id_by_path(directory_path)
        if result_dict and result_dict.get('conversation_id'):
            convo_id = result_dict['conversation_id']
            return self.get_conversations_by_id(convo_id)
        return None

    def get_conversations_by_id(self, conversation_id: str) -> List[Dict[str, Any]]:
        sql = """SELECT id, message_id, timestamp, role, content, conversation_id,
                 directory_path, model, provider, npc, team
                 FROM conversation_history WHERE conversation_id = ? ORDER BY timestamp ASC"""
        results = self._fetch_all(sql, (conversation_id,))
        for message_dict in results:
             attachments = self.get_message_attachments(message_dict["message_id"])
             if attachments: message_dict["attachments"] = attachments
        return results

    def get_npc_conversation_stats(self, start_date=None, end_date=None) -> pd.DataFrame:
        date_filter = ""
        params = {} # Use dict for named parameters with SQLAlchemy/read_sql
        if start_date and end_date:
            date_filter = "WHERE timestamp BETWEEN :start_date AND :end_date"
            params = {"start_date": start_date, "end_date": end_date}
        elif start_date:
            date_filter = "WHERE timestamp >= :start_date"
            params = {"start_date": start_date}
        elif end_date:
            date_filter = "WHERE timestamp <= :end_date"
            params = {"end_date": end_date}


        query = f"""
        SELECT
            npc,
            COUNT(*) as total_messages,
            AVG(LENGTH(content)) as avg_message_length,
            COUNT(DISTINCT conversation_id) as total_conversations,
            COUNT(DISTINCT model) as models_used,
            COUNT(DISTINCT provider) as providers_used,
            GROUP_CONCAT(DISTINCT model) as model_list,
            GROUP_CONCAT(DISTINCT provider) as provider_list,
            MIN(timestamp) as first_conversation,
            MAX(timestamp) as last_conversation
        FROM conversation_history
        {date_filter}
        GROUP BY npc
        ORDER BY total_messages DESC
        """
        try:
            # Use pd.read_sql with the appropriate connection object
            if self._is_sqlalchemy:
                # read_sql works directly with SQLAlchemy Engine
                df = pd.read_sql(sql=text(query), con=self.conn, params=params)
            else:
                # read_sql works directly with sqlite3 Connection
                df = pd.read_sql(sql=query, con=self.conn, params=params)
            return df
        except Exception as e:
             print(f"Error fetching conversation stats with pandas: {e}")
             # Fallback or return empty DataFrame
             return pd.DataFrame(columns=[
                 'npc', 'total_messages', 'avg_message_length', 'total_conversations',
                 'models_used', 'providers_used', 'model_list', 'provider_list',
                 'first_conversation', 'last_conversation'
             ])


    def get_command_patterns(self, timeframe='day') -> pd.DataFrame:
        time_group_formats = {
            'hour': "strftime('%Y-%m-%d %H', timestamp)",
            'day': "strftime('%Y-%m-%d', timestamp)",
            'week': "strftime('%Y-%W', timestamp)",
            'month': "strftime('%Y-%m', timestamp)"
        }
        time_group = time_group_formats.get(timeframe, "strftime('%Y-%m-%d', timestamp)") # Default to day

        query = f"""
        WITH parsed_commands AS (
            SELECT
                {time_group} as time_bucket,
                CASE
                    WHEN command LIKE '/%%' THEN SUBSTR(command, 2, INSTR(SUBSTR(command, 2), ' ') - 1)
                    WHEN command LIKE 'npc %%' THEN SUBSTR(command, 5, INSTR(SUBSTR(command, 5), ' ') - 1)
                    ELSE command
                END as base_command
            FROM command_history
            WHERE timestamp IS NOT NULL -- Added check for null timestamps
        )
        SELECT
            time_bucket,
            base_command,
            COUNT(*) as usage_count
        FROM parsed_commands
        WHERE base_command IS NOT NULL AND base_command != '' -- Filter out potential null/empty commands
        GROUP BY time_bucket, base_command
        ORDER BY time_bucket DESC, usage_count DESC
        """
        try:
             # Use pd.read_sql
             if self._is_sqlalchemy:
                  df = pd.read_sql(sql=text(query), con=self.conn)
             else:
                  df = pd.read_sql(sql=query, con=self.conn)
             return df
        except Exception as e:
             print(f"Error fetching command patterns with pandas: {e}")
             return pd.DataFrame(columns=['time_period', 'command', 'count'])

    # --- Implementation for search_commands and get_all_commands (used by show_history) ---
    # These are examples; adjust based on your needs.
    def search_commands(self, search_term: str) -> List[Dict]:
        """Searches command history table for a term."""
        # Use LOWER() for case-insensitive search
        sql = """
            SELECT id, timestamp, command, subcommands, output, location
            FROM command_history
            WHERE LOWER(command) LIKE LOWER(?) OR LOWER(output) LIKE LOWER(?)
            ORDER BY timestamp DESC
            LIMIT 5
        """
        like_term = f"%{search_term}%"
        return self._fetch_all(sql, (like_term, like_term))

    def get_all_commands(self, limit: int = 100) -> List[Dict]:
        """Gets the most recent commands."""
        sql = """
            SELECT id, timestamp, command, subcommands, output, location
            FROM command_history
            ORDER BY id DESC
            LIMIT ?
        """
        return self._fetch_all(sql, (limit,))


    def close(self):
        """Closes the connection if it's a direct sqlite3 connection."""
        if not self._is_sqlalchemy and self.conn:
            try:
                self.conn.close()
                print("Closed sqlite3 connection.")
            except Exception as e:
                print(f"Error closing sqlite3 connection: {e}")
        elif self._is_sqlalchemy:
             print("SQLAlchemy Engine pool managed automatically.")
        self.conn = None
def start_new_conversation(prepend = '') -> str:
    """
    Starts a new conversation and returns a unique conversation ID.
    """
    if prepend =='':
        prepend = 'npcsh'
    return f"{prepend}_{datetime.now().strftime('%Y%m%d%H%M%S')}"


def save_conversation_message(
    command_history: CommandHistory,
    conversation_id: str,
    role: str,
    content: str,
    wd: str = None,
    model: str = None,
    provider: str = None,
    npc: str = None,
    team: str = None,
    attachments: List[Dict] = None,
    message_id: str = None,
):
    """
    Saves a conversation message linked to a conversation ID with optional attachments.

    Args:
        command_history: The CommandHistory instance
        conversation_id: The conversation identifier
        role: The message sender role ('user', 'assistant', etc.)
        content: The message content
        wd: Working directory (defaults to current directory)
        model: The model identifier (optional)
        provider: The provider identifier (optional)
        npc: The NPC identifier (optional)
        attachments: List of attachment dictionaries (optional)
            Each attachment dict should have:
            - name: Filename/title
            - type: MIME type or extension
            - data: Binary blob data
            - size: Size in bytes (optional)

    Returns:
        The message ID
    """
    if wd is None:
        wd = os.getcwd()

    return command_history.add_conversation(
        role=role,
        content=content,
        conversation_id=conversation_id,
        directory_path=wd,
        model=model,
        provider=provider,
        npc=npc,
        team=team,
        attachments=attachments,
        message_id=message_id,
    )


def retrieve_last_conversation(
    command_history: CommandHistory, conversation_id: str
) -> str:
    """
    Retrieves and formats all messages from the last conversation.
    """
    last_message = command_history.get_last_conversation(conversation_id)
    if last_message:
        return last_message[3]  # content
    return "No previous conversation messages found."


def save_attachment_to_message(
    command_history: CommandHistory,
    message_id: str,
    file_path: str,
    attachment_name: str = None,
    attachment_type: str = None,
):
    """
    Helper function to save a file from disk as an attachment.

    Args:
        command_history: The CommandHistory instance
        message_id: The message ID to attach to
        file_path: Path to the file on disk
        attachment_name: Name to save (defaults to basename)
        attachment_type: MIME type (defaults to guessing from extension)

    Returns:
        Boolean indicating success
    """
    try:
        # Get file name if not specified
        if not attachment_name:
            attachment_name = os.path.basename(file_path)

        # Try to guess MIME type if not specified
        if not attachment_type:
            _, ext = os.path.splitext(file_path)
            if ext:
                attachment_type = ext.lower()[1:]  # Remove the dot

        # Read file data
        with open(file_path, "rb") as f:
            data = f.read()

        # Add attachment
        command_history.add_attachment(
            message_id=message_id,
            attachment_name=attachment_name,
            attachment_type=attachment_type,
            attachment_data=data,
            attachment_size=len(data),
        )
        return True
    except Exception as e:
        print(f"Error saving attachment: {str(e)}")
        return False

def get_available_tables(db_path: str) -> str:
    """
    Function Description:
        This function gets the available tables in the database.
    Args:
        db_path (str): The database path.
    Keyword Args:
        None
    Returns:
        str: The available tables in the database.
    """
    if '~' in db_path:
        db_path = os.path.expanduser(db_path)
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name != 'command_history'"
            )
            tables = cursor.fetchall()

            return tables
    except Exception as e:
        print(f"Error getting available tables: {e}")
        return ""
    
    

'''
from npcpy.memory.command_history import CommandHistory
command_history = CommandHistory()

sibiji_messages = command_history.get_messages_by_npc('sibiji', n_last=10)

stats = command_history.get_npc_conversation_stats()




from npcpy.memory.command_history import CommandHistory
command_history = CommandHistory()
command_history.create_tool_call_table()
'''