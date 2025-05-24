"""
Memory integration for all REPL loops in npcsh.

This module provides integration between different REPL loops and 
the knowledge graph memory system, allowing consistent memory access
across all interaction modes.
"""

import os
import json
import sqlite3
import datetime
from typing import Dict, List, Any, Optional

from npcpy.memory.sleep import breathe, sleep, recall
from npcpy.memory.knowledge_graph import (
    extract_facts,
    extract_mistakes,
    extract_lessons_learned,
    process_text
)

def initialize_memory_db(db_path: str) -> sqlite3.Connection:
    """
    Initialize the database for memory operations.
    Uses the same structure as command_history to ensure compatibility.
    
    Args:
        db_path: Path to the SQLite database
        
    Returns:
        SQLite connection object
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables for knowledge graph
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS facts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT NOT NULL UNIQUE,
        source TEXT NOT NULL,
        created_at TIMESTAMP NOT NULL
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS mistakes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT NOT NULL UNIQUE,
        source TEXT NOT NULL,
        created_at TIMESTAMP NOT NULL
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS lessons (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT NOT NULL UNIQUE,
        source TEXT NOT NULL,
        created_at TIMESTAMP NOT NULL
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS actions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT NOT NULL UNIQUE,
        source TEXT NOT NULL,
        created_at TIMESTAMP NOT NULL
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        content TEXT NOT NULL UNIQUE,
        source TEXT NOT NULL,
        created_at TIMESTAMP NOT NULL
    )
    ''')
    
    # Create memory breathing sessions table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS breathing_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        repl_type TEXT NOT NULL,
        session_id TEXT NOT NULL,
        start_time TIMESTAMP NOT NULL,
        end_time TIMESTAMP,
        extraction_count INTEGER DEFAULT 0,
        status TEXT NOT NULL
    )
    ''')
    
    conn.commit()
    return conn

def process_repl_memory(
    conversation: List[Dict[str, str]],
    repl_type: str,
    session_id: str,
    db_path: str,
    chroma_path: str,
    model: str = "llama3.2",
    provider: str = "ollama",
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Process memory from any REPL loop and integrate it with knowledge graph.
    
    Args:
        conversation: List of conversation turns in dict format
        repl_type: Type of REPL (alicanto, yap, guac, etc.)
        session_id: Unique identifier for the session
        db_path: Path to the SQLite database
        chroma_path: Path to the ChromaDB directory
        model: LLM model to use
        provider: LLM provider
        verbose: Whether to print verbose output
        
    Returns:
        Dict with results of the memory processing
    """
    results = {
        "success": False,
        "extraction_count": 0,
        "facts": [],
        "mistakes": [],
        "lessons": [],
        "actions": [],
        "decisions": [],
        "start_time": datetime.datetime.now().isoformat(),
        "end_time": None,
        "duration": 0,
        "errors": []
    }
    
    try:
        # Initialize database
        conn = initialize_memory_db(db_path)
        cursor = conn.cursor()
        
        # Log start of breathing session
        cursor.execute(
            "INSERT INTO breathing_sessions (repl_type, session_id, start_time, status) VALUES (?, ?, datetime('now'), ?)",
            (repl_type, session_id, "started")
        )
        breathing_id = cursor.lastrowid
        conn.commit()
        
        if verbose:
            print(f"🧠 {repl_type} is breathing and processing memories...")
        
        # Convert conversation to text
        conversation_text = ""
        for turn in conversation:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            conversation_text += f"{role}: {content}\n\n"
        
        # Extract knowledge using breathe function
        if verbose:
            print("🔍 Extracting knowledge from conversation...")
        
        context = f"REPL type: {repl_type}, Session: {session_id}"
        extraction_results = {
            "facts": extract_facts(conversation_text, model=model, provider=provider, context=context),
            "mistakes": extract_mistakes(conversation_text, model=model, provider=provider, context=context),
            "lessons": extract_lessons_learned(conversation_text, model=model, provider=provider, context=context)
        }
        
        # Store results
        results["facts"] = extraction_results.get("facts", [])
        results["mistakes"] = extraction_results.get("mistakes", [])
        results["lessons"] = extraction_results.get("lessons", [])
        
        # Calculate extraction count
        extraction_count = sum(len(items) for items in extraction_results.values() if isinstance(items, list))
        results["extraction_count"] = extraction_count
        
        # Store in SQLite
        if verbose:
            print("💾 Storing knowledge in database...")
            
        # Store facts
        for fact in results["facts"]:
            cursor.execute(
                "INSERT OR IGNORE INTO facts (content, source, created_at) VALUES (?, ?, datetime('now'))",
                (fact, f"{repl_type}:{session_id}")
            )
            
        # Store mistakes
        for mistake in results["mistakes"]:
            cursor.execute(
                "INSERT OR IGNORE INTO mistakes (content, source, created_at) VALUES (?, ?, datetime('now'))",
                (mistake, f"{repl_type}:{session_id}")
            )
            
        # Store lessons
        for lesson in results["lessons"]:
            cursor.execute(
                "INSERT OR IGNORE INTO lessons (content, source, created_at) VALUES (?, ?, datetime('now'))",
                (lesson, f"{repl_type}:{session_id}")
            )
        
        # Update session information
        end_time = datetime.datetime.now()
        results["end_time"] = end_time.isoformat()
        results["duration"] = (end_time - datetime.datetime.fromisoformat(results["start_time"])).total_seconds()
        
        cursor.execute(
            "UPDATE breathing_sessions SET end_time = datetime('now'), status = ?, extraction_count = ? WHERE id = ?",
            ("completed", extraction_count, breathing_id)
        )
        conn.commit()
        
        # Store in vector database for semantic search
        from npcpy.memory.knowledge_graph import store_in_vector_db
        store_in_vector_db(chroma_path, conversation_text, extraction_results, f"{repl_type}:{session_id}")
        
        if verbose:
            print(f"✅ Breathing completed. Extracted {extraction_count} items.")
            print(f"⏱️ Duration: {results['duration']:.2f} seconds")
        
        results["success"] = True
    except Exception as e:
        results["errors"].append(str(e))
        if verbose:
            print(f"❌ Error during breathing: {str(e)}")
        
        # Log error in database if breathing_id exists
        try:
            if 'breathing_id' in locals():
                cursor.execute(
                    "UPDATE breathing_sessions SET status = ? WHERE id = ?",
                    ("error", breathing_id)
                )
                conn.commit()
        except Exception as inner_e:
            results["errors"].append(f"Failed to log error: {str(inner_e)}")
    finally:
        # Close database connection
        if 'conn' in locals():
            conn.close()
    
    return results

def get_relevant_memories(
    query: str,
    db_path: str,
    chroma_path: str,
    repl_type: Optional[str] = None,
    top_k: int = 5,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Retrieve relevant memories for a given query.
    
    Args:
        query: The query to search for memories
        db_path: Path to the SQLite database
        chroma_path: Path to the ChromaDB directory
        repl_type: Optional filter for specific REPL type
        top_k: Maximum number of results to return
        verbose: Whether to print verbose output
        
    Returns:
        Dict with search results
    """
    results = {
        "facts": [],
        "mistakes": [],
        "lessons": [],
        "vector_results": [],
        "success": False,
        "errors": []
    }
    
    try:
        # Initialize database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        if verbose:
            print(f"🔍 Searching memories for: '{query}'")
        
        # Build source filter if repl_type is provided
        source_filter = f"AND source LIKE '{repl_type}:%'" if repl_type else ""
        
        # Search for facts
        cursor.execute(
            f"SELECT id, content, source FROM facts WHERE content LIKE ? {source_filter} LIMIT ?",
            (f"%{query}%", top_k)
        )
        results["facts"] = [{"id": row[0], "content": row[1], "source": row[2]} for row in cursor.fetchall()]
        
        # Search for mistakes
        cursor.execute(
            f"SELECT id, content, source FROM mistakes WHERE content LIKE ? {source_filter} LIMIT ?",
            (f"%{query}%", top_k)
        )
        results["mistakes"] = [{"id": row[0], "content": row[1], "source": row[2]} for row in cursor.fetchall()]
        
        # Search for lessons
        cursor.execute(
            f"SELECT id, content, source FROM lessons WHERE content LIKE ? {source_filter} LIMIT ?",
            (f"%{query}%", top_k)
        )
        results["lessons"] = [{"id": row[0], "content": row[1], "source": row[2]} for row in cursor.fetchall()]
        
        # Vector search from ChromaDB
        from npcpy.memory.knowledge_graph import retrieve_relevant_memory
        vector_results = retrieve_relevant_memory(query, chroma_path, top_k)
        
        # Filter by repl_type if specified
        if repl_type and vector_results:
            vector_results = [r for r in vector_results if r.get("source", "").startswith(f"{repl_type}:")]
            
        results["vector_results"] = vector_results
        
        if verbose:
            total_results = sum(len(results[key]) for key in ["facts", "mistakes", "lessons"])
            vector_count = len(results["vector_results"])
            print(f"✅ Found {total_results} direct matches and {vector_count} semantic matches.")
        
        results["success"] = True
    except Exception as e:
        results["errors"].append(str(e))
        if verbose:
            print(f"❌ Error retrieving memories: {str(e)}")
    finally:
        # Close database connection
        if 'conn' in locals():
            conn.close()
    
    return results

def list_breathing_sessions(
    db_path: str,
    limit: int = 10,
    repl_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    List recent breathing sessions.
    
    Args:
        db_path: Path to the SQLite database
        limit: Maximum number of sessions to return
        repl_type: Optional filter for specific REPL type
        
    Returns:
        List of session information dictionaries
    """
    sessions = []
    
    try:
        # Initialize database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Build query with optional repl_type filter
        query = "SELECT id, repl_type, session_id, start_time, end_time, extraction_count, status FROM breathing_sessions"
        params = []
        
        if repl_type:
            query += " WHERE repl_type = ?"
            params.append(repl_type)
            
        query += " ORDER BY start_time DESC LIMIT ?"
        params.append(limit)
        
        # Execute query
        cursor.execute(query, params)
        
        for row in cursor.fetchall():
            session_id, repl_type, session_key, start_time, end_time, extraction_count, status = row
            
            # Calculate duration if available
            duration = None
            if start_time and end_time:
                try:
                    start_dt = datetime.datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                    end_dt = datetime.datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                    duration = (end_dt - start_dt).total_seconds()
                except:
                    pass
            
            sessions.append({
                "id": session_id,
                "repl_type": repl_type,
                "session_id": session_key,
                "start_time": start_time,
                "end_time": end_time,
                "extraction_count": extraction_count,
                "status": status,
                "duration": duration
            })
        
        conn.close()
    except Exception as e:
        print(f"Error listing breathing sessions: {str(e)}")
    
    return sessions