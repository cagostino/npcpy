"""
Sleep module for NPC agents.

This module provides functions for the "sleep" process of NPC agents, which includes:
1. Breathing: extracting facts, lessons, and mistakes from conversation history
2. Processing: organizing and storing these extractions in knowledge graphs
3. Consolidation: merging similar memories and updating belief structures
4. Integration: connecting new knowledge with existing knowledge
"""

import os
import json
import time
import datetime
import sqlite3
from typing import Dict, List, Tuple, Any, Optional


def initialize_sleep_db(db_path: str) -> sqlite3.Connection:
    """
    Initialize the database for sleep operations.
    
    Args:
        db_path: Path to the SQLite database
        
    Returns:
        SQLite connection object
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create sleep_sessions table to track sleeping sessions
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sleep_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        start_time TIMESTAMP NOT NULL,
        end_time TIMESTAMP,
        conversation_source TEXT NOT NULL,
        status TEXT NOT NULL,
        extraction_count INTEGER DEFAULT 0,
        consolidation_count INTEGER DEFAULT 0
    )
    ''')
    
    # Create sleep_logs table for detailed logging
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sleep_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        timestamp TIMESTAMP NOT NULL,
        event_type TEXT NOT NULL,
        details TEXT,
        FOREIGN KEY (session_id) REFERENCES sleep_sessions(id)
    )
    ''')
    
    conn.commit()
    return conn

def sleep(conversation_history: List[Dict], 
          db_path: str, 
          chroma_path: str,
          agent_name: str = "default_agent",
          sleep_duration: int = 5,
          verbose: bool = False) -> Dict:
    """
    Main sleep function for NPC agents. This processes conversation history,
    extracts knowledge, and integrates it into the agent's memory.
    
    Args:
        conversation_history: List of conversation turns in dict format
        db_path: Path to the SQLite database
        chroma_path: Path to the ChromaDB directory
        agent_name: Name of the agent
        sleep_duration: Duration to simulate sleep in seconds
        verbose: Whether to print verbose output
        
    Returns:
        Dict with results of the sleep operation
    """
    # Initialize results dictionary
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
        # Connect to database
        conn = initialize_sleep_db(db_path)
        cursor = conn.cursor()
        
        # Log start of sleep session
        cursor.execute(
            "INSERT INTO sleep_sessions (start_time, conversation_source, status) VALUES (datetime('now'), ?, ?)",
            (agent_name, "started")
        )
        session_id = cursor.lastrowid
        conn.commit()
        
        if verbose:
            print(f"🌙 {agent_name} is sleeping and processing memories...")
        
        # Simulate sleep processing time
        time.sleep(sleep_duration)
        
        # Process conversation history
        conversation_text = ""
        for turn in conversation_history:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            conversation_text += f"{role}: {content}\n\n"
        
        # Extract knowledge using breathe
        if verbose:
            print("🧠 Extracting knowledge from conversation...")
            
        extraction_results = breathe(conversation_text)
        
        # Store results
        results["facts"] = extraction_results.get("facts", [])
        results["mistakes"] = extraction_results.get("mistakes", [])
        results["lessons"] = extraction_results.get("lessons", [])
        results["actions"] = extraction_results.get("actions", [])
        results["decisions"] = extraction_results.get("decisions", [])
        
        # Calculate extraction count
        extraction_count = sum(len(items) for items in extraction_results.values() if isinstance(items, list))
        results["extraction_count"] = extraction_count
        
        # Store in SQLite
        if verbose:
            print("💾 Storing knowledge in database...")
            
        # Store facts
        for fact in results["facts"]:
            add_fact(conn, fact, agent_name)
            cursor.execute(
                "INSERT INTO sleep_logs (session_id, timestamp, event_type, details) VALUES (?, datetime('now'), ?, ?)",
                (session_id, "fact_added", fact)
            )
        
        # Store mistakes
        for mistake in results["mistakes"]:
            add_mistake(conn, mistake, agent_name)
            cursor.execute(
                "INSERT INTO sleep_logs (session_id, timestamp, event_type, details) VALUES (?, datetime('now'), ?, ?)",
                (session_id, "mistake_added", mistake)
            )
        
        # Store lessons
        for lesson in results["lessons"]:
            add_lesson(conn, lesson, agent_name)
            cursor.execute(
                "INSERT INTO sleep_logs (session_id, timestamp, event_type, details) VALUES (?, datetime('now'), ?, ?)",
                (session_id, "lesson_added", lesson)
            )
        
        # Store actions
        for action in results["actions"]:
            add_action(conn, action, agent_name)
            cursor.execute(
                "INSERT INTO sleep_logs (session_id, timestamp, event_type, details) VALUES (?, datetime('now'), ?, ?)",
                (session_id, "action_added", action)
            )
        
        # Store decisions
        for decision in results["decisions"]:
            add_decision(conn, decision, agent_name)
            cursor.execute(
                "INSERT INTO sleep_logs (session_id, timestamp, event_type, details) VALUES (?, datetime('now'), ?, ?)",
                (session_id, "decision_added", decision)
            )
        
        # Store in vector database
        store_in_vector_db(chroma_path, conversation_text, extraction_results, agent_name)
        
        # Update session information
        end_time = datetime.datetime.now()
        results["end_time"] = end_time.isoformat()
        results["duration"] = (end_time - datetime.datetime.fromisoformat(results["start_time"])).total_seconds()
        
        cursor.execute(
            "UPDATE sleep_sessions SET end_time = datetime('now'), status = ?, extraction_count = ? WHERE id = ?",
            ("completed", extraction_count, session_id)
        )
        conn.commit()
        
        if verbose:
            print(f"✅ Sleep completed. Extracted {extraction_count} items.")
            print(f"⏱️ Duration: {results['duration']:.2f} seconds")
        
        results["success"] = True
    except Exception as e:
        results["errors"].append(str(e))
        if verbose:
            print(f"❌ Error during sleep: {str(e)}")
        
        # Log error in database if session_id exists
        try:
            if 'session_id' in locals():
                cursor.execute(
                    "UPDATE sleep_sessions SET status = ? WHERE id = ?",
                    ("error", session_id)
                )
                cursor.execute(
                    "INSERT INTO sleep_logs (session_id, timestamp, event_type, details) VALUES (?, datetime('now'), ?, ?)",
                    (session_id, "error", str(e))
                )
                conn.commit()
        except Exception as inner_e:
            results["errors"].append(f"Failed to log error: {str(inner_e)}")
    finally:
        # Close database connection
        if 'conn' in locals():
            conn.close()
    
    return results

def recall(query: str, 
           db_path: str, 
           chroma_path: str, 
           top_k: int = 5, 
           include_vector_search: bool = True,
           verbose: bool = False) -> Dict:
    """
    Recall information from the agent's memory.
    
    Args:
        query: The query to search for
        db_path: Path to the SQLite database
        chroma_path: Path to the ChromaDB directory
        top_k: Number of results to return
        include_vector_search: Whether to include vector search results
        verbose: Whether to print verbose output
        
    Returns:
        Dict with search results
    """
    results = {
        "facts": [],
        "mistakes": [],
        "lessons": [],
        "actions": [],
        "decisions": [],
        "vector_results": [],
        "success": False,
        "errors": []
    }
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        
        # Search in SQL database
        if verbose:
            print(f"🔍 Searching for: '{query}'")
            
        # Search for facts
        facts = search_facts(conn, query)
        results["facts"] = [{"id": f[0], "content": f[1], "source": f[2]} for f in facts]
        
        # Search for mistakes
        mistakes = search_mistakes(conn, query)
        results["mistakes"] = [{"id": m[0], "content": m[1], "source": m[2]} for m in mistakes]
        
        # Search for lessons
        lessons = search_lessons(conn, query)
        results["lessons"] = [{"id": l[0], "content": l[1], "source": l[2]} for l in lessons]
        
        # Search for actions
        actions = search_actions(conn, query)
        results["actions"] = [{"id": a[0], "content": a[1], "source": a[2]} for a in actions]
        
        # Search for decisions
        decisions = search_decisions(conn, query)
        results["decisions"] = [{"id": d[0], "content": d[1], "source": d[2]} for d in decisions]
        
        # Vector search
        if include_vector_search:
            vector_results = retrieve_relevant_memory(query, chroma_path, top_k)
            results["vector_results"] = vector_results
            
        if verbose:
            total_results = sum(len(results[key]) for key in ["facts", "mistakes", "lessons", "actions", "decisions"])
            vector_count = len(results["vector_results"])
            print(f"✅ Found {total_results} direct matches and {vector_count} semantic matches.")
            
        results["success"] = True
    except Exception as e:
        results["errors"].append(str(e))
        if verbose:
            print(f"❌ Error during recall: {str(e)}")
    finally:
        # Close database connection
        if 'conn' in locals():
            conn.close()
            
    return results

def list_sleep_sessions(db_path: str, limit: int = 10) -> List[Dict]:
    """
    List recent sleep sessions.
    
    Args:
        db_path: Path to the SQLite database
        limit: Maximum number of sessions to return
        
    Returns:
        List of session information dictionaries
    """
    sessions = []
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Query sessions
        cursor.execute("""
            SELECT id, start_time, end_time, conversation_source, status, extraction_count, consolidation_count 
            FROM sleep_sessions 
            ORDER BY start_time DESC 
            LIMIT ?
        """, (limit,))
        
        for row in cursor.fetchall():
            session_id, start_time, end_time, source, status, extraction_count, consolidation_count = row
            
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
                "start_time": start_time,
                "end_time": end_time,
                "source": source,
                "status": status,
                "extraction_count": extraction_count,
                "consolidation_count": consolidation_count,
                "duration": duration
            })
        
        conn.close()
    except Exception as e:
        print(f"Error listing sleep sessions: {str(e)}")
    
    return sessions

def get_session_details(db_path: str, session_id: int) -> Dict:
    """
    Get detailed information about a sleep session.
    
    Args:
        db_path: Path to the SQLite database
        session_id: ID of the session to retrieve
        
    Returns:
        Dictionary with session details and logs
    """
    details = {
        "session": None,
        "logs": [],
        "success": False,
        "error": None
    }
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Query session information
        cursor.execute("""
            SELECT id, start_time, end_time, conversation_source, status, extraction_count, consolidation_count 
            FROM sleep_sessions 
            WHERE id = ?
        """, (session_id,))
        
        row = cursor.fetchone()
        if not row:
            details["error"] = f"Session {session_id} not found"
            return details
            
        session_id, start_time, end_time, source, status, extraction_count, consolidation_count = row
        
        # Calculate duration if available
        duration = None
        if start_time and end_time:
            try:
                start_dt = datetime.datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                end_dt = datetime.datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                duration = (end_dt - start_dt).total_seconds()
            except:
                pass
        
        details["session"] = {
            "id": session_id,
            "start_time": start_time,
            "end_time": end_time,
            "source": source,
            "status": status,
            "extraction_count": extraction_count,
            "consolidation_count": consolidation_count,
            "duration": duration
        }
        
        # Query session logs
        cursor.execute("""
            SELECT id, timestamp, event_type, details 
            FROM sleep_logs 
            WHERE session_id = ? 
            ORDER BY timestamp ASC
        """, (session_id,))
        
        for log_row in cursor.fetchall():
            log_id, timestamp, event_type, log_details = log_row
            details["logs"].append({
                "id": log_id,
                "timestamp": timestamp,
                "event_type": event_type,
                "details": log_details
            })
        
        details["success"] = True
        conn.close()
    except Exception as e:
        details["error"] = str(e)
    
    return details

def forget(query: str, 
           db_path: str, 
           chroma_path: str,
           source: Optional[str] = None,
           verbose: bool = False) -> Dict:
    """
    Forget/remove specific memories from the agent's memory based on query.
    
    Args:
        query: The query to identify memories to forget
        db_path: Path to the SQLite database
        chroma_path: Path to the ChromaDB directory
        source: Optional source filter (e.g., agent name)
        verbose: Whether to print verbose output
        
    Returns:
        Dict with deletion results
    """
    results = {
        "success": False,
        "removed": {
            "facts": 0,
            "mistakes": 0,
            "lessons": 0,
            "actions": 0,
            "decisions": 0,
            "vector_items": 0
        },
        "errors": []
    }
    
    try:
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        if verbose:
            print(f"🗑️ Forgetting memories matching: '{query}'")
            
        # Build source filter if provided
        source_filter = "AND source = ?" if source else ""
        params = (f"%{query}%", source) if source else (f"%{query}%",)
        
        # Delete facts
        cursor.execute(
            f"DELETE FROM facts WHERE content LIKE ? {source_filter}",
            params
        )
        results["removed"]["facts"] = cursor.rowcount
        
        # Delete mistakes
        cursor.execute(
            f"DELETE FROM mistakes WHERE content LIKE ? {source_filter}",
            params
        )
        results["removed"]["mistakes"] = cursor.rowcount
        
        # Delete lessons
        cursor.execute(
            f"DELETE FROM lessons WHERE content LIKE ? {source_filter}",
            params
        )
        results["removed"]["lessons"] = cursor.rowcount
        
        # Delete actions
        cursor.execute(
            f"DELETE FROM actions WHERE content LIKE ? {source_filter}",
            params
        )
        results["removed"]["actions"] = cursor.rowcount
        
        # Delete decisions
        cursor.execute(
            f"DELETE FROM decisions WHERE content LIKE ? {source_filter}",
            params
        )
        results["removed"]["decisions"] = cursor.rowcount
        
        conn.commit()
        
        # Delete from vector database
        try:
            from npcpy.memory.knowledge_graph import remove_from_vector_db
            vector_count = remove_from_vector_db(chroma_path, query, source)
            results["removed"]["vector_items"] = vector_count
        except ImportError:
            results["errors"].append("Vector database removal function not available")
            
        # Calculate total removed items
        total_removed = sum(results["removed"].values())
        
        if verbose:
            print(f"✅ Removed {total_removed} memories matching the query.")
            print(f"  - Facts: {results['removed']['facts']}")
            print(f"  - Mistakes: {results['removed']['mistakes']}")
            print(f"  - Lessons: {results['removed']['lessons']}")
            print(f"  - Actions: {results['removed']['actions']}")
            print(f"  - Decisions: {results['removed']['decisions']}")
            print(f"  - Vector items: {results['removed']['vector_items']}")
            
        results["success"] = True
    except Exception as e:
        results["errors"].append(str(e))
        if verbose:
            print(f"❌ Error during forget operation: {str(e)}")
    finally:
        # Close database connection
        if 'conn' in locals():
            conn.close()
            
    return results