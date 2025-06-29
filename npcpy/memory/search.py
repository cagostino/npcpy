from npcpy.data.load import load_file_contents
from npcpy.data.web import search_web
from npcpy.memory.command_history import setup_chroma_db
from npcpy.gen.embeddings import get_ollama_embeddings
from npcpy.llm_funcs import get_llm_response
from npcpy.npc_sysenv import render_markdown
from typing import Any, Dict, List, Optional, Union 
import numpy as np
from datetime import datetime
import traceback
try:
    import chromadb
except ImportError:
    chromadb = None
except Exception as e: 
    print(f"An error occurred: {e}")
    chromadb = None
    

def search_similar_texts(
    query: str,
    embedding_model: str,
    embedding_provider: str,    
    chroma_client = None,
    
    docs_to_embed: Optional[List[str]] = None,
    top_k: int = 15,
) -> List[Dict[str, any]]:
    """
    Search for similar texts using either a Chroma database or direct embedding comparison.
    With duplicate filtering.
    """

    print(f"\nQuery to embed: {query}")
    embedded_search_term = get_ollama_embeddings([query], embedding_model)[0]

    if docs_to_embed is None:
        # Fetch from the database if no documents to embed are provided
        collection_name = f"{embedding_provider}_{embedding_model}_embeddings"
        collection = chroma_client.get_collection(collection_name)
        results = collection.query(
            query_embeddings=[embedded_search_term], n_results=top_k * 2  # Fetch more to account for filtering
        )
        
        # Filter out duplicates while preserving order
        seen_texts = set()
        filtered_results = []
        
        for idx, (id, distance, document) in enumerate(zip(
            results["ids"][0], results["distances"][0], results["documents"][0]
        )):
            # Check if this is a command (starts with /) and if we've seen it before
            if document not in seen_texts:
                seen_texts.add(document)
                filtered_results.append({
                    "id": id, 
                    "score": float(distance), 
                    "text": document
                })
                
                # Break if we have enough unique results
                if len(filtered_results) >= top_k:
                    break
                    
        return filtered_results

    print(f"\nNumber of documents to embed: {len(docs_to_embed)}")

    # Get embeddings for provided documents - use np.unique to remove duplicates
    unique_docs = list(dict.fromkeys(docs_to_embed))  # Preserves order while removing duplicates
    raw_embeddings = get_ollama_embeddings(unique_docs, embedding_model)

    output_embeddings = []
    unique_doc_indices = []
    
    for idx, emb in enumerate(raw_embeddings):
        if emb:  # Exclude any empty embeddings
            output_embeddings.append(emb)
            unique_doc_indices.append(idx)

    # Convert to numpy arrays for calculations
    doc_embeddings = np.array(output_embeddings)
    query_embedding = np.array(embedded_search_term)

    # Check for zero-length embeddings
    if len(doc_embeddings) == 0:
        raise ValueError("No valid document embeddings found")

    # Normalize embeddings to avoid division by zeros
    doc_norms = np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    query_norm = np.linalg.norm(query_embedding)

    # Ensure no zero vectors are being used in cosine similarity
    if query_norm == 0:
        raise ValueError("Query embedding is zero-length")

    # Calculate cosine similarities
    cosine_similarities = np.dot(doc_embeddings, query_embedding) / (
        doc_norms.flatten() * query_norm
    )

    # Get indices of top K documents
    top_indices = np.argsort(cosine_similarities)[::-1][:top_k]

    return [
        {
            "id": str(unique_doc_indices[idx]),
            "score": float(cosine_similarities[idx]),
            "text": unique_docs[unique_doc_indices[idx]],
        }
        for idx in top_indices
    ]
def execute_search_command(
    command: str,
    messages=None,
    provider: str = None,
):
    """
    Function Description:

    Args:
        command : str : Command
        db_path : str : Database path

    Keyword Args:
        embedding_model : None : Embedding model
        current_npc : None : Current NPC
        text_data : None : Text data
        text_data_embedded : None : Embedded text data
        messages : None : Messages
    Returns:
        dict : dict : Dictionary

    """

    search_command = command.split()
    if any("-p" in s for s in search_command) or any(
        "--provider" in s for s in search_command
    ):
        provider = (
            search_command[search_command.index("-p") + 1]
            if "-p" in search_command
            else search_command[search_command.index("--provider") + 1]
        )
    else:
        provider = None
    if any("-n" in s for s in search_command) or any(
        "--num_results" in s for s in search_command
    ):
        num_results = (
            search_command[search_command.index("-n") + 1]
            if "-n" in search_command
            else search_command[search_command.index("--num_results") + 1]
        )
    else:
        num_results = 5

    # remove the -p and provider from the command string
    command = command.replace(f"-p {provider}", "").replace(
        f"--provider {provider}", ""
    )
    result = search_web(command, num_results=num_results, provider=provider)
    if messages is None:
        messages = []
        messages.append({"role": "user", "content": command})

    messages.append(
        {"role": "assistant", "content": result[0] + f" \n Citation Links: {result[1]}"}
    )

    return {
        "messages": messages,
        "output": result[0] + f"\n\n\n Citation Links: {result[1]}",
    }
def execute_rag_command(
    command: str,
    vector_db_path: str, 
    embedding_model: str,
    embedding_provider: str,
    messages=None,
    top_k: int = 15,
    file_contents=None,  # List of file content chunks
    **kwargs
) -> dict:
    """
    Execute the RAG command with support for embedding generation.
    When file_contents is provided, it searches those instead of the database.
    """
    # ANSI color codes for terminal output
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    # Format header
    header = f"\n{BOLD}{BLUE}RAG Query: {RESET}{GREEN}{command}{RESET}\n"
    
    # If we have file contents, search those instead of the database
    if file_contents and len(file_contents) > 0:
        similar_chunks = search_similar_texts(
            command,
            embedding_model,
            embedding_provider,
            chroma_client=None,  # Not using ChromaDB when searching files
            
            docs_to_embed=file_contents,  # Pass file chunks directly to embed
            top_k=top_k
        )
        
        # Process the results for display
        file_info = f"{BOLD}{BLUE}Files Processed: {RESET}{YELLOW}{len(file_contents)}{RESET}\n"
        separator = f"{YELLOW}{'-' * 100}{RESET}\n"
        
        # Format chunk results for display
        chunk_results = []
        for i, chunk in enumerate(similar_chunks, 1):
            score = chunk['score']
            text = chunk['text']
            
            # Truncate for display if needed
            display_text = text[:150] + ("..." if len(text) > 150 else "")
            chunk_results.append(f"{BOLD}{i:2d}{RESET}. {CYAN}[{score:.2f}]{RESET} {display_text}")
        
        # Display the file search results
        file_results = header + file_info + separator + "\n".join(chunk_results)
        render_markdown(f"FILE SEARCH RESULTS:\n{file_results}")
        
        # Prepare the chunks for the prompt (plain text version)
        plain_chunks = [f"{i+1}. {chunk['text']}" for i, chunk in enumerate(similar_chunks)]
        plain_results = "\n\n".join(plain_chunks)
        
        # Build the prompt focusing on file contents
        prompt = f"""
        The user asked: {command}
        
        Here are the most relevant sections from the file(s):
        
        {plain_results}
        
        Please respond to the user query based on these file contents.
        """
        
        # Get LLM response
        response = get_llm_response(
            prompt,
            messages=messages,
            **kwargs
        )
        return response
    
    else:
        # No file contents, search the database instead
        try:
            # Setup ChromaDB connection
            chroma_client, chroma_collection = setup_chroma_db( 
                f"{embedding_provider}_{embedding_model}_embeddings",
                "Conversation embeddings",
                vector_db_path
            )
            
            # Search for similar texts in the database
            similar_texts = search_similar_texts(
                command, 
                embedding_model,
                embedding_provider,
                chroma_client=chroma_client,
                top_k=top_k,
            )
            
            # Process the results for display
            separator = f"{YELLOW}{'-' * 100}{RESET}\n"
            
            # Format results
            processed_texts = []
            for i, similar_text in enumerate(similar_texts, 1):
                text = similar_text['text']
                score = similar_text['score']
                
                # Format timestamp if available
                timestamp_str = ""
                try:
                    if 'id' in similar_text and '_' in similar_text['id']:
                        timestamp = datetime.fromisoformat(similar_text['id'].split('_')[1])
                        timestamp_str = f"{YELLOW}({timestamp.strftime('%Y-%m-%d')}){RESET}"
                except (IndexError, ValueError, TypeError):
                    pass
                
                # Clean up the text
                text = text.replace('\n', ' ').strip()
                snippet = text[:85] + ("..." if len(text) > 85 else "")
                
                # Format with colors
                processed_texts.append(
                    f"{BOLD}{i:2d}{RESET}. {CYAN}[{score:.2f}]{RESET} {snippet} {timestamp_str}"
                )
            
            # Combine for display
            knowledge_results = header + separator + "\n".join(processed_texts)
            render_markdown(f"KNOWLEDGE BASE: {knowledge_results}")
            
            # Prepare plain text for the prompt
            plain_texts = [f"{i+1}. {similar_texts[i]['text']}" for i in range(len(similar_texts))]
            plain_results = "\n\n".join(plain_texts)
            
            # Build the prompt
            prompt = f"""
            The user asked: {command}
            
            Here are the most similar texts found in the knowledge base:
            
            {plain_results}
            
            Please respond to the user query based on the above information.
            """
            
            # Get LLM response
            response = get_llm_response(
                prompt,
                messages=messages,
                **kwargs
            )
            return response
            
        except Exception as e:
            traceback.print_exc()
            return {"output": f"Error searching knowledge base: {e}", "messages": messages}
        
        
def execute_brainblast_command(
    command: str,
    command_history,
    messages=None,
    top_k: int = 5,  # Fewer results per chunk to keep total manageable
    **kwargs
) -> dict:
    """
    Execute a comprehensive "brainblast" search on command history.
    Breaks the query into words and searches for combinations of those words.
    """
    # ANSI color codes for terminal output
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    print(f"\nBrainblast Query: {command}")
    
    # Format header for display
    header = f"\n{BOLD}{BLUE}BRAINBLAST Query: {RESET}{GREEN}{command}{RESET}\n"
    separator = f"{YELLOW}{'-' * 100}{RESET}\n"
    
    try:
        # Split the command into words
        words = command.split()
        
        if not words:
            return {"output": "Please provide search terms to use brainblast.", "messages": messages or []}
        
        # Generate different chunk sizes for searching
        all_chunks = []
        
        # Add individual words
        all_chunks.extend(words)
        
        # Add pairs of words
        if len(words) >= 2:
            for i in range(len(words) - 1):
                all_chunks.append(f"{words[i]} {words[i+1]}")
        
        # Add groups of 4 words
        if len(words) >= 4:
            for i in range(len(words) - 3):
                all_chunks.append(f"{words[i]} {words[i+1]} {words[i+2]} {words[i+3]}")
        
        # Add the entire query
        if len(words) > 1:
            all_chunks.append(command)
        
        # Remove duplicates while preserving order
        unique_chunks = []
        for chunk in all_chunks:
            if chunk not in unique_chunks:
                unique_chunks.append(chunk)
        
        # Search for each chunk
        all_results = []
        chunk_results = {}
        
        for chunk in unique_chunks:
            results = command_history.search_commands(chunk)
            if results:
                chunk_results[chunk] = results[:top_k]  # Limit results per chunk
                all_results.extend(results[:top_k])
        
        # Remove duplicate results while preserving order
        unique_results = []
        seen_ids = set()
        for result in all_results:
            result_id = result.get('id')
            if result_id not in seen_ids:
                seen_ids.add(result_id)
                unique_results.append(result)
        
        if not unique_results:
            result_message = f"No matches found for any combination of terms in: {command}"
            render_markdown(f"BRAINBLAST SEARCH: {header}{separator}{result_message}")
            
            # Get LLM response
            prompt = f"""
            The user asked for a brainblast search with: {command}
            
            No matching commands were found in the command history for any combination 
            of these search terms. Please let the user know and suggest they try different terms.
            """
            
            response = get_llm_response(
                prompt,
                messages=messages,
                **kwargs
            )
            return response
        
        # Process the results for display
        processed_chunks = []
        for chunk, results in chunk_results.items():
            if results:
                chunk_display = f"{BOLD}{BLUE}Results for '{chunk}':{RESET}\n"
                
                for i, result in enumerate(results[:3], 1):  # Just show top 3 for each chunk
                    cmd = result.get('command', '')
                    timestamp = result.get('timestamp', '')
                    
                    chunk_display += f"  {i}. {CYAN}[{timestamp}]{RESET} {cmd}\n"
                
                if len(results) > 3:
                    chunk_display += f"  {YELLOW}...and {len(results) - 3} more results{RESET}\n"
                
                processed_chunks.append(chunk_display)
        
        # Display summarized chunk results
        chunks_output = header + separator + "\n".join(processed_chunks)
        render_markdown(f"BRAINBLAST SEARCH: {chunks_output}")
        
        # Prepare the consolidated results for the prompt
        plain_results = []
        for i, result in enumerate(unique_results[:15], 1):  # Limit to 15 total unique results
            cmd = result.get('command', '')
            output = result.get('output', '')
            timestamp = result.get('timestamp', '')
            location = result.get('location', '')
            
            # Format without ANSI colors
            plain_results.append(
                f"{i}. [{timestamp}] Command: {cmd}\n   Location: {location}\n   Output: {output[:150] + ('...' if len(output) > 150 else '')}"
            )
        
        # Summary of which terms matched what
        term_summary = []
        for chunk, results in chunk_results.items():
            if results:
                term_summary.append(f"Term '{chunk}' matched {len(results)} commands")
        
        # Build the prompt
        f=', '.join(term_summary)
        e="\n\n".join(plain_results)
        prompt = f"""
        The user asked for a brainblast search with: {command}
        
        I searched for individual words, pairs of words, groups of 4 words, and the full query.
        Here's a summary of what matched:
        {f}
        
        Here are the top unique matching commands from the search:
        
        {e}
        
        Please analyze these results and provide insights to the user. Look for:
        1. Patterns in the commands they've used
        2. Common themes or topics
        3. Any interesting or unusual commands that might be useful
        4. Suggestions for how the user might better leverage their command history
        
        Focus on being helpful and surfacing valuable insights from their command history.
        """
        
        # Get LLM response
        response = get_llm_response(
            prompt,
            messages=messages,
            **kwargs
        )
        return response
        
    except Exception as e:
        traceback.print_exc()
        return {"output": f"Error executing brainblast command: {e}", "messages": messages or []}