
# pti
import json 
from typing import Dict, List, Optional, Any, Generator 
import os 
from npcpy.memory.command_history import CommandHistory, save_attachment_to_message, start_new_conversation,save_conversation_message
from npcpy.npc_sysenv import (NPCSH_REASONING_MODEL,
                              NPCSH_REASONING_PROVIDER, 
                              NPCSH_CHAT_MODEL, 
                              NPCSH_CHAT_PROVIDER, 
                              NPCSH_API_URL, 
                              NPCSH_STREAM_OUTPUT,print_and_process_stream_with_markdown)
from npcpy.llm_funcs import get_llm_response, handle_request_input

from npcpy.npc_compiler import NPC 
from npcpy.data.load import load_csv, load_pdf
from npcpy.data.text import rag_search






def enter_reasoning_human_in_the_loop(
    user_input=None,
    messages: List[Dict[str, str]] = None,
    reasoning_model: str = NPCSH_REASONING_MODEL,
    reasoning_provider: str = NPCSH_REASONING_PROVIDER,
    files : List = None, 
    npc: Any = None,
    conversation_id : str= False,
    answer_only: bool = False,
    context=None,
) :
    """
    Stream responses while checking for think tokens and handling human input when needed.

    Args:
        messages: List of conversation messages
        model: LLM model to use
        provider: Model provider
        npc: NPC instance if applicable

    """
    # Get the initial stream
    loaded_content = {}  # New dictionary to hold loaded content

    # Create conversation ID if not provided
    if not conversation_id:
        conversation_id = start_new_conversation()

    command_history = CommandHistory()
    # Load specified files if any
    if files:
        for file in files:
            extension = os.path.splitext(file)[1].lower()
            try:
                if extension == ".pdf":
                    content = load_pdf(file)["texts"].iloc[0]
                elif extension == ".csv":
                    content = load_csv(file)
                else:
                    print(f"Unsupported file type: {file}")
                    continue
                loaded_content[file] = content
                print(f"Loaded content from: {file}")
            except Exception as e:
                print(f"Error loading {file}: {str(e)}")


    try:
        while True:

            if loaded_content:
                context_content = ""
                for filename, content in loaded_content.items():
                    retrieved_docs = rag_search(
                        user_input,
                        content,
                    )
                    if retrieved_docs:
                        context_content += (
                            f"\n\nLoaded content from: {filename}\n{content}\n\n"
                        )
                if len(context_content) > 0:
                    user_input += f"""
                    Here is the loaded content that may be relevant to your query:
                        {context_content}
                    Please reference it explicitly in your response and use it for answering.
                    """
            if answer_only:
                response = get_llm_response(
                user_input, 
                model = reasoning_model, 
                provider=reasoning_provider, 
                messages=messages, 
                stream=True,
                )
                assistant_reply, messages = response['response'], response['messages']
                assistant_reply = print_and_process_stream_with_markdown(assistant_reply, reasoning_model, reasoning_provider)
                messages.append({'role':'assistant', 'content':assistant_reply})
                return enter_reasoning_human_in_the_loop(user_input = None, 
                                                         messages=messages, 
                                                         reasoning_model=reasoning_model, 
                                                         reasoning_provider=reasoning_provider, answer_only=False)
            else:
                message= "Think first though and use <think> tags in your chain of thought. Once finished, either answer plainly or write a request for input by beginning with the <request_for_input> tag. and close it with a </request_for_input>"
                if user_input is None:
                    user_input = input('🐻‍❄️>')
                
                message_id = save_conversation_message(
                    command_history,
                    conversation_id,
                    "user",
                    user_input,
                    wd=os.getcwd(),
                    model=reasoning_model,
                    provider=reasoning_provider,
                    npc=npc.name if npc else None,
                    
                )                    
                response = get_llm_response(
                    user_input+message, 
                    model = reasoning_model, 
                    provider=reasoning_provider, 
                    messages=messages, 
                    stream=True,
                )

                assistant_reply, messages = response['response'], response['messages']
                thoughts = []
                response_chunks = []
                in_think_block = False # the thinking chain generated after reasoning
                
                thinking = False # the reasoning content 
                

                for chunk in assistant_reply:       
                    if thinking:
                        if not in_think_block:
                            in_think_block = True
                    try:
                        
                        if reasoning_provider == "ollama":
                            chunk_content = chunk.get("message", {}).get("content", "")
                        else:
                            chunk_content = ''
                            reasoning_content = ''
                            for c in chunk.choices:
                                if hasattr(c.delta, "reasoning_content"):
                                    
                                    reasoning_content += c.delta.reasoning_content
                                    
                            if reasoning_content:
                                thinking = True
                                chunk_content = reasoning_content
                            chunk_content += "".join(
                                choice.delta.content
                                for choice in chunk.choices
                                if choice.delta.content is not None
                            )
                        response_chunks.append(chunk_content)
                        print(chunk_content, end='')
                        combined_text = "".join(response_chunks)

                        if in_think_block:
                            if '</thinking>' in combined_text:
                                in_think_block = False
                            thoughts.append(chunk_content)
                            
                        if "</request_for_input>" in combined_text:
                            # Process the LLM's input request
                            request_text = "".join(thoughts)

                            print("\nPlease provide the requested information: ")

                            user_input = input('🐻‍❄️>')

                            messages.append({"role": "assistant", "content": request_text})

                            print("\n[Continuing with provided information...]\n")
                            return enter_reasoning_human_in_the_loop( user_input = user_input,
                                                                     messages=messages, 
                                                                     reasoning_model=reasoning_model,
                                                                     reasoning_provider=reasoning_provider,
                                                                     npc=npc, 
                                                                     answer_only=True)
                            
                        
                    except KeyboardInterrupt:        
                        user_interrupt = input("\n[Stream interrupted by user]\n Enter your additional input: ")
                        

                        # Add the interruption to messages and restart stream
                        messages.append(
                            {"role": "user", "content": f"[INTERRUPT] {user_interrupt}"}
                        )
                        print(f"\n[Continuing with added context...]\n")        
                        
    except KeyboardInterrupt:
        user_interrupt = input("\n[Stream interrupted by user]\n 🔴🔴🔴🔴\nEnter your additional input: ")
        

        # Add the interruption to messages and restart stream
        messages.append(
            {"role": "user", "content": f"[INTERRUPT] {user_interrupt}"}
        )
        print(f"\n[Continuing with added context...]\n")        
        
    return {'messages':messages, }
        

def main():
    # Example usage
    import argparse    
    parser = argparse.ArgumentParser(description="Enter PTI mode for chatting with an LLM")
    parser.add_argument("--npc", default='~/.npcsh/npc_team/frederic.npc', help="Path to NPC File")    
    parser.add_argument("--model", default=NPCSH_REASONING_MODEL, help="Model to use")
    parser.add_argument("--provider", default=NPCSH_REASONING_PROVIDER, help="Provider to use")
    parser.add_argument("--files", nargs="*", help="Files to load into context")
    args = parser.parse_args()
    
    npc = NPC(file=args.npc)
    enter_reasoning_human_in_the_loop(
        messages = [],
        npc=npc,
        reasoning_model=args.model,
        reasoning_provider=args.provider,
        files=args.files,
    )

if __name__ == "__main__":
    main()
    
