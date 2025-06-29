
try:
    from faster_whisper import WhisperModel
    from gtts import gTTS
    import torch
    import pyaudio
    import wave
    import queue

    from npcpy.data.audio import (
        cleanup_temp_files,
        FORMAT,
        CHANNELS,
        RATE,
        CHUNK,
        transcribe_recording,
        convert_mp3_to_wav,
    )
    import threading
    import tempfile
    import os
    import re
    import time
    import numpy as np
    

except Exception as e:
    print(
        "Exception: "
        + str(e)
        + "\n"
        + "Could not load the whisper package. If you want to use tts/stt features, please run `pip install npcsh[audio]` and follow the instructions in the npcsh github readme to  ensure your OS can handle the audio dependencies."
    )
from npcpy.data.load import load_csv, load_pdf
from npcpy.npc_sysenv import (
    NPCSH_CHAT_MODEL,
    NPCSH_CHAT_PROVIDER,
    NPCSH_DB_PATH,
    NPCSH_API_URL,
    NPCSH_STREAM_OUTPUT,
    get_system_message,
    print_and_process_stream_with_markdown, 
    render_markdown
    

)
from sqlalchemy import create_engine
from npcpy.llm_funcs import check_llm_command
from npcpy.data.text import rag_search
from npcpy.npc_compiler import (
    NPC, Team
)
from npcpy.memory.command_history import CommandHistory, save_conversation_message,start_new_conversation
from typing import Dict, Any, List
def enter_yap_mode(
    
    model: str ,
    provider: str ,
    messages: list = None,
    npc = None,    
    team=  None,
    tts_model="kokoro",
    voice="af_heart", 
    files: List[str] = None,
    rag_similarity_threshold: float = 0.3,
    stream: bool = NPCSH_STREAM_OUTPUT, 
    conversation_id = None,
) -> Dict[str, Any]:
    running = True
    is_recording = False
    recording_data = []
    buffer_data = []
    last_speech_time = 0
    vad_model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
        verbose=False,
        
    )
    device = 'cpu'
    vad_model.to(device)
    

    print("Entering yap mode. Initializing...")

    concise_instruction = "Please provide brief responses of 1-2 sentences unless the user specifically asks for more detailed information. Keep responses clear and concise."

    provider = (
        NPCSH_CHAT_PROVIDER if npc is None else npc.provider or NPCSH_CHAT_PROVIDER
    )
    api_url = NPCSH_API_URL if npc is None else npc.api_url or NPCSH_API_URL

    print(f"\nUsing model: {model} with provider: {provider}")

    system_message = get_system_message(npc) if npc else "You are a helpful assistant."

    # Add conciseness instruction to the system message
    system_message = system_message + " " + concise_instruction

    if messages is None:
        messages = [{"role": "system", "content": system_message}]
    elif messages is not None and messages[0]['role'] != 'system':
        messages.insert(0, {"role": "system", "content": system_message})

    kokoro_pipeline = None
    if tts_model == "kokoro":
        try:
            from kokoro import KPipeline
            import soundfile as sf

            kokoro_pipeline = KPipeline(lang_code="a")
            print("Kokoro TTS model initialized")
        except ImportError:
            print("Kokoro not installed, falling back to gTTS")
            tts_model = "gtts"

    # Initialize PyAudio
    pyaudio_instance = pyaudio.PyAudio()
    audio_stream = None  # We'll open and close as needed
    transcription_queue = queue.Queue()

    # Create and properly use the is_speaking event
    is_speaking = threading.Event()
    is_speaking.clear()  # Not speaking initially

    speech_queue = queue.Queue(maxsize=20)
    speech_thread_active = threading.Event()
    speech_thread_active.set()

    def speech_playback_thread():
        nonlocal running, audio_stream

        while running and speech_thread_active.is_set():
            try:
                # Get next speech item from queue
                if not speech_queue.empty():
                    text_to_speak = speech_queue.get(timeout=0.1)

                    # Only process if there's text to speak
                    if text_to_speak.strip():
                        # IMPORTANT: Set is_speaking flag BEFORE starting audio output
                        is_speaking.set()

                        # Safely close the audio input stream before speaking
                        current_audio_stream = audio_stream
                        audio_stream = (
                            None  # Set to None to prevent capture thread from using it
                        )

                        if current_audio_stream and current_audio_stream.is_active():
                            current_audio_stream.stop_stream()
                            current_audio_stream.close()

                        print(f"Speaking full response...")

                        # Generate and play speech
                        generate_and_play_speech(text_to_speak)

                        # Delay after speech to prevent echo
                        time.sleep(0.005 * len(text_to_speak))
                        print(len(text_to_speak))

                        # Clear the speaking flag to allow listening again
                        is_speaking.clear()
                else:
                    time.sleep(0.5)
            except Exception as e:
                print(f"Error in speech thread: {e}")
                is_speaking.clear()  # Make sure to clear the flag if there's an error
                time.sleep(0.1)

    def safely_close_audio_stream(stream):
        """Safely close an audio stream with error handling"""
        if stream:
            try:
                if stream.is_active():
                    stream.stop_stream()
                stream.close()
            except Exception as e:
                print(f"Error closing audio stream: {e}")

    # Start speech thread
    speech_thread = threading.Thread(target=speech_playback_thread)
    speech_thread.daemon = True
    speech_thread.start()

    def generate_and_play_speech(text):
        try:
            # Create a temporary file for audio
            unique_id = str(time.time()).replace(".", "")
            temp_dir = tempfile.gettempdir()
            wav_file = os.path.join(temp_dir, f"temp_{unique_id}.wav")

            # Generate speech based on selected TTS model
            if tts_model == "kokoro" and kokoro_pipeline:
                # Use Kokoro for generation
                generator = kokoro_pipeline(text, voice=voice)

                # Get the audio from the generator
                for _, _, audio in generator:
                    # Save audio to WAV file
                    import soundfile as sf

                    sf.write(wav_file, audio, 24000)
                    break  # Just use the first chunk for now
            else:
                # Fall back to gTTS
                mp3_file = os.path.join(temp_dir, f"temp_{unique_id}.mp3")
                tts = gTTS(text=text, lang="en", slow=False)
                tts.save(mp3_file)
                convert_mp3_to_wav(mp3_file, wav_file)

            # Play the audio
            wf = wave.open(wav_file, "rb")
            p = pyaudio.PyAudio()

            stream = p.open(
                format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True,
            )

            data = wf.readframes(4096)
            while data and running:
                stream.write(data)
                data = wf.readframes(4096)

            stream.stop_stream()
            stream.close()
            p.terminate()

            # Cleanup temp files
            try:
                if os.path.exists(wav_file):
                    os.remove(wav_file)
                if tts_model == "gtts" and "mp3_file" in locals():
                    if os.path.exists(mp3_file):
                        os.remove(mp3_file)
            except Exception as e:
                print(f"Error removing temp file: {e}")

        except Exception as e:
            print(f"Error in TTS process: {e}")

    # Modified speak_text function that just queues text
    def speak_text(text):
        speech_queue.put(text)

    def process_input(user_input, messages):
        #try:
        full_response = ""

        # Use get_stream for streaming response
        check = check_llm_command(
            user_input,
            npc=npc,
            team=team,
            messages=messages,
            model=model,
            provider=provider,
            stream=False,
        )
        #mport pdb 
        #pdb.set_trace()
        assistant_reply = check["output"]
        messages = check['messages']
        #print(messages)
        #import pdb 
        #pdb.set_trace()
        if stream and not isinstance(assistant_reply,str) and not isinstance(assistant_reply, dict):
            assistant_reply = print_and_process_stream_with_markdown(assistant_reply, model, provider)
        elif isinstance(assistant_reply,dict):
            # assume its a jinx output, to fix later
            assistant_reply = assistant_reply.get('output')
            render_markdown(assistant_reply)
        full_response += assistant_reply

        print("\n")  # End the progress display

        # Process and speak the entire response at once
        if full_response.strip():
            processed_text = process_text_for_tts(full_response)
            speak_text(processed_text)

        # Add assistant's response to messages
        messages.append({"role": "assistant", "content": full_response})
        return messages 
        #except Exception as e:
        #    print(f"Error in LLM response: {e}")
        #    speak_text("I'm sorry, there was an error processing your request.")

    # Function to capture and process audio


    def capture_audio():
        nonlocal is_recording, recording_data, buffer_data, last_speech_time, running, is_speaking
        nonlocal audio_stream, transcription_queue

        # Don't try to record if we're speaking
        if is_speaking.is_set():
            return False

        try:
            # Only create a new audio stream if we don't have one
            if audio_stream is None and not is_speaking.is_set():
                audio_stream = pyaudio_instance.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                )

            # Initialize or reset the recording variables
            is_recording = False
            recording_data = []
            buffer_data = []

            print("\nListening for speech...")

            while (
                running
                and audio_stream
                and audio_stream.is_active()
                and not is_speaking.is_set()
            ):
                try:
                    data = audio_stream.read(CHUNK, exception_on_overflow=False)
                    if data:
                        audio_array = np.frombuffer(data, dtype=np.int16)
                        audio_float = audio_array.astype(np.float32) / 32768.0

                        tensor = torch.from_numpy(audio_float).to(device)
                        speech_prob = vad_model(tensor, RATE).item()
                        current_time = time.time()

                        if speech_prob > 0.5:  # VAD threshold
                            last_speech_time = current_time
                            if not is_recording:
                                is_recording = True
                                print("\nSpeech detected, listening...")
                                recording_data.extend(buffer_data)
                                buffer_data = []
                            recording_data.append(data)
                        else:
                            if is_recording:
                                if (
                                    current_time - last_speech_time > 1
                                ):  # silence duration
                                    is_recording = False
                                    print("Speech ended, transcribing...")

                                    # Stop stream before transcribing
                                    safely_close_audio_stream(audio_stream)
                                    audio_stream = None

                                    # Transcribe in this thread to avoid race conditions
                                    transcription = transcribe_recording(recording_data)
                                    if transcription:
                                        transcription_queue.put(transcription)
                                    recording_data = []
                                    return True  # Got speech
                            else:
                                buffer_data.append(data)
                                if len(buffer_data) > int(
                                    0.65 * RATE / CHUNK
                                ):  # buffer duration
                                    buffer_data.pop(0)

                    # Check frequently if we need to stop capturing
                    if is_speaking.is_set():
                        safely_close_audio_stream(audio_stream)
                        audio_stream = None
                        return False

                except Exception as e:
                    print(f"Error processing audio frame: {e}")
                    time.sleep(0.1)

        except Exception as e:
            print(f"Error in audio capture: {e}")

        # Close stream if we exit without finding speech
        safely_close_audio_stream(audio_stream)
        audio_stream = None

        return False

    def process_text_for_tts(text):
        # Remove special characters that might cause issues in TTS
        text = re.sub(r"[*<>{}()\[\]&%#@^_=+~]", "", text)
        text = text.strip()
        # Add spaces after periods that are followed by words (for better pronunciation)
        text = re.sub(r"(\w)\.(\w)\.", r"\1 \2 ", text)
        text = re.sub(r"([.!?])(\w)", r"\1 \2", text)
        return text

    # Now that functions are defined, play welcome messages
    speak_text("Entering yap mode. Please wait.")

    try:
        loaded_content = {}  # New dictionary to hold loaded content
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



        while running:

            # First check for typed input (non-blocking)
            import select
            import sys

            # Don't spam the console with prompts when speaking
            if not is_speaking.is_set():
                print(
                    "🎤🎤🎤🎤\n Speak or type your message (or 'exit' to quit): ",
                    end="",
                    flush=True,
                )

            rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
            if rlist:
                user_input = sys.stdin.readline().strip()
                if user_input.lower() in ("exit", "quit", "goodbye"):
                    print("\nExiting yap mode.")
                    break
                if user_input:
                    print(f"\nYou (typed): {user_input}")
                    # Handle RAG context
                    if loaded_content:
                        context_content = ""
                        for filename, content in loaded_content.items():
                            retrieved_docs = rag_search(
                                user_input,
                                content,
                                similarity_threshold=rag_similarity_threshold,
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
                    message_id = save_conversation_message(
                        command_history,
                        conversation_id,
                        "user",
                        user_input,
                        wd=os.getcwd(),
                        model=model,
                        provider=provider,
                        npc=npc.name if npc else None,
                    )

                            
                    messages= process_input(user_input, messages)

                    message_id = save_conversation_message(
                        command_history,
                        conversation_id,
                        "assistant",
                        messages[-1]["content"],
                        wd=os.getcwd(),
                        model=model,
                        provider=provider,
                        npc=npc.name if npc else None,
                    )


                    continue  # Skip audio capture this cycle

            # Then try to capture some audio (if no typed input)
            if not is_speaking.is_set():  # Only capture if not currently speaking
                got_speech = capture_audio()

                # If we got speech, process it
                if got_speech:
                    try:
                        transcription = transcription_queue.get_nowait()
                        print(f"\nYou (spoke): {transcription}")
                        messages = process_input(transcription, messages)
                    except queue.Empty:
                        pass
            else:
                # If we're speaking, just wait a bit without spamming the console
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        # Set running to False to signal threads to exit
        running = False
        speech_thread_active.clear()

        # Clean up audio resources
        safely_close_audio_stream(audio_stream)

        if pyaudio_instance:
            pyaudio_instance.terminate()

        print("\nExiting yap mode.")
        speak_text("Exiting yap mode. Goodbye!")
        time.sleep(1)
        cleanup_temp_files()

    return {"messages": messages, "output": "yap mode session ended."}

def main():
    # Example usage
    import argparse    
    parser = argparse.ArgumentParser(description="Enter yap mode for chatting with an NPC")
    parser.add_argument("--model", default=NPCSH_CHAT_MODEL, help="Model to use")
    parser.add_argument("--provider", default=NPCSH_CHAT_PROVIDER, help="Provider to use")
    parser.add_argument("--files", nargs="*", help="Files to load into context")
    parser.add_argument("--stream", default="true", help="Use streaming mode")
    parser.add_argument("--npc", type=str, default=os.path.expanduser('~/.npcsh/npc_team/sibiji.npc'), help="Path to NPC file")
    args = parser.parse_args()
    npc_db_conn = create_engine(
        f"sqlite:///{NPCSH_DB_PATH}")
    
    sibiji = NPC(file=args.npc, db_conn=npc_db_conn)
    
    team = Team(team_path = '~/.npcsh/npc_team/', db_conn=npc_db_conn, forenpc= sibiji)
    if sibiji.model is None:
        sibiji.model = args.model
        model = args.model
    else:
        model = sibiji.model
    if sibiji.provider is None:
        sibiji.provider = args.provider
        provider = args.provider
    else:
        provider = sibiji.provider        
    # Enter spool mode
    enter_yap_mode(
        model,
        provider,
        messages=None,
        npc=sibiji,
        team = team,
        files=args.files,
        stream= args.stream.lower() == "true",
    )

if __name__ == "__main__":
    main()