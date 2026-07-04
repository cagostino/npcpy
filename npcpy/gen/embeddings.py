

from typing import List, Dict, Optional
import numpy as np
from datetime import datetime

try:
    from openai import OpenAI
    import anthropic
except: 
    pass

def get_ollama_embeddings(
    inputs: List[str], model: str = "nomic-embed-text"
) -> List[List[float]]:
    """Generate embeddings using Ollama. Supports both text prompts and image file paths.
    If an input is a path to an existing image file, it is embedded as an image; otherwise as text."""
    import os
    import ollama

    embeddings = []
    for item in inputs:
        if os.path.exists(item) and item.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")):
            response = ollama.embed(model=model, input=item)
        else:
            response = ollama.embeddings(model=model, prompt=item)
        embeddings.append(response["embedding"])
    return embeddings

def get_openai_embeddings(
    texts: List[str], model: str = "text-embedding-3-small"
) -> List[List[float]]:
    """Generate embeddings using OpenAI."""
    client = OpenAI()
    response = client.embeddings.create(input=texts, model=model)
    return [embedding.embedding for embedding in response.data]

def store_embeddings_for_model(
    texts,
    embeddings,
    chroma_client,
    model,
    provider,
    metadata=None,
):
    collection_name = f"{provider}_{model}_embeddings"
    collection = chroma_client.get_collection(collection_name)

    
    if metadata is None:
        metadata = [{"text_length": len(text)} for text in texts]  
        print(
            "metadata is none, creating metadata for each document as the length of the text"
        )
    
    collection.add(
        ids=[str(i) for i in range(len(texts))],
        embeddings=embeddings,
        metadatas=metadata,  
        documents=texts,
    )

def delete_embeddings_from_collection(collection, ids):
    """Delete embeddings by id from Chroma collection."""
    if ids:
        collection.delete(ids=ids)  

def _normalize_embedding(vec: np.ndarray) -> List[float]:
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


def _load_nomic_vision_onnx():
    from huggingface_hub import hf_hub_download
    import onnxruntime as ort
    from transformers import AutoImageProcessor
    model_name = "nomic-ai/nomic-embed-vision-v1.5"
    onnx_path = hf_hub_download(repo_id=model_name, filename="onnx/model.onnx")
    processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return processor, session


def get_hf_image_embeddings(
    image_paths: List[str],
    model_name: str = "nomic-ai/nomic-embed-vision-v1.5",
    device: str = "cpu",
) -> List[List[float]]:
    """Generate normalized image embeddings using a Hugging Face vision model.
    Model and processor are loaded once and cached in-process for repeated calls.
    For the Nomic vision model the exported ONNX runtime is used because the
    PyTorch implementation returns NaN embeddings on CPU in this environment."""
    import os
    from PIL import Image
    import torch

    cache_key = f"_hf_image_embedder_{model_name}_{device}"
    if cache_key not in globals():
        use_onnx = (
            model_name == "nomic-ai/nomic-embed-vision-v1.5"
        )
        if use_onnx:
            try:
                globals()[cache_key] = _load_nomic_vision_onnx()
            except Exception:
                use_onnx = False
        if not use_onnx:
            from transformers import AutoConfig, AutoModel, AutoImageProcessor, PreTrainedModel
            try:
                config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            except Exception:
                # Some remote configs pass float values where ints are expected.
                # Patch the cached config.json and retry.
                import json
                from huggingface_hub import hf_hub_download
                config_path = hf_hub_download(repo_id=model_name, filename="config.json")
                with open(config_path) as f:
                    cfg = json.load(f)
                for key in ("n_inner", "n_positions", "num_hidden_layers", "vocab_size", "n_embd"):
                    if key in cfg:
                        cfg[key] = int(cfg[key])
                with open(config_path, "w") as f:
                    json.dump(cfg, f)
                config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
            processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
            # Some custom remote models lack attributes that newer transformers expects during load.
            if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
                PreTrainedModel.all_tied_weights_keys = property(lambda self: {})
            model = AutoModel.from_pretrained(model_name, config=config, trust_remote_code=True).to(device).eval()
            globals()[cache_key] = (processor, model)
    processor, backend = globals()[cache_key]

    embeddings = []
    if isinstance(backend, torch.nn.Module):
        model = backend
        with torch.no_grad():
            for path in image_paths:
                img = Image.open(path).convert("RGB")
                inputs = processor(images=img, return_tensors="pt").to(device)
                outputs = model(**inputs)
                # Mean-pool the final hidden state and L2-normalize.
                vec = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                embeddings.append(_normalize_embedding(vec))
    else:
        session = backend
        for path in image_paths:
            img = Image.open(path).convert("RGB")
            inputs = processor(images=img, return_tensors="np")
            pixel_values = inputs["pixel_values"].astype("float32")
            outputs = session.run(None, {"pixel_values": pixel_values})
            last_hidden_state = outputs[0]
            # Mean-pool the final hidden state and L2-normalize.
            vec = last_hidden_state.mean(axis=1).squeeze()
            embeddings.append(_normalize_embedding(vec))
    return embeddings


def get_embeddings(
    texts: List[str],
    model: str ,
    provider: str,
) -> List[List[float]]:
    """Generate embeddings using the specified provider and store them in Chroma."""
    if provider == "ollama":
        embeddings = get_ollama_embeddings(texts, model)
    elif provider == "openai":
        embeddings = get_openai_embeddings(texts, model)
    else:
        raise ValueError(f"Unsupported provider: {provider}")



    return embeddings
