import os
import base64
import io
from typing import Union, List, Optional

import PIL
from PIL import Image

from litellm import image_generation



def generate_image_diffusers(
    prompt: str,
    model: str = "runwayml/stable-diffusion-v1-5",
    device: str = "cpu",
    height: int = 512,
    width: int = 512,
):
    """Generate an image using the Stable Diffusion API."""
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(model)
    pipe = pipe.to(device)

    # Generate the image
    image = pipe(prompt, height=height, width=width)
    image = image.images[0]
    
    return image


def openai_image_gen(
    prompt: str,
    model: str = "dall-e-2",
    attachments: Union[List[Union[str, bytes, Image.Image]], None] = None,
    height: int = 1024,
    width: int = 1024,
    n_images: int = 1,
):
    """Generate or edit an image using the OpenAI API."""
    from openai import OpenAI
    
    client = OpenAI()
    
    if height is None:
        height = 1024
    if width is None:
        width = 1024  
    # Prepare image for editing if provided
    if attachments is not None:
        # Process the attachments into the format OpenAI expects
        processed_images = []
    
        for attachment in attachments:
            if isinstance(attachment, str):
                # Assume it's a file path
                processed_images.append(open(attachment, "rb"))
            elif isinstance(attachment, bytes):
                processed_images.append(io.BytesIO(attachment))
            elif isinstance(attachment, Image.Image):
                img_byte_arr = io.BytesIO()
                attachment.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                processed_images.append(img_byte_arr)
        # Use images.edit for image editing
        result = client.images.edit(
            model=model,
            image=processed_images[0],  # Edit only supports a single image
            prompt=prompt,
            n=n_images,
            size=f"{width}x{height}",
        )
    else:
        # Use images.generate for new image generation
        result = client.images.generate(
            model=model,
            prompt=prompt,
            n=n_images,
            size=f"{width}x{height}",
        )
    if model =='gpt-image-1':
        image_base64 = result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes))
        image.save('generated_image.png') 
    elif model == 'dall-e-2' or model == 'dall-e-3':
        image_base64 = result.data[0].url
        import requests
        response = requests.get(image_base64)
        image = Image.open(io.BytesIO(response.content))
        image.save('generated_image.png')
        
    return image


def gemini_image_gen(
    prompt: str,
    model: str = "imagen-3.0-generate-002",
    attachments: Union[List[Union[str, bytes, Image.Image]], None] = None,
    n_images: int = 1,
    api_key: Optional[str] = None,
):
    """Generate or edit images using Google's Gemini API."""
    from google import genai
    from google.genai import types
    from io import BytesIO

    # Use environment variable if api_key not provided
    if api_key is None:
        api_key = os.environ.get('GEMINI_API_KEY')
    
    client = genai.Client(api_key=api_key)
    
    if attachments is not None:
        # Process the attachments
        processed_images = []
        for attachment in attachments:
            if isinstance(attachment, str):
                # Assume it's a file path
                processed_images.append(Image.open(attachment))
            elif isinstance(attachment, bytes):
                processed_images.append(Image.open(BytesIO(attachment)))
            elif isinstance(attachment, Image.Image):
                processed_images.append(attachment)
        
        # Use generate_content for image editing with gemini-2.0-flash-exp-image-generation
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp-image-generation",
            contents=[prompt, processed_images],
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )
        
        # Process the response
        images = []
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                image = Image.open(BytesIO(part.inline_data.data))
                images.append(image)
        
        return images[0] if images else None
    else:
        # Use generate_images for new image generation with imagen model
        response = client.models.generate_images(
            model=model,
            prompt=prompt,
            config=types.GenerateImagesConfig(
                number_of_images=n_images,
            )
        )
        
        # Process the response
        images = []
        for generated_image in response.generated_images:
            image = Image.open(BytesIO(generated_image.image.image_bytes))
            images.append(image)
        
        return images[0] if images else None


def generate_image(
    prompt: str,
    model: str ,
    provider: str ,
    height: int = 1024,
    width: int = 1024,
    n_images: int = 1,
    api_key: Optional[str] = None,
    api_url: Optional[str] = None,
    attachments: Union[List[Union[str, bytes, Image.Image]], None] = None,
    save_path: Optional[str] = None,
):
    """
    Unified function to generate or edit images using various providers.
    
    Args:
        prompt (str): The prompt for generating/editing the image.
        model (str): The model to use.
        provider (str): The provider to use ('openai', 'diffusers', 'gemini').
        height (int): The height of the output image.
        width (int): The width of the output image.
        n_images (int): Number of images to generate.
        api_key (str): API key for the provider.
        api_url (str): API URL for the provider.
        attachments (list): List of images for editing. Can be file paths, bytes, or PIL Images.
        save_path (str): Path to save the generated image.
        
    Returns:
        PIL.Image: The generated image.
    """
    # Set default model if none provided
    if model is None:
        if provider == "openai":
            model = "dall-e-2"
        elif provider == "diffusers":
            model = "runwayml/stable-diffusion-v1-5"
        elif provider == "gemini":
            model = "imagen-3.0-generate-002"
    
    # Generate or edit the image based on provider
    if provider == "diffusers":
        image = generate_image_diffusers(
            prompt=prompt, 
            model=model, 
            height=height, 
            width=width
        )
    elif provider == "openai":
        image = openai_image_gen(
            prompt=prompt,
            model=model,
            attachments=attachments,
            height=height,
            width=width,
            n_images=n_images
        )
    elif provider == "gemini":
        image = gemini_image_gen(
            prompt=prompt,
            model=model,
            attachments=attachments,
            n_images=n_images,
            api_key=api_key
        )
    else:
        # Validate image size for litellm
        valid_sizes = ["256x256", "512x512", "1024x1024", "1024x1792", "1792x1024"]
        size = f"{height}x{width}"
        if size not in valid_sizes:
            raise ValueError(
                f"Invalid image size: {size}. Please use one of the following: {', '.join(valid_sizes)}"
            )
        
        # litellm doesn't support editing yet, so raise an error if attachments provided
        if attachments is not None:
            raise ValueError("Image editing not supported with litellm provider")
        
        # Generate image using litellm
        result = image_generation(
            prompt=prompt,
            model=f"{provider}/{model}",
            n=n_images,
            size=size,
            api_key=api_key,
            api_base=api_url,
        )
        # Convert the URL result to a PIL image
        # This assumes image_generation returns a URL
        from urllib.request import urlopen
        with urlopen(result) as response:
            image = Image.open(response)
    
    # Save the image if a save path is provided
    if save_path and image:
        image.save(save_path)
    
    return image


def edit_image(
    prompt: str,
    image_path: str,
    provider: str = "openai",
    model: str = None,
    height: int = 1024,
    width: int = 1024,
    save_path: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """
    Convenience function to edit an existing image.
    
    Args:
        prompt (str): The editing instructions.
        image_path (str): Path to the image to edit.
        provider (str): The provider to use for editing.
        model (str): The model to use.
        height (int): The height of the output image.
        width (int): The width of the output image.
        save_path (str): Path to save the edited image.
        api_key (str): API key for the provider.
        
    Returns:
        PIL.Image: The edited image.
    """
    # Set default model based on provider
    if model is None:
        if provider == "openai":
            model = "gpt-image-1"
        elif provider == "gemini":
            model = "gemini-2.0-flash-exp-image-generation"
    
    # Use the generate_image function with attachments
    image = generate_image(
        prompt=prompt,
        provider=provider,
        model=model,
        attachments=[image_path],
        height=height,
        width=width,
        save_path=save_path,
        api_key=api_key
    )
    
    return image


