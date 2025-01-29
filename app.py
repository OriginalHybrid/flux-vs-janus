import gradio as gr
from PIL import Image
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from PIL import Image

import numpy as np
import os
import time
from Upsample import RealESRGAN
import spaces

def generate(vl_gpt, vl_chat_processor, tokenizer, input_ids,
             width,
             height,
             temperature: float = 1,
             parallel_size: int = 5,
             cfg_weight: float = 5,
             image_token_num_per_image: int = 576,
             patch_size: int = 16,
             ):
    
    cuda_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Clear CUDA cache before generating
    torch.cuda.empty_cache()
    
    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.int).to(cuda_device)
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id
    inputs_embeds = vl_gpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).to(cuda_device)

    pkv = None
    for i in range(image_token_num_per_image):
        with torch.no_grad():
            outputs = vl_gpt.language_model.model(inputs_embeds=inputs_embeds,
                                                use_cache=True,
                                                past_key_values=pkv)
            pkv = outputs.past_key_values
            hidden_states = outputs.last_hidden_state
            logits = vl_gpt.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)
            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)

            img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
            inputs_embeds = img_embeds.unsqueeze(dim=1)

    

    patches = vl_gpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int),
                                                 shape=[parallel_size, 8, width // patch_size, height // patch_size])

    return generated_tokens.to(dtype=torch.int), patches

def unpack(dec, width, height, parallel_size=5):
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, width, height, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    return visual_img

def image_upsample(img: Image.Image) -> Image.Image:
    if img is None:
        raise Exception("Image not uploaded")
    
    width, height = img.size
    
    if width >= 5000 or height >= 5000:
        raise Exception("The image is too large.")

    global sr_model
    result = sr_model.predict(img.convert('RGB'))
    # Stage 2: Upscale from 768x768 to 1024x1024 (can be done with bicubic resizing)
    result = result.resize((1024, 1024), Image.Resampling.LANCZOS)  
    return result

def setupflux():
    # Load the Flux model
    import torch
    from diffusers import FluxPipeline
    import os

    # Set the environment variable to help with memory fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    # Clear CUDA cache and avoid tracking gradients
    torch.cuda.empty_cache()

    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)

    print("Flux model has been set up!")
    
    return pipe

def setupdeepseek():
    # Load the Deepseek model
    # Load model and processor
    model_path = "deepseek-ai/Janus-Pro-7B"
    config = AutoConfig.from_pretrained(model_path)
    language_config = config.language_config
    language_config._attn_implementation = 'eager'
    vl_gpt = AutoModelForCausalLM.from_pretrained(model_path,
                                                language_config=language_config,
                                                trust_remote_code=True)
    if torch.cuda.is_available():
        vl_gpt = vl_gpt.to(torch.bfloat16).cuda()
    else:
        vl_gpt = vl_gpt.to(torch.float16)

    vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer
    cuda_device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # SR model
    global sr_model
    sr_model = RealESRGAN(torch.device('cuda' if torch.cuda.is_available() else 'cpu'), scale=2)
    sr_model.load_weights(f'weights/RealESRGAN_x2.pth', download=True)

    print("Deepseek model has been set up!")
    return vl_gpt, vl_chat_processor, tokenizer

def create_gradio_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# Image Generation Comparison: Flux vs Deepseek")

        # Define inputs
        prompt_input = gr.Textbox(
            label="Enter your prompt",
            value="A scenic landscape with mountains and lake"
        )

        # Create row for outputs and buttons
        with gr.Row():
            # Flux column
            with gr.Column():
                flux_button = gr.Button("Generate Flux Image")
                flux_output = gr.Image(
                    label="Flux Output",
                    type="pil"
                )

            # Deepseek column
            with gr.Column():
                deepseek_button = gr.Button("Generate Deepseek Image")
                deepseek_output = gr.Image(
                    label="Deepseek Output",
                    type="pil"
                )

        # Define click events
        flux_button.click(
            fn=generate_flux,
            inputs=prompt_input,
            outputs=flux_output
        )

        deepseek_button.click(
            fn=generate_deepseek,
            inputs=prompt_input,
            outputs=deepseek_output
        )

    return demo

def generate_flux(prompt):
    pipe = setupflux()
    print(f"Generating Flux image for prompt: {prompt}")
    pipe.enable_model_cpu_offload()

    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        num_inference_steps=50,
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]

    # Save the image
    image.save("flux-dev.png")
    return image

def generate_deepseek(prompt):
    vl_gpt, vl_chat_processor, tokenizer = setupdeepseek()
    print(f"Generating Deepseek image for prompt: {prompt}")
    torch.cuda.empty_cache()

    width = 384
    height = 384
    parallel_size = 1

    with torch.no_grad():
        messages = [{'role': '<|User|>', 'content': prompt},
                   {'role': '<|Assistant|>', 'content': ''}]
        text = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=messages,
            sft_format=vl_chat_processor.sft_format,
            system_prompt=''
        )
        text = text + vl_chat_processor.image_start_tag

        input_ids = torch.LongTensor(tokenizer.encode(text))
        output, patches = generate( vl_gpt, vl_chat_processor, tokenizer,
            input_ids,
            width // 16 * 16,
            height // 16 * 16,
            cfg_weight=3.5,
            parallel_size=parallel_size
        )

        images = unpack(
            patches,
            width // 16 * 16,
            height // 16 * 16,
            parallel_size=parallel_size
        )

        ret_images = [image_upsample(Image.fromarray(images[i])) for i in range(parallel_size)]

    return ret_images[0] if ret_images else None

# Create and launch the Gradio app
app = create_gradio_interface()
app.launch()