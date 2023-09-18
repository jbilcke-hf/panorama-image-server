import io
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
from diffusers import DiffusionPipeline, AutoPipelineForInpainting 
from cog_sdxl.dataset_and_utils import TokenEmbeddingsHandler 
from huggingface_hub import hf_hub_download 
from realesrgan import RealESRGANer 
from basicsr.archs.rrdbnet_arch import RRDBNet 
import gradio as gr

class Generator:
    def __init__(self):
        self.pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir="sdxl-cache"
        ).to("cuda")
        self.pipe.load_lora_weights(
            "jbilcke-hf/sdxl-panorama",
            weight_name="lora.safetensors",
            cache_dir="lora-cache"
        )
        text_encoders = [self.pipe.text_encoder, self.pipe.text_encoder_2]
        tokenizers = [self.pipe.tokenizer, self.pipe.tokenizer_2]
        embedding_path = hf_hub_download(
            repo_id="jbilcke-hf/sdxl-panorama",
            filename="embeddings.pti",
            repo_type="model",
            cache_dir="embedding-cache"
        )
        embhandler = TokenEmbeddingsHandler(text_encoders, tokenizers)
        embhandler.load_embeddings(embedding_path)

        self.pipeInpaint = AutoPipelineForInpainting.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir="sdxl-inpaint-cache"
        ).to("cuda")

    def predict(self, prompt, seed):
        prompt_sdxl = prompt + " in the style of <s0><s1>"
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        image = self.pipe(
            prompt_sdxl,
            cross_attention_kwargs={"scale": 0.8},
            width=1024,
            height=512,
            generator=torch.manual_seed(seed),
        ).images[0]

        # Split and swap
        width, height = image.size
        midpoint = width // 2
        left_half = image.crop((0, 0, midpoint, height))
        right_half = image.crop((midpoint, 0, width, height))
        image.paste(right_half, (0, 0))
        image.paste(left_half, (midpoint, 0))

        # Upscale image using RealESRGAN
        model_name = 'RealESRGAN_x4plus'
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        model_path = os.path.join('realesrgan', model_name + ".pth")
        upsampler = RealESRGANer(
            scale=4,
            model_path=model_path,
            model=model,
            tile=0,
            tile_pad=10,
            pre_pad=0,
            half=True
        )
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        output, _ = upsampler.enhance(img, outscale=2)
        image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

        # Crop the image and create a mask
        width, height = image.size
        left = (width - height) // 2
        top = 0
        right = ((width - height) // 2) + height
        bottom = height
        image_mask_area = image.crop((left, top, right, bottom))

        mask = self.create_mask(width, height)

        # Inpainting
        generator = torch.Generator(device="cuda").manual_seed(seed)
        inpainted = self.pipeInpaint(
            prompt=prompt,
            image=image_mask_area,
            mask_image=mask,
            guidance_scale=8.0,
            num_inference_steps=20,
            strength=0.99,
            generator=generator,
        ).images[0]

        image.paste(inpainted, (left, 0))

        return image

    @staticmethod
    def create_mask(width, height):
        """Creates the mask used for inpainting."""
        # Create a new black image
        image = Image.new('L', (width, height), 'black')
        draw = ImageDraw.Draw(image)

        left = width // 6
        right = width // 6 * 5
        draw.rectangle([left, 0, right, height], fill='white')

        # Turn the Image object into a numpy array and normalize it
        mask = np.array(image) / 255.0
        return mask

if __name__=="__main__":
    def gui_interface(prompt, seed):
        generator = Generator()
        return generator.predict(prompt, seed)

    iface = gr.Interface(fn=gui_interface, inputs=["text", "number"], outputs="image")
    iface.launch()