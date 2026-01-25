import torch 
from PIL import Image
import io
import base64
import numpy as np
from io import BytesIO
from .model import FlorenceInput, FlorenceOutput
from transformers import AutoProcessor, AutoModelForCausalLM
from utils.memory import clear_vram

class FlorenceView:
    def __init__(self):
        self.model_id = 'microsoft/Florence-2-large'
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = None
        self.processor = None
    
    def _load_model(self):
        if self.model is not None:
            print("--- Florence-2 Model already loaded ---")
            return
        print(f"--- Loading Florence-2 Model: {self.model_id} ---")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            attn_implementation="eager"
        ).to(self.device).eval()

        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

    def unload_model(self):
        print("--- Unloading Florence-2 Model to free VRAM ---")
        if self.model is not None:
            del self.model
        if self.processor is not None:
            del self.processor
        self.model = None
        self.processor = None
        clear_vram()

    def _decode_image(self, image_b64: str) -> Image.Image:
        try:
            if "," in image_b64:
                image_b64 = image_b64.split(",")[1]
            img_data = base64.b64decode(image_b64)
            return Image.open(BytesIO(img_data)).convert("RGB")
        except Exception as e:
            print(f"Error decoding base64: {e}")
            return None

    def generate_caption(self, data: FlorenceInput):
        """
        handle inference for a list of FlorenceInput and return a list of FlorenceOutput
        """
        self._load_model()
        results = []
        task_prompt = '<DETAILED_CAPTION>' 
        try:
            print("Decoding image from base64...")
            image = self._decode_image(data.image_b64)
            inputs = self.processor(text=task_prompt, images=image, return_tensors="pt")

            print("Generating caption with Florence-2...")
            generated_ids = self.model.generate(
                    input_ids=inputs["input_ids"].to(self.device),
                    pixel_values=inputs["pixel_values"].to(self.device, self.torch_dtype),
                    max_new_tokens=1024,
                    early_stopping=False,
                    do_sample=False,
                    num_beams=3,
                )
            
            print("Decoding generated text...")
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            
            print("Post-processing generated text...")
            parsed_answer = self.processor.post_process_generation(
                generated_text,
                task=task_prompt,
                image_size=(image.width, image.height)
            )
            caption_text = parsed_answer.get(task_prompt, "")

        except Exception as e:
            print(f"Error during Florence inference: {str(e)}")
        return caption_text