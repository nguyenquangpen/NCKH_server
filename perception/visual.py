# perception/visual.py
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

class VisualPerception:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "microsoft/Florence-2-large" # Hoặc "Florence-2-base" cho nhẹ
        
        print(f"Loading Florence-2 on {self.device}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device).eval()
        
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)

    def generate_caption(self, image_pil, task_prompt="<DETAILED_CAPTION>"):
        if image_pil is None:
            return "No image provided"
            
        inputs = self.processor(text=task_prompt, images=image_pil, return_tensors="pt").to(self.device)
        if self.device == "cuda":
            inputs = {k: v.to(torch.float16) if v.dtype == torch.float32 else v for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=512,
                num_beams=3
            )
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text