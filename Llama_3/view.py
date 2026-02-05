import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
import h5py
import os
from tqdm import tqdm
import numpy as np

class LlamaView:
    def __init__(self):
        self.model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.torch_dtype = torch.bfloat16

    def _load_model(self):
        if self.model is not None:
            return
        print(f"--- Loading Llama-3 Model: {self.model_id} ---")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.user_tag = "<|eot_id|>"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",        
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True 
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            output_hidden_states=True,
            trust_remote_code=True,
            use_cache=True,
        )
        self.model.eval()

    def unload_model(self):
        print("--- Unloading Llama-3 Model to free VRAM ---")
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        self.model = None
        self.tokenizer = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # @torch.inference_mode()
    # def extract_dual_embeddings(self, full_prompt):
    #     """Generate embeddings for the given text input."""
    #     inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
    #     outputs = self.model(**inputs)

    #     last_hidden_states = outputs.hidden_states[-1]

    #     x2 = last_hidden_states[0, -1, :].to(torch.float32).cpu().numpy()

    #     token_ids = inputs['input_ids'][0].tolist()
    #     eot_id = self.tokenizer.convert_tokens_to_ids(self.user_tag)
    #     eot_indices = [i for i, x in enumerate(token_ids) if x == eot_id]

    #     if len(eot_indices) >= 2:
    #         user_idx = eot_indices[-2]
    #         x1 = last_hidden_states[0, user_idx, :].to(torch.float32).cpu().numpy()
    #     else:
    #         x1 = last_hidden_states[0, 0, :].to(torch.float32).cpu().numpy()
    #     return x1, x2

    @torch.inference_mode()
    def extract_dual_embeddings(self, full_prompt):
        """Generate embeddings for the given text input with targeted pooling."""
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        # 1. Sử dụng Layer -2 hoặc -4 để có tính phân hóa (Discriminative Power) tốt hơn lớp cuối
        # Lớp cuối cùng của Instruct model thường bị "bó" lại để chuẩn bị ra token Score:
        selected_layer = outputs.hidden_states[-2].to(torch.float32) 

        # 2. Chỉ Mean Pooling phần đuôi của Prompt (nơi chứa thông tin Shot thay đổi)
        # Thông thường phần Video Data và mô tả nằm ở 1/3 cuối của Prompt
        seq_len = selected_layer.shape[1]
        dynamic_window = min(seq_len, 300) # Lấy tối đa 300 tokens cuối cùng
        
        # Pool duy nhất phần dynamic để làm nổi bật sự khác biệt giữa các shot
        x1 = torch.mean(selected_layer[:, -dynamic_window:, :], dim=1).squeeze().to(torch.float32).cpu().numpy()

        # x2 vẫn lấy token cuối cùng làm đặc trưng cho "sự hội tụ" của prompt
        x2 = selected_layer[0, -1, :].to(torch.float32).cpu().numpy()

        return x1, x2

    def get_segment_embeddings(self, segment_data):
        """Extract embeddings for a given segment data."""
        try:
            prompt_text = str(segment_data.get('prompt', ''))
            x1, x2 = self.extract_dual_embeddings(prompt_text)
            return x1, x2
        except Exception as e:
            print(f"❌ Failed to extract embeddings: {str(e)}")
            return None, None