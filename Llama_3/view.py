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
    
    @torch.inference_mode()
    def extract_dual_embeddings(self, full_prompt):
        """Generate embeddings for the given text input."""
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        last_hidden_states = outputs.hidden_states[-1]

        x2 = last_hidden_states[0, -1, :].cpu().numpy()

        token_ids = inputs['input_ids'][0].tolist()
        eot_id = self.tokenizer.convert_tokens_to_ids(self.user_tag)
        eot_indices = [i for i, x in enumerate(token_ids) if x == eot_id]

        if len(eot_indices) >= 2:
            user_idx = eot_indices[-2]
            x1 = last_hidden_states[0, user_idx, :].cpu().numpy()
        else:
            x1 = last_hidden_states[0, 0, :].cpu().numpy()
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