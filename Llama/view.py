import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import json
import h5py
import os
from tqdm import tqdm
import numpy as np

PROMPT_SYSTEM = "prompt_config.md"

class LlamaView:
    def __init__(self):
        self.model_id = "meta-llama/Llama-2-13b-chat-hf"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.torch_dtype = torch.float16
        self.prefix_cache = None
        self.prefix_idx = None

    def _load_model(self):
        if self.model is not None:
            return
        print(f"--- Loading Llama Model: {self.model_id} ---")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",        
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True 
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            torch_dtype=self.torch_dtype,
            device_map="auto",
            output_hidden_states=True,
            trust_remote_code=True,
            use_cache=False,
        )
        self.model.eval()

    def unload_model(self):
        print("--- Unloading Llama-2 Model to free VRAM ---")
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
    def set_global_prefix(self):
        """Set a global prefix for all inputs to the model."""
        self._load_model()
        print("--- Setting global prefix for Llama Model ---")

        with open(PROMPT_SYSTEM, 'r', encoding='utf-8') as f:
            system_prompt = f.read().strip()

        inputs = self.tokenizer(system_prompt, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, use_cache=True)
        self.prefix_cache = outputs.past_key_values
        self.prefix_idx = inputs.input_ids.shape[1]

    @torch.inference_mode()
    def extract_dual_embeddings(self, variable_text):
        """Generate embeddings for the given text input with targeted pooling."""
        if self.prefix_cache is None:
            self.set_global_prefix()

        inputs_variable = self.tokenizer(
            variable_text + " [/INST]", 
            return_tensors="pt", 
            add_special_tokens=False 
        ).to(self.device)

        prefix_mask = torch.ones((1, self.prefix_idx), dtype=torch.long).to(self.device)
        full_mask = torch.cat([prefix_mask, inputs_variable.attention_mask], dim=1)

        outputs = self.model(
            input_ids=inputs_variable.input_ids,
            attention_mask=full_mask,
            past_key_values=self.prefix_cache,
            use_cache=False,
            output_hidden_states=True
        )

        hidden_layer = outputs.hidden_states[-2] 
        x1 = torch.mean(hidden_layer[:, -min(hidden_layer.shape[1], 300):, :], dim=1).squeeze().cpu().to(torch.float32).numpy()
        x2 = hidden_layer[0, -1, :].cpu().to(torch.float32).numpy()

        del outputs
        del inputs_variable
        del full_mask

        return x1, x2

    def get_segment_embeddings(self, dynamic_data):
        """Extract embeddings for a given segment data."""
        try:
            prompt_text = str(dynamic_data.get('prompt', ''))
            x1, x2 = self.extract_dual_embeddings(prompt_text)
            return x1, x2
        except Exception as e:
            print(f"❌ Failed to extract embeddings: {str(e)}")
            torch.cuda.empty_cache()
            return None, None
        

    # test generate text
    @torch.inference_mode()
    def generate_text(self, variable_text, max_new_tokens=200):
        """Sinh văn bản từ model dựa trên prefix đã cache."""
        if self.prefix_cache is None:
            self.set_global_prefix()

        inputs_variable = self.tokenizer(
            variable_text + " [/INST]", 
            return_tensors="pt", 
            add_special_tokens=False 
        ).to(self.device)

        prefix_mask = torch.ones((1, self.prefix_idx), dtype=torch.long).to(self.device)
        full_mask = torch.cat([prefix_mask, inputs_variable.attention_mask], dim=1)

        output_ids = self.model.generate(
            input_ids=inputs_variable.input_ids,
            attention_mask=full_mask,
            past_key_values=self.prefix_cache,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=self.tokenizer.eos_token_id
        )

        new_tokens = output_ids[0]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return response