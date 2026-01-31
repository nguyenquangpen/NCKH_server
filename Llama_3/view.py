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

    def process_metadata_to_h5(self, prompt_json_path, base_output_dir):
        """Process metadata JSON file and store embeddings in H5 file."""

        with open(prompt_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        video_id = data['video_id']
        prompts = data['prompts']

        user_dir = os.path.join(base_output_dir, "user_prompt")
        gen_dir = os.path.join(base_output_dir, "gen")
        os.makedirs(user_dir, exist_ok=True)
        os.makedirs(gen_dir, exist_ok=True)

        user_h5_path = os.path.join(user_dir, "user_prompt_pool.h5")
        gen_h5_path = os.path.join(gen_dir, "gen_pool.h5")

        x1_list, x2_list = [], []
        print(f"🧪 Processing {video_id}...")

        for p in tqdm(prompts):
            x1, x2 = self.extract_dual_embeddings(p['prompt'])
            x1_list.append(x1)
            x2_list.append(x2)
            
        x1_array = np.array(x1_list)[:, np.newaxis, :]
        x2_array = np.array(x2_list)[:, np.newaxis, :]

        for path, arr in [(user_h5_path, x1_array), (gen_h5_path, x2_array)]:
            with h5py.File(path, 'a') as f:
                if video_id in f:
                    del f[video_id]
                f.create_dataset(video_id, data=arr.astype(np.float16))

        print(f"✅ Extracted features for {video_id} to {user_dir} and {gen_dir}")

# if __name__ == "__main__":
#     extractor = LlamaView()
#     # Phải dùng file PROMPTS (đã qua prompt_generator.py) chứ không dùng metadata thô
#     PROMPT_JSON = "prompts/videoplayback (9).mp4_prompts.json"
#     OUTPUT_DIR = "llama_emb/tvsum_sum" 

#     try:
#         extractor.process_to_h5(PROMPT_JSON, OUTPUT_DIR)
#     finally:
#         extractor.unload_model()