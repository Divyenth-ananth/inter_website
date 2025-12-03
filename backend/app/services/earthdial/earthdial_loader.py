import sys
import os

# Add EarthDial src folder to PYTHONPATH
EARTHDIAL_SRC = "/Data4/MeetRanaInterIIT/vlm-docker/LongVA/EarthDial/src"
sys.path.append(EARTHDIAL_SRC)
from app.config import MODEL_PATH
from app.services.earthdial.constants import (
    IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
    QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN, REF_END_TOKEN,
    BOX_START_TOKEN, BOX_END_TOKEN,
    S2_RGB_10_TOKEN, L8_RGB_30_TOKEN, S2_MS_10_TOKEN,
    HIGH_RGB_05_TOKEN, HIGH_RGB_05_TEMP_TOKEN, HIGH_RGBI_05,
    S1_VH_10_TOKEN, S1_VH_1_TOKEN, S1_VH_TEMP_10,
    TREECLASSIFY, GROUNDING, REFER, CLASSIFY, IDENTIFY, CAPTION, CHANGEDET,UHI, L8_MS_30, HYPER_RGB_3, MB_TOKEN_START, MB_TOKEN_END
)
import torch
from transformers import AutoTokenizer, AutoModel

class EarthDial:
    def __init__(self,model_path):
        print("Loading EarthDial from:", MODEL_PATH)

        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            use_fast=False,
            padding_side='left'
        )
        token_list = [
            IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
            QUAD_START_TOKEN, QUAD_END_TOKEN, REF_START_TOKEN, REF_END_TOKEN,
            BOX_START_TOKEN, BOX_END_TOKEN,
            S2_RGB_10_TOKEN, L8_RGB_30_TOKEN, S2_MS_10_TOKEN,
            HIGH_RGB_05_TOKEN, HIGH_RGB_05_TEMP_TOKEN, HIGH_RGBI_05,
            S1_VH_10_TOKEN, S1_VH_1_TOKEN, S1_VH_TEMP_10,
            TREECLASSIFY, GROUNDING, REFER, CLASSIFY, IDENTIFY, CAPTION, CHANGEDET,
            UHI, L8_MS_30, HYPER_RGB_3, MB_TOKEN_START, MB_TOKEN_END
        ]
        num_new_tokens = self.tokenizer.add_tokens(token_list, special_tokens=True)

        if num_new_tokens > 0:
            print(f"Added {num_new_tokens} missing special tokens to tokenizer.")
        test_ids = self.tokenizer.encode(IMG_CONTEXT_TOKEN, add_special_tokens=False)
        if len(test_ids) != 1:
            print(f"CRITICAL WARNING: {IMG_CONTEXT_TOKEN} is splitting into {test_ids}. Forcing re-add.")
            self.tokenizer.add_special_tokens({'additional_special_tokens': [IMG_CONTEXT_TOKEN]})
        
        # Get the context token ID for the model config
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        print(f"IMG_CONTEXT_TOKEN ID: {self.img_context_token_id}")
        self.model = AutoModel.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        ).eval().cuda()
        target_vocab_size = len(self.tokenizer)
        
        # InternVL stores vocab_size in llm_config, not the root config
        if hasattr(self.model.config, "llm_config"):
            current_vocab_size = self.model.config.llm_config.vocab_size
        else:
            # Fallback for standard HF models
            current_vocab_size = getattr(self.model.config, "vocab_size", 0)

        if current_vocab_size != target_vocab_size:
            print(f"Resizing model embeddings: {current_vocab_size} -> {target_vocab_size}")

            # If using InternVL structure, access the inner language model
            if hasattr(self.model, "language_model"):
                 self.model.language_model.resize_token_embeddings(target_vocab_size)
                 self.model.config.llm_config.vocab_size = target_vocab_size
                 self.model.language_model.config.vocab_size = target_vocab_size
            else:
                 self.model.resize_token_embeddings(target_vocab_size)

        self.model.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.device = next(self.model.parameters()).device
        self.image_size = self.model.config.force_image_size

    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer

