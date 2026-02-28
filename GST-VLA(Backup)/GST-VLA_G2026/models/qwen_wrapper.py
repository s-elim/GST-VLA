import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

class FrozenQwenVLWrapper(nn.Module):
    def __init__(
        self, 
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", 
        embed_dim: int = 3584,
        dtype: torch.dtype = torch.bfloat16
    ):
        """
        Wraps Qwen2.5-VL, freezes all parameters, and exposes a forward 
        pass that accepts custom continuous spatial tokens (from GST).
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.dtype = dtype
        
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Load the base model in bfloat16 to save memory
        print(f"Loading frozen VLM: {model_name}...")
        self.vlm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map="auto" # Automatically distribute across available GPUs
        )
        
        # 1. Freeze the entire model
        self.vlm.requires_grad_(False)
        self.vlm.eval() # Ensure dropout, etc., are disabled
        
        # We only need the text embedding layer and the decoder layers
        self.text_embedder = self.vlm.get_input_embeddings()

    def forward(
        self, 
        text_prompts: list[str], 
        z_spatial: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Args:
            text_prompts: List of B strings containing the language instruction I.
            z_spatial: (B, 128, 3584) continuous spatial tokens from the GST.
            device: Target device for execution.
        Returns:
            hidden_states: (B, Seq_Len, 3584) the final hidden states before the LM head.
        """
        B = z_spatial.shape[0]
        
        # 1. Tokenize Text (Language I)
        # Assuming simple instructional prompts like "Pick up the red block."
        text_inputs = self.tokenizer(
            text_prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).to(device)
        
        # 2. Get standard discrete text embeddings: (B, Text_Seq_Len, 3584)
        with torch.no_grad(): # Double-check no gradients flow here
            text_embeds = self.text_embedder(text_inputs.input_ids)
        
        # 3. Modality Fusion (Concatenation)
        # We prepend the spatial tokens to the text tokens.
        # Format: [Spatial Tokens] + [Text Tokens]
        # z_spatial must be cast to the same dtype as the VLM (e.g., bfloat16)
        inputs_embeds = torch.cat([z_spatial.to(self.dtype), text_embeds], dim=1)
        
        # 4. Attention Mask Update
        # Since we added 128 spatial tokens, we need to extend the attention mask.
        spatial_mask = torch.ones((B, z_spatial.shape[1]), dtype=torch.long, device=device)
        attention_mask = torch.cat([spatial_mask, text_inputs.attention_mask], dim=1)
        
        # 5. VLM Forward Pass
        # We pass the custom inputs_embeds instead of input_ids.
        # We extract the hidden states from the last layer to feed the Action Expert.
        with torch.no_grad():
            outputs = self.vlm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            
            # Extract the final layer's hidden states: (B, 128 + Text_Seq_Len, 3584)
            final_hidden_states = outputs.hidden_states[-1]
            
        return final_hidden_states