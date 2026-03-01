"""
Frozen VLM Wrapper: Qwen2.5-VL-7B for DEAD-VLA
=================================================
The VLM is FROZEN during all training stages.
LoRA is applied in Stage 2 (not included in this file; added externally).

Input sequence to VLM:
    [z_spatial_tokens | language_tokens | state_token]
    where:
        z_spatial_tokens : (B, N_g, d_vlm)  projected GST tokens
        language_tokens  : (B, L, d_vlm)    embedded language instruction
        state_token      : (B, 1, d_vlm)    encoded robot state (optional)

Output:
    h_vlm ∈ R^(B, N_g + L + 1, d_vlm)   combined hidden states
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple


class QwenVLMWrapper(nn.Module):
    """
    Frozen Qwen2.5-VL-7B wrapper.

    Prepends spatial tokens (and appends state token) to the language
    embedding sequence, then runs the frozen VLM backbone to obtain
    hidden states for the action expert.

    d_vlm = 3584 for Qwen2.5-VL-7B.
    """

    D_VLM = 3584

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        use_mock: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.use_mock   = use_mock
        self._model     = None
        self._processor = None

    def _load(self):
        if self.use_mock:
            return
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
            import torch
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name, 
                quantization_config=quantization_config,
                device_map="auto",
            )
            self._model.eval()
            for p in self._model.parameters():
                p.requires_grad_(False)
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            print(f"[QwenVLM] Loaded {self.model_name} in 4-bit (frozen)")
        except Exception as e:
            print(f"[QwenVLM] Failed to load: {e}. Using mock.")
            self.use_mock = True

    @property
    def model(self):
        if self._model is None and not self.use_mock:
            self._load()
        return self._model

    def forward(
        self,
        spatial_tokens: torch.Tensor,             # (B, N_g, d_vlm) projected GST tokens
        input_ids: torch.Tensor,                   # (B, L) language token ids
        attention_mask: Optional[torch.Tensor] = None,
        state_token: Optional[torch.Tensor] = None,  # (B, 1, d_vlm) MLP(s_t) token
    ) -> torch.Tensor:
        """
        Forward through frozen VLM.

        Input sequence: [spatial | language | state]
        Returns h_vlm: (B, N_g + L [+ 1], d_vlm)
        """
        if self.use_mock or self.model is None:
            return self._mock_forward(spatial_tokens, input_ids, state_token)

        B, N_g, d = spatial_tokens.shape
        L = input_ids.shape[1]
        base_model = getattr(self.model, "model", self.model)

        embed_layer = self.model.get_input_embeddings()
        embed_device = embed_layer.weight.device
        input_ids = input_ids.to(embed_device)

        # Language token embeddings
        lang_emb = embed_layer(input_ids)  # (B, L, d_vlm)

        spatial_tokens = spatial_tokens.to(device=lang_emb.device, dtype=lang_emb.dtype)
        if state_token is not None:
            state_token = state_token.to(device=lang_emb.device, dtype=lang_emb.dtype)

        # Build combined sequence: [spatial | language | state?]
        parts = [spatial_tokens, lang_emb]
        if state_token is not None:
            parts.append(state_token)  # (B, 1, d_vlm)
        combined = torch.cat(parts, dim=1)  # (B, N_g + L [+ 1], d_vlm)

        # Build combined attention mask
        N_total = combined.shape[1]
        if attention_mask is not None:
            attention_mask = attention_mask.to(lang_emb.device)
            spatial_mask = torch.ones(B, N_g, dtype=attention_mask.dtype, device=lang_emb.device)
            if state_token is not None:
                state_mask = torch.ones(B, 1, dtype=attention_mask.dtype, device=lang_emb.device)
                combined_mask = torch.cat([spatial_mask, attention_mask, state_mask], dim=1)
            else:
                combined_mask = torch.cat([spatial_mask, attention_mask], dim=1)
        else:
            combined_mask = torch.ones(B, N_total, dtype=torch.long, device=lang_emb.device)

        outputs = base_model(
            inputs_embeds=combined,
            attention_mask=combined_mask,
            output_hidden_states=False,
            use_cache=False,
        )
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state  # (B, N_total, d_vlm)
        return outputs[0]

    def _mock_forward(
        self,
        spatial_tokens: torch.Tensor,
        input_ids: torch.Tensor,
        state_token: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns mock hidden states that include inputs in the computation graph.
        The 0.0-weighted contributions keep the gradient path intact (zero but not None)
        while the random base provides realistic-looking mock features.
        """
        B, N_g, d_vlm = spatial_tokens.shape
        L = input_ids.shape[1]
        N_total = N_g + L + (1 if state_token is not None else 0)
        device  = spatial_tokens.device

        base = torch.randn(B, N_total, d_vlm, device=device, dtype=spatial_tokens.dtype)

        # Include inputs in the autograd graph so gradients flow to callers.
        # Multiplied by 0.0 so the random values are unaffected numerically.
        base = base + 0.0 * spatial_tokens.sum()
        if state_token is not None:
            base = base + 0.0 * state_token.sum()
        return base

    def tokenize(
        self,
        instructions: List[str],
        device: torch.device,
        max_length: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a batch of instruction strings."""
        if self._processor is None and not self.use_mock:
            self._load()
        if self.use_mock or self._processor is None:
            B = len(instructions)
            return (
                torch.ones(B, max_length, dtype=torch.long, device=device),
                torch.ones(B, max_length, dtype=torch.long, device=device),
            )
        enc = self._processor.tokenizer(
            instructions, return_tensors="pt",
            padding=True, truncation=True, max_length=max_length,
        )
        return enc.input_ids.to(device), enc.attention_mask.to(device)

    def get_d_vlm(self) -> int:
        return self.D_VLM
