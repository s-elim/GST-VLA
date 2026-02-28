"""
Frozen VLM Wrapper: Qwen2.5-VL-7B for GST-VLA
================================================
The VLM is FROZEN for ACCV 2026 submission.
Only the cross-attention projector (GSTtoVLMProjector) is trainable.

For ACCV:  No CoT block. VLM acts as a frozen reasoner.
For ICRA:  DA-CoT block will be added (future work).

Input:
    language_tokens: tokenized language instruction
    spatial_tokens:  projected GST tokens ∈ R^(B, N_g, d_vlm)

Output:
    h_vlm: ∈ R^(B, N, d_vlm)  — hidden states for action expert
"""

import torch
import torch.nn as nn
from typing import Optional, List, Tuple


class QwenVLMWrapper(nn.Module):
    """
    Frozen Qwen2.5-VL-7B wrapper.
    
    Prepends projected spatial tokens to the language token sequence,
    then runs frozen VLM to get hidden states.
    
    Architecture:
        [spatial_tokens | language_tokens] → Qwen2.5-VL-7B → h_vlm
        
    The spatial_tokens are the projected GST output (via GSTtoVLMProjector).
    d_vlm = 3584 for Qwen2.5-VL-7B.
    """

    D_VLM = 3584  # Qwen2.5-VL-7B hidden dim

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        use_mock: bool = False,
        max_new_tokens: int = 1,   # For ACCV: no generation, just hidden states
    ):
        super().__init__()
        self.model_name     = model_name
        self.use_mock       = use_mock
        self.max_new_tokens = max_new_tokens
        self._model     = None
        self._tokenizer = None
        self._processor = None

    def _load(self):
        if self.use_mock:
            return
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            import transformers

            self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            self._model.eval()
            # Freeze ALL VLM parameters
            for name, p in self._model.named_parameters():
                p.requires_grad_(False)

            self._processor = AutoProcessor.from_pretrained(self.model_name)
            print(f"[QwenVLM] Loaded {self.model_name} (frozen)")
        except Exception as e:
            print(f"[QwenVLM] Failed to load model: {e}. Using mock.")
            self.use_mock = True

    @property
    def model(self):
        if self._model is None and not self.use_mock:
            self._load()
        return self._model

    @torch.no_grad()
    def forward(
        self,
        spatial_tokens: torch.Tensor,      # (B, N_g, d_vlm) — projected GST tokens
        input_ids: torch.Tensor,            # (B, L) language token ids
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through frozen VLM.
        
        Prepends spatial tokens to the language embedding sequence,
        then extracts the final hidden states.
        
        Returns:
            h_vlm: (B, N_g + L, d_vlm)  combined hidden states
        """
        if self.use_mock or self.model is None:
            return self._mock_forward(spatial_tokens, input_ids)

        B, N_g, d = spatial_tokens.shape
        device = spatial_tokens.device

        # Get language token embeddings
        embeddings = self.model.model.embed_tokens(input_ids)  # (B, L, d_vlm)

        # Prepend spatial tokens: [spatial | language]
        combined = torch.cat([spatial_tokens, embeddings], dim=1)  # (B, N_g+L, d_vlm)

        # Build attention mask for combined sequence
        if attention_mask is not None:
            spatial_mask = torch.ones(B, N_g, dtype=attention_mask.dtype, device=device)
            combined_mask = torch.cat([spatial_mask, attention_mask], dim=1)
        else:
            combined_mask = torch.ones(B, N_g + input_ids.shape[1], dtype=torch.long, device=device)

        # Forward through VLM backbone (encoder layers only, no generation)
        outputs = self.model.model(
            inputs_embeds=combined,
            attention_mask=combined_mask,
            output_hidden_states=False,
            use_cache=False,
        )

        h_vlm = outputs.last_hidden_state  # (B, N_g+L, d_vlm)
        return h_vlm

    def _mock_forward(
        self,
        spatial_tokens: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Mock VLM for testing."""
        B, N_g, d_vlm = spatial_tokens.shape
        L = input_ids.shape[1]
        device = spatial_tokens.device
        return torch.randn(B, N_g + L, d_vlm, device=device, dtype=spatial_tokens.dtype)

    def get_d_vlm(self) -> int:
        return self.D_VLM

    def tokenize(
        self,
        instructions: List[str],
        device: torch.device,
        max_length: int = 64,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize a list of instruction strings.
        
        Returns:
            input_ids:      (B, L)
            attention_mask: (B, L)
        """
        if self._processor is None and not self.use_mock:
            self._load()

        if self.use_mock or self._processor is None:
            # Mock tokenization
            B = len(instructions)
            input_ids = torch.ones(B, max_length, dtype=torch.long, device=device)
            attention_mask = torch.ones(B, max_length, dtype=torch.long, device=device)
            return input_ids, attention_mask

        encoding = self._processor.tokenizer(
            instructions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        return encoding.input_ids.to(device), encoding.attention_mask.to(device)
