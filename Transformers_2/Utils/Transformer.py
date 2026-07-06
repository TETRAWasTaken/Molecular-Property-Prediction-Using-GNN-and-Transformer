"""ChemBERTa encoder with Dual Pooling for the hybrid fusion model.

Key improvements over the baseline ``AttentionPooling`` approach:
- **Dual Pooling**: concatenates the ``[CLS]`` token embedding (global
  sentence-level signal) with an attention-weighted mean over non-padding
  tokens (content-focused summary), doubling the information fed to the
  fusion MLP.
- **Pooling dropout**: a dropout layer on the pooled output provides light
  regularisation before the fusion projector.
- **``pooled_hidden_size`` attribute**: exposes the actual output dimension
  (``hidden_size * 2``) so the fusion model can set its projector size
  automatically without hard-coding 768.
"""

import importlib
import subprocess
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class DualPooling(nn.Module):
    """Concatenate the ``[CLS]`` token with an attention-weighted mean.

    The ``[CLS]`` token captures a holistic sentence-level summary while the
    attention-weighted mean emphasises the most chemically relevant tokens.
    Combining both gives the downstream fusion MLP complementary signals.

    Output dimension: ``hidden_size * 2``.

    Args:
        hidden_size: Hidden dimension of the transformer (e.g. 768).
        dropout: Dropout probability applied to the concatenated output.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attention_scorer = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool the transformer hidden states into a fixed-size vector.

        Args:
            hidden_states: Transformer output, shape
                ``[batch, seq_len, hidden_size]``.
            attention_mask: Token validity mask (1 = real, 0 = padding),
                shape ``[batch, seq_len]``.

        Returns:
            Pooled embedding, shape ``[batch, hidden_size * 2]``.
        """
        # --- [CLS] token (position 0) ---
        cls_output = hidden_states[:, 0, :]  # [batch, hidden_size]

        # --- Attention-weighted mean over non-padding tokens ---
        # Raw scores: [batch, seq_len]
        scores = self.attention_scorer(hidden_states).squeeze(-1)
        # Push padding positions to -inf so their softmax weight → 0
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)  # [batch, seq_len]
        attn_output = torch.bmm(
            attn_weights.unsqueeze(1), hidden_states
        ).squeeze(1)  # [batch, hidden_size]

        # Concatenate and apply dropout
        pooled = torch.cat([cls_output, attn_output], dim=-1)  # [batch, hidden_size*2]
        return self.dropout(pooled)


class StandaloneChemBERTa(nn.Module):
    """ChemBERTa text encoder with Dual Pooling.

    Exposes two size attributes for callers:

    - ``hidden_size``: the raw transformer hidden dimension (e.g. 768).
    - ``pooled_hidden_size``: the size of the vector produced by the pooling
      layer (``hidden_size * 2``).  Use this when constructing the fusion
      projector so the projector width is always correct.

    Args:
        model_name: Hugging Face model identifier.
        num_targets: Number of regression targets for the prediction head.
        pool_dropout: Dropout rate applied after the dual pooling operation.
    """

    def __init__(
        self,
        model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        num_targets: int = 12,
        pool_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # Avoid importing the full transformers stack in a broken environment.
        # The ChemBERTa model is a Roberta-based checkpoint, and loading it can
        # trigger torchaudio/torchvision import side effects. Disable those
        # optional features if they are present.
        try:
            import os
            os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            import transformers
            if hasattr(transformers, "utils") and hasattr(transformers.utils, "import_utils"):
                pass
        except Exception:
            pass

        # Remove broken optional audio dependencies that can break model import.
        try:
            import builtins
            import importlib
            import sys
            if "torchaudio" in sys.modules:
                sys.modules.pop("torchaudio", None)
            if "torchvision" in sys.modules:
                sys.modules.pop("torchvision", None)
        except Exception:
            pass

        try:
            self.transformer = AutoModel.from_pretrained(
                model_name, local_files_only=True, trust_remote_code=False
            )
        except Exception:
            try:
                self.transformer = AutoModel.from_pretrained(
                    model_name, local_files_only=False, trust_remote_code=False
                )
            except Exception:
                # Fallback: try the generic RobertaModel class directly if the
                # model config can be resolved from the checkpoint metadata.
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(model_name, local_files_only=False)
                from transformers.models.roberta.modeling_roberta import RobertaModel
                self.transformer = RobertaModel(config)

        self.hidden_size: int = self.transformer.config.hidden_size
        # Output size of the pooling layer — used by fusion model.
        self.pooled_hidden_size: int = self.hidden_size * 2

        self.dual_pool = DualPooling(self.hidden_size, dropout=pool_dropout)
        self.prediction_head = nn.Linear(self.pooled_hidden_size, num_targets)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through ChemBERTa → Dual Pooling → prediction head.

        Args:
            input_ids: Token IDs, shape ``[batch, seq_len]``.
            attention_mask: Attention mask, shape ``[batch, seq_len]``.

        Returns:
            Per-molecule predictions, shape ``[batch, num_targets]``.
        """
        outputs = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask
        )
        pooled = self.dual_pool(outputs.last_hidden_state, attention_mask)
        return self.prediction_head(pooled)