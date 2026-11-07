import numpy as np
import torch
import torch.nn as nn
from collections import deque
from sklearn.metrics import confusion_matrix
import json
import pandas as pd
import math
import inspect
from transformers import AutoModelForCausalLM
from typing import List

def get_sinusoidal_position_embedding(
    max_position_embeddings, embedding_dim, dtype=torch.float32
):
    """
    生成正弦位置编码
    
    Args:
        max_position_embeddings: 最大位置数
        embedding_dim: 嵌入维度
        dtype: 数据类型
        
    Returns:
        position_embeddings: 形状为 [max_position_embeddings, embedding_dim] 的位置编码
    """
    position = torch.arange(max_position_embeddings, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embedding_dim, 2, dtype=dtype)
        * (-math.log(10000.0) / embedding_dim)
    )
    
    pe = torch.zeros(max_position_embeddings, embedding_dim, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe


class ClassifierBlock(nn.Module):
    """
    A single transformer-style block for the classifier backbone.
    Implements layer normalization, MLP with expansion, and residual connections.
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        expansion_factor=4,
        dropout_rate=0.3,
        dtype=torch.float32
    ):
        super().__init__()
        self.dtype = dtype
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Pre-layer normalization
        self.layer_norm = nn.LayerNorm(input_dim, dtype=dtype)
        
        # MLP with expansion
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim * expansion_factor, dtype=dtype),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim * expansion_factor, output_dim, dtype=dtype),
            nn.Dropout(dropout_rate),
        )
        
        # Dimension change projection if needed
        self.dim_change = (
            nn.Linear(input_dim, output_dim, dtype=dtype)
            if input_dim != output_dim
            else nn.Identity()
        )
        
    def forward(self, x):
        # Apply layer norm
        normalized = self.layer_norm(x)
        
        # Handle dimension change for residual if needed
        residual = self.dim_change(x)
        
        # MLP block with residual connection
        return residual + self.mlp(normalized)


class ClassifierBackbone(nn.Module):
    """
    Backbone architecture for all classifiers.
    Implements transformer-style MLP blocks with residual connections.
    """

    def __init__(
        self,
        input_dim,
        output_dim=1,  # Default to 1 for binary classification
        hidden_dims=[256, 512, 256],  # Bottleneck structure
        expansion_factor=4,  # Transformer-style expansion
        dropout_rate=0.3,
        dtype=torch.float32,
        use_position_embedding=False,
        max_position_embeddings=1024,
    ):
        super().__init__()
        self.dtype = dtype
        self.use_position_embedding = use_position_embedding
        
        if self.use_position_embedding:
            # Generate sinusoidal position encoding and register as buffer
            position_embedding = get_sinusoidal_position_embedding(
                max_position_embeddings, hidden_dims[0], dtype=dtype
            )
            self.register_buffer("position_embedding", position_embedding)
        
        # Build transformer-style blocks
        self.blocks = nn.ModuleList()
        
        # First projection to match the first hidden dimension
        self.input_projection = nn.Linear(input_dim, hidden_dims[0], dtype=dtype)
        
        block_dims = hidden_dims + [hidden_dims[-1]] # for the last block, the output dimension is the same as the input dimension

        # Create classifier blocks with proper input and output dimensions
        for i in range(len(block_dims) - 1):
            # For the last block, output dimension is the same as input
            # For other blocks, output dimension is the next hidden dimension
            block_input_dim = block_dims[i]
            block_output_dim = block_dims[i+1]
            
            # Create a single block
            self.blocks.append(
                ClassifierBlock(
                    input_dim=block_input_dim,
                    output_dim=block_output_dim,
                    expansion_factor=expansion_factor,
                    dropout_rate=dropout_rate,
                    dtype=dtype
                )
            )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_dims[-1], dtype=dtype),
            nn.Linear(hidden_dims[-1], output_dim, dtype=dtype),
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def apply_position_embedding(self, x, position_ids=None):
        """Apply position embeddings to the input if enabled."""
        if self.use_position_embedding:
            if position_ids is None:
                # Use default position ids (0, 1, 2, ...)
                batch_size = x.shape[0]
                seq_length = x.shape[1] if len(x.shape) > 2 else 1
                position_ids = torch.arange(seq_length, device=x.device).expand(
                    (batch_size, -1)
                )
            
            # Get position embeddings
            pos_emb = self.position_embedding[position_ids]
            
            # Add position embeddings to input
            x = x + pos_emb
        
        return x
    
    def forward(self, x, position_ids=None):
        """
        Forward pass through the backbone.
        
        Args:
            x: Input tensor
            position_ids: Optional tensor of position ids for position embeddings
            
        Returns:
            Model output (logits)
        """
        # Project input to first hidden dimension
        x = self.input_projection(x)
        
        # Apply position embeddings if enabled
        x = self.apply_position_embedding(x, position_ids)
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Output layer
        return self.output_layer(x)
    

class HiddenStatesTokenLMHeadLogitsClassifier(nn.Module):
    """
    A classifier that uses a pretrained LM head for processing token and also takes logits as input.
    This model accepts hidden states, sampled token, and logits as input.
    Uses the ClassifierBackbone for processing.
    """

    def __init__(
        self,
        hidden_states_size=1536,
        logits_size=100,
        hidden_dims=[512, 1024, 512],  # Bottleneck structure
        expansion_factor=4,  # Transformer-style expansion
        dropout_rate=0.3,
        dtype=torch.bfloat16,
        pretrained_model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        pretrained_model=None,
        use_position_embedding=False,
        max_position_embeddings=1024,
        topk=None,
        normalize_input=False,
        freeze_lm_head=True,
    ):
        super().__init__()
        self.dtype = dtype
        self.topk = topk  # Store topk parameter
        self.pretrained_model_name = pretrained_model_name
        self.normalize_input = normalize_input
        self.freeze_lm_head = freeze_lm_head
        
        # Layer normalization for inputs if normalization is enabled
        if self.normalize_input:
            self.layer_norm_hidden_states = nn.LayerNorm(hidden_states_size, dtype=dtype)
            self.layer_norm_token = nn.LayerNorm(hidden_states_size, dtype=dtype)
            # No layer norm for logits as we'll apply softmax instead if normalize_input is True
        
        # Copy weights from a pretrained LM head
        try:
            if pretrained_model is None:
                pretrained_model = AutoModelForCausalLM.from_pretrained(
                    pretrained_model_name
                )
            
            # Get the token embeddings from the pretrained model and create a copy
            embedding_layer = pretrained_model.get_input_embeddings()
            embed_dim = embedding_layer.embedding_dim

            self.token_embeddings = nn.Embedding(
                embedding_layer.num_embeddings,
                embed_dim,
                dtype=dtype
            )
            with torch.no_grad():     
                self.token_embeddings.weight.copy_(embedding_layer.weight.to(dtype=dtype))
            
            print("Successfully copied weights")
        except Exception as e:
            print(f"Failed to load pretrained model: {e}")
            # Create a simple embedding layer as fallback
            vocab_size = 50257  # Default GPT-2 vocab size
            embed_dim = hidden_dims[0]
            self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
            print(f"Using randomly initialized embeddings")

        if self.freeze_lm_head:
            self.token_embeddings.weight.requires_grad = False
        
        # Projection for logits
        self.logits_projection = nn.Linear(logits_size, hidden_states_size, dtype=dtype)
        
        # Combined projection for hidden states + token embeddings + logits projection
        combined_size = hidden_states_size + embed_dim + hidden_states_size
        self.combined_projection = nn.Linear(
            combined_size, hidden_dims[0], dtype=dtype
        )
        
        # Create backbone
        self.backbone = ClassifierBackbone(
            input_dim=hidden_dims[0],
            output_dim=1,
            hidden_dims=hidden_dims,
            expansion_factor=expansion_factor,
            dropout_rate=dropout_rate,
            dtype=dtype,
            use_position_embedding=use_position_embedding,
            max_position_embeddings=max_position_embeddings,
        )
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.token_embeddings:
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, hidden_states, token=None, logits=None, position_ids=None):
        """
        Forward pass through the model.
        
        Args:
            hidden_states: Tensor of hidden states [batch_size, hidden_dim]
            token: Tensor of token IDs [batch_size]
            logits: Tensor of logits [batch_size, logits_size]
            position_ids: Optional tensor of position ids for position embeddings
            
        Returns:
            Model output (logits)
        """
        # Process inputs to correct dtype
        hidden_states = hidden_states.to(dtype=self.dtype)
        logits = logits.to(dtype=self.dtype)
        
        # Get token embeddings
        token_embeddings = self.token_embeddings(token).to(dtype=self.dtype)

        # Apply normalization if enabled
        if self.normalize_input:
            hidden_states = self.layer_norm_hidden_states(hidden_states)
            token_embeddings = self.layer_norm_token(token_embeddings)
            logits = torch.nn.functional.softmax(logits, dim=-1)
        
        # Process logits
        logits_features = self.logits_projection(logits)
        
        # Concatenate all features
        combined_features = torch.cat([hidden_states, token_embeddings, logits_features], dim=-1)
        
        # Project the combined features
        x = self.combined_projection(combined_features)

        # Process through backbone
        return self.backbone(x, position_ids)