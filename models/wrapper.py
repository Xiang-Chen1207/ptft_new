import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from .backbone import CBraModBackbone

class CBraModWrapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_config = config.get('model', {})
        self._init_backbone(model_config)
        self._init_task_heads(config, model_config)
        self._load_pretrained_weights(model_config)

    def _init_backbone(self, model_config):
        self.backbone = CBraModBackbone(
            in_dim=model_config.get('in_dim', 200),
            d_model=model_config.get('d_model', 200),
            dim_feedforward=model_config.get('dim_feedforward', 800),
            seq_len=model_config.get('seq_len', 30),
            n_layer=model_config.get('n_layer', 12),
            nhead=model_config.get('nhead', 8)
        )

    def _init_task_heads(self, config, model_config):
        self.task_type = config.get('task_type', 'classification')
        self.num_classes = model_config.get('num_classes', 2)
        self.dropout = model_config.get('dropout', 0.1)
        self.d_model = model_config.get('d_model', 200)
        
        # Pretraining Config
        self.pretrain_tasks = model_config.get('pretrain_tasks', ['reconstruction'])
        self.feature_token_type = model_config.get('feature_token_type', 'gap')
        self.feature_token_strategy = model_config.get('feature_token_strategy', 'single')
        self.feature_group_count = model_config.get('feature_group_count', 5)
        
        if self.task_type == 'pretraining':
            self._init_pretraining_heads(model_config)
        else:
            self._init_classification_head(model_config)

    def _init_pretraining_heads(self, model_config):
        # Reconstruction Head
        if 'reconstruction' in self.pretrain_tasks:
            self.head = nn.Linear(self.d_model, model_config.get('out_dim', 200))
        else:
            self.head = None
        
        # Feature Prediction Head
        self.feature_dim = model_config.get('feature_dim', 0)
        if 'feature_pred' in self.pretrain_tasks and self.feature_dim > 0:
            if self.feature_token_type == 'cross_attn':
                self._init_cross_attn_feature_head()
            else:
                self._init_gap_feature_head()
        else:
            self.feature_head = None

    def _init_cross_attn_feature_head(self):
        # Determine tokens based on strategy
        if self.feature_token_strategy == 'single':
            num_tokens, out_dim = 1, self.feature_dim
        elif self.feature_token_strategy == 'all':
            num_tokens, out_dim = self.feature_dim, 1
        elif self.feature_token_strategy == 'group':
            num_tokens, out_dim = self.feature_group_count, None # Handled by MLP
        else:
            raise ValueError(f"Unknown strategy: {self.feature_token_strategy}")
        
        self.feat_query = nn.Parameter(torch.zeros(1, num_tokens, self.d_model))
        self.feat_attn = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=4, batch_first=True)
        
        if self.feature_token_strategy == 'group':
            self.feature_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(num_tokens * self.d_model, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.feature_dim)
            )
        else:
            self.feature_head = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, out_dim)
            )

    def _init_gap_feature_head(self):
        self.feature_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.feature_dim)
        )

    def _init_classification_head(self, model_config):
        head_type = model_config.get('head_type', 'flatten')
        self.cls_head_type = model_config.get('cls_head_type', 'eeg')  # eeg, feat, full

        if head_type == 'pooling':
            self.fc_norm = nn.LayerNorm(self.d_model)

            # Determine input dimension based on cls_head_type
            if self.cls_head_type == 'full':
                # EEG (GAP) + Feature token concatenated
                cls_input_dim = self.d_model * 2
                self._init_cls_cross_attn()
            elif self.cls_head_type == 'feat':
                # Feature token only
                cls_input_dim = self.d_model
                self._init_cls_cross_attn()
            else:
                # EEG (GAP) only - default
                cls_input_dim = self.d_model

            self.head = nn.Linear(cls_input_dim, self.num_classes)
        else:
            self.fc_norm = nn.Identity()
            self.head = nn.Sequential(
                Rearrange('b c s d -> b (c s d)'),
                nn.LazyLinear(10 * 200),
                nn.ELU(),
                nn.Dropout(self.dropout),
                nn.Linear(10 * 200, 200),
                nn.ELU(),
                nn.Dropout(self.dropout),
                nn.Linear(200, self.num_classes),
            )

    def _init_cls_cross_attn(self):
        """Initialize cross-attention components for classification feature token."""
        self.cls_feat_query = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.cls_feat_attn = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=4, batch_first=True)

    def _load_pretrained_weights(self, model_config):
        if not model_config.get('use_pretrained', False):
            return
            
        pretrained_path = model_config.get('pretrained_path')
        if not pretrained_path:
            return
            
        self.load_pretrained(pretrained_path)

    def load_pretrained(self, checkpoint_path):
        print(f"Loading pretrained weights from {checkpoint_path}")
        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        except TypeError:
             # Fallback for older torch versions that don't support weights_only arg
             state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle 'model' key wrapper if present
        if 'model' in state_dict:
            state_dict = state_dict['model']
            
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('proj_out'): # Skip projection head from pretraining
                continue
                
            # If checkpoint has 'patch_embedding' etc directly, map to 'backbone.patch_embedding'
            if not k.startswith('backbone.'):
                # Check if it belongs to backbone components
                if any(k.startswith(p) for p in ['patch_embedding', 'encoder']):
                    new_key = f"backbone.{k}"
                else:
                    new_key = k # Might be other keys or unexpected
            else:
                new_key = k
                
            new_state_dict[new_key] = v
            
        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)
        print(f"Missing keys: {missing}")
        # print(f"Unexpected keys: {unexpected}")

    def forward(self, x, mask=None):
        if self.task_type == 'pretraining':
            return self._forward_pretraining(x, mask)
        else:
            return self._forward_classification(x)

    def _forward_pretraining(self, x, mask=None):
        if mask is None:
            mask = self._generate_mask(x)
            
        feats = self.backbone(x, mask)
        
        # Reconstruction Output
        out = self.head(feats) if self.head is not None else None
        
        # Feature Prediction Output
        feature_pred = None
        if getattr(self, 'feature_head', None) is not None:
            if self.feature_token_type == 'cross_attn':
                feature_pred = self._forward_cross_attn_head(feats)
            else:
                feature_pred = self._forward_gap_head(feats)
                
        return out, mask, feature_pred

    def _forward_classification(self, x):
        feats = self.backbone(x)

        if isinstance(self.fc_norm, nn.LayerNorm):
            # Global Average Pooling for EEG features
            eeg_feat = feats.mean(dim=[1, 2])  # (B, D)

            cls_head_type = getattr(self, 'cls_head_type', 'eeg')

            if cls_head_type == 'full':
                # EEG + Feature token concatenated
                # Apply LayerNorm to each feature separately before concatenation
                feat_token = self._get_cls_feat_token(feats)  # (B, D)
                eeg_normed = self.fc_norm(eeg_feat)
                feat_normed = self.fc_norm(feat_token)
                combined = torch.cat([eeg_normed, feat_normed], dim=-1)  # (B, 2*D)
                out = self.head(combined)
            elif cls_head_type == 'feat':
                # Feature token only
                feat_token = self._get_cls_feat_token(feats)  # (B, D)
                out = self.head(self.fc_norm(feat_token))
            else:
                # EEG (GAP) only - default
                out = self.head(self.fc_norm(eeg_feat))
        else:
            out = self.head(feats)

        if self.num_classes == 1:
            out = out.view(-1)
        return out

    def _get_cls_feat_token(self, feats):
        """Extract feature token via cross-attention for classification."""
        B, C, N, D = feats.shape
        feats_flat = feats.view(B, C * N, D)

        # Expand query: (B, 1, D)
        query = self.cls_feat_query.expand(B, -1, -1)

        # Cross Attention
        attn_output, _ = self.cls_feat_attn(query, feats_flat, feats_flat)

        return attn_output.squeeze(1)  # (B, D)

    def _generate_mask(self, x):
        # Generate random mask (50% masking)
        B, C, N, P = x.shape
        return torch.bernoulli(torch.full((B, C, N), 0.5, device=x.device)).to(x.device)

    def _forward_cross_attn_head(self, feats):
        # feats: (B, C, N, D) -> (B, S, D) where S = C*N
        B, C, N, D = feats.shape
        feats_flat = feats.view(B, C * N, D)
        
        # Expand query: (B, num_tokens, D)
        query = self.feat_query.expand(B, -1, -1)
        
        # Cross Attention
        attn_output, _ = self.feat_attn(query, feats_flat, feats_flat)
        
        if self.feature_token_strategy == 'single':
            return self.feature_head(attn_output).squeeze(1)
        elif self.feature_token_strategy == 'all':
            return self.feature_head(attn_output).squeeze(-1)
        elif self.feature_token_strategy == 'group':
            return self.feature_head(attn_output)
        return None

    def _forward_gap_head(self, feats):
        global_feat = feats.mean(dim=[1, 2])
        return self.feature_head(global_feat)
