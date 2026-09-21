import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNetBlock1D(nn.Module):
    """
    1D Residual Block with Conv1D -> BatchNorm1D -> ReLU -> Conv1D -> BatchNorm1D + Residual Shortcut.
    """

    def __init__(self, in_channels, out_channels, stride=1, kernel_size=5):
        super(ResNetBlock1D, self).__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    """
    Deep Residual Network for 12-Lead ECG 1D Signal Classification.
    Benchmark champion for PTB-XL ECG classification.
    """

    def __init__(self, in_channels=12, num_classes=5, base_filters=64):
        super(ResNet1D, self).__init__()

        self.prep_conv = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = nn.Sequential(
            ResNetBlock1D(base_filters, base_filters, stride=1),
            ResNetBlock1D(base_filters, base_filters, stride=1)
        )

        self.layer2 = nn.Sequential(
            ResNetBlock1D(base_filters, base_filters * 2, stride=2),
            ResNetBlock1D(base_filters * 2, base_filters * 2, stride=1)
        )

        self.layer3 = nn.Sequential(
            ResNetBlock1D(base_filters * 2, base_filters * 4, stride=2),
            ResNetBlock1D(base_filters * 4, base_filters * 4, stride=1)
        )

        self.layer4 = nn.Sequential(
            ResNetBlock1D(base_filters * 4, base_filters * 8, stride=2),
            ResNetBlock1D(base_filters * 8, base_filters * 8, stride=1)
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(base_filters * 8, num_classes)

    def forward(self, x):
        # x: (batch_size, 12, 1000)
        out = self.prep_conv(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.global_pool(out).squeeze(-1) # (batch_size, channels)
        out = self.dropout(out)
        logits = self.fc(out) # (batch_size, num_classes)
        return logits


class Attention1D(nn.Module):
    """
    Temporal Self-Attention Mechanism for 1D sequences.
    """

    def __init__(self, feature_dim):
        super(Attention1D, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1)
        )

    def forward(self, x):
        # x: (batch_size, seq_len, feature_dim)
        weights = self.attention(x) # (batch_size, seq_len, 1)
        weights = F.softmax(weights, dim=1)
        context = torch.sum(x * weights, dim=1) # (batch_size, feature_dim)
        return context, weights


class CNN_BiLSTM_Attention(nn.Module):
    """
    Spatial-Temporal Hybrid Deep Network:
    1D-CNN (Local Feature Extractor) -> BiLSTM (Temporal Sequence Modeling) -> Attention Mechanism -> Classifier.
    """

    def __init__(self, in_channels=12, num_classes=5):
        super(CNN_BiLSTM_Attention, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )

        self.bilstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        self.attention = Attention1D(feature_dim=256) # 128 * 2 for bidirectional
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (batch_size, 12, 1000)
        features = self.feature_extractor(x) # (batch_size, 256, seq_len)
        features = features.permute(0, 2, 1) # (batch_size, seq_len, 256)

        lstm_out, _ = self.bilstm(features) # (batch_size, seq_len, 256)
        context, _ = self.attention(lstm_out) # (batch_size, 256)

        logits = self.fc(context) # (batch_size, num_classes)
        return logits


class PatchEmbedding1D(nn.Module):
    """
    1D Patch Embedding layer for temporal biomedical signals.
    Divides a multi-channel 1D signal into non-overlapping patches and projects to embed_dim.
    """

    def __init__(self, in_channels=4, patch_size=50, embed_dim=128):
        super(PatchEmbedding1D, self).__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (batch_size, in_channels, seq_len)
        x = self.proj(x) # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2) # (batch_size, num_patches, embed_dim)
        x = self.norm(x)
        return x


class ECG_ViT1D(nn.Module):
    """
    1D Vision Transformer (ViT) Architecture for ECG Signal Classification on Selected Leads.
    
    Flow:
      Selected Leads Signal (e.g. 4 leads x 1000 time steps)
      -> 1D Patch Embedding (non-overlapping temporal patches)
      -> Prepend learnable [CLS] token
      -> Add learnable 1D Positional Encodings
      -> L Transformer Encoder Layers (Pre-LN Multi-Head Self-Attention + FFN)
      -> Classification Head ([CLS] output -> LayerNorm -> Linear -> Logits)
    """

    def __init__(
        self,
        in_channels=4,
        seq_len=1000,
        patch_size=50,
        embed_dim=128,
        depth=4,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.1,
        num_classes=5
    ):
        super(ECG_ViT1D, self).__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size
        self.embed_dim = embed_dim

        # 1. Patch Embedding
        self.patch_embed = PatchEmbedding1D(in_channels=in_channels, patch_size=patch_size, embed_dim=embed_dim)

        # 2. Learnable [CLS] Token and 1D Positional Embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)

        # 3. Transformer Encoder Blocks (Pre-LN)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 4. Norm and Classification Head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(embed_dim, num_classes)
        )

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def extract_features(self, x):
        """Extracts latent [CLS] feature vector (for downstream feature concatenation)."""
        B = x.shape[0]
        x = self.patch_embed(x) # (B, num_patches, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (B, num_patches + 1, embed_dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.transformer(x)
        cls_feature = self.norm(x[:, 0]) # (B, embed_dim)
        return cls_feature

    def forward(self, x):
        # x shape: (batch_size, in_channels, seq_len)
        features = self.extract_features(x)
        logits = self.head(features) # (batch_size, num_classes)
        return logits


def build_model(model_name, in_channels=12, num_classes=5, seq_len=1000):
    if model_name == "ResNet1D":
        return ResNet1D(in_channels=in_channels, num_classes=num_classes)
    elif model_name == "CNN_BiLSTM_Attention":
        return CNN_BiLSTM_Attention(in_channels=in_channels, num_classes=num_classes)
    elif model_name in ["ECG_ViT1D", "ViT1D", "ViT"]:
        return ECG_ViT1D(in_channels=in_channels, seq_len=seq_len, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")
