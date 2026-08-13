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


def build_model(model_name, in_channels=12, num_classes=5):
    if model_name == "ResNet1D":
        return ResNet1D(in_channels=in_channels, num_classes=num_classes)
    elif model_name == "CNN_BiLSTM_Attention":
        return CNN_BiLSTM_Attention(in_channels=in_channels, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model architecture: {model_name}")
