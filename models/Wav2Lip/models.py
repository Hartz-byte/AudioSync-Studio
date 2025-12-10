import torch
from torch import nn
from torch.nn import functional as F
import math

class Conv2d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm2d(cout)
        )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)

class nonlocal_block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.query_conv = nn.Conv2d(c, c // 8, 1)
        self.key_conv = nn.Conv2d(c, c // 8, 1)
        self.value_conv = nn.Conv2d(c, c // 4, 1)
        self.out_conv = nn.Conv2d(c // 4, c, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        query = self.query_conv(x).view(batch_size, -1, height * width)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        value = self.value_conv(x).view(batch_size, -1, height * width)
        
        query = query.permute(0, 2, 1)
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, -1, height, width)
        out = self.out_conv(out)
        
        return x + self.gamma * out

class Wav2Lip(nn.Module):
    def __init__(self):
        super(Wav2Lip, self).__init__()
        
        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(6, 16, 7, 1, 3)),
            nn.Sequential(Conv2d(16, 32, 3, 2, 1), Conv2d(32, 32, 3, 1, 1), Conv2d(32, 32, 3, 1, 1)),
            nn.Sequential(Conv2d(32, 64, 3, 2, 1), Conv2d(64, 64, 3, 1, 1), Conv2d(64, 64, 3, 1, 1)),
            nn.Sequential(Conv2d(64, 64, 3, 2, 1), Conv2d(64, 64, 3, 1, 1), Conv2d(64, 64, 3, 1, 1)),
            nn.Sequential(Conv2d(64, 64, 3, 1, 0), Conv2d(64, 128, 3, 1, 0), Conv2d(128, 128, 3, 1, 0)),
        ])

        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, 3, 1, 1),
            Conv2d(32, 32, 3, 1, 1),
            nn.MaxPool2d(1, (3, 2)),
            Conv2d(32, 64, 3, 1, 1),
            Conv2d(64, 64, 3, 1, 1),
            nn.MaxPool2d(1, (2, 2)),
            Conv2d(64, 64, 3, 1, 1),
            Conv2d(64, 64, 3, 1, 1),
            nn.MaxPool2d(1, (2, 2)),
            Conv2d(64, 128, 2, 1, 0),
            Conv2d(128, 128, 3, 1, 0),
            Conv2d(128, 128, 3, 1, 0),
        )

        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(128, 64, 3, 1, 0)),
            nn.Sequential(Conv2d(64, 64, 3, 1, 1), Conv2d(64, 64, 3, 1, 1)),
            nn.Sequential(Conv2d(64, 64, 3, 1, 1), Conv2d(64, 64, 3, 1, 1)),
            nn.Sequential(Conv2d(64, 32, 3, 1, 1), Conv2d(32, 32, 3, 1, 1)),
            nn.Sequential(Conv2d(32, 16, 3, 1, 1), Conv2d(16, 16, 3, 1, 1)),
        ])

        self.face_upsample_blocks = nn.ModuleList([
            nn.Upsample(scale_factor=2),
            nn.Upsample(scale_factor=2),
            nn.Upsample(scale_factor=2),
            nn.Upsample(scale_factor=2),
            nn.Upsample(scale_factor=2),
        ])

        self.mouth_net = nn.Sequential(
            Conv2d(16, 16, 3, 1, 1),
            nn.Conv2d(16, 3, 1, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, audio_sequences, face_sequences):
        # audio_sequences: (B, 1, 80, T) where T is audio timesteps
        # face_sequences: (B, 6, 96, 96) where 6 = 2 frames * 3 channels (masked + original)
        
        B = audio_sequences.size(0)
        
        # Audio encoding
        audio_encoded = self.audio_encoder(audio_sequences)
        
        # Face encoding
        face_encoded = face_sequences
        for block in self.face_encoder_blocks:
            face_encoded = block(face_encoded)
        
        # Concatenate audio and face features
        audio_encoded_expanded = audio_encoded.expand(B, -1, 12, 96)
        
        x = torch.cat([face_encoded, audio_encoded_expanded], dim=1)
        
        # Face decoding with audio guidance
        for idx, block in enumerate(self.face_decoder_blocks):
            x = block(x)
            x = self.face_upsample_blocks[idx](x)
        
        # Generate mouth prediction
        mouth_pred = self.mouth_net(x)
        
        return mouth_pred

class SyncNet_color(nn.Module):
    def __init__(self):
        super(SyncNet_color, self).__init__()
        
        self.face_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        self.audio_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(1, (3, 2)),
            
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(1, (2, 2)),
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 12 * 6, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward_audio(self, audio_sequences):
        audio_embeddings = self.audio_encoder(audio_sequences)
        audio_embeddings = audio_embeddings.view(audio_embeddings.size(0), -1)
        return audio_embeddings

    def forward_lip(self, face_sequences):
        face_embeddings = self.face_encoder(face_sequences)
        face_embeddings = face_embeddings.view(face_embeddings.size(0), -1)
        return face_embeddings

    def forward(self, audio_sequences, face_sequences):
        audio_embeddings = self.forward_audio(audio_sequences)
        lip_embeddings = self.forward_lip(face_sequences)
        
        audio_lip_embeddings = torch.cat([audio_embeddings, lip_embeddings], dim=1)
        
        output = self.fc(audio_lip_embeddings)
        return output

class SyncNet_color_for_load(nn.Module):
    def __init__(self):
        super(SyncNet_color_for_load, self).__init__()
        self.model = SyncNet_color()

    def forward(self, audio_sequences, face_sequences):
        return self.model(audio_sequences, face_sequences)
