import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WaveNetVocoder(nn.Module):
    """Simple WaveNet-style vocoder for mel-to-audio conversion"""
    
    def __init__(self, mel_channels=80, residual_channels=512, skip_channels=256, 
                 num_blocks=3, num_layers=10, kernel_size=2):
        super(WaveNetVocoder, self).__init__()
        
        self.mel_channels = mel_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        
        # Input convolution
        self.start_conv = nn.Conv1d(1, residual_channels, kernel_size=1)
        
        # Mel conditioning network
        self.mel_conv = nn.Conv1d(mel_channels, residual_channels, kernel_size=1)
        
        # WaveNet layers
        self.res_blocks = nn.ModuleList()
        for b in range(num_blocks):
            for l in range(num_layers):
                dilation = 2 ** l
                self.res_blocks.append(
                    ResidualBlock(residual_channels, skip_channels, kernel_size, dilation)
                )
        
        # Output layers
        self.final_conv1 = nn.Conv1d(skip_channels, skip_channels, kernel_size=1)
        self.final_conv2 = nn.Conv1d(skip_channels, 1, kernel_size=1)
        
    def forward(self, mel_spec, audio=None):
        """
        Args:
            mel_spec: Mel spectrogram (batch, mel_channels, time)
            audio: Audio waveform (batch, 1, time) - only for training
        """
        if audio is None:
            # Inference mode - generate audio from mel spectrogram
            return self.inference(mel_spec)
        
        # Training mode
        batch_size, _, time_steps = audio.shape
        
        # Input processing
        x = self.start_conv(audio)
        
        # Mel conditioning
        mel_cond = self.mel_conv(mel_spec)
        
        # Upsample mel to match audio length if necessary
        if mel_cond.size(-1) != x.size(-1):
            mel_cond = F.interpolate(mel_cond, size=x.size(-1), mode='linear', align_corners=False)
        
        # Add mel conditioning
        x = x + mel_cond
        
        # WaveNet blocks
        skip_connections = []
        for block in self.res_blocks:
            x, skip = block(x, mel_cond)
            skip_connections.append(skip)
        
        # Sum skip connections
        x = torch.stack(skip_connections, dim=0).sum(dim=0)
        
        # Final layers
        x = torch.relu(self.final_conv1(x))
        x = self.final_conv2(x)
        
        return torch.tanh(x)
    
    def inference(self, mel_spec, hop_length=256):
        """Generate audio from mel spectrogram"""
        batch_size, mel_channels, mel_time = mel_spec.shape
        audio_length = mel_time * hop_length
        
        # Initialize audio with zeros
        audio = torch.zeros(batch_size, 1, audio_length, device=mel_spec.device)
        
        # Mel conditioning
        mel_cond = self.mel_conv(mel_spec)
        mel_cond = F.interpolate(mel_cond, size=audio_length, mode='linear', align_corners=False)
        
        # Generate audio sample by sample
        for t in range(audio_length):
            # Get current audio context
            start = max(0, t - 1000)  # Use last 1000 samples as context
            audio_context = audio[:, :, start:t+1]
            
            # Pad if necessary
            if audio_context.size(-1) < 1000:
                padding = 1000 - audio_context.size(-1)
                audio_context = F.pad(audio_context, (padding, 0))
            
            # Forward pass
            x = self.start_conv(audio_context)
            x = x + mel_cond[:, :, start:start+x.size(-1)]
            
            skip_connections = []
            for block in self.res_blocks:
                x, skip = block(x, mel_cond[:, :, start:start+x.size(-1)])
                skip_connections.append(skip)
            
            x = torch.stack(skip_connections, dim=0).sum(dim=0)
            x = torch.relu(self.final_conv1(x))
            x = self.final_conv2(x)
            
            # Get the last sample
            sample = torch.tanh(x[:, :, -1:])
            audio[:, :, t:t+1] = sample
        
        return audio

class ResidualBlock(nn.Module):
    """WaveNet residual block with gated activation"""
    
    def __init__(self, residual_channels, skip_channels, kernel_size, dilation):
        super(ResidualBlock, self).__init__()
        
        self.dilated_conv = nn.Conv1d(
            residual_channels, 2 * residual_channels,
            kernel_size=kernel_size, dilation=dilation,
            padding=dilation * (kernel_size - 1) // 2
        )
        
        self.residual_conv = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, kernel_size=1)
        
    def forward(self, x, conditioning=None):
        residual = x
        
        # Dilated convolution
        x = self.dilated_conv(x)
        
        # Split for gated activation
        filter_x, gate_x = x.chunk(2, dim=1)
        
        # Gated activation
        x = torch.tanh(filter_x) * torch.sigmoid(gate_x)
        
        # Residual and skip connections
        residual_out = self.residual_conv(x)
        skip_out = self.skip_conv(x)
        
        return residual + residual_out, skip_out

class SimpleVocoder(nn.Module):
    """Simplified vocoder for quick prototyping"""
    
    def __init__(self, mel_channels=80, hidden_dim=256, num_layers=4):
        super(SimpleVocoder, self).__init__()
        
        self.mel_channels = mel_channels
        self.hidden_dim = hidden_dim
        
        # Upsampling layers
        self.upsample_layers = nn.ModuleList([
            nn.ConvTranspose1d(mel_channels, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
            nn.ConvTranspose1d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1),
        ])
        
        # Refinement layers
        self.refinement_layers = nn.ModuleList([
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3),
        ])
        
        # Output layer
        self.output_layer = nn.Conv1d(hidden_dim, 1, kernel_size=7, padding=3)
        
        # Activation
        self.activation = nn.LeakyReLU(0.1)
        
    def forward(self, mel_spec):
        """
        Args:
            mel_spec: Mel spectrogram (batch, mel_channels, time)
        Returns:
            audio: Audio waveform (batch, 1, time * hop_length)
        """
        x = mel_spec
        
        # Upsampling
        for layer in self.upsample_layers:
            x = self.activation(layer(x))
        
        # Refinement
        for layer in self.refinement_layers:
            residual = x
            x = self.activation(layer(x))
            x = x + residual  # Residual connection
        
        # Output
        x = torch.tanh(self.output_layer(x))
        
        return x

class MelGAN(nn.Module):
    """MelGAN-style generator for high-quality audio synthesis"""
    
    def __init__(self, mel_channels=80, ngf=32, num_upsamples=4):
        super(MelGAN, self).__init__()
        
        self.num_upsamples = num_upsamples
        
        # Initial convolution
        self.init_conv = nn.Conv1d(mel_channels, ngf * (2 ** num_upsamples), kernel_size=7, padding=3)
        
        # Upsampling layers
        self.upsample_layers = nn.ModuleList()
        for i in range(num_upsamples):
            self.upsample_layers.append(
                nn.ConvTranspose1d(
                    ngf * (2 ** (num_upsamples - i)),
                    ngf * (2 ** (num_upsamples - i - 1)),
                    kernel_size=16, stride=8, padding=4
                )
            )
        
        # Residual blocks
        self.res_blocks = nn.ModuleList()
        for i in range(num_upsamples):
            blocks = nn.ModuleList()
            channels = ngf * (2 ** (num_upsamples - i - 1))
            for j in range(3):  # 3 residual blocks per upsampling layer
                blocks.append(ResBlock(channels))
            self.res_blocks.append(blocks)
        
        # Final convolution
        self.final_conv = nn.Conv1d(ngf, 1, kernel_size=7, padding=3)
        
    def forward(self, mel_spec):
        x = self.init_conv(mel_spec)
        
        for i, upsample in enumerate(self.upsample_layers):
            x = F.leaky_relu(x, 0.1)
            x = upsample(x)
            
            # Apply residual blocks
            for res_block in self.res_blocks[i]:
                x = res_block(x)
        
        x = F.leaky_relu(x, 0.1)
        x = self.final_conv(x)
        x = torch.tanh(x)
        
        return x

class ResBlock(nn.Module):
    """Residual block for MelGAN"""
    
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock, self).__init__()
        
        self.convs1 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1, 
                     padding=kernel_size//2 * d, dilation=d)
            for d in dilation
        ])
        
        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel_size, 1,
                     padding=kernel_size//2 * d, dilation=d)
            for d in dilation
        ])
        
    def forward(self, x):
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = x
            x = F.leaky_relu(x, 0.1)
            x = conv1(x)
            x = F.leaky_relu(x, 0.1)
            x = conv2(x)
            x = x + residual
        return x
