import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TacotronEncoder(nn.Module):
    """Tacotron encoder with bidirectional LSTM"""
    
    def __init__(self, vocab_size, embedding_dim, encoder_dim):
        super(TacotronEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(embedding_dim, encoder_dim, kernel_size=5, padding=2),
            nn.Conv1d(encoder_dim, encoder_dim, kernel_size=5, padding=2),
            nn.Conv1d(encoder_dim, encoder_dim, kernel_size=5, padding=2)
        ])
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(encoder_dim),
            nn.BatchNorm1d(encoder_dim),
            nn.BatchNorm1d(encoder_dim)
        ])
        self.lstm = nn.LSTM(encoder_dim, encoder_dim // 2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x)  # (batch, seq_len, embedding_dim)
        x = x.transpose(1, 2)  # (batch, embedding_dim, seq_len)
        
        # Convolutional layers
        for conv, bn in zip(self.conv_layers, self.batch_norms):
            x = self.dropout(torch.relu(bn(conv(x))))
        
        x = x.transpose(1, 2)  # (batch, seq_len, encoder_dim)
        
        # LSTM
        outputs, _ = self.lstm(x)
        return outputs

class BahdanauAttention(nn.Module):
    """Bahdanau attention mechanism"""
    
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(BahdanauAttention, self).__init__()
        self.encoder_proj = nn.Linear(encoder_dim, attention_dim)
        self.decoder_proj = nn.Linear(decoder_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1)
        
    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs: (batch, seq_len, encoder_dim)
        # decoder_hidden: (batch, decoder_dim)
        
        seq_len = encoder_outputs.size(1)
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, seq_len, 1)
        
        # Calculate attention scores
        energy = torch.tanh(
            self.encoder_proj(encoder_outputs) + self.decoder_proj(decoder_hidden)
        )
        attention_weights = torch.softmax(self.v(energy).squeeze(2), dim=1)
        
        # Apply attention weights
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
        return context.squeeze(1), attention_weights

class TacotronDecoder(nn.Module):
    """Tacotron decoder with attention"""
    
    def __init__(self, encoder_dim, decoder_dim, attention_dim, num_mels):
        super(TacotronDecoder, self).__init__()
        self.num_mels = num_mels
        self.decoder_dim = decoder_dim
        
        self.attention = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)
        self.prenet = nn.Sequential(
            nn.Linear(num_mels, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        self.attention_rnn = nn.LSTMCell(128 + encoder_dim, decoder_dim)
        self.decoder_rnn = nn.LSTMCell(decoder_dim, decoder_dim)
        
        self.projection = nn.Linear(decoder_dim + encoder_dim, num_mels)
        self.stop_projection = nn.Linear(decoder_dim + encoder_dim, 1)
        
    def forward(self, encoder_outputs, targets=None, max_len=1000):
        batch_size = encoder_outputs.size(0)
        device = encoder_outputs.device
        
        # Initialize hidden states
        attention_hidden = torch.zeros(batch_size, self.decoder_dim, device=device)
        attention_cell = torch.zeros(batch_size, self.decoder_dim, device=device)
        decoder_hidden = torch.zeros(batch_size, self.decoder_dim, device=device)
        decoder_cell = torch.zeros(batch_size, self.decoder_dim, device=device)
        
        # Initialize first input
        mel_input = torch.zeros(batch_size, self.num_mels, device=device)
        
        outputs = []
        attention_weights = []
        stop_tokens = []
        
        if targets is not None:
            # Teacher forcing mode
            max_len = targets.size(1)
            
        for t in range(max_len):
            # Prenet
            prenet_out = self.prenet(mel_input)
            
            # Attention
            context, attn_weights = self.attention(encoder_outputs, attention_hidden)
            attention_weights.append(attn_weights)
            
            # Attention RNN
            attention_input = torch.cat([prenet_out, context], dim=1)
            attention_hidden, attention_cell = self.attention_rnn(
                attention_input, (attention_hidden, attention_cell)
            )
            
            # Decoder RNN
            decoder_hidden, decoder_cell = self.decoder_rnn(
                attention_hidden, (decoder_hidden, decoder_cell)
            )
            
            # Output projection
            decoder_output = torch.cat([decoder_hidden, context], dim=1)
            mel_output = self.projection(decoder_output)
            stop_token = torch.sigmoid(self.stop_projection(decoder_output))
            
            outputs.append(mel_output)
            stop_tokens.append(stop_token)
            
            # Update input for next step
            if targets is not None:
                # Teacher forcing
                mel_input = targets[:, t] if t < targets.size(1) - 1 else mel_output
            else:
                mel_input = mel_output
                # Stop if stop token is high (using very high threshold for untrained models)
                # Only stop if we have at least 100 frames and stop token is very confident
                if t > 100 and stop_token.mean() > 0.95:
                    break
        
        outputs = torch.stack(outputs, dim=1)
        attention_weights = torch.stack(attention_weights, dim=1)
        stop_tokens = torch.stack(stop_tokens, dim=1)
        
        return outputs, attention_weights, stop_tokens

class PostNet(nn.Module):
    """Post-processing network to refine mel spectrograms"""
    
    def __init__(self, num_mels, postnet_dim=512, num_layers=5):
        super(PostNet, self).__init__()
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # First layer
        self.convs.append(nn.Conv1d(num_mels, postnet_dim, kernel_size=5, padding=2))
        self.batch_norms.append(nn.BatchNorm1d(postnet_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(nn.Conv1d(postnet_dim, postnet_dim, kernel_size=5, padding=2))
            self.batch_norms.append(nn.BatchNorm1d(postnet_dim))
        
        # Last layer
        self.convs.append(nn.Conv1d(postnet_dim, num_mels, kernel_size=5, padding=2))
        self.batch_norms.append(nn.BatchNorm1d(num_mels))
        
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # x: (batch, seq_len, num_mels)
        # Expected input: (batch, seq_len, num_mels) where num_mels=80
        # Conv1d expects: (batch, channels, seq_len)

        print(f"PostNet input shape: {x.shape}")

        # Transpose to Conv1d format: (batch, num_mels, seq_len)
        x = x.transpose(1, 2)

        print(f"PostNet after transpose: {x.shape}")

        for i, (conv, bn) in enumerate(zip(self.convs, self.batch_norms)):
            x = conv(x)
            x = bn(x)

            if i < len(self.convs) - 1:
                x = torch.tanh(x)
                x = self.dropout(x)

        # Transpose back to (batch, seq_len, num_mels)
        x = x.transpose(1, 2)
        return x

class TacotronTTS(nn.Module):
    """Complete Tacotron TTS model"""
    
    def __init__(self, config):
        super(TacotronTTS, self).__init__()
        self.config = config
        
        # Extract configuration
        vocab_size = config.get('vocab_size', 256)
        embedding_dim = config.get('embedding_dim', 512)
        encoder_dim = config.get('encoder_dim', 512)
        decoder_dim = config.get('decoder_dim', 1024)
        attention_dim = config.get('attention_dim', 128)
        num_mels = config.get('num_mels', 80)
        
        # Model components
        self.encoder = TacotronEncoder(vocab_size, embedding_dim, encoder_dim)
        self.decoder = TacotronDecoder(encoder_dim, decoder_dim, attention_dim, num_mels)
        self.postnet = PostNet(num_mels)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
    def forward(self, text_inputs, mel_targets=None):
        # Encode text
        encoder_outputs = self.encoder(text_inputs)
        
        # Decode to mel spectrograms
        mel_outputs, attention_weights, stop_tokens = self.decoder(
            encoder_outputs, mel_targets
        )
        
        # Post-processing
        # Debug: print tensor shapes
        print(f"mel_outputs shape before postnet: {mel_outputs.shape}")
        
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet
        
        return {
            'mel_outputs': mel_outputs,
            'mel_outputs_postnet': mel_outputs_postnet,
            'attention_weights': attention_weights,
            'stop_tokens': stop_tokens
        }
    
    def inference(self, text_inputs, max_len=1000):
        """Inference mode without teacher forcing"""
        self.eval()
        with torch.no_grad():
            encoder_outputs = self.encoder(text_inputs)
            mel_outputs, attention_weights, stop_tokens = self.decoder(
                encoder_outputs, targets=None, max_len=max_len
            )
            # Debug: print tensor shapes  
            print(f"mel_outputs shape before postnet (inference): {mel_outputs.shape}")
            
            mel_outputs_postnet = self.postnet(mel_outputs)
            mel_outputs_postnet = mel_outputs + mel_outputs_postnet
            
        return {
            'mel_outputs': mel_outputs,
            'mel_outputs_postnet': mel_outputs_postnet,
            'attention_weights': attention_weights,
            'stop_tokens': stop_tokens
        }
    
    def calculate_loss(self, outputs, targets):
        """Calculate training loss"""
        mel_targets = targets['mel_targets']
        stop_targets = targets['stop_targets']
        
        # Mel spectrogram loss
        mel_loss = self.criterion(outputs['mel_outputs'], mel_targets)
        mel_postnet_loss = self.criterion(outputs['mel_outputs_postnet'], mel_targets)
        
        # Stop token loss
        stop_loss = F.binary_cross_entropy(
            outputs['stop_tokens'].squeeze(-1), 
            stop_targets.float()
        )
        
        total_loss = mel_loss + mel_postnet_loss + stop_loss
        
        return {
            'total_loss': total_loss,
            'mel_loss': mel_loss,
            'mel_postnet_loss': mel_postnet_loss,
            'stop_loss': stop_loss
        }
