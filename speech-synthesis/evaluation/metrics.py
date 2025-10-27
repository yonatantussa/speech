import numpy as np
import librosa
import torch
import torch.nn.functional as F
from scipy import signal
from scipy.spatial.distance import cosine
from sklearn.metrics import mean_squared_error
import pesq
import pystoi
from typing import Dict, List, Tuple, Optional

class AudioMetrics:
    """Comprehensive audio quality evaluation metrics"""
    
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
    
    def evaluate_audio(self, audio, sample_rate=None):
        """Evaluate audio quality with multiple metrics"""
        if sample_rate is None:
            sample_rate = self.sample_rate
        
        metrics = {}
        
        # Basic audio features
        metrics.update(self._basic_features(audio, sample_rate))
        
        # Spectral features
        metrics.update(self._spectral_features(audio, sample_rate))
        
        # Perceptual features
        metrics.update(self._perceptual_features(audio, sample_rate))
        
        return metrics
    
    def _basic_features(self, audio, sample_rate):
        """Calculate basic audio features"""
        features = {}
        
        # RMS Energy
        features['rms'] = np.sqrt(np.mean(audio**2))
        
        # Zero Crossing Rate
        features['zcr'] = np.mean(librosa.feature.zero_crossing_rate(audio))
        
        # Dynamic range
        features['dynamic_range'] = np.max(audio) - np.min(audio)
        
        # Signal-to-noise ratio (rough estimate)
        noise_estimate = np.std(audio[:1000])  # Use first 1000 samples as noise estimate
        signal_power = np.mean(audio**2)
        features['snr_estimate'] = 10 * np.log10(signal_power / (noise_estimate**2 + 1e-8))
        
        return features
    
    def _spectral_features(self, audio, sample_rate):
        """Calculate spectral features"""
        features = {}
        
        # Compute STFT
        stft = librosa.stft(audio)
        magnitude = np.abs(stft)
        
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sample_rate)
        features['spectral_centroid'] = np.mean(spectral_centroid)
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sample_rate)
        features['spectral_rolloff'] = np.mean(spectral_rolloff)
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=sample_rate)
        features['spectral_bandwidth'] = np.mean(spectral_bandwidth)
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(S=magnitude, sr=sample_rate)
        features['spectral_contrast'] = np.mean(spectral_contrast)
        
        # Spectral flatness
        spectral_flatness = librosa.feature.spectral_flatness(S=magnitude)
        features['spectral_flatness'] = np.mean(spectral_flatness)
        
        return features
    
    def _perceptual_features(self, audio, sample_rate):
        """Calculate perceptual features"""
        features = {}
        
        # Mel-frequency cepstral coefficients
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfccs, axis=1)
        features['mfcc_std'] = np.std(mfccs, axis=1)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        features['chroma_mean'] = np.mean(chroma, axis=1)
        
        # Tonnetz (tonal centroid features)
        tonnetz = librosa.feature.tonnetz(y=audio, sr=sample_rate)
        features['tonnetz_mean'] = np.mean(tonnetz, axis=1)
        
        return features
    
    def calculate_similarity(self, audio1, audio2, method='mfcc'):
        """Calculate similarity between two audio signals"""
        if method == 'mfcc':
            return self._mfcc_similarity(audio1, audio2)
        elif method == 'spectral':
            return self._spectral_similarity(audio1, audio2)
        elif method == 'mel':
            return self._mel_similarity(audio1, audio2)
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    def _mfcc_similarity(self, audio1, audio2):
        """Calculate MFCC-based similarity"""
        mfcc1 = librosa.feature.mfcc(y=audio1, sr=self.sample_rate, n_mfcc=13)
        mfcc2 = librosa.feature.mfcc(y=audio2, sr=self.sample_rate, n_mfcc=13)
        
        # Take mean across time
        mfcc1_mean = np.mean(mfcc1, axis=1)
        mfcc2_mean = np.mean(mfcc2, axis=1)
        
        # Calculate cosine similarity
        similarity = 1 - cosine(mfcc1_mean, mfcc2_mean)
        return similarity
    
    def _spectral_similarity(self, audio1, audio2):
        """Calculate spectral similarity"""
        stft1 = np.abs(librosa.stft(audio1))
        stft2 = np.abs(librosa.stft(audio2))
        
        # Take mean across time
        spec1_mean = np.mean(stft1, axis=1)
        spec2_mean = np.mean(stft2, axis=1)
        
        # Calculate cosine similarity
        similarity = 1 - cosine(spec1_mean, spec2_mean)
        return similarity
    
    def _mel_similarity(self, audio1, audio2):
        """Calculate mel spectrogram similarity"""
        mel1 = librosa.feature.melspectrogram(y=audio1, sr=self.sample_rate)
        mel2 = librosa.feature.melspectrogram(y=audio2, sr=self.sample_rate)
        
        # Take mean across time
        mel1_mean = np.mean(mel1, axis=1)
        mel2_mean = np.mean(mel2, axis=1)
        
        # Calculate cosine similarity
        similarity = 1 - cosine(mel1_mean, mel2_mean)
        return similarity

class TTSMetrics:
    """Metrics specifically for TTS model evaluation"""
    
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
        self.audio_metrics = AudioMetrics(sample_rate)
    
    def evaluate_mel_spectrogram(self, predicted_mel, target_mel):
        """Evaluate mel spectrogram prediction quality"""
        metrics = {}
        
        # Mean Squared Error
        metrics['mel_mse'] = np.mean((predicted_mel - target_mel)**2)
        
        # Mean Absolute Error
        metrics['mel_mae'] = np.mean(np.abs(predicted_mel - target_mel))
        
        # Structural Similarity Index
        metrics['mel_ssim'] = self._compute_ssim(predicted_mel, target_mel)
        
        # Spectral convergence
        metrics['spectral_convergence'] = self._spectral_convergence(predicted_mel, target_mel)
        
        return metrics
    
    def evaluate_attention_alignment(self, attention_weights):
        """Evaluate attention alignment quality"""
        metrics = {}
        
        # Attention alignment score (diagonal-ness)
        metrics['alignment_score'] = self._compute_alignment_score(attention_weights)
        
        # Attention entropy
        metrics['attention_entropy'] = self._compute_attention_entropy(attention_weights)
        
        return metrics
    
    def evaluate_prosody(self, audio, reference_audio=None):
        """Evaluate prosodic features"""
        metrics = {}
        
        # Fundamental frequency
        f0 = librosa.yin(audio, fmin=80, fmax=400, sr=self.sample_rate)
        f0_valid = f0[~np.isnan(f0)]
        
        if len(f0_valid) > 0:
            metrics['f0_mean'] = np.mean(f0_valid)
            metrics['f0_std'] = np.std(f0_valid)
            metrics['f0_range'] = np.max(f0_valid) - np.min(f0_valid)
        
        # Energy contour
        energy = librosa.feature.rms(y=audio, hop_length=512)[0]
        metrics['energy_mean'] = np.mean(energy)
        metrics['energy_std'] = np.std(energy)
        
        # Speaking rate (rough estimate)
        zero_crossings = librosa.zero_crossings(audio)
        metrics['speaking_rate'] = np.sum(zero_crossings) / (len(audio) / self.sample_rate)
        
        # If reference is provided, calculate prosody similarity
        if reference_audio is not None:
            metrics.update(self._prosody_similarity(audio, reference_audio))
        
        return metrics
    
    def _compute_ssim(self, pred, target):
        """Compute Structural Similarity Index for mel spectrograms"""
        # Convert to torch tensors if needed
        if isinstance(pred, np.ndarray):
            pred = torch.tensor(pred).float()
        if isinstance(target, np.ndarray):
            target = torch.tensor(target).float()
        
        # Normalize to [0, 1]
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        target = (target - target.min()) / (target.max() - target.min() + 1e-8)
        
        # Compute SSIM (simplified version)
        mu_pred = torch.mean(pred)
        mu_target = torch.mean(target)
        
        sigma_pred = torch.var(pred)
        sigma_target = torch.var(target)
        sigma_cross = torch.mean((pred - mu_pred) * (target - mu_target))
        
        c1 = 0.01**2
        c2 = 0.03**2
        
        ssim = ((2 * mu_pred * mu_target + c1) * (2 * sigma_cross + c2)) / \
               ((mu_pred**2 + mu_target**2 + c1) * (sigma_pred + sigma_target + c2))
        
        return ssim.item()
    
    def _spectral_convergence(self, pred, target):
        """Compute spectral convergence"""
        return np.linalg.norm(target - pred) / np.linalg.norm(target)
    
    def _compute_alignment_score(self, attention_weights):
        """Compute attention alignment score (how diagonal the attention is)"""
        if len(attention_weights.shape) == 3:
            # Take first item in batch
            attention = attention_weights[0]
        else:
            attention = attention_weights
        
        decoder_len, encoder_len = attention.shape
        
        # Create ideal diagonal attention
        ideal_attention = np.zeros_like(attention)
        for i in range(decoder_len):
            # Map decoder position to encoder position
            encoder_pos = int(i * encoder_len / decoder_len)
            if encoder_pos < encoder_len:
                ideal_attention[i, encoder_pos] = 1.0
        
        # Calculate similarity to ideal diagonal
        attention_norm = attention / (np.sum(attention, axis=1, keepdims=True) + 1e-8)
        alignment_score = np.sum(attention_norm * ideal_attention)
        
        return alignment_score
    
    def _compute_attention_entropy(self, attention_weights):
        """Compute attention entropy (measure of attention sharpness)"""
        if len(attention_weights.shape) == 3:
            attention = attention_weights[0]
        else:
            attention = attention_weights
        
        # Normalize attention
        attention_norm = attention / (np.sum(attention, axis=1, keepdims=True) + 1e-8)
        
        # Compute entropy for each decoder step
        entropies = []
        for i in range(attention_norm.shape[0]):
            entropy = -np.sum(attention_norm[i] * np.log(attention_norm[i] + 1e-8))
            entropies.append(entropy)
        
        return np.mean(entropies)
    
    def _prosody_similarity(self, audio, reference_audio):
        """Compare prosodic features between generated and reference audio"""
        metrics = {}
        
        # F0 comparison
        try:
            f0_gen = librosa.yin(audio, fmin=80, fmax=400, sr=self.sample_rate)
            f0_ref = librosa.yin(reference_audio, fmin=80, fmax=400, sr=self.sample_rate)
            
            f0_gen_valid = f0_gen[~np.isnan(f0_gen)]
            f0_ref_valid = f0_ref[~np.isnan(f0_ref)]
            
            if len(f0_gen_valid) > 0 and len(f0_ref_valid) > 0:
                metrics['f0_similarity'] = 1 - np.abs(np.mean(f0_gen_valid) - np.mean(f0_ref_valid)) / np.mean(f0_ref_valid)
        except:
            metrics['f0_similarity'] = 0.0
        
        # Energy comparison
        energy_gen = librosa.feature.rms(y=audio)[0]
        energy_ref = librosa.feature.rms(y=reference_audio)[0]
        
        energy_corr = np.corrcoef(energy_gen, energy_ref)[0, 1]
        metrics['energy_correlation'] = energy_corr if not np.isnan(energy_corr) else 0.0
        
        return metrics

class IntelligibilityMetrics:
    """Metrics for evaluating speech intelligibility"""
    
    def __init__(self, sample_rate=22050):
        self.sample_rate = sample_rate
    
    def evaluate_intelligibility(self, audio, reference_audio=None):
        """Evaluate speech intelligibility"""
        metrics = {}
        
        # Short-Time Objective Intelligibility (STOI)
        if reference_audio is not None:
            try:
                stoi_score = pystoi.stoi(reference_audio, audio, self.sample_rate, extended=False)
                metrics['stoi'] = stoi_score
            except:
                metrics['stoi'] = 0.0
        
        # Perceptual Evaluation of Speech Quality (PESQ)
        if reference_audio is not None:
            try:
                pesq_score = pesq.pesq(self.sample_rate, reference_audio, audio, 'wb')
                metrics['pesq'] = pesq_score
            except:
                metrics['pesq'] = 0.0
        
        # Articulation-based metrics
        metrics.update(self._articulation_metrics(audio))
        
        return metrics
    
    def _articulation_metrics(self, audio):
        """Calculate articulation-based intelligibility metrics"""
        metrics = {}
        
        # Modulation depth (simplified)
        envelope = np.abs(signal.hilbert(audio))
        envelope_smooth = signal.savgol_filter(envelope, 101, 3)
        
        if np.max(envelope_smooth) > 0:
            modulation_depth = (np.max(envelope_smooth) - np.min(envelope_smooth)) / np.max(envelope_smooth)
            metrics['modulation_depth'] = modulation_depth
        else:
            metrics['modulation_depth'] = 0.0
        
        # Spectral tilt (measure of high-frequency content)
        freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)
        fft = np.fft.fft(audio)
        magnitude = np.abs(fft)
        
        low_freq_energy = np.sum(magnitude[(freqs >= 0) & (freqs <= 1000)])
        high_freq_energy = np.sum(magnitude[(freqs >= 2000) & (freqs <= 8000)])
        
        if low_freq_energy > 0:
            spectral_tilt = high_freq_energy / low_freq_energy
            metrics['spectral_tilt'] = spectral_tilt
        else:
            metrics['spectral_tilt'] = 0.0
        
        return metrics

def calculate_model_metrics(model, dataloader, device='cpu'):
    """Calculate comprehensive model performance metrics"""
    model.eval()
    metrics = {
        'mel_mse': [],
        'mel_mae': [],
        'stop_accuracy': [],
        'attention_scores': []
    }
    
    tts_metrics = TTSMetrics()
    
    with torch.no_grad():
        for batch in dataloader:
            text_inputs = batch['text'].to(device)
            mel_targets = batch['mel'].to(device)
            stop_targets = batch['stop_tokens'].to(device)
            
            # Forward pass
            outputs = model(text_inputs, mel_targets)
            
            # Calculate metrics
            pred_mel = outputs['mel_outputs_postnet'].cpu().numpy()
            target_mel = mel_targets.cpu().numpy()
            
            for i in range(pred_mel.shape[0]):
                mel_metrics = tts_metrics.evaluate_mel_spectrogram(pred_mel[i], target_mel[i])
                metrics['mel_mse'].append(mel_metrics['mel_mse'])
                metrics['mel_mae'].append(mel_metrics['mel_mae'])
            
            # Stop token accuracy
            stop_pred = torch.sigmoid(outputs['stop_tokens']) > 0.5
            stop_acc = (stop_pred.cpu() == stop_targets.cpu()).float().mean()
            metrics['stop_accuracy'].append(stop_acc.item())
            
            # Attention alignment
            attention = outputs['attention_weights'].cpu().numpy()
            for i in range(attention.shape[0]):
                alignment_score = tts_metrics._compute_alignment_score(attention[i])
                metrics['attention_scores'].append(alignment_score)
    
    # Average metrics
    final_metrics = {}
    for key, values in metrics.items():
        final_metrics[key] = np.mean(values)
        final_metrics[f'{key}_std'] = np.std(values)
    
    return final_metrics
