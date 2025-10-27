import librosa
import librosa.display
import numpy as np
import torch
import soundfile as sf
from scipy import signal
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class AudioProcessor:
    """Audio preprocessing pipeline for TTS training and inference"""
    
    def __init__(self, sample_rate=22050, n_fft=1024, hop_length=256, win_length=1024,
                 n_mels=80, fmin=0, fmax=8000, preemphasis=0.97, min_level_db=-100, ref_level_db=20):
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.preemphasis = preemphasis
        self.min_level_db = min_level_db
        self.ref_level_db = ref_level_db
        
        # Create mel filterbank
        self.mel_basis = librosa.filters.mel(
            sr=sample_rate, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
        )
        
        # Normalization parameters
        self.max_norm = 4.0  # Maximum normalization value
        self.clip_norm = True  # Whether to clip normalized values
        
    def load_audio(self, file_path, trim_silence=True):
        """Load audio file and preprocess"""
        try:
            # Load audio
            audio, sr = librosa.load(file_path, sr=self.sample_rate)
            
            # Trim silence
            if trim_silence:
                audio, _ = librosa.effects.trim(audio, top_db=20)
            
            # Apply preemphasis
            if self.preemphasis > 0:
                audio = self._apply_preemphasis(audio)
            
            return audio
        
        except Exception as e:
            raise ValueError(f"Error loading audio file {file_path}: {str(e)}")
    
    def save_audio(self, audio, file_path):
        """Save audio to file"""
        try:
            # Remove preemphasis if applied
            if self.preemphasis > 0:
                audio = self._remove_preemphasis(audio)
            
            # Ensure audio is in valid range
            audio = np.clip(audio, -1.0, 1.0)
            
            sf.write(file_path, audio, self.sample_rate)
        
        except Exception as e:
            raise ValueError(f"Error saving audio file {file_path}: {str(e)}")
    
    def _apply_preemphasis(self, audio):
        """Apply preemphasis filter"""
        return signal.lfilter([1, -self.preemphasis], [1], audio)
    
    def _remove_preemphasis(self, audio):
        """Remove preemphasis filter"""
        return signal.lfilter([1], [1, -self.preemphasis], audio)
    
    def compute_stft(self, audio):
        """Compute Short-Time Fourier Transform"""
        return librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window='hann',
            center=True,
            pad_mode='constant'
        )
    
    def compute_magnitude_spectrogram(self, audio):
        """Compute magnitude spectrogram"""
        stft = self.compute_stft(audio)
        magnitude = np.abs(stft)
        return magnitude
    
    def compute_mel_spectrogram(self, audio):
        """Compute mel spectrogram from audio"""
        # Compute magnitude spectrogram
        magnitude = self.compute_magnitude_spectrogram(audio)
        
        # Convert to mel scale
        mel_spec = np.dot(self.mel_basis, magnitude)
        
        # Convert to dB
        mel_spec = self._amp_to_db(mel_spec)
        
        # Normalize
        mel_spec = self._normalize_db(mel_spec)
        
        return mel_spec
    
    def mel_to_audio(self, mel_spec, vocoder=None):
        """Convert mel spectrogram back to audio"""
        if vocoder is not None:
            # Use neural vocoder
            mel_tensor = torch.tensor(mel_spec).unsqueeze(0)
            with torch.no_grad():
                audio = vocoder(mel_tensor)
            return audio.squeeze().numpy()
        else:
            # Use Griffin-Lim algorithm
            return self._mel_to_audio_griffin_lim(mel_spec)
    
    def _mel_to_audio_griffin_lim(self, mel_spec, n_iter=50):
        """Convert mel spectrogram to audio using Griffin-Lim algorithm"""
        # Denormalize
        mel_spec = self._denormalize_db(mel_spec)
        
        # Convert from dB
        mel_spec = self._db_to_amp(mel_spec)
        
        # Convert from mel to linear scale
        linear_spec = np.dot(self.mel_basis.T, mel_spec)
        
        # Apply Griffin-Lim
        audio = librosa.griffinlim(
            linear_spec,
            n_iter=n_iter,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window='hann',
            center=True,
            length=None
        )
        
        return audio
    
    def _amp_to_db(self, amplitude):
        """Convert amplitude to decibels"""
        return 20 * np.log10(np.maximum(1e-5, amplitude))
    
    def _db_to_amp(self, db):
        """Convert decibels to amplitude"""
        return np.power(10.0, db * 0.05)
    
    def _normalize_db(self, db_spec):
        """Normalize dB spectrogram"""
        normalized = (db_spec - self.ref_level_db - self.min_level_db) / (-self.min_level_db)
        
        if self.clip_norm:
            normalized = np.clip(normalized, 0, self.max_norm)
        
        return normalized
    
    def _denormalize_db(self, normalized_spec):
        """Denormalize dB spectrogram"""
        return (normalized_spec * (-self.min_level_db)) + self.ref_level_db + self.min_level_db
    
    def extract_f0(self, audio, method='pyin'):
        """Extract fundamental frequency (F0) from audio"""
        if method == 'pyin':
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7'),
                sr=self.sample_rate,
                hop_length=self.hop_length
            )
        elif method == 'harvest':
            # Alternative method (requires pyworld)
            try:
                import pyworld as pw
                f0, _ = pw.harvest(
                    audio.astype(np.float64),
                    self.sample_rate,
                    frame_period=self.hop_length / self.sample_rate * 1000
                )
            except ImportError:
                raise ImportError("pyworld is required for harvest F0 extraction")
        else:
            raise ValueError(f"Unknown F0 extraction method: {method}")
        
        # Handle NaN values
        f0 = np.nan_to_num(f0, nan=0.0)
        
        return f0
    
    def extract_energy(self, audio):
        """Extract energy (RMS) from audio"""
        # Compute frame-wise RMS energy
        energy = librosa.feature.rms(
            y=audio,
            frame_length=self.win_length,
            hop_length=self.hop_length,
            center=True
        )[0]
        
        # Convert to dB
        energy_db = 20 * np.log10(np.maximum(1e-5, energy))
        
        return energy_db
    
    def time_stretch(self, audio, rate):
        """Time-stretch audio by a given factor"""
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def pitch_shift(self, audio, n_steps):
        """Pitch-shift audio by n semitones"""
        return librosa.effects.pitch_shift(audio, sr=self.sample_rate, n_steps=n_steps)
    
    def add_noise(self, audio, noise_factor=0.005):
        """Add random noise to audio for data augmentation"""
        noise = np.random.randn(len(audio))
        return audio + noise_factor * noise
    
    def normalize_audio(self, audio, target_db=-20):
        """Normalize audio to target dB level"""
        # Calculate current RMS
        rms = np.sqrt(np.mean(audio**2))
        
        # Calculate target RMS
        target_rms = 10**(target_db / 20)
        
        # Apply normalization
        if rms > 0:
            audio = audio * (target_rms / rms)
        
        return audio
    
    def extract_audio_features(self, audio):
        """Extract comprehensive audio features"""
        features = {}
        
        # Basic statistics
        features['duration'] = len(audio) / self.sample_rate
        features['rms'] = np.sqrt(np.mean(audio**2))
        features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(audio))
        
        # Spectral features
        stft = self.compute_stft(audio)
        magnitude = np.abs(stft)
        
        features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(S=magnitude, sr=self.sample_rate))
        features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(S=magnitude, sr=self.sample_rate))
        features['spectral_bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(S=magnitude, sr=self.sample_rate))
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfccs, axis=1)
        features['mfcc_std'] = np.std(mfccs, axis=1)
        
        # Fundamental frequency
        try:
            f0 = self.extract_f0(audio)
            features['f0_mean'] = np.mean(f0[f0 > 0])
            features['f0_std'] = np.std(f0[f0 > 0])
        except:
            features['f0_mean'] = 0.0
            features['f0_std'] = 0.0
        
        return features
    
    def plot_spectrogram(self, spectrogram, title="Spectrogram", sr=None):
        """Plot spectrogram"""
        if sr is None:
            sr = self.sample_rate
        
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(
            spectrogram,
            sr=sr,
            hop_length=self.hop_length,
            x_axis='time',
            y_axis='mel'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        return plt.gcf()
    
    def align_sequences(self, mel_spec, text_length):
        """Align mel spectrogram with text sequence"""
        mel_length = mel_spec.shape[1]
        
        if mel_length != text_length:
            # Simple linear interpolation for alignment
            x_old = np.linspace(0, 1, mel_length)
            x_new = np.linspace(0, 1, text_length)
            
            aligned_mel = np.zeros((mel_spec.shape[0], text_length))
            for i in range(mel_spec.shape[0]):
                f = interp1d(x_old, mel_spec[i], kind='linear', fill_value='extrapolate')
                aligned_mel[i] = f(x_new)
            
            return aligned_mel
        
        return mel_spec
