import numpy as np
import soundfile as sf
import librosa
import torch
import io
from pathlib import Path
from typing import Tuple, Optional, Union, List
import warnings

def load_audio(file_path: Union[str, Path], 
               sample_rate: Optional[int] = None,
               normalize: bool = True) -> Tuple[np.ndarray, int]:
    """
    Load audio file with optional resampling and normalization
    
    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate (None to keep original)
        normalize: Whether to normalize audio to [-1, 1]
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    try:
        # Load audio file
        audio, sr = librosa.load(str(file_path), sr=sample_rate, mono=True)
        
        # Normalize if requested
        if normalize:
            audio = normalize_audio(audio)
        
        return audio, sr
    
    except Exception as e:
        raise ValueError(f"Error loading audio file {file_path}: {str(e)}")

def save_audio(audio: np.ndarray, 
               sample_rate: int,
               file_path: Optional[Union[str, Path]] = None,
               format: str = 'wav',
               normalize: bool = True) -> Union[bytes, None]:
    """
    Save audio to file or return as bytes
    
    Args:
        audio: Audio data
        sample_rate: Sample rate
        file_path: Output file path (None to return bytes)
        format: Audio format ('wav', 'mp3', 'flac')
        normalize: Whether to normalize audio
        
    Returns:
        None if file_path provided, bytes otherwise
    """
    if normalize:
        audio = normalize_audio(audio)
    
    # Ensure audio is in valid range
    audio = np.clip(audio, -1.0, 1.0)
    
    if file_path is not None:
        # Save to file
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        sf.write(str(file_path), audio, sample_rate, format=format.upper())
        return None
    else:
        # Return as bytes
        buffer = io.BytesIO()
        sf.write(buffer, audio, sample_rate, format=format.upper())
        return buffer.getvalue()

def normalize_audio(audio: np.ndarray, 
                   target_db: Optional[float] = None,
                   method: str = 'peak') -> np.ndarray:
    """
    Normalize audio signal
    
    Args:
        audio: Input audio signal
        target_db: Target level in dB (None for peak normalization)
        method: Normalization method ('peak', 'rms')
        
    Returns:
        Normalized audio signal
    """
    if len(audio) == 0:
        return audio
    
    if method == 'peak':
        # Peak normalization
        peak = np.max(np.abs(audio))
        if peak > 0:
            if target_db is not None:
                target_linear = 10**(target_db / 20)
                audio = audio * (target_linear / peak)
            else:
                audio = audio / peak * 0.95  # Leave some headroom
    
    elif method == 'rms':
        # RMS normalization
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            if target_db is not None:
                target_rms = 10**(target_db / 20)
                audio = audio * (target_rms / rms)
            else:
                audio = audio / rms * 0.1  # Conservative RMS level
    
    return audio

def trim_silence(audio: np.ndarray, 
                sample_rate: int,
                top_db: float = 20,
                frame_length: int = 2048,
                hop_length: int = 512) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Trim silence from beginning and end of audio
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        top_db: dB threshold for silence detection
        frame_length: Frame length for analysis
        hop_length: Hop length for analysis
        
    Returns:
        Tuple of (trimmed_audio, (start_sample, end_sample))
    """
    trimmed, (start_frame, end_frame) = librosa.effects.trim(
        audio, 
        top_db=top_db,
        frame_length=frame_length,
        hop_length=hop_length
    )
    
    start_sample = start_frame * hop_length
    end_sample = end_frame * hop_length
    
    return trimmed, (start_sample, end_sample)

def add_noise(audio: np.ndarray, 
              noise_factor: float = 0.005,
              noise_type: str = 'gaussian') -> np.ndarray:
    """
    Add noise to audio for data augmentation
    
    Args:
        audio: Input audio signal
        noise_factor: Noise level (standard deviation for Gaussian)
        noise_type: Type of noise ('gaussian', 'uniform')
        
    Returns:
        Audio with added noise
    """
    if noise_type == 'gaussian':
        noise = np.random.normal(0, noise_factor, audio.shape)
    elif noise_type == 'uniform':
        noise = np.random.uniform(-noise_factor, noise_factor, audio.shape)
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    return audio + noise

def change_speed(audio: np.ndarray, 
                speed_factor: float) -> np.ndarray:
    """
    Change audio speed without changing pitch
    
    Args:
        audio: Input audio signal
        speed_factor: Speed change factor (>1 faster, <1 slower)
        
    Returns:
        Speed-changed audio
    """
    return librosa.effects.time_stretch(audio, rate=speed_factor)

def change_pitch(audio: np.ndarray, 
                sample_rate: int,
                n_steps: float) -> np.ndarray:
    """
    Change audio pitch without changing speed
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        n_steps: Pitch change in semitones (positive = higher, negative = lower)
        
    Returns:
        Pitch-shifted audio
    """
    return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=n_steps)

def apply_fade(audio: np.ndarray, 
               sample_rate: int,
               fade_in_duration: float = 0.01,
               fade_out_duration: float = 0.01) -> np.ndarray:
    """
    Apply fade in/out to audio to reduce clicks
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        fade_in_duration: Fade in duration in seconds
        fade_out_duration: Fade out duration in seconds
        
    Returns:
        Audio with fade applied
    """
    audio = audio.copy()
    
    # Calculate fade lengths in samples
    fade_in_samples = int(fade_in_duration * sample_rate)
    fade_out_samples = int(fade_out_duration * sample_rate)
    
    # Apply fade in
    if fade_in_samples > 0 and fade_in_samples < len(audio):
        fade_in_curve = np.linspace(0, 1, fade_in_samples)
        audio[:fade_in_samples] *= fade_in_curve
    
    # Apply fade out
    if fade_out_samples > 0 and fade_out_samples < len(audio):
        fade_out_curve = np.linspace(1, 0, fade_out_samples)
        audio[-fade_out_samples:] *= fade_out_curve
    
    return audio

def concatenate_audio(audio_list: List[np.ndarray], 
                     crossfade_duration: float = 0.01,
                     sample_rate: int = 22050) -> np.ndarray:
    """
    Concatenate multiple audio segments with optional crossfading
    
    Args:
        audio_list: List of audio arrays
        crossfade_duration: Crossfade duration in seconds
        sample_rate: Sample rate
        
    Returns:
        Concatenated audio
    """
    if len(audio_list) == 0:
        return np.array([])
    
    if len(audio_list) == 1:
        return audio_list[0]
    
    crossfade_samples = int(crossfade_duration * sample_rate)
    
    # Start with first audio segment
    result = audio_list[0].copy()
    
    for audio in audio_list[1:]:
        if crossfade_samples > 0 and len(result) >= crossfade_samples and len(audio) >= crossfade_samples:
            # Apply crossfade
            fade_out = np.linspace(1, 0, crossfade_samples)
            fade_in = np.linspace(0, 1, crossfade_samples)
            
            # Overlap the segments
            overlap_region = (result[-crossfade_samples:] * fade_out + 
                            audio[:crossfade_samples] * fade_in)
            
            # Concatenate with overlap
            result = np.concatenate([result[:-crossfade_samples], 
                                   overlap_region, 
                                   audio[crossfade_samples:]])
        else:
            # Simple concatenation without crossfade
            result = np.concatenate([result, audio])
    
    return result

def detect_voice_activity(audio: np.ndarray, 
                         sample_rate: int,
                         frame_length: int = 2048,
                         hop_length: int = 512,
                         energy_threshold: float = 0.01) -> np.ndarray:
    """
    Detect voice activity in audio
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        frame_length: Frame length for analysis
        hop_length: Hop length for analysis
        energy_threshold: Energy threshold for voice detection
        
    Returns:
        Boolean array indicating voice activity per frame
    """
    # Compute frame-wise energy
    energy = librosa.feature.rms(
        y=audio, 
        frame_length=frame_length, 
        hop_length=hop_length
    )[0]
    
    # Simple energy-based VAD
    voice_activity = energy > energy_threshold
    
    # Apply some smoothing to reduce false positives
    from scipy import ndimage
    voice_activity = ndimage.binary_opening(voice_activity, structure=np.ones(3))
    voice_activity = ndimage.binary_closing(voice_activity, structure=np.ones(3))
    
    return voice_activity

def convert_sample_rate(audio: np.ndarray, 
                       orig_sr: int, 
                       target_sr: int) -> np.ndarray:
    """
    Convert audio sample rate
    
    Args:
        audio: Input audio signal
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled audio
    """
    if orig_sr == target_sr:
        return audio
    
    return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)

def audio_to_tensor(audio: np.ndarray, 
                   dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Convert numpy audio to PyTorch tensor
    
    Args:
        audio: Input audio array
        dtype: Target tensor dtype
        
    Returns:
        Audio tensor
    """
    return torch.tensor(audio, dtype=dtype)

def tensor_to_audio(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert PyTorch tensor to numpy audio
    
    Args:
        tensor: Input audio tensor
        
    Returns:
        Audio array
    """
    if tensor.requires_grad:
        tensor = tensor.detach()
    
    return tensor.cpu().numpy()

def calculate_audio_stats(audio: np.ndarray, 
                         sample_rate: int) -> dict:
    """
    Calculate comprehensive audio statistics
    
    Args:
        audio: Input audio signal
        sample_rate: Sample rate
        
    Returns:
        Dictionary with audio statistics
    """
    stats = {}
    
    # Basic stats
    stats['duration'] = len(audio) / sample_rate
    stats['sample_rate'] = sample_rate
    stats['num_samples'] = len(audio)
    
    # Amplitude stats
    stats['max_amplitude'] = np.max(np.abs(audio))
    stats['rms'] = np.sqrt(np.mean(audio**2))
    stats['mean'] = np.mean(audio)
    stats['std'] = np.std(audio)
    
    # Dynamic range
    stats['dynamic_range'] = np.max(audio) - np.min(audio)
    
    # Zero crossings
    stats['zero_crossings'] = np.sum(np.diff(np.signbit(audio)))
    stats['zero_crossing_rate'] = stats['zero_crossings'] / len(audio)
    
    # Spectral features
    try:
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0]
        stats['spectral_centroid_mean'] = np.mean(spectral_centroids)
        stats['spectral_centroid_std'] = np.std(spectral_centroids)
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0]
        stats['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate)[0]
        stats['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        
    except Exception as e:
        # If spectral analysis fails, set defaults
        stats['spectral_centroid_mean'] = 0.0
        stats['spectral_centroid_std'] = 0.0
        stats['spectral_rolloff_mean'] = 0.0
        stats['spectral_bandwidth_mean'] = 0.0
    
    return stats

def validate_audio(audio: np.ndarray, 
                  sample_rate: int,
                  min_duration: float = 0.1,
                  max_duration: float = 30.0) -> Tuple[bool, str]:
    """
    Validate audio data
    
    Args:
        audio: Audio signal to validate
        sample_rate: Sample rate
        min_duration: Minimum duration in seconds
        max_duration: Maximum duration in seconds
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check if audio is empty
    if len(audio) == 0:
        return False, "Audio is empty"
    
    # Check duration
    duration = len(audio) / sample_rate
    if duration < min_duration:
        return False, f"Audio too short: {duration:.2f}s < {min_duration}s"
    
    if duration > max_duration:
        return False, f"Audio too long: {duration:.2f}s > {max_duration}s"
    
    # Check for NaN or infinite values
    if np.any(np.isnan(audio)):
        return False, "Audio contains NaN values"
    
    if np.any(np.isinf(audio)):
        return False, "Audio contains infinite values"
    
    # Check amplitude range
    max_amplitude = np.max(np.abs(audio))
    if max_amplitude == 0:
        return False, "Audio is silent (all zeros)"
    
    if max_amplitude > 10:  # Suspiciously high amplitude
        return False, f"Audio amplitude too high: {max_amplitude}"
    
    # Check sample rate
    if sample_rate <= 0:
        return False, f"Invalid sample rate: {sample_rate}"
    
    return True, "Audio is valid"
