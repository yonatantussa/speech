import torch
import numpy as np
import soundfile as sf
from pathlib import Path
import time

from preprocessing.text_processor import TextProcessor
from preprocessing.audio_processor import AudioProcessor
from models.vocoder import SimpleVocoder, MelGAN

class TTSSynthesizer:
    """Text-to-Speech synthesis engine"""
    
    def __init__(self, tts_model, vocoder=None, text_processor=None, audio_processor=None):
        self.tts_model = tts_model
        self.vocoder = vocoder or SimpleVocoder()
        self.text_processor = text_processor or TextProcessor()
        self.audio_processor = audio_processor or AudioProcessor()
        
        # Set models to evaluation mode
        self.tts_model.eval()
        if self.vocoder is not None:
            self.vocoder.eval()
        
        # Device
        self.device = next(self.tts_model.parameters()).device
        if self.vocoder is not None:
            self.vocoder.to(self.device)
        
        # Synthesis parameters
        self.default_speed = 1.0
        self.default_pitch = 0.0
        self.default_energy = 1.0
    
    def synthesize(self, text, speed=None, pitch=None, energy=None, 
                   use_phonemes=False, max_decoder_steps=1000):
        """
        Synthesize speech from text
        
        Args:
            text: Input text string
            speed: Speech speed factor (1.0 = normal)
            pitch: Pitch shift in semitones (0.0 = no change)
            energy: Energy scaling factor (1.0 = normal)
            use_phonemes: Whether to use phoneme conversion
            max_decoder_steps: Maximum decoder steps for synthesis
            
        Returns:
            audio: Generated audio waveform
            sample_rate: Audio sample rate
            mel_spectrogram: Generated mel spectrogram
        """
        # Set default values
        speed = speed or self.default_speed
        pitch = pitch or self.default_pitch
        energy = energy or self.default_energy
        
        start_time = time.time()
        
        # Preprocess text
        processed_text = self._preprocess_text(text)
        text_sequence = self.text_processor.text_to_sequence(processed_text, use_phonemes)
        
        # Convert to tensor
        text_tensor = torch.tensor([text_sequence], dtype=torch.long).to(self.device)
        
        # Generate mel spectrogram
        with torch.no_grad():
            outputs = self.tts_model.inference(text_tensor, max_len=max_decoder_steps)
            mel_outputs = outputs['mel_outputs_postnet']
        
        # Apply prosody modifications
        mel_outputs = self._apply_prosody_control(mel_outputs, speed, pitch, energy)
        
        # Convert mel to audio
        audio = self._mel_to_audio(mel_outputs)
        
        synthesis_time = time.time() - start_time
        audio_duration = len(audio) / self.audio_processor.sample_rate
        rtf = synthesis_time / audio_duration  # Real-time factor
        
        print(f"Synthesis completed in {synthesis_time:.2f}s (RTF: {rtf:.2f})")
        
        return audio, self.audio_processor.sample_rate, mel_outputs.squeeze().cpu().numpy()
    
    def synthesize_batch(self, texts, **kwargs):
        """Synthesize multiple texts in batch"""
        results = []
        
        for text in texts:
            audio, sr, mel = self.synthesize(text, **kwargs)
            results.append({
                'text': text,
                'audio': audio,
                'sample_rate': sr,
                'mel_spectrogram': mel
            })
        
        return results
    
    def _preprocess_text(self, text):
        """Preprocess text for synthesis"""
        # Apply all text preprocessing steps
        processed = self.text_processor.normalize_text(text)
        processed = self.text_processor.expand_abbreviations(processed)
        processed = self.text_processor.convert_numbers(processed)
        processed = self.text_processor.clean_text(processed)
        
        return processed
    
    def _apply_prosody_control(self, mel_outputs, speed, pitch, energy):
        """Apply prosody control to mel spectrogram"""
        mel_outputs = mel_outputs.clone()
        
        # Speed control (time stretching)
        if speed != 1.0:
            mel_outputs = self._time_stretch_mel(mel_outputs, speed)
        
        # Pitch control (frequency shifting)
        if pitch != 0.0:
            mel_outputs = self._pitch_shift_mel(mel_outputs, pitch)
        
        # Energy control (amplitude scaling)
        if energy != 1.0:
            mel_outputs = mel_outputs * energy
        
        return mel_outputs
    
    def _time_stretch_mel(self, mel_spec, rate):
        """Time-stretch mel spectrogram"""
        # Simple linear interpolation for time stretching
        batch_size, mel_dim, time_steps = mel_spec.shape
        new_time_steps = int(time_steps / rate)
        
        # Create interpolation indices
        old_indices = torch.linspace(0, time_steps - 1, time_steps).to(mel_spec.device)
        new_indices = torch.linspace(0, time_steps - 1, new_time_steps).to(mel_spec.device)
        
        # Interpolate
        stretched_mel = torch.zeros(batch_size, mel_dim, new_time_steps).to(mel_spec.device)
        
        for b in range(batch_size):
            for m in range(mel_dim):
                stretched_mel[b, m] = torch.interp(new_indices, old_indices, mel_spec[b, m])
        
        return stretched_mel
    
    def _pitch_shift_mel(self, mel_spec, semitones):
        """Pitch-shift mel spectrogram"""
        # Simple frequency domain shifting
        # This is a simplified approach - real pitch shifting is more complex
        shift_bins = int(semitones * mel_spec.size(1) / 24)  # Rough approximation
        
        if shift_bins == 0:
            return mel_spec
        
        shifted_mel = torch.zeros_like(mel_spec)
        
        if shift_bins > 0:
            # Shift up
            shifted_mel[:, shift_bins:, :] = mel_spec[:, :-shift_bins, :]
        else:
            # Shift down
            shifted_mel[:, :shift_bins, :] = mel_spec[:, -shift_bins:, :]
        
        return shifted_mel
    
    def _mel_to_audio(self, mel_outputs):
        """Convert mel spectrogram to audio waveform"""
        if self.vocoder is not None:
            # Use neural vocoder
            # Vocoder expects (batch, mel_channels, time)
            # mel_outputs is (batch, time, mel_channels), so transpose
            mel_outputs_transposed = mel_outputs.transpose(1, 2)

            with torch.no_grad():
                audio = self.vocoder(mel_outputs_transposed)
                audio = audio.squeeze().cpu().numpy()
        else:
            # Use Griffin-Lim algorithm
            # mel_outputs is (batch, time, mel_channels), need (mel_channels, time)
            mel_numpy = mel_outputs.squeeze(0).cpu().numpy()  # (time, mel_channels)
            mel_numpy = mel_numpy.T  # (mel_channels, time)
            audio = self.audio_processor.mel_to_audio(mel_numpy)

        # Post-process audio
        audio = self._postprocess_audio(audio)

        return audio
    
    def _postprocess_audio(self, audio):
        """Post-process generated audio"""
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.95
        
        # Apply slight fade in/out to reduce clicks
        fade_samples = int(0.01 * self.audio_processor.sample_rate)  # 10ms fade
        
        if len(audio) > 2 * fade_samples:
            # Fade in
            fade_in = np.linspace(0, 1, fade_samples)
            audio[:fade_samples] *= fade_in
            
            # Fade out
            fade_out = np.linspace(1, 0, fade_samples)
            audio[-fade_samples:] *= fade_out
        
        return audio
    
    def save_audio(self, audio, output_path, sample_rate=None):
        """Save audio to file"""
        sample_rate = sample_rate or self.audio_processor.sample_rate
        
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save audio
        sf.write(output_path, audio, sample_rate)
        
        return output_path
    
    def get_synthesis_info(self, text):
        """Get information about synthesis without actually generating audio"""
        processed_text = self._preprocess_text(text)
        text_sequence = self.text_processor.text_to_sequence(processed_text)
        
        # Estimate synthesis parameters
        estimated_duration = len(text_sequence) * 0.05  # Rough estimate
        estimated_mel_frames = int(estimated_duration * self.audio_processor.sample_rate / self.audio_processor.hop_length)
        
        info = {
            'original_text': text,
            'processed_text': processed_text,
            'text_length': len(text_sequence),
            'estimated_duration': estimated_duration,
            'estimated_mel_frames': estimated_mel_frames,
            'vocab_size': self.text_processor.vocab_size
        }
        
        return info

class RealTimeSynthesizer(TTSSynthesizer):
    """Real-time speech synthesis with streaming capabilities"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chunk_size = kwargs.get('chunk_size', 1000)  # Mel frames per chunk
        self.overlap = kwargs.get('overlap', 100)  # Overlap between chunks
    
    def synthesize_streaming(self, text, chunk_callback=None):
        """Synthesize speech with streaming output"""
        # Preprocess text
        processed_text = self._preprocess_text(text)
        text_sequence = self.text_processor.text_to_sequence(processed_text)
        text_tensor = torch.tensor([text_sequence], dtype=torch.long).to(self.device)
        
        # Generate mel spectrogram in chunks
        full_audio = []
        
        with torch.no_grad():
            outputs = self.tts_model.inference(text_tensor)
            mel_outputs = outputs['mel_outputs_postnet']
            
            # Process in chunks
            mel_length = mel_outputs.size(-1)
            for start in range(0, mel_length, self.chunk_size - self.overlap):
                end = min(start + self.chunk_size, mel_length)
                
                # Extract chunk
                mel_chunk = mel_outputs[:, :, start:end]
                
                # Convert to audio
                audio_chunk = self._mel_to_audio(mel_chunk)
                full_audio.append(audio_chunk)
                
                # Callback for real-time processing
                if chunk_callback:
                    chunk_callback(audio_chunk, self.audio_processor.sample_rate)
        
        # Concatenate all chunks
        full_audio = np.concatenate(full_audio)
        
        return full_audio, self.audio_processor.sample_rate

class BatchSynthesizer(TTSSynthesizer):
    """Efficient batch synthesis for processing multiple texts"""
    
    def synthesize_batch_efficient(self, texts, batch_size=8):
        """Efficiently synthesize multiple texts using batching"""
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_results = self._synthesize_batch_internal(batch_texts)
            results.extend(batch_results)
        
        return results
    
    def _synthesize_batch_internal(self, texts):
        """Internal batch synthesis method"""
        # Preprocess all texts
        sequences = []
        max_length = 0
        
        for text in texts:
            processed_text = self._preprocess_text(text)
            sequence = self.text_processor.text_to_sequence(processed_text)
            sequences.append(sequence)
            max_length = max(max_length, len(sequence))
        
        # Pad sequences
        padded_sequences = []
        for sequence in sequences:
            padded = sequence + [0] * (max_length - len(sequence))
            padded_sequences.append(padded)
        
        # Convert to tensor
        text_tensor = torch.tensor(padded_sequences, dtype=torch.long).to(self.device)
        
        # Generate mel spectrograms
        with torch.no_grad():
            outputs = self.tts_model.inference(text_tensor)
            mel_outputs = outputs['mel_outputs_postnet']
        
        # Convert each mel to audio
        results = []
        for i, text in enumerate(texts):
            mel_spec = mel_outputs[i:i+1]
            audio = self._mel_to_audio(mel_spec)
            
            results.append({
                'text': text,
                'audio': audio,
                'sample_rate': self.audio_processor.sample_rate,
                'mel_spectrogram': mel_spec.squeeze().cpu().numpy()
            })
        
        return results
