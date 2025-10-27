import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import json
import random
from typing import List, Dict, Tuple, Optional

from preprocessing.text_processor import TextProcessor
from preprocessing.audio_processor import AudioProcessor

class TTSDataset(Dataset):
    """Dataset class for TTS training"""
    
    def __init__(self, data_dir: str, metadata_file: str = None, 
                 text_processor: TextProcessor = None,
                 audio_processor: AudioProcessor = None,
                 max_text_length: int = 200,
                 max_mel_length: int = 1000,
                 use_phonemes: bool = False):
        
        self.data_dir = Path(data_dir)
        self.max_text_length = max_text_length
        self.max_mel_length = max_mel_length
        self.use_phonemes = use_phonemes
        
        # Initialize processors
        self.text_processor = text_processor or TextProcessor()
        self.audio_processor = audio_processor or AudioProcessor()
        
        # Load metadata
        self.metadata = self._load_metadata(metadata_file)
        
        # Filter by length constraints
        self.metadata = self._filter_by_length()
        
        print(f"Loaded {len(self.metadata)} samples")
    
    def _load_metadata(self, metadata_file: str) -> List[Dict]:
        """Load dataset metadata"""
        if metadata_file is None:
            # Create synthetic metadata for demo
            return self._create_synthetic_metadata()
        
        metadata_path = self.data_dir / metadata_file
        
        if metadata_path.suffix == '.csv':
            # LJSpeech format: filename|text|normalized_text (no header)
            df = pd.read_csv(metadata_path, sep='|', header=None,
                           names=['filename', 'text', 'normalized_text'],
                           dtype=str)  # Force all columns to string to avoid float parsing
            # Use normalized text if available, otherwise use original text
            df['text'] = df['normalized_text'].fillna(df['text'])
            df['filename'] = df['filename'] + '.wav'  # Add .wav extension

            # Drop any rows with missing text
            df = df.dropna(subset=['text'])

            metadata = df[['filename', 'text']].to_dict('records')
        elif metadata_path.suffix == '.json':
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            # Assume it's a text file with format: filename|text
            metadata = []
            with open(metadata_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('|')
                    if len(parts) >= 2:
                        metadata.append({
                            'filename': parts[0],
                            'text': '|'.join(parts[1:])
                        })
        
        return metadata
    
    def _create_synthetic_metadata(self) -> List[Dict]:
        """Create synthetic metadata for demo purposes"""
        sample_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Speech synthesis is the artificial production of human speech.",
            "Deep learning has revolutionized text to speech systems.",
            "Natural language processing enables computers to understand text.",
            "Machine learning models can generate realistic human voices.",
            "Artificial intelligence continues to advance rapidly.",
            "Voice assistants use speech synthesis technology.",
            "Neural networks can learn complex patterns in data.",
            "Text preprocessing is crucial for good speech quality.",
            "Mel spectrograms represent audio in the frequency domain."
        ]
        
        metadata = []
        for i, text in enumerate(sample_texts):
            metadata.append({
                'filename': f'sample_{i:03d}.wav',
                'text': text,
                'duration': len(text) * 0.05  # Approximate duration
            })
        
        return metadata
    
    def _filter_by_length(self) -> List[Dict]:
        """Filter samples by length constraints"""
        filtered = []
        
        for item in self.metadata:
            text = item['text']
            text_length = len(self.text_processor.text_to_sequence(text, self.use_phonemes))
            
            # Estimate mel length (rough approximation)
            duration = item.get('duration', len(text) * 0.05)
            mel_length = int(duration * self.audio_processor.sample_rate / self.audio_processor.hop_length)
            
            # Filter by length
            if text_length <= self.max_text_length and mel_length <= self.max_mel_length:
                item['text_length'] = text_length
                item['mel_length'] = mel_length
                filtered.append(item)
        
        return filtered
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample"""
        item = self.metadata[idx]
        
        # Process text
        text = item['text']
        text_sequence = self.text_processor.text_to_sequence(text, self.use_phonemes)
        
        # Load and process audio (or create synthetic for demo)
        # Try wavs subdirectory first (for LJSpeech), then try data_dir directly
        audio_path = self.data_dir / "wavs" / item['filename']
        if not audio_path.exists():
            audio_path = self.data_dir / item['filename']

        if audio_path.exists():
            audio = self.audio_processor.load_audio(str(audio_path))
            mel_spec = self.audio_processor.compute_mel_spectrogram(audio)
        else:
            # Create synthetic mel spectrogram for demo
            mel_spec = self._create_synthetic_mel(item['mel_length'])
        
        # Create stop tokens (1 for end of sequence, 0 otherwise)
        mel_length = mel_spec.shape[1]
        stop_tokens = np.zeros(mel_length)
        stop_tokens[-1] = 1.0
        
        # Pad sequences
        text_padded = self._pad_sequence(text_sequence, self.max_text_length)
        mel_padded = self._pad_mel(mel_spec, self.max_mel_length)
        stop_padded = self._pad_sequence(stop_tokens, self.max_mel_length)

        # Transpose mel to (time, mel_channels) for decoder compatibility
        mel_padded = mel_padded.T  # (mel_channels, time) -> (time, mel_channels)

        return {
            'text': torch.tensor(text_padded, dtype=torch.long),
            'mel': torch.tensor(mel_padded, dtype=torch.float32),
            'stop_tokens': torch.tensor(stop_padded, dtype=torch.float32),
            'text_length': len(text_sequence),
            'mel_length': mel_length,
            'filename': item['filename']
        }
    
    def _create_synthetic_mel(self, length: int) -> np.ndarray:
        """Create synthetic mel spectrogram for demo"""
        # Generate random mel spectrogram with some structure
        mel_spec = np.random.randn(self.audio_processor.n_mels, length) * 0.5
        
        # Add some harmonic structure
        for i in range(self.audio_processor.n_mels):
            freq_component = np.sin(2 * np.pi * i * np.arange(length) / length)
            mel_spec[i] += freq_component * 0.3
        
        # Normalize
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)
        
        return mel_spec
    
    def _pad_sequence(self, sequence, max_length: int):
        """Pad sequence to max length"""
        # Convert to numpy array if it's a list
        if isinstance(sequence, list):
            sequence = np.array(sequence)

        if len(sequence) >= max_length:
            return sequence[:max_length]
        else:
            # Pad with zeros
            padding = np.zeros(max_length - len(sequence))
            return np.concatenate([sequence, padding])
    
    def _pad_mel(self, mel_spec: np.ndarray, max_length: int) -> np.ndarray:
        """Pad mel spectrogram to max length"""
        if mel_spec.shape[1] >= max_length:
            return mel_spec[:, :max_length]
        else:
            padding = np.zeros((mel_spec.shape[0], max_length - mel_spec.shape[1]))
            return np.concatenate([mel_spec, padding], axis=1)
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Custom collate function for batching"""
        # Sort batch by text length (for more efficient packing)
        batch = sorted(batch, key=lambda x: x['text_length'], reverse=True)
        
        # Stack tensors
        text_batch = torch.stack([item['text'] for item in batch])
        mel_batch = torch.stack([item['mel'] for item in batch])
        stop_batch = torch.stack([item['stop_tokens'] for item in batch])
        
        # Get lengths
        text_lengths = torch.tensor([item['text_length'] for item in batch])
        mel_lengths = torch.tensor([item['mel_length'] for item in batch])
        
        return {
            'text': text_batch,
            'mel': mel_batch,
            'stop_tokens': stop_batch,
            'text_lengths': text_lengths,
            'mel_lengths': mel_lengths,
            'filenames': [item['filename'] for item in batch]
        }

class LJSpeechDataset(TTSDataset):
    """LJSpeech dataset implementation"""
    
    def __init__(self, data_dir: str, **kwargs):
        super().__init__(data_dir, metadata_file='metadata.csv', **kwargs)
    
    def _load_metadata(self, metadata_file: str) -> List[Dict]:
        """Load LJSpeech metadata"""
        metadata_path = self.data_dir / metadata_file
        metadata = []
        
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('|')
                    if len(parts) >= 3:
                        metadata.append({
                            'filename': f"wavs/{parts[0]}.wav",
                            'text': parts[2],  # Use normalized text
                            'raw_text': parts[1]
                        })
        else:
            # Fallback to synthetic data
            return self._create_synthetic_metadata()
        
        return metadata

class CustomDataset(TTSDataset):
    """Custom dataset for user-provided data"""
    
    def __init__(self, audio_files: List[str], texts: List[str], **kwargs):
        self.audio_files = audio_files
        self.texts = texts
        
        # Create metadata from provided lists
        metadata = []
        for audio_file, text in zip(audio_files, texts):
            metadata.append({
                'filename': audio_file,
                'text': text,
                'duration': 5.0  # Default duration
            })
        
        # Initialize without calling parent __init__
        self.data_dir = Path('.')
        self.max_text_length = kwargs.get('max_text_length', 200)
        self.max_mel_length = kwargs.get('max_mel_length', 1000)
        self.use_phonemes = kwargs.get('use_phonemes', False)
        
        self.text_processor = kwargs.get('text_processor') or TextProcessor()
        self.audio_processor = kwargs.get('audio_processor') or AudioProcessor()
        
        self.metadata = self._filter_by_length_custom(metadata)

def create_data_loader(dataset: Dataset, batch_size: int = 16, 
                      shuffle: bool = True, num_workers: int = 0) -> DataLoader:
    """Create a DataLoader for the dataset"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=torch.cuda.is_available(),
        drop_last=True
    )

def get_dataset_statistics(dataset: TTSDataset) -> Dict:
    """Calculate dataset statistics"""
    text_lengths = []
    mel_lengths = []
    
    for item in dataset.metadata:
        text_lengths.append(item.get('text_length', 0))
        mel_lengths.append(item.get('mel_length', 0))
    
    stats = {
        'num_samples': len(dataset),
        'text_length_mean': np.mean(text_lengths),
        'text_length_std': np.std(text_lengths),
        'text_length_min': np.min(text_lengths),
        'text_length_max': np.max(text_lengths),
        'mel_length_mean': np.mean(mel_lengths),
        'mel_length_std': np.std(mel_lengths),
        'mel_length_min': np.min(mel_lengths),
        'mel_length_max': np.max(mel_lengths),
        'vocab_size': dataset.text_processor.vocab_size
    }
    
    return stats
