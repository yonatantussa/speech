import re
import string
import unicodedata
import os
import nltk
from phonemizer import phonemize
from phonemizer.backend import EspeakBackend
import torch
import numpy as np

# Set espeak path for phonemizer
espeak_bin_path = '/nix/store/02sy4i533rf5zcqal2yblk6mcyfpdsh8-espeak-ng-1.51.1/bin'
current_path = os.environ.get('PATH', '')
if espeak_bin_path not in current_path:
    os.environ['PATH'] = f"{espeak_bin_path}:{current_path}"

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TextProcessor:
    """Text preprocessing pipeline for TTS"""
    
    def __init__(self, language='en-us'):
        self.language = language
        
        # Initialize espeak backend with error handling
        try:
            self.backend = EspeakBackend(language=language)
            self.phonemizer_available = True
        except Exception as e:
            print(f"Warning: Phonemizer backend initialization failed: {e}")
            self.backend = None
            self.phonemizer_available = False
        
        # Character to index mapping
        self._init_char_mappings()
        
        # Abbreviation mappings
        self.abbreviations = {
            "mr.": "mister",
            "mrs.": "misses",
            "dr.": "doctor",
            "prof.": "professor",
            "sr.": "senior",
            "jr.": "junior",
            "vs.": "versus",
            "etc.": "etcetera",
            "i.e.": "that is",
            "e.g.": "for example",
            "u.s.": "united states",
            "u.k.": "united kingdom",
            "u.s.a.": "united states of america",
        }
        
        # Number to word mappings
        self.ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        self.teens = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", 
                     "sixteen", "seventeen", "eighteen", "nineteen"]
        self.tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        
    def _init_char_mappings(self):
        """Initialize character to index mappings"""
        # Basic characters
        chars = list(string.ascii_lowercase) + list(string.digits) + [' ', '.', ',', '!', '?', ';', ':', '-', "'"]
        
        # Add special tokens
        special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        
        # Create mappings
        self.char_to_idx = {char: idx for idx, char in enumerate(special_tokens + chars)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.vocab_size = len(self.char_to_idx)
        
        # Special token indices
        self.pad_idx = self.char_to_idx['<pad>']
        self.sos_idx = self.char_to_idx['<sos>']
        self.eos_idx = self.char_to_idx['<eos>']
        self.unk_idx = self.char_to_idx['<unk>']
    
    def normalize_text(self, text):
        """Normalize text by removing diacritics and converting to lowercase"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove diacritics
        text = unicodedata.normalize('NFD', text)
        text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def expand_abbreviations(self, text):
        """Expand common abbreviations"""
        words = text.split()
        expanded_words = []
        
        for word in words:
            # Check if word is an abbreviation
            lower_word = word.lower().rstrip('.,!?;:')
            if lower_word in self.abbreviations:
                expanded_words.append(self.abbreviations[lower_word])
            else:
                expanded_words.append(word)
        
        return ' '.join(expanded_words)
    
    def convert_numbers(self, text):
        """Convert numbers to words"""
        # Pattern for numbers
        number_pattern = r'\b\d+\b'
        
        def number_to_words(match):
            number = int(match.group())
            return self._number_to_words(number)
        
        text = re.sub(number_pattern, number_to_words, text)
        return text
    
    def _number_to_words(self, number):
        """Convert a number to its word representation"""
        if number == 0:
            return "zero"
        
        if number < 10:
            return self.ones[number]
        elif number < 20:
            return self.teens[number - 10]
        elif number < 100:
            return self.tens[number // 10] + ("" if number % 10 == 0 else " " + self.ones[number % 10])
        elif number < 1000:
            return (self.ones[number // 100] + " hundred" + 
                   ("" if number % 100 == 0 else " " + self._number_to_words(number % 100)))
        elif number < 1000000:
            return (self._number_to_words(number // 1000) + " thousand" + 
                   ("" if number % 1000 == 0 else " " + self._number_to_words(number % 1000)))
        else:
            return str(number)  # For very large numbers, return as string
    
    def clean_text(self, text):
        """Clean text by removing unwanted characters"""
        # Remove multiple punctuation
        text = re.sub(r'[.]{2,}', '.', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Remove special characters except basic punctuation
        allowed_chars = set(string.ascii_letters + string.digits + ' .,!?;:\'-')
        text = ''.join(char for char in text if char in allowed_chars)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def text_to_phonemes(self, text):
        """Convert text to phonemes using phonemizer"""
        if not self.phonemizer_available:
            print("Phonemizer not available, using character-level fallback")
            return list(self.clean_text(text).lower())
        
        try:
            # Clean text first
            text = self.clean_text(text)
            
            # Convert to phonemes using espeak with proper path
            phonemes = phonemize(
                text,
                language=self.language,
                backend='espeak',
                strip=True,
                preserve_punctuation=True,
                with_stress=False  # Disable stress to avoid issues
            )
            
            # Split into list and filter empty strings
            phoneme_list = [p for p in phonemes.split() if p.strip()]
            
            return phoneme_list if phoneme_list else list(text.lower())
        
        except Exception as e:
            print(f"Error in phoneme conversion: {e}")
            # Fallback to character-level
            return list(text.lower())
    
    def text_to_sequence(self, text, use_phonemes=False):
        """Convert text to sequence of indices"""
        if use_phonemes:
            phonemes = self.text_to_phonemes(text)
            # For simplicity, map phonemes to characters
            # In practice, you'd want a separate phoneme vocabulary
            sequence = [self.char_to_idx.get(char, self.unk_idx) for char in ' '.join(phonemes).lower()]
        else:
            # Character-level encoding
            text = self.normalize_text(text)
            text = self.clean_text(text)
            sequence = [self.char_to_idx.get(char, self.unk_idx) for char in text]
        
        # Add start and end tokens
        sequence = [self.sos_idx] + sequence + [self.eos_idx]
        
        return sequence
    
    def sequence_to_text(self, sequence):
        """Convert sequence of indices back to text"""
        chars = [self.idx_to_char.get(idx, '<unk>') for idx in sequence]
        
        # Remove special tokens
        chars = [char for char in chars if char not in ['<pad>', '<sos>', '<eos>']]
        
        return ''.join(chars)
    
    def pad_sequences(self, sequences, max_length=None):
        """Pad sequences to the same length"""
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)
        
        padded_sequences = []
        for seq in sequences:
            if len(seq) < max_length:
                # Pad with pad token
                padded_seq = seq + [self.pad_idx] * (max_length - len(seq))
            else:
                # Truncate if too long
                padded_seq = seq[:max_length]
            
            padded_sequences.append(padded_seq)
        
        return padded_sequences
    
    def preprocess_batch(self, texts, use_phonemes=False, max_length=None):
        """Preprocess a batch of texts"""
        # Convert texts to sequences
        sequences = []
        for text in texts:
            # Apply all preprocessing steps
            processed_text = self.normalize_text(text)
            processed_text = self.expand_abbreviations(processed_text)
            processed_text = self.convert_numbers(processed_text)
            processed_text = self.clean_text(processed_text)
            
            # Convert to sequence
            sequence = self.text_to_sequence(processed_text, use_phonemes)
            sequences.append(sequence)
        
        # Pad sequences
        padded_sequences = self.pad_sequences(sequences, max_length)
        
        # Convert to tensor
        tensor = torch.tensor(padded_sequences, dtype=torch.long)
        
        return tensor
    
    def get_text_statistics(self, text):
        """Get statistics about the text"""
        stats = {
            'character_count': len(text),
            'word_count': len(text.split()),
            'sentence_count': len(nltk.sent_tokenize(text)),
            'unique_characters': len(set(text.lower())),
            'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0
        }
        
        return stats
