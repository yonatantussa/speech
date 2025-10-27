# Overview

This is a comprehensive Text-to-Speech (TTS) research pipeline built with Python and Streamlit. The system provides a complete framework for training, evaluating, and synthesizing speech from text using deep learning models. It implements the Tacotron architecture with attention mechanisms and includes support for various vocoders (WaveNet-style and MelGAN). The application serves as both a research tool and an interactive platform for experimenting with TTS model architectures and synthesis parameters.

## Recent Changes
- **August 2025**: Added Cartesia Sonic API integration for external TTS service comparison and benchmarking
- **August 2025**: Resolved espeak dependency issues by implementing graceful fallback to character-level processing when phoneme conversion is unavailable
- **System Status**: Fully operational with robust error handling and external API integration capabilities

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Streamlit-based Web Interface**: Multi-page application with sections for text preprocessing, model architecture, training, synthesis, evaluation, and configuration
- **Interactive Visualization**: Real-time plotting of spectrograms, training metrics, and audio waveforms using Matplotlib and Plotly
- **Session State Management**: Persistent model and training state across user interactions

## Backend Architecture
- **Modular Design**: Separated into distinct modules for models, preprocessing, training, synthesis, evaluation, and utilities
- **Model Architecture**: Implements Tacotron TTS with encoder-decoder architecture and Bahdanau attention mechanism
- **Vocoder Support**: Multiple vocoder options including WaveNet-style and MelGAN for mel-spectrogram to audio conversion
- **Training Pipeline**: Complete training infrastructure with gradient clipping, learning rate scheduling, and checkpoint management

## Data Processing Pipeline
- **Text Processing**: Comprehensive text normalization, phoneme conversion using Phonemizer with eSpeak backend, and character-to-index mapping
- **Audio Processing**: Mel-spectrogram extraction using librosa, audio normalization, preemphasis filtering, and silence trimming
- **Dataset Management**: Flexible dataset loading supporting CSV metadata with length filtering and batch processing

## Training Infrastructure
- **PyTorch-based Training**: Adam optimizer with configurable learning rate scheduling and gradient clipping
- **Loss Functions**: Multi-component loss including mel-spectrogram loss and post-net loss with configurable weights
- **Metrics and Evaluation**: Comprehensive audio quality metrics including RMS energy, zero-crossing rate, spectral features, and perceptual metrics
- **Logging and Monitoring**: Structured logging with colored console output and file logging capabilities

## Synthesis Engine
- **Real-time Synthesis**: Text-to-speech conversion with configurable speed, pitch, and energy parameters
- **Multi-stage Processing**: Text normalization → Phoneme conversion → Mel-spectrogram generation → Audio synthesis
- **Quality Control**: Audio normalization and post-processing for consistent output quality

# External Dependencies

## Core ML/Audio Libraries
- **PyTorch**: Deep learning framework for model implementation and training
- **librosa**: Audio analysis and feature extraction (mel-spectrograms, STFT)
- **soundfile**: Audio file I/O operations
- **numpy**: Numerical computations and array operations

## Text Processing
- **phonemizer**: Text-to-phoneme conversion with eSpeak backend support
- **nltk**: Natural language processing for text tokenization and normalization

## Visualization and UI
- **streamlit**: Web application framework for the interactive interface
- **matplotlib**: Static plotting for spectrograms and training visualizations
- **plotly**: Interactive plotting and real-time visualization
- **seaborn**: Statistical visualization styling

## Audio Quality Assessment
- **pesq**: Perceptual Evaluation of Speech Quality metrics
- **pystoi**: Short-Time Objective Intelligibility measure
- **scipy**: Signal processing utilities and statistical functions

## Configuration and Logging
- **yaml**: Configuration file parsing and management
- **colorlog**: Colored console logging for better debugging experience
- **pandas**: Data manipulation for metadata handling

## Scientific Computing
- **sklearn**: Machine learning utilities for metrics computation
- **scipy**: Advanced mathematical functions and signal processing