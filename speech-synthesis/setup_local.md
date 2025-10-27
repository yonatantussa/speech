# Local Setup Guide for TTS Research Pipeline

## Prerequisites

1. **Python 3.11+** - Make sure you have Python 3.11 or higher installed
2. **espeak** - Required for phoneme conversion (see installation instructions below)

## Installation Steps

### 1. Clone and Setup
```bash
# Install Python dependencies
pip install -r local_requirements.txt

# Download required NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('cmudict')"
```

### 2. Install espeak (System-level dependency)

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install espeak espeak-data
```

**macOS:**
```bash
brew install espeak
```

**Windows:**
- Download and install espeak from: http://espeak.sourceforge.net/download.html
- Add espeak to your system PATH

### 3. GPU Support (Optional)

For CUDA-enabled GPU acceleration:
```bash
# Uninstall CPU-only torch
pip uninstall torch

# Install CUDA version (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Run the Application

```bash
# Start the Streamlit web interface
streamlit run app.py --server.port 5000

# Or test the Cartesia API directly
python test_cartesia.py
```

## Configuration

The application will automatically create default configuration files on first run. You can customize:

- `config/model_config.yaml` - Model architecture and training parameters
- `.streamlit/config.toml` - Web interface settings

## Troubleshooting

### Common Issues

1. **"espeak not found"** - Install espeak as shown above
2. **Import errors** - Make sure all dependencies are installed: `pip install -r local_requirements.txt`
3. **CUDA issues** - Ensure you have the right PyTorch version for your CUDA installation
4. **Port conflicts** - Change the port in the streamlit command if 5000 is busy

### Fallback Mode

The application includes robust error handling:
- If espeak is not available, it falls back to character-level text processing
- The Cartesia API integration is optional and won't break the main application if unavailable

## Features Available

- ✅ Text preprocessing with phoneme conversion
- ✅ Custom TTS model training (Tacotron architecture)
- ✅ Audio synthesis and evaluation
- ✅ Interactive web interface with real-time visualizations
- ✅ External API testing with Cartesia Sonic
- ✅ Audio quality metrics and benchmarking
- ✅ Model configuration and checkpoint management

## Next Steps

1. Configure your model parameters in the Configuration tab
2. Test text preprocessing with your target language
3. Train custom models or test with the Cartesia API
4. Compare different TTS approaches using the evaluation tools