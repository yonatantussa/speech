# External TTS APIs Integration

This module provides a unified interface to test and compare multiple Text-to-Speech (TTS) APIs.

## Supported Providers

### 1. **ElevenLabs**
- Website: https://elevenlabs.io
- High-quality voice cloning and synthesis
- Multiple voice options
- API Key required

### 2. **Cartesia (Sonic)**
- Website: https://cartesia.ai
- Ultra-low latency TTS
- Real-time streaming capabilities
- API Key required

### 3. **Hume AI**
- Website: https://hume.ai
- Empathic voice with emotion control
- Prosody and emotion parameters
- API Key required

### 4. **Play.ht**
- Website: https://play.ht
- Wide range of voices
- Multiple languages
- API Key + User ID required

### 5. **Microsoft Azure**
- Website: https://azure.microsoft.com/en-us/services/cognitive-services/text-to-speech/
- Neural voices
- SSML support
- API Key + Region required

### 6. **Google Cloud**
- Website: https://cloud.google.com/text-to-speech
- WaveNet and Neural2 voices
- Multiple languages
- API Key required

### 7. **Amazon Polly**
- Website: https://aws.amazon.com/polly/
- Neural and standard voices
- AWS integration
- AWS credentials required

## Setup

### Installation

No additional installation needed if you've already installed the project dependencies.

For Amazon Polly support, install boto3:
```bash
pip install boto3
```

### API Keys

Set up your API keys in one of these ways:

#### Option 1: Environment Variables (Recommended)
```bash
export ELEVENLABS_API_KEY="your_key_here"
export CARTESIA_API_KEY="your_key_here"
export HUME_API_KEY="your_key_here"
export PLAYHT_API_KEY="your_key_here"
export PLAYHT_USER_ID="your_user_id_here"
export AZURE_SPEECH_KEY="your_key_here"
export AZURE_SPEECH_REGION="eastus"
export GOOGLE_CLOUD_API_KEY="your_key_here"
export AWS_ACCESS_KEY_ID="your_key_here"
export AWS_SECRET_ACCESS_KEY="your_secret_here"
```

#### Option 2: Streamlit UI
Enter your API keys directly in the web interface under "External TTS APIs" tab.

## Usage

### Via Streamlit UI

1. Navigate to **"External TTS APIs"** page
2. Go to **"Quick Test"** tab
3. Select your provider from the dropdown
4. Enter your API key
5. Enter text to synthesize
6. Click **"Synthesize"**

### Programmatic Usage

```python
from external_apis.tts_api_client import get_tts_client

# Get a client for any provider
client = get_tts_client("elevenlabs", api_key="your_key")

# Synthesize speech
audio_bytes = client.synthesize(
    text="Hello, world!",
    voice="21m00Tcm4TlvDq8ikWAM"  # Voice ID
)

# Save to file
with open("output.wav", "wb") as f:
    f.write(audio_bytes)
```

### Provider-Specific Examples

#### ElevenLabs
```python
client = get_tts_client("elevenlabs", api_key="your_key")
audio = client.synthesize(
    text="Hello!",
    voice="21m00Tcm4TlvDq8ikWAM",  # Rachel
    stability=0.5,
    similarity_boost=0.75
)
```

#### Cartesia
```python
client = get_tts_client("cartesia", api_key="your_key")
audio = client.synthesize(
    text="Hello!",
    voice="a0e99841-438c-4a64-b679-ae501e7d6091",
    model="sonic-english",
    sample_rate=22050
)
```

#### Azure
```python
client = get_tts_client("azure", api_key="your_key", region="eastus")
audio = client.synthesize(
    text="Hello!",
    voice="en-US-AriaNeural"
)
```

## Voice IDs / Names

### ElevenLabs Popular Voices
- `21m00Tcm4TlvDq8ikWAM` - Rachel (female)
- `29vD33N1CtxCmqQRPOHJ` - Drew (male)
- `ErXwobaYiN019PkySvjV` - Antoni (male)

### Azure Popular Voices
- `en-US-AriaNeural` - Female
- `en-US-GuyNeural` - Male
- `en-US-JennyNeural` - Female

### Google Popular Voices
- `en-US-Neural2-A` - Male
- `en-US-Neural2-C` - Female
- `en-US-Neural2-D` - Male

### Amazon Polly Popular Voices
- `Joanna` - Female
- `Matthew` - Male
- `Ivy` - Female (child)

## Pricing (as of 2024)

- **ElevenLabs**: ~$0.30 per 1K characters (Free tier: 10K/month)
- **Cartesia**: ~$0.05 per 1K characters (Free tier available)
- **Play.ht**: ~$0.06 per 1K characters (Free tier: 2.5K words)
- **Azure**: ~$16 per 1M characters (Free tier: 0.5M/month)
- **Google**: ~$16 per 1M characters (Free tier: 1M/month)
- **Amazon Polly**: ~$16 per 1M characters (Free tier: 5M/month first year)

*Note: Prices subject to change. Check provider websites for current pricing.*

## Comparison Features

Use the **"API Comparison"** tab in the Streamlit UI to:
- Test the same text across multiple providers
- Compare synthesis times
- Compare audio quality
- Evaluate cost-effectiveness

## Troubleshooting

### "Module not found" errors
Make sure all dependencies are installed:
```bash
pip install requests numpy soundfile
```

### Authentication errors
- Double-check your API keys
- Ensure they have the correct permissions
- Check if your account has available credits

### Amazon Polly errors
Install boto3:
```bash
pip install boto3
```

## Contributing

To add a new TTS provider:

1. Create a new client class in `tts_api_client.py` that inherits from `TTSAPIClient`
2. Implement `synthesize()` method
3. Add to the `providers` dict in `get_tts_client()`
4. Update this README

## License

See main project LICENSE file.
