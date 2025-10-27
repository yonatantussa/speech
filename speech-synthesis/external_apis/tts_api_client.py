"""
Unified TTS API Client for multiple external services
Supports: ElevenLabs, Cartesia, Hume AI, Azure, Google, Amazon Polly, PlayHT
"""

import requests
import json
import os
from typing import Optional, Dict, List
import base64
from pathlib import Path


class TTSAPIClient:
    """Base class for TTS API clients"""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key

    def synthesize(self, text: str, voice: str = None, **kwargs) -> bytes:
        """Synthesize speech from text. Returns audio bytes."""
        raise NotImplementedError

    def list_voices(self) -> List[Dict]:
        """List available voices"""
        raise NotImplementedError


class ElevenLabsClient(TTSAPIClient):
    """ElevenLabs TTS API Client"""

    BASE_URL = "https://api.elevenlabs.io/v1"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key or os.getenv("ELEVENLABS_API_KEY"))
        self.headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }

    def synthesize(self, text: str, voice: str = "21m00Tcm4TlvDq8ikWAM",
                   model: str = "eleven_monolingual_v1", **kwargs) -> bytes:
        """
        Synthesize speech using ElevenLabs

        Args:
            text: Text to synthesize
            voice: Voice ID (default: Rachel)
            model: Model to use
            **kwargs: Additional parameters (stability, similarity_boost, etc.)
        """
        url = f"{self.BASE_URL}/text-to-speech/{voice}"

        payload = {
            "text": text,
            "model_id": model,
            "voice_settings": {
                "stability": kwargs.get("stability", 0.5),
                "similarity_boost": kwargs.get("similarity_boost", 0.75),
                "style": kwargs.get("style", 0.0),
                "use_speaker_boost": kwargs.get("use_speaker_boost", True)
            }
        }

        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status()

        return response.content

    def list_voices(self) -> List[Dict]:
        """List available ElevenLabs voices"""
        url = f"{self.BASE_URL}/voices"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()

        return response.json()["voices"]


class CartesiaClient(TTSAPIClient):
    """Cartesia Sonic TTS API Client"""

    BASE_URL = "https://api.cartesia.ai"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key or os.getenv("CARTESIA_API_KEY"))
        self.headers = {
            "X-API-Key": self.api_key,
            "Cartesia-Version": "2024-06-10",
            "Content-Type": "application/json"
        }

    def synthesize(self, text: str, voice: str = "a0e99841-438c-4a64-b679-ae501e7d6091",
                   model: str = "sonic-english", **kwargs) -> bytes:
        """
        Synthesize speech using Cartesia Sonic

        Args:
            text: Text to synthesize
            voice: Voice ID
            model: Model to use (sonic-english, sonic-multilingual)
        """
        url = f"{self.BASE_URL}/tts/bytes"

        payload = {
            "model_id": model,
            "transcript": text,
            "voice": {
                "mode": "id",
                "id": voice
            },
            "output_format": {
                "container": "wav",
                "encoding": "pcm_s16le",
                "sample_rate": kwargs.get("sample_rate", 22050)
            }
        }

        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status()

        return response.content


class HumeAIClient(TTSAPIClient):
    """Hume AI Empathic Voice Interface Client"""

    BASE_URL = "https://api.hume.ai/v0"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key or os.getenv("HUME_API_KEY"))
        self.headers = {
            "X-Hume-Api-Key": self.api_key,
            "Content-Type": "application/json"
        }

    def synthesize(self, text: str, voice: str = None, **kwargs) -> bytes:
        """
        Synthesize speech using Hume AI

        Args:
            text: Text to synthesize
            voice: Voice configuration
        """
        url = f"{self.BASE_URL}/evi/chat"

        payload = {
            "text": text,
            "config": {
                "voice": voice or {},
                "prosody": kwargs.get("prosody", {}),
                "emotion": kwargs.get("emotion", {})
            }
        }

        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status()

        # Hume returns base64 encoded audio
        result = response.json()
        audio_base64 = result.get("audio", "")
        return base64.b64decode(audio_base64)


class PlayHTClient(TTSAPIClient):
    """Play.ht TTS API Client"""

    BASE_URL = "https://api.play.ht/api/v2"

    def __init__(self, api_key: Optional[str] = None, user_id: Optional[str] = None):
        super().__init__(api_key or os.getenv("PLAYHT_API_KEY"))
        self.user_id = user_id or os.getenv("PLAYHT_USER_ID")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "X-User-Id": self.user_id,
            "Content-Type": "application/json"
        }

    def synthesize(self, text: str, voice: str = "larry", **kwargs) -> bytes:
        """
        Synthesize speech using Play.ht

        Args:
            text: Text to synthesize
            voice: Voice ID or name
        """
        url = f"{self.BASE_URL}/tts"

        payload = {
            "text": text,
            "voice": voice,
            "quality": kwargs.get("quality", "premium"),
            "output_format": kwargs.get("output_format", "wav"),
            "speed": kwargs.get("speed", 1.0),
            "sample_rate": kwargs.get("sample_rate", 24000)
        }

        response = requests.post(url, json=payload, headers=self.headers)
        response.raise_for_status()

        # Play.ht returns URL to download audio
        result = response.json()
        audio_url = result.get("url")

        if audio_url:
            audio_response = requests.get(audio_url)
            audio_response.raise_for_status()
            return audio_response.content

        return b""


class AzureTTSClient(TTSAPIClient):
    """Microsoft Azure Cognitive Services TTS Client"""

    def __init__(self, api_key: Optional[str] = None, region: str = "eastus"):
        super().__init__(api_key or os.getenv("AZURE_SPEECH_KEY"))
        self.region = region or os.getenv("AZURE_SPEECH_REGION", "eastus")
        self.base_url = f"https://{self.region}.tts.speech.microsoft.com"

    def synthesize(self, text: str, voice: str = "en-US-AriaNeural", **kwargs) -> bytes:
        """
        Synthesize speech using Azure TTS

        Args:
            text: Text to synthesize
            voice: Voice name (e.g., en-US-AriaNeural)
        """
        url = f"{self.base_url}/cognitiveservices/v1"

        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "audio-24khz-48kbitrate-mono-mp3"
        }

        # Create SSML
        ssml = f"""
        <speak version='1.0' xml:lang='en-US'>
            <voice xml:lang='en-US' name='{voice}'>
                {text}
            </voice>
        </speak>
        """

        response = requests.post(url, headers=headers, data=ssml.encode('utf-8'))
        response.raise_for_status()

        return response.content


class GoogleTTSClient(TTSAPIClient):
    """Google Cloud Text-to-Speech Client"""

    BASE_URL = "https://texttospeech.googleapis.com/v1"

    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key or os.getenv("GOOGLE_CLOUD_API_KEY"))

    def synthesize(self, text: str, voice: str = "en-US-Neural2-A", **kwargs) -> bytes:
        """
        Synthesize speech using Google Cloud TTS

        Args:
            text: Text to synthesize
            voice: Voice name
        """
        url = f"{self.BASE_URL}/text:synthesize?key={self.api_key}"

        # Parse voice name
        parts = voice.split("-")
        language_code = f"{parts[0]}-{parts[1]}" if len(parts) >= 2 else "en-US"

        payload = {
            "input": {"text": text},
            "voice": {
                "languageCode": language_code,
                "name": voice
            },
            "audioConfig": {
                "audioEncoding": "LINEAR16",
                "sampleRateHertz": kwargs.get("sample_rate", 24000)
            }
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        result = response.json()
        audio_base64 = result.get("audioContent", "")
        return base64.b64decode(audio_base64)


class AmazonPollyClient(TTSAPIClient):
    """Amazon Polly TTS Client"""

    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None,
                 region: str = "us-east-1"):
        super().__init__(api_key or os.getenv("AWS_ACCESS_KEY_ID"))
        self.secret_key = secret_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        self.region = region

    def synthesize(self, text: str, voice: str = "Joanna", **kwargs) -> bytes:
        """
        Synthesize speech using Amazon Polly

        Note: This is a simplified implementation. For production use,
        use boto3 library for proper AWS authentication.

        Args:
            text: Text to synthesize
            voice: Voice ID
        """
        try:
            import boto3

            polly = boto3.client(
                'polly',
                aws_access_key_id=self.api_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region
            )

            response = polly.synthesize_speech(
                Text=text,
                OutputFormat='mp3',
                VoiceId=voice,
                Engine=kwargs.get("engine", "neural")
            )

            return response['AudioStream'].read()

        except ImportError:
            raise ImportError("boto3 is required for Amazon Polly. Install with: pip install boto3")


# Factory function to get the right client
def get_tts_client(provider: str, **kwargs) -> TTSAPIClient:
    """
    Get TTS client for specified provider

    Args:
        provider: Provider name (elevenlabs, cartesia, hume, playht, azure, google, polly)
        **kwargs: Additional arguments for client initialization

    Returns:
        TTSAPIClient instance
    """
    providers = {
        "elevenlabs": ElevenLabsClient,
        "cartesia": CartesiaClient,
        "hume": HumeAIClient,
        "playht": PlayHTClient,
        "azure": AzureTTSClient,
        "google": GoogleTTSClient,
        "polly": AmazonPollyClient
    }

    provider_lower = provider.lower()

    if provider_lower not in providers:
        raise ValueError(f"Unknown provider: {provider}. Available: {list(providers.keys())}")

    return providers[provider_lower](**kwargs)
