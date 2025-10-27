"""
Cartesia Sonic TTS API Test Module

This module provides functionality to test and compare the Cartesia Sonic TTS model
with the custom TTS pipeline. It's useful for benchmarking and research purposes.
"""

import requests
import json
import os
import soundfile as sf
import numpy as np
from io import BytesIO
from pathlib import Path
import time
from typing import Optional, Dict, Any

class CartesiaSonicTester:
    """Test interface for Cartesia Sonic TTS API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Cartesia Sonic tester
        
        Args:
            api_key: Cartesia API key. If None, will try to get from environment
        """
        self.api_key = api_key or os.getenv('CARTESIA_API_KEY')
        self.base_url = "https://api.cartesia.ai/tts/bytes"
        self.version = "2024-06-10"
        
        if not self.api_key:
            raise ValueError("Cartesia API key is required. Set CARTESIA_API_KEY environment variable or pass api_key parameter.")
        
        # Default configuration
        self.default_voice_id = "bf0a246a-8642-498a-9950-80c35e9276b5"
        self.default_model = "sonic-2"
        
    def synthesize(self, 
                   text: str,
                   voice_id: Optional[str] = None,
                   model_id: str = "sonic-2",
                   language: str = "en",
                   output_format: Optional[Dict[str, Any]] = None) -> tuple:
        """
        Synthesize speech using Cartesia Sonic API
        
        Args:
            text: Text to synthesize
            voice_id: Voice ID to use (defaults to preset voice)
            model_id: Model to use (sonic-2, etc.)
            language: Language code
            output_format: Audio output format configuration
            
        Returns:
            tuple: (audio_data, sample_rate, response_info)
        """
        # Set defaults
        voice_id = voice_id or self.default_voice_id
        output_format = output_format or {
            "container": "wav",
            "encoding": "pcm_f32le", 
            "sample_rate": 44100
        }
        
        # Prepare request
        headers = {
            "Cartesia-Version": self.version,
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }
        
        payload = {
            "model_id": model_id,
            "transcript": text,
            "voice": {
                "mode": "id",
                "id": voice_id
            },
            "output_format": output_format,
            "language": language
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(self.base_url, headers=headers, json=payload)
            response.raise_for_status()
            
            synthesis_time = time.time() - start_time
            
            # Parse audio data
            audio_bytes = response.content
            audio_data, sample_rate = sf.read(BytesIO(audio_bytes))
            
            # Response info for analysis
            response_info = {
                "synthesis_time": synthesis_time,
                "audio_duration": len(audio_data) / sample_rate,
                "rtf": synthesis_time / (len(audio_data) / sample_rate),
                "status_code": response.status_code,
                "audio_size_bytes": len(audio_bytes),
                "model_used": model_id,
                "voice_used": voice_id
            }
            
            return audio_data, sample_rate, response_info
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Cartesia API request failed: {str(e)}")
        except Exception as e:
            raise Exception(f"Audio processing failed: {str(e)}")
    
    def test_basic_synthesis(self, test_text: str = "Hello, this is a test of the Cartesia Sonic speech synthesis system."):
        """
        Perform basic synthesis test
        
        Args:
            test_text: Text to use for testing
            
        Returns:
            dict: Test results
        """
        print(f"Testing Cartesia Sonic with text: '{test_text}'")
        
        try:
            audio_data, sample_rate, info = self.synthesize(test_text)
            
            results = {
                "success": True,
                "text": test_text,
                "audio_data": audio_data,
                "sample_rate": sample_rate,
                "synthesis_time": info["synthesis_time"],
                "audio_duration": info["audio_duration"],
                "rtf": info["rtf"],
                "audio_quality": {
                    "rms": np.sqrt(np.mean(audio_data**2)),
                    "max_amplitude": np.max(np.abs(audio_data)),
                    "zero_crossings": np.sum(np.diff(np.signbit(audio_data)))
                }
            }
            
            print(f"✓ Synthesis successful!")
            print(f"  Duration: {results['audio_duration']:.2f}s")
            print(f"  Synthesis time: {results['synthesis_time']:.2f}s")
            print(f"  RTF: {results['rtf']:.2f}")
            
            return results
            
        except Exception as e:
            print(f"✗ Synthesis failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "text": test_text
            }
    
    def compare_voices(self, text: str, voice_ids: list):
        """
        Compare different voices using the same text
        
        Args:
            text: Text to synthesize
            voice_ids: List of voice IDs to test
            
        Returns:
            dict: Comparison results
        """
        results = {}
        
        for voice_id in voice_ids:
            print(f"Testing voice: {voice_id}")
            try:
                audio_data, sample_rate, info = self.synthesize(text, voice_id=voice_id)
                results[voice_id] = {
                    "success": True,
                    "audio_data": audio_data,
                    "sample_rate": sample_rate,
                    "synthesis_time": info["synthesis_time"],
                    "rtf": info["rtf"],
                    "audio_duration": info["audio_duration"]
                }
                print(f"  ✓ Success - RTF: {info['rtf']:.2f}")
            except Exception as e:
                results[voice_id] = {
                    "success": False,
                    "error": str(e)
                }
                print(f"  ✗ Failed: {str(e)}")
        
        return results
    
    def benchmark_performance(self, test_texts: list, iterations: int = 3):
        """
        Benchmark synthesis performance across multiple texts and iterations
        
        Args:
            test_texts: List of texts to test
            iterations: Number of iterations per text
            
        Returns:
            dict: Benchmark results
        """
        results = {
            "total_tests": len(test_texts) * iterations,
            "successful_tests": 0,
            "failed_tests": 0,
            "synthesis_times": [],
            "rtf_values": [],
            "text_results": {}
        }
        
        for text in test_texts:
            text_results = []
            print(f"\nBenchmarking text: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            
            for i in range(iterations):
                try:
                    audio_data, sample_rate, info = self.synthesize(text)
                    
                    text_results.append({
                        "iteration": i + 1,
                        "success": True,
                        "synthesis_time": info["synthesis_time"],
                        "rtf": info["rtf"],
                        "audio_duration": info["audio_duration"]
                    })
                    
                    results["synthesis_times"].append(info["synthesis_time"])
                    results["rtf_values"].append(info["rtf"])
                    results["successful_tests"] += 1
                    
                    print(f"  Iteration {i+1}: RTF {info['rtf']:.2f}")
                    
                except Exception as e:
                    text_results.append({
                        "iteration": i + 1,
                        "success": False,
                        "error": str(e)
                    })
                    results["failed_tests"] += 1
                    print(f"  Iteration {i+1}: Failed - {str(e)}")
            
            results["text_results"][text] = text_results
        
        # Calculate statistics
        if results["synthesis_times"]:
            results["avg_synthesis_time"] = np.mean(results["synthesis_times"])
            results["avg_rtf"] = np.mean(results["rtf_values"])
            results["min_rtf"] = np.min(results["rtf_values"])
            results["max_rtf"] = np.max(results["rtf_values"])
        
        return results
    
    def save_audio(self, audio_data: np.ndarray, sample_rate: int, 
                   filename: str, output_dir: str = "cartesia_outputs"):
        """
        Save audio data to file
        
        Args:
            audio_data: Audio data array
            sample_rate: Sample rate
            filename: Output filename
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        file_path = output_path / filename
        sf.write(file_path, audio_data, sample_rate)
        print(f"Audio saved to: {file_path}")
        
        return file_path


def main():
    """Main function for testing Cartesia Sonic API"""
    print("Cartesia Sonic TTS API Tester")
    print("=" * 40)
    
    try:
        # Initialize tester
        tester = CartesiaSonicTester()
        
        # Basic synthesis test
        print("\n1. Basic Synthesis Test")
        basic_result = tester.test_basic_synthesis()
        
        if basic_result["success"]:
            # Save the audio
            tester.save_audio(
                basic_result["audio_data"], 
                basic_result["sample_rate"],
                "basic_test.wav"
            )
        
        # Performance benchmark
        print("\n2. Performance Benchmark")
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Speech synthesis technology has advanced significantly in recent years.",
            "This is a longer sentence to test the performance with more complex text input."
        ]
        
        benchmark_results = tester.benchmark_performance(test_texts, iterations=2)
        
        print(f"\nBenchmark Summary:")
        print(f"  Total tests: {benchmark_results['total_tests']}")
        print(f"  Successful: {benchmark_results['successful_tests']}")
        print(f"  Failed: {benchmark_results['failed_tests']}")
        
        if benchmark_results.get("avg_rtf"):
            print(f"  Average RTF: {benchmark_results['avg_rtf']:.2f}")
            print(f"  RTF range: {benchmark_results['min_rtf']:.2f} - {benchmark_results['max_rtf']:.2f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nPlease ensure you have set the CARTESIA_API_KEY environment variable.")


if __name__ == "__main__":
    main()