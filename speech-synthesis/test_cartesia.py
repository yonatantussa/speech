#!/usr/bin/env python3
"""
Quick test script for Cartesia Sonic API
"""

import os
from external_apis.cartesia_test import CartesiaSonicTester

def main():
    # Test with your API key
    api_key = "sk_car_Lmi1PeaDrpqwggZ3J1Q4Qa"  # Replace with your actual key
    
    try:
        print("Testing Cartesia Sonic API...")
        tester = CartesiaSonicTester(api_key=api_key)
        
        # Basic test
        result = tester.test_basic_synthesis("Hello world, this is a test of Cartesia Sonic.")
        
        if result["success"]:
            print("✓ API test successful!")
            print(f"  Audio duration: {result['audio_duration']:.2f}s")
            print(f"  RTF: {result['rtf']:.2f}")
            
            # Save audio
            tester.save_audio(result["audio_data"], result["sample_rate"], "test_output.wav")
            
        else:
            print(f"✗ API test failed: {result['error']}")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()