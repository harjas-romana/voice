#!/usr/bin/env python3
"""
Test script for streaming TTS functionality.

This script demonstrates how to use the streaming TTS functionality
in both voice.py and through the WebSocket API.
"""

import asyncio
import time
import wave
import io
import sys
import os
from pathlib import Path
import websockets
import json

# Add parent directory to path to import voice module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import voice

async def test_direct_streaming():
    """Test streaming TTS directly through voice.py."""
    print("Testing direct streaming TTS...")
    
    # Create output directory
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Test text
    test_text = "Kia ora, welcome to QuantAI Restaurant in Auckland! We're delighted to have you join us today. Our chef has prepared some amazing specials that I think you'll love. Would you like to hear about them?"
    
    # Prepare WAV file
    wav_file = output_dir / "streaming_output.wav"
    
    # Start timing
    start_time = time.time()
    
    # Create a list to collect all audio chunks
    all_chunks = []
    
    # Stream audio chunks
    print("Generating streaming audio...")
    chunk_count = 0
    async for chunk in voice.speak_streaming(test_text, language="en"):
        chunk_count += 1
        all_chunks.append(chunk)
        print(f"Received chunk {chunk_count}: {len(chunk)} bytes")
    
    # Combine all chunks into one WAV file
    with open(wav_file, "wb") as f:
        for chunk in all_chunks:
            f.write(chunk)
    
    # Calculate total time
    total_time = time.time() - start_time
    
    print(f"Streaming TTS completed in {total_time:.2f} seconds")
    print(f"Total chunks: {chunk_count}")
    print(f"Output saved to {wav_file}")
    
    # For comparison, test non-streaming TTS
    print("\nTesting non-streaming TTS for comparison...")
    start_time = time.time()
    audio_data = voice.speak(test_text, language="en")
    total_time = time.time() - start_time
    
    # Save non-streaming output
    non_streaming_file = output_dir / "non_streaming_output.wav"
    with open(non_streaming_file, "wb") as f:
        f.write(audio_data)
    
    print(f"Non-streaming TTS completed in {total_time:.2f} seconds")
    print(f"Output saved to {non_streaming_file}")

async def test_websocket_streaming():
    """Test streaming TTS through WebSocket API."""
    print("\nTesting WebSocket streaming TTS...")
    
    # WebSocket URL
    ws_url = "ws://localhost:8006/ws/tts"
    
    # Test text
    test_text = "Kia ora, welcome to QuantAI Restaurant in Auckland! We're delighted to have you join us today. Our chef has prepared some amazing specials that I think you'll love. Would you like to hear about them?"
    
    # Create output directory
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)
    
    # Prepare WAV file
    wav_file = output_dir / "websocket_streaming_output.wav"
    
    # Start timing
    start_time = time.time()
    
    # Create a list to collect all audio chunks
    all_chunks = []
    
    try:
        # Connect to WebSocket
        async with websockets.connect(ws_url) as websocket:
            print("Connected to WebSocket")
            
            # Send TTS request
            request = {
                "text": test_text,
                "language": "en",
                "chunk_size": 1024
            }
            await websocket.send(json.dumps(request))
            print("Request sent")
            
            # Receive and process messages
            chunk_count = 0
            while True:
                message = await websocket.recv()
                
                if isinstance(message, bytes):
                    # Binary audio chunk
                    chunk_count += 1
                    all_chunks.append(message)
                    print(f"Received audio chunk {chunk_count}: {len(message)} bytes")
                else:
                    # JSON message
                    data = json.loads(message)
                    print(f"Received message: {data}")
                    
                    if data.get("status") == "end":
                        print("Streaming completed")
                        break
                    elif data.get("error"):
                        print(f"Error: {data['error']}")
                        break
    
    except Exception as e:
        print(f"WebSocket error: {e}")
    
    # Combine all chunks into one WAV file
    if all_chunks:
        with open(wav_file, "wb") as f:
            for chunk in all_chunks:
                f.write(chunk)
    
    # Calculate total time
    total_time = time.time() - start_time
    
    print(f"WebSocket streaming TTS completed in {total_time:.2f} seconds")
    print(f"Total chunks: {chunk_count}")
    print(f"Output saved to {wav_file}")

async def test_conversation_websocket():
    """Test conversation through WebSocket API."""
    print("\nTesting WebSocket conversation...")
    
    # WebSocket URL
    ws_url = "ws://localhost:8006/ws/conversation"
    
    # Test query
    test_query = "What's on the menu today?"
    
    try:
        # Connect to WebSocket
        async with websockets.connect(ws_url) as websocket:
            print("Connected to WebSocket")
            
            # Send conversation request
            request = {
                "text": test_query,
                "language": "en"
            }
            await websocket.send(json.dumps(request))
            print("Request sent")
            
            # Create output directory for audio
            output_dir = Path("test_output")
            output_dir.mkdir(exist_ok=True)
            audio_chunks = []
            
            # Receive and process messages
            response_text = ""
            while True:
                message = await websocket.recv()
                
                if isinstance(message, bytes):
                    # Binary audio chunk
                    audio_chunks.append(message)
                    print(f"Received audio chunk: {len(message)} bytes")
                else:
                    # JSON message
                    data = json.loads(message)
                    
                    if data.get("type") == "text" and "chunk" in data:
                        response_text += data["chunk"]
                        print(f"Text chunk: {data['chunk']}")
                    elif data.get("status") == "complete":
                        print("Conversation completed")
                        print(f"Full response: {response_text}")
                        break
                    elif data.get("error"):
                        print(f"Error: {data['error']}")
                        break
                    else:
                        print(f"Received message: {data}")
            
            # Save audio if received
            if audio_chunks:
                audio_file = output_dir / "conversation_audio.wav"
                with open(audio_file, "wb") as f:
                    for chunk in audio_chunks:
                        f.write(chunk)
                print(f"Audio saved to {audio_file}")
    
    except Exception as e:
        print(f"WebSocket error: {e}")

async def main():
    """Run all tests."""
    # Test direct streaming
    await test_direct_streaming()
    
    # Test WebSocket streaming
    await test_websocket_streaming()
    
    # Test conversation WebSocket
    await test_conversation_websocket()

if __name__ == "__main__":
    asyncio.run(main()) 