import streamlit as st
import torch
from torch.utils.data import DataLoader
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import yaml
import os
import tempfile
import io

from models.tacotron import TacotronTTS
from preprocessing.text_processor import TextProcessor
from preprocessing.audio_processor import AudioProcessor
from synthesis.synthesizer import TTSSynthesizer
from evaluation.metrics import AudioMetrics
from utils.config import load_config
from utils.visualization import plot_spectrogram, plot_training_metrics
from utils.audio_utils import save_audio, load_audio
from training.trainer import TTSTrainer
from training.dataset import TTSDataset

# External APIs (optional)
try:
    from external_apis.cartesia_test import CartesiaSonicTester
    CARTESIA_AVAILABLE = True
except ImportError:
    CARTESIA_AVAILABLE = False

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'trainer' not in st.session_state:
    st.session_state.trainer = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = []

def main():
    st.title("Speech Synthesis Pipeline")
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    pages = ["Text Preprocessing", "Model Architecture", "Training", "Synthesis", "Evaluation", "Configuration"]
    if CARTESIA_AVAILABLE:
        pages.append("External APIs")
    
    page = st.sidebar.selectbox("Choose a section:", pages)
    
    if page == "Text Preprocessing":
        text_preprocessing_page()
    elif page == "Model Architecture":
        model_architecture_page()
    elif page == "Training":
        training_page()
    elif page == "Synthesis":
        synthesis_page()
    elif page == "Evaluation":
        evaluation_page()
    elif page == "Configuration":
        configuration_page()
    elif page == "External APIs" and CARTESIA_AVAILABLE:
        external_apis_page()

def text_preprocessing_page():
    st.header("Text Preprocessing")
    
    # Text input
    text_input = st.text_area(
        "Enter text to preprocess:",
        value="Hello, this is a test of the speech synthesis system.",
        height=100
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Preprocessing Options")
        normalize_text = st.checkbox("Normalize Text", value=True)
        expand_abbreviations = st.checkbox("Expand Abbreviations", value=True)
        convert_numbers = st.checkbox("Convert Numbers to Words", value=True)
        phoneme_conversion = st.checkbox("Convert to Phonemes", value=True)
    
    if st.button("Process Text"):
        try:
            processor = TextProcessor()
            
            # Show original text
            st.subheader("Original Text")
            st.text(text_input)
            
            # Apply preprocessing steps
            processed_text = text_input
            
            if normalize_text:
                processed_text = processor.normalize_text(processed_text)
                st.subheader("Normalized Text")
                st.text(processed_text)
            
            if expand_abbreviations:
                processed_text = processor.expand_abbreviations(processed_text)
                st.subheader("Expanded Abbreviations")
                st.text(processed_text)
            
            if convert_numbers:
                processed_text = processor.convert_numbers(processed_text)
                st.subheader("Numbers to Words")
                st.text(processed_text)
            
            if phoneme_conversion:
                phonemes = processor.text_to_phonemes(processed_text)
                st.subheader("Phonemes")
                st.text(" ".join(phonemes))
            
            # Character and phoneme statistics
            with col2:
                st.subheader("Statistics")
                st.metric("Character Count", len(processed_text))
                if phoneme_conversion:
                    st.metric("Phoneme Count", len(phonemes))
                st.metric("Word Count", len(processed_text.split()))
                
        except Exception as e:
            st.error(f"Error during preprocessing: {str(e)}")

def model_architecture_page():
    st.header("Model Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Configuration")
        
        # Model parameters
        embedding_dim = st.slider("Embedding Dimension", 128, 1024, 512, 64)
        encoder_dim = st.slider("Encoder Dimension", 256, 1024, 512, 64)
        decoder_dim = st.slider("Decoder Dimension", 512, 2048, 1024, 64)
        attention_dim = st.slider("Attention Dimension", 64, 512, 128, 32)
        num_mels = st.slider("Mel Spectrogram Channels", 40, 128, 80, 8)
        
        # Model type selection
        model_type = st.selectbox("Model Architecture", ["Tacotron", "FastSpeech", "Custom"])
        
    with col2:
        st.subheader("Model Statistics")
        
        # Calculate approximate model size
        if st.button("Initialize Model"):
            try:
                # Initialize text processor to get vocab size
                text_processor = TextProcessor()
                
                config = {
                    'vocab_size': text_processor.vocab_size,
                    'embedding_dim': embedding_dim,
                    'encoder_dim': encoder_dim,
                    'decoder_dim': decoder_dim,
                    'attention_dim': attention_dim,
                    'num_mels': num_mels
                }
                
                model = TacotronTTS(config)
                st.session_state.model = model
                
                # Count parameters
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                st.metric("Total Parameters", f"{total_params:,}")
                st.metric("Trainable Parameters", f"{trainable_params:,}")
                st.metric("Model Size (MB)", f"{total_params * 4 / 1024 / 1024:.2f}")
                
                st.success("Model initialized successfully!")
                
                # Show model architecture
                st.subheader("Model Architecture")
                st.text(str(model))
                
            except Exception as e:
                st.error(f"Error initializing model: {str(e)}")

def training_page():
    st.header("Model Training")
    
    if st.session_state.model is None:
        st.warning("Please initialize a model first in the Model Architecture section.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Training Configuration")
        
        # Training parameters
        learning_rate = st.number_input("Learning Rate", 0.0001, 0.01, 0.001, format="%.4f")
        batch_size = st.selectbox("Batch Size", [8, 16, 32, 64], index=1)
        num_epochs = st.slider("Number of Epochs", 1, 100, 10)
        
        # Dataset selection
        st.subheader("Dataset")
        dataset_type = st.selectbox("Dataset Type", ["Synthetic", "LJSpeech", "Custom"], index=0)

        if dataset_type == "Synthetic":
            st.info("ℹ️ Synthetic mode uses 10 sample sentences for quick testing")
        
        if dataset_type == "Custom":
            uploaded_file = st.file_uploader("Upload Dataset (ZIP)", type=['zip'])
        
    with col2:
        st.subheader("Training Progress")
        
        if st.button("Start Training"):
            try:
                st.info("Initializing training...")

                # Create training configuration
                training_config = {
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'num_epochs': num_epochs,
                    'gradient_clip_val': 1.0,
                    'mel_loss_weight': 1.0,
                    'postnet_loss_weight': 1.0,
                    'stop_loss_weight': 50.0
                }

                # Create dataset
                st.write("📚 Creating dataset...")
                if dataset_type == "Synthetic":
                    # Use synthetic data for quick testing
                    dataset = TTSDataset(
                        data_dir="./data/synthetic",
                        metadata_file=None,  # Will create synthetic metadata
                        text_processor=TextProcessor(),
                        audio_processor=AudioProcessor(),
                        max_text_length=200,
                        max_mel_length=1000
                    )
                elif dataset_type == "LJSpeech":
                    # Try to use LJSpeech if available, otherwise fallback to synthetic
                    ljspeech_path = Path("./data/LJSpeech-1.1")
                    if ljspeech_path.exists():
                        dataset = TTSDataset(
                            data_dir=str(ljspeech_path),  # Base directory
                            metadata_file="metadata.csv",  # Just filename, will be joined with data_dir
                            text_processor=TextProcessor(),
                            audio_processor=AudioProcessor()
                        )
                        st.success(f"✅ Loaded LJSpeech dataset with {len(dataset)} samples")
                    else:
                        st.warning("⚠️ LJSpeech not found, using synthetic data instead")
                        dataset = TTSDataset(
                            data_dir="./data/synthetic",
                            metadata_file=None,
                            text_processor=TextProcessor(),
                            audio_processor=AudioProcessor()
                        )
                else:
                    st.error("Custom dataset not yet implemented")
                    return

                # Create data loader
                # Adjust batch size if dataset is too small
                effective_batch_size = min(batch_size, len(dataset))

                if effective_batch_size < batch_size:
                    st.warning(f"⚠️ Batch size reduced from {batch_size} to {effective_batch_size} due to small dataset")

                train_loader = DataLoader(
                    dataset,
                    batch_size=effective_batch_size,
                    shuffle=True,
                    num_workers=0,  # Keep 0 for Streamlit compatibility
                    drop_last=False  # Keep all samples, especially important for small datasets
                )

                st.write(f"✅ Dataset ready: {len(dataset)} samples, {len(train_loader)} batches per epoch (batch size: {effective_batch_size})")

                # Initialize trainer
                trainer = TTSTrainer(
                    model=st.session_state.model,
                    config=training_config
                )

                st.session_state.trainer = trainer

                # Create progress display
                st.write("🏋️ Starting training...")
                epoch_progress = st.progress(0)
                loss_display = st.empty()

                # Real training loop
                loss_history = []

                for epoch in range(num_epochs):
                    st.session_state.model.train()
                    epoch_losses = []

                    # Training batches
                    for batch_idx, batch in enumerate(train_loader):
                        # Perform training step
                        loss_dict = trainer._train_step(batch)
                        epoch_losses.append(loss_dict['total_loss'].item())

                        # Update display every 5 batches
                        if batch_idx % 5 == 0:
                            current_loss = np.mean(epoch_losses)
                            loss_display.text(
                                f"Epoch {epoch + 1}/{num_epochs} | "
                                f"Batch {batch_idx + 1}/{len(train_loader)} | "
                                f"Loss: {current_loss:.4f}"
                            )

                    # Update epoch progress
                    epoch_progress.progress((epoch + 1) / num_epochs)
                    avg_loss = np.mean(epoch_losses)
                    loss_history.append(avg_loss)

                    # Log epoch results
                    st.write(f"✅ Epoch {epoch + 1}/{num_epochs} completed - Avg Loss: {avg_loss:.4f}")

                st.session_state.training_history = loss_history
                st.success("🎉 Training completed!")

                # Plot training curve
                fig = px.line(
                    x=range(1, len(loss_history) + 1),
                    y=loss_history,
                    title="Training Loss",
                    labels={'x': 'Epoch', 'y': 'Loss'}
                )
                st.plotly_chart(fig)

            except Exception as e:
                st.error(f"❌ Error during training: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

def synthesis_page():
    st.header("Audio Synthesis")
    
    if st.session_state.model is None:
        st.warning("Please initialize and train a model first.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Text to Speech")
        
        # Text input for synthesis
        synthesis_text = st.text_area(
            "Enter text to synthesize:",
            value="The quick brown fox jumps over the lazy dog.",
            height=100
        )
        
        # Synthesis parameters
        st.subheader("Synthesis Parameters")
        speed = st.slider("Speed", 0.5, 2.0, 1.0, 0.1)
        pitch = st.slider("Pitch", -1.0, 1.0, 0.0, 0.1)
        energy = st.slider("Energy", 0.5, 2.0, 1.0, 0.1)
        
        if st.button("Synthesize Audio"):
            try:
                # Use vocoder=None to fall back to Griffin-Lim (untrained vocoder produces bad output)
                synthesizer = TTSSynthesizer(st.session_state.model, vocoder=None)

                # Generate audio
                with st.spinner("Generating audio..."):
                    audio_data, sample_rate, mel_spectrogram = synthesizer.synthesize(
                        synthesis_text,
                        speed=speed,
                        pitch=pitch,
                        energy=energy
                    )
                
                # Display audio player
                st.audio(audio_data, sample_rate=sample_rate)
                
                # Show spectrogram
                st.subheader("Mel Spectrogram")
                fig = plot_spectrogram(mel_spectrogram)
                st.pyplot(fig)
                
                # Audio statistics
                with col2:
                    st.subheader("Audio Statistics")
                    duration = len(audio_data) / sample_rate
                    st.metric("Duration (seconds)", f"{duration:.2f}")
                    st.metric("Sample Rate", f"{sample_rate} Hz")
                    st.metric("Audio Length", f"{len(audio_data):,} samples")
                
                # Download button
                audio_bytes = save_audio(audio_data, sample_rate)
                st.download_button(
                    label="Download Audio",
                    data=audio_bytes,
                    file_name="synthesized_audio.wav",
                    mime="audio/wav"
                )
                
            except Exception as e:
                st.error(f"Error during synthesis: {str(e)}")

def evaluation_page():
    st.header("Model Evaluation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Audio Quality Metrics")
        
        # File upload for evaluation
        uploaded_audio = st.file_uploader("Upload Audio for Evaluation", type=['wav', 'mp3'])
        reference_audio = st.file_uploader("Upload Reference Audio (optional)", type=['wav', 'mp3'])
        
        if uploaded_audio is not None:
            # Load and display audio
            audio_data, sample_rate = sf.read(io.BytesIO(uploaded_audio.read()))
            st.audio(uploaded_audio)
            
            if st.button("Evaluate Audio"):
                try:
                    metrics = AudioMetrics()
                    
                    # Calculate metrics
                    results = metrics.evaluate_audio(audio_data, sample_rate)
                    
                    # Display metrics
                    st.metric("Spectral Centroid", f"{results['spectral_centroid']:.2f}")
                    st.metric("Zero Crossing Rate", f"{results['zcr']:.4f}")
                    st.metric("RMS Energy", f"{results['rms']:.4f}")
                    st.metric("Spectral Rolloff", f"{results['spectral_rolloff']:.2f}")
                    
                    # If reference audio is provided, calculate similarity
                    if reference_audio is not None:
                        ref_data, ref_sr = sf.read(io.BytesIO(reference_audio.read()))
                        similarity = metrics.calculate_similarity(audio_data, ref_data)
                        st.metric("Similarity Score", f"{similarity:.3f}")
                    
                except Exception as e:
                    st.error(f"Error during evaluation: {str(e)}")
    
    with col2:
        st.subheader("Training Metrics")
        
        if st.session_state.training_history:
            # Plot training history
            fig = px.line(
                x=range(1, len(st.session_state.training_history) + 1),
                y=st.session_state.training_history,
                title="Training Loss History",
                labels={'x': 'Epoch', 'y': 'Loss'}
            )
            st.plotly_chart(fig)
            
            # Display final metrics
            final_loss = st.session_state.training_history[-1]
            st.metric("Final Training Loss", f"{final_loss:.4f}")
            
            # Training statistics
            min_loss = min(st.session_state.training_history)
            max_loss = max(st.session_state.training_history)
            st.metric("Best Loss", f"{min_loss:.4f}")
            st.metric("Loss Reduction", f"{((max_loss - min_loss) / max_loss * 100):.1f}%")

def configuration_page():
    st.header("Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Configuration")
        
        # Load current config
        try:
            config = load_config("config/model_config.yaml")
            
            # Display editable config
            st.json(config)
            
            # Config editor
            st.subheader("Edit Configuration")
            new_config = st.text_area(
                "Configuration (YAML format):",
                value=yaml.dump(config, default_flow_style=False),
                height=300
            )
            
            if st.button("Save Configuration"):
                try:
                    # Parse and save new config
                    updated_config = yaml.safe_load(new_config)
                    
                    # Save to file
                    with open("config/model_config.yaml", 'w') as f:
                        yaml.dump(updated_config, f)
                    
                    st.success("Configuration saved successfully!")
                    
                except Exception as e:
                    st.error(f"Error saving configuration: {str(e)}")
        
        except Exception as e:
            st.error(f"Error loading configuration: {str(e)}")
    
    with col2:
        st.subheader("System Information")
        
        # Display system info
        st.metric("PyTorch Version", torch.__version__)
        st.metric("CUDA Available", torch.cuda.is_available())
        if torch.cuda.is_available():
            st.metric("CUDA Device", torch.cuda.get_device_name(0))
        
        # Model info
        if st.session_state.model is not None:
            total_params = sum(p.numel() for p in st.session_state.model.parameters())
            st.metric("Model Parameters", f"{total_params:,}")
        
        # Export/Import buttons
        st.subheader("Model Management")
        
        if st.button("Export Model"):
            if st.session_state.model is not None:
                # Save model state
                torch.save(st.session_state.model.state_dict(), "model_checkpoint.pth")
                st.success("Model exported successfully!")
            else:
                st.warning("No model to export")
        
        uploaded_model = st.file_uploader("Import Model", type=['pth'])
        if uploaded_model is not None:
            if st.button("Load Model"):
                try:
                    # Load model state
                    if st.session_state.model is not None:
                        state_dict = torch.load(io.BytesIO(uploaded_model.read()))
                        st.session_state.model.load_state_dict(state_dict)
                        st.success("Model loaded successfully!")
                    else:
                        st.warning("Please initialize a model first")
                except Exception as e:
                    st.error(f"Error loading model: {str(e)}")

def external_apis_page():
    st.header("External TTS APIs")
    st.markdown("Compare your custom TTS models with external services")

    tab1, tab2 = st.tabs(["Quick Test", "API Comparison"])

    with tab1:
        st.subheader("TTS API Test")
        st.markdown("Test any TTS provider with a simple interface")

        col1, col2 = st.columns(2)

        with col1:
            # Provider selection
            provider = st.selectbox(
                "TTS Provider",
                ["ElevenLabs", "Cartesia", "Hume AI", "Play.ht", "Azure", "Google", "Amazon Polly"],
                help="Select which TTS API to use"
            )

            # API Key input
            api_key = st.text_input(
                f"{provider} API Key",
                type="password",
                help=f"Enter your {provider} API key"
            )

            # Additional credentials for specific providers
            if provider == "Play.ht":
                user_id = st.text_input("User ID", type="password")
            elif provider == "Azure":
                region = st.text_input("Region", value="eastus")
            elif provider == "Amazon Polly":
                secret_key = st.text_input("AWS Secret Key", type="password")
                region = st.text_input("AWS Region", value="us-east-1")

            # Text input
            test_text = st.text_area(
                "Text to synthesize:",
                value="Hello! This is a test of the text to speech system.",
                height=100
            )

            # Voice selection (provider-specific defaults)
            voice_defaults = {
                "ElevenLabs": "21m00Tcm4TlvDq8ikWAM",  # Rachel
                "Cartesia": "a0e99841-438c-4a64-b679-ae501e7d6091",
                "Hume AI": "default",
                "Play.ht": "larry",
                "Azure": "en-US-AriaNeural",
                "Google": "en-US-Neural2-A",
                "Amazon Polly": "Joanna"
            }

            voice = st.text_input("Voice ID/Name", value=voice_defaults.get(provider, "default"))

        with col2:
            st.markdown("### Test Results")

            if st.button("🎵 Synthesize", use_container_width=True):
                if not api_key:
                    st.error(f"Please enter your {provider} API key")
                else:
                    try:
                        from external_apis.tts_api_client import get_tts_client
                        import time

                        with st.spinner(f"Synthesizing with {provider}..."):
                            # Get client
                            kwargs = {"api_key": api_key}
                            if provider == "Play.ht" and 'user_id' in locals():
                                kwargs["user_id"] = user_id
                            elif provider == "Azure" and 'region' in locals():
                                kwargs["region"] = region
                            elif provider == "Amazon Polly" and 'secret_key' in locals():
                                kwargs["secret_key"] = secret_key
                                kwargs["region"] = region

                            client = get_tts_client(provider.lower().replace(" ", "").replace(".", ""), **kwargs)

                            # Synthesize
                            start_time = time.time()
                            audio_bytes = client.synthesize(test_text, voice=voice)
                            synthesis_time = time.time() - start_time

                        st.success(f"✅ Synthesis successful with {provider}!")

                        # Metrics
                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.metric("Synthesis Time", f"{synthesis_time:.2f}s")
                        with col_b:
                            st.metric("Audio Size", f"{len(audio_bytes) / 1024:.1f} KB")

                        # Play audio
                        st.audio(audio_bytes, format="audio/wav")

                        # Download button
                        st.download_button(
                            label="📥 Download Audio",
                            data=audio_bytes,
                            file_name=f"{provider.lower()}_synthesis.wav",
                            mime="audio/wav"
                        )

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        import traceback
                        with st.expander("Show error details"):
                            st.code(traceback.format_exc())

    with tab2:
        st.subheader("📈 API Comparison")
        st.markdown("Compare different TTS services and models")
        
        if 'comparison_results' not in st.session_state:
            st.session_state.comparison_results = []
        
        # Comparison interface
        if st.button("🔄 Add Current Test to Comparison"):
            if api_key:
                # This would store the last test result for comparison
                st.info("Feature coming soon: Store and compare multiple API results")
            else:
                st.warning("Run a test first to add to comparison")
        
        # Display comparison table
        if st.session_state.comparison_results:
            df = pd.DataFrame(st.session_state.comparison_results)
            st.dataframe(df)
        else:
            st.info("No comparison data yet. Run tests to start comparing APIs.")

if __name__ == "__main__":
    main()
