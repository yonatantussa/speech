import yaml
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Configuration file not found: {config_path}")
        return get_default_config()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        logger.info(f"Configuration loaded from: {config_path}")
        return config
    
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {str(e)}")
        return get_default_config()

def save_config(config: Dict[str, Any], config_path: str) -> bool:
    """
    Save configuration to YAML or JSON file
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
        
    Returns:
        True if successful, False otherwise
    """
    config_path = Path(config_path)
    
    # Create directory if it doesn't exist
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            elif config_path.suffix.lower() == '.json':
                json.dump(config, f, indent=2)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        logger.info(f"Configuration saved to: {config_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error saving configuration to {config_path}: {str(e)}")
        return False

def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration for TTS model
    
    Returns:
        Default configuration dictionary
    """
    return {
        # Model architecture
        'model': {
            'vocab_size': 256,
            'embedding_dim': 512,
            'encoder_dim': 512,
            'decoder_dim': 1024,
            'attention_dim': 128,
            'num_mels': 80,
            'postnet_dim': 512,
            'postnet_layers': 5
        },
        
        # Audio processing
        'audio': {
            'sample_rate': 22050,
            'n_fft': 1024,
            'hop_length': 256,
            'win_length': 1024,
            'n_mels': 80,
            'fmin': 0,
            'fmax': 8000,
            'preemphasis': 0.97,
            'min_level_db': -100,
            'ref_level_db': 20
        },
        
        # Training parameters
        'training': {
            'learning_rate': 0.001,
            'batch_size': 16,
            'num_epochs': 100,
            'gradient_clip_val': 1.0,
            'lr_step_size': 50000,
            'lr_gamma': 0.5,
            'patience': 10,
            'mel_loss_weight': 1.0,
            'postnet_loss_weight': 1.0,
            'stop_loss_weight': 50.0
        },
        
        # Dataset parameters
        'dataset': {
            'max_text_length': 200,
            'max_mel_length': 1000,
            'use_phonemes': False,
            'data_dir': 'data',
            'metadata_file': 'metadata.csv'
        },
        
        # Synthesis parameters
        'synthesis': {
            'max_decoder_steps': 1000,
            'griffin_lim_iters': 50,
            'power': 1.5,
            'use_vocoder': True
        },
        
        # Paths and directories
        'paths': {
            'checkpoint_dir': 'checkpoints',
            'log_dir': 'logs',
            'output_dir': 'outputs',
            'tensorboard_dir': 'tensorboard'
        },
        
        # Logging
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'log_to_file': True,
            'log_file': 'logs/tts_pipeline.log'
        }
    }

def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override base with
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary
    
    Args:
        config: Configuration to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_sections = ['model', 'audio', 'training', 'dataset']
    
    for section in required_sections:
        if section not in config:
            logger.error(f"Missing required configuration section: {section}")
            return False
    
    # Validate model parameters
    model_config = config['model']
    required_model_params = ['vocab_size', 'embedding_dim', 'encoder_dim', 'decoder_dim', 'num_mels']
    
    for param in required_model_params:
        if param not in model_config:
            logger.error(f"Missing required model parameter: {param}")
            return False
        
        if not isinstance(model_config[param], int) or model_config[param] <= 0:
            logger.error(f"Invalid model parameter {param}: must be positive integer")
            return False
    
    # Validate audio parameters
    audio_config = config['audio']
    if audio_config['sample_rate'] <= 0:
        logger.error("Invalid sample_rate: must be positive")
        return False
    
    if audio_config['hop_length'] <= 0:
        logger.error("Invalid hop_length: must be positive")
        return False
    
    # Validate training parameters
    training_config = config['training']
    if training_config['learning_rate'] <= 0:
        logger.error("Invalid learning_rate: must be positive")
        return False
    
    if training_config['batch_size'] <= 0:
        logger.error("Invalid batch_size: must be positive")
        return False
    
    logger.info("Configuration validation passed")
    return True

def setup_directories(config: Dict[str, Any]) -> None:
    """
    Create necessary directories based on configuration
    
    Args:
        config: Configuration dictionary
    """
    paths = config.get('paths', {})
    
    directories = [
        paths.get('checkpoint_dir', 'checkpoints'),
        paths.get('log_dir', 'logs'),
        paths.get('output_dir', 'outputs'),
        paths.get('tensorboard_dir', 'tensorboard')
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {directory}")

def get_config_from_env() -> Dict[str, Any]:
    """
    Get configuration overrides from environment variables
    
    Returns:
        Configuration dictionary with environment variable overrides
    """
    env_config = {}
    
    # Model parameters
    if os.getenv('TTS_BATCH_SIZE'):
        env_config.setdefault('training', {})['batch_size'] = int(os.getenv('TTS_BATCH_SIZE'))
    
    if os.getenv('TTS_LEARNING_RATE'):
        env_config.setdefault('training', {})['learning_rate'] = float(os.getenv('TTS_LEARNING_RATE'))
    
    if os.getenv('TTS_NUM_EPOCHS'):
        env_config.setdefault('training', {})['num_epochs'] = int(os.getenv('TTS_NUM_EPOCHS'))
    
    # Audio parameters
    if os.getenv('TTS_SAMPLE_RATE'):
        env_config.setdefault('audio', {})['sample_rate'] = int(os.getenv('TTS_SAMPLE_RATE'))
    
    if os.getenv('TTS_N_MELS'):
        env_config.setdefault('audio', {})['n_mels'] = int(os.getenv('TTS_N_MELS'))
        env_config.setdefault('model', {})['num_mels'] = int(os.getenv('TTS_N_MELS'))
    
    # Paths
    if os.getenv('TTS_DATA_DIR'):
        env_config.setdefault('dataset', {})['data_dir'] = os.getenv('TTS_DATA_DIR')
    
    if os.getenv('TTS_CHECKPOINT_DIR'):
        env_config.setdefault('paths', {})['checkpoint_dir'] = os.getenv('TTS_CHECKPOINT_DIR')
    
    # Logging
    if os.getenv('TTS_LOG_LEVEL'):
        env_config.setdefault('logging', {})['level'] = os.getenv('TTS_LOG_LEVEL')
    
    return env_config

def load_config_with_overrides(config_path: str, env_overrides: bool = True) -> Dict[str, Any]:
    """
    Load configuration with environment variable overrides
    
    Args:
        config_path: Path to configuration file
        env_overrides: Whether to apply environment variable overrides
        
    Returns:
        Final configuration dictionary
    """
    # Load base configuration
    config = load_config(config_path)
    
    # Apply environment overrides
    if env_overrides:
        env_config = get_config_from_env()
        if env_config:
            config = merge_configs(config, env_config)
            logger.info("Applied environment variable overrides")
    
    # Validate final configuration
    if not validate_config(config):
        logger.warning("Configuration validation failed, using default config")
        config = get_default_config()
    
    # Setup directories
    setup_directories(config)
    
    return config
