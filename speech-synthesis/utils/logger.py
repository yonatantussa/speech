import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import colorlog
from datetime import datetime

def setup_logger(name: str = 'tts_pipeline', 
                level: str = 'INFO',
                log_file: Optional[str] = None,
                format_string: Optional[str] = None,
                use_colors: bool = True) -> logging.Logger:
    """
    Setup a comprehensive logger with console and file output
    
    Args:
        name: Logger name
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Optional file path for logging to file
        format_string: Custom format string
        use_colors: Whether to use colored output for console
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Clear existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Set logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    if use_colors:
        # Colored formatter for console
        color_formatter = colorlog.ColoredFormatter(
            '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s%(reset)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
        console_handler.setFormatter(color_formatter)
    else:
        # Standard formatter
        formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
        console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_handler.setLevel(numeric_level)
        
        file_formatter = logging.Formatter(format_string, datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        
        logger.addHandler(file_handler)
    
    return logger

def setup_training_logger(experiment_name: str,
                         log_dir: str = 'logs',
                         level: str = 'INFO') -> logging.Logger:
    """
    Setup logger specifically for training experiments
    
    Args:
        experiment_name: Name of the experiment
        log_dir: Directory for log files
        level: Logging level
        
    Returns:
        Configured logger for training
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f"{log_dir}/training_{experiment_name}_{timestamp}.log"
    
    logger = setup_logger(
        name=f'training_{experiment_name}',
        level=level,
        log_file=log_file,
        format_string='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )
    
    logger.info(f"Training logger initialized for experiment: {experiment_name}")
    logger.info(f"Log file: {log_file}")
    
    return logger

def log_model_info(logger: logging.Logger, model, config: Dict[str, Any]) -> None:
    """
    Log comprehensive model information
    
    Args:
        logger: Logger instance
        model: PyTorch model
        config: Model configuration dictionary
    """
    logger.info("=" * 60)
    logger.info("MODEL INFORMATION")
    logger.info("=" * 60)
    
    # Model architecture
    logger.info(f"Model type: {type(model).__name__}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Model size estimation
    param_size = total_params * 4  # Assuming float32
    buffer_size = sum(buf.numel() * 4 for buf in model.buffers())
    total_size = param_size + buffer_size
    
    logger.info(f"Estimated model size: {total_size / 1024 / 1024:.2f} MB")
    
    # Configuration
    logger.info("Model configuration:")
    for key, value in config.items():
        if isinstance(value, dict):
            logger.info(f"  {key}:")
            for sub_key, sub_value in value.items():
                logger.info(f"    {sub_key}: {sub_value}")
        else:
            logger.info(f"  {key}: {value}")
    
    logger.info("=" * 60)

def log_training_step(logger: logging.Logger,
                     epoch: int,
                     step: int,
                     total_steps: int,
                     losses: Dict[str, float],
                     learning_rate: float,
                     step_time: float) -> None:
    """
    Log training step information
    
    Args:
        logger: Logger instance
        epoch: Current epoch
        step: Current step
        total_steps: Total steps in epoch
        losses: Dictionary of loss values
        learning_rate: Current learning rate
        step_time: Time taken for this step
    """
    progress_pct = (step / total_steps) * 100
    
    loss_str = " | ".join([f"{k}: {v:.4f}" for k, v in losses.items()])
    
    logger.info(
        f"Epoch {epoch:3d} | "
        f"Step {step:4d}/{total_steps:4d} ({progress_pct:5.1f}%) | "
        f"{loss_str} | "
        f"LR: {learning_rate:.2e} | "
        f"Time: {step_time:.2f}s"
    )

def log_epoch_summary(logger: logging.Logger,
                     epoch: int,
                     train_losses: Dict[str, float],
                     val_losses: Optional[Dict[str, float]] = None,
                     epoch_time: float = 0.0,
                     is_best: bool = False) -> None:
    """
    Log epoch summary information
    
    Args:
        logger: Logger instance
        epoch: Current epoch
        train_losses: Training losses
        val_losses: Validation losses (optional)
        epoch_time: Time taken for epoch
        is_best: Whether this is the best model so far
    """
    logger.info("=" * 80)
    logger.info(f"EPOCH {epoch} SUMMARY")
    logger.info("=" * 80)
    
    # Training losses
    logger.info("Training losses:")
    for loss_name, loss_value in train_losses.items():
        logger.info(f"  {loss_name}: {loss_value:.6f}")
    
    # Validation losses
    if val_losses:
        logger.info("Validation losses:")
        for loss_name, loss_value in val_losses.items():
            logger.info(f"  {loss_name}: {loss_value:.6f}")
    
    # Timing
    if epoch_time > 0:
        logger.info(f"Epoch time: {epoch_time:.2f}s")
    
    # Best model indicator
    if is_best:
        logger.info("🏆 NEW BEST MODEL!")
    
    logger.info("=" * 80)

def log_synthesis_info(logger: logging.Logger,
                      text: str,
                      audio_duration: float,
                      synthesis_time: float,
                      sample_rate: int,
                      model_params: Dict[str, Any]) -> None:
    """
    Log speech synthesis information
    
    Args:
        logger: Logger instance
        text: Input text
        audio_duration: Generated audio duration
        synthesis_time: Time taken for synthesis
        sample_rate: Audio sample rate
        model_params: Model parameters used
    """
    rtf = synthesis_time / audio_duration if audio_duration > 0 else float('inf')
    
    logger.info("=" * 60)
    logger.info("SYNTHESIS INFORMATION")
    logger.info("=" * 60)
    logger.info(f"Input text: '{text[:100]}{'...' if len(text) > 100 else ''}'")
    logger.info(f"Text length: {len(text)} characters")
    logger.info(f"Audio duration: {audio_duration:.2f}s")
    logger.info(f"Sample rate: {sample_rate} Hz")
    logger.info(f"Synthesis time: {synthesis_time:.2f}s")
    logger.info(f"Real-time factor: {rtf:.2f}x")
    
    if model_params:
        logger.info("Model parameters:")
        for key, value in model_params.items():
            logger.info(f"  {key}: {value}")
    
    logger.info("=" * 60)

def log_evaluation_results(logger: logging.Logger,
                          metrics: Dict[str, float],
                          test_name: str = "Evaluation") -> None:
    """
    Log evaluation results
    
    Args:
        logger: Logger instance
        metrics: Dictionary of evaluation metrics
        test_name: Name of the evaluation test
    """
    logger.info("=" * 60)
    logger.info(f"{test_name.upper()} RESULTS")
    logger.info("=" * 60)
    
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            logger.info(f"{metric_name}: {metric_value:.4f}")
        else:
            logger.info(f"{metric_name}: {metric_value}")
    
    logger.info("=" * 60)

def log_system_info(logger: logging.Logger) -> None:
    """
    Log system and environment information
    
    Args:
        logger: Logger instance
    """
    import platform
    import torch
    import sys
    
    logger.info("=" * 60)
    logger.info("SYSTEM INFORMATION")
    logger.info("=" * 60)
    
    # Python and system info
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Architecture: {platform.architecture()}")
    logger.info(f"Processor: {platform.processor()}")
    
    # PyTorch info
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    logger.info("=" * 60)

class TrainingProgressLogger:
    """
    Context manager for tracking training progress
    """
    
    def __init__(self, logger: logging.Logger, total_epochs: int):
        self.logger = logger
        self.total_epochs = total_epochs
        self.start_time = None
        self.epoch_start_time = None
        
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.info(f"Training started at {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Total epochs: {self.total_epochs}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        total_time = end_time - self.start_time
        
        self.logger.info("=" * 60)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Total training time: {total_time}")
        self.logger.info(f"Average time per epoch: {total_time / self.total_epochs}")
        self.logger.info("=" * 60)
        
        if exc_type is not None:
            self.logger.error(f"Training interrupted due to: {exc_type.__name__}: {exc_val}")
    
    def start_epoch(self, epoch: int):
        self.epoch_start_time = datetime.now()
        self.logger.info(f"\nStarting epoch {epoch}/{self.total_epochs}")
        
    def end_epoch(self, epoch: int):
        if self.epoch_start_time:
            epoch_time = datetime.now() - self.epoch_start_time
            self.logger.info(f"Epoch {epoch} completed in {epoch_time}")
