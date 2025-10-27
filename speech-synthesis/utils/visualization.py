import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import librosa
import librosa.display
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
from typing import Optional, Tuple, List, Dict, Any
import torch

# Set style for matplotlib
plt.style.use('default')
sns.set_palette("husl")

def plot_spectrogram(spectrogram: np.ndarray, 
                    title: str = "Spectrogram",
                    sample_rate: int = 22050,
                    hop_length: int = 256,
                    y_axis: str = 'mel',
                    x_axis: str = 'time',
                    figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
    """
    Plot mel spectrogram using matplotlib
    
    Args:
        spectrogram: Spectrogram array (n_mels, time)
        title: Plot title
        sample_rate: Audio sample rate
        hop_length: Hop length for STFT
        y_axis: Y-axis type ('mel', 'hz', 'linear')
        x_axis: X-axis type ('time', 'frames')
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    img = librosa.display.specshow(
        spectrogram,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis=x_axis,
        y_axis=y_axis,
        ax=ax,
        cmap='viridis'
    )
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    
    plt.tight_layout()
    return fig

def plot_spectrogram_comparison(spec1: np.ndarray, 
                               spec2: np.ndarray,
                               titles: List[str] = ["Original", "Generated"],
                               sample_rate: int = 22050,
                               hop_length: int = 256,
                               figsize: Tuple[int, int] = (15, 8)) -> plt.Figure:
    """
    Plot two spectrograms side by side for comparison
    
    Args:
        spec1: First spectrogram
        spec2: Second spectrogram
        titles: Titles for each spectrogram
        sample_rate: Audio sample rate
        hop_length: Hop length for STFT
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Plot first spectrogram
    img1 = librosa.display.specshow(
        spec1,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel',
        ax=axes[0],
        cmap='viridis'
    )
    axes[0].set_title(titles[0], fontsize=14, fontweight='bold')
    fig.colorbar(img1, ax=axes[0], format='%+2.0f dB')
    
    # Plot second spectrogram
    img2 = librosa.display.specshow(
        spec2,
        sr=sample_rate,
        hop_length=hop_length,
        x_axis='time',
        y_axis='mel',
        ax=axes[1],
        cmap='viridis'
    )
    axes[1].set_title(titles[1], fontsize=14, fontweight='bold')
    fig.colorbar(img2, ax=axes[1], format='%+2.0f dB')
    
    plt.tight_layout()
    return fig

def plot_attention_weights(attention_weights: np.ndarray,
                          text_tokens: Optional[List[str]] = None,
                          title: str = "Attention Weights",
                          figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
    """
    Plot attention alignment matrix
    
    Args:
        attention_weights: Attention weights (decoder_steps, encoder_steps)
        text_tokens: Text tokens for x-axis labels
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Transpose for correct orientation (time on x-axis, text on y-axis)
    attention_weights = attention_weights.T
    
    im = ax.imshow(attention_weights, aspect='auto', origin='lower', cmap='Blues')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Decoder Steps (Time)', fontsize=12)
    ax.set_ylabel('Encoder Steps (Text)', fontsize=12)
    
    # Add text tokens as y-axis labels if provided
    if text_tokens is not None:
        ax.set_yticks(range(len(text_tokens)))
        ax.set_yticklabels(text_tokens, fontsize=10)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', fontsize=12)
    
    plt.tight_layout()
    return fig

def plot_training_metrics(history: Dict[str, List[float]],
                         title: str = "Training Metrics",
                         figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Plot training metrics over time
    
    Args:
        history: Dictionary with metric names and values
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    # Separate training and validation metrics
    train_metrics = {k: v for k, v in history.items() if k.startswith('train_')}
    val_metrics = {k: v for k, v in history.items() if k.startswith('val_')}
    
    # Determine number of subplots needed
    unique_metrics = set()
    for key in train_metrics.keys():
        metric_name = key.replace('train_', '')
        unique_metrics.add(metric_name)
    
    n_metrics = len(unique_metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    axes = axes.flatten()
    
    for i, metric in enumerate(unique_metrics):
        ax = axes[i]
        
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'
        
        if train_key in history:
            epochs = range(1, len(history[train_key]) + 1)
            ax.plot(epochs, history[train_key], label=f'Train {metric}', linewidth=2)
        
        if val_key in history:
            epochs = range(1, len(history[val_key]) + 1)
            ax.plot(epochs, history[val_key], label=f'Val {metric}', linewidth=2)
        
        ax.set_title(f'{metric.title()} Over Time', fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.title())
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_audio_waveform(audio: np.ndarray,
                       sample_rate: int = 22050,
                       title: str = "Audio Waveform",
                       figsize: Tuple[int, int] = (12, 4)) -> plt.Figure:
    """
    Plot audio waveform
    
    Args:
        audio: Audio signal
        sample_rate: Sample rate
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    time_axis = np.linspace(0, len(audio) / sample_rate, len(audio))
    
    ax.plot(time_axis, audio, linewidth=0.5, color='blue')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Amplitude', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_f0_contour(f0: np.ndarray,
                   time_axis: Optional[np.ndarray] = None,
                   title: str = "F0 Contour",
                   figsize: Tuple[int, int] = (12, 4)) -> plt.Figure:
    """
    Plot fundamental frequency contour
    
    Args:
        f0: F0 values
        time_axis: Time axis (if None, use frame indices)
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if time_axis is None:
        time_axis = np.arange(len(f0))
    
    # Only plot voiced regions (non-zero F0)
    voiced_mask = f0 > 0
    
    ax.plot(time_axis[voiced_mask], f0[voiced_mask], 'b-', linewidth=2, label='F0')
    ax.scatter(time_axis[voiced_mask], f0[voiced_mask], c='red', s=10, alpha=0.6)
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (frames)' if time_axis is None else 'Time (seconds)', fontsize=12)
    ax.set_ylabel('F0 (Hz)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig

def plot_energy_contour(energy: np.ndarray,
                       time_axis: Optional[np.ndarray] = None,
                       title: str = "Energy Contour",
                       figsize: Tuple[int, int] = (12, 4)) -> plt.Figure:
    """
    Plot energy contour
    
    Args:
        energy: Energy values
        time_axis: Time axis (if None, use frame indices)
        title: Plot title
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if time_axis is None:
        time_axis = np.arange(len(energy))
    
    ax.plot(time_axis, energy, 'g-', linewidth=2, label='Energy')
    ax.fill_between(time_axis, energy, alpha=0.3, color='green')
    
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (frames)' if time_axis is None else 'Time (seconds)', fontsize=12)
    ax.set_ylabel('Energy (dB)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    return fig

def create_interactive_spectrogram(spectrogram: np.ndarray,
                                  title: str = "Interactive Spectrogram",
                                  sample_rate: int = 22050,
                                  hop_length: int = 256) -> go.Figure:
    """
    Create interactive spectrogram using Plotly
    
    Args:
        spectrogram: Spectrogram array
        title: Plot title
        sample_rate: Sample rate
        hop_length: Hop length
        
    Returns:
        Plotly Figure object
    """
    # Create time and frequency axes
    time_frames = spectrogram.shape[1]
    freq_bins = spectrogram.shape[0]
    
    time_axis = np.linspace(0, time_frames * hop_length / sample_rate, time_frames)
    freq_axis = np.linspace(0, sample_rate // 2, freq_bins)
    
    fig = go.Figure(data=go.Heatmap(
        z=spectrogram,
        x=time_axis,
        y=freq_axis,
        colorscale='Viridis',
        colorbar=dict(title="dB")
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Time (seconds)",
        yaxis_title="Frequency (Hz)",
        width=800,
        height=500
    )
    
    return fig

def create_interactive_attention(attention_weights: np.ndarray,
                               text_tokens: Optional[List[str]] = None,
                               title: str = "Interactive Attention Weights") -> go.Figure:
    """
    Create interactive attention visualization using Plotly
    
    Args:
        attention_weights: Attention weights matrix
        text_tokens: Text tokens for axis labels
        title: Plot title
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure(data=go.Heatmap(
        z=attention_weights.T,  # Transpose for correct orientation
        colorscale='Blues',
        colorbar=dict(title="Attention Weight")
    ))
    
    if text_tokens is not None:
        fig.update_layout(
            yaxis=dict(
                tickmode='array',
                tickvals=list(range(len(text_tokens))),
                ticktext=text_tokens
            )
        )
    
    fig.update_layout(
        title=title,
        xaxis_title="Decoder Steps (Time)",
        yaxis_title="Encoder Steps (Text)",
        width=800,
        height=600
    )
    
    return fig

def create_training_dashboard(history: Dict[str, List[float]]) -> go.Figure:
    """
    Create interactive training dashboard using Plotly
    
    Args:
        history: Training history dictionary
        
    Returns:
        Plotly Figure object with subplots
    """
    # Extract unique metrics
    train_metrics = {k: v for k, v in history.items() if k.startswith('train_')}
    val_metrics = {k: v for k, v in history.items() if k.startswith('val_')}
    
    unique_metrics = set()
    for key in train_metrics.keys():
        metric_name = key.replace('train_', '')
        unique_metrics.add(metric_name)
    
    # Create subplots
    n_metrics = len(unique_metrics)
    n_cols = 2
    n_rows = (n_metrics + 1) // 2
    
    subplot_titles = [f'{metric.title()} Over Time' for metric in unique_metrics]
    
    fig = make_subplots(
        rows=n_rows, 
        cols=n_cols,
        subplot_titles=subplot_titles
    )
    
    for i, metric in enumerate(unique_metrics):
        row = i // n_cols + 1
        col = i % n_cols + 1
        
        train_key = f'train_{metric}'
        val_key = f'val_{metric}'
        
        if train_key in history:
            epochs = list(range(1, len(history[train_key]) + 1))
            fig.add_trace(
                go.Scatter(
                    x=epochs, 
                    y=history[train_key], 
                    name=f'Train {metric}',
                    line=dict(width=2)
                ),
                row=row, col=col
            )
        
        if val_key in history:
            epochs = list(range(1, len(history[val_key]) + 1))
            fig.add_trace(
                go.Scatter(
                    x=epochs, 
                    y=history[val_key], 
                    name=f'Val {metric}',
                    line=dict(width=2)
                ),
                row=row, col=col
            )
    
    fig.update_layout(
        height=300 * n_rows,
        title_text="Training Metrics Dashboard",
        showlegend=True
    )
    
    # Update x-axis labels
    for i in range(1, n_rows + 1):
        for j in range(1, n_cols + 1):
            fig.update_xaxes(title_text="Epoch", row=i, col=j)
    
    return fig

def plot_model_architecture(model, input_shape: Tuple[int, ...], 
                           title: str = "Model Architecture") -> plt.Figure:
    """
    Visualize model architecture (simplified representation)
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
        title: Plot title
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # This is a simplified visualization
    # In practice, you might want to use tools like torchviz or tensorboard
    
    layer_names = []
    layer_params = []
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            param_count = sum(p.numel() for p in module.parameters())
            if param_count > 0:
                layer_names.append(name)
                layer_params.append(param_count)
    
    # Create horizontal bar chart
    y_pos = np.arange(len(layer_names))
    
    bars = ax.barh(y_pos, layer_params, color='skyblue', alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(layer_names, fontsize=10)
    ax.set_xlabel('Number of Parameters', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add parameter count labels
    for i, (bar, param_count) in enumerate(zip(bars, layer_params)):
        ax.text(bar.get_width() + max(layer_params) * 0.01, 
               bar.get_y() + bar.get_height()/2, 
               f'{param_count:,}', 
               ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig

def save_plot(fig, filepath: str, dpi: int = 300) -> None:
    """
    Save matplotlib figure to file
    
    Args:
        fig: matplotlib Figure object
        filepath: Output file path
        dpi: Resolution in dots per inch
    """
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close(fig)

def save_interactive_plot(fig: go.Figure, filepath: str) -> None:
    """
    Save Plotly figure to HTML file
    
    Args:
        fig: Plotly Figure object
        filepath: Output file path
    """
    fig.write_html(filepath)
