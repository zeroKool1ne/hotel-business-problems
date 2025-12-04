from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
from IPython.display import Image, display



# Resolve the project root by going one level up from /src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Resolve the project root by going one level up from /src/
FIG_DIR = PROJECT_ROOT / "reports" / "figures"

# Base data directory and its subfolders
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure that a directory exists.

    This function creates the directory (including parent directories)
    if it does not already exist. Always returns the directory as a Path
    object, regardless of whether it was newly created or previously existed.

    Args:
        path (str | Path): Path to the directory that should be ensured.

    Returns:
        Path: A Path object pointing to the ensured directory.
    """
    # Convert input to Path object
    path = Path(path)

    # Create directory if it does not exist (including parent folders)
    path.mkdir(parents=True, exist_ok=True)
    
    return path


# ---- Figure-Helpers --------------------------------------------------------

def get_fig_path(name: str, ext: str = "png") -> Path:
    """
    Build a consistent path to a figure file.

    Automatically ensures the figure directory exists and appends the
    file extension if the user omitted it.

    Example:
        get_fig_path("lead_time_barplot")
        -> PROJECT_ROOT/reports/figures/lead_time_barplot.png

    Args:
        name (str): Base filename of the figure (with or without extension).
        ext (str, optional): File extension to apply if missing. Defaults to "png".

    Returns:
        Path: Full path to the figure file.
    """
    # Ensure the figure directory exists
    ensure_dir(FIG_DIR)

    # Append extension if user did not include it
    if not name.endswith(f".{ext}"):
        name = f"{name}.{ext}"

    # Return full path to the figure file  
    return FIG_DIR / name


def save_current_fig(name: str, dpi: int = 300, tight: bool = True) -> Path:
    """
    Save the currently active Matplotlib figure.

    Saves the figure inside PROJECT_ROOT/reports/figures/ under the specified name.
    Optionally applies tight layout before saving.

    Args:
        name (str): Name of the figure file (without extension).
        dpi (int, optional): Image resolution. Defaults to 300.
        tight (bool, optional): Apply tight_layout() before saving. Defaults to True.

    Returns:
        Path: Path to the saved figure file.
    """
    # Build the full path where the figure will be saved
    fig_path = get_fig_path(name)
    
    # Apply tight layout if requested
    if tight:
        plt.tight_layout()

    # Save the active Matplotlib figure
    plt.savefig(fig_path, dpi=dpi)

    return fig_path


def show_saved_fig(name: str, ext: str = "png", width: int = 800) -> None:
    """
    Display a previously saved figure inside a Jupyter Notebook.

    Loads the figure from the reports/figures directory and renders it
    inline using IPython.display.

    Example:
        show_saved_fig("lead_time_barplot")

    Args:
        name (str): Base filename of the figure (with or without extension).
        ext (str, optional): Expected figure file extension. Defaults to "png".
        width (int, optional): Display width in pixels. Defaults to 800.

    Raises:
        FileNotFoundError: If the requested figure file does not exist.
    """
    # Build the path to the saved figure file
    fig_path = get_fig_path(name, ext=ext)

    # Ensure the file exists before displaying it
    if not fig_path.exists():
        raise FileNotFoundError(f"Figure not found: {fig_path}")
        
    # Display image inline inside the notebook    
    display(Image(filename=str(fig_path), width=width))


