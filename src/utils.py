# src/utils.py

from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
from IPython.display import Image, display


# ---- Paths -----------------------------------------------------------------

# Projekt-Root = Ordner, der "src" enthält
PROJECT_ROOT = Path(__file__).resolve().parents[1]

FIG_DIR = PROJECT_ROOT / "reports" / "figures"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Erzeugt den Ordner, falls er noch nicht existiert.
    Gibt immer ein Path-Objekt zurück.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---- Figure-Helpers --------------------------------------------------------

def get_fig_path(name: str, ext: str = "png") -> Path:
    """
    Erzeuge einen konsistenten Pfad für eine Figure,
    z.B. get_fig_path("lead_time_barplot") -> PROJECT_ROOT/figures/lead_time_barplot.png
    """
    ensure_dir(FIG_DIR)
    if not name.endswith(f".{ext}"):
        name = f"{name}.{ext}"
    return FIG_DIR / name


def save_current_fig(name: str, dpi: int = 300, tight: bool = True) -> Path:
    """
    Speichert die aktuell aktive Matplotlib-Figure unter figures/<name>.png
    und gibt den Pfad zurück.
    """
    fig_path = get_fig_path(name)
    if tight:
        plt.tight_layout()
    plt.savefig(fig_path, dpi=dpi)
    return fig_path


def show_saved_fig(name: str, ext: str = "png", width: int = 800) -> None:
    """
    Zeigt eine zuvor gespeicherte Figure im Notebook an.
    Beispiel: show_saved_fig("lead_time_barplot")
    """
    fig_path = get_fig_path(name, ext=ext)
    if not fig_path.exists():
        raise FileNotFoundError(f"Figure not found: {fig_path}")
    display(Image(filename=str(fig_path), width=width))


