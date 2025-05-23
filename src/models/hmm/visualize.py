# HMM Visualization Script 
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any

from utils.logger import get_logger
# Import base HMM class for type hinting, handle potential import error
try:
    from hmmlearn import hmm 
except ImportError:
    hmm = None # Set to None if hmmlearn is not installed

logger = get_logger("hmm_visualize")

DEFAULT_VISUALIZATION_ARGS = {
    "enabled": True,
    "output_dir": "src/results/figures/hmm", # Default updated path
    "show": False
}

def visualize(
    saved_path: Path, 
    args: Dict[str, Any] = None
) -> None:
    """Visualizes the transition matrices of the trained HMM models.

    Loads the dictionary of HMM models and generates a heatmap for the 
    transition matrix of each class-specific HMM.

    Args:
        saved_path (Path): Path to the saved joblib file containing the dictionary of trained HMMs.
        args (Dict[str, Any]): Dictionary containing model and visualization arguments from config.
                                Expects nested keys like args['visualization']['output_dir'].
    """
    
    if hmm is None:
        logger.error("Package 'hmmlearn' is required for HMM visualization but is not installed. Cannot proceed.")
        return
        
    vis_config = DEFAULT_VISUALIZATION_ARGS.copy()
    # Load visualization settings specifically from the 'visualization' sub-dictionary if present
    if args and isinstance(args.get("visualization"), dict):
        vis_config.update(args["visualization"])

    if not vis_config.get("enabled", False):
        logger.info("HMM visualization is disabled in the configuration.")
        return

    if not saved_path.is_file():
        logger.error(f"Saved model file not found at {saved_path} for visualization.")
        return

    logger.info(f"Loading HMM models from {saved_path} for visualization...")
    try:
        # Type hint assumes GaussianHMM, adjust if other HMM types are used
        trained_models: Dict[Any, hmm.GaussianHMM] = joblib.load(saved_path)
    except Exception as e:
        logger.error(f"Failed to load models from {saved_path} for visualization: {e}")
        return

    if not trained_models:
        logger.warning("No trained HMM models found in the loaded file.")
        return

    output_dir = Path(vis_config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating HMM transition matrix heatmaps...")
    plot_generated = False
    for cls_label, model in trained_models.items():
        # Check if the model has the transition matrix attribute
        if not hasattr(model, "transmat_") or model.transmat_ is None:
            logger.warning(f"Model for class '{cls_label}' does not have a trained transition matrix (transmat_). Skipping visualization for this class.")
            continue

        n_components = model.n_components
        transition_matrix = model.transmat_

        plt.figure(figsize=(max(6, n_components * 1.5), max(5, n_components * 1.2)))
        sns.heatmap(transition_matrix, annot=True, fmt=".3f", cmap="viridis", 
                    xticklabels=[f"State {i}" for i in range(n_components)], 
                    yticklabels=[f"State {i}" for i in range(n_components)],
                    linewidths=.5)
        plt.title(f"HMM Transition Matrix - Class '{cls_label}' (States: {n_components})")
        plt.xlabel("To State")
        plt.ylabel("From State")
        plt.tight_layout()
        
        # Sanitize class label for filename (replace spaces, slashes etc.)
        safe_cls_label = str(cls_label).replace("/", "_").replace("\\", "_").replace(" ", "_")
        save_fig_path = output_dir / f"hmm_transition_matrix_class_{safe_cls_label}.png"
        
        try:
            plt.savefig(save_fig_path, bbox_inches="tight")
            logger.info(f"Saved transition matrix heatmap to {save_fig_path}")
            plot_generated = True
        except Exception as e:
             logger.error(f"Failed to save heatmap for class {cls_label} to {save_fig_path}: {e}")

        # Show plot if configured, otherwise close to free memory
        if vis_config.get("show", False):
            plt.show()
        else:
             plt.close() 

    if not plot_generated:
         logger.warning("No transition matrices were generated for any class.")
         
    logger.info("HMM visualization finished.")

    # --- Installation Note --- 
    try:
        import matplotlib
        import seaborn
    except ImportError:
        logger.warning("Packages 'matplotlib' and 'seaborn' are required for visualization. Please install them: pip install matplotlib seaborn") 