from .bayes_net import *

def visualize(saved_path=None, model=None, args=None):
    """
    Visualize Bayesian Network model.
    
    Parameters:
    -----------
    saved_path : str or Path, optional
        Path to the saved model
    model : BayesianNetwork, optional
        Already loaded model instance
    args : dict, optional
        Visualization arguments
    """
    
    print(model)
    print(args)
    if model is None and saved_path:
        from utils.utils import load_pkl
        model = load_pkl(saved_path)
    
    if model is None:
        print("No model available for visualization")
        return
        
    # Get visualization parameters
    if args is None:
        print("No visualization arguments provided")
        args = {}
        
    viz_config = args.get("visualization", {})
    if not isinstance(viz_config, dict):
        viz_config = {}
        
    # Check if visualization is enabled
    if not viz_config.get("enabled", False):
        return
        
    viz_type = viz_config.get("type", "network")
    output_file = viz_config.get("output_file", None)
    show_viz = viz_config.get("show", False)
    
    # Create output directory if needed
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    if viz_type == "network":
        print("Visualizing Bayesian Network structure...")
        visualize_bayesian_network(model, output_file=output_file, show=show_viz)
    else:
        print(f"Unknown visualization type: {viz_type}")
    print("Visualization successful.")