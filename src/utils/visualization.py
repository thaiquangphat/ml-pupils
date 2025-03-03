import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path
import seaborn as sns

def visualize_bayesian_network(model, output_file=None, show=True):
    """
    Visualize the structure of a Bayesian Network model.
    """
    # Create a directed graph from the model
    G = nx.DiGraph()
    
    # Add nodes and edges
    for node in model.nodes():
        G.add_node(node)
    
    for edge in model.edges():
        G.add_edge(edge[0], edge[1])
    
    # Set up the plot
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)  # For reproducible layout
    
    # Draw nodes with different colors for tumor_type vs features
    node_colors = ['lightblue' if node != 'tumor_type' else 'salmon' 
                  for node in G.nodes()]
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color=node_colors, alpha=0.8)
    nx.draw_networkx_edges(G, pos, arrowsize=20, width=2, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
    
    plt.title("Bayesian Network Model Structure", fontsize=16)
    plt.axis('off')
    
    # Save if an output file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to {output_file}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()

def visualize_feature_relations(features_path=None, npz_data_dir=None, output_file=None, show=True):
    """
    Visualize relationships between features using correlation matrix.
    Works with either features CSV or processed NPZ files.
    
    Parameters:
    -----------
    features_path : str, optional
        Path to CSV file with extracted features
    npz_data_dir : str, optional
        Path to directory containing processed NPZ files
    output_file : str, optional
        Path to save the visualization
    show : bool
        Whether to display the plot
    """
    # Data collection logic - either from CSV or NPZ
    if features_path and os.path.exists(features_path):
        # Load from features CSV
        df = pd.read_csv(features_path)
        features_df = df.select_dtypes(include=[np.number]).drop(['split'], axis=1, errors='ignore')
    
    elif npz_data_dir and os.path.exists(npz_data_dir):
        # Load from NPZ files and extract features on-the-fly
        from utils.feature_extraction import extract_features_from_npz
        
        # Create temporary features file
        temp_features = "temp_features.csv"
        extract_features_from_npz(npz_data_dir, temp_features)
        
        # Load the features
        df = pd.read_csv(temp_features)
        features_df = df.select_dtypes(include=[np.number]).drop(['split'], axis=1, errors='ignore')
        
        # Clean up temp file
        try:
            os.remove(temp_features)
        except:
            pass
    
    else:
        raise ValueError("Either features_path or npz_data_dir must be provided")
    
    # Create correlation matrix
    corr = features_df.corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    
    # Use seaborn if available
    try:
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='coolwarm', vmax=1, vmin=-1, center=0,
                   square=True, linewidths=.5, annot=True, fmt=".2f")
    except ImportError:
        plt.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        plt.colorbar()
        plt.xticks(range(len(corr)), corr.columns, rotation=90)
        plt.yticks(range(len(corr)), corr.columns)
    
    plt.title("Feature Correlation Matrix", fontsize=16)
    plt.tight_layout()
    
    # Save if output file is provided
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Correlation matrix saved to {output_file}")
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close()

# def visualize_class_distribution(npz_data_dir=None, features_path=None, output_file=None, show=True):
#     """
#     Visualize class distribution from dataset.
    
#     Parameters:
#     -----------
#     npz_data_dir : str, optional
#         Path to directory containing processed NPZ files
#     features_path : str, optional
#         Path to CSV file with extracted features
#     output_file : str, optional
#         Path to save the visualization
#     show : bool
#         Whether to display the plot
#     """
#     class_names = ["No Tumor", "Glioma", "Meningioma", "Pituitary"]
    
#     if npz_data_dir and os.path.exists(npz_data_dir):
#         # Load from NPZ files
#         npz_data_dir = Path(npz_data_dir)
#         train_path = npz_data_dir / "train.npz"
#         test_path = npz_data_dir / "test.npz"
        
#         train_data = np.load(train_path) if os.path.exists(train_path) else None
#         test_data = np.load(test_path) if os.path.exists(test_path) else None
        
#         # Count classes
#         counts = {'Training': {}, 'Testing': {}}
        
#         if train_data is not None:
#             y_train = train_data['y']
#             for i in range(4):  # 4 classes
#                 counts['Training'][class_names[i]] = np.sum(y_train == i)
        
#         if test_data is not None:
#             y_test = test_data['y']
#             for i in range(4):
#                 counts['Testing'][class_names[i]] = np.sum(y_test == i)
    
#     elif features_path and os.path.exists(features_path):
#         # Load from features CSV
#         df = pd.read_csv(features_path)
#         # Count classes
#         counts = df.groupby(['split', 'label']).size().unstack(fill_value=0).to_dict()
    
#     else:
#         raise ValueError("Either npz_data_dir or features_path must be provided")
    
#     # Plot class distribution
#     plt.figure(figsize=(12, 6))
    
#     if npz_data_dir:
#         # For NPZ data
#         x = np.arange(len(class_names))
#         width = 0.35
        
#         train_counts = [counts['Training'].get(class_name, 0) for class_name in class_names]
#         test_counts = [counts['Testing'].get(class_name, 0) for class_name in class_names]
        
#         plt.bar(x - width/2, train_counts, width, label='Training')
#         plt.bar(x + width/2, test_counts, width, label='Testing')
        
#         plt.xlabel('Class')
#         plt.ylabel('Count')
#         plt.title('Class Distribution')
#         plt.xticks(x, class_names)
#         plt.legend()
    
#     else:
#         # For features CSV
#         df_train = df[df['split'] == 'Training']['label'].value_counts().sort_index()
#         df_test = df[df['split'] == 'Testing']['label'].value_counts().sort_index()
        
#         x = np.arange(len(df_train))
#         width = 0.35
        
#         plt.bar(x - width/2, df_train.values, width, label='Training')
#         plt.bar(x + width/2, df_test.values, width, label='Testing')
        
#         plt.xlabel('Class')
#         plt.ylabel('Count')
#         plt.title('Class Distribution')
#         plt.xticks(x, df_train.index)
#         plt.legend()
    
#     plt.tight_layout()
    
#     # Save if output file is provided
#     if output_file:
#         plt.savefig(output_file, dpi=300, bbox_inches='tight')
#         print(f"Class distribution saved to {output_file}")
    
#     # Show if requested
#     if show:
#         plt.show()
#     else:
#         plt.close()

# # Add a new function to show feature distributions by class
# def visualize_feature_distributions(features_path, output_dir=None, show=True):
#     """
#     Visualize the distribution of each feature across different tumor types.
    
#     Parameters:
#     -----------
#     features_path : str
#         Path to CSV file with extracted features
#     output_dir : str, optional
#         Directory to save visualizations
#     show : bool
#         Whether to display plots
#     """
#     df = pd.read_csv(features_path)
    
#     # Create output directory if needed
#     if output_dir:
#         os.makedirs(output_dir, exist_ok=True)
    
#     # Get numerical feature columns
#     feature_cols = ['area', 'perimeter', 'mean_intensity', 
#                    'eccentricity', 'solidity', 'contrast', 'homogeneity']
    
#     # Create distribution plot for each feature
#     for feature in feature_cols:
#         plt.figure(figsize=(10, 6))
        
#         # Plot distributions by tumor type
#         for tumor_type in df['label'].unique():
#             subset = df[df['label'] == tumor_type]
#             if len(subset) > 0:  # Check if we have data
#                 sns.kdeplot(data=subset, x=feature, label=tumor_type)
        
#         plt.title(f"Distribution of {feature} by Tumor Type", fontsize=14)
#         plt.xlabel(feature, fontsize=12)
#         plt.ylabel("Density", fontsize=12)
#         plt.legend()
#         plt.tight_layout()
        
#         # Save if output_dir is provided
#         if output_dir:
#             plt.savefig(os.path.join(output_dir, f"{feature}_distribution.png"))
            
#         if show:
#             plt.show()
#         else:
#             plt.close()