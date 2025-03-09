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