from .bayes_net import *

def build_bayesian_network(naive=False):
    """Define BN structure with more tumor features"""
    if naive:
        # Naive Bayes structure: class variable is parent of all features
        model = BayesianNetwork([
            ("tumor_type", "area"),
            ("tumor_type", "perimeter"),
            ("tumor_type", "eccentricity"),
            ("tumor_type", "solidity"),
            ("tumor_type", "contrast"),
            ("tumor_type", "homogeneity"),
            ("tumor_type", "energy"),
            ("tumor_type", "correlation")
        ])
        print("Building Naive Bayes structure...")
    else:
        # Regular Bayesian network structure
        model = BayesianNetwork([
            # tumor_type influences discriminative features directly
            ('tumor_type', 'area'),
            ('tumor_type', 'eccentricity'),
            ('tumor_type', 'homogeneity'),
            ('tumor_type', 'contrast'),
            
            # Feature dependencies based on correlations
            ('area', 'perimeter'),
            ('eccentricity', 'solidity'),
            ('homogeneity', 'energy'),
            ('energy', 'correlation')
        ])
        print("Building regular Bayesian Network structure...")
        print("Nodes:", model.nodes())
        print("Edges:", model.edges())
    return model

def extract_features_from_imagedataset(dataset, output_dir="features/output", split_name="train", chunk_size=1000):
    """
    Extract features directly from an ImageDataset object.
    """
    os.makedirs(output_dir, exist_ok=True)
    features_path = os.path.join(output_dir, f"features_{split_name}.csv")
    if os.path.exists(features_path):
        print(f"Features already extracted for {split_name}. Skipping.")
        return features_path
    
    try:
        if not hasattr(dataset, 'images') or not hasattr(dataset, 'labels'):
            raise ValueError("Dataset object must have 'images' and 'labels' attributes")
        
        images = dataset.images
        labels = dataset.labels
        
        print(f"Processing {len(images)} images from ImageDataset")
        features_list = []
        chunk_counter = 0
        
        # Class mapping
        class_names = ["notumor", "glioma", "meningioma", "pituitary"]
        
        # Process each image with tqdm progress bar
        for idx in tqdm(range(len(images)), desc="Extracting features", unit="img"):
            img = images[idx]
            label_idx = int(labels[idx])
            
            # Map label index to name
            label = class_names[label_idx] if label_idx < len(class_names) else f"class_{label_idx}"
            
            # Extract features
            is_tumor = (label != "notumor")
            feats = segment_and_extract_features(img, is_tumor)
            feats["label"] = label
            feats["split"] = split_name
            features_list.append(feats)
            
            # Save in chunks to manage memory
            if len(features_list) >= chunk_size or idx == len(images) - 1:
                df_chunk = pd.DataFrame(features_list)
                
                mode = 'w' if chunk_counter == 0 else 'a'
                header = True if chunk_counter == 0 else False
                df_chunk.to_csv(features_path, mode=mode, header=header, index=False)
                
                features_list = []
                chunk_counter += 1
        
        return features_path
        
    except Exception as e:
        print(f"Error extracting features: {str(e)}")
        return None

