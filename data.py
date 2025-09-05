import re
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import logging
import gc

from config import ProjectConfig


logger = logging.getLogger(__name__)

class EmbeddingExtractor:
    """Unified embedding extraction with proper memory management"""
    
    def __init__(self, device, keep_on_gpu=True):
        self.device = device
        self.keep_on_gpu = keep_on_gpu
    
    def _get_model_outputs(self, model, inputs):
        with torch.no_grad():
            if hasattr(model, 'esm_model') and hasattr(model, 'mlp'):
                # ESMFeatureExtractionModel (LoRA)
                return model.esm_model(**inputs)
            else:
                # Traditional ESM-2
                return model(**inputs)
    
    def generate_embeddings(self, data, model, tokenizer, batch_size=16):
        X = []
        y = []
        
        model.eval()
        is_esm_feature_model = hasattr(model, 'esm_model') and hasattr(model, 'mlp')
        print(f"Model type detected: ESMFeatureExtractionModel={is_esm_feature_model}")
        
        for i in tqdm(range(0, len(data), batch_size), desc="Generating embeddings (optimized)"):
            batch_data = data[i:i + batch_size]
            
            try:
                seq1_list = [item['seq1'] for item in batch_data]
                seq2_list = [item['seq2'] for item in batch_data]
                labels_list = [item['interaction'] for item in batch_data]
                
                seq1_inputs = tokenizer(seq1_list, return_tensors="pt", padding=True, truncation=True, max_length=512)
                seq2_inputs = tokenizer(seq2_list, return_tensors="pt", padding=True, truncation=True, max_length=512)
                
                # -> GPU
                seq1_inputs = {k: v.to(self.device, non_blocking=True) for k, v in seq1_inputs.items()}
                seq2_inputs = {k: v.to(self.device, non_blocking=True) for k, v in seq2_inputs.items()}
                
                seq1_outputs = self._get_model_outputs(model, seq1_inputs)
                seq2_outputs = self._get_model_outputs(model, seq2_inputs)
                
                # mean pooling
                seq1_embeddings = seq1_outputs.last_hidden_state.mean(dim=1)
                seq2_embeddings = seq2_outputs.last_hidden_state.mean(dim=1)
                
                combined_embeddings = torch.cat([seq1_embeddings, seq2_embeddings], dim=1)
                batch_embeddings = combined_embeddings.cpu().numpy()
            
                for j, embedding in enumerate(batch_embeddings):
                    X.append(embedding)
                    y.append(labels_list[j])
                    
            except Exception as e:
                print(f"Batch processing failed at batch {i//batch_size + 1}: {e}")
                continue 
        
        if len(X) == 0:
            print(f"ERROR: No embeddings generated. Total batches processed: {len(data)//batch_size + 1}")
            print(f"Model type: {type(model)}, Device: {self.device}")
            raise RuntimeError("Failed to generate any embeddings. All processing attempts failed.")
        
        return np.array(X), np.array(y)
    
    def cleanup(self):
        """Clean up resources"""
        torch.cuda.empty_cache()
        gc.collect()

def parse(txt):
    """Parse protein interaction data from text"""
    lines = txt.strip().split('\n')
    
    if len(lines) < 5:
        return None
    
    # Extract interaction label from line 3 (>Interaction: 1 & 0)
    interaction_line = lines[2]
    interaction = 1 if 'Interaction: 1' in interaction_line else 0
    
    # Extract sequences from lines 4 and 5
    seq1 = lines[3].strip()
    seq2 = lines[4].strip()
    
    return {'seq1': seq1, 'seq2': seq2, 'interaction': interaction}

def load_data(file_path):
    """Load and parse data from file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Split by double newlines to separate entries
        entries = content.strip().split('\n\n')
        data = []
        
        for entry in entries:
            parsed = parse(entry)
            if parsed:
                data.append(parsed)
        
        print(f"Loaded {len(data)} protein pairs")
        return data
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Failed to load data: {e}")


def apply_smote(X, y, random_state=None):
    """Apply SMOTE for handling class imbalance"""
    if random_state is None:
        defaults = ProjectConfig.get_default_config()
        random_state = defaults['data']['random_state']
    
    y = y.astype(int)
    
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    y_resampled = np.round(y_resampled).astype(int)
    y_resampled = np.clip(y_resampled, 0, 1)
    
    print(f"SMOTE applied: {len(X)} -> {len(X_resampled)} samples")
    return X_resampled, y_resampled


# ============================================================================
# Plot
# ============================================================================

def visualize_embeddings_pca(X, y, output_path=None, n_components=2, figsize=None, dpi=None, alpha=None):
    """Visualize embeddings using PCA
    Args:
        X: Input embeddings (n_samples, n_features)
        y: Labels (n_samples,)
    """
    defaults = ProjectConfig.get_default_config()
    
    figsize = figsize or defaults['visualization']['figsize']
    dpi = dpi or defaults['visualization']['dpi']
    alpha = alpha or defaults['visualization']['alpha']
    
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy() # scikit-learn only works in numpy no PyTorch, need GPU ->CPU
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    
    plt.figure(figsize=figsize)
    
    if n_components == 2:
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=alpha)
        plt.colorbar(scatter, label='Interaction')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    
    plt.title('PCA Visualization of Protein Embeddings')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"PCA visualization saved to {output_path}")
    
    plt.close()
    return X_pca, pca.explained_variance_ratio_
