import os
import json
from datetime import datetime
from typing import Dict, Any
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy arrays and other numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

class ProjectConfig:
    """Unified project configuration management"""
    
    # Model mappings
    MODEL_MAPPING = {
        '6': 'facebook/esm2_t6_8M_UR50D',
        '12': 'facebook/esm2_t12_35M_UR50D', 
        '30': 'facebook/esm2_t30_150M_UR50D',
        '33': 'facebook/esm2_t33_650M_UR50D'
    }
    
    EXPERIMENT_NAMES = {
        'exp1': 'Train(INTRA₀) → Test(INTRA₁)',
        'exp2': 'Train(INTER) → Test(INTRA₀)', 
        'exp3': 'Train(INTER) → Test(INTRA₁)',
        'exp4': 'Random 80/20 Split'
    }
    
    EXPERIMENT_COLORS = {
        'exp1': 'blue', 
        'exp2': 'red', 
        'exp3': 'green', 
        'exp4': 'purple'
    }
    
    DEFAULT_RF_PARAMS = {
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'class_weight': 'balanced',
    }
    
    # Default configurations
    DEFAULT_LORA_CONFIG = {
        'rank': 8,
        'alpha': 16,
        'dropout': 0.05,
        'target_modules': ['query', 'key', 'value']
    }
    
    DEFAULT_TRAINING_CONFIG = {
        'batch_size': 8,
        'epochs': 10,
        'learning_rate': 5e-5,
        'weight_decay': 0.01,
        'warmup_ratio': 0.1,
        'max_grad_norm': 1.0, 
        'early_stopping_patience': 3,
        'early_stopping_min_delta': 1e-4, 
    }
    
    DEFAULT_MLP_CONFIG = {
        'mlp_hidden_dim': 512, 
        'mlp_dropout': 0.1, 
        'max_length': 512, 
    }
    
    DEFAULT_DATA_CONFIG = {
        'max_length': 512,
        'min_bitscore': 50,
        'random_state': 42
    }
    
    DEFAULT_VIS_CONFIG = {
        'figsize': (10, 8),
        'dpi': 300,
        'alpha': 0.7
    }
    
    DEFAULT_BAYESIAN_CONFIG = {
        'n_calls': 10, 
        'random_state': 42, 
        'dimensions': {
            'lora_rank': {'type': 'Integer', 'low': 4, 'high': 32},
            'lora_alpha': {'type': 'Real', 'low': 8.0, 'high': 64.0},
            'lora_dropout': {'type': 'Real', 'low': 0.01, 'high': 0.2},
            'learning_rate': {'type': 'Real', 'low': 1e-5, 'high': 1e-4},
            'batch_size': {'type': 'Integer', 'low': 4, 'high': 16},
            'mlp_hidden_dim': {'type': 'Integer', 'low': 256, 'high': 1024},
            'mlp_dropout': {'type': 'Real', 'low': 0.05, 'high': 0.3}
        }
    }
    
    @classmethod
    def get_model_id(cls, model_name: str) -> str:
        """Get model ID from model name"""
        return cls.MODEL_MAPPING.get(model_name, "unknown_model")
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "lora": cls.DEFAULT_LORA_CONFIG,
            "training": cls.DEFAULT_TRAINING_CONFIG,
            "data": cls.DEFAULT_DATA_CONFIG,
            "visualization": cls.DEFAULT_VIS_CONFIG,
            "model": cls.DEFAULT_MLP_CONFIG,    
            "bayesian_opt": cls.DEFAULT_BAYESIAN_CONFIG
        }
    
    def create_output_directory(self, base_dir: str, model_size: str, method_type: str = "None") -> str:
        """Create timestamped output directory with method type suffix"""
        timestamp = datetime.now().strftime("%m%d_%H")
        output_dir = os.path.join(base_dir, model_size, f"experiment_{timestamp}_{method_type}")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    def save_config(self, args, output_dir: str):
        """Save experiment configuration"""
        config = {
            'args': vars(args),
            'defaults': self.get_default_config(),
            'model_id': self.get_model_id(args.model_size),
            'timestamp': datetime.now().isoformat()
        }
        
        config_path = os.path.join(output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4, cls=NumpyEncoder)
        
        print(f"Configuration saved to {config_path}")
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Config file not found: {config_path}")
            return self.get_default_config()
        except Exception as e:
            print(f"Failed to load config: {e}")
            return self.get_default_config()
