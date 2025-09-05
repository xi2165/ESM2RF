import os
import argparse
import numpy as np
import json
from datetime import datetime
import warnings
import torch
import gc
import traceback

from transformers import EsmModel, EsmTokenizer
from peft import PeftModel, get_peft_model, LoraConfig, TaskType

# Ignore scikit-learn deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*_validate_data.*")

from config import ProjectConfig, NumpyEncoder
from data import EmbeddingExtractor, load_data, visualize_embeddings_pca
from rf import rf_evaluation
from lora import train_lora_opt, get_lora_config, ESMFeatureExtractionModel
from checkpoint import CheckpointManager



class ModelManager:
    """Unified model management with proper resource cleanup"""
    
    def __init__(self, model_id):
        self.model_id = model_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._tokenizer = None
        self._base_model = None
        self._lora_model = None
        print(f"ModelManager initialized with device: {self.device}")
    
    def get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = EsmTokenizer.from_pretrained(self.model_id)
        return self._tokenizer
    
    def get_base_model(self):
        if self._base_model is None:
            self._base_model = EsmModel.from_pretrained(self.model_id).to(self.device)
        return self._base_model
    
    
    def get_lora_model(self, checkpoint_path=None):

        if self._lora_model is None and checkpoint_path:
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            print(f"Loading LoRA model from: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            base_model = self.get_base_model()
            
            lora_config_params = self._detect_lora_config_from_checkpoint(checkpoint)
            print(f"Detected LoRA config: {lora_config_params}")
            
            lora_config = get_lora_config(
                lora_rank=lora_config_params['lora_rank'],
                lora_alpha=lora_config_params['lora_alpha'],
                lora_dropout=lora_config_params['lora_dropout'],
                inference_mode=False  
            )
            
            peft_model = get_peft_model(base_model, lora_config)
            
            if not hasattr(peft_model, 'peft_config'):
                raise RuntimeError(f"PEFT model creation failed. Got {type(peft_model)}")
            
            print(f"PEFT model created successfully: {type(peft_model)}")
            
            model_state_dict = checkpoint['model_state_dict']
            
            cleaned_state_dict = {}
            for key, value in model_state_dict.items():
                if key.startswith('esm_model.'):
                    clean_key = key[len('esm_model.'):]
                    cleaned_state_dict[clean_key] = value
                elif not key.startswith('mlp.'):
                    cleaned_state_dict[key] = value
            
            missing_keys, unexpected_keys = peft_model.load_state_dict(cleaned_state_dict, strict=False)
            
            if missing_keys:
                print(f"Warning: Missing keys in state dict: {len(missing_keys)}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys in state dict: {len(unexpected_keys)}")
            
            peft_model.eval()
            if hasattr(peft_model, 'enable_adapter'):
                peft_model.enable_adapter()
                print("LoRA adapter enabled")
            
            self._lora_model = ESMFeatureExtractionModel(peft_model)
            
            mlp_state_dict = {k: v for k, v in model_state_dict.items() if k.startswith('mlp.')}
            if mlp_state_dict:
                self._lora_model.load_state_dict(mlp_state_dict, strict=False)
                print(f"Loaded MLP parameters: {len(mlp_state_dict)} keys")
            
            self._lora_model.to(self.device)
            
            self._verify_lora_activation()
            
            print(f"✅ LoRA model loaded successfully")
        
        return self._lora_model

    def _detect_lora_config_from_checkpoint(self, checkpoint):
        """Optimized LoRA config detection with fallback hierarchy"""
        
        # Parameter source priority mapping
        param_sources = [
            ('training_params', checkpoint.get('training_params')),
            ('lora_config', checkpoint.get('lora_config')),
            ('dynamic_detection', self._detect_from_weights(checkpoint))
        ]
        
        # Find first valid parameter source
        params, source = next(
            ((params, source) for source, params in param_sources if params),
            (None, None)
        )
        
        if not params:
            raise ValueError("No valid LoRA configuration found in checkpoint")
        
        # Standardize parameter extraction
        config_params = {
            'lora_rank': int(params['lora_rank']),
            'lora_alpha': float(params['lora_alpha']),
            'lora_dropout': float(params['lora_dropout'])
        }
        
        # Validate against actual weights if available
        if 'model_state_dict' in checkpoint:
            actual_rank = self._get_actual_rank(checkpoint['model_state_dict'])
            if actual_rank and actual_rank != config_params['lora_rank']:
                print(f"❌ Rank mismatch detected. Config: {config_params['lora_rank']}, Actual: {actual_rank}")
                if checkpoint.get('bayesian_optimized'):
                    print(f"✅ Using actual rank for Bayesian optimized checkpoint: {actual_rank}")
                    config_params['lora_rank'] = actual_rank
        
        # Log optimization status
        if checkpoint.get('bayesian_optimized'):
            print(f"✅ Bayesian optimized checkpoint (loss: {checkpoint.get('best_loss', 'N/A')})")
        
        print(f"📋 Config loaded from {source}: {config_params}")
        return config_params
    
    def _detect_from_weights(self, checkpoint):
        """Detect LoRA parameters from model weights"""
        model_state = checkpoint.get('model_state_dict', {})
        
        # Find LoRA rank from weights
        for key, tensor in model_state.items():
            if 'lora_A' in key and 'weight' in key:
                return {
                    'lora_rank': tensor.shape[0],
                    'lora_alpha': 16,  # Default
                    'lora_dropout': 0.05  # Default
                }
        return None
    
    def _get_actual_rank(self, model_state_dict):
        """Extract actual LoRA rank from model weights"""
        for key, tensor in model_state_dict.items():
            if 'lora_A' in key and 'weight' in key:
                return tensor.shape[0]
        return None

    def _verify_lora_activation(self):
        if self._lora_model is None:
            print("❌ No LoRA model loaded")
            return
            
        model = self._lora_model
        esm_model = model.esm_model
        
        print("\n=== LoRA Activation Verification ===")
        
        # 1. PEFT cofig
        if hasattr(esm_model, 'peft_config'):
            print(f"✅ PEFT config found: {esm_model.peft_config}")
            
            if hasattr(esm_model, 'active_adapters'):
                print(f"Active adapters: {esm_model.active_adapters}")
            
            if hasattr(esm_model, 'enable_adapter'):
                esm_model.enable_adapter()
                print("LoRA adapter explicitly enabled")
        else:
            print("❌ No PEFT config found - model may not be properly loaded")
            return
        
        # 2. LoRA config
        lora_params = []
        for name, param in esm_model.named_parameters():
            if 'lora' in name.lower():
                lora_params.append((name, param.norm().item()))
        
        if lora_params:
            print(f"✅ Found {len(lora_params)} LoRA parameters")
            for name, norm in lora_params[:3]:
                print(f"  {name}: norm={norm:.6f}")
        else:
            print("❌ No LoRA parameters found")
            return
        
        print("=== Verification Complete ===\n")
    
    def cleanup(self):
        """Clean up resources"""
        for attr in ['_base_model', '_lora_model', '_tokenizer']:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                delattr(self, attr)
        torch.cuda.empty_cache()
        gc.collect()
        print("ModelManager resources cleaned up")


class ExperimentRunner:
    """Simplified experiment runner with clear stage management"""
    
    def __init__(self, args):
        self.args = args
        self.config = ProjectConfig()
        self.checkpoint_manager = CheckpointManager("./checkpoints")
        self.model_manager = None
        self.embedding_extractor = None
        
        # Experiment data
        self.all_data = None
        self.X = None
        self.y = None
        self.lora_results = None
        
        if args.use_lora:
            if args.use_bayesian_opt:
                self.method_type = "opt_lora"
            else:
                self.method_type = "lora"
        else:
            self.method_type = "traditional"
        
        # Create output directory with method type suffix
        self.result_dir = self.config.create_output_directory(args.output_dir, args.model_size, self.method_type)
        print(f"Results will be saved to: {self.result_dir}")
    
    def setup_experiment(self):
        """Setup experiment environment"""
        # Save configuration
        self.config.save_config(self.args, self.result_dir)
        
        # Initialize model manager
        model_id = self.config.get_model_id(self.args.model_size)
        self.model_manager = ModelManager(model_id)
        
        # Initialize embedding extractor
        self.embedding_extractor = EmbeddingExtractor(self.model_manager.device)
        
        print(f"Using model: {model_id}")
    
    def load_data(self):
        """Load and prepare data"""
        if self.all_data is None:
            file_path = os.path.join("..", "database.txt")
            self.all_data = load_data(file_path)
            print(f"Data loaded: {len(self.all_data)} samples")
    
    def run_lora_training(self):
        """Run LoRA training if enabled"""
        if not self.args.use_lora:
            return
            
        print("\n===== LoRA Training Stage =====")
        
        # Consolidated cache key generation
        cache_key = f"{self.args.model_size}{'_bayesian' if self.args.use_bayesian_opt else ''}"
        
        # Single checkpoint load attempt
        self.lora_results = self.checkpoint_manager.load_model_results(
            'lora', self.args.model_size, self.args.use_bayesian_opt
        )
        
        if self.lora_results is None:
            # Consolidated parameter dictionary
            training_params = {
                param: getattr(self.args, param) 
                for param in ['lora_rank', 'lora_alpha', 'lora_dropout', 'batch_size', 
                             'epochs', 'learning_rate', 'min_bitscore', 'model_size']
            }
            
            # Single training call
            self.lora_results, _ = train_lora_opt(
                self.all_data, self.model_manager.model_id, self.result_dir,
                checkpoint_manager=self.checkpoint_manager,
                use_bayesian_opt=self.args.use_bayesian_opt,
                **training_params
            )
            
            # Conditional save
            if self.lora_results:
                self.checkpoint_manager.save_model_results(
                    self.lora_results, 'lora', self.args.model_size, self.args.use_bayesian_opt
                )
        else:
            print("LoRA results loaded from checkpoint")
    
    def run_embedding_generation(self):
        """Generate embeddings using the appropriate model"""
        print("\n===== Embedding Generation =====")
        
        # load from checkpoint
        self.X, self.y = self.checkpoint_manager.load_embeddings(self.args.model_size, self.method_type, self.args.use_bayesian_opt)
        
        if self.X is None or self.y is None:
            if self.args.use_lora:
                checkpoint_path = self.checkpoint_manager.get_latest_checkpoint_path(self.args.model_size, self.args.use_bayesian_opt)
                if checkpoint_path is None:
                    raise FileNotFoundError(f"No LoRA checkpoint found")
                model = self.model_manager.get_lora_model(checkpoint_path)
                print("Using LoRA-tuned model for embedding extraction")
            else:
                model = self.model_manager.get_base_model()
                print("Using base ESM model for embedding extraction")
            
            tokenizer = self.model_manager.get_tokenizer()
            
            self.X, self.y = self.embedding_extractor.generate_embeddings(self.all_data, model, tokenizer, batch_size=32)
            
            if self.X is not None and self.y is not None:
                self.checkpoint_manager.save_embeddings(self.X, self.y, self.args.model_size, self.method_type, self.args.use_bayesian_opt)
                print(f"Generated and saved embeddings: {self.X.shape}")
        else:
            print(f"Embeddings loaded from cache: {self.X.shape}")
    
    def run_evaluation(self):
        """Run evaluation stage"""
        if self.X is None or self.y is None:
            print("Warning: No embeddings available for evaluation")
            return
            
        print("\n===== Evaluation Stage =====")
        
        from split import load_partition_results, generate_fixed_partitions
        
        # check if global partition exists
        partition, protein_index, all_proteins, partition_exists = load_partition_results("./output")
        if partition_exists:
            # calculate global partition statistics
            num_proteins = len(partition)
            num_partitions = len(set(partition.values()))
            print(f"Using existing global partition: {num_proteins} proteins, {num_partitions} partitions")
        else:
            print("Global partition not found, generating...")
            generate_fixed_partitions("database.txt", "./output", self.args.min_bitscore)
        
        # Convert to numpy if needed
        if isinstance(self.X, torch.Tensor):
            self.X = self.X.cpu().numpy()
        if isinstance(self.y, torch.Tensor):
            self.y = self.y.cpu().numpy()
        
        self.y = self.y.astype(int)
        
        print(f"Features shape: {self.X.shape}, Labels distribution: {np.bincount(self.y)}")
        
        # Visualize embeddings
        pca_path = os.path.join(self.result_dir, "embedding_pca.png")
        visualize_embeddings_pca(self.X, self.y, pca_path)
        

        print("Running integrated RF evaluation...")
        partition_results = rf_evaluation(
            X=self.X, 
            y=self.y, 
            data=self.all_data,          
            output_dir=self.result_dir,
            min_bitscore=self.args.min_bitscore,
            random_state=self.args.random_state,
            use_grid_search=self.args.use_grid_search,
            target_experiment=None       
        )
        
        # Save results
        self.save_results(partition_results)
    
    
    def save_results(self, partition_results):
        """Save evaluation results"""
        try:
            # Save partition results
            serializable_results = {}
            for exp_name, exp_data in partition_results['results'].items():
                serializable_results[exp_name] = {'metrics': exp_data['metrics']}
            
            partition_path = os.path.join(self.result_dir, "partition_evaluation.json")
            with open(partition_path, 'w') as f:
                json.dump(serializable_results, f, indent=4, cls=NumpyEncoder)
            
            print("Evaluation results saved successfully")
            
        except Exception as e:
            print(f"Failed to save results: {e}")
    
    def run(self):
        """Run complete experiment pipeline"""
        try:
            print("Starting experiment pipeline...")
            
            # Setup
            self.setup_experiment()
            
            # Load data
            self.load_data()
            
            # LoRA training (if enabled)
            self.run_lora_training()
            
            # Generate embeddings
            self.run_embedding_generation()
            
            # Evaluation
            self.run_evaluation()
            
            # Optional analysis
            
            print("\n===== Experiment Completed Successfully =====")
            
        except KeyboardInterrupt:
            print("\nExperiment interrupted by user")
        except Exception as e:
            print(f"Experiment failed: {str(e)}")
            traceback.print_exc()
        finally:
            # Cleanup
            if self.model_manager:
                self.model_manager.cleanup()
            if self.embedding_extractor:
                self.embedding_extractor.cleanup()
            self.checkpoint_manager.cleanup_old_checkpoints()

def parse_arguments():
    """Parse command line arguments"""

    defaults = ProjectConfig.get_default_config()
    
    parser = argparse.ArgumentParser(description='PPI Prediction with ESM2')

    # Basic parameters
    parser.add_argument('--model_size', type=str, default='6', choices=['6', '12', '30', '33'], help='ESM2 model size')
    parser.add_argument('--output_dir', type=str, default='output', help='Output directory')
    parser.add_argument('--random_state', type=int, default=defaults['data']['random_state'], help='Random seed')
    parser.add_argument('--min_bitscore', type=float, default=defaults['data']['min_bitscore'], help='Minimum bitscore threshold')
    
    # LoRA parameters
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA fine-tuning')
    parser.add_argument('--use_bayesian_opt', action='store_true', help='Use Bayesian optimization')
    parser.add_argument('--lora_rank', type=int, default=defaults['lora']['rank'], help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=defaults['lora']['alpha'], help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float, default=defaults['lora']['dropout'], help='LoRA dropout')
    parser.add_argument('--batch_size', type=int, default=defaults['training']['batch_size'], help='Batch size')
    parser.add_argument('--epochs', type=int, default=defaults['training']['epochs'], help='Training epochs')
    parser.add_argument('--learning_rate', type=float, default=defaults['training']['learning_rate'], help='Learning rate')
    
    # Analysis parameters
    parser.add_argument('--use_grid_search', action='store_true', help='Use GridSearchCV')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    runner = ExperimentRunner(args)
    runner.run()

if __name__ == "__main__":
    main()
