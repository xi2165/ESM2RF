import os
import json
import pickle
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime, timedelta
import torch
from config import NumpyEncoder

logger = logging.getLogger(__name__)

class CheckpointManager:
    """Simplified checkpoint management"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Checkpoint file paths
        self.state_file = self.checkpoint_dir / "experiment_state.json"
        self.embeddings_dir = self.checkpoint_dir / "embeddings"
        self.embeddings_dir.mkdir(exist_ok=True)
        
        logger.info(f"CheckpointManager initialized: {self.checkpoint_dir}")
    
    def get_data_hash(self, data) -> str:
        """Generate hash for data consistency check"""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()[:8]
    
    def save_experiment_progress(self, stage: str, args, data_hash: str, metadata: Dict = None):
        """Save experiment progress"""
        progress = {
            'stage': stage,
            'timestamp': datetime.now().isoformat(),
            'data_hash': data_hash,
            'args': vars(args),
            'metadata': metadata or {}
        }
        
        with open(self.state_file, 'w') as f:
            json.dump(progress, f, indent=2, cls=NumpyEncoder)
        
        logger.info(f"Progress saved: {stage}")
    
    def load_experiment_progress(self) -> Tuple[Optional[Dict], str]:
        """Load experiment progress"""
        if not self.state_file.exists():
            return None, 'data_loading'
        
        try:
            with open(self.state_file, 'r') as f:
                progress = json.load(f)
            return progress, progress.get('stage', 'data_loading')
        except Exception as e:
            logger.error(f"Failed to load progress: {e}")
            return None, 'data_loading'
    
    def save_model_results(self, results: Dict, model_type: str, model_size: str, use_bayesian_opt: bool = False):
        """Save model training results with consistent naming"""
        timestamp = datetime.now().strftime("%Y%m%d")
        opt_suffix = "_bayesian" if use_bayesian_opt else ""
        filename = f"results_{model_type}_{model_size}{opt_suffix}_{timestamp}.pkl"
        filepath = self.checkpoint_dir / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(results, f)
        
        # Create latest symlink
        latest_path = self.checkpoint_dir / f"results_{model_type}_{model_size}{opt_suffix}_latest.pkl"
        if latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(filename)
        
        logger.info(f"Model results saved: {filepath}")
    
    def load_model_results(self, model_type: str, model_size: str, use_bayesian_opt: bool = False) -> Optional[Dict]:
        """Load model training results with consistent naming"""
        opt_suffix = "_bayesian" if use_bayesian_opt else ""
        latest_path = self.checkpoint_dir / f"results_{model_type}_{model_size}{opt_suffix}_latest.pkl"
        
        if not latest_path.exists():
            return None
        
        try:
            with open(latest_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Failed to load model results: {e}")
            return None
    
    def save_embeddings(self, X, y, model_size: str, method_type: str, use_bayesian_opt: bool = False):
        """Save embeddings with consistent naming"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        filename = f"embeddings_{model_size}_{method_type}_{timestamp}.npz"
        filepath = self.embeddings_dir / filename
        
        np.savez_compressed(filepath, X=X, y=y)
        
        # Create latest symlink
        latest_path = self.embeddings_dir / f"embeddings_{model_size}_{method_type}_latest.npz"
        if latest_path.exists():
            latest_path.unlink()
        latest_path.symlink_to(filename)
        
        logger.info(f"Embeddings saved: {filepath}")
    
    def load_embeddings(self, model_size: str, method_type: str, use_bayesian_opt: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load embeddings from file"""
        latest_path = self.embeddings_dir / f"embeddings_{model_size}_{method_type}_latest.npz"
        
        if not latest_path.exists():
            return None, None
        
        try:
            data = np.load(latest_path)
            return data['X'], data['y']
        except Exception as e:
            logger.error(f"Failed to load embeddings: {e}")
            return None, None

    def cleanup_old_checkpoints(self, keep_days: int = 7):
        """Clean up old checkpoint files"""
        cutoff_date = datetime.now() - timedelta(days=keep_days)
        
        for file_path in self.checkpoint_dir.rglob("*"):
            if file_path.is_file() and not file_path.name.endswith('_latest.npz') and not file_path.name.endswith('_latest.pkl'):
                try:
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_date:
                        file_path.unlink()
                        logger.info(f"Cleaned up old file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up {file_path}: {e}")
    
    def save_lora_checkpoint(self, checkpoint: Dict, model_size: str, use_bayesian_opt: bool = False):
        """Save LoRA checkpoint with model size and optimization method"""
        opt_suffix = "_bayesian" if use_bayesian_opt else ""
        filename = f"lora_checkpoint_{model_size}{opt_suffix}_latest.pt"
        filepath = self.checkpoint_dir / filename
        
        checkpoint['metadata'] = {
            'model_size': model_size,
            'use_bayesian_opt': use_bayesian_opt,
            'exp_name': 'exp1' 
        }
        
        torch.save(checkpoint, filepath)
        logger.info(f"LoRA checkpoint saved: {filepath}")
        return str(filepath)
    
    def load_lora_checkpoint(self, model_size: str, use_bayesian_opt: bool = False) -> Optional[Dict]:
        """Load LoRA checkpoint with model size and optimization method"""
        opt_suffix = "_bayesian" if use_bayesian_opt else ""
        filename = f"lora_checkpoint_{model_size}{opt_suffix}_latest.pt"
        filepath = self.checkpoint_dir / filename
        
        if not filepath.exists():
            return None
        
        try:
            checkpoint = torch.load(filepath, map_location='cpu')
            logger.info(f"LoRA checkpoint loaded: {filepath}")
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load LoRA checkpoint: {e}")
            return None
    
    def get_latest_checkpoint_path(self, model_size: str, use_bayesian_opt: bool = False) -> Optional[str]:
        """Get path to latest checkpoint for model size and optimization method"""
        opt_suffix = "_bayesian" if use_bayesian_opt else ""
        pattern = f"lora_checkpoint_{model_size}{opt_suffix}_latest.pt"
        checkpoint_path = self.checkpoint_dir / pattern
        
        if checkpoint_path.exists():
            return str(checkpoint_path)
        return None
    
    def save_checkpoint(self, model_state_dict, epoch, val_loss, exp_name, training_params=None):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': model_state_dict,
            'epoch': epoch,
            'val_loss': val_loss,
            'training_params': training_params or {}
        }
        
        checkpoint_path = self.checkpoint_dir / f"lora_checkpoint_{exp_name}_latest.pt"
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def save_optimization_results(self, output_dir, best_params, best_loss):
        """Save Bayesian optimization results"""
        results_data = {
            'best_params': best_params,
            'best_loss': float(best_loss),
            'method': 'LoRA Bayesian Optimization'
        }
        
        results_path = os.path.join(output_dir, "bayesian_optimization_results.json")
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2, cls=NumpyEncoder)
        logger.info(f"Optimization results saved: {results_path}")
        return results_path
    
    def update_bayesian_checkpoint(self, results, final_params, best_params, best_loss):
        """Update checkpoint with Bayesian optimization results"""
        exp1_result = results.get('exp1', {})
        if 'checkpoint' not in exp1_result:
            logger.warning("No checkpoint found in exp1 results")
            return
        
        # Direct checkpoint modification
        checkpoint = exp1_result['checkpoint']
        checkpoint.update({
            'training_params': {
                'lora_rank': int(best_params['lora_rank']),
                'lora_alpha': float(best_params['lora_alpha']),
                'lora_dropout': float(best_params['lora_dropout']),
                'lr': float(best_params.get('learning_rate', final_params.get('learning_rate', 5e-5)))
            },
            'bayesian_optimized': True,
            'best_loss': float(best_loss)
        })
        
        # Save updated checkpoint
        filepath = self.save_lora_checkpoint(
            checkpoint, 
            model_size=final_params.get('model_size'), 
            use_bayesian_opt=True
        )
        
        logger.info(f"✅ Optimized checkpoint saved: rank={best_params['lora_rank']}, "
                   f"alpha={best_params['lora_alpha']:.4f}, dropout={best_params['lora_dropout']:.4f}")
        return filepath
