import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import EsmModel, EsmTokenizer, get_linear_schedule_with_warmup #ESM-2
from peft import get_peft_model, LoraConfig, TaskType #LoRA
from tqdm import tqdm
from skopt import gp_minimize #Bayesian optimization
from skopt.space import Real, Integer
from skopt.utils import use_named_args

from config import ProjectConfig
from split import kahip_split

def get_lora_config(lora_rank=None, lora_alpha=None, lora_dropout=None, inference_mode=None):
    """Get LoRA configuration with defaults from config"""
    
    defaults = ProjectConfig.get_default_config()
    
    lora_rank = lora_rank if lora_rank is not None else defaults['lora']['rank']
    lora_alpha = lora_alpha if lora_alpha is not None else defaults['lora']['alpha']
    lora_dropout = lora_dropout if lora_dropout is not None else defaults['lora']['dropout']
    
    if inference_mode is None:
        inference_mode = False
    
    return LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=inference_mode,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=defaults['lora']['target_modules'],
        bias="none"
    )

# ============================================================================
# 1. data processing
# ============================================================================

class PPIFeatureDataset(torch.utils.data.Dataset):
    """Dataset for protein-protein interaction with feature extraction"""
    
    def __init__(self, data, tokenizer, max_length=None):
        self.data = data
        self.tokenizer = tokenizer
        
        if max_length is None:
            defaults = ProjectConfig.get_default_config()
            max_length = defaults['model']['max_length']
        
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize both proteins
        protein1_tokens = self.tokenizer(
            item['seq1'],
            truncation=True, 
            padding='max_length', 
            max_length=self.max_length, 
            return_tensors='pt'
        )
        
        protein2_tokens = self.tokenizer(
            item['seq2'],
            truncation=True, 
            padding='max_length', 
            max_length=self.max_length, 
            return_tensors='pt'
        )
        
        return {
            'input_ids_1': protein1_tokens['input_ids'].squeeze(),
            'attention_mask_1': protein1_tokens['attention_mask'].squeeze(),
            'input_ids_2': protein2_tokens['input_ids'].squeeze(),
            'attention_mask_2': protein2_tokens['attention_mask'].squeeze(),
            'labels': torch.tensor(item['interaction'], dtype=torch.float)
        }

def collate_fn_feature(batch):
    """Batch processing function for feature extraction"""
    input_ids_1 = torch.stack([item['input_ids_1'] for item in batch])
    attention_mask_1 = torch.stack([item['attention_mask_1'] for item in batch])
    input_ids_2 = torch.stack([item['input_ids_2'] for item in batch])
    attention_mask_2 = torch.stack([item['attention_mask_2'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'input_ids_1': input_ids_1,
        'attention_mask_1': attention_mask_1,
        'input_ids_2': input_ids_2,
        'attention_mask_2': attention_mask_2,
        'labels': labels
    }

def create_data_loaders(train_data, test_data, tokenizer, batch_size=None, max_length=None):
    """Create data loaders for training and testing"""
    defaults = ProjectConfig.get_default_config()
    
    batch_size = batch_size if batch_size is not None else defaults['training']['batch_size']
    max_length = max_length if max_length is not None else defaults['model']['max_length']
    
    batch_size = int(batch_size)
    
    train_dataset = PPIFeatureDataset(train_data, tokenizer, max_length=max_length)
    test_dataset = PPIFeatureDataset(test_data, tokenizer, max_length=max_length)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, test_loader

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# 2. Feature Extraction Model with MLP Classifier
# ============================================================================

class ESMFeatureExtractionModel(nn.Module):
    """ESM model for feature extraction with MLP classifier
        Input:
            input token ids1 & 2: Tensor of shape (batch_size, seq_len)
            attention masks 1 & 2: Tensor of shape (batch_size, seq_len)
        Output:
            logits: Tensor of shape (batch_size)
    """
    
    def __init__(self, esm_model, hidden_dim=None, mlp_hidden_dim=None, mlp_dropout=None):
        super().__init__()
        
        defaults = ProjectConfig.get_default_config()
        
        self.esm_model = esm_model
        self.hidden_dim = hidden_dim or esm_model.config.hidden_size
        
        mlp_hidden_dim = mlp_hidden_dim or defaults['model']['mlp_hidden_dim']
        mlp_dropout = mlp_dropout or defaults['model']['mlp_dropout']
        
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_dim, 1)
        )
    
    def forward(self, input_ids_1, attention_mask_1, input_ids_2, attention_mask_2):
        # Extract features from both proteins
        outputs1 = self.esm_model(input_ids=input_ids_1, attention_mask=attention_mask_1)
        outputs2 = self.esm_model(input_ids=input_ids_2, attention_mask=attention_mask_2)
        
        # Use mean pooling
        emb1 = outputs1.last_hidden_state.mean(dim=1)
        emb2 = outputs2.last_hidden_state.mean(dim=1)
        
        # Concatenate embeddings [batch_size, hidden_dim]->[batch_size, hidden_dim*2]
        combined_emb = torch.cat([emb1, emb2], dim=1)
        
        # Pass through MLP classifier
        logits = self.mlp(combined_emb)
        
       
        return logits.squeeze(-1)


# ============================================================================
# 3. model configuration
# ============================================================================

def setup_lora_model(model_id, lora_rank=None, lora_alpha=None, lora_dropout=None):
    """Setup LoRA model with configuration
    
        Output:
            model: ESMFeatureExtractionModel - Model with LoRA applied  
    """
    defaults = ProjectConfig.get_default_config()
    
    lora_rank = lora_rank if lora_rank is not None else defaults['lora']['rank']
    lora_alpha = lora_alpha if lora_alpha is not None else defaults['lora']['alpha']
    lora_dropout = lora_dropout if lora_dropout is not None else defaults['lora']['dropout']
    
    base_model = EsmModel.from_pretrained(model_id)
    
    lora_config = get_lora_config(
        lora_rank=lora_rank,
        lora_alpha=lora_alpha, 
        lora_dropout=lora_dropout
    )
    
    # Apply LoRA
    lora_model = get_peft_model(base_model, lora_config)
    
    # Wrap with feature extraction model
    model = ESMFeatureExtractionModel(lora_model)
    
    device = get_device()
    model = model.to(device)
    
    return model

# ============================================================================
# 5. Training
# ============================================================================

def train_lora(model, train_loader, val_loader, device, epochs=None, lr=None, 
               output_dir="./output", checkpoint_manager=None, exp_name="exp1",
               model_size=None, use_bayesian_opt=False, **kwargs):
    """Train LoRA model with feature extraction and MLP"""
    
    defaults = ProjectConfig.get_default_config()
    
    epochs = epochs if epochs is not None else defaults['training']['epochs']
    lr = lr if lr is not None else defaults['training']['learning_rate']
    
    training_config = defaults['training']
    
    max_grad_norm = training_config.get('max_grad_norm', 1.0)
    early_stopping_patience = training_config.get('early_stopping_patience', 3)
    early_stopping_min_delta = training_config.get('early_stopping_min_delta', 1e-4)
    
    lora_rank = kwargs.get('lora_rank', defaults['lora']['rank'])
    lora_alpha = kwargs.get('lora_alpha', defaults['lora']['alpha'])
    lora_dropout = kwargs.get('lora_dropout', defaults['lora']['dropout'])

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, 
                                 weight_decay=defaults['training']['weight_decay'])
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(defaults['training']['warmup_ratio'] * total_steps), 
        num_training_steps=total_steps
    )
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Training history
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    early_stopped = False
    
    print(f"Training {exp_name} with LoRA (rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout})")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0.0
        epoch_train_correct = 0
        epoch_train_total = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            optimizer.zero_grad()
            
            input_ids_1 = batch['input_ids_1'].to(device)
            attention_mask_1 = batch['attention_mask_1'].to(device)
            input_ids_2 = batch['input_ids_2'].to(device)
            attention_mask_2 = batch['attention_mask_2'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            logits = model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm) # gradiant clip
            optimizer.step()
            scheduler.step()
            
            # Statistics
            epoch_train_loss += loss.item()
            predictions = torch.sigmoid(logits)
            predicted_labels = (predictions > 0.5).float()
            epoch_train_correct += (predicted_labels == labels).sum().item()
            epoch_train_total += labels.size(0)
        
        # Calculate training metrics
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_accuracy = epoch_train_correct / epoch_train_total
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        epoch_val_correct = 0
        epoch_val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                input_ids_1 = batch['input_ids_1'].to(device)
                attention_mask_1 = batch['attention_mask_1'].to(device)
                input_ids_2 = batch['input_ids_2'].to(device)
                attention_mask_2 = batch['attention_mask_2'].to(device)
                labels = batch['labels'].to(device)
                
                # Forward pass
                logits = model(input_ids_1, attention_mask_1, input_ids_2, attention_mask_2)
                loss = criterion(logits, labels)
                
                # Statistics
                epoch_val_loss += loss.item()
                predictions = torch.sigmoid(logits)
                predicted_labels = (predictions > 0.5).float()
                epoch_val_correct += (predicted_labels == labels).sum().item()
                epoch_val_total += labels.size(0)
        
        # Calculate validation metrics
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_accuracy = epoch_val_correct / epoch_val_total
        
        # Record history
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
        
        # Early stopping logic
        if avg_val_loss < best_val_loss - early_stopping_min_delta:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            if checkpoint_manager:
                checkpoint_data = {
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch,
                    'val_loss': avg_val_loss,
                    'training_params': {
                        'lora_rank': lora_rank,
                        'lora_alpha': lora_alpha,
                        'lora_dropout': lora_dropout,
                        'lr': lr
                    },
                    'bayesian_optimized': use_bayesian_opt
                }
                checkpoint_manager.save_lora_checkpoint(checkpoint_data, model_size=model_size, use_bayesian_opt=use_bayesian_opt)
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            print(f'Early stopping at epoch {epoch+1}')
            early_stopped = True
            break
    
    # Return results
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'early_stopped': early_stopped,
        'total_epochs': epoch + 1,
        'lora_rank': lora_rank,
        'lora_alpha': lora_alpha,
        'lora_dropout': lora_dropout,
        'training_params': {
            'lora_rank': lora_rank,
            'lora_alpha': lora_alpha,
            'lora_dropout': lora_dropout,
            'lr': lr
        }
    }
    
    return {
        'checkpoint': final_checkpoint,
        'training_history': {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies
        },
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'early_stopped': early_stopped
    }

def train_lora_kahip(all_data, model_id, output_dir, checkpoint_manager=None, device=None, **params):
    """Train LoRA using KaHIP data splits - only exp1"""
    device = get_device() if device is None else device
    
    print("=== LoRA Training with Feature Extraction (exp1 only) ===")
    
    defaults = ProjectConfig.get_default_config()
    
    # Prepare data for kahip_split
    protein_pairs = all_data
    y = np.array([item['interaction'] for item in all_data])
    X = None
    
    # Get data splits
    data_splits = kahip_split(X, y, protein_pairs, output_dir=output_dir, 
                             min_bitscore=params.get('min_bitscore', defaults['data']['min_bitscore']))
    
    # Consolidated LoRA parameter extraction
    lora_params = {
        'lora_rank': params.get('lora_rank', defaults['lora']['rank']),
        'lora_alpha': params.get('lora_alpha', defaults['lora']['alpha']),
        'lora_dropout': params.get('lora_dropout', defaults['lora']['dropout'])
    }
    
    print(f"🔧 LoRA config: {lora_params}")
    
    # Only train exp1
    results = {}
    tokenizer = EsmTokenizer.from_pretrained(model_id)
    
    exp_key = 'exp1'
    train_idx, test_idx = data_splits[exp_key]
    exp_name = "exp1"
    
    train_data = [protein_pairs[idx] for idx in train_idx]
    val_data = [protein_pairs[idx] for idx in test_idx]
    
    # Setup model with extracted parameters
    model = setup_lora_model(model_id, **lora_params)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_data, val_data, tokenizer, 
        batch_size=params.get('batch_size')
    )
    
    # Consolidated training parameters
    train_params = {
        **params,  # Include all original parameters
        **lora_params,  # Override with LoRA-specific parameters
        'output_dir': output_dir,
        'checkpoint_manager': checkpoint_manager,
        'exp_name': exp_name
    }
    
    training_results = train_lora(model, train_loader, val_loader, device, **train_params)
    results[exp_name] = training_results
    
    return results

def opt_lora(all_data, model_id, output_dir, n_calls=None, random_state=None, device=None, **base_params):
    """Bayesian optimization for LoRA hyperparameters - only exp1"""
    if device is None:
        device = get_device()

    config = ProjectConfig.get_default_config()
    bayesian_config = config['bayesian_opt']
    dimensions_config = bayesian_config['dimensions']
    
    if n_calls is None:
        n_calls = bayesian_config.get('n_calls', 10)
    
    if random_state is None:
        random_state = bayesian_config.get('random_state', 42)

    dimensions = []
    for param_name, param_config in dimensions_config.items():
        if param_config['type'] == 'Integer':
            dimensions.append(Integer(param_config['low'], param_config['high'], name=param_name))
        elif param_config['type'] == 'Real':
            dimensions.append(Real(param_config['low'], param_config['high'], name=param_name))
    
    @use_named_args(dimensions)
    def objective(**opt_params):
        """Objective function for optimization"""
        print(f"\nTesting parameters: {opt_params}")
        
        defaults = ProjectConfig.get_default_config()
        
        current_params = base_params.copy()
        current_params.update(opt_params)
        
        if 'model_size' not in current_params:
            current_params['model_size'] = base_params.get('model_size')
        current_params['use_bayesian_opt'] = True
        
        try:
            # Get data splits
            protein_pairs = all_data
            y = np.array([item['interaction'] for item in all_data])
            X = None
            
            data_splits = kahip_split(X, y, protein_pairs, output_dir=output_dir, 
                                     min_bitscore=current_params.get('min_bitscore', 50))
            
            # Only train exp1
            tokenizer = EsmTokenizer.from_pretrained(model_id)
            
            exp_key = 'exp1'
            train_idx, test_idx = data_splits[exp_key]
            
            train_data = [protein_pairs[idx] for idx in train_idx]
            val_data = [protein_pairs[idx] for idx in test_idx]
            
            # Setup model
            model = setup_lora_model(
                model_id, 
                lora_rank=current_params.get('lora_rank', defaults['lora']['rank']),
                lora_alpha=current_params.get('lora_alpha', defaults['lora']['alpha']),
                lora_dropout=current_params.get('lora_dropout', defaults['lora']['dropout'])
            )
            
            # Create data loaders
            train_loader, val_loader = create_data_loaders(
                train_data, val_data, tokenizer, 
                batch_size=int(current_params.get('batch_size', 8))
            )
            
            # Train model
            result = train_lora(
                model, train_loader, val_loader, device,
                epochs=current_params.get('epochs', 3),
                lr=current_params.get('learning_rate', 5e-5),
                output_dir=output_dir,
                checkpoint_manager=None,
                exp_name="exp1",
                model_size=base_params.get('model_size'),  
                use_bayesian_opt=base_params.get('use_bayesian_opt', False) 
            )
            
            # Get final validation loss
            final_val_loss = result['training_history']['val_losses'][-1]
            print(f"Validation loss: {final_val_loss:.4f}")
            
            return final_val_loss if not np.isinf(final_val_loss) else 5.0
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            return 5.0  # Return penalty for failed runs
    
    # Run optimization
    print("Starting Bayesian optimization...")
    result = gp_minimize(objective, dimensions, n_calls=n_calls, random_state=random_state)
    
    # Extract best parameters
    best_params = {}
    for i, dim in enumerate(dimensions):
        best_params[dim.name] = result.x[i]
    
    print(f"\nBest parameters found: {best_params}")
    print(f"Best validation loss: {result.fun:.4f}")
    
    return best_params, result.fun

def train_lora_opt(all_data, model_id, output_dir, checkpoint_manager=None, 
                     use_bayesian_opt=False, device=None, **params):
    """Unified interface for LoRA training with optional Bayesian optimization - only exp1"""
    device = get_device() if device is None else device
    
    print("=== LoRA Training with Feature Extraction + MLP (exp1 only) ===")
    
    # Consolidate parameter handling
    final_params = params.copy()
    final_params['use_bayesian_opt'] = use_bayesian_opt
    
    best_params = None
    if use_bayesian_opt:
        print("Using Bayesian optimization for hyperparameter tuning...")
        best_params, best_loss = opt_lora(all_data, model_id, output_dir, device=device, **params)
        
        # Direct parameter update - no intermediate variables
        final_params.update(best_params)
        print(f"Training final model with optimized parameters: {best_params}")
        
        # Save optimization results through checkpoint manager
        if checkpoint_manager:
            checkpoint_manager.save_optimization_results(output_dir, best_params, best_loss)
    else:
        print("Using default parameters for training...")
    
    # Single training call with optimized parameters
    results = train_lora_kahip(all_data, model_id, output_dir, checkpoint_manager, device, **final_params)
    
    # Update checkpoint through checkpoint manager
    if use_bayesian_opt and results and checkpoint_manager and best_params:
        checkpoint_manager.update_bayesian_checkpoint(results, final_params, best_params, best_loss)
    
    if results:
        lora_training_curves(results, output_dir)
    
    return results, best_params


def lora_training_curves(results, output_dir):
    """Plot training curves for exp1 with train and validation combined in single plots"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    exp_name = 'exp1'
    if exp_name in results:
        result = results[exp_name]
        history = result['training_history']
        
        epochs = range(1, len(history['train_losses']) + 1)
        
        # Combined Loss plot
        axes[0].plot(epochs, history['train_losses'], label='Training Loss', marker='o', color='blue', linewidth=2)
        axes[0].plot(epochs, history['val_losses'], label='Validation Loss', marker='s', color='red', linewidth=2)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        axes[0].set_xticks(list(epochs))
        
        all_losses = history['train_losses'] + history['val_losses']
        min_loss = min(all_losses)
        max_loss = max(all_losses)
        
        loss_range = max_loss - min_loss
        if loss_range <= 0.1:
            tick_interval = 0.02
        elif loss_range <= 0.5:
            tick_interval = 0.05
        else:
            tick_interval = 0.10
        
        y_min = (int(min_loss / tick_interval) - 1) * tick_interval
        y_max = (int(max_loss / tick_interval) + 2) * tick_interval
        axes[0].set_ylim(max(0, y_min), y_max)
        
        import numpy as np
        y_ticks = np.arange(max(0, y_min), y_max + tick_interval/2, tick_interval)
        axes[0].set_yticks(y_ticks)
        axes[0].set_yticklabels([f'{tick:.2f}' for tick in y_ticks])
        
        # Combined Accuracy plot
        axes[1].plot(epochs, history['train_accuracies'], label='Training Accuracy', marker='o', color='blue', linewidth=2)
        axes[1].plot(epochs, history['val_accuracies'], label='Validation Accuracy', marker='s', color='red', linewidth=2)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        axes[1].set_xticks(list(epochs))
        
        all_accuracies = history['train_accuracies'] + history['val_accuracies']
        min_acc = min(all_accuracies)
        max_acc = max(all_accuracies)
        
        acc_range = max_acc - min_acc
        if acc_range <= 0.2:
            acc_tick_interval = 0.02
        else:
            acc_tick_interval = 0.05
        
        acc_y_min = max(0, (int(min_acc / acc_tick_interval) - 1) * acc_tick_interval)
        acc_y_max = min(1, (int(max_acc / acc_tick_interval) + 2) * acc_tick_interval)
        axes[1].set_ylim(acc_y_min, acc_y_max)
        
        acc_y_ticks = np.arange(acc_y_min, acc_y_max + acc_tick_interval/2, acc_tick_interval)
        axes[1].set_yticks(acc_y_ticks)
        axes[1].set_yticklabels([f'{tick:.2f}' for tick in acc_y_ticks])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'LoRA_training.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Combined training curves saved to {output_dir}/LoRA_training.png")
