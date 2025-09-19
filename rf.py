import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    balanced_accuracy_score, confusion_matrix, roc_auc_score,
    average_precision_score, roc_curve, auc, matthews_corrcoef, log_loss
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import glob
from datetime import datetime

from split import kahip_split
from calibration import CalibrationEvaluator, calculate_uncertainty
from config import ProjectConfig, NumpyEncoder

# ============================================================================
# Evaluation and Plot
# ============================================================================

def analyze_uncertainty_cutoffs(y_true, y_proba, cutoffs=[0.2, 0.4, 0.6, 0.8]):
    """
    Analyze precision and true positives under different uncertainty cutoffs
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        cutoffs: List of uncertainty cutoff thresholds
    
    Returns:
        dict: Contains precision and true positives for each cutoff
    """
    uncertainty = calculate_uncertainty(y_proba)
    results = {}
    
    for cutoff in cutoffs:
        # Select samples with uncertainty below cutoff (high confidence samples)
        high_confidence_mask = uncertainty <= cutoff
        
        if high_confidence_mask.sum() == 0:
            results[cutoff] = {'precision': 0.0, 'true_positives': 0}
            continue
            
        # Get predictions and true labels for high confidence samples
        y_true_hc = y_true[high_confidence_mask]
        y_proba_hc = y_proba[high_confidence_mask]
        
        # Use 0.5 as classification threshold
        y_pred_hc = (y_proba_hc > 0.5).astype(int)
        
        # Calculate precision and true positives
        true_positives = ((y_pred_hc == 1) & (y_true_hc == 1)).sum()
        predicted_positives = (y_pred_hc == 1).sum()
        
        precision = true_positives / predicted_positives if predicted_positives > 0 else 0.0
        
        results[cutoff] = {
            'precision': precision,
            'true_positives': int(true_positives),
            'predicted_positives': int(predicted_positives),
            'total_samples': int(high_confidence_mask.sum())
        }
    
    return results

def calculate_metrics(y_true, y_pred, y_proba=None):
    """
    Calculate all evaluation metrics in one place
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }
    
    # Calculate specificity
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    if y_proba is not None:
        metrics['auc'] = roc_auc_score(y_true, y_proba)
        metrics['prc'] = average_precision_score(y_true, y_proba)
        
        # Calculate ECE
        cal = CalibrationEvaluator(n_bins=10)
        
        calibration_metrics = cal.calculate_detailed_metrics(y_true, y_proba)
        metrics.update(calibration_metrics)
        
        # Calculate test loss
        metrics['test_loss'] = log_loss(y_true, y_proba)
    
    return metrics

def print_evaluation_results(y_true, y_pred, y_proba, test_loss=None):
    """
    Print detailed evaluation results including uncertainty cutoff analysis
    """
    metrics = calculate_metrics(y_true, y_pred, y_proba)
    
    # Print results in consistent format
    print(f"Test Loss           AUC                 PRC                 Accuracy            Sensitivity         Specificity         Precision           F1                  MCC")
    print(f"{metrics.get('test_loss', 0.0):<18} {metrics.get('auc', 0.0):<18} {metrics.get('prc', 0.0):<18} {metrics['accuracy']:<18} {metrics['recall']:<18} {metrics['specificity']:<18} {metrics['precision']:<18} {metrics['f1']:<18} {metrics['mcc']}")
    
    if y_proba is not None:
        print(f"Expected Calibration Error (ECE): {metrics['ece']}")
        
        # Uncertainty cutoff analysis
        cutoff_results = analyze_uncertainty_cutoffs(y_true, y_proba)
        for cutoff, result in cutoff_results.items():
            print(f"Uncertainty Cutoff {cutoff}: Precision - {result['precision']}, True Positives - {result['true_positives']}")

def plot_confusion_matrix(y_true, y_pred, output_path=None):
    """
    Plot confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                annot_kws={'size': 16})  
    plt.xlabel('Predicted', fontsize=14)  
    plt.ylabel('True', fontsize=14)      
    plt.title('Confusion Matrix', fontsize=16)  
    
    # 设置坐标轴刻度标签的字体大小
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {output_path}")
    
    plt.close()

def plot_roc_curve(y_true, y_proba, output_path=None):
    """
    Plot ROC curve
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {output_path}")
    
    plt.close()

def plot_combined_roc_curves(partition_results, output_path=None):
    """
    Plot ROC curves for all experiments on the same figure
    """
    plt.figure(figsize=(10, 8))
    
    colors = ProjectConfig.EXPERIMENT_COLORS
    exp_names = ProjectConfig.EXPERIMENT_NAMES
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', alpha=0.8, label='Random Classifier')
    
    # Plot ROC curve for each experiment
    for exp_name in ['exp1', 'exp2', 'exp3', 'exp4']:
        if exp_name in partition_results['results']:
            exp_data = partition_results['results'][exp_name]
            predictions = exp_data['predictions']
            
            if predictions['y_proba'] is not None:
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(predictions['y_true'], predictions['y_proba'])
                roc_auc = auc(fpr, tpr)
                
                # Plot ROC curve
                plt.plot(fpr, tpr, color=colors[exp_name], lw=2.5, alpha=0.8,
                        label=f'{exp_name}: {exp_names[exp_name]} (AUC = {roc_auc:.3f})')
            else:
                print(f"Warning: {exp_name} has no probability predictions, skipping ROC curve")
    
    # Set chart properties
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison - Graph Partition Based Evaluation', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Combined ROC curves saved to {output_path}")
    
    plt.close()

def plot_combined_metrics_comparison(partition_results, output_path=None):
    """
    Plot comparison of all metrics across experiments
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Extract data
    experiments = ['exp1', 'exp2', 'exp3', 'exp4']
    exp_labels = ['Exp1\n(INTRA₀→INTRA₁)', 'Exp2\n(INTER→INTRA₀)', 
                'Exp3\n(INTER→INTRA₁)', 'Exp4\n(Random 80/20)']
    metrics = ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'auc']
    metric_labels = ['Accuracy', 'Balanced Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
    colors = list(ProjectConfig.EXPERIMENT_COLORS.values())
    
    for i, (metric, metric_label) in enumerate(zip(metrics, metric_labels)):
        values = []
        for exp in experiments:
            if exp in partition_results['results'] and metric in partition_results['results'][exp]['metrics']:
                values.append(partition_results['results'][exp]['metrics'][metric])
            else:
                values.append(0)  # If metric doesn't exist, set to 0
        
        # Plot bar chart
        bars = axes[i].bar(exp_labels, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
        
        # Add value labels
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        axes[i].set_title(f'{metric_label}', fontsize=12, fontweight='bold')
        axes[i].set_ylim(0, 1.1)
        axes[i].grid(True, alpha=0.3, axis='y')
        
        # Set y-axis ticks
        axes[i].set_yticks(np.arange(0, 1.1, 0.2))
    
    plt.suptitle('Performance Metrics Comparison Across Graph Partition Experiments', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison chart saved to {output_path}")
    
    plt.close()

def _generate_plots(exp_name, y_true, y_pred, y_proba, output_dir):
    """
    Generate visualization plots for experiment
    """
    # Confusion matrix
    cm_path = os.path.join(output_dir, f"{exp_name}_cm.png")
    plot_confusion_matrix(y_true, y_pred, output_path=cm_path)
    
    # ROC curve
    if y_proba is not None:
        roc_path = os.path.join(output_dir, f"{exp_name}_roc.png")
        plot_roc_curve(y_true, y_proba, output_path=roc_path)
        
        # Calibration diagram
        cal = CalibrationEvaluator(n_bins=10)
        calibration_path = os.path.join(output_dir, f"{exp_name}_calibration.png")
        cal.plot_calibration_curve(y_proba, y_true, 
                                 save_path=os.path.join(output_dir, f"{exp_name}_calibration.png"))

# ============================================================================
# RF
# ============================================================================

def rf_evaluation(X, y, data, output_dir="./output", min_bitscore=50, 
                           use_grid_search=True, random_state=42, target_experiment=None):
    """
    Output: dicts-conclude all results and splits
    """
    print("=== Random forest step ===")
    
    os.makedirs(output_dir, exist_ok=True)
    

    
    print("Protein partition:")
    splits = kahip_split(X, y, data, output_dir=output_dir, min_bitscore=min_bitscore)
    
    if target_experiment:
        experiments_to_run = [target_experiment]
        print(f"evaluatesingle exp: {target_experiment}")
    else:
        experiments_to_run = ['exp1', 'exp2', 'exp3', 'exp4']
        print("evaluate all exp")
    
    # Step 1: Grid search
    best_params = None
    if use_grid_search and (target_experiment is None or target_experiment == 'exp1'):
        print("\n=== Step1: Hyper parameter search in exp1===")
        exp1_train_idx, exp1_test_idx = splits['exp1']
        X_exp1_train, y_exp1_train = X[exp1_train_idx], y[exp1_train_idx]
        
        
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 3, 4],
            'class_weight': [None, 'balanced'],
        }
        
        rf_base = RandomForestClassifier(random_state=random_state)
        scorer = make_scorer(f1_score)
        
        grid_search = GridSearchCV(
            estimator=rf_base,
            param_grid=param_grid,
            scoring=scorer,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state),
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_exp1_train, y_exp1_train)
        best_params = grid_search.best_params_
        print(f"best params: {best_params}")
    else:
        best_params = ProjectConfig.DEFAULT_RF_PARAMS.copy()
        print(f"use default params: {best_params}")
    
    # Step 2: 3-fold cv
    print("\n=== Step 2: 3-fold cross validation ===")
    
    results = {}
    experiment_names = ProjectConfig.EXPERIMENT_NAMES
    
    for exp_name in experiments_to_run:
        if exp_name not in splits:
            print(f"Warning: {exp_name} not in splits...")
            continue
            
        print(f"\n----- EXPERIMENT {exp_name}: {experiment_names[exp_name]} -----")
        
        train_idx, test_idx = splits[exp_name]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        print("perform 3-fold CV...")
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        
        cv_scores = []
        cv_predictions = []
        cv_probabilities = []
        cv_true_labels = []
        
        fold = 1
        for train_cv_idx, val_cv_idx in cv.split(X_train, y_train):
            print(f"  Fold {fold}/3")
            
            X_train_cv, X_val_cv = X_train[train_cv_idx], X_train[val_cv_idx]
            y_train_cv, y_val_cv = y_train[train_cv_idx], y_train[val_cv_idx]
            
            rf_model = RandomForestClassifier(**best_params, random_state=random_state)
            rf_model.fit(X_train_cv, y_train_cv)
            
            y_val_pred = rf_model.predict(X_val_cv)
            y_val_proba = rf_model.predict_proba(X_val_cv)[:, 1]
            
            fold_metrics = calculate_metrics(y_val_cv, y_val_pred, y_val_proba)
            cv_scores.append(fold_metrics)
            
            cv_predictions.extend(y_val_pred)
            cv_probabilities.extend(y_val_proba)
            cv_true_labels.extend(y_val_cv)
            
            fold += 1
        
        cv_mean_metrics = {}
        for metric in cv_scores[0].keys():
            cv_mean_metrics[f'cv_{metric}_mean'] = np.mean([score[metric] for score in cv_scores])
            cv_mean_metrics[f'cv_{metric}_std'] = np.std([score[metric] for score in cv_scores])
        
        print("final model training...")
        final_rf_model = RandomForestClassifier(**best_params, random_state=random_state)
        final_rf_model.fit(X_train, y_train)
        
        y_pred = final_rf_model.predict(X_test)
        y_proba = final_rf_model.predict_proba(X_test)[:, 1]
    
        test_metrics = calculate_metrics(y_test, y_pred, y_proba)
        uncertainty = calculate_uncertainty(y_proba) if y_proba is not None else None
        
        # save
        results[exp_name] = {
            'metrics': test_metrics,
            'cv_metrics': cv_mean_metrics,
            'cv_scores': cv_scores,
            'predictions': {
                'y_true': y_test,
                'y_pred': y_pred,
                'y_proba': y_proba,
                'uncertainty': uncertainty
            },
            'cv_predictions': {
                'y_true': np.array(cv_true_labels),
                'y_pred': np.array(cv_predictions),
                'y_proba': np.array(cv_probabilities)
            },
            'model': final_rf_model,
            'best_params': best_params
        }
        
        # Print results
        print(f"\n{exp_name} final model results:")
        for metric in ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 'auc']:
            if f'cv_{metric}_mean' in cv_mean_metrics:
                mean_val = cv_mean_metrics[f'cv_{metric}_mean']
                std_val = cv_mean_metrics[f'cv_{metric}_std']
                print(f"  CV {metric}: {mean_val:.4f} ± {std_val:.4f}")
        
        print(f"\n{exp_name} final model test results:")
        print_evaluation_results(y_test, y_pred, y_proba)
        
        _generate_plots(exp_name, y_test, y_pred, y_proba, output_dir)
    
    # Step 3: generate combined plot
    partition_results = {'results': results, 'splits': splits}
    
    if target_experiment is None:
        print("\n=== Step 3: generate combined plot ===")
        
        print("generate combined ROC curve...")
        combined_roc_path = os.path.join(output_dir, "combined_roc.png")
        plot_combined_roc_curves(partition_results, combined_roc_path)
        
        print("generate combined metrics comparison plot...")
        metrics_comparison_path = os.path.join(output_dir, "combined_metrics.png")
        plot_combined_metrics_comparison(partition_results, metrics_comparison_path)
    
    
    print("\n=== Evaluation completed ===")
    return partition_results
