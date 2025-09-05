import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import seaborn as sns

class CalibrationEvaluator:
    """
    Evaluator for model calibration metrics including ECE, calibration plots, and detailed metrics.
    """
    
    def __init__(self, n_bins=10):
        self.n_bins = n_bins
    
    def get_ece(self, S, T):
        """
        Calculate Expected Calibration Error (ECE).
        
        Args:
            S (array-like): Predicted probabilities
            T (array-like): True binary labels
            
        Returns:
            float: Expected Calibration Error
        """
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (S > bin_lower) & (S <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = T[in_bin].mean()
                avg_confidence_in_bin = S[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return ece
    
    def plot_calibration_curve(self, S, T, save_path=None):
        """
        Plot calibration curve.
        """
        fraction_of_positives, mean_predicted_value = calibration_curve(
            T, S, n_bins=self.n_bins
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.title("Calibration Plot")
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def calculate_detailed_metrics(self, T, S):
        """
        Calculate detailed calibration metrics.

        Returns:
            dict: Dictionary containing ECE, Brier score, and MCE
        """
        ece = self.get_ece(S, T)
        brier_score = brier_score_loss(T, S)
        
        # Calculate Maximum Calibration Error (MCE)
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (S > bin_lower) & (S <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = T[in_bin].mean()
                avg_confidence_in_bin = S[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return {'ece': ece,'brier_score': brier_score,'mce': mce}

def calculate_uncertainty(S):
    """
    Calculate uncertainty based on prediction probabilities.
    """
    # Avoid log(0) by adding small epsilon
    epsilon = 1e-15
    S = np.clip(S, epsilon, 1 - epsilon)
    
    # Calculate entropy as uncertainty measure
    uncertainty = -(S * np.log(S) + (1 - S) * np.log(1 - S))
    return uncertainty