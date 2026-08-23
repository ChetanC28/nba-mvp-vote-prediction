"""Adaptive weighted ensemble for NBA MVP vote prediction.

Learns a convex combination of base-model predictions by directly minimising
leave-one-season-out squared error, then swaps in a different weight vector at
prediction time depending on how much the base models disagree about a player.

The class below is the original implementation, unchanged. It was extracted
from a single-file script (`weighted_model.py`, still in this repo's history on
the `weighted-ensemble` branch) so that the benchmark driver can import it.
"""

import numpy as np
from scipy.optimize import minimize


class AdaptiveWeightedEnsemble:
    """
    Learns optimal base model weights using leave-one-season-out CV,
    then adaptively adjusts per-prediction based on model agreement.
    
    When models strongly disagree on a player, weights shift toward
    the models that were more reliable on similar disagreement levels
    during training. Built from scratch.
    """
    
    def __init__(self, n_base_models):
        self.n_base = n_base_models
        self.global_weights = None
        self.high_agreement_weights = None
        self.low_agreement_weights = None
        self.disagreement_threshold = None
    
    def _optimize_weights(self, base_preds, y_true, years, min_weight=0.05):
        n_models = base_preds.shape[1]
        unique_years = np.unique(years)
        
        def loso_objective(w):
            fold_errors = []
            for year in unique_years:
                val_mask = years == year
                val_pred = base_preds[val_mask] @ w
                val_true = y_true[val_mask]
                fold_errors.append(np.mean((val_pred - val_true) ** 2))
            return np.mean(fold_errors)
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(min_weight, 1)] * n_models
        
        best_result = None
        best_loss = np.inf
        
        for _ in range(30):
            w0 = np.random.dirichlet(np.ones(n_models))
            w0 = np.clip(w0, min_weight, None)
            w0 /= w0.sum()
            result = minimize(loso_objective, w0, method='SLSQP',
                            bounds=bounds, constraints=constraints)
            if result.fun < best_loss:
                best_loss = result.fun
                best_result = result
        
        return best_result.x
    
    def fit(self, base_preds, years, y_true):
        self.global_weights = self._optimize_weights(base_preds, y_true, years)
        print(f"Global weights: {self.global_weights.round(3)}")
        
        disagreements = np.std(base_preds, axis=1)
        self.disagreement_threshold = np.median(disagreements)
        
        high_agree_mask = disagreements <= self.disagreement_threshold
        low_agree_mask = disagreements > self.disagreement_threshold
        
        print(f"Disagreement threshold: {self.disagreement_threshold:.2f}")
        print(f"High agreement samples: {high_agree_mask.sum()}, Low: {low_agree_mask.sum()}")
        
        if high_agree_mask.sum() > 500:
            self.high_agreement_weights = self._optimize_weights(
                base_preds[high_agree_mask], y_true[high_agree_mask], 
                years[high_agree_mask]
            )
            print(f"High agreement weights: {self.high_agreement_weights.round(3)}")
        else:
            self.high_agreement_weights = self.global_weights
        
        if low_agree_mask.sum() > 500:
            self.low_agreement_weights = self._optimize_weights(
                base_preds[low_agree_mask], y_true[low_agree_mask],
                years[low_agree_mask]
            )
            print(f"Low agreement weights: {self.low_agreement_weights.round(3)}")
        else:
            self.low_agreement_weights = self.global_weights
    
    def predict(self, base_preds):
        disagreements = np.std(base_preds, axis=1)
        predictions = np.zeros(len(base_preds))
        weights_used = np.zeros((len(base_preds), self.n_base))
        
        for i in range(len(base_preds)):
            if disagreements[i] <= self.disagreement_threshold:
                w = self.high_agreement_weights
            else:
                blend = min(disagreements[i] / (2 * self.disagreement_threshold), 1.0)
                w = (1 - blend) * self.low_agreement_weights + blend * self.global_weights
            
            predictions[i] = base_preds[i] @ w
            weights_used[i] = w
        
        return predictions, weights_used
