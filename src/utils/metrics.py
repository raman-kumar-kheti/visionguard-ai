import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Tuple, List

def compute_metrics(labels: List[int], scores: List[float]) -> Tuple[float, float]:
    """
    Computes AUROC and a basic threshold for F1-score optimization.
    """
    labels_np = np.array(labels)
    scores_np = np.array(scores)
    
    if len(np.unique(labels_np)) == 1:
        # Cannot compute AUROC with only one class
        return 0.0, np.max(scores_np)
        
    auroc = roc_auc_score(labels_np, scores_np)
    
    # Simple thresholding: middle of min anomaly and max normal (if separable), 
    # or just use percentile/mean + std on normal scores
    normal_scores = scores_np[labels_np == 0]
    threshold = np.mean(normal_scores) + 3 * np.std(normal_scores) if len(normal_scores) > 0 else 0.5
    
    return float(auroc), float(threshold)
