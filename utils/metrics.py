import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate(model, X_val, y_val, name="Model"):
    y_pred = model.predict(X_val)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_val)[:, 1]
    else:
        y_prob = y_pred
    
    auc = roc_auc_score(y_val, y_prob)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    
    print(f"=== {name} ===")
    print(f"ROC-AUC:   {auc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))
    print("\\n")
    return {"ROC-AUC": auc, "Precision": prec, "Recall": rec, "F1-score": f1}
