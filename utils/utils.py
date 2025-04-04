from numba import njit
import numpy as np

# @njit
def segment_metrics(y_val, y_pred):
    if y_val.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Both arrays must be 1D.")
    if y_val.shape[0] != y_pred.shape[0]:
        raise ValueError("Arrays must have the same length.")
    
    TPe = 0
    FNe = 0
    FPe = 0
    FPt = 0
    Nt = np.sum(y_pred)

    in_val_segment=False
    in_pred_segment=False
    count_val=0
    count_pred=0
    
    for i in range(y_val.shape[0]):
        if y_pred[i]==1 and y_val[i]==1:
            count_val+=1
            count_pred+=1

        if y_pred[i]==1 and y_val[i]==0:
            FPt+=1

        if y_val[i]==1:
            in_val_segment=True
        elif in_val_segment:
            in_val_segment=False
            if count_val>0:
                TPe+=1
            else:
                FNe+=1
            count_val=0
            
        if y_pred[i]==1:
            in_pred_segment=True
        elif in_pred_segment:
            in_pred_segment=False
            if count_pred==0:
                FPe+=1
            
            count_pred=0

    return TPe, FNe, FPe, FPt, Nt

def corrected_eventwise_F0_5_score(y_val, y_pred):
    TPe, FNe, FPe, FPt, Nt = segment_metrics(y_val, y_pred)

    precision_e_corr = TPe/(TPe+FPe)*(1-FPt/Nt)
    recall_e = TPe/(TPe+FNe)
    score = (1+0.5**2)*precision_e_corr*recall_e/(0.5**2*precision_e_corr+recall_e)

    return float(score)
