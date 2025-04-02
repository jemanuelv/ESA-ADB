from numba import njit

@njit
def segment_metrics(y_val, y_pred):
    if y_val.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("Both arrays must be 1D.")
    if y_val.shape[0] != y_pred.shape[0]:
        raise ValueError("Arrays must have the same length.")
    
    TPe = 0
    FNe = 0
    FPe = 0
    FPt = 0
    Nt = 0


    return TPe, FNe, FPe, FPt, Nt
