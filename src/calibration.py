import numpy as np, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class LogLRCalibrator:
    def __init__(self, C=1.0):
        self.scaler = StandardScaler()
        self.model = LogisticRegression(C=C, solver="lbfgs", max_iter=2000)

    def fit(self, scores, y):
        X = np.asarray(scores, float).reshape(-1, 1)
        y = np.asarray(y, int)
        X = self.scaler.fit_transform(X)
        nt, nn = max(y.sum(),1), max((1-y).sum(),1)
        w = np.where(y == 1, len(y)/(2*nt), len(y)/(2*nn))
        self.model.fit(X, y, sample_weight=w)
        return self

    def transform(self, scores):
        X = self.scaler.transform(np.asarray(scores,float).reshape(-1,1))
        return self.model.decision_function(X)  # natural-log LLR, equal prior

# cal.fit(calibration.raw_score, calibration.target)
# validation["loglr"] = cal.transform(validation.raw_score)
# joblib.dump(cal, "models/calibrator.joblib")
