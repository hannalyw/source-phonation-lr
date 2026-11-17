import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_curve

LOG2 = np.log(2.0)


def cllr(llr, y):
    llr, y = np.asarray(llr, float), np.asarray(y, int)
    tar, non = llr[y == 1], llr[y == 0]

    a = np.mean(np.logaddexp(0.0, -tar)) / LOG2
    b = np.mean(np.logaddexp(0.0, non)) / LOG2

    return 0.5 * (a + b)


def empirical_min_cllr(raw_score, y):
    s, y = np.asarray(raw_score, float), np.asarray(y, int)
    nt, nn = max(y.sum(), 1), max((1 - y).sum(), 1)
    w = np.where(y == 1, 0.5 / nt, 0.5 / nn)

    iso = IsotonicRegression(
        y_min=1e-6,
        y_max=1-1e-6,
        increasing=True,
        out_of_bounds="clip"
    )

    p = iso.fit_transform(s, y, sample_weight=w)
    llr_opt = np.log(p) - np.log1p(-p)

    return cllr(llr_opt, y)


def eer(raw_score, y):
    fpr, tpr, _ = roc_curve(y, raw_score, pos_label=1)
    fnr = 1.0 - tpr
    i = np.nanargmin(np.abs(fpr - fnr))

    return float((fpr[i] + fnr[i]) / 2)


def misleading_rates(llr, y, strong_lr=10.0):
    llr, y = np.asarray(llr, float), np.asarray(y, int)
    wrong = np.where(y == 1, llr < 0, llr > 0)

    t = np.log(strong_lr)
    strong = np.where(y == 1, llr < -t, llr > t)

    return {
        "misleading": wrong.mean(),
        "strong_misleading": strong.mean()
    }
