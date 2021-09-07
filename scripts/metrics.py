# Code referred from:
# https://github.com/pokaxpoka/deep_Mahalanobis_detector/blob/master/calculate_log.py

import tensorflow as tf
import numpy as np


def get_confidence(loader, model, stype="generalized_odin"):
    if stype == "generalized_odin":
        logits = model.predict(loader)
        confidence = tf.reduce_max(logits, 1)
    else:
        confidence = model.predict(loader)
        confidence = tf.reduce_max(confidence, 1)
    return confidence.numpy()


def calculate_auroc(in_loader, out_loader, model, stype):
    tp, fp = dict(), dict()
    tnr_at_tpr95 = dict()

    confidence_in = get_confidence(in_loader, model=model, stype=stype)
    confidence_out = get_confidence(out_loader, model=model, stype=stype)

    confidence_in.sort()
    confidence_out.sort()

    end = np.max([np.max(confidence_in), np.max(confidence_out)])
    start = np.min([np.min(confidence_in), np.min(confidence_out)])

    num_k = confidence_in.shape[0]
    num_n = confidence_out.shape[0]
    tp[stype] = -np.ones([num_k + num_n + 1], dtype=int)
    fp[stype] = -np.ones([num_k + num_n + 1], dtype=int)
    tp[stype][0], fp[stype][0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k + num_n):
        if k == num_k:
            tp[stype][l + 1 :] = tp[stype][l]
            fp[stype][l + 1 :] = np.arange(fp[stype][l] - 1, -1, -1)
            break
        elif n == num_n:
            tp[stype][l + 1 :] = np.arange(tp[stype][l] - 1, -1, -1)
            fp[stype][l + 1 :] = fp[stype][l]
            break
        else:
            if confidence_out[n] < confidence_in[k]:
                n += 1
                tp[stype][l + 1] = tp[stype][l]
                fp[stype][l + 1] = fp[stype][l] - 1
            else:
                k += 1
                tp[stype][l + 1] = tp[stype][l] - 1
                fp[stype][l + 1] = fp[stype][l]
    tpr95_pos = np.abs(tp[stype] / num_k - 0.95).argmin()
    tnr_at_tpr95[stype] = 1.0 - fp[stype][tpr95_pos] / num_n

    return tp, fp, tnr_at_tpr95


def metric(in_loader, out_loader, model, stype):
    tp, fp, tnr_at_tpr95 = calculate_auroc(in_loader, out_loader, model, stype)

    results = dict()
    results[stype] = dict()

    # TNR
    mtype = "TNR"
    results[stype][mtype] = tnr_at_tpr95[stype]

    # AUROC
    mtype = "AUROC"
    tpr = np.concatenate([[1.0], tp[stype] / tp[stype][0], [0.0]])
    fpr = np.concatenate([[1.0], fp[stype] / fp[stype][0], [0.0]])
    results[stype][mtype] = -np.trapz(1.0 - fpr, tpr)

    return results
