from . import calibration_tools
import tensorflow as tf


def get_confidence(loader, model, mask, type="odin"):
    if type == "odin":
        logits = model.predict(loader)[:, mask]
        confidence = tf.nn.softmax(logits, axis=1)
        confidence = tf.reduce_max(confidence, 1)
    else:
        confidence = model.predict(loader)[:, mask]
        confidence = tf.reduce_max(confidence, 1)
    return confidence.numpy()


def calculate_auroc(in_loader, out_loader, model, mask, type):
    confidence_in, correct, num_correct = get_confidence(
        in_loader, model=model, mask=mask, type=type
    )
    in_score = -confidence_in
    confidence_out, correct_out, num_correct_out = get_confidence(
        out_loader, model=model, mask=mask, type=type
    )
    out_score = -confidence_out

    auroc = calibration_tools.get_measures(out_score, in_score)
    return auroc
