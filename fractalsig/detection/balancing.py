"""Shared class-balancing utilities -- de-duplicates the identical
inverse-frequency formula and WeightedTrainer class that were independently
hand-rolled in p3_train_eval.py, roberta_finetune.py, p4_task5_roberta.py,
and p4_b2_ladder_score_all.py (found by code review 2026-08-29). Math is
UNCHANGED from the original -- this is a pure extraction, not a switch to
sklearn.utils.class_weight.compute_sample_weight (checked and rejected:
that function has no max(n,1) guard, so it diverges from this formula in
the single-class-in-a-fold case this project's LODO/LOGO splits can
produce).

torch/transformers are imported LAZILY, inside make_weighted_trainer(), so
that importing this module for balanced_class_weights/balanced_sample_weight
alone (p3_train_eval.py's use case) does not drag torch into every CPU-only
P3 script that currently has no torch dependency.
"""
import numpy as np


def balanced_class_weights(y):
    """Inverse-frequency (w_neg, w_pos) scalars, matching
    class_weight='balanced' in spirit. y: array-like of 0/1 labels."""
    y = np.asarray(y)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    w_pos = len(y) / (2.0 * max(n_pos, 1))
    w_neg = len(y) / (2.0 * max(n_neg, 1))
    return w_neg, w_pos


def balanced_sample_weight(y):
    """Per-row sample_weight array for estimators with no native
    class_weight (e.g. MLPClassifier via Pipeline fit_params)."""
    y = np.asarray(y)
    w_neg, w_pos = balanced_class_weights(y)
    return np.where(y == 1, w_pos, w_neg)


def make_weighted_trainer(class_weights):
    """Returns a Trainer subclass with compute_loss weighted by
    class_weights (a torch tensor [w_neg, w_pos]). Factory, not a shared
    class instance, so each caller's class_weights closes over correctly."""
    import torch
    from transformers import Trainer

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
            loss = loss_fct(logits, labels)
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer
