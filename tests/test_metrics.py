"""Evaluation metrics against hand-computed values and SciPy references."""

import numpy as np
import pytest
from scipy import stats

from src.evaluate import (
    classification_metrics,
    equal_error_rate,
    expected_calibration_error,
    fit_temperature,
    pearson,
    plot_confusion_matrix,
    plot_phone_distributions,
    roc_auc,
    softmax_np,
    spearman,
)


def test_classification_metrics_by_hand():
    y_true = [0, 0, 0, 0, 1, 1, 2, 2]
    y_pred = [0, 0, 0, 1, 1, 2, 2, 2]
    m = classification_metrics(y_true, y_pred, ["a", "b", "c"])
    assert m["accuracy"] == pytest.approx(6 / 8)
    # recalls: a 3/4, b 1/2, c 2/2 -> UAR = mean recall (balanced accuracy)
    assert m["uar"] == pytest.approx((0.75 + 0.5 + 1.0) / 3)
    # precisions: a 3/3, b 1/2, c 2/3
    f1 = [2 * p * r / (p + r) for p, r in ((1.0, 0.75), (0.5, 0.5), (2 / 3, 1.0))]
    assert m["macro_f1"] == pytest.approx(np.mean(f1))
    assert m["per_class"]["b"] == pytest.approx(
        {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 2}
    )
    assert m["confusion_matrix"] == [[3, 1, 0], [0, 1, 1], [0, 0, 2]]


def test_absent_classes_do_not_drag_macro_scores_down():
    m = classification_metrics([0, 1], [0, 1], ["a", "b", "never_seen"])
    assert m["macro_f1"] == 1.0 and m["uar"] == 1.0


def test_ece_counts_full_confidence_predictions():
    probs = np.array([[1.0, 0.0], [1.0, 0.0], [0.6, 0.4]])
    ece, bins = expected_calibration_error(probs, [0, 1, 0], n_bins=10)
    assert sum(b["count"] for b in bins) == 3  # nothing falls out of the top bin
    # bin 1.0: acc 0.5 vs conf 1.0 (2/3 of samples); bin 0.6: acc 1 vs conf 0.6 (1/3)
    assert ece == pytest.approx(2 / 3 * 0.5 + 1 / 3 * 0.4)


def test_temperature_scaling_recovers_a_known_temperature():
    rng = np.random.default_rng(0)
    true_t = 2.5
    logits = rng.normal(0, 3, size=(4000, 5))
    labels = [rng.choice(5, p=p) for p in softmax_np(logits, true_t)]
    assert fit_temperature(logits, labels) == pytest.approx(true_t, rel=0.1)


def test_correlations_match_scipy_including_ties():
    x = [1, 2, 2, 3, 5, 5, 5, 8]
    y = [2, 1, 4, 3, 6, 7, 5, 9]
    assert pearson(x, y) == pytest.approx(stats.pearsonr(x, y)[0])
    assert spearman(x, y) == pytest.approx(stats.spearmanr(x, y)[0])
    assert np.isnan(pearson([1, 1, 1], [1, 2, 3]))  # constant input: undefined, not a crash


def test_auc_and_eer():
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    perfect = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
    assert roc_auc(perfect, labels) == 1.0
    assert equal_error_rate(perfect, labels)[0] == pytest.approx(0.0)
    assert roc_auc([0.5] * 8, labels) == 0.5  # ties count half
    inverted = perfect[::-1]
    assert roc_auc(inverted, labels) == 0.0
    # one negative above one positive: AUC = 1 - 1/16
    swapped = [0.1, 0.2, 0.3, 0.65, 0.6, 0.7, 0.8, 0.9]
    assert roc_auc(swapped, labels) == pytest.approx(15 / 16)
    eer, _ = equal_error_rate(swapped, labels)
    assert 0.0 < eer <= 0.25
    rng = np.random.default_rng(0)
    rand_labels = rng.integers(0, 2, 4000)
    eer_random, _ = equal_error_rate(rng.random(4000), rand_labels)
    assert eer_random == pytest.approx(0.5, abs=0.05)
    assert np.isnan(roc_auc([0.1, 0.2], [1, 1]))  # one class only


def test_plots_are_written(tmp_path):
    cm = np.array([[5, 1], [2, 4]])
    assert (tmp_path / "cm.png").exists() is False
    plot_confusion_matrix(cm, ["Tamil", "Hindi"], str(tmp_path / "cm.png"))
    plot_phone_distributions({"θ": [0.2, 0.4, 0.3], "s": [0.8, 0.9]}, str(tmp_path / "p.png"))
    assert (tmp_path / "cm.png").stat().st_size > 1000
    assert (tmp_path / "p.png").stat().st_size > 1000
