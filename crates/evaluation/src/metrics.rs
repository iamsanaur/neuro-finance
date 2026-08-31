//! Classification metrics (project spec §28, §31): accuracy and per-class
//! F1, computed from parallel `predicted`/`actual` class-index slices.
//! Regression error metrics, AUC, and directional accuracy are deferred —
//! no regression or binary-direction task exists yet (§26/§27, later
//! milestones); adding metrics for tasks that don't exist would be
//! untestable.

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub struct ClassMetrics {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub support: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ClassificationReport {
    pub accuracy: f64,
    /// Keyed by class index.
    pub per_class: HashMap<usize, ClassMetrics>,
}

/// `predicted` and `actual` must be the same length and non-empty.
pub fn classification_report(
    predicted: &[usize],
    actual: &[usize],
    num_classes: usize,
) -> ClassificationReport {
    assert_eq!(
        predicted.len(),
        actual.len(),
        "predicted and actual must be the same length"
    );
    assert!(
        !predicted.is_empty(),
        "cannot compute metrics over zero examples"
    );

    let n = predicted.len();
    let correct = predicted.iter().zip(actual).filter(|(p, a)| p == a).count();
    let accuracy = correct as f64 / n as f64;

    let mut per_class = HashMap::new();
    for class in 0..num_classes {
        let true_positive = predicted
            .iter()
            .zip(actual)
            .filter(|(&p, &a)| p == class && a == class)
            .count();
        let predicted_positive = predicted.iter().filter(|&&p| p == class).count();
        let actual_positive = actual.iter().filter(|&&a| a == class).count();

        let precision = if predicted_positive == 0 {
            0.0
        } else {
            true_positive as f64 / predicted_positive as f64
        };
        let recall = if actual_positive == 0 {
            0.0
        } else {
            true_positive as f64 / actual_positive as f64
        };
        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };

        per_class.insert(
            class,
            ClassMetrics {
                precision,
                recall,
                f1,
                support: actual_positive,
            },
        );
    }

    ClassificationReport {
        accuracy,
        per_class,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_predictions_give_accuracy_and_f1_of_one() {
        let predicted = vec![0, 1, 2, 0, 1];
        let actual = vec![0, 1, 2, 0, 1];
        let report = classification_report(&predicted, &actual, 3);
        assert!((report.accuracy - 1.0).abs() < 1e-12);
        for class in 0..3 {
            let metrics = &report.per_class[&class];
            assert!(
                (metrics.f1 - 1.0).abs() < 1e-12,
                "class {class} f1 should be 1.0"
            );
        }
    }

    #[test]
    fn all_wrong_predictions_give_zero_accuracy() {
        let predicted = vec![1, 2, 0];
        let actual = vec![0, 1, 2];
        let report = classification_report(&predicted, &actual, 3);
        assert_eq!(report.accuracy, 0.0);
    }

    #[test]
    fn class_never_predicted_has_zero_precision_but_defined_recall() {
        // Class 2 exists in `actual` but the model never predicts it.
        let predicted = vec![0, 0, 1];
        let actual = vec![0, 2, 1];
        let report = classification_report(&predicted, &actual, 3);
        let class2 = &report.per_class[&2];
        assert_eq!(class2.precision, 0.0);
        assert_eq!(class2.recall, 0.0); // never correctly predicted either
        assert_eq!(class2.support, 1);
    }

    #[test]
    fn support_counts_true_occurrences_per_class() {
        let predicted = vec![0, 0, 0, 0];
        let actual = vec![0, 1, 1, 2];
        let report = classification_report(&predicted, &actual, 3);
        assert_eq!(report.per_class[&0].support, 1);
        assert_eq!(report.per_class[&1].support, 2);
        assert_eq!(report.per_class[&2].support, 1);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn rejects_mismatched_lengths() {
        classification_report(&[0, 1], &[0], 3);
    }

    #[test]
    #[should_panic(expected = "zero examples")]
    fn rejects_empty_input() {
        classification_report(&[], &[], 3);
    }
}
