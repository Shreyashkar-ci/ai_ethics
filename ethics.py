import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
import math
from scipy.stats import ks_2samp, entropy, gaussian_kde
from typing import List, Dict, Union, Optional


class FairnessMetrics:
    def __init__(self,
                 predictions: np.ndarray,
                 ground_truth: np.ndarray):
        """
        Initialize Fairness Metrics base class
        Parameters:
        -----------
        predictions: np.ndarray
        ground_truth: np.ndarray
        """
        self.predictions = np.array(predictions)
        self.ground_truth = np.array(ground_truth)

        # Validate inputs
        assert len(self.predictions) == len(self.ground_truth), \
            "Predictions and ground truth must have the same length"
        
class GroupFairnessScore:
    def __init__(self,
                 sensitive_features: pd.DataFrame,
                 statistical_target_feature: Optional[str] = None):
        """
        Initialize Group Fairness Analyzer

        Parameters:
        -----------
        sensitive_features : pd.DataFrame
            DataFrame containing sensitive feature columns
        statistical_target_feature : str, optional
            Feature to use for certain conditional parity metrics
        """
        self.sensitive_features = sensitive_features
        self.statistical_target_feature = statistical_target_feature

    def _get_group_masks(self, feature_name: str) -> Dict[str, np.ndarray]:
        """
        Utility method to get boolean masks for each group of a given feature.
        """
        masks = {}
        for group in self.sensitive_features[feature_name].unique():
            masks[group] = (self.sensitive_features[feature_name] == group)
        return masks

    def _get_confusion_counts(self, fairness_metrics: FairnessMetrics, group_mask: np.ndarray) -> Dict[str, int]:
        """
        Compute confusion matrix counts: TP, FP, TN, FN for a given group mask.
        """
        preds = fairness_metrics.predictions[group_mask]
        truth = fairness_metrics.ground_truth[group_mask]

        tp = np.sum((preds == 1) & (truth == 1))
        fp = np.sum((preds == 1) & (truth == 0))
        tn = np.sum((preds == 0) & (truth == 0))
        fn = np.sum((preds == 0) & (truth == 1))

        return {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}

    def demographic_parity(self,
                            fairness_metrics: FairnessMetrics,
                            feature_name: str) -> Dict:
        group_rates = {}
        masks = self._get_group_masks(feature_name)
        for group, mask in masks.items():
            group_rates[group] = np.mean(fairness_metrics.predictions[mask] == 1)

        rates = list(group_rates.values())
        # Extracting keys and values

        disparity = max(rates) - min(rates) if len(rates) > 1 else 0

        return {
            'feature': feature_name,
            'disparity': disparity,
        }

    def equal_opportunity(self,
                           fairness_metrics: FairnessMetrics,
                           feature_name: str) -> Dict:
        group_tpr = {}
        masks = self._get_group_masks(feature_name)
        for group, mask in masks.items():
            group_positive_mask = (fairness_metrics.ground_truth[mask] == 1)
            if np.sum(group_positive_mask) > 0:
                group_tpr[group] = np.mean(
                    fairness_metrics.predictions[mask][group_positive_mask] == 1
                )
            else:
                group_tpr[group] = 0

        tpr_values = list(group_tpr.values())
        disparity = max(tpr_values) - min(tpr_values) if len(tpr_values) > 1 else 0

        return {
            'feature': feature_name,
            'disparity': disparity,
        }

    def predictive_parity(self,
                           fairness_metrics: FairnessMetrics,
                           feature_name: str) -> Dict:
        group_precision = {}
        masks = self._get_group_masks(feature_name)
        for group, mask in masks.items():
            pred_pos = fairness_metrics.predictions[mask] == 1
            if np.sum(pred_pos) > 0:
                group_precision[group] = np.mean(
                    fairness_metrics.ground_truth[mask][pred_pos] == 1
                )
            else:
                group_precision[group] = 0

        precision_values = list(group_precision.values())
        disparity = max(precision_values) - min(precision_values) if len(precision_values) > 1 else 0

        return {
            'feature': feature_name,
            'disparity': disparity,
        }

    def js_divergence(self,
                      fairness_metrics: FairnessMetrics,
                      feature_name: str) -> Dict:

        masks = self._get_group_masks(feature_name)
        js_divergences = {}

        for group, mask in masks.items():
            preds = fairness_metrics.predictions[mask]
            truth = fairness_metrics.ground_truth[mask]

            def estimate_distribution(data, min_val=0, max_val=1):
              if len(data) < 2:
                return None
              kde = gaussian_kde(data)
              x_range = np.linspace(min_val, max_val, 100)
              return kde(x_range), x_range

            pred_dist = estimate_distribution(preds)
            truth_dist = estimate_distribution(truth)

            if pred_dist is None or truth_dist is None:
              js_divergences[group] = np.nan
              continue

            pred_pdf, x_range = pred_dist
            truth_pdf, _ = truth_dist

            # Normalize distributions
            pred_pdf = pred_pdf / np.sum(pred_pdf)
            truth_pdf = truth_pdf / np.sum(truth_pdf)

            avg_pdf = 0.5 * (pred_pdf + truth_pdf)

            eps = 1e-10
            js_div = 0.5*(
                np.sum(pred_pdf*np.log((pred_pdf+eps)/(avg_pdf+eps))) +
                np.sum(truth_pdf*np.log((truth_pdf+eps)/(avg_pdf+eps)))
            )
            js_divergences[group] = js_div

        max_div = max(js_divergences.values())
        min_div = min(js_divergences.values())

        js_div = max_div - min_div

        return {
            'feature': feature_name,
            'js_divergence_diff': js_div
        }

    def disparate_impact(self,
                         fairness_metrics: FairnessMetrics,
                         feature_name: str) -> Dict:
        # Disparate Impact: ratio of positive rates between groups
        masks = self._get_group_masks(feature_name)
        group_rates = {g: np.mean(fairness_metrics.predictions[m] == 1) for g, m in masks.items()}

        rates = list(group_rates.values())
        if len(rates) < 2:
            di = 1.0  # If only one group, no disparity
        else:
            max_rate = max(rates)
            min_rate = min(rates)
            if max_rate == 0:
                di = 1.0
            else:
                di = min_rate / max_rate

        return {
            'feature': feature_name,
            'disparate_impact': di,
        }

    def equal_opportunity_difference(self,
                                     fairness_metrics: FairnessMetrics,
                                     feature_name: str) -> Dict:
        # Equal Opportunity Difference: difference in TPR between groups
        group_tpr = {}
        masks = self._get_group_masks(feature_name)
        for group, mask in masks.items():
            group_positive_mask = (fairness_metrics.ground_truth[mask] == 1)
            if np.sum(group_positive_mask) > 0:
                group_tpr[group] = np.mean(fairness_metrics.predictions[mask][group_positive_mask] == 1)
            else:
                group_tpr[group] = 0

        tpr_values = list(group_tpr.values())
        eod = max(tpr_values) - min(tpr_values) if len(tpr_values) > 1 else 0

        return {
            'feature': feature_name,
            'equal_opportunity_difference': eod,
        }

    def equalized_odds(self,
                       fairness_metrics: FairnessMetrics,
                       feature_name: str) -> Dict:
        # Equalized Odds: TPR and FPR should be equal across groups
        # We'll measure disparity in both TPR and FPR.
        masks = self._get_group_masks(feature_name)
        group_tpr = {}
        group_fpr = {}

        for group, mask in masks.items():
            counts = self._get_confusion_counts(fairness_metrics, mask)
            tp = counts['TP']
            fp = counts['FP']
            fn = counts['FN']
            tn = counts['TN']

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            group_tpr[group] = tpr
            group_fpr[group] = fpr

        tpr_values = list(group_tpr.values())
        fpr_values = list(group_fpr.values())

        tpr_diff = max(tpr_values) - min(tpr_values) if len(tpr_values) > 1 else 0
        fpr_diff = max(fpr_values) - min(fpr_values) if len(fpr_values) > 1 else 0

        return {
            'feature': feature_name,
            'diff': {
                'tpr_diff': tpr_diff,
                'fpr_diff': fpr_diff
            },
        }

    def statistical_parity_difference(self,
                                      fairness_metrics: FairnessMetrics,
                                      feature_name: str) -> Dict:
        # Statistical Parity Difference: difference in predicted positive rate
        # This is essentially what we computed in demographic_parity but by name.
        masks = self._get_group_masks(feature_name)
        group_rates = {g: np.mean(fairness_metrics.predictions[m] == 1) for g,m in masks.items()}

        rates = list(group_rates.values())
        sp_diff = max(rates) - min(rates) if len(rates) > 1 else 0

        return {
            'feature': feature_name,
            'statistical_parity_difference': sp_diff,
        }

    def treatment_equality(self,
                           fairness_metrics: FairnessMetrics,
                           feature_name: str) -> Dict:
        # Treatment Equality: equality in ratio of FPR to FNR across groups
        masks = self._get_group_masks(feature_name)
        group_te = {}

        for group, mask in masks.items():
            counts = self._get_confusion_counts(fairness_metrics, mask)
            tp, fp, tn, fn = counts['TP'], counts['FP'], counts['TN'], counts['FN']

            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

            if fnr == 0:
                te_val = np.inf if fpr > 0 else 1.0
            else:
                te_val = fpr / fnr

            group_te[group] = te_val

        te_values = [v for v in group_te.values() if np.isfinite(v)]
        if len(te_values) < 2:
            te_diff = 0
        else:
            te_diff = max(te_values) - min(te_values)

        return {
            'feature': feature_name,
            'treatment_equality_diff': te_diff,
        }

    def calibration_by_group(self,
                             fairness_metrics: FairnessMetrics,
                             feature_name: str) -> Dict:
        # Calibration by group: predictions match actual outcomes in frequency
        # Without predicted probabilities, we approximate calibration as
        # difference between predicted positive rate and actual positive rate.
        masks = self._get_group_masks(feature_name)
        group_cal = {}

        for group, mask in masks.items():
            pred_pos_rate = np.mean(fairness_metrics.predictions[mask] == 1)
            actual_pos_rate = np.mean(fairness_metrics.ground_truth[mask] == 1)
            calibration_error = abs(pred_pos_rate - actual_pos_rate)
            group_cal[group] = {
                'predicted_positive_rate': pred_pos_rate,
                'actual_positive_rate': actual_pos_rate,
                'calibration_error': calibration_error
            }

        # Disparity can be considered as difference in calibration error across groups
        errors = [g['calibration_error'] for g in group_cal.values()]
        cal_disparity = max(errors) - min(errors) if len(errors) > 1 else 0

        return {
            'feature': feature_name,
            'calibration_error_disparity': cal_disparity,
        }

    def conditional_parity(self,
                           fairness_metrics: FairnessMetrics,
                           feature_name: str) -> Dict:
        # Conditional parity:
        # For simplicity, if statistical_target_feature is provided, we measure
        # parity within each category of that feature, then average disparities.

        if self.statistical_target_feature is None:
            return {
                'feature': feature_name,
                'disparity': None
            }

        target_values = self.sensitive_features[self.statistical_target_feature].unique()
        masks_feature = self._get_group_masks(feature_name)

        # For each target_value, compute rates per group and measure disparity
        disparity_rates = {}
        disparities = []
        for tv in target_values:
            tv_mask = self.sensitive_features[self.statistical_target_feature] == tv
            group_rates_tv = {}
            for group, mask in masks_feature.items():
                combined_mask = mask & tv_mask
                if np.sum(combined_mask) > 0:
                    group_rates_tv[group] = np.mean(fairness_metrics.predictions[combined_mask] == 1)
                else:
                    group_rates_tv[group] = 0
            rates = list(group_rates_tv.values())
            if len(rates) > 1:
                disparities.append(max(rates) - min(rates))
                disparity_rates[tv] = (max(rates)-min(rates))

        if len(disparities) > 0:
            avg_disparity = np.mean(disparities)
        else:
            avg_disparity = 0

        return {
            'feature': self.statistical_target_feature,
            'average_disparity_across_conditions': avg_disparity,
        }

    def raw_outcome_disparity(self,
                              fairness_metrics: FairnessMetrics,
                              feature_name: str) -> Dict:
        # Raw Outcome Disparity: difference in actual outcome rates
        masks = self._get_group_masks(feature_name)
        group_outcomes = {g: np.mean(fairness_metrics.ground_truth[m]) for g,m in masks.items()}

        outcomes = list(group_outcomes.values())
        disparity = max(outcomes) - min(outcomes) if len(outcomes) > 1 else 0

        return {
            'feature': feature_name,
            'raw_outcome_disparity': disparity,
        }

    def generate_fairness_report(self,
                                 fairness_metrics: FairnessMetrics,
                                 features_to_analyze: List[str]) -> pd.DataFrame:
        """
        Generate comprehensive fairness report with a single metric_outcome column.

        Returns a DataFrame with columns: feature, metric, metric_outcome
        """
        report_data = []

        # Metrics to compute
        metrics = [
            ('Demographic Parity', self.demographic_parity),
            ('Equal Opportunity', self.equal_opportunity),
            ('Predictive Parity', self.predictive_parity),
            ('JS Divergence', self.js_divergence),
            ('Disparate Impact', self.disparate_impact),
            ('Equal Opportunity Difference', self.equal_opportunity_difference),
            ('Equalized Odds', self.equalized_odds),
            ('Statistical Parity Difference', self.statistical_parity_difference),
            ('Treatment Equality', self.treatment_equality),
            ('Calibration by Group', self.calibration_by_group),
            ('Conditional Parity', self.conditional_parity),
            ('Raw Outcome Disparity', self.raw_outcome_disparity),
        ]

        # Compute metrics for each feature
        for feature in features_to_analyze:
            for metric_name, metric_func in metrics:
                try:
                    result = metric_func(fairness_metrics, feature)

                    # Handle different result structures
                    if metric_name == 'Equalized Odds':
                        # Special case for Equalized Odds which returns a dict with 'diff'
                        metric_outcome = result.get('diff', {}).get('tpr_diff', 0)
                    elif isinstance(result, dict):
                        # For most metrics, find the numeric value
                        metric_outcome = next(
                            (v for k, v in result.items() if k not in ['feature', 'metric'] and isinstance(v, (int, float))),
                            0
                        )
                    else:
                        metric_outcome = 0

                    # Create a standardized report entry
                    report_entry = {
                        'feature': feature,
                        'metric': metric_name,
                        'metric_outcome': metric_outcome
                    }
                    report_data.append(report_entry)
                except Exception as e:
                    print(f"Could not compute {metric_name} for {feature}: {e}")

        # Create DataFrame with consistent columns
        df = pd.DataFrame(report_data)

        return df
    
class GroupFairnessAnalyzer:
    def __init__(self,
                 sensitive_features: pd.DataFrame,
                 statistical_target_feature: Optional[str] = None):
        """
        Initialize Group Fairness Analyzer

        Parameters:
        -----------
        sensitive_features : pd.DataFrame
            DataFrame containing sensitive feature columns
        statistical_target_feature : str, optional
            Feature to use for certain conditional parity metrics
        """
        self.sensitive_features = sensitive_features
        self.statistical_target_feature = statistical_target_feature

    def _get_group_masks(self, feature_name: str) -> Dict[str, np.ndarray]:
        """
        Utility method to get boolean masks for each group of a given feature.
        """
        masks = {}
        for group in self.sensitive_features[feature_name].unique():
            masks[group] = (self.sensitive_features[feature_name] == group)
        return masks

    def _get_confusion_counts(self, fairness_metrics: FairnessMetrics, group_mask: np.ndarray) -> Dict[str, int]:
        """
        Compute confusion matrix counts: TP, FP, TN, FN for a given group mask.
        """
        preds = fairness_metrics.predictions[group_mask]
        truth = fairness_metrics.ground_truth[group_mask]

        tp = np.sum((preds == 1) & (truth == 1))
        fp = np.sum((preds == 1) & (truth == 0))
        tn = np.sum((preds == 0) & (truth == 0))
        fn = np.sum((preds == 0) & (truth == 1))

        return {'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn}

    def demographic_parity(self,
                            fairness_metrics: FairnessMetrics,
                            feature_name: str) -> Dict:
        group_rates = {}
        masks = self._get_group_masks(feature_name)
        for group, mask in masks.items():
            group_rates[group] = np.mean(fairness_metrics.predictions[mask] == 1)

        rates = list(group_rates.values())
        # Extracting keys and values

        disparity = max(rates) - min(rates) if len(rates) > 1 else 0

        return {
            'feature': feature_name,
            'group_rates': group_rates
        }

    def equal_opportunity(self,
                           fairness_metrics: FairnessMetrics,
                           feature_name: str) -> Dict:
        group_tpr = {}
        masks = self._get_group_masks(feature_name)
        for group, mask in masks.items():
            group_positive_mask = (fairness_metrics.ground_truth[mask] == 1)
            if np.sum(group_positive_mask) > 0:
                group_tpr[group] = np.mean(
                    fairness_metrics.predictions[mask][group_positive_mask] == 1
                )
            else:
                group_tpr[group] = 0

        tpr_values = list(group_tpr.values())
        disparity = max(tpr_values) - min(tpr_values) if len(tpr_values) > 1 else 0

        return {
            'feature': feature_name,
            'group_rates': group_tpr
        }

    def predictive_parity(self,
                           fairness_metrics: FairnessMetrics,
                           feature_name: str) -> Dict:
        group_precision = {}
        masks = self._get_group_masks(feature_name)
        for group, mask in masks.items():
            pred_pos = fairness_metrics.predictions[mask] == 1
            if np.sum(pred_pos) > 0:
                group_precision[group] = np.mean(
                    fairness_metrics.ground_truth[mask][pred_pos] == 1
                )
            else:
                group_precision[group] = 0

        precision_values = list(group_precision.values())
        disparity = max(precision_values) - min(precision_values) if len(precision_values) > 1 else 0

        return {
            'feature': feature_name,
            'group_rates': group_precision
        }

    def js_divergence(self,
                      fairness_metrics: FairnessMetrics,
                      feature_name: str) -> Dict:

        masks = self._get_group_masks(feature_name)
        js_divergences = {}

        for group, mask in masks.items():
            preds = fairness_metrics.predictions[mask]
            truth = fairness_metrics.ground_truth[mask]

            def estimate_distribution(data, min_val=0, max_val=1):
              if len(data) < 2:
                return None
              kde = gaussian_kde(data)
              x_range = np.linspace(min_val, max_val, 100)
              return kde(x_range), x_range

            pred_dist = estimate_distribution(preds)
            truth_dist = estimate_distribution(truth)

            if pred_dist is None or truth_dist is None:
              js_divergences[group] = np.nan
              continue

            pred_pdf, x_range = pred_dist
            truth_pdf, _ = truth_dist

            # Normalize distributions
            pred_pdf = pred_pdf / np.sum(pred_pdf)
            truth_pdf = truth_pdf / np.sum(truth_pdf)

            avg_pdf = 0.5 * (pred_pdf + truth_pdf)

            eps = 1e-10
            js_div = 0.5*(
                np.sum(pred_pdf*np.log((pred_pdf+eps)/(avg_pdf+eps))) +
                np.sum(truth_pdf*np.log((truth_pdf+eps)/(avg_pdf+eps)))
            )
            js_divergences[group] = js_div

        return {
            'feature': feature_name,
            'group_rates': js_divergences,
        }

    def disparate_impact(self,
                         fairness_metrics: FairnessMetrics,
                         feature_name: str) -> Dict:
        # Disparate Impact: ratio of positive rates between groups
        masks = self._get_group_masks(feature_name)
        group_rates = {g: np.mean(fairness_metrics.predictions[m] == 1) for g, m in masks.items()}

        rates = list(group_rates.values())
        if len(rates) < 2:
            di = 1.0  # If only one group, no disparity
        else:
            max_rate = max(rates)
            min_rate = min(rates)
            if max_rate == 0:
                di = 1.0
            else:
                di = min_rate / max_rate

        return {
            'feature': feature_name,
            'group_rates': group_rates,
        }

    def equal_opportunity_difference(self,
                                     fairness_metrics: FairnessMetrics,
                                     feature_name: str) -> Dict:
        # Equal Opportunity Difference: difference in TPR between groups
        group_tpr = {}
        masks = self._get_group_masks(feature_name)
        for group, mask in masks.items():
            group_positive_mask = (fairness_metrics.ground_truth[mask] == 1)
            if np.sum(group_positive_mask) > 0:
                group_tpr[group] = np.mean(fairness_metrics.predictions[mask][group_positive_mask] == 1)
            else:
                group_tpr[group] = 0

        tpr_values = list(group_tpr.values())
        eod = max(tpr_values) - min(tpr_values) if len(tpr_values) > 1 else 0

        return {
            'feature': feature_name,
            'group_rates': group_tpr
        }

    def equalized_odds(self,
                       fairness_metrics: FairnessMetrics,
                       feature_name: str) -> Dict:
        # Equalized Odds: TPR and FPR should be equal across groups
        # We'll measure disparity in both TPR and FPR.
        masks = self._get_group_masks(feature_name)
        group_tpr = {}
        group_fpr = {}

        for group, mask in masks.items():
            counts = self._get_confusion_counts(fairness_metrics, mask)
            tp = counts['TP']
            fp = counts['FP']
            fn = counts['FN']
            tn = counts['TN']

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

            group_tpr[group] = tpr
            group_fpr[group] = fpr

        tpr_values = list(group_tpr.values())
        fpr_values = list(group_fpr.values())

        tpr_diff = max(tpr_values) - min(tpr_values) if len(tpr_values) > 1 else 0
        fpr_diff = max(fpr_values) - min(fpr_values) if len(fpr_values) > 1 else 0

        return {
            'feature': feature_name,
            'group_rates':{
                'tpr_rates': group_tpr,
                'fpr_rates': group_fpr,
            }
        }

    def statistical_parity_difference(self,
                                      fairness_metrics: FairnessMetrics,
                                      feature_name: str) -> Dict:
        # Statistical Parity Difference: difference in predicted positive rate
        # This is essentially what we computed in demographic_parity but by name.
        masks = self._get_group_masks(feature_name)
        group_rates = {g: np.mean(fairness_metrics.predictions[m] == 1) for g,m in masks.items()}

        rates = list(group_rates.values())
        sp_diff = max(rates) - min(rates) if len(rates) > 1 else 0

        return {
            'feature': feature_name,
            'group_rates': group_rates,
        }

    def treatment_equality(self,
                           fairness_metrics: FairnessMetrics,
                           feature_name: str) -> Dict:
        # Treatment Equality: equality in ratio of FPR to FNR across groups
        masks = self._get_group_masks(feature_name)
        group_te = {}

        for group, mask in masks.items():
            counts = self._get_confusion_counts(fairness_metrics, mask)
            tp, fp, tn, fn = counts['TP'], counts['FP'], counts['TN'], counts['FN']

            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

            if fnr == 0:
                te_val = np.inf if fpr > 0 else 1.0
            else:
                te_val = fpr / fnr

            group_te[group] = te_val

        te_values = [v for v in group_te.values() if np.isfinite(v)]
        if len(te_values) < 2:
            te_diff = 0
        else:
            te_diff = max(te_values) - min(te_values)

        return {
            'feature': feature_name,
            'group_rates': group_te
        }

    def calibration_by_group(self,
                             fairness_metrics: FairnessMetrics,
                             feature_name: str) -> Dict:
        # Calibration by group: predictions match actual outcomes in frequency
        # Without predicted probabilities, we approximate calibration as
        # difference between predicted positive rate and actual positive rate.
        masks = self._get_group_masks(feature_name)
        group_cal = {}

        for group, mask in masks.items():
            pred_pos_rate = np.mean(fairness_metrics.predictions[mask] == 1)
            actual_pos_rate = np.mean(fairness_metrics.ground_truth[mask] == 1)
            calibration_error = abs(pred_pos_rate - actual_pos_rate)
            group_cal[group] = {
                'predicted_positive_rate': pred_pos_rate,
                'actual_positive_rate': actual_pos_rate,
                'calibration_error': calibration_error
            }

        # Disparity can be considered as difference in calibration error across groups
        errors = [g['calibration_error'] for g in group_cal.values()]
        cal_disparity = max(errors) - min(errors) if len(errors) > 1 else 0

        return {
            'feature': feature_name,
            'group_rates': group_cal,
        }

    def conditional_parity(self,
                           fairness_metrics: FairnessMetrics,
                           feature_name: str) -> Dict:
        # Conditional parity:
        # For simplicity, if statistical_target_feature is provided, we measure
        # parity within each category of that feature, then average disparities.

        if self.statistical_target_feature is None:
            return {
                'feature': feature_name,
                'group_rates': None
            }

        target_values = self.sensitive_features[self.statistical_target_feature].unique()
        masks_feature = self._get_group_masks(feature_name)

        # For each target_value, compute rates per group and measure disparity
        disparity_rates = {}
        disparities = []
        for tv in target_values:
            tv_mask = self.sensitive_features[self.statistical_target_feature] == tv
            group_rates_tv = {}
            for group, mask in masks_feature.items():
                combined_mask = mask & tv_mask
                if np.sum(combined_mask) > 0:
                    group_rates_tv[group] = np.mean(fairness_metrics.predictions[combined_mask] == 1)
                else:
                    group_rates_tv[group] = 0
            rates = list(group_rates_tv.values())
            if len(rates) > 1:
                disparities.append(max(rates) - min(rates))
                disparity_rates[tv] = (max(rates)-min(rates))

        if len(disparities) > 0:
            avg_disparity = np.mean(disparities)
        else:
            avg_disparity = 0

        return {
            'feature': self.statistical_target_feature,
            'group_rates': disparity_rates,
        }

    def raw_outcome_disparity(self,
                              fairness_metrics: FairnessMetrics,
                              feature_name: str) -> Dict:
        # Raw Outcome Disparity: difference in actual outcome rates
        masks = self._get_group_masks(feature_name)
        group_outcomes = {g: np.mean(fairness_metrics.ground_truth[m]) for g,m in masks.items()}

        outcomes = list(group_outcomes.values())
        disparity = max(outcomes) - min(outcomes) if len(outcomes) > 1 else 0

        return {
            'feature': feature_name,
            'group_rates': group_outcomes,
        }

    def generate_fairness_report(self,
                                 fairness_metrics: FairnessMetrics,
                                 features_to_analyze: List[str]) -> pd.DataFrame:
        """
        Generate comprehensive fairness report with a single metric_outcome column.

        Returns a DataFrame with columns: feature, metric, metric_outcome
        """
        report_data = []

        # Metrics to compute
        metrics = [
            ('Demographic Parity', self.demographic_parity),
            ('Equal Opportunity', self.equal_opportunity),
            ('Predictive Parity', self.predictive_parity),
            ('JS Divergence', self.js_divergence),
            ('Disparate Impact', self.disparate_impact),
            ('Equal Opportunity Difference', self.equal_opportunity_difference),
            ('Equalized Odds', self.equalized_odds),
            ('Statistical Parity Difference', self.statistical_parity_difference),
            ('Treatment Equality', self.treatment_equality),
            ('Calibration by Group', self.calibration_by_group),
            ('Conditional Parity', self.conditional_parity),
            ('Raw Outcome Disparity', self.raw_outcome_disparity),
        ]

        # Compute metrics for each feature
        for feature in features_to_analyze:
            for metric_name, metric_func in metrics:
                try:
                    result = metric_func(fairness_metrics, feature)
                    result['metric'] = metric_name
                    report_data.append(result)
                except Exception as e:
                    print(f"Could not compute {metric_name} for {feature}: {e}")

        # Create DataFrame with consistent columns
        df = pd.DataFrame(report_data)

        return df
        
