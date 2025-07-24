"""
This program evaluates AR predictions by comparing them against human-annotated ground truth data.

Input files:
1. Human ground truth file: gt_llm_sample.txt (format: video_name:1,1,1,1,1,1)
2. Prediction file: prediction.json (LLM prediction results)

Output file:
evaluation_reports/report_human_gt.txt

Usage:
python AREvaluationAnalyzer_human_gt.py [human_gt_file] [prediction_file] [report_file]
python AREvaluationAnalyzer_human_gt.py gt_llm.txt predictions/results_eval_gemma_hf_nofewshot.json evaluation_reports/human/report_simplified_gemma_nofewshot.txt

Note: 
- Human annotation: 1 = no issue, 0 = has issue
- LLM prediction: true = has issue, false = no issue
"""

import json
import os
import sys
from collections import defaultdict


def load_human_ground_truth(filepath):
    """Load and parse human-annotated ground truth file.
    
    Args:
        filepath (str): Path to human ground truth file
        
    Returns:
        dict: Dictionary with (video_name, metric) as key and boolean issue as value
    """
    # Metric order in human annotation
    metrics_order = [
        "Object Placement",
        "Object Movement", 
        "Occlusion",
        "Lighting",
        "Visual Artifacts and Rendering Issues",
        "Black Screen"
    ]
    
    ground_truth = {}
    
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            for line_num, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    # Parse line format: video_name:1,1,1,1,1,1
                    video_name, scores_str = line.split(":", 1)
                    scores = [int(x.strip()) for x in scores_str.split(",")]
                    
                    # Add .mp4 extension if missing
                    if not video_name.endswith(".mp4"):
                        video_name = video_name + ".mp4"
                    
                    # Validate scores length
                    if len(scores) != 6:
                        print(f"Warning: Line {line_num} has {len(scores)} scores instead of 6: {line}")
                        continue
                    
                    # Convert human annotation to LLM format
                    # Human: 1 = no issue, 0 = has issue
                    # LLM: true = has issue, false = no issue
                    for i, score in enumerate(scores):
                        metric = metrics_order[i]
                        key = (video_name, metric)
                        # Convert: human 0 (has issue) -> LLM true (has issue)
                        #          human 1 (no issue) -> LLM false (no issue)
                        ground_truth[key] = (score == 0)
                        
                except ValueError as e:
                    print(f"Error parsing line {line_num}: {line} - {e}")
                    continue
                    
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return {}
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return {}
    
    return ground_truth


def load_llm_predictions(filepath):
    """Load and parse LLM prediction JSON file.
    
    Args:
        filepath (str): Path to prediction JSON file
        
    Returns:
        dict: Dictionary with (video_name, metric) as key and boolean issue as value
    """
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            data = json.load(file)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file {filepath}: {e}")
        return {}
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return {}
    
    predictions = {}
    
    for entry in data:
        video_name = entry.get("Video_name")
        metric = entry.get("Metrics")
        issue = entry.get("Issue")
        
        if not all([video_name, metric, isinstance(issue, bool)]):
            print(f"Warning: Invalid entry format in predictions: {entry}")
            continue
        
        key = (video_name, metric)
        predictions[key] = issue
    
    return predictions


def analyze_ground_truth_distribution(ground_truth):
    """Analyze the distribution of True/False in ground truth for each metric.
    
    Args:
        ground_truth (dict): Processed ground truth data
        
    Returns:
        dict: Distribution statistics for each metric
    """
    metric_distribution = defaultdict(lambda: {"positive": 0, "negative": 0, "total": 0})
    
    for (video, metric), issue in ground_truth.items():
        metric_distribution[metric]["total"] += 1
        if issue:
            metric_distribution[metric]["positive"] += 1
        else:
            metric_distribution[metric]["negative"] += 1
    
    return metric_distribution


def calculate_confusion_matrix(y_true, y_pred):
    """Calculate confusion matrix components manually.
    
    Args:
        y_true (list): True labels (boolean)
        y_pred (list): Predicted labels (boolean)
        
    Returns:
        dict: Dictionary with TP, FP, TN, FN counts
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    tp = fp = tn = fn = 0
    
    for true_val, pred_val in zip(y_true, y_pred):
        if true_val and pred_val:
            tp += 1
        elif not true_val and pred_val:
            fp += 1
        elif not true_val and not pred_val:
            tn += 1
        elif true_val and not pred_val:
            fn += 1
    
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn}


def calculate_metrics_from_confusion_matrix(confusion_matrix):
    """Calculate precision, recall, F1-score, and accuracy from confusion matrix.
    
    Args:
        confusion_matrix (dict): Dictionary with TP, FP, TN, FN
        
    Returns:
        dict: Dictionary with calculated metrics
    """
    tp = confusion_matrix["TP"]
    fp = confusion_matrix["FP"]
    tn = confusion_matrix["TN"]
    fn = confusion_matrix["FN"]
    
    total = tp + fp + tn + fn
    
    # Calculate metrics
    accuracy = (tp + tn) / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }


def calculate_detailed_statistics(ground_truth, predictions):
    """Calculate detailed statistical metrics for the predictions.
    
    Args:
        ground_truth (dict): Processed ground truth data
        predictions (dict): Processed prediction data
        
    Returns:
        tuple: (aspect_metrics, overall_metrics, missing_entries, gt_distribution)
    """
    aspect_metrics = defaultdict(lambda: {"y_true": [], "y_pred": [], "matches": 0, "total": 0})
    missing_entries = defaultdict(set)
    
    # Analyze ground truth distribution
    gt_distribution = analyze_ground_truth_distribution(ground_truth)
    
    # Generate statistics by comparing ground truth to predictions
    for (video, aspect), gt_issue in ground_truth.items():
        # Check if this video-aspect pair exists in predictions
        if (video, aspect) in predictions:
            pred_issue = predictions[(video, aspect)]
            
            # Add to aspect-specific metrics
            aspect_metrics[aspect]["y_true"].append(gt_issue)
            aspect_metrics[aspect]["y_pred"].append(pred_issue)
            aspect_metrics[aspect]["total"] += 1
            
            # Count matches
            if gt_issue == pred_issue:
                aspect_metrics[aspect]["matches"] += 1
        else:
            # Track missing entries in predictions
            missing_entries[video].add(aspect)
    
    # Calculate overall metrics
    overall_y_true = []
    overall_y_pred = []
    
    for aspect, metrics in aspect_metrics.items():
        overall_y_true.extend(metrics["y_true"])
        overall_y_pred.extend(metrics["y_pred"])
    
    return aspect_metrics, (overall_y_true, overall_y_pred), missing_entries, gt_distribution


def generate_detailed_report(aspect_metrics, overall_results, missing_entries, gt_distribution, 
                           pred_distribution, report_file, ground_truth_type="Human"):
    """Generate and save the detailed evaluation report.
    
    Args:
        aspect_metrics (dict): Metrics for each aspect
        overall_results (tuple): Overall true and predicted values
        missing_entries (dict): Missing entries by video
        gt_distribution (dict): Ground truth distribution for each metric
        pred_distribution (dict): Prediction distribution for each metric
        report_file (str): Path to save the report
        ground_truth_type (str): Type of ground truth (Human/LLM)
    """
    overall_y_true, overall_y_pred = overall_results
    
    report_lines = [f"== AR Evaluation Report: LLM Predictions vs {ground_truth_type} Ground Truth =="]
    
    # Ground Truth Distribution Analysis
    report_lines.append(f"\n== {ground_truth_type} Ground Truth Distribution Analysis ==")
    for metric, dist in gt_distribution.items():
        positive_rate = dist["positive"] / dist["total"] if dist["total"] > 0 else 0
        report_lines.append(f"\nMetric: {metric}")
        report_lines.append(f"  - Total samples: {dist['total']}")
        report_lines.append(f"  - Positive cases (Issue=True): {dist['positive']} ({positive_rate:.4f})")
        report_lines.append(f"  - Negative cases (Issue=False): {dist['negative']} ({1-positive_rate:.4f})")
    
    # Calculate aspect-specific metrics with detailed confusion matrix
    aspect_accuracies = {}
    
    report_lines.append("\n== Detailed Metrics Analysis ==")
    
    for aspect, data in aspect_metrics.items():
        y_true, y_pred = data["y_true"], data["y_pred"]
        matches, total = data["matches"], data["total"]
        
        if not y_true or not y_pred:
            print(f"[WARNING] Skipping {aspect} (Empty y_true or y_pred)")
            continue
        
        # Calculate confusion matrix
        confusion_matrix = calculate_confusion_matrix(y_true, y_pred)
        metrics = calculate_metrics_from_confusion_matrix(confusion_matrix)
        
        report_lines.append(f"\nAspect: {aspect}")
        report_lines.append(f"  - Total samples: {total}")
        report_lines.append(f"  - Direct matches: {matches}/{total} ({matches/total:.4f})")
        
        # Confusion Matrix
        report_lines.append(f"  - Confusion Matrix:")
        report_lines.append(f"    * True Positive (TP):  {confusion_matrix['TP']}")
        report_lines.append(f"    * False Positive (FP): {confusion_matrix['FP']}")
        report_lines.append(f"    * True Negative (TN):  {confusion_matrix['TN']}")
        report_lines.append(f"    * False Negative (FN): {confusion_matrix['FN']}")
        
        # Calculated Metrics
        report_lines.append(f"  - Calculated Metrics:")
        report_lines.append(f"    * Accuracy:  {metrics['accuracy']:.4f}")
        report_lines.append(f"    * Precision: {metrics['precision']:.4f}")
        report_lines.append(f"    * Recall:    {metrics['recall']:.4f}")
        report_lines.append(f"    * F1-Score:  {metrics['f1_score']:.4f}")
        
        # Interpretation
        if confusion_matrix['TP'] + confusion_matrix['FP'] == 0:
            report_lines.append(f"    * Note: Precision is 0 because no positive predictions were made")
        if confusion_matrix['TP'] + confusion_matrix['FN'] == 0:
            report_lines.append(f"    * Note: Recall is 0 because no positive ground truth cases exist")
        
        aspect_accuracies[aspect] = metrics['accuracy']
    
    # Find best and worst performing aspects
    if aspect_accuracies:
        best_performance = max(aspect_accuracies, key=aspect_accuracies.get)
        worst_performance = min(aspect_accuracies, key=aspect_accuracies.get)
        
        report_lines.append("\n== Best & Worst Performance ==")
        report_lines.append(f"  - Best Performance: {best_performance} (Accuracy: {aspect_accuracies[best_performance]:.4f})")
        report_lines.append(f"  - Worst Performance: {worst_performance} (Accuracy: {aspect_accuracies[worst_performance]:.4f})")
    
    # Calculate overall statistics
    if overall_y_true and overall_y_pred:
        overall_confusion = calculate_confusion_matrix(overall_y_true, overall_y_pred)
        overall_metrics = calculate_metrics_from_confusion_matrix(overall_confusion)
        
        overall_match_count = sum(1 for t, p in zip(overall_y_true, overall_y_pred) if t == p)
        overall_match_rate = overall_match_count / len(overall_y_true)
    else:
        overall_confusion = {"TP": 0, "FP": 0, "TN": 0, "FN": 0}
        overall_metrics = {"accuracy": 0, "precision": 0, "recall": 0, "f1_score": 0}
        overall_match_rate = 0
    
    report_lines.append("\n== Overall Statistics ==")
    report_lines.append(f"  - Total samples: {len(overall_y_true)}")
    report_lines.append(f"  - Direct Match Rate: {overall_match_rate:.4f}")
    
    report_lines.append(f"  - Overall Confusion Matrix:")
    report_lines.append(f"    * True Positive (TP):  {overall_confusion['TP']}")
    report_lines.append(f"    * False Positive (FP): {overall_confusion['FP']}")
    report_lines.append(f"    * True Negative (TN):  {overall_confusion['TN']}")
    report_lines.append(f"    * False Negative (FN): {overall_confusion['FN']}")
    
    report_lines.append(f"  - Overall Metrics:")
    report_lines.append(f"    * Accuracy:  {overall_metrics['accuracy']:.4f}")
    report_lines.append(f"    * Precision: {overall_metrics['precision']:.4f}")
    report_lines.append(f"    * Recall:    {overall_metrics['recall']:.4f}")
    report_lines.append(f"    * F1-Score:  {overall_metrics['f1_score']:.4f}")
    
    # Report on missing entries
    if missing_entries:
        report_lines.append(f"\n== Missing Entries in Predictions ==")
        report_lines.append(f"  - Videos with missing aspects: {len(missing_entries)}")
        
        for vid, aspects in sorted(missing_entries.items()):
            report_lines.append(f"  - Video {vid} is missing: {', '.join(sorted(aspects))}")
    
    # Data comparison summary
    gt_total = len([k for k in gt_distribution.values()])
    pred_total = len(overall_y_pred)
    
    report_lines.append(f"\n== Data Comparison Summary ==")
    report_lines.append(f"  - {ground_truth_type} Ground Truth entries: {len(overall_y_true)}")
    report_lines.append(f"  - LLM Prediction entries: {pred_total}")
    report_lines.append(f"  - Coverage Rate: {pred_total/len(overall_y_true):.4f}" if len(overall_y_true) > 0 else "  - Coverage Rate: N/A")
    
    # Save report
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print(f"[INFO] Detailed evaluation complete. Report saved to {report_file}")


def main():
    # Default file paths (can be overridden with command line arguments)
    human_gt_file = "gt_llm_sample.txt"
    prediction_file = "predictions/results_eval_model.json"
    report_file = "evaluation_reports/report_human_gt.txt"
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        human_gt_file = sys.argv[1]
    if len(sys.argv) > 2:
        prediction_file = sys.argv[2]
    if len(sys.argv) > 3:
        report_file = sys.argv[3]
    
    # Load and process data
    print(f"[INFO] Loading human ground truth from {human_gt_file}")
    ground_truth = load_human_ground_truth(human_gt_file)
    
    print(f"[INFO] Loading LLM predictions from {prediction_file}")
    predictions = load_llm_predictions(prediction_file)
    
    print(f"[INFO] Human ground truth entries: {len(ground_truth)}")
    print(f"[INFO] LLM prediction entries: {len(predictions)}")
    
    if not ground_truth:
        print("[ERROR] No valid ground truth data loaded. Please check the file format.")
        return
    
    if not predictions:
        print("[ERROR] No valid prediction data loaded. Please check the file format.")
        return
    
    # Calculate detailed statistics
    aspect_metrics, overall_results, missing_entries, gt_distribution = calculate_detailed_statistics(
        ground_truth, predictions
    )
    
    # Calculate prediction distribution for comparison
    pred_distribution = analyze_ground_truth_distribution(predictions)
    
    # Generate and save detailed report
    generate_detailed_report(aspect_metrics, overall_results, missing_entries, gt_distribution, 
                           pred_distribution, report_file, ground_truth_type="Human")


if __name__ == "__main__":
    main()