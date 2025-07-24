"""
This program evaluates AR predictions by comparing them against ground truth data.

Input files:
1. Ground truth file: final_gt.json
2. Prediction file: prediction.json (both files now use the same JSON format)

Output file:
evaluation_reports/report_simplified.txt

Usage:
python AREvaluationAnalyzer_no_API.py [ground_truth_file] [prediction_file] [report_file]
python AREvaluationAnalyzer_no_API.py labels/final_gt_9.json predictions/results_eval_InternVL2_5_nofewshot_20250524_083416.json evaluation_reports/report_simplified_InternVL_nofewshot.txt
"""


import json
import os
import sys
from collections import defaultdict


def load_json_file(filepath):
    """Load and parse a JSON file."""
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON file {filepath}: {e}")
        return []
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return []


def process_json_data(data):
    """Process JSON data into a structured dictionary.
    
    Args:
        data (list): List of dictionaries with Video_name, Metrics, Issue, Reason
        
    Returns:
        dict: Dictionary with Video_name and Metrics as keys, and Issue as value
    """
    processed_data = {}
    
    for entry in data:
        video_name = entry.get("Video_name")
        metric = entry.get("Metrics")
        issue = entry.get("Issue")
        
        if not all([video_name, metric, isinstance(issue, bool)]):
            print(f"Warning: Invalid entry format in data: {entry}")
            continue
        
        key = (video_name, metric)
        processed_data[key] = issue
    
    return processed_data


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


def calculate_detailed_statistics(ground_truth, predictions, expected_aspects):
    """Calculate detailed statistical metrics for the predictions.
    
    Args:
        ground_truth (dict): Processed ground truth data
        predictions (dict): Processed prediction data
        expected_aspects (set): Set of expected aspect names
        
    Returns:
        tuple: (aspect_metrics, overall_metrics, missing_entries, gt_distribution)
    """
    aspect_metrics = defaultdict(lambda: {"y_true": [], "y_pred": [], "matches": 0, "total": 0})
    video_counts = defaultdict(int)
    missing_entries = defaultdict(set)
    
    # Analyze ground truth distribution
    gt_distribution = analyze_ground_truth_distribution(ground_truth)
    
    # Collect all unique video names
    all_videos = set([key[0] for key in ground_truth.keys()])
    
    # Generate statistics by comparing ground truth to predictions
    for (video, aspect), gt_issue in ground_truth.items():
        # Track video occurrences
        video_counts[video] += 1
        
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
    
    # Check for videos missing expected aspects in ground truth
    for video in all_videos:
        for aspect in expected_aspects:
            if (video, aspect) not in ground_truth:
                missing_entries[video].add(aspect)
    
    # Calculate overall metrics
    overall_y_true = []
    overall_y_pred = []
    
    for aspect, metrics in aspect_metrics.items():
        overall_y_true.extend(metrics["y_true"])
        overall_y_pred.extend(metrics["y_pred"])
    
    return aspect_metrics, (overall_y_true, overall_y_pred), missing_entries, video_counts, gt_distribution


def generate_detailed_report(aspect_metrics, overall_results, missing_entries, video_counts, gt_distribution, report_file):
    """Generate and save the detailed evaluation report.
    
    Args:
        aspect_metrics (dict): Metrics for each aspect
        overall_results (tuple): Overall true and predicted values
        missing_entries (dict): Missing entries by video
        video_counts (dict): Count of aspects per video
        gt_distribution (dict): Ground truth distribution for each metric
        report_file (str): Path to save the report
    """
    overall_y_true, overall_y_pred = overall_results
    
    report_lines = ["== Detailed AR Evaluation Report =="]
    
    # Ground Truth Distribution Analysis
    report_lines.append("\n== Ground Truth Distribution Analysis ==")
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
    
    # Report on data integrity
    total_videos = len(video_counts)
    expected_aspects_per_video = 6  # Object Placement, Object Movement, etc.
    total_expected_entries = total_videos * expected_aspects_per_video
    total_actual_entries = sum(video_counts.values())
    
    report_lines.append(f"\n== Data Integrity Check ==")
    report_lines.append(f"  - Total Videos Processed: {total_videos}")
    report_lines.append(f"  - Expected Entries Count: {total_expected_entries}")
    report_lines.append(f"  - Actual Entries Count: {total_actual_entries}")
    report_lines.append(f"  - Completeness Rate: {total_actual_entries/total_expected_entries:.4f}")
    
    # Report on missing entries
    if missing_entries:
        report_lines.append(f"\n== Missing Entries ==")
        report_lines.append(f"  - Videos with missing aspects: {len(missing_entries)}")
        
        for vid, aspects in sorted(missing_entries.items()):
            report_lines.append(f"  - Video {vid} is missing: {', '.join(sorted(aspects))}")
    
    # Save report
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print(f"[INFO] Detailed evaluation complete. Report saved to {report_file}")


def main():
    # Define file paths (can be overridden with command line arguments)
    #ground_truth_file = "labels/final_gt_9.json"
    #prediction_file = "predictions/results_eval_InternVL2_5_fewshot_img_scaling_20176673.json"
    #report_file = "evaluation_reports/report_detailed_internvl.txt"
    ground_truth_file = ""
    prediction_file = ""
    report_file = ""
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        ground_truth_file = sys.argv[1]
    if len(sys.argv) > 2:
        prediction_file = sys.argv[2]
    if len(sys.argv) > 3:
        report_file = sys.argv[3]
    
    # Expected aspects in AR evaluation
    expected_aspects = {
        "Object Placement",
        "Object Movement",
        "Occlusion",
        "Lighting",
        "Visual Artifacts and Rendering Issues",
        "Black Screen"
    }
    
    # Load and process data
    print(f"[INFO] Loading ground truth from {ground_truth_file}")
    ground_truth_data = load_json_file(ground_truth_file)
    
    print(f"[INFO] Loading predictions from {prediction_file}")
    prediction_data = load_json_file(prediction_file)
    
    # Process data into structured dictionaries
    ground_truth = process_json_data(ground_truth_data)
    predictions = process_json_data(prediction_data)
    
    print(f"[INFO] Ground truth entries: {len(ground_truth)}")
    print(f"[INFO] Prediction entries: {len(predictions)}")
    
    # Calculate detailed statistics
    aspect_metrics, overall_results, missing_entries, video_counts, gt_distribution = calculate_detailed_statistics(
        ground_truth, predictions, expected_aspects
    )
    
    # Generate and save detailed report
    generate_detailed_report(aspect_metrics, overall_results, missing_entries, video_counts, gt_distribution, report_file)


if __name__ == "__main__":
    main()