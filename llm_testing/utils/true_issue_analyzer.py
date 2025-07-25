"""
AR True Issue and Bug Statistical Analyzer (Multi-Mode Version)

This program supports multiple working modes, using binary bits to control which data sources to use.
Binary bits from left to right represent: human_label, llm_label, prediction
Bit value 0 means not using that data source, bit value 1 means using it.

Working modes (1-7):
1 (001): Use prediction only
2 (010): Use llm_label only  
3 (011): Use llm_label + prediction
4 (100): Use human_label only
5 (101): Use human_label + prediction
6 (110): Use human_label + llm_label
7 (111): Use all three data sources (default mode)

Usage:
python true_issue_analyzer.py [mode] [human_gt_file] [llm_gt_file] [prediction_file] [report_file]

Examples:
python true_issue_analyzer.py 5 gt_llm.txt final_gt.json predictions/results_eval_model.json reports/true_issue_analysis.txt
python true_issue_analyzer.py 7 gt_llm.txt final_gt.json predictions/results_eval_model.json reports/true_issue_analysis.txt
"""

import json
import os
import sys
from collections import defaultdict


def load_human_ground_truth(filepath):
    """Load and parse human-annotated ground truth file"""
    # Metric order (corresponding to human annotation)
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
                    # Parse format: video_name:1,1,1,1,1,1
                    video_name, scores_str = line.split(":", 1)
                    scores = [int(x.strip()) for x in scores_str.split(",")]
                    
                    # Add .mp4 extension if missing
                    if not video_name.endswith(".mp4"):
                        video_name = video_name + ".mp4"
                    
                    # Validate scores length
                    if len(scores) != 6:
                        print(f"Warning: Line {line_num} has {len(scores)} scores instead of 6: {line}")
                        continue
                    
                    # Convert human annotation format to LLM format
                    # Human: 1 = no issue, 0 = has issue
                    # LLM: true = has issue, false = no issue
                    for i, score in enumerate(scores):
                        metric = metrics_order[i]
                        key = (video_name, metric)
                        # Convert: human 0(has issue) -> LLM true(has issue)
                        #         human 1(no issue) -> LLM false(no issue)
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


def load_json_file(filepath):
    """Load and parse JSON file"""
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
    """Process JSON data into structured dictionary"""
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


def parse_mode(mode_num):
    """Parse working mode, return flags for which data sources to use"""
    if not 1 <= mode_num <= 7:
        raise ValueError("Mode must be between 1 and 7")
    
    # Convert to 3-bit binary, from left to right: human_label, llm_label, prediction
    binary = format(mode_num, '03b')
    
    use_human = binary[0] == '1'
    use_llm = binary[1] == '1'
    use_prediction = binary[2] == '1'
    
    mode_description = f"Mode {mode_num} ({binary}): "
    sources = []
    if use_human:
        sources.append("human_label")
    if use_llm:
        sources.append("llm_label")
    if use_prediction:
        sources.append("prediction")
    
    mode_description += " + ".join(sources) if sources else "no data sources"
    
    return use_human, use_llm, use_prediction, mode_description


def find_true_issues_multi_mode(human_gt, llm_gt, prediction, use_human, use_llm, use_prediction):
    """Find true issues that meet the criteria according to specified mode"""
    if not any([use_human, use_llm, use_prediction]):
        return {}
    
    true_issues = {}
    
    # Get all possible (video, metric) keys
    all_keys = set()
    if use_human:
        all_keys.update(human_gt.keys())
    if use_llm:
        all_keys.update(llm_gt.keys())
    if use_prediction:
        all_keys.update(prediction.keys())
    
    for key in all_keys:
        video_name, metric = key
        
        # Check each data source according to mode
        conditions = []
        
        if use_human:
            human_issue = human_gt.get(key, False)
            conditions.append(human_issue)
            
        if use_llm:
            llm_issue = llm_gt.get(key, False)
            conditions.append(llm_issue)
            
        if use_prediction:
            pred_issue = prediction.get(key, False)
            conditions.append(pred_issue)
        
        # Only when all enabled data sources mark as issue, consider it as true issue
        if conditions and all(conditions):
            true_issues[key] = True
            
    return true_issues


def count_bugs_by_video(true_issues):
    """Count the number of true issues per video and total bug count"""
    video_true_issues = defaultdict(list)
    
    # Group true issues by video
    for (video_name, metric), is_true_issue in true_issues.items():
        if is_true_issue:
            video_true_issues[video_name].append(metric)
    
    # Count bugs (videos with at least one true issue)
    bug_count = len(video_true_issues)
    
    return video_true_issues, bug_count


def generate_detailed_analysis_multi_mode(human_gt, llm_gt, prediction, true_issues, video_true_issues, 
                                        bug_count, report_file, mode_description, 
                                        use_human, use_llm, use_prediction):
    """Generate detailed analysis report for multi-mode"""
    
    # Count true issues by metric
    metric_true_issues = defaultdict(int)
    for (video_name, metric), is_true_issue in true_issues.items():
        if is_true_issue:
            metric_true_issues[metric] += 1
    
    # Calculate total true issues
    total_true_issues = sum(metric_true_issues.values())
    
    # Count total issues for each data source (only enabled ones)
    data_source_stats = []
    if use_human:
        human_total_issues = sum(1 for issue in human_gt.values() if issue)
        data_source_stats.append(f"Human annotation total issues: {human_total_issues}")
    if use_llm:
        llm_total_issues = sum(1 for issue in llm_gt.values() if issue)
        data_source_stats.append(f"LLM GT total issues: {llm_total_issues}")
    if use_prediction:
        pred_total_issues = sum(1 for issue in prediction.values() if issue)
        data_source_stats.append(f"Prediction total issues: {pred_total_issues}")
    
    # Get all video names
    all_videos = set()
    if use_human:
        for key in human_gt.keys():
            all_videos.add(key[0])
    if use_llm:
        for key in llm_gt.keys():
            all_videos.add(key[0])
    if use_prediction:
        for key in prediction.keys():
            all_videos.add(key[0])
    
    total_videos = len(all_videos)
    clean_videos = total_videos - bug_count
    
    report_lines = [
        "=" * 60,
        "AR True Issue and Bug Statistical Analysis Report (Multi-Mode)",
        "=" * 60,
        "",
        "== Working Mode ==",
        mode_description,
        "",
        "== Data Overview ==",
    ]
    
    # Add enabled data source statistics
    report_lines.extend(data_source_stats)
    report_lines.extend([
        f"True issues total (selected data sources consistent): {total_true_issues}",
        "",
        f"Total video count: {total_videos}",
        f"Videos with bugs found: {bug_count}",
        f"Videos without bugs: {clean_videos}",
        f"Bug detection rate: {bug_count/total_videos:.2%}" if total_videos > 0 else "Bug detection rate: N/A",
        "",
        "== True Issues Statistics by Metric ==",
    ])
    
    # Statistics by metric
    metrics = [
        "Object Placement",
        "Object Movement", 
        "Occlusion",
        "Lighting",
        "Visual Artifacts and Rendering Issues",
        "Black Screen"
    ]
    
    for metric in metrics:
        count = metric_true_issues[metric]
        report_lines.append(f"{metric}: {count} true issues")
    
    report_lines.extend([
        "",
        "== Details of Videos with Bugs ==",
    ])
    
    if video_true_issues:
        for video_name, issues in sorted(video_true_issues.items()):
            report_lines.append(f"Video: {video_name}")
            report_lines.append(f"  True issue count: {len(issues)}")
            report_lines.append(f"  Involved metrics: {', '.join(issues)}")
            report_lines.append("")
    else:
        report_lines.append("No true issues found")
    
    # Add data consistency analysis (only for enabled data sources)
    report_lines.extend([
        "== Data Consistency Analysis ==",
    ])
    
    if use_human and human_gt:
        human_total_issues = sum(1 for issue in human_gt.values() if issue)
        if human_total_issues > 0:
            report_lines.append(f"True issues as percentage of human annotations: {total_true_issues/human_total_issues:.2%}")
    
    if use_llm and llm_gt:
        llm_total_issues = sum(1 for issue in llm_gt.values() if issue)
        if llm_total_issues > 0:
            report_lines.append(f"True issues as percentage of LLM GT: {total_true_issues/llm_total_issues:.2%}")
    
    if use_prediction and prediction:
        pred_total_issues = sum(1 for issue in prediction.values() if issue)
        if pred_total_issues > 0:
            report_lines.append(f"True issues as percentage of predictions: {total_true_issues/pred_total_issues:.2%}")
    
    # Build description of used data sources
    sources_used = []
    if use_human:
        sources_used.append("human_gt")
    if use_llm:
        sources_used.append("LLM_gt")
    if use_prediction:
        sources_used.append("prediction")
    sources_desc = " & ".join(sources_used)
    
    report_lines.extend([
        "",
        "== Summary ==",
        f"This analysis is based on {mode_description}",
        f"Requires selected data sources ({sources_desc}) all mark as issue",
        f"Identified {total_true_issues} true issues, involving {bug_count} videos",
        f"This indicates that among {total_videos} test videos, {bug_count} videos have reliable AR quality issues"
    ])
    
    # Save report
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    
    print(f"[INFO] True issue analysis completed. Report saved to {report_file}")
    
    # Output key statistics to console
    print("\n" + "=" * 50)
    print("Key Statistical Results:")
    print("=" * 50)
    print(f"Working mode: {mode_description}")
    print(f"Total true issues: {total_true_issues}")
    print(f"Videos with bugs found: {bug_count} / {total_videos}")
    print(f"Bug detection rate: {bug_count/total_videos:.2%}" if total_videos > 0 else "Bug detection rate: N/A")
    print("=" * 50)


def print_mode_help():
    """Print mode help information"""
    print("\nWorking Mode Instructions:")
    print("Binary bits from left to right represent: human_label, llm_label, prediction")
    print("Bit value 0 means not using that data source, bit value 1 means using it")
    print("\nAvailable modes:")
    print("1 (001): Use prediction only")
    print("2 (010): Use llm_label only")  
    print("3 (011): Use llm_label + prediction")
    print("4 (100): Use human_label only")
    print("5 (101): Use human_label + prediction")
    print("6 (110): Use human_label + llm_label")
    print("7 (111): Use all three data sources (strictest mode)")
    print()


def main():
    # Check number of arguments
    if len(sys.argv) < 2:
        print("Error: Missing mode parameter")
        print_mode_help()
        print("Usage: python true_issue_analyzer.py [mode] [human_gt_file] [llm_gt_file] [prediction_file] [report_file]")
        return
    
    # Parse mode parameter
    try:
        mode = int(sys.argv[1])
        use_human, use_llm, use_prediction, mode_description = parse_mode(mode)
    except (ValueError, IndexError):
        print("Error: Invalid mode parameter")
        print_mode_help()
        return
    
    # Default file paths (can be overridden by command line arguments)
    human_gt_file = "gt_llm_sample.txt"
    llm_gt_file = "final_gt_sample.json" 
    prediction_file = "predictions/results_eval_model.json"
    report_file = f"reports/true_issue_analysis_mode_{mode}.txt"
    
    # Check other command line arguments
    if len(sys.argv) > 2:
        human_gt_file = sys.argv[2]
    if len(sys.argv) > 3:
        llm_gt_file = sys.argv[3]
    if len(sys.argv) > 4:
        prediction_file = sys.argv[4]
    if len(sys.argv) > 5:
        report_file = sys.argv[5]
    
    print(f"[INFO] Using {mode_description}")
    
    # Load corresponding data sources according to mode
    human_gt = {}
    llm_gt = {}
    prediction = {}
    
    if use_human:
        print(f"[INFO] Loading human Ground Truth: {human_gt_file}")
        human_gt = load_human_ground_truth(human_gt_file)
        if not human_gt:
            print("[ERROR] No valid human ground truth data loaded. Please check file format.")
            return
        print(f"[INFO] Human GT entries: {len(human_gt)}")
    
    if use_llm:
        print(f"[INFO] Loading LLM Ground Truth: {llm_gt_file}")
        llm_gt_data = load_json_file(llm_gt_file)
        llm_gt = process_json_data(llm_gt_data)
        if not llm_gt:
            print("[ERROR] No valid LLM ground truth data loaded. Please check file format.")
            return
        print(f"[INFO] LLM GT entries: {len(llm_gt)}")
    
    if use_prediction:
        print(f"[INFO] Loading prediction results: {prediction_file}")
        prediction_data = load_json_file(prediction_file)
        prediction = process_json_data(prediction_data)
        if not prediction:
            print("[ERROR] No valid prediction data loaded. Please check file format.")
            return
        print(f"[INFO] Prediction entries: {len(prediction)}")
    
    # Find true issues
    print(f"[INFO] Analyzing true issues that meet mode {mode} criteria...")
    true_issues = find_true_issues_multi_mode(human_gt, llm_gt, prediction, 
                                            use_human, use_llm, use_prediction)
    
    # Count bugs
    video_true_issues, bug_count = count_bugs_by_video(true_issues)
    
    # Generate detailed analysis report
    generate_detailed_analysis_multi_mode(human_gt, llm_gt, prediction, true_issues, 
                                        video_true_issues, bug_count, report_file, 
                                        mode_description, use_human, use_llm, use_prediction)


if __name__ == "__main__":
    main()