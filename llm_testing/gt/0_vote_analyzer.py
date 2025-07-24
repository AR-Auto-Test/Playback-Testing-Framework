"""
python 0_vote_analyzer.py json/9_final_alldata --video_folder ../video_clips_6s_2 --gt_file final_gt_9.json
"""

import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple, DefaultDict, Set

class ModelAssessmentAggregator:
    def __init__(self, json_folder: str, video_folder: str):
        self.json_folder = json_folder
        self.video_folder = video_folder
        self.video_names = self._get_video_names()
        self.model_data = []
        self.model_files = []  # Store model file names
        self.aggregated_results = defaultdict(lambda: defaultdict(list))
        self.vote_statistics = defaultdict(lambda: defaultdict(int))
        self.final_ground_truth = {}  # Store final ground truth for evaluation
        
    def _get_video_names(self) -> Set[str]:
        """Get list of video names from the video folder."""
        if not os.path.exists(self.video_folder):
            raise ValueError(f"Video folder not found: {self.video_folder}")
            
        video_names = set()
        for file in os.listdir(self.video_folder):
            if file.lower().endswith(('.mp4', '.avi', '.mov')):
                video_names.add(file)
                
        if not video_names:
            raise ValueError(f"No video files found in {self.video_folder}")
            
        print(f"Found {len(video_names)} videos in {self.video_folder}")
        return video_names
        
    def load_json_files(self) -> None:
        """Load all JSON files from the specified folder."""
        json_files = [f for f in os.listdir(self.json_folder) if f.endswith('.json')]
        print(f"Found {len(json_files)} JSON files")
        
        for json_file in json_files:
            file_path = os.path.join(self.json_folder, json_file)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    self.model_data.append(data)
                    self.model_files.append(json_file)  # Store file name
                print(f"Successfully loaded {json_file}")
            except Exception as e:
                print(f"Error loading {json_file}: {str(e)}")

    def process_assessments(self) -> None:
        """Process assessments from all models based on video list."""
        invalid_videos = set()
        
        for video_name in self.video_names:
            for metric in REQUIRED_METRICS:
                self.aggregated_results[video_name][metric] = []
        
        for model_idx, model_assessments in enumerate(self.model_data):
            for assessment in model_assessments:
                video_name = assessment["Video_name"]
                
                if video_name not in self.video_names:
                    invalid_videos.add(video_name)
                    continue
                    
                metric = assessment["Metrics"]
                if metric in REQUIRED_METRICS:
                    self.aggregated_results[video_name][metric].append(assessment)
                
        if invalid_videos:
            print(f"\nWarning: Found assessments for {len(invalid_videos)} videos that don't exist in video folder:")
            for video in sorted(invalid_videos):
                print(f"- {video}")

    def get_majority_vote(self, assessments: List[Dict], video_name: str, metric: str) -> Tuple[bool, List[str]]:
        """Determine majority vote and collect reasons for matching votes."""
        if not assessments:
            print(f"Warning: No assessments found for {video_name} - {metric}")
            return False, ["No assessments available"]
            
        true_count = sum(1 for a in assessments if a["Issue"])
        false_count = len(assessments) - true_count
        
        vote_key = f"{true_count}:{false_count}"
        self.vote_statistics[metric][vote_key] += 1
        
        is_issue = true_count > false_count
        matching_reasons = [a["Reason"] for a in assessments if a["Issue"] == is_issue]
        
        if len(assessments) != len(self.model_data):
            print(f"Warning: {video_name} {metric} has {len(assessments)} assessments instead of {len(self.model_data)}")
        
        return is_issue, matching_reasons

    def generate_final_assessment(self) -> List[Dict]:
        """Generate final assessment based on majority voting."""
        final_assessments = []
        
        for video_name in sorted(self.video_names):
            for metric in sorted(REQUIRED_METRICS):
                assessments = self.aggregated_results[video_name][metric]
                is_issue, matching_reasons = self.get_majority_vote(assessments, video_name, metric)
                
                final_assessment = {
                    "Video_name": video_name,
                    "Metrics": metric,
                    "Issue": is_issue,
                    "Reason": "\n\n".join(matching_reasons) if matching_reasons else "No reason provided"
                }
                
                # Store in final_ground_truth for later evaluation
                key = (video_name, metric)
                self.final_ground_truth[key] = is_issue
                
                final_assessments.append(final_assessment)
        
        return final_assessments

    def evaluate_models(self) -> Dict:
        """Evaluate each model's accuracy against the final ground truth and record disagreements."""
        model_evaluations = {}

        for model_idx, (model_data, model_file) in enumerate(zip(self.model_data, self.model_files)):
            correct_predictions = defaultdict(int)
            total_predictions = defaultdict(int)
            model_name = f"Model_{model_idx + 1}_{model_file}"

            # 新增: 记录与ground truth不符的条目
            disagreements = []

            for assessment in model_data:
                video_name = assessment["Video_name"]
                if video_name not in self.video_names:
                    continue

                metric = assessment["Metrics"]
                if metric not in REQUIRED_METRICS:
                    continue

                key = (video_name, metric)
                if key in self.final_ground_truth:
                    total_predictions[metric] += 1
                    if assessment["Issue"] == self.final_ground_truth[key]:
                        correct_predictions[metric] += 1
                    else:
                        # 记录不一致的条目
                        disagreements.append({
                            "video_name": video_name,
                            "metric": metric,
                            "model_assessment": assessment["Issue"],
                            "ground_truth": self.final_ground_truth[key]
                        })

            # 计算各指标和总体准确率
            metric_accuracies = {}
            total_correct = 0
            total_predictions_all = 0

            for metric in REQUIRED_METRICS:
                if total_predictions[metric] > 0:
                    accuracy = (correct_predictions[metric] / total_predictions[metric]) * 100
                    metric_accuracies[metric] = {
                        "accuracy": round(accuracy, 2),
                        "correct": correct_predictions[metric],
                        "total": total_predictions[metric]
                    }
                    total_correct += correct_predictions[metric]
                    total_predictions_all += total_predictions[metric]

            overall_accuracy = (total_correct / total_predictions_all * 100) if total_predictions_all > 0 else 0

            # 将不一致条目添加到评估结果中
            model_evaluations[model_name] = {
                "overall_accuracy": round(overall_accuracy, 2),
                "total_correct": total_correct,
                "total_predictions": total_predictions_all,
                "metric_accuracies": metric_accuracies,
                "disagreements": disagreements  # 新增字段
            }

        return model_evaluations

    def generate_vote_statistics_report(self) -> Dict:
        """Generate statistics report for voting distributions."""
        stats_report = {}
        vote_patterns = ["0:3", "1:2", "2:1", "3:0"]

        # 用于统计所有投票情况
        total_votes_across_all_metrics = 0
        unanimous_votes = 0  # 3:0 和 0:3 votes
        split_votes = 0      # 2:1 和 1:2 votes

        for metric in sorted(REQUIRED_METRICS):
            metric_stats = {
                pattern: self.vote_statistics[metric][pattern] 
                for pattern in vote_patterns
            }
            total_votes = sum(metric_stats.values())
            metric_stats["total"] = total_votes
            stats_report[metric] = metric_stats

            # 累计所有指标的投票情况
            total_votes_across_all_metrics += total_votes
            unanimous_votes += self.vote_statistics[metric]["3:0"] + self.vote_statistics[metric]["0:3"]
            split_votes += self.vote_statistics[metric]["2:1"] + self.vote_statistics[metric]["1:2"]

        # 计算百分比
        if total_votes_across_all_metrics > 0:
            unanimous_percentage = (unanimous_votes / total_votes_across_all_metrics) * 100
            split_percentage = (split_votes / total_votes_across_all_metrics) * 100
        else:
            unanimous_percentage = 0
            split_percentage = 0

        # 添加汇总统计到报告中
        stats_report["summary"] = {
            "total_votes": total_votes_across_all_metrics,
            "unanimous_votes": unanimous_votes,
            "unanimous_percentage": round(unanimous_percentage, 2),
            "split_votes": split_votes,
            "split_percentage": round(split_percentage, 2)
        }

        return stats_report

    def process_and_save(self, gt_file) -> None:
        """Process all data and save results."""
        print("\nLoading JSON files...")
        self.load_json_files()

        print("\nProcessing assessments...")
        self.process_assessments()

        print("\nGenerating final assessment...")
        final_assessments = self.generate_final_assessment()
        with open(gt_file, 'w') as f:
            json.dump(final_assessments, f, indent=2)
        print("Saved final ground truth to final_gt.json")

        print("\nGenerating voting statistics...")
        stats_report = self.generate_vote_statistics_report()
        stats_report["metadata"] = {
            "total_videos": len(self.video_names),
            "total_models": len(self.model_data)
        }

        with open('voting_statistics.json', 'w') as f:
            json.dump(stats_report, f, indent=2)
        print("Saved voting statistics to voting_statistics.json")

        # 打印投票一致性统计
        if "summary" in stats_report:
            print("\nVoting Agreement Summary:")
            print(f"Unanimous agreement (3:0): {stats_report['summary']['unanimous_votes']}/{stats_report['summary']['total_votes']} ({stats_report['summary']['unanimous_percentage']}%)")
            print(f"Split decisions (2:1): {stats_report['summary']['split_votes']}/{stats_report['summary']['total_votes']} ({stats_report['summary']['split_percentage']}%)")

        print("\nEvaluating model performances...")
        model_evaluations = self.evaluate_models()
        with open('model_evaluations.json', 'w') as f:
            json.dump(model_evaluations, f, indent=2)
        print("Saved model evaluations to model_evaluations.json")

        # Print summary of model performances
        print("\nModel Performance Summary:")
        for model_name, evaluation in model_evaluations.items():
            print(f"\n{model_name}:")
            print(f"Overall Accuracy: {evaluation['overall_accuracy']}%")
            print(f"Total Correct: {evaluation['total_correct']}/{evaluation['total_predictions']}")
            print("Metric-wise Accuracies:")
            for metric, metric_eval in evaluation['metric_accuracies'].items():
                print(f"  {metric}: {metric_eval['accuracy']}% ({metric_eval['correct']}/{metric_eval['total']})")

def main():
    # Constants
    global REQUIRED_METRICS
    REQUIRED_METRICS = {
        "Object Placement",
        "Object Movement",
        "Occlusion",
        "Lighting",
        "Visual Artifacts and Rendering Issues",
        "Black Screen"
    }
    
    parser = argparse.ArgumentParser(description='Aggregate model assessments and generate final ground truth')
    parser.add_argument('json_folder', help='Path to the folder containing JSON assessment files')
    parser.add_argument('--video_folder', default='../video_clips_6s_2',
                      help='Path to the folder containing video files (default: ../video_clips_6s_2)')
    parser.add_argument('--gt_file', default='final_gt.json',
                      help='Path to the json file for ground truth (default: final_gt.json)')
    args = parser.parse_args()
    
    try:
        aggregator = ModelAssessmentAggregator(args.json_folder, args.video_folder)
        aggregator.process_and_save(args.gt_file)
        print("\nProcess completed successfully!")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import argparse
    main()