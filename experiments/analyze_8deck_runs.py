#!/usr/bin/env python3
"""
Comprehensive analysis script for 8-deck curriculum learning runs.
Extracts statistics for research paper including stage accuracies, completion rates,
and comparative metrics across multiple runs.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


class EightDeckAnalyzer:
    def __init__(self, logs_dir="logs/logs_version2/8deck-runs"):
        self.logs_dir = Path(logs_dir)
        self.runs_data = {}
        self.results = {}

    def load_all_runs(self):
        """Load data from all 8-deck runs."""
        print("Loading 8-deck curriculum runs...")

        run_dirs = [
            d for d in self.logs_dir.iterdir() if d.is_dir() and "8-deck" in str(d)
        ]
        print(f"Found {len(run_dirs)} runs: {[d.name for d in run_dirs]}")

        for run_dir in sorted(run_dirs):
            run_name = run_dir.name
            print(f"\nProcessing {run_name}...")

            try:
                run_data = self._load_single_run(run_dir)
                if run_data:
                    self.runs_data[run_name] = run_data
                    print(f"  ✓ Loaded {run_name}")
                else:
                    print(f"  ✗ Failed to load {run_name}")
            except Exception as e:
                print(f"  ✗ Error loading {run_name}: {e}")

        print(f"\nSuccessfully loaded {len(self.runs_data)} runs")
        return self.runs_data

    def _load_single_run(self, run_dir):
        """Load data from a single run directory."""
        run_data = {
            "run_name": run_dir.name,
            "agents": {},
            "stages": {},
            "summary": {},
            "config": {},
        }

        # Load run summary
        summary_files = list(run_dir.glob("run_summary_*.json"))
        if summary_files:
            with open(summary_files[0]) as f:
                summary = json.load(f)
                run_data["config"] = summary.get("training_config", {})

        # Load curriculum training report
        report_files = list(
            (run_dir / "reports").glob("curriculum_training_report_*.json")
        )
        if report_files:
            with open(report_files[0]) as f:
                report = json.load(f)
                run_data["summary"] = report

                # Extract curriculum stages info
                for stage in report.get("curriculum_stages", []):
                    stage_id = stage["stage_id"]
                    run_data["stages"][stage_id] = {
                        "name": stage["name"],
                        "available_actions": stage["available_actions"],
                        "success_threshold": stage["success_threshold"],
                        "difficulty": stage["difficulty"],
                    }

        # Load evaluation logs for each agent and stage
        eval_dir = run_dir / "evaluation"
        if eval_dir.exists():
            for eval_file in eval_dir.glob("evaluation_log_*.json"):
                try:
                    with open(eval_file) as f:
                        eval_data = json.load(f)

                    agent_id = eval_data["agent_id"]
                    agent_type = eval_data["agent_type"]
                    stage_id = eval_data.get("stage_id")

                    if agent_id not in run_data["agents"]:
                        run_data["agents"][agent_id] = {
                            "agent_type": agent_type,
                            "stages": {},
                        }

                    if stage_id:
                        run_data["agents"][agent_id]["stages"][stage_id] = {
                            "win_rate": eval_data["summary"]["win_rate"],
                            "avg_reward": eval_data["summary"]["avg_reward"],
                            "evaluation_episodes": eval_data["evaluation_episodes"],
                            "timestamp": eval_data["timestamp"],
                        }
                except Exception as e:
                    print(f"    Warning: Could not load {eval_file.name}: {e}")

        return run_data if run_data["agents"] else None

    def analyze_stage_progression(self):
        """Analyze stage progression across all runs."""
        print("\nAnalyzing stage progression...")

        stage_stats = {}
        agent_progression = {"dqn": [], "tabular": []}
        best_performance_per_run = {"dqn": [], "tabular": []}

        for run_name, run_data in self.runs_data.items():
            print(f"  Processing {run_name}...")

            # Track curriculum stages for this run
            run_stages = run_data.get("stages", {})
            num_curriculum_stages = len(run_stages)

            for agent_id, agent_data in run_data["agents"].items():
                agent_type = agent_data["agent_type"]
                stages_completed = list(agent_data["stages"].keys())
                max_stage = max(stages_completed) if stages_completed else 0

                # Find best performing stage for this agent in this run
                best_stage_id = None
                best_win_rate = -1
                best_avg_reward = -999

                for stage_id, stage_data in agent_data["stages"].items():
                    if stage_data["win_rate"] > best_win_rate:
                        best_win_rate = stage_data["win_rate"]
                        best_avg_reward = stage_data["avg_reward"]
                        best_stage_id = stage_id

                # Store best performance for this agent/run
                if best_stage_id is not None:
                    best_performance_per_run[agent_type].append(
                        {
                            "run": run_name,
                            "agent_id": agent_id,
                            "best_stage_id": best_stage_id,
                            "best_win_rate": best_win_rate,
                            "best_avg_reward": best_avg_reward,
                            "total_stages_in_curriculum": num_curriculum_stages,
                            "stages_completed": len(stages_completed),
                            "max_stage_reached": max_stage,
                        }
                    )

                agent_progression[agent_type].append(
                    {
                        "run": run_name,
                        "agent_id": agent_id,
                        "max_stage": max_stage,
                        "stages_completed": len(stages_completed),
                        "final_win_rate": (
                            agent_data["stages"].get(max_stage, {}).get("win_rate", 0)
                            if max_stage > 0
                            else 0
                        ),
                        "best_win_rate": best_win_rate if best_stage_id else 0,
                        "best_stage_id": best_stage_id,
                        "curriculum_stages": num_curriculum_stages,
                    }
                )

                # Collect stage-wise statistics (all stages across all runs)
                for stage_id, stage_data in agent_data["stages"].items():
                    key = f"stage_{stage_id}_{agent_type}"
                    if key not in stage_stats:
                        stage_stats[key] = {
                            "win_rates": [],
                            "avg_rewards": [],
                            "agent_type": agent_type,
                            "stage_id": stage_id,
                        }

                    stage_stats[key]["win_rates"].append(stage_data["win_rate"])
                    stage_stats[key]["avg_rewards"].append(stage_data["avg_reward"])

        self.results["stage_stats"] = stage_stats
        self.results["agent_progression"] = agent_progression
        self.results["best_performance_per_run"] = best_performance_per_run

        return stage_stats, agent_progression

    def calculate_summary_statistics(self):
        """Calculate comprehensive summary statistics."""
        print("\nCalculating summary statistics...")

        stats = {
            "runs_analyzed": len(self.runs_data),
            "total_stages": 7,  # From curriculum design
            "agent_types": ["dqn", "tabular"],
            "stage_completion_rates": {},
            "stage_accuracies": {},
            "final_performance": {},
            "best_performance_analysis": {},
            "curriculum_effectiveness": {},
        }

        # Stage completion rates
        for agent_type in ["dqn", "tabular"]:
            progression_data = self.results["agent_progression"][agent_type]

            stats["stage_completion_rates"][agent_type] = {
                "mean_stages_completed": np.mean(
                    [p["stages_completed"] for p in progression_data]
                ),
                "std_stages_completed": np.std(
                    [p["stages_completed"] for p in progression_data]
                ),
                "max_stages_reached": np.max(
                    [p["max_stage"] for p in progression_data]
                ),
                "completion_rate_all_stages": sum(
                    1 for p in progression_data if p["max_stage"] >= 7
                )
                / len(progression_data),
            }

            stats["final_performance"][agent_type] = {
                "mean_final_win_rate": np.mean(
                    [p["final_win_rate"] for p in progression_data]
                ),
                "std_final_win_rate": np.std(
                    [p["final_win_rate"] for p in progression_data]
                ),
                "min_final_win_rate": np.min(
                    [p["final_win_rate"] for p in progression_data]
                ),
                "max_final_win_rate": np.max(
                    [p["final_win_rate"] for p in progression_data]
                ),
            }

        # Best performance analysis across all runs
        for agent_type in ["dqn", "tabular"]:
            best_perf_data = self.results["best_performance_per_run"][agent_type]

            if best_perf_data:
                best_win_rates = [p["best_win_rate"] for p in best_perf_data]
                best_avg_rewards = [p["best_avg_reward"] for p in best_perf_data]
                best_stage_ids = [p["best_stage_id"] for p in best_perf_data]
                curriculum_sizes = [
                    p["total_stages_in_curriculum"] for p in best_perf_data
                ]

                stats["best_performance_analysis"][agent_type] = {
                    "mean_best_win_rate": np.mean(best_win_rates),
                    "std_best_win_rate": np.std(best_win_rates),
                    "min_best_win_rate": np.min(best_win_rates),
                    "max_best_win_rate": np.max(best_win_rates),
                    "mean_best_avg_reward": np.mean(best_avg_rewards),
                    "std_best_avg_reward": np.std(best_avg_rewards),
                    "most_common_best_stage": max(
                        set(best_stage_ids), key=best_stage_ids.count
                    ),
                    "best_stage_distribution": {
                        stage: best_stage_ids.count(stage)
                        for stage in set(best_stage_ids)
                    },
                    "mean_curriculum_size": np.mean(curriculum_sizes),
                    "curriculum_size_range": [
                        np.min(curriculum_sizes),
                        np.max(curriculum_sizes),
                    ],
                    "sample_size": len(best_perf_data),
                }

        # Stage-wise accuracy statistics
        for key, stage_data in self.results["stage_stats"].items():
            stage_id = stage_data["stage_id"]
            agent_type = stage_data["agent_type"]

            if stage_id not in stats["stage_accuracies"]:
                stats["stage_accuracies"][stage_id] = {}

            stats["stage_accuracies"][stage_id][agent_type] = {
                "mean_win_rate": np.mean(stage_data["win_rates"]),
                "std_win_rate": np.std(stage_data["win_rates"]),
                "mean_avg_reward": np.mean(stage_data["avg_rewards"]),
                "std_avg_reward": np.std(stage_data["avg_rewards"]),
                "sample_size": len(stage_data["win_rates"]),
            }

        self.results["summary_stats"] = stats
        return stats

    def generate_paper_tables(self, output_dir="paper_results"):
        """Generate LaTeX tables for research paper."""
        print(f"\nGenerating paper tables in {output_dir}/...")

        os.makedirs(output_dir, exist_ok=True)
        stats = self.results["summary_stats"]

        # Table 1: Stage Completion Statistics
        with open(f"{output_dir}/table1_stage_completion.tex", "w") as f:
            f.write("\\begin{table}[h!]\n")
            f.write("\\centering\n")
            f.write(
                "\\caption{Stage Completion Statistics for 8-Deck Curriculum Learning}\n"
            )
            f.write("\\label{tab:stage_completion_8deck}\n")
            f.write("\\begin{tabular}{lcc}\n")
            f.write("\\hline\n")
            f.write("Metric & DQN Agent & Tabular Agent \\\\\n")
            f.write("\\hline\n")

            dqn_comp = stats["stage_completion_rates"]["dqn"]
            tab_comp = stats["stage_completion_rates"]["tabular"]

            f.write(
                f"Mean Stages Completed & {dqn_comp['mean_stages_completed']:.2f} ± {dqn_comp['std_stages_completed']:.2f} & {tab_comp['mean_stages_completed']:.2f} ± {tab_comp['std_stages_completed']:.2f} \\\\\n"
            )
            f.write(
                f"Max Stage Reached & {dqn_comp['max_stages_reached']} & {tab_comp['max_stages_reached']} \\\\\n"
            )
            f.write(
                f"Full Curriculum Completion Rate & {dqn_comp['completion_rate_all_stages']:.1%} & {tab_comp['completion_rate_all_stages']:.1%} \\\\\n"
            )
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        # Table 2: Stage-wise Win Rates
        with open(f"{output_dir}/table2_stage_accuracies.tex", "w") as f:
            f.write("\\begin{table}[h!]\n")
            f.write("\\centering\n")
            f.write(
                "\\caption{Stage-wise Win Rates (Mean ± Std) for 8-Deck Curriculum}\n"
            )
            f.write("\\label{tab:stage_accuracies_8deck}\n")
            f.write("\\begin{tabular}{lccc}\n")
            f.write("\\hline\n")
            f.write("Stage & Stage Name & DQN Agent & Tabular Agent \\\\\n")
            f.write("\\hline\n")

            for stage_id in sorted(stats["stage_accuracies"].keys()):
                stage_data = stats["stage_accuracies"][stage_id]
                stage_name = self._get_stage_name(stage_id)

                dqn_data = stage_data.get(
                    "dqn", {"mean_win_rate": 0, "std_win_rate": 0}
                )
                tab_data = stage_data.get(
                    "tabular", {"mean_win_rate": 0, "std_win_rate": 0}
                )

                f.write(
                    f"{stage_id} & {stage_name} & {dqn_data['mean_win_rate']:.3f} ± {dqn_data['std_win_rate']:.3f} & {tab_data['mean_win_rate']:.3f} ± {tab_data['std_win_rate']:.3f} \\\\\n"
                )

            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        # Table 3: Best Performance Analysis
        with open(f"{output_dir}/table3_best_performance.tex", "w") as f:
            f.write("\\begin{table}[h!]\n")
            f.write("\\centering\n")
            f.write(
                "\\caption{Best Stage Performance Analysis for 8-Deck Curriculum Learning}\n"
            )
            f.write("\\label{tab:best_performance_8deck}\n")
            f.write("\\begin{tabular}{lcc}\n")
            f.write("\\hline\n")
            f.write("Metric & DQN Agent & Tabular Agent \\\\\n")
            f.write("\\hline\n")

            dqn_best = stats["best_performance_analysis"]["dqn"]
            tab_best = stats["best_performance_analysis"]["tabular"]

            f.write(
                f"Mean Best Win Rate & {dqn_best['mean_best_win_rate']:.3f} ± {dqn_best['std_best_win_rate']:.3f} & {tab_best['mean_best_win_rate']:.3f} ± {tab_best['std_best_win_rate']:.3f} \\\\\n"
            )
            f.write(
                f"Peak Win Rate & {dqn_best['max_best_win_rate']:.3f} & {tab_best['max_best_win_rate']:.3f} \\\\\n"
            )
            f.write(
                f"Most Common Best Stage & {dqn_best['most_common_best_stage']} & {tab_best['most_common_best_stage']} \\\\\n"
            )
            f.write(
                f"Mean Curriculum Size & {dqn_best['mean_curriculum_size']:.1f} & {tab_best['mean_curriculum_size']:.1f} \\\\\n"
            )
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        # Table 4: Final Performance Summary
        with open(f"{output_dir}/table4_final_performance.tex", "w") as f:
            f.write("\\begin{table}[h!]\n")
            f.write("\\centering\n")
            f.write(
                "\\caption{Final Performance Summary for 8-Deck Curriculum Learning}\n"
            )
            f.write("\\label{tab:final_performance_8deck}\n")
            f.write("\\begin{tabular}{lcc}\n")
            f.write("\\hline\n")
            f.write("Metric & DQN Agent & Tabular Agent \\\\\n")
            f.write("\\hline\n")

            dqn_final = stats["final_performance"]["dqn"]
            tab_final = stats["final_performance"]["tabular"]

            f.write(
                f"Mean Final Win Rate & {dqn_final['mean_final_win_rate']:.3f} ± {dqn_final['std_final_win_rate']:.3f} & {tab_final['mean_final_win_rate']:.3f} ± {tab_final['std_final_win_rate']:.3f} \\\\\n"
            )
            f.write(
                f"Best Final Win Rate & {dqn_final['max_final_win_rate']:.3f} & {tab_final['max_final_win_rate']:.3f} \\\\\n"
            )
            f.write(
                f"Worst Final Win Rate & {dqn_final['min_final_win_rate']:.3f} & {tab_final['min_final_win_rate']:.3f} \\\\\n"
            )
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")

        print(f"  ✓ Generated LaTeX tables in {output_dir}/")

    def _get_stage_name(self, stage_id):
        """Get shortened stage name for tables."""
        stage_names = {
            1: "Hit \\& Stand",
            2: "Double Down",
            3: "Splitting",
            4: "Basic Strategy",
            5: "Insurance",
            6: "Surrender",
            7: "All Actions",
        }
        return stage_names.get(stage_id, f"Stage {stage_id}")

    def generate_visualizations(self, output_dir="paper_results"):
        """Generate advanced scientific visualizations for the paper."""
        print(f"\nGenerating advanced visualizations in {output_dir}/...")

        # Set scientific plotting style
        plt.style.use("seaborn-v0_8-whitegrid")
        plt.rcParams.update(
            {
                "font.size": 12,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 11,
                "figure.titlesize": 16,
            }
        )

        # Color palette
        colors = {
            "dqn": "#2E86AB",  # Blue
            "tabular": "#A23B72",  # Purple
            "dqn_light": "#87CEEB",  # Light blue
            "tabular_light": "#DDA0DD",  # Light purple
        }

        # Figure 1: Comprehensive Performance Analysis
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1.1: Best Performance Distribution with Violin Plots
        ax1 = fig.add_subplot(gs[0, 0])
        dqn_best = [
            p["best_win_rate"] for p in self.results["best_performance_per_run"]["dqn"]
        ]
        tab_best = [
            p["best_win_rate"]
            for p in self.results["best_performance_per_run"]["tabular"]
        ]

        parts = ax1.violinplot(
            [dqn_best, tab_best], positions=[1, 2], showmeans=True, showmedians=True
        )
        parts["bodies"][0].set_facecolor(colors["dqn_light"])
        parts["bodies"][1].set_facecolor(colors["tabular_light"])

        ax1.scatter(
            [1] * len(dqn_best),
            dqn_best,
            alpha=0.7,
            color=colors["dqn"],
            s=30,
            label="DQN",
        )
        ax1.scatter(
            [2] * len(tab_best),
            tab_best,
            alpha=0.7,
            color=colors["tabular"],
            s=30,
            label="Tabular",
        )

        ax1.set_xticks([1, 2])
        ax1.set_xticklabels(["DQN", "Tabular"])
        ax1.set_ylabel("Best Win Rate")
        ax1.set_title("Best Performance Distribution")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 1.2: Stage Completion with Error Bars
        ax2 = fig.add_subplot(gs[0, 1])
        dqn_stages = [
            p["stages_completed"] for p in self.results["agent_progression"]["dqn"]
        ]
        tab_stages = [
            p["stages_completed"] for p in self.results["agent_progression"]["tabular"]
        ]

        # Fix: Use only 7 stages consistently
        stage_bins = np.arange(1, 9)  # 1 to 8 for histogram bins
        dqn_hist, _ = np.histogram(dqn_stages, bins=stage_bins)
        tab_hist, _ = np.histogram(tab_stages, bins=stage_bins)

        width = 0.35
        x = np.arange(1, 8)  # Display stages 1-7
        ax2.bar(
            x - width / 2, dqn_hist, width, label="DQN", color=colors["dqn"], alpha=0.8
        )
        ax2.bar(
            x + width / 2,
            tab_hist,
            width,
            label="Tabular",
            color=colors["tabular"],
            alpha=0.8,
        )

        ax2.set_xlabel("Stages Completed")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Stage Completion Distribution")
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"{i}" for i in x])
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 1.3: Best Stage Distribution
        ax3 = fig.add_subplot(gs[0, 2])
        dqn_best_stages = [
            p["best_stage_id"] for p in self.results["best_performance_per_run"]["dqn"]
        ]
        tab_best_stages = [
            p["best_stage_id"]
            for p in self.results["best_performance_per_run"]["tabular"]
        ]

        # Fix: Use only stages 1-7 consistently
        all_stages = list(range(1, 8))  # Always show stages 1-7
        dqn_stage_counts = [dqn_best_stages.count(s) for s in all_stages]
        tab_stage_counts = [tab_best_stages.count(s) for s in all_stages]

        x = np.arange(len(all_stages))
        ax3.bar(
            x - width / 2,
            dqn_stage_counts,
            width,
            label="DQN",
            color=colors["dqn"],
            alpha=0.8,
        )
        ax3.bar(
            x + width / 2,
            tab_stage_counts,
            width,
            label="Tabular",
            color=colors["tabular"],
            alpha=0.8,
        )

        ax3.set_xlabel("Best Performing Stage")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Most Common Best Stages")
        ax3.set_xticks(x)
        ax3.set_xticklabels([f"{s}" for s in all_stages])
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 2.1: Stage-wise Performance Evolution with Confidence Intervals
        ax4 = fig.add_subplot(gs[1, :])
        # Fix: Use only stages 1-7 consistently
        stages = list(range(1, 8))  # Always use stages 1-7

        dqn_means = []
        dqn_stds = []
        dqn_raw_data = []
        tab_means = []
        tab_stds = []
        tab_raw_data = []

        for stage_id in stages:
            # Get raw data for each stage
            dqn_stage_key = f"stage_{stage_id}_dqn"
            tab_stage_key = f"stage_{stage_id}_tabular"

            if dqn_stage_key in self.results["stage_stats"]:
                dqn_data = self.results["stage_stats"][dqn_stage_key]["win_rates"]
                dqn_means.append(np.mean(dqn_data))
                dqn_stds.append(np.std(dqn_data))
                dqn_raw_data.append(dqn_data)
            else:
                dqn_means.append(np.nan)
                dqn_stds.append(np.nan)
                dqn_raw_data.append([])

            if tab_stage_key in self.results["stage_stats"]:
                tab_data = self.results["stage_stats"][tab_stage_key]["win_rates"]
                tab_means.append(np.mean(tab_data))
                tab_stds.append(np.std(tab_data))
                tab_raw_data.append(tab_data)
            else:
                tab_means.append(np.nan)
                tab_stds.append(np.nan)
                tab_raw_data.append([])

        x = np.array(stages)

        # Plot confidence intervals as filled areas
        for i, stage in enumerate(stages):
            if dqn_raw_data[i]:
                y_vals = dqn_raw_data[i]
                x_vals = [stage] * len(y_vals)
                ax4.scatter(x_vals, y_vals, alpha=0.3, color=colors["dqn"], s=20)

        for i, stage in enumerate(stages):
            if tab_raw_data[i]:
                y_vals = tab_raw_data[i]
                x_vals = [stage] * len(y_vals)
                ax4.scatter(x_vals, y_vals, alpha=0.3, color=colors["tabular"], s=20)

        # Plot means with error bars
        valid_dqn = ~np.isnan(dqn_means)
        valid_tab = ~np.isnan(tab_means)

        ax4.errorbar(
            x[valid_dqn],
            np.array(dqn_means)[valid_dqn],
            yerr=np.array(dqn_stds)[valid_dqn],
            marker="o",
            linewidth=3,
            markersize=8,
            capsize=5,
            label="DQN",
            color=colors["dqn"],
        )
        ax4.errorbar(
            x[valid_tab],
            np.array(tab_means)[valid_tab],
            yerr=np.array(tab_stds)[valid_tab],
            marker="s",
            linewidth=3,
            markersize=8,
            capsize=5,
            label="Tabular",
            color=colors["tabular"],
        )

        ax4.set_xlabel("Curriculum Stage")
        ax4.set_ylabel("Win Rate")
        ax4.set_title("Stage-wise Performance Evolution with Individual Data Points")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.set_xticks(stages)

        # 3.1: Performance vs Curriculum Size
        ax5 = fig.add_subplot(gs[2, 0])
        dqn_curriculum_sizes = [
            p["total_stages_in_curriculum"]
            for p in self.results["best_performance_per_run"]["dqn"]
        ]
        tab_curriculum_sizes = [
            p["total_stages_in_curriculum"]
            for p in self.results["best_performance_per_run"]["tabular"]
        ]

        ax5.scatter(
            dqn_curriculum_sizes,
            dqn_best,
            alpha=0.7,
            color=colors["dqn"],
            s=60,
            label="DQN",
        )
        ax5.scatter(
            tab_curriculum_sizes,
            tab_best,
            alpha=0.7,
            color=colors["tabular"],
            s=60,
            label="Tabular",
        )

        # Add trend lines
        z_dqn = np.polyfit(dqn_curriculum_sizes, dqn_best, 1)
        p_dqn = np.poly1d(z_dqn)
        ax5.plot(
            sorted(dqn_curriculum_sizes),
            p_dqn(sorted(dqn_curriculum_sizes)),
            "--",
            color=colors["dqn"],
            alpha=0.8,
            linewidth=2,
        )

        z_tab = np.polyfit(tab_curriculum_sizes, tab_best, 1)
        p_tab = np.poly1d(z_tab)
        ax5.plot(
            sorted(tab_curriculum_sizes),
            p_tab(sorted(tab_curriculum_sizes)),
            "--",
            color=colors["tabular"],
            alpha=0.8,
            linewidth=2,
        )

        ax5.set_xlabel("Curriculum Size (Stages)")
        ax5.set_ylabel("Best Win Rate")
        ax5.set_title("Performance vs Curriculum Size")
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        # 3.2: Final vs Best Performance Comparison
        ax6 = fig.add_subplot(gs[2, 1])
        dqn_final = [
            p["final_win_rate"] for p in self.results["agent_progression"]["dqn"]
        ]
        tab_final = [
            p["final_win_rate"] for p in self.results["agent_progression"]["tabular"]
        ]

        # Perfect correlation line
        min_val = min(min(dqn_final + tab_final), min(dqn_best + tab_best))
        max_val = max(max(dqn_final + tab_final), max(dqn_best + tab_best))
        ax6.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, linewidth=1)

        ax6.scatter(
            dqn_final, dqn_best, alpha=0.7, color=colors["dqn"], s=60, label="DQN"
        )
        ax6.scatter(
            tab_final,
            tab_best,
            alpha=0.7,
            color=colors["tabular"],
            s=60,
            label="Tabular",
        )

        ax6.set_xlabel("Final Win Rate")
        ax6.set_ylabel("Best Win Rate")
        ax6.set_title("Final vs Best Performance")
        ax6.legend()
        ax6.grid(True, alpha=0.3)

        # 3.3: Clean Statistical Summary
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis("off")

        stats = self.results["summary_stats"]

        # Create clean, professional text summary
        summary_text = f"""PERFORMANCE SUMMARY

DQN Agent:
  Best Win Rate: {stats['best_performance_analysis']['dqn']['mean_best_win_rate']:.3f} ± {stats['best_performance_analysis']['dqn']['std_best_win_rate']:.3f}
  Peak Performance: {stats['best_performance_analysis']['dqn']['max_best_win_rate']:.3f}
  Completion Rate: {stats['stage_completion_rates']['dqn']['completion_rate_all_stages']:.0%}
  Best Stage: Stage {stats['best_performance_analysis']['dqn']['most_common_best_stage']}

Tabular Agent:
  Best Win Rate: {stats['best_performance_analysis']['tabular']['mean_best_win_rate']:.3f} ± {stats['best_performance_analysis']['tabular']['std_best_win_rate']:.3f}
  Peak Performance: {stats['best_performance_analysis']['tabular']['max_best_win_rate']:.3f}
  Completion Rate: {stats['stage_completion_rates']['tabular']['completion_rate_all_stages']:.0%}
  Best Stage: Stage {stats['best_performance_analysis']['tabular']['most_common_best_stage']}

STUDY PARAMETERS
  Runs Analyzed: {stats['runs_analyzed']}
  Curriculum Stages: 1-7
  Sample Size: {stats['best_performance_analysis']['dqn']['sample_size']} per agent"""

        ax7.text(
            0.05,
            1.0,
            summary_text,
            transform=ax7.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#F8F9FA",
                alpha=0.9,
                edgecolor="#DEE2E6",
            ),
        )

        plt.suptitle(
            "8-Deck Curriculum Learning: Comprehensive Performance Analysis",
            fontsize=16,
            y=0.98,
        )
        plt.savefig(
            f"{output_dir}/figure1_comprehensive_analysis.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Figure 2: Advanced Stage Evolution with Variance Clouds
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Top plot: Individual trajectories with variance clouds
        for i, (run_name, run_data) in enumerate(self.runs_data.items()):
            for agent_id, agent_data in run_data["agents"].items():
                agent_type = agent_data["agent_type"]
                stages = sorted(agent_data["stages"].keys())
                win_rates = [agent_data["stages"][s]["win_rate"] for s in stages]

                color = colors["dqn"] if agent_type == "dqn" else colors["tabular"]
                alpha = 0.3 if i > 0 else 0.6  # First trajectory more visible

                ax1.plot(stages, win_rates, color=color, alpha=alpha, linewidth=1)

                # Mark best performance
                best_stage = max(
                    agent_data["stages"].keys(),
                    key=lambda s: agent_data["stages"][s]["win_rate"],
                )
                best_win_rate = agent_data["stages"][best_stage]["win_rate"]
                ax1.scatter(
                    best_stage,
                    best_win_rate,
                    color=color,
                    s=50,
                    alpha=0.8,
                    marker="*" if agent_type == "dqn" else "D",
                )

        # Add mean trajectories with confidence bands
        all_stages = sorted(
            set().union(
                *[list(self.results["summary_stats"]["stage_accuracies"].keys())]
            )
        )

        for agent_type in ["dqn", "tabular"]:
            stage_means = []
            stage_stds = []
            stage_positions = []

            for stage in all_stages:
                key = f"stage_{stage}_{agent_type}"
                if key in self.results["stage_stats"]:
                    data = self.results["stage_stats"][key]["win_rates"]
                    stage_means.append(np.mean(data))
                    stage_stds.append(np.std(data))
                    stage_positions.append(stage)

            stage_positions = np.array(stage_positions)
            stage_means = np.array(stage_means)
            stage_stds = np.array(stage_stds)

            color = colors["dqn"] if agent_type == "dqn" else colors["tabular"]

            # Mean line
            ax1.plot(
                stage_positions,
                stage_means,
                color=color,
                linewidth=4,
                label=f"{agent_type.upper()} Mean",
                marker="o",
                markersize=8,
            )

            # Confidence band (±1 std)
            ax1.fill_between(
                stage_positions,
                stage_means - stage_stds,
                stage_means + stage_stds,
                color=color,
                alpha=0.2,
                label=f"{agent_type.upper()} ±1σ",
            )

        ax1.set_xlabel("Curriculum Stage")
        ax1.set_ylabel("Win Rate")
        ax1.set_title("Individual Learning Trajectories with Statistical Summaries")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Bottom plot: Box plots for each stage
        stage_data_dqn = []
        stage_data_tab = []
        stage_labels = []

        for stage in all_stages:
            dqn_key = f"stage_{stage}_dqn"
            tab_key = f"stage_{stage}_tabular"

            if dqn_key in self.results["stage_stats"]:
                stage_data_dqn.append(self.results["stage_stats"][dqn_key]["win_rates"])
            else:
                stage_data_dqn.append([])

            if tab_key in self.results["stage_stats"]:
                stage_data_tab.append(self.results["stage_stats"][tab_key]["win_rates"])
            else:
                stage_data_tab.append([])

            stage_labels.append(f"Stage {stage}")

        # Create box plots
        positions_dqn = np.arange(len(all_stages)) * 2 - 0.3
        positions_tab = np.arange(len(all_stages)) * 2 + 0.3

        bp1 = ax2.boxplot(
            stage_data_dqn,
            positions=positions_dqn,
            widths=0.5,
            patch_artist=True,
            boxprops=dict(facecolor=colors["dqn_light"]),
        )
        bp2 = ax2.boxplot(
            stage_data_tab,
            positions=positions_tab,
            widths=0.5,
            patch_artist=True,
            boxprops=dict(facecolor=colors["tabular_light"]),
        )

        ax2.set_xlabel("Curriculum Stage")
        ax2.set_ylabel("Win Rate")
        ax2.set_title("Performance Distribution by Stage")
        ax2.set_xticks(np.arange(len(all_stages)) * 2)
        ax2.set_xticklabels(stage_labels)
        ax2.grid(True, alpha=0.3)

        # Add legend
        ax2.legend([bp1["boxes"][0], bp2["boxes"][0]], ["DQN", "Tabular"])

        plt.tight_layout()
        plt.savefig(
            f"{output_dir}/figure2_advanced_stage_evolution.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print(f"  ✓ Generated advanced scientific visualizations in {output_dir}/")

    def save_raw_data(self, output_dir="paper_results"):
        """Save raw data as CSV for further analysis."""
        print(f"\nSaving raw data to {output_dir}/...")

        # Agent progression data
        progression_data = []
        for agent_type in ["dqn", "tabular"]:
            for prog in self.results["agent_progression"][agent_type]:
                progression_data.append(
                    {
                        "run": prog["run"],
                        "agent_type": agent_type,
                        "agent_id": prog["agent_id"],
                        "max_stage": prog["max_stage"],
                        "stages_completed": prog["stages_completed"],
                        "final_win_rate": prog["final_win_rate"],
                    }
                )

        df_progression = pd.DataFrame(progression_data)
        df_progression.to_csv(f"{output_dir}/agent_progression_data.csv", index=False)

        # Best performance data
        best_performance_data = []
        for agent_type in ["dqn", "tabular"]:
            for perf in self.results["best_performance_per_run"][agent_type]:
                best_performance_data.append(
                    {
                        "run": perf["run"],
                        "agent_type": agent_type,
                        "agent_id": perf["agent_id"],
                        "best_stage_id": perf["best_stage_id"],
                        "best_win_rate": perf["best_win_rate"],
                        "best_avg_reward": perf["best_avg_reward"],
                        "total_stages_in_curriculum": perf[
                            "total_stages_in_curriculum"
                        ],
                        "stages_completed": perf["stages_completed"],
                        "max_stage_reached": perf["max_stage_reached"],
                    }
                )

        df_best_performance = pd.DataFrame(best_performance_data)
        df_best_performance.to_csv(
            f"{output_dir}/best_performance_data.csv", index=False
        )

        # Stage-wise performance data
        stage_data = []
        for key, data in self.results["stage_stats"].items():
            for i, (win_rate, avg_reward) in enumerate(
                zip(data["win_rates"], data["avg_rewards"])
            ):
                stage_data.append(
                    {
                        "stage_id": data["stage_id"],
                        "agent_type": data["agent_type"],
                        "run_index": i,
                        "win_rate": win_rate,
                        "avg_reward": avg_reward,
                    }
                )

        df_stage = pd.DataFrame(stage_data)
        df_stage.to_csv(f"{output_dir}/stage_performance_data.csv", index=False)

        # Summary statistics as JSON (convert numpy types)
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(v) for v in obj]
            else:
                return convert_numpy(obj)

        stats_serializable = recursive_convert(self.results["summary_stats"])

        with open(f"{output_dir}/summary_statistics.json", "w") as f:
            json.dump(stats_serializable, f, indent=2)

        print(f"  ✓ Saved raw data to {output_dir}/")

    def print_summary_report(self):
        """Print a comprehensive summary report."""
        print("\n" + "=" * 60)
        print("8-DECK CURRICULUM LEARNING ANALYSIS REPORT")
        print("=" * 60)

        stats = self.results["summary_stats"]

        print(f"\nRuns Analyzed: {stats['runs_analyzed']}")
        print(f"Total Curriculum Stages: {stats['total_stages']}")
        print(f"Agent Types: {', '.join(stats['agent_types'])}")

        print(f"\nSTAGE COMPLETION SUMMARY:")
        print("-" * 30)
        for agent_type in ["dqn", "tabular"]:
            comp_data = stats["stage_completion_rates"][agent_type]
            print(f"\n{agent_type.upper()} Agent:")
            print(
                f"  Mean Stages Completed: {comp_data['mean_stages_completed']:.2f} ± {comp_data['std_stages_completed']:.2f}"
            )
            print(f"  Maximum Stage Reached: {comp_data['max_stages_reached']}")
            print(
                f"  Full Curriculum Completion: {comp_data['completion_rate_all_stages']:.1%}"
            )

        print(f"\nBEST PERFORMANCE ANALYSIS (Peak Stage per Run):")
        print("-" * 30)
        for agent_type in ["dqn", "tabular"]:
            best_data = stats["best_performance_analysis"][agent_type]
            print(f"\n{agent_type.upper()} Agent:")
            print(
                f"  Mean Best Win Rate: {best_data['mean_best_win_rate']:.3f} ± {best_data['std_best_win_rate']:.3f}"
            )
            print(f"  Peak Win Rate Achieved: {best_data['max_best_win_rate']:.3f}")
            print(f"  Most Common Best Stage: {best_data['most_common_best_stage']}")
            print(f"  Best Stage Distribution: {best_data['best_stage_distribution']}")
            print(
                f"  Mean Curriculum Size: {best_data['mean_curriculum_size']:.1f} stages"
            )

        print(f"\nFINAL PERFORMANCE SUMMARY:")
        print("-" * 30)
        for agent_type in ["dqn", "tabular"]:
            perf_data = stats["final_performance"][agent_type]
            print(f"\n{agent_type.upper()} Agent:")
            print(
                f"  Mean Final Win Rate: {perf_data['mean_final_win_rate']:.3f} ± {perf_data['std_final_win_rate']:.3f}"
            )
            print(f"  Best Performance: {perf_data['max_final_win_rate']:.3f}")
            print(f"  Worst Performance: {perf_data['min_final_win_rate']:.3f}")

        print(f"\nSTAGE-WISE ACCURACY SUMMARY:")
        print("-" * 30)
        for stage_id in sorted(stats["stage_accuracies"].keys()):
            stage_data = stats["stage_accuracies"][stage_id]
            print(f"\S {stage_id} ({self._get_stage_name(stage_id)}):")

            for agent_type in ["dqn", "tabular"]:
                if agent_type in stage_data:
                    data = stage_data[agent_type]
                    print(
                        f"  {agent_type.upper()}: {data['mean_win_rate']:.3f} ± {data['std_win_rate']:.3f} (n={data['sample_size']})"
                    )


def main():
    print("8-Deck Curriculum Learning Analysis")
    print("=" * 40)

    # Initialize analyzer
    analyzer = EightDeckAnalyzer()

    # Load all runs
    runs_data = analyzer.load_all_runs()
    if not runs_data:
        print("No runs found! Check the logs directory.")
        return

    # Analyze stage progression
    analyzer.analyze_stage_progression()

    # Calculate summary statistics
    analyzer.calculate_summary_statistics()

    # Generate outputs
    output_dir = "paper_results_8deck"
    analyzer.generate_paper_tables(output_dir)
    analyzer.generate_visualizations(output_dir)
    analyzer.save_raw_data(output_dir)

    # Print summary report
    analyzer.print_summary_report()

    print(f"\n✅ Analysis complete! Results saved to {output_dir}/")
    print(f"\nGenerated files:")
    print(f"  - LaTeX tables: table*.tex")
    print(f"  - Figures: figure*.png")
    print(f"  - Raw data: *.csv, summary_statistics.json")


if __name__ == "__main__":
    main()
