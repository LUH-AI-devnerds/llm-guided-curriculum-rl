#!/usr/bin/env python3
"""
Comprehensive analysis script for no-curriculum (baseline) learning runs.
Extracts statistics for research paper including performance metrics,
action distributions, and comparative analysis across multiple runs.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings

warnings.filterwarnings("ignore")


class NoCurriculumAnalyzer:
    def __init__(self, logs_dir="logs"):
        self.logs_dir = Path(logs_dir)
        self.runs_data = {}
        self.results = {}
        self.output_dir = Path("no_curriculum_analysis")
        self.output_dir.mkdir(exist_ok=True)

    def find_no_curriculum_runs(self):
        """Find all no-curriculum run directories."""
        pattern_dirs = []

        # Look for standard no-curriculum runs
        for item in self.logs_dir.iterdir():
            if (
                item.is_dir()
                and "standard" in item.name
                and "no-curriculum" in item.name
            ):
                pattern_dirs.append(item)

        print(f"Found {len(pattern_dirs)} no-curriculum runs:")
        for d in sorted(pattern_dirs):
            print(f"  - {d.name}")

        return sorted(pattern_dirs)

    def load_all_runs(self):
        """Load data from all no-curriculum runs."""
        print("Loading no-curriculum baseline runs...")

        run_dirs = self.find_no_curriculum_runs()

        for run_dir in run_dirs:
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
        """Load data from a single no-curriculum run."""
        run_data = {
            "name": run_dir.name,
            "path": str(run_dir),
            "agents": {},
            "metadata": {},
        }

        # Extract metadata from directory name
        parts = run_dir.name.split("-")
        if len(parts) >= 4:
            run_data["metadata"]["date"] = parts[1]
            run_data["metadata"]["deck_type"] = parts[3]
            run_data["metadata"]["penetration"] = parts[4] if len(parts) > 4 else "0.9"

        # Load evaluation data
        eval_dir = run_dir / "evaluation"
        if not eval_dir.exists():
            print(f"    No evaluation directory found in {run_dir}")
            return None

        # Load agent evaluations
        eval_files = list(eval_dir.glob("evaluation_log_agent_*.json"))
        if not eval_files:
            print(f"    No evaluation files found in {eval_dir}")
            return None

        for eval_file in eval_files:
            try:
                with open(eval_file, "r") as f:
                    eval_data = json.load(f)

                agent_id = eval_data.get("agent_id", "unknown")
                agent_type = eval_data.get("agent_type", "unknown")

                if agent_id not in run_data["agents"]:
                    run_data["agents"][agent_id] = {
                        "agent_type": agent_type,
                        "evaluations": [],
                    }

                run_data["agents"][agent_id]["evaluations"].append(eval_data)

            except Exception as e:
                print(f"    Error loading {eval_file}: {e}")

        # Sort evaluations by timestamp for each agent
        for agent_id in run_data["agents"]:
            evals = run_data["agents"][agent_id]["evaluations"]
            evals.sort(key=lambda x: x.get("timestamp", ""))

        return run_data if run_data["agents"] else None

    def analyze_performance_metrics(self):
        """Analyze key performance metrics across all runs."""
        print("\n" + "=" * 50)
        print("PERFORMANCE METRICS ANALYSIS")
        print("=" * 50)

        metrics = {
            "win_rates": [],
            "avg_rewards": [],
            "agent_types": [],
            "run_names": [],
            "evaluation_episodes": [],
            "timestamps": [],
        }

        # Collect metrics from all runs
        for run_name, run_data in self.runs_data.items():
            for agent_id, agent_data in run_data["agents"].items():
                agent_type = agent_data["agent_type"]

                # Use the latest evaluation for each agent
                if agent_data["evaluations"]:
                    latest_eval = agent_data["evaluations"][-1]
                    summary = latest_eval.get("summary", {})

                    metrics["win_rates"].append(summary.get("win_rate", 0.0))
                    metrics["avg_rewards"].append(summary.get("avg_reward", 0.0))
                    metrics["agent_types"].append(agent_type)
                    metrics["run_names"].append(run_name)
                    metrics["evaluation_episodes"].append(
                        latest_eval.get("evaluation_episodes", 0)
                    )
                    metrics["timestamps"].append(latest_eval.get("timestamp", ""))

        # Convert to DataFrame for analysis
        df = pd.DataFrame(metrics)

        # Summary statistics
        print(f"\nDataset Overview:")
        print(f"  Total evaluations: {len(df)}")
        print(f"  Agent types: {df['agent_types'].value_counts().to_dict()}")
        print(f"  Runs analyzed: {df['run_names'].nunique()}")

        # Performance by agent type
        print(f"\nPerformance by Agent Type:")
        perf_stats = (
            df.groupby("agent_types")
            .agg(
                {
                    "win_rates": ["mean", "std", "min", "max", "count"],
                    "avg_rewards": ["mean", "std", "min", "max"],
                }
            )
            .round(4)
        )

        print(perf_stats)

        # Overall statistics
        print(f"\nOverall Performance:")
        print(f"  Win Rate: {df['win_rates'].mean():.4f} ± {df['win_rates'].std():.4f}")
        print(
            f"  Avg Reward: {df['avg_rewards'].mean():.4f} ± {df['avg_rewards'].std():.4f}"
        )
        print(
            f"  Win Rate Range: [{df['win_rates'].min():.4f}, {df['win_rates'].max():.4f}]"
        )
        print(
            f"  Avg Reward Range: [{df['avg_rewards'].min():.4f}, {df['avg_rewards'].max():.4f}]"
        )

        self.results["performance_metrics"] = {
            "dataframe": df,
            "summary_stats": perf_stats,
            "overall_stats": {
                "mean_win_rate": df["win_rates"].mean(),
                "std_win_rate": df["win_rates"].std(),
                "mean_avg_reward": df["avg_rewards"].mean(),
                "std_avg_reward": df["avg_rewards"].std(),
                "min_win_rate": df["win_rates"].min(),
                "max_win_rate": df["win_rates"].max(),
                "min_avg_reward": df["avg_rewards"].min(),
                "max_avg_reward": df["avg_rewards"].max(),
            },
        }

        return df

    def analyze_action_distributions(self):
        """Analyze action usage patterns across all runs."""
        print("\n" + "=" * 50)
        print("ACTION DISTRIBUTION ANALYSIS")
        print("=" * 50)

        action_names = {
            0: "Stand",
            1: "Hit",
            2: "Double",
            3: "Split",
            4: "Surrender",
            5: "Insurance",
        }

        all_action_data = []

        # Collect action performance data
        for run_name, run_data in self.runs_data.items():
            for agent_id, agent_data in run_data["agents"].items():
                if agent_data["evaluations"]:
                    latest_eval = agent_data["evaluations"][-1]
                    action_perf = latest_eval.get("summary", {}).get(
                        "action_performance", {}
                    )

                    for action_str, perf in action_perf.items():
                        action_id = int(action_str)
                        all_action_data.append(
                            {
                                "run_name": run_name,
                                "agent_id": agent_id,
                                "agent_type": agent_data["agent_type"],
                                "action_id": action_id,
                                "action_name": action_names.get(
                                    action_id, f"Action_{action_id}"
                                ),
                                "count": perf.get("count", 0),
                                "avg_reward": perf.get("avg_reward", 0.0),
                                "std_reward": perf.get("std_reward", 0.0),
                            }
                        )

        action_df = pd.DataFrame(all_action_data)

        if action_df.empty:
            print("No action data found!")
            return None

        # Action usage statistics
        print(f"\nAction Usage Summary:")
        action_usage = (
            action_df.groupby("action_name")
            .agg({"count": ["sum", "mean", "std"], "avg_reward": ["mean", "std"]})
            .round(4)
        )

        print(action_usage)

        # Action usage by agent type
        print(f"\nAction Usage by Agent Type:")
        agent_action_usage = (
            action_df.groupby(["agent_type", "action_name"])["count"]
            .sum()
            .unstack(fill_value=0)
        )

        # Calculate percentages
        agent_action_pct = (
            agent_action_usage.div(agent_action_usage.sum(axis=1), axis=0) * 100
        )
        print(agent_action_pct.round(2))

        self.results["action_distributions"] = {
            "dataframe": action_df,
            "usage_stats": action_usage,
            "agent_usage": agent_action_usage,
            "agent_usage_pct": agent_action_pct,
        }

        return action_df

    def analyze_learning_progression(self):
        """Analyze learning progression over time for runs with multiple evaluations."""
        print("\n" + "=" * 50)
        print("LEARNING PROGRESSION ANALYSIS")
        print("=" * 50)

        progression_data = []

        for run_name, run_data in self.runs_data.items():
            for agent_id, agent_data in run_data["agents"].items():
                evaluations = agent_data["evaluations"]

                if len(evaluations) > 1:  # Only analyze if multiple evaluations
                    for i, eval_data in enumerate(evaluations):
                        summary = eval_data.get("summary", {})
                        progression_data.append(
                            {
                                "run_name": run_name,
                                "agent_id": agent_id,
                                "agent_type": agent_data["agent_type"],
                                "evaluation_index": i,
                                "timestamp": eval_data.get("timestamp", ""),
                                "win_rate": summary.get("win_rate", 0.0),
                                "avg_reward": summary.get("avg_reward", 0.0),
                            }
                        )

        if not progression_data:
            print("No multi-evaluation runs found for progression analysis.")
            return None

        prog_df = pd.DataFrame(progression_data)

        print(f"\nLearning Progression Summary:")
        print(f"  Runs with multiple evaluations: {prog_df['run_name'].nunique()}")
        print(f"  Total evaluation points: {len(prog_df)}")

        # Analyze improvement trends
        improvement_stats = []
        for (run_name, agent_id), group in prog_df.groupby(["run_name", "agent_id"]):
            if len(group) >= 2:
                first_wr = group.iloc[0]["win_rate"]
                last_wr = group.iloc[-1]["win_rate"]
                first_ar = group.iloc[0]["avg_reward"]
                last_ar = group.iloc[-1]["avg_reward"]

                improvement_stats.append(
                    {
                        "run_name": run_name,
                        "agent_id": agent_id,
                        "agent_type": group.iloc[0]["agent_type"],
                        "win_rate_improvement": last_wr - first_wr,
                        "avg_reward_improvement": last_ar - first_ar,
                        "evaluations_count": len(group),
                    }
                )

        if improvement_stats:
            imp_df = pd.DataFrame(improvement_stats)
            print(f"\nImprovement Statistics:")
            print(
                f"  Mean Win Rate Improvement: {imp_df['win_rate_improvement'].mean():.4f}"
            )
            print(
                f"  Mean Avg Reward Improvement: {imp_df['avg_reward_improvement'].mean():.4f}"
            )
            print(
                f"  Agents that improved (win rate): {(imp_df['win_rate_improvement'] > 0).sum()}/{len(imp_df)}"
            )

        self.results["learning_progression"] = {
            "progression_df": prog_df,
            "improvement_stats": imp_df if improvement_stats else None,
        }

        return prog_df

    def generate_visualizations(self):
        """Generate comprehensive visualizations."""
        print("\n" + "=" * 50)
        print("GENERATING VISUALIZATIONS")
        print("=" * 50)

        # Set style
        plt.style.use("default")
        sns.set_palette("husl")

        # 1. Performance comparison by agent type
        if "performance_metrics" in self.results:
            self._plot_performance_comparison()

        # 2. Action distribution analysis
        if "action_distributions" in self.results:
            self._plot_action_distributions()

        # 3. Learning progression (if available)
        if (
            "learning_progression" in self.results
            and self.results["learning_progression"]["progression_df"] is not None
        ):
            self._plot_learning_progression()

        print(f"\nVisualizations saved to: {self.output_dir}/")

    def _plot_performance_comparison(self):
        """Plot performance comparison between agent types."""
        df = self.results["performance_metrics"]["dataframe"]

        fig, axes = plt.subplots(1, 2, figsize=(15, 6))

        # Win Rate comparison
        sns.boxplot(data=df, x="agent_types", y="win_rates", ax=axes[0])
        axes[0].set_title("Win Rate Distribution by Agent Type")
        axes[0].set_xlabel("Agent Type")
        axes[0].set_ylabel("Win Rate")
        axes[0].tick_params(axis="x", rotation=45)

        # Average Reward comparison
        sns.boxplot(data=df, x="agent_types", y="avg_rewards", ax=axes[1])
        axes[1].set_title("Average Reward Distribution by Agent Type")
        axes[1].set_xlabel("Agent Type")
        axes[1].set_ylabel("Average Reward")
        axes[1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "performance_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # Performance summary table
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis("tight")
        ax.axis("off")

        summary_stats = self.results["performance_metrics"]["summary_stats"]
        table_data = []

        for agent_type in summary_stats.index:
            row = [
                agent_type,
                f"{summary_stats.loc[agent_type, ('win_rates', 'mean')]:.4f}",
                f"{summary_stats.loc[agent_type, ('win_rates', 'std')]:.4f}",
                f"{summary_stats.loc[agent_type, ('avg_rewards', 'mean')]:.4f}",
                f"{summary_stats.loc[agent_type, ('avg_rewards', 'std')]:.4f}",
                f"{int(summary_stats.loc[agent_type, ('win_rates', 'count')])}",
            ]
            table_data.append(row)

        headers = [
            "Agent Type",
            "Win Rate (Mean)",
            "Win Rate (Std)",
            "Avg Reward (Mean)",
            "Avg Reward (Std)",
            "Count",
        ]
        table = ax.table(
            cellText=table_data, colLabels=headers, cellLoc="center", loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        ax.set_title(
            "No-Curriculum Performance Summary", pad=20, fontsize=14, fontweight="bold"
        )
        plt.savefig(
            self.output_dir / "performance_summary_table.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _plot_action_distributions(self):
        """Plot action usage distributions."""
        action_df = self.results["action_distributions"]["dataframe"]
        agent_usage_pct = self.results["action_distributions"]["agent_usage_pct"]

        # Action usage by agent type (stacked bar)
        fig, ax = plt.subplots(figsize=(12, 8))
        agent_usage_pct.plot(kind="bar", stacked=True, ax=ax)
        ax.set_title("Action Usage Distribution by Agent Type (%)")
        ax.set_xlabel("Agent Type")
        ax.set_ylabel("Percentage of Actions")
        ax.legend(title="Actions", bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "action_distributions.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # Action performance (reward) by action type
        fig, ax = plt.subplots(figsize=(12, 6))

        # Filter out actions with zero counts for cleaner visualization
        filtered_df = action_df[action_df["count"] > 0]

        if not filtered_df.empty:
            sns.boxplot(data=filtered_df, x="action_name", y="avg_reward", ax=ax)
            ax.set_title("Action Performance (Average Reward) by Action Type")
            ax.set_xlabel("Action Type")
            ax.set_ylabel("Average Reward")
            ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "action_performance.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _plot_learning_progression(self):
        """Plot learning progression over time."""
        prog_df = self.results["learning_progression"]["progression_df"]

        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Win rate progression
        for agent_type in prog_df["agent_type"].unique():
            type_data = prog_df[prog_df["agent_type"] == agent_type]
            for (run_name, agent_id), group in type_data.groupby(
                ["run_name", "agent_id"]
            ):
                axes[0].plot(
                    group["evaluation_index"],
                    group["win_rate"],
                    marker="o",
                    alpha=0.7,
                    label=f"{agent_type} (Run: {run_name[-10:]})",
                )

        axes[0].set_title("Win Rate Progression Over Evaluations")
        axes[0].set_xlabel("Evaluation Index")
        axes[0].set_ylabel("Win Rate")
        axes[0].grid(True, alpha=0.3)

        # Average reward progression
        for agent_type in prog_df["agent_type"].unique():
            type_data = prog_df[prog_df["agent_type"] == agent_type]
            for (run_name, agent_id), group in type_data.groupby(
                ["run_name", "agent_id"]
            ):
                axes[1].plot(
                    group["evaluation_index"],
                    group["avg_reward"],
                    marker="o",
                    alpha=0.7,
                    label=f"{agent_type} (Run: {run_name[-10:]})",
                )

        axes[1].set_title("Average Reward Progression Over Evaluations")
        axes[1].set_xlabel("Evaluation Index")
        axes[1].set_ylabel("Average Reward")
        axes[1].grid(True, alpha=0.3)

        # Add legends (limit to avoid clutter)
        if len(prog_df.groupby(["run_name", "agent_id"])) <= 10:
            axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
            axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "learning_progression.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def generate_report(self):
        """Generate comprehensive analysis report."""
        print("\n" + "=" * 50)
        print("GENERATING ANALYSIS REPORT")
        print("=" * 50)

        report_file = self.output_dir / "no_curriculum_analysis_report.md"

        with open(report_file, "w") as f:
            f.write("# No-Curriculum Baseline Analysis Report\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Dataset overview
            f.write("## Dataset Overview\n\n")
            f.write(f"- **Total runs analyzed**: {len(self.runs_data)}\n")

            if "performance_metrics" in self.results:
                df = self.results["performance_metrics"]["dataframe"]
                f.write(f"- **Total evaluations**: {len(df)}\n")
                f.write(f"- **Agent types**: {', '.join(df['agent_types'].unique())}\n")
                f.write(f"- **Unique runs**: {df['run_names'].nunique()}\n\n")

            # Performance metrics
            if "performance_metrics" in self.results:
                f.write("## Performance Metrics\n\n")
                stats = self.results["performance_metrics"]["overall_stats"]
                f.write(f"### Overall Performance\n")
                f.write(
                    f"- **Mean Win Rate**: {stats['mean_win_rate']:.4f} ± {stats['std_win_rate']:.4f}\n"
                )
                f.write(
                    f"- **Mean Average Reward**: {stats['mean_avg_reward']:.4f} ± {stats['std_avg_reward']:.4f}\n"
                )
                f.write(
                    f"- **Win Rate Range**: [{stats['min_win_rate']:.4f}, {stats['max_win_rate']:.4f}]\n"
                )
                f.write(
                    f"- **Avg Reward Range**: [{stats['min_avg_reward']:.4f}, {stats['max_avg_reward']:.4f}]\n\n"
                )

                f.write("### Performance by Agent Type\n")
                summary_stats = self.results["performance_metrics"]["summary_stats"]
                f.write(
                    "| Agent Type | Win Rate (Mean ± Std) | Avg Reward (Mean ± Std) | Count |\n"
                )
                f.write(
                    "|------------|----------------------|-------------------------|-------|\n"
                )

                for agent_type in summary_stats.index:
                    wr_mean = summary_stats.loc[agent_type, ("win_rates", "mean")]
                    wr_std = summary_stats.loc[agent_type, ("win_rates", "std")]
                    ar_mean = summary_stats.loc[agent_type, ("avg_rewards", "mean")]
                    ar_std = summary_stats.loc[agent_type, ("avg_rewards", "std")]
                    count = int(summary_stats.loc[agent_type, ("win_rates", "count")])

                    f.write(
                        f"| {agent_type} | {wr_mean:.4f} ± {wr_std:.4f} | {ar_mean:.4f} ± {ar_std:.4f} | {count} |\n"
                    )
                f.write("\n")

            # Action distributions
            if "action_distributions" in self.results:
                f.write("## Action Usage Analysis\n\n")
                agent_usage_pct = self.results["action_distributions"][
                    "agent_usage_pct"
                ]

                f.write("### Action Usage by Agent Type (%)\n")
                f.write(
                    "| Agent Type | Stand | Hit | Double | Split | Surrender | Insurance |\n"
                )
                f.write(
                    "|------------|-------|-----|--------|-------|-----------|----------|\n"
                )

                for agent_type in agent_usage_pct.index:
                    row = f"| {agent_type} |"
                    for action in [
                        "Stand",
                        "Hit",
                        "Double",
                        "Split",
                        "Surrender",
                        "Insurance",
                    ]:
                        pct = (
                            agent_usage_pct.loc[agent_type, action]
                            if action in agent_usage_pct.columns
                            else 0.0
                        )
                        row += f" {pct:.2f}% |"
                    f.write(row + "\n")
                f.write("\n")

            # Learning progression
            if (
                "learning_progression" in self.results
                and self.results["learning_progression"]["improvement_stats"]
                is not None
            ):
                f.write("## Learning Progression\n\n")
                imp_stats = self.results["learning_progression"]["improvement_stats"]

                f.write(f"- **Runs with multiple evaluations**: {len(imp_stats)}\n")
                f.write(
                    f"- **Mean Win Rate Improvement**: {imp_stats['win_rate_improvement'].mean():.4f}\n"
                )
                f.write(
                    f"- **Mean Avg Reward Improvement**: {imp_stats['avg_reward_improvement'].mean():.4f}\n"
                )
                f.write(
                    f"- **Agents that improved (win rate)**: {(imp_stats['win_rate_improvement'] > 0).sum()}/{len(imp_stats)}\n\n"
                )

            # Files generated
            f.write("## Generated Files\n\n")
            f.write(
                "- `performance_comparison.png` - Performance comparison by agent type\n"
            )
            f.write("- `performance_summary_table.png` - Summary statistics table\n")
            f.write("- `action_distributions.png` - Action usage distributions\n")
            f.write("- `action_performance.png` - Action performance analysis\n")
            if (
                "learning_progression" in self.results
                and self.results["learning_progression"]["progression_df"] is not None
            ):
                f.write(
                    "- `learning_progression.png` - Learning progression over time\n"
                )
            f.write("- `no_curriculum_analysis_report.md` - This report\n")
            f.write("- `analysis_results.json` - Raw analysis data\n\n")

        print(f"Report saved to: {report_file}")

        # Save raw results as JSON
        results_file = self.output_dir / "analysis_results.json"

        # Convert DataFrames to dictionaries for JSON serialization
        json_results = {}
        for key, value in self.results.items():
            json_results[key] = {}
            for subkey, subvalue in value.items():
                if isinstance(subvalue, pd.DataFrame):
                    # Handle multi-level columns by flattening them
                    df_copy = subvalue.copy()
                    if isinstance(df_copy.columns, pd.MultiIndex):
                        df_copy.columns = [
                            "_".join(map(str, col)).strip()
                            for col in df_copy.columns.values
                        ]
                    json_results[key][subkey] = df_copy.to_dict("records")
                elif isinstance(subvalue, pd.Series):
                    # Handle multi-level index by flattening
                    series_copy = subvalue.copy()
                    if isinstance(series_copy.index, pd.MultiIndex):
                        series_copy.index = [
                            "_".join(map(str, idx)).strip()
                            for idx in series_copy.index.values
                        ]
                    json_results[key][subkey] = series_copy.to_dict()
                else:
                    json_results[key][subkey] = subvalue

        with open(results_file, "w") as f:
            json.dump(json_results, f, indent=2, default=str)

        print(f"Raw results saved to: {results_file}")

    def run_complete_analysis(self):
        """Run the complete analysis pipeline."""
        print("=" * 60)
        print("NO-CURRICULUM BASELINE ANALYSIS")
        print("=" * 60)

        # Load data
        self.load_all_runs()

        if not self.runs_data:
            print("No data loaded. Exiting.")
            return

        # Run analyses
        self.analyze_performance_metrics()
        self.analyze_action_distributions()
        self.analyze_learning_progression()

        # Generate outputs
        self.generate_visualizations()
        self.generate_report()

        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"Results saved to: {self.output_dir}/")
        print("\nKey findings:")

        if "performance_metrics" in self.results:
            stats = self.results["performance_metrics"]["overall_stats"]
            print(
                f"  - Overall win rate: {stats['mean_win_rate']:.4f} ± {stats['std_win_rate']:.4f}"
            )
            print(
                f"  - Overall avg reward: {stats['mean_avg_reward']:.4f} ± {stats['std_avg_reward']:.4f}"
            )

        if "action_distributions" in self.results:
            usage_stats = self.results["action_distributions"]["usage_stats"]
            print(f"  - Most used action: {usage_stats[('count', 'sum')].idxmax()}")

        print(f"  - Total runs analyzed: {len(self.runs_data)}")


def main():
    """Main function to run the analysis."""
    analyzer = NoCurriculumAnalyzer()
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
