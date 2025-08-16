#!/usr/bin/env python3
"""
Multi-seed evaluation automation script for curriculum vs no-curriculum comparison.
Runs each configuration with multiple seeds and provides statistical analysis.
"""

import os
import json
import subprocess
import time
import numpy as np
from datetime import datetime
import argparse
from pathlib import Path


class MultiSeedEvaluator:
    def __init__(
        self,
        seeds=10,
        episodes=500000,
        eval_episodes=50000,
        max_episodes_per_stage=20000,
    ):
        self.seeds = seeds
        self.episodes = episodes
        self.eval_episodes = eval_episodes
        self.max_episodes_per_stage = max_episodes_per_stage
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f"multi_seed_results_{self.timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)

    def run_single_experiment(self, deck_type, use_curriculum, seed):
        """Run a single experiment with given parameters."""
        print(
            f"Running {deck_type} {'curriculum' if use_curriculum else 'no-curriculum'} seed {seed}"
        )

        # Set random seed for reproducibility
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = str(seed)

        cmd = [
            "python",
            "scripts/curriculum_multi_agent_rl.py",
            "--episodes",
            str(self.episodes),
            "--deck-type",
            deck_type,
            "--eval-episodes",
            str(self.eval_episodes),
        ]

        if not use_curriculum:
            cmd.append("--no-curriculum")
        else:
            cmd.append("--max-episodes-per-stage")
            cmd.append(str(self.max_episodes_per_stage))

        try:
            # Run with timeout to prevent hanging
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200,  # 2 hour timeout
                env=env,
            )

            if result.returncode != 0:
                print(f"Error in seed {seed}: {result.stderr}")
                return None

            return self.extract_results_from_logs(deck_type, use_curriculum)

        except subprocess.TimeoutExpired:
            print(f"Timeout for seed {seed}")
            return None
        except Exception as e:
            print(f"Exception in seed {seed}: {e}")
            return None

    def extract_results_from_logs(self, deck_type, use_curriculum):
        """Extract results from the most recent log files."""
        try:
            # Find the most recent log directory
            log_pattern = f"logs-*-{deck_type}-0.9"
            if not use_curriculum:
                log_pattern += "-no-curriculum"

            logs_dir = Path("logs")
            matching_dirs = list(logs_dir.glob(log_pattern))

            if not matching_dirs:
                print(f"No log directory found for pattern: {log_pattern}")
                return None

            latest_dir = max(matching_dirs, key=lambda x: x.stat().st_mtime)

            # Extract evaluation results
            eval_dir = latest_dir / "evaluation"
            if not eval_dir.exists():
                return None

            eval_files = list(eval_dir.glob("evaluation_log_*.json"))
            if not eval_files:
                return None

            results = []
            for eval_file in eval_files:
                with open(eval_file, "r") as f:
                    data = json.load(f)
                    summary = data.get("summary", {})
                    results.append(
                        {
                            "agent_type": data.get("agent_type", "unknown"),
                            "win_rate": summary.get("win_rate", 0),
                            "avg_reward": summary.get("avg_reward", 0),
                            "evaluation_episodes": data.get("evaluation_episodes", 0),
                            "time_taken": summary.get("time_taken", 0),
                        }
                    )

            return results

        except Exception as e:
            print(f"Error extracting results: {e}")
            return None

    def run_configuration(self, deck_type, use_curriculum):
        """Run all seeds for a specific configuration."""
        config_name = (
            f"{deck_type}_{'curriculum' if use_curriculum else 'no_curriculum'}"
        )
        print(f"\n{'='*60}")
        print(f"Running configuration: {config_name}")
        print(f"{'='*60}")

        self.results[config_name] = []

        for seed in range(self.seeds):
            print(f"\nSeed {seed + 1}/{self.seeds}")
            result = self.run_single_experiment(deck_type, use_curriculum, seed)

            if result:
                self.results[config_name].append(
                    {
                        "seed": seed,
                        "agents": result,
                        "timestamp": datetime.now().isoformat(),
                    }
                )
                print(f"✓ Seed {seed} completed")
            else:
                print(f"✗ Seed {seed} failed")

            # Save intermediate results
            self.save_intermediate_results()

    def save_intermediate_results(self):
        """Save current results to file."""
        results_file = os.path.join(self.output_dir, "intermediate_results.json")
        with open(results_file, "w") as f:
            json.dump(self.results, f, indent=2)

    def calculate_statistics(self):
        """Calculate mean, variance, std dev, and confidence intervals."""
        stats = {}

        for config_name, runs in self.results.items():
            if not runs:
                continue

            # Group by agent type
            agent_stats = {}

            for run in runs:
                for agent in run["agents"]:
                    agent_type = agent["agent_type"]
                    if agent_type not in agent_stats:
                        agent_stats[agent_type] = {
                            "win_rates": [],
                            "avg_rewards": [],
                            "times": [],
                        }

                    agent_stats[agent_type]["win_rates"].append(agent["win_rate"])
                    agent_stats[agent_type]["avg_rewards"].append(agent["avg_reward"])
                    agent_stats[agent_type]["times"].append(agent["time_taken"])

            # Calculate statistics for each agent type
            config_stats = {}
            for agent_type, data in agent_stats.items():
                if data["win_rates"]:
                    win_rates = np.array(data["win_rates"])
                    avg_rewards = np.array(data["avg_rewards"])
                    times = np.array(data["times"])

                    config_stats[agent_type] = {
                        "win_rate": {
                            "mean": float(np.mean(win_rates)),
                            "std": float(np.std(win_rates)),
                            "var": float(np.var(win_rates)),
                            "min": float(np.min(win_rates)),
                            "max": float(np.max(win_rates)),
                            "median": float(np.median(win_rates)),
                            "ci_95": [
                                float(np.percentile(win_rates, 2.5)),
                                float(np.percentile(win_rates, 97.5)),
                            ],
                        },
                        "avg_reward": {
                            "mean": float(np.mean(avg_rewards)),
                            "std": float(np.std(avg_rewards)),
                            "var": float(np.var(avg_rewards)),
                            "min": float(np.min(avg_rewards)),
                            "max": float(np.max(avg_rewards)),
                            "median": float(np.median(avg_rewards)),
                            "ci_95": [
                                float(np.percentile(avg_rewards, 2.5)),
                                float(np.percentile(avg_rewards, 97.5)),
                            ],
                        },
                        "time_taken": {
                            "mean": float(np.mean(times)),
                            "std": float(np.std(times)),
                            "total": float(np.sum(times)),
                        },
                        "sample_size": len(win_rates),
                    }

            stats[config_name] = config_stats

        return stats

    def generate_report(self):
        """Generate comprehensive report with statistics."""
        stats = self.calculate_statistics()

        report = {
            "metadata": {
                "timestamp": self.timestamp,
                "seeds": self.seeds,
                "episodes": self.episodes,
                "eval_episodes": self.eval_episodes,
                "total_experiments": sum(len(runs) for runs in self.results.values()),
            },
            "configurations": list(self.results.keys()),
            "statistics": stats,
            "raw_results": self.results,
        }

        # Save detailed report
        report_file = os.path.join(self.output_dir, "evaluation_report.json")
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2)

        # Generate human-readable summary
        self.generate_summary_report(stats)

        return report

    def generate_summary_report(self, stats):
        """Generate human-readable summary."""
        summary_file = os.path.join(self.output_dir, "summary_report.txt")

        with open(summary_file, "w") as f:
            f.write("MULTI-SEED EVALUATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Seeds: {self.seeds}\n")
            f.write(f"Episodes: {self.episodes:,}\n")
            f.write(f"Evaluation Episodes: {self.eval_episodes:,}\n")
            f.write(f"Timestamp: {self.timestamp}\n\n")

            for config_name, config_stats in stats.items():
                f.write(f"\n{config_name.upper()}\n")
                f.write("-" * len(config_name) + "\n")

                for agent_type, agent_stats in config_stats.items():
                    f.write(f"\n{agent_type.upper()} Agent:\n")

                    wr = agent_stats["win_rate"]
                    ar = agent_stats["avg_reward"]

                    f.write(f"  Win Rate: {wr['mean']:.4f} ± {wr['std']:.4f} ")
                    f.write(f"(95% CI: [{wr['ci_95'][0]:.4f}, {wr['ci_95'][1]:.4f}])\n")
                    f.write(f"  Avg Reward: {ar['mean']:.4f} ± {ar['std']:.4f} ")
                    f.write(f"(95% CI: [{ar['ci_95'][0]:.4f}, {ar['ci_95'][1]:.4f}])\n")
                    f.write(f"  Sample Size: {agent_stats['sample_size']}\n")
                    f.write(f"  Avg Time: {agent_stats['time_taken']['mean']:.1f}s\n")

        print(f"\nReports saved to: {self.output_dir}/")

    def run_no_curriculum_configurations(self, deck_types=None):
        """Run all deck types with no-curriculum only."""
        if deck_types is None:
            deck_types = ["1-deck", "4-deck", "8-deck", "infinite"]

        print("\n" + "=" * 60)
        print("RUNNING NO-CURRICULUM EXPERIMENTS")
        print("=" * 60)

        for deck_type in deck_types:
            self.run_configuration(deck_type, use_curriculum=False)

        # Generate report for no-curriculum only
        print("\nGenerating no-curriculum report...")
        self.generate_report()
        print(
            f"✅ No-curriculum experiments completed! Check {self.output_dir}/ for results."
        )

    def run_curriculum_configurations(self, deck_types=None):
        """Run all deck types with curriculum only."""
        if deck_types is None:
            deck_types = ["1-deck", "4-deck", "8-deck", "infinite"]

        print("\n" + "=" * 60)
        print("RUNNING CURRICULUM EXPERIMENTS")
        print("=" * 60)

        for deck_type in deck_types:
            self.run_configuration(deck_type, use_curriculum=True)

        # Generate report for curriculum only
        print("\nGenerating curriculum report...")
        self.generate_report()
        print(
            f"✅ Curriculum experiments completed! Check {self.output_dir}/ for results."
        )

    def run_all_configurations(self, deck_types=None):
        """Run all deck types with both curriculum and no-curriculum."""
        if deck_types is None:
            deck_types = ["1-deck", "4-deck", "8-deck", "infinite"]

        print("\n" + "=" * 60)
        print("RUNNING ALL EXPERIMENTS (NO-CURRICULUM + CURRICULUM)")
        print("=" * 60)

        # Run no-curriculum first (faster)
        print("\nPhase 1: No-Curriculum Experiments")
        print("-" * 40)
        for deck_type in deck_types:
            self.run_configuration(deck_type, use_curriculum=False)

        # Run curriculum
        print("\nPhase 2: Curriculum Experiments")
        print("-" * 40)
        for deck_type in deck_types:
            self.run_configuration(deck_type, use_curriculum=True)

        # Generate final report
        print("\nGenerating final comparative report...")
        self.generate_report()
        print(f"✅ All experiments completed! Check {self.output_dir}/ for results.")


def main():
    parser = argparse.ArgumentParser(description="Multi-seed evaluation automation")
    parser.add_argument("--seeds", type=int, default=10, help="Number of seeds to run")
    parser.add_argument(
        "--episodes", type=int, default=500000, help="Training episodes"
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=50000, help="Evaluation episodes"
    )
    parser.add_argument(
        "--deck-types",
        nargs="+",
        default=["1-deck", "4-deck", "8-deck", "infinite"],
        help="Deck types to test",
    )
    parser.add_argument(
        "--test-run", action="store_true", help="Quick test with small parameters"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "no-curriculum", "curriculum"],
        default="all",
        help="Run mode: all (both), no-curriculum only, or curriculum only",
    )
    parser.add_argument(
        "--max-episodes-per-stage",
        type=int,
        default=20000,
        help="Maximum episodes per stage before forcing advancement",
    )

    args = parser.parse_args()

    if args.test_run:
        print("Running test with small parameters...")
        evaluator = MultiSeedEvaluator(seeds=2, episodes=1000, eval_episodes=100)
    else:
        evaluator = MultiSeedEvaluator(
            args.seeds, args.episodes, args.eval_episodes, args.max_episodes_per_stage
        )

    print(f"Starting multi-seed evaluation with {evaluator.seeds} seeds")
    print(f"Training episodes: {evaluator.episodes:,}")
    print(f"Evaluation episodes: {evaluator.eval_episodes:,}")
    print(f"Mode: {args.mode}")

    start_time = time.time()

    if args.mode == "no-curriculum":
        evaluator.run_no_curriculum_configurations(args.deck_types)
    elif args.mode == "curriculum":
        evaluator.run_curriculum_configurations(args.deck_types)
    else:  # args.mode == "all"
        evaluator.run_all_configurations(args.deck_types)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n✅ Total time: {total_time/3600:.1f} hours")


if __name__ == "__main__":
    main()
