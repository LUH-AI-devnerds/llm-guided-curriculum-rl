#!/usr/bin/env python3

import json
import os
import glob
import statistics


def analyze_curriculum_runs():
    """Analyze 8-deck curriculum runs (run 1-10) using training reports"""
    base_path = "logs/logs_version2/8deck-runs"

    all_times = []
    all_bust_rates = []
    run_data = {}

    # Process runs 1-10
    for run_num in range(1, 11):
        run_pattern = f"*-8-deck-0.9-run{run_num}"
        run_dirs = glob.glob(os.path.join(base_path, run_pattern))

        if not run_dirs:
            print(f"Warning: No directory found for run {run_num}")
            continue

        run_dir = run_dirs[0]

        # Look for training report
        report_files = glob.glob(
            os.path.join(run_dir, "reports", "curriculum_training_report_*.json")
        )
        if not report_files:
            print(f"Warning: No training report for run {run_num}")
            continue

        report_file = report_files[0]

        try:
            with open(report_file, "r") as f:
                report_data = json.load(f)

            # Extract times from global_performance_log
            run_times = []
            if "global_performance_log" in report_data:
                for stage_log in report_data["global_performance_log"]:
                    if "results" in stage_log:
                        for agent_id, result in stage_log["results"].items():
                            # Only include DQN agents (agent_0)
                            if "time_taken" in result and agent_id == "agent_0":
                                run_times.append(result["time_taken"])

            # Get bust rates from evaluation files - only from best performing stage per agent
            eval_dir = os.path.join(run_dir, "evaluation")
            run_busts = []
            agent_best_stages = {}

            # First, find the best win rate stage for each agent from the report
            if "global_performance_log" in report_data:
                for stage_log in report_data["global_performance_log"]:
                    if "results" in stage_log:
                        stage_id = stage_log["stage"]["stage_id"]
                        for agent_id, result in stage_log["results"].items():
                            # Only analyze DQN agents (agent_0)
                            if "win_rate" in result and agent_id == "agent_0":
                                agent_num = int(
                                    agent_id.split("_")[1]
                                )  # Extract agent number as int

                                # Track best stage for this agent
                                if agent_num not in agent_best_stages:
                                    agent_best_stages[agent_num] = {
                                        "win_rate": result["win_rate"],
                                        "stage": stage_id,
                                    }
                                elif (
                                    result["win_rate"]
                                    > agent_best_stages[agent_num]["win_rate"]
                                ):
                                    agent_best_stages[agent_num] = {
                                        "win_rate": result["win_rate"],
                                        "stage": stage_id,
                                    }

            # Now get bust rates only from evaluation files of best stages
            if os.path.exists(eval_dir):
                for eval_file in glob.glob(os.path.join(eval_dir, "*.json")):
                    try:
                        with open(eval_file, "r") as f:
                            eval_data = json.load(f)

                        if (
                            "summary" in eval_data
                            and "game_outcomes" in eval_data["summary"]
                        ):
                            # Check if this evaluation is from the best stage for this agent
                            # Only include DQN agents (agent_id == 0)
                            agent_id = eval_data.get("agent_id", -1)
                            stage_id = eval_data.get("stage_id", 0)

                            if (
                                agent_id == 0
                                and agent_id in agent_best_stages
                                and stage_id == agent_best_stages[agent_id]["stage"]
                            ):
                                outcomes = eval_data["summary"]["game_outcomes"]
                                busts = outcomes.get("busts", 0)
                                total_games = sum(outcomes.values())

                                if total_games > 0:
                                    bust_rate = busts / total_games
                                    run_busts.append(bust_rate)

                    except Exception as e:
                        continue

            if run_times and run_busts:
                avg_time = statistics.mean(run_times)
                avg_bust_rate = statistics.mean(run_busts)

                run_data[f"Run {run_num}"] = {
                    "avg_time": avg_time,
                    "avg_bust_rate": avg_bust_rate,
                    "num_training_sessions": len(run_times),
                    "num_evaluations": len(run_busts),
                    "best_stages": agent_best_stages,
                }

                all_times.extend(run_times)
                all_bust_rates.extend(run_busts)

                # Show detailed bust rate information for each agent's best stage
                agent_bust_info = {}
                for eval_file in glob.glob(os.path.join(eval_dir, "*.json")):
                    try:
                        with open(eval_file, "r") as f:
                            eval_data = json.load(f)

                        if (
                            "summary" in eval_data
                            and "game_outcomes" in eval_data["summary"]
                        ):
                            agent_id = eval_data.get("agent_id", -1)
                            stage_id = eval_data.get("stage_id", 0)

                            if (
                                agent_id in agent_best_stages
                                and stage_id == agent_best_stages[agent_id]["stage"]
                            ):
                                outcomes = eval_data["summary"]["game_outcomes"]
                                busts = outcomes.get("busts", 0)
                                total_games = sum(outcomes.values())

                                if total_games > 0:
                                    bust_rate = busts / total_games
                                    agent_bust_info[agent_id] = {
                                        "bust_rate": bust_rate,
                                        "stage": stage_id,
                                        "win_rate": agent_best_stages[agent_id][
                                            "win_rate"
                                        ],
                                    }

                    except Exception as e:
                        continue

                print(
                    f"Run {run_num}: Avg time = {avg_time:.2f}s, Combined avg bust rate = {avg_bust_rate:.3f}"
                )

                # Show individual agent bust rates
                for agent_id, info in agent_bust_info.items():
                    print(
                        f"  Agent {agent_id}: Stage {info['stage']} - Bust rate: {info['bust_rate']:.3f}, Win rate: {info['win_rate']:.3f}"
                    )

                if not agent_bust_info:
                    stage_info = ", ".join(
                        [
                            f"Agent {agent}: Stage {info['stage']} (WR: {info['win_rate']:.3f})"
                            for agent, info in agent_best_stages.items()
                        ]
                    )
                    print(f"  Best stages used: {stage_info}")

        except Exception as e:
            print(f"Error processing {report_file}: {e}")

    return all_times, all_bust_rates, run_data


def analyze_no_curriculum_runs():
    """Analyze no-curriculum runs"""
    eval_dir = "logs/logs-20250817-standard-8-deck-0.9-no-curriculum/evaluation"

    all_times = []
    all_bust_rates = []

    if not os.path.exists(eval_dir):
        print(f"Warning: No evaluation directory found at {eval_dir}")
        return [], [], {}

    for eval_file in glob.glob(os.path.join(eval_dir, "*.json")):
        try:
            with open(eval_file, "r") as f:
                data = json.load(f)

            # Only include DQN agents (agent_id == 0)
            agent_id = data.get("agent_id", -1)
            if agent_id != 0:
                continue

            if "summary" in data:
                # Get time information
                if "time_taken" in data["summary"]:
                    all_times.append(data["summary"]["time_taken"])

                # Get bust information
                if "game_outcomes" in data["summary"]:
                    outcomes = data["summary"]["game_outcomes"]
                    busts = outcomes.get("busts", 0)
                    total_games = sum(outcomes.values())

                    if total_games > 0:
                        bust_rate = busts / total_games
                        all_bust_rates.append(bust_rate)

        except Exception as e:
            print(f"Error processing {eval_file}: {e}")

    return all_times, all_bust_rates, {}


def main():
    print("=== 8-Deck Curriculum vs No-Curriculum Analysis (DQN Agents Only) ===\n")

    # Analyze curriculum runs
    print("Analyzing curriculum runs (runs 1-10) - DQN agents only...")
    curr_times, curr_busts, curr_run_data = analyze_curriculum_runs()

    print(f"\n📊 CURRICULUM SUMMARY:")
    print(f"Total training sessions: {len(curr_times)}")
    print(f"Total evaluations: {len(curr_busts)}")

    if curr_times:
        print(f"Average training time: {statistics.mean(curr_times):.2f}s")
        print(f"Training time std dev: {statistics.stdev(curr_times):.2f}s")
        print(f"Training time range: {min(curr_times):.2f}s - {max(curr_times):.2f}s")

    if curr_busts:
        print(f"Average bust rate: {statistics.mean(curr_busts):.3f}")
        print(f"Bust rate std dev: {statistics.stdev(curr_busts):.3f}")
        print(f"Bust rate range: {min(curr_busts):.3f} - {max(curr_busts):.3f}")

    # Analyze no-curriculum runs
    print("\nAnalyzing no-curriculum runs - DQN agents only...")
    no_curr_times, no_curr_busts, _ = analyze_no_curriculum_runs()

    print(f"\n📊 NO-CURRICULUM SUMMARY:")
    print(f"Total evaluations: {len(no_curr_times)}")

    if no_curr_times:
        print(f"Average evaluation time: {statistics.mean(no_curr_times):.2f}s")
        print(f"Evaluation time std dev: {statistics.stdev(no_curr_times):.2f}s")
        print(
            f"Evaluation time range: {min(no_curr_times):.2f}s - {max(no_curr_times):.2f}s"
        )

    if no_curr_busts:
        print(f"Average bust rate: {statistics.mean(no_curr_busts):.3f}")
        print(f"Bust rate std dev: {statistics.stdev(no_curr_busts):.3f}")
        print(f"Bust rate range: {min(no_curr_busts):.3f} - {max(no_curr_busts):.3f}")

    # Comparison
    print("\n🔍 === COMPARISON ===")

    if curr_busts and no_curr_busts:
        curr_avg_bust = statistics.mean(curr_busts)
        no_curr_avg_bust = statistics.mean(no_curr_busts)
        bust_diff = curr_avg_bust - no_curr_avg_bust
        bust_percent_diff = (bust_diff / no_curr_avg_bust) * 100

        print(f"\n🎯 BUST RATE COMPARISON:")
        print(f"  Curriculum:    {curr_avg_bust:.3f}")
        print(f"  No-Curriculum: {no_curr_avg_bust:.3f}")
        print(f"  Difference:    {bust_diff:+.3f} ({bust_percent_diff:+.1f}%)")

        if bust_diff < 0:
            print("  → Curriculum approach has LOWER bust rate (better)")
        else:
            print("  → No-curriculum approach has LOWER bust rate (better)")

    # Note about time comparison
    print(f"\n⏱️  TIME COMPARISON NOTE:")
    print("  Curriculum times are training session times (learning)")
    print("  No-curriculum times are evaluation times (testing)")
    print("  Direct comparison not meaningful due to different contexts")

    if curr_times and no_curr_times:
        print(f"  Curriculum avg training time: {statistics.mean(curr_times):.2f}s")
        print(f"  No-curriculum avg eval time: {statistics.mean(no_curr_times):.2f}s")


if __name__ == "__main__":
    main()
