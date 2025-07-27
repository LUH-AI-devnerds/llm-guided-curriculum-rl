#!/usr/bin/env python3
"""
Log Analysis Script for Blackjack RL Training
Generates visual summaries similar to strategy tables and performance metrics
Now properly handles curriculum stages with stage-specific analysis
"""

import json
import os
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def load_evaluation_log(log_path):
    """Load evaluation log from JSON file."""
    with open(log_path, "r") as f:
        return json.load(f)


def load_training_log(log_path):
    """Load training log from JSON file."""
    with open(log_path, "r") as f:
        return json.load(f)


def load_run_summary(summary_path):
    """Load run summary from JSON file."""
    with open(summary_path, "r") as f:
        return json.load(f)


def load_curriculum_report(report_path):
    """Load curriculum training report from JSON file."""
    with open(report_path, "r") as f:
        return json.load(f)


def get_logs_from_summary(summary_data, log_dir):
    """Extract evaluation and training log paths from run summary."""
    evaluation_logs = []
    training_logs = []
    curriculum_report = None

    # Get the directories from summary
    eval_log_dir = os.path.join(log_dir, "evaluation")
    training_log_dir = os.path.join(log_dir, "training")
    reports_dir = os.path.join(log_dir, "reports")

    # Get list of evaluation log files
    eval_log_files = summary_data.get("files_generated", {}).get("evaluation", [])
    training_log_files = summary_data.get("files_generated", {}).get("training", [])

    for log_file in eval_log_files:
        log_path = os.path.join(eval_log_dir, log_file)
        if os.path.exists(log_path):
            evaluation_logs.append(log_path)
        else:
            print(f"⚠️  Warning: Evaluation log not found: {log_path}")

    for log_file in training_log_files:
        log_path = os.path.join(training_log_dir, log_file)
        if os.path.exists(log_path):
            training_logs.append(log_path)
        else:
            print(f"⚠️  Warning: Training log not found: {log_path}")

    # Look for curriculum training report
    if os.path.exists(reports_dir):
        report_files = [
            f
            for f in os.listdir(reports_dir)
            if f.startswith("curriculum_training_report")
        ]
        if report_files:
            # Use the most recent report
            report_files.sort()
            curriculum_report = os.path.join(reports_dir, report_files[-1])

    return evaluation_logs, training_logs, curriculum_report


def group_logs_by_agent_and_stage(
    evaluation_logs, training_logs, curriculum_report=None, run_summary=None
):
    """Group logs by agent and stage for curriculum analysis."""
    agent_stage_data = defaultdict(lambda: defaultdict(dict))

    # Get deck type and reward type from run summary
    deck_type = "unknown"
    reward_type = "simplified"  # Default to simplified reward system
    if run_summary and "training_config" in run_summary:
        deck_type = run_summary["training_config"].get("deck_type", "unknown")
        reward_type = run_summary["training_config"].get("reward_type", "simplified")

    # Process evaluation logs
    for log_path in evaluation_logs:
        try:
            log_data = load_evaluation_log(log_path)
            agent_id = log_data["agent_id"]
            agent_type = log_data["agent_type"]
            stage_id = log_data.get("stage_id", "unknown")

            agent_key = f"{agent_type}_{agent_id}"
            agent_stage_data[agent_key]["evaluation"][stage_id] = log_data
            # Store deck type and reward type at agent level
            if "deck_type" not in agent_stage_data[agent_key]:
                agent_stage_data[agent_key]["deck_type"] = deck_type
            if "reward_type" not in agent_stage_data[agent_key]:
                agent_stage_data[agent_key]["reward_type"] = reward_type
        except Exception as e:
            print(f"❌ Error loading evaluation log {log_path}: {e}")

    # Process training logs
    for log_path in training_logs:
        try:
            log_data = load_training_log(log_path)
            agent_id = log_data["agent_id"]
            agent_type = log_data["agent_type"]
            stage_id = log_data.get("stage_id", "unknown")

            agent_key = f"{agent_type}_{agent_id}"
            if "training" not in agent_stage_data[agent_key]:
                agent_stage_data[agent_key]["training"] = {}
            agent_stage_data[agent_key]["training"][stage_id] = log_data

            # Store deck type and reward type at agent level if not already set
            if "deck_type" not in agent_stage_data[agent_key]:
                agent_stage_data[agent_key]["deck_type"] = deck_type
            if "reward_type" not in agent_stage_data[agent_key]:
                agent_stage_data[agent_key]["reward_type"] = reward_type
        except Exception as e:
            print(f"❌ Error loading training log {log_path}: {e}")

    # Extract training times from curriculum report
    if curriculum_report:
        try:
            with open(curriculum_report, "r") as f:
                report_data = json.load(f)

            # Extract training times from global_performance_log
            if "global_performance_log" in report_data:
                for stage_log in report_data["global_performance_log"]:
                    stage_id = stage_log["stage"]["stage_id"]
                    results = stage_log["results"]

                    for agent_key, agent_result in results.items():
                        # Convert agent_key (e.g., "agent_0") to our format (e.g., "dqn_0")
                        agent_id = agent_key.split("_")[1]
                        agent_type = agent_result["agent_type"]
                        time_taken = agent_result.get("time_taken", 0)

                        our_agent_key = f"{agent_type}_{agent_id}"

                        # Store training time
                        if "training_times" not in agent_stage_data[our_agent_key]:
                            agent_stage_data[our_agent_key]["training_times"] = {}
                        agent_stage_data[our_agent_key]["training_times"][
                            stage_id
                        ] = time_taken

        except Exception as e:
            print(f"❌ Error loading curriculum report {curriculum_report}: {e}")

    return agent_stage_data


def create_strategy_table_heatmap(log_data, output_dir, stage_info="", deck_type=""):
    """Create a heatmap similar to the Blackjack strategy table."""
    strategy_table = log_data["summary"]["strategy_table"]

    # Debug: Print some strategy table info
    print(f"  📊 Strategy table has {len(strategy_table)} states")
    if strategy_table:
        sample_state = list(strategy_table.keys())[0]
        sample_stats = strategy_table[sample_state]
        print(f"  📊 Sample state {sample_state}: {sample_stats}")

    # Extract player sums and dealer cards
    player_sums = set()
    dealer_cards = set()

    for state_key in strategy_table.keys():
        # Parse state key: P{player_sum}_D{dealer_up}_A{has_ace}
        parts = state_key.split("_")
        player_sum = int(parts[0][1:])  # Remove 'P' prefix
        dealer_card = int(parts[1][1:])  # Remove 'D' prefix
        has_ace = parts[2] == "ATrue"

        player_sums.add(player_sum)
        dealer_cards.add(dealer_card)

    player_sums = sorted(list(player_sums))
    dealer_cards = sorted(list(dealer_cards))

    # Create matrices for each action (only 4 main actions: stand, hit, double, split)
    stand_matrix = np.zeros((len(player_sums), len(dealer_cards)))
    hit_matrix = np.zeros((len(player_sums), len(dealer_cards)))
    double_matrix = np.zeros((len(player_sums), len(dealer_cards)))
    split_matrix = np.zeros((len(player_sums), len(dealer_cards)))

    # Fill matrices
    for state_key, stats in strategy_table.items():
        parts = state_key.split("_")
        player_sum = int(parts[0][1:])
        dealer_card = int(parts[1][1:])

        p_idx = player_sums.index(player_sum)
        d_idx = dealer_cards.index(dealer_card)

        stand_matrix[p_idx, d_idx] = stats.get("stand_percent", 0)
        hit_matrix[p_idx, d_idx] = stats.get("hit_percent", 0)
        double_matrix[p_idx, d_idx] = stats.get("double_percent", 0)
        split_matrix[p_idx, d_idx] = stats.get("split_percent", 0)

    # Create subplots - back to 2x2 for 4 main actions
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    title = f'Agent Strategy Table - {log_data["agent_type"].upper()} Agent'
    if deck_type:
        title += f" ({deck_type})"
    if stage_info:
        title += f" - {stage_info}"
    fig.suptitle(title, fontsize=16)

    # Stand actions
    sns.heatmap(
        stand_matrix,
        annot=True,
        fmt=".1f",
        cmap="Reds",
        xticklabels=dealer_cards,
        yticklabels=player_sums,
        ax=axes[0, 0],
        cbar_kws={"label": "Stand %"},
    )
    axes[0, 0].set_title("Stand Actions (%)")
    axes[0, 0].set_xlabel("Dealer Up Card")
    axes[0, 0].set_ylabel("Player Sum")

    # Hit actions
    sns.heatmap(
        hit_matrix,
        annot=True,
        fmt=".1f",
        cmap="Greens",
        xticklabels=dealer_cards,
        yticklabels=player_sums,
        ax=axes[0, 1],
        cbar_kws={"label": "Hit %"},
    )
    axes[0, 1].set_title("Hit Actions (%)")
    axes[0, 1].set_xlabel("Dealer Up Card")
    axes[0, 1].set_ylabel("Player Sum")

    # Double actions
    sns.heatmap(
        double_matrix,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=dealer_cards,
        yticklabels=player_sums,
        ax=axes[1, 0],
        cbar_kws={"label": "Double %"},
    )
    axes[1, 0].set_title("Double Actions (%)")
    axes[1, 0].set_xlabel("Dealer Up Card")
    axes[1, 0].set_ylabel("Player Sum")

    # Split actions
    sns.heatmap(
        split_matrix,
        annot=True,
        fmt=".1f",
        cmap="Purples",
        xticklabels=dealer_cards,
        yticklabels=player_sums,
        ax=axes[1, 1],
        cbar_kws={"label": "Split %"},
    )
    axes[1, 1].set_title("Split Actions (%)")
    axes[1, 1].set_xlabel("Dealer Up Card")
    axes[1, 1].set_ylabel("Player Sum")

    plt.tight_layout()
    output_path = os.path.join(output_dir, "strategy_table_heatmap.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Strategy table heatmap saved to: {output_path}")


def create_performance_summary(log_data, output_dir, stage_info="", deck_type=""):
    """Create a performance summary table similar to Table 1."""
    summary = log_data["summary"]

    # Create performance table
    fig, ax = plt.subplots(
        figsize=(12, 10)
    )  # Increased height to accommodate more rows
    ax.axis("tight")
    ax.axis("off")

    # Table data - now includes surrender and insurance statistics
    table_data = [
        ["Metric", "Value", "Percentage"],
        [
            "Win Rate",
            f"{summary.get('win_rate', 0):.3f}",
            f"{summary.get('win_rate', 0)*100:.2f}%",
        ],
        ["Average Reward", f"{summary.get('avg_reward', 0):.3f}", ""],
        ["Total Wins", f"{summary.get('total_wins', 0)}", ""],
    ]

    # Add game outcome percentages if available
    game_outcomes = summary.get("game_outcome_percentages", {})
    if game_outcomes:
        table_data.extend(
            [
                [
                    "Win/Loss Ratio",
                    f"{game_outcomes.get('win_loss_ratio', 0):.2f}",
                    "",
                ],
                [
                    "Bust Rate",
                    f"{game_outcomes.get('bust_percent', 0):.1f}%",
                    "",
                ],
                [
                    "Blackjack Rate",
                    f"{game_outcomes.get('blackjack_percent', 0):.1f}%",
                    "",
                ],
                [
                    "Push Rate",
                    f"{game_outcomes.get('push_percent', 0):.1f}%",
                    "",
                ],
            ]
        )

    # Add surrender and insurance statistics if available
    if "surrenders" in summary:
        table_data.append(
            [
                "Surrenders",
                f"{summary['surrenders']}",
                f"{summary['surrenders']/summary.get('total_hands', 1)*100:.1f}%",
            ]
        )

    if "insurance_bets" in summary:
        table_data.append(
            [
                "Insurance Bets",
                f"{summary['insurance_bets']}",
                f"{summary['insurance_bets']/summary.get('total_hands', 1)*100:.1f}%",
            ]
        )

    # Create table
    table = ax.table(
        cellText=table_data[1:], colLabels=table_data[0], cellLoc="center", loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 2)

    # Style the table
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    for i in range(1, len(table_data)):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#f0f0f0")

    title = f'Performance Summary - {log_data["agent_type"].upper()} Agent'
    if deck_type:
        title += f" ({deck_type})"
    title += f'\nEvaluation Episodes: {log_data.get("evaluation_episodes", "Unknown")}'
    if stage_info:
        title += f"\n{stage_info}"

    plt.title(title, fontsize=14, pad=20)

    output_path = os.path.join(output_dir, "performance_summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Performance summary saved to: {output_path}")


def create_action_distribution_chart(log_data, output_dir, stage_info="", deck_type=""):
    """Create action distribution charts."""
    # Check if action_performance exists, otherwise try to derive from other data
    action_performance = log_data["summary"].get("action_performance", {})

    # If action_performance is empty, try to get action usage from strategy table
    if not action_performance:
        strategy_table = log_data["summary"].get("strategy_table", {})
        if strategy_table:
            # Derive action usage from strategy table
            action_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
            action_rewards = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}

            for state_stats in strategy_table.values():
                total_actions = state_stats.get("total_actions", 0)
                avg_reward = state_stats.get("avg_reward", 0)

                # Calculate action counts
                action_counts[0] += int(
                    state_stats.get("stand_percent", 0) * total_actions / 100
                )
                action_counts[1] += int(
                    state_stats.get("hit_percent", 0) * total_actions / 100
                )
                action_counts[2] += int(
                    state_stats.get("double_percent", 0) * total_actions / 100
                )
                action_counts[3] += int(
                    state_stats.get("split_percent", 0) * total_actions / 100
                )
                action_counts[4] += int(
                    state_stats.get("surrender_percent", 0) * total_actions / 100
                )
                action_counts[5] += int(
                    state_stats.get("insurance_percent", 0) * total_actions / 100
                )

                # Collect rewards for each action (weighted by action percentage)
                for action_id in range(6):
                    action_percent = state_stats.get(
                        f"{['stand', 'hit', 'double', 'split', 'surrender', 'insurance'][action_id]}_percent",
                        0,
                    )
                    if action_percent > 0:
                        # Add the average reward weighted by action percentage
                        action_rewards[action_id].extend(
                            [avg_reward] * int(action_percent * total_actions / 100)
                        )

            # Convert to the expected format with calculated average rewards
            action_performance = {}
            for i in range(6):
                count = action_counts[i]
                avg_reward = np.mean(action_rewards[i]) if action_rewards[i] else 0.0
                action_performance[str(i)] = {"count": count, "avg_reward": avg_reward}
        else:
            # If no strategy table either, create empty data
            action_performance = {
                str(i): {"count": 0, "avg_reward": 0.0} for i in range(6)
            }

    # Action names (now 6 actions)
    action_names = ["Stand", "Hit", "Double", "Split", "Surrender", "Insurance"]
    action_counts = []
    action_avg_rewards = []

    # Get data for all 6 actions, handling missing actions gracefully
    for i in range(6):
        action_key = str(i)
        if action_key in action_performance:
            action_counts.append(action_performance[action_key]["count"])
            action_avg_rewards.append(action_performance[action_key]["avg_reward"])
        else:
            action_counts.append(0)
            action_avg_rewards.append(0)

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Action count pie chart
    colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#ff9966", "#cccccc"]
    # Only show actions that were actually used
    used_actions = [
        (name, count) for name, count in zip(action_names, action_counts) if count > 0
    ]

    if used_actions:
        used_names, used_counts = zip(*used_actions)
        used_colors = [colors[action_names.index(name)] for name in used_names]

        ax1.pie(
            used_counts,
            labels=used_names,
            autopct="%1.1f%%",
            colors=used_colors,
            startangle=90,
        )
    else:
        ax1.text(
            0.5,
            0.5,
            "No actions recorded",
            ha="center",
            va="center",
            transform=ax1.transAxes,
        )
    ax1.set_title("Action Distribution")

    # Action average rewards bar chart
    bars = ax2.bar(action_names, action_avg_rewards, color=colors)
    ax2.set_title("Average Reward per Action")
    ax2.set_ylabel("Average Reward")
    ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    # Add value labels on bars
    for bar, reward in zip(bars, action_avg_rewards):
        height = bar.get_height()
        if height != 0:  # Only label non-zero values
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{reward:.3f}",
                ha="center",
                va="bottom" if height > 0 else "top",
            )

    title = f"{log_data['agent_type'].upper()} Agent"
    if deck_type:
        title += f" ({deck_type})"
    if stage_info:
        title += f" - {stage_info}"
    fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "action_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Action distribution chart saved to: {output_path}")


def create_state_value_analysis(log_data, output_dir, stage_info="", deck_type=""):
    """Create state-value analysis similar to the 3D plots."""
    state_reward_stats = log_data["summary"].get("state_reward_stats", {})

    if not state_reward_stats:
        print(f"  ⚠️  No state reward statistics available for state value analysis")
        return

    # Separate hard and soft totals
    hard_states = {}
    soft_states = {}

    for state_key, stats in state_reward_stats.items():
        parts = state_key.split("_")
        player_sum = int(parts[0][1:])
        dealer_card = int(parts[1][1:])
        has_ace = parts[2] == "ATrue"

        if has_ace:
            soft_states[(player_sum, dealer_card)] = stats["avg"]
        else:
            hard_states[(player_sum, dealer_card)] = stats["avg"]

    # Create 3D plots
    fig = plt.figure(figsize=(15, 6))

    # Hard totals (no ace)
    if hard_states:
        ax1 = fig.add_subplot(121, projection="3d")
        x = [state[1] for state in hard_states.keys()]  # dealer card
        y = [state[0] for state in hard_states.keys()]  # player sum
        z = list(hard_states.values())  # state value

        scatter = ax1.scatter(x, y, z, c=z, cmap="viridis", s=50)
        ax1.set_xlabel("Dealer Up Card")
        ax1.set_ylabel("Player Sum")
        ax1.set_zlabel("State Value")
        ax1.set_title("State Values - Hard Totals (No Ace)")
        fig.colorbar(scatter, ax=ax1, shrink=0.5, aspect=5)

    # Soft totals (with ace)
    if soft_states:
        ax2 = fig.add_subplot(122, projection="3d")
        x = [state[1] for state in soft_states.keys()]  # dealer card
        y = [state[0] for state in soft_states.keys()]  # player sum
        z = list(soft_states.values())  # state value

        scatter = ax2.scatter(x, y, z, c=z, cmap="viridis", s=50)
        ax2.set_xlabel("Dealer Up Card")
        ax2.set_ylabel("Player Sum")
        ax2.set_zlabel("State Value")
        ax2.set_title("State Values - Soft Totals (With Ace)")
        fig.colorbar(scatter, ax=ax2, shrink=0.5, aspect=5)

    title = f"{log_data['agent_type'].upper()} Agent"
    if deck_type:
        title += f" ({deck_type})"
    if stage_info:
        title += f" - {stage_info}"
    fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "state_value_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 State value analysis saved to: {output_path}")


def create_stage_progression_charts(agent_stage_data, output_dir):
    """Create charts showing performance progression across stages."""

    for agent_key, agent_data in agent_stage_data.items():
        evaluation_data = agent_data.get("evaluation", {})
        training_data = agent_data.get("training", {})
        deck_type = agent_data.get("deck_type", "")

        if not evaluation_data:
            continue

        # Sort stages - handle both string and integer keys
        stages = []
        for key in evaluation_data.keys():
            if key != "unknown":
                try:
                    stages.append(int(key))
                except (ValueError, TypeError):
                    continue
        stages.sort()

        if len(stages) < 2:
            continue

        # Extract performance metrics across stages
        win_rates = []
        avg_rewards = []
        win_loss_ratios = []
        bust_rates = []
        blackjack_rates = []
        training_times = []  # Add training times
        stage_names = []

        for stage_id in stages:
            # Try both string and integer keys
            stage_data = None
            if str(stage_id) in evaluation_data:
                stage_data = evaluation_data[str(stage_id)]
            elif stage_id in evaluation_data:
                stage_data = evaluation_data[stage_id]

            if stage_data is None:
                print(
                    f"⚠️  Warning: Could not find stage {stage_id} data for {agent_key}"
                )
                continue

            summary = stage_data["summary"]

            win_rates.append(summary.get("win_rate", 0) * 100)
            avg_rewards.append(summary.get("avg_reward", 0))

            game_outcomes = summary.get("game_outcome_percentages", {})
            win_loss_ratios.append(game_outcomes.get("win_loss_ratio", 0))
            bust_rates.append(game_outcomes.get("bust_percent", 0))
            blackjack_rates.append(game_outcomes.get("blackjack_percent", 0))

            # Get training time for this stage
            training_time = agent_data.get("training_times", {}).get(str(stage_id), 0)
            training_times.append(training_time)

            # Get stage name from training data if available
            stage_name = f"Stage {stage_id}"
            if str(stage_id) in training_data:
                stage_name = training_data[str(stage_id)].get(
                    "stage_name", f"Stage {stage_id}"
                )
            elif stage_id in training_data:
                stage_name = training_data[stage_id].get(
                    "stage_name", f"Stage {stage_id}"
                )
            stage_names.append(stage_name)

        # Create progression charts - back to 2x2 layout
        title = f"Stage Progression - {agent_key.upper()}"
        title_parts = []
        if deck_type and deck_type != "unknown":
            title_parts.append(f"Deck: {deck_type}")
        reward_type = agent_data.get("reward_type", "")
        if reward_type and reward_type != "unknown":
            title_parts.append(f"Reward: {reward_type}")
        if title_parts:
            title += f" ({' | '.join(title_parts)})"
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16)

        # Win rate progression
        axes[0, 0].plot(
            stages, win_rates, "o-", linewidth=2, markersize=8, color="#4CAF50"
        )
        axes[0, 0].set_title("Win Rate Progression")
        axes[0, 0].set_xlabel("Stage")
        axes[0, 0].set_ylabel("Win Rate (%)")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_xticks(stages)

        # Add stage names as x-tick labels
        axes[0, 0].set_xticklabels(
            [f"{s}\n{stage_names[i]}" for i, s in enumerate(stages)],
            rotation=45,
            ha="right",
        )

        # Average reward progression
        axes[0, 1].plot(
            stages, avg_rewards, "o-", linewidth=2, markersize=8, color="#2196F3"
        )
        axes[0, 1].set_title("Average Reward Progression")
        axes[0, 1].set_xlabel("Stage")
        axes[0, 1].set_ylabel("Average Reward")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_xticks(stages)
        axes[0, 1].set_xticklabels(
            [f"{s}\n{stage_names[i]}" for i, s in enumerate(stages)],
            rotation=45,
            ha="right",
        )
        axes[0, 1].axhline(y=0, color="black", linestyle="-", alpha=0.3)

        # Win/Loss ratio progression
        axes[1, 0].plot(
            stages, win_loss_ratios, "o-", linewidth=2, markersize=8, color="#FF9800"
        )
        axes[1, 0].set_title("Win/Loss Ratio Progression")
        axes[1, 0].set_xlabel("Stage")
        axes[1, 0].set_ylabel("Win/Loss Ratio")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xticks(stages)
        axes[1, 0].set_xticklabels(
            [f"{s}\n{stage_names[i]}" for i, s in enumerate(stages)],
            rotation=45,
            ha="right",
        )
        axes[1, 0].axhline(y=1, color="black", linestyle="-", alpha=0.3)

        # Training time progression
        if any(t > 0 for t in training_times):
            axes[1, 1].plot(
                stages, training_times, "o-", linewidth=2, markersize=8, color="#9C27B0"
            )
            axes[1, 1].set_title("Training Time Progression")
            axes[1, 1].set_xlabel("Stage")
            axes[1, 1].set_ylabel("Training Time (seconds)")
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].set_xticks(stages)
            axes[1, 1].set_xticklabels(
                [f"{s}\n{stage_names[i]}" for i, s in enumerate(stages)],
                rotation=45,
                ha="right",
            )
        else:
            # If no training times available, show game outcomes breakdown
            x = np.arange(len(stages))
            width = 0.35

            axes[1, 1].bar(
                x - width / 2,
                bust_rates,
                width,
                label="Bust Rate",
                color="#f44336",
                alpha=0.7,
            )
        axes[1, 1].bar(
            x + width / 2,
            blackjack_rates,
            width,
            label="Blackjack Rate",
            color="#4CAF50",
            alpha=0.7,
        )

        axes[1, 1].set_title("Game Outcomes by Stage")
        axes[1, 1].set_xlabel("Stage")
        axes[1, 1].set_ylabel("Percentage (%)")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(
            [f"{s}\n{stage_names[i]}" for i, s in enumerate(stages)],
            rotation=45,
            ha="right",
        )
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = os.path.join(output_dir, f"stage_progression_{agent_key}.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"📊 Stage progression chart saved to: {output_path}")


def create_comparative_analysis(agent_stage_data, output_dir):
    """Create comparative analysis across all agents and stages."""

    # Extract all agent-stage combinations
    all_agent_stages = []
    training_times = {}  # Store training times for each agent

    for agent_key, agent_data in agent_stage_data.items():
        evaluation_data = agent_data.get("evaluation", {})
        deck_type = agent_data.get("deck_type", "")
        reward_type = agent_data.get("reward_type", "")

        # Extract training times from curriculum report data
        training_times_data = agent_data.get("training_times", {})
        if training_times_data:
            total_time = sum(training_times_data.values())
            training_times[agent_key] = total_time

        for stage_id, stage_data in evaluation_data.items():
            if stage_id == "unknown":
                continue

            summary = stage_data["summary"]
            all_agent_stages.append(
                {
                    "agent_key": agent_key,
                    "stage_id": int(stage_id),
                    "win_rate": summary.get("win_rate", 0) * 100,
                    "avg_reward": summary.get("avg_reward", 0),
                    "win_loss_ratio": summary.get("game_outcome_percentages", {}).get(
                        "win_loss_ratio", 0
                    ),
                    "bust_rate": summary.get("game_outcome_percentages", {}).get(
                        "bust_percent", 0
                    ),
                    "blackjack_rate": summary.get("game_outcome_percentages", {}).get(
                        "blackjack_percent", 0
                    ),
                    "push_rate": summary.get("game_outcome_percentages", {}).get(
                        "push_percent", 0
                    ),
                    "surrender_rate": summary.get("surrenders", 0)
                    / summary.get("total_hands", 1)
                    * 100,
                    "insurance_rate": summary.get("insurance_bets", 0)
                    / summary.get("total_hands", 1)
                    * 100,
                    "deck_type": deck_type,
                    "reward_type": reward_type,
                }
            )

    if not all_agent_stages:
        print("❌ No valid agent-stage data found for comparative analysis")
        return

    # Get deck type and reward type for title
    deck_types = set(
        [
            data["deck_type"]
            for data in all_agent_stages
            if data["deck_type"] != "unknown"
        ]
    )
    reward_types = set(
        [
            data["reward_type"]
            for data in all_agent_stages
            if data["reward_type"] != "unknown"
        ]
    )

    title_parts = []
    if deck_types:
        title_parts.append(f"Deck: {', '.join(deck_types)}")
    if reward_types:
        title_parts.append(f"Reward: {', '.join(reward_types)}")

    title_str = f" ({' | '.join(title_parts)})" if title_parts else ""

    # Create comparative charts - back to 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    fig.suptitle(f"Comparative Agent Performance Across Stages{title_str}", fontsize=16)

    # Agent labels
    agent_labels = list(set([data["agent_key"] for data in all_agent_stages]))
    agent_labels.sort()

    # Win rates comparison by stage
    stages = sorted(list(set([data["stage_id"] for data in all_agent_stages])))

    for agent_key in agent_labels:
        agent_data = [d for d in all_agent_stages if d["agent_key"] == agent_key]
        agent_data.sort(key=lambda x: x["stage_id"])

        win_rates = [d["win_rate"] for d in agent_data]
        stage_ids = [d["stage_id"] for d in agent_data]

        axes[0, 0].plot(
            stage_ids,
            win_rates,
            "o-",
            linewidth=2,
            markersize=8,
            label=agent_key.upper(),
        )

    axes[0, 0].set_title("Win Rates by Stage")
    axes[0, 0].set_xlabel("Stage")
    axes[0, 0].set_ylabel("Win Rate (%)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(stages)

    # Average rewards comparison by stage
    for agent_key in agent_labels:
        agent_data = [d for d in all_agent_stages if d["agent_key"] == agent_key]
        agent_data.sort(key=lambda x: x["stage_id"])

        avg_rewards = [d["avg_reward"] for d in agent_data]
        stage_ids = [d["stage_id"] for d in agent_data]

        axes[0, 1].plot(
            stage_ids,
            avg_rewards,
            "o-",
            linewidth=2,
            markersize=8,
            label=agent_key.upper(),
        )

    axes[0, 1].set_title("Average Rewards by Stage")
    axes[0, 1].set_xlabel("Stage")
    axes[0, 1].set_ylabel("Average Reward")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(stages)
    axes[0, 1].axhline(y=0, color="black", linestyle="-", alpha=0.3)

    # Bust rates comparison by stage
    for agent_key in agent_labels:
        agent_data = [d for d in all_agent_stages if d["agent_key"] == agent_key]
        agent_data.sort(key=lambda x: x["stage_id"])

        bust_rates = [d["bust_rate"] for d in agent_data]
        stage_ids = [d["stage_id"] for d in agent_data]

        axes[1, 0].plot(
            stage_ids,
            bust_rates,
            "o-",
            linewidth=2,
            markersize=8,
            label=agent_key.upper(),
        )

    axes[1, 0].set_title("Bust Rates by Stage")
    axes[1, 0].set_xlabel("Stage")
    axes[1, 0].set_ylabel("Bust Rate (%)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xticks(stages)

    # Training time comparison
    if training_times:
        agent_names = list(training_times.keys())
        total_times = list(training_times.values())

        bars = axes[1, 1].bar(
            agent_names, total_times, color=["#4CAF50", "#2196F3", "#FF9800"]
        )
        axes[1, 1].set_title("Total Training Time Comparison")
        axes[1, 1].set_xlabel("Agent")
        axes[1, 1].set_ylabel("Total Training Time (seconds)")
        axes[1, 1].grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, time_val in zip(bars, total_times):
            height = bar.get_height()
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + max(total_times) * 0.01,
                f"{time_val:.1f}s",
                ha="center",
                va="bottom",
            )
    else:
        # Final stage performance comparison
        final_stage = max(stages)
        final_performances = [
            d for d in all_agent_stages if d["stage_id"] == final_stage
        ]

        agent_names = [d["agent_key"].upper() for d in final_performances]
        final_win_rates = [d["win_rate"] for d in final_performances]
        final_bust_rates = [d["bust_rate"] for d in final_performances]

        x = np.arange(len(agent_names))
        width = 0.35

        bars1 = axes[1, 1].bar(
            x - width / 2,
            final_win_rates,
            width,
            label="Win Rate",
            color="#4CAF50",
            alpha=0.7,
        )
        bars2 = axes[1, 1].bar(
            x + width / 2,
            final_bust_rates,
            width,
            label="Bust Rate",
            color="#f44336",
            alpha=0.7,
        )

        axes[1, 1].set_title(f"Final Stage ({final_stage}) Performance")
        axes[1, 1].set_xlabel("Agent")
        axes[1, 1].set_ylabel("Rate (%)")
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(agent_names, rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                label = f"{height:.1f}%"
                axes[1, 1].text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.5,
                    label,
                    ha="center",
                    va="bottom",
                )

    plt.tight_layout()
    output_path = os.path.join(output_dir, "comparative_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Comparative analysis saved to: {output_path}")


def analyze_agent_stages(agent_key, agent_data, output_dir):
    """Analyze all stages for a single agent."""
    print(f"📊 Analyzing agent: {agent_key}")

    evaluation_data = agent_data.get("evaluation", {})
    training_data = agent_data.get("training", {})
    deck_type = agent_data.get("deck_type", "")

    if not evaluation_data:
        print(f"❌ No evaluation data found for {agent_key}")
        return

    # Create agent-specific output directory
    agent_output_dir = os.path.join(output_dir, agent_key)
    os.makedirs(agent_output_dir, exist_ok=True)

    # Analyze each stage
    for stage_id, stage_data in evaluation_data.items():
        if stage_id == "unknown":
            continue

        print(f"  📊 Analyzing Stage {stage_id}")

        # Get stage info
        stage_info = f"Stage {stage_id}"
        if str(stage_id) in training_data:
            stage_name = training_data[str(stage_id)].get(
                "stage_name", f"Stage {stage_id}"
            )
            stage_info = f"Stage {stage_id}: {stage_name}"
        elif stage_id in training_data:
            stage_name = training_data[stage_id].get("stage_name", f"Stage {stage_id}")
            stage_info = f"Stage {stage_id}: {stage_name}"

        # Create stage-specific output directory
        stage_output_dir = os.path.join(agent_output_dir, f"stage_{stage_id}")
        os.makedirs(stage_output_dir, exist_ok=True)

        # Generate visualizations for this stage
        create_strategy_table_heatmap(
            stage_data, stage_output_dir, stage_info, deck_type
        )
        create_performance_summary(stage_data, stage_output_dir, stage_info, deck_type)
        create_action_distribution_chart(
            stage_data, stage_output_dir, stage_info, deck_type
        )
        create_state_value_analysis(stage_data, stage_output_dir, stage_info, deck_type)

    # Create stage progression charts
    create_stage_progression_charts({agent_key: agent_data}, agent_output_dir)


def main():
    parser = argparse.ArgumentParser(description="Analyze Blackjack RL training logs")
    parser.add_argument(
        "input_path", help="Path to evaluation log JSON file OR run summary JSON file"
    )
    parser.add_argument(
        "--output-dir",
        default="analysis_output",
        help="Output directory for generated plots",
    )

    args = parser.parse_args()

    # Check if input is a run summary or single log
    if "run_summary" in args.input_path:
        # Analyze all logs from run summary
        print(f"📊 Loading run summary: {args.input_path}")

        try:
            summary_data = load_run_summary(args.input_path)
            log_dir = os.path.dirname(args.input_path)

            # Get all evaluation and training logs from summary
            evaluation_logs, training_logs, curriculum_report = get_logs_from_summary(
                summary_data, log_dir
            )

            if not evaluation_logs:
                print("❌ No evaluation logs found in run summary!")
                return

            print(
                f"📊 Found {len(evaluation_logs)} evaluation logs and {len(training_logs)} training logs to analyze"
            )

            # Group logs by agent and stage
            agent_stage_data = group_logs_by_agent_and_stage(
                evaluation_logs, training_logs, curriculum_report, summary_data
            )

            # Create output directory based on run summary
            run_timestamp = summary_data.get(
                "run_timestamp", datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            output_dir = os.path.join(
                log_dir, "analysis", "run_analysis", run_timestamp
            )
            os.makedirs(output_dir, exist_ok=True)

            # Analyze each agent's stages
            for agent_key, agent_data in agent_stage_data.items():
                analyze_agent_stages(agent_key, agent_data, output_dir)

            # Create comparative analysis if we have multiple agents or stages
            if len(agent_stage_data) > 1 or any(
                len(agent_data.get("evaluation", {})) > 1
                for agent_data in agent_stage_data.values()
            ):
                print(f"\n📊 Creating comparative analysis...")
                create_comparative_analysis(agent_stage_data, output_dir)

            # Print run summary
            print(f"\n✅ RUN ANALYSIS COMPLETE!")
            print(f"📁 All analyses saved to: {output_dir}")
            print(f"\n📋 RUN SUMMARY:")
            print(f"Total Agents Analyzed: {len(agent_stage_data)}")

            for agent_key, agent_data in agent_stage_data.items():
                evaluation_data = agent_data.get("evaluation", {})
                # Handle both string and integer keys
                stages = []
                for key in evaluation_data.keys():
                    if key != "unknown":
                        try:
                            stages.append(int(key))
                        except (ValueError, TypeError):
                            continue
                stages.sort()
                print(f"  {agent_key.upper()}: {len(stages)} stages ({stages})")

                # Show final stage performance
                if stages:
                    final_stage = max(stages)
                    # Try both string and integer keys
                    final_data = None
                    if str(final_stage) in evaluation_data:
                        final_data = evaluation_data[str(final_stage)]
                    elif final_stage in evaluation_data:
                        final_data = evaluation_data[final_stage]

                    if final_data:
                        summary = final_data["summary"]
                        game_outcomes = summary.get("game_outcome_percentages", {})
                        print(
                            f"    Final Stage {final_stage}: Win Rate: {summary.get('win_rate', 0)*100:.2f}%, "
                            f"Win/Loss Ratio: {game_outcomes.get('win_loss_ratio', 0):.2f}"
                        )

        except Exception as e:
            print(f"❌ Error processing run summary: {e}")
            import traceback

            traceback.print_exc()

    else:
        # Analyze single log file (original functionality)
        try:
            log_data = load_evaluation_log(args.input_path)
            print(f"📊 Loaded evaluation log: {args.input_path}")
        except Exception as e:
            print(f"❌ Error loading log file: {e}")
            return

        # Try to find corresponding curriculum report
        curriculum_report = None
        log_dir = os.path.dirname(args.input_path)
        reports_dir = os.path.join(log_dir, "..", "reports")
        if os.path.exists(reports_dir):
            report_files = [
                f
                for f in os.listdir(reports_dir)
                if f.startswith("curriculum_training_report")
            ]
            if report_files:
                report_files.sort()
                curriculum_report = os.path.join(reports_dir, report_files[-1])
                print(f"📊 Found curriculum report: {curriculum_report}")

        # Generate visualizations
        print(f"\n🎨 Generating visualizations...")

        try:
            # Parse log filename to extract agent info
            log_filename = os.path.basename(args.input_path)
            parts = log_filename.replace(".json", "").split("_")
            agent_type = parts[4]  # tabular/dqn
            agent_id = parts[3]
            date = parts[5]  # 20250719
            time = parts[6]  # 120449

            # Get the directory of the log file
            log_dir = os.path.dirname(args.input_path)

            # Create analysis output directory within the same folder as the log file
            output_dir = os.path.join(
                log_dir,
                "analysis",
                f"{agent_type}_{agent_id}",
                date,
                time,
            )

            # Create the output directory structure
            os.makedirs(output_dir, exist_ok=True)

            # Get stage info and deck type
            stage_info = ""
            deck_type = log_data.get("deck_type", "")
            if "stage_id" in log_data:
                stage_id = log_data["stage_id"]
                stage_info = f"Stage {stage_id}"

            # Generate each visualization with individual error handling
            try:
                create_strategy_table_heatmap(
                    log_data, output_dir, stage_info, deck_type
                )
                print("  ✅ Strategy table heatmap created")
            except Exception as e:
                print(f"  ❌ Error creating strategy table heatmap: {e}")

            try:
                create_performance_summary(log_data, output_dir, stage_info, deck_type)
                print("  ✅ Performance summary created")
            except Exception as e:
                print(f"  ❌ Error creating performance summary: {e}")

            try:
                create_action_distribution_chart(
                    log_data, output_dir, stage_info, deck_type
                )
                print("  ✅ Action distribution chart created")
            except Exception as e:
                print(f"  ❌ Error creating action distribution chart: {e}")

            try:
                create_state_value_analysis(log_data, output_dir, stage_info, deck_type)
                print("  ✅ State value analysis created")
            except Exception as e:
                print(f"  ❌ Error creating state value analysis: {e}")

            print(f"\n✅ Analysis complete! All visualizations saved to: {output_dir}")

            # Print summary
            summary = log_data["summary"]
            print(f"\n📋 QUICK SUMMARY:")
            print(f"Agent Type: {log_data['agent_type'].upper()}")
            if "stage_id" in log_data:
                print(f"Stage: {log_data['stage_id']}")
            print(f"Win Rate (Episodes): {summary.get('win_rate', 0)*100:.2f}%")
            # print(f"Win Rate (Hands): {summary.get('hand_win_rate', 0)*100:.2f}%")
            print(f"Average Reward: {summary.get('avg_reward', 0):.3f}")

            game_outcomes = summary.get("game_outcome_percentages", {})
            print(f"Win/Loss Ratio: {game_outcomes.get('win_loss_ratio', 0):.2f}")
            print(f"Bust Rate: {game_outcomes.get('bust_percent', 0):.2f}%")

        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
