#!/usr/bin/env python3
"""
Log Analysis Script for Blackjack RL Training
Generates visual summaries similar to strategy tables and performance metrics
"""

import json
import os
import sys
import argparse
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_evaluation_log(log_path):
    """Load evaluation log from JSON file."""
    with open(log_path, "r") as f:
        return json.load(f)


def create_strategy_table_heatmap(log_data, output_dir):
    """Create a heatmap similar to the Blackjack strategy table."""
    strategy_table = log_data["summary"]["strategy_table"]

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

    # Create matrices for each action
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

        stand_matrix[p_idx, d_idx] = stats["stand_percent"]
        hit_matrix[p_idx, d_idx] = stats["hit_percent"]
        double_matrix[p_idx, d_idx] = stats["double_percent"]
        split_matrix[p_idx, d_idx] = stats["split_percent"]

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        f'Agent Strategy Table - {log_data["agent_type"].upper()} Agent', fontsize=16
    )

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


def create_performance_summary(log_data, output_dir):
    """Create a performance summary table similar to Table 1."""
    summary = log_data["summary"]

    # Create performance table
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("tight")
    ax.axis("off")

    # Table data
    table_data = [
        ["Metric", "Value", "Percentage"],
        ["Win Rate", f"{summary['win_rate']:.3f}", f"{summary['win_rate']*100:.2f}%"],
        ["Average Reward", f"{summary['avg_reward']:.3f}", ""],
        [
            "Total Wins",
            summary["game_outcomes"]["wins"],
            f"{summary['game_outcome_percentages']['win_percent']:.2f}%",
        ],
        [
            "Busts",
            summary["game_outcomes"]["busts"],
            f"{summary['game_outcome_percentages']['bust_percent']:.2f}%",
        ],
        [
            "Pushes",
            summary["game_outcomes"]["pushes"],
            f"{summary['game_outcome_percentages']['push_percent']:.2f}%",
        ],
        [
            "Blackjacks",
            summary["game_outcomes"]["blackjacks"],
            f"{summary['game_outcome_percentages']['blackjack_percent']:.2f}%",
        ],
        [
            "Net Wins",
            "",
            f"{summary['game_outcome_percentages']['net_wins_percent']:.2f}%",
        ],
    ]

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

    plt.title(
        f'Performance Summary - {log_data["agent_type"].upper()} Agent\n'
        f'Evaluation Episodes: {log_data["evaluation_episodes"]}',
        fontsize=14,
        pad=20,
    )

    output_path = os.path.join(output_dir, "performance_summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Performance summary saved to: {output_path}")


def create_action_distribution_chart(log_data, output_dir):
    """Create action distribution charts."""
    action_performance = log_data["summary"]["action_performance"]

    # Action names
    action_names = ["Stand", "Hit", "Double", "Split"]
    action_counts = [action_performance[str(i)]["count"] for i in range(4)]
    action_avg_rewards = [action_performance[str(i)]["avg_reward"] for i in range(4)]

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Action count pie chart
    colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99"]
    ax1.pie(
        action_counts,
        labels=action_names,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
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
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{reward:.3f}",
            ha="center",
            va="bottom" if height > 0 else "top",
        )

    plt.tight_layout()
    output_path = os.path.join(output_dir, "action_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 Action distribution chart saved to: {output_path}")


def create_state_value_analysis(log_data, output_dir):
    """Create state-value analysis similar to the 3D plots."""
    state_reward_stats = log_data["summary"]["state_reward_stats"]

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

    plt.tight_layout()
    output_path = os.path.join(output_dir, "state_value_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"📊 State value analysis saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Blackjack RL training logs")
    parser.add_argument("log_path", help="Path to evaluation log JSON file")
    parser.add_argument(
        "--output-dir",
        default="analysis_output",
        help="Output directory for generated plots",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load log data
    try:
        log_data = load_evaluation_log(args.log_path)
        print(f"📊 Loaded evaluation log: {args.log_path}")
    except Exception as e:
        print(f"❌ Error loading log file: {e}")
        return

    # Generate visualizations
    print(f"\n🎨 Generating visualizations...")

    try:
        # Parse log filename to extract agent info
        log_filename = os.path.basename(args.log_path)
        # Format: evaluation_log_agent_1_tabular_20250719_120449.json
        parts = log_filename.replace(".json", "").split("_")
        agent_type = parts[4]  # tabular
        date = parts[5]  # 20250719
        time = parts[6]  # 120449

        # Get the directory of the log file
        log_dir = os.path.dirname(args.log_path)

        # Create analysis output directory within the same folder as the log file
        output_dir = os.path.join(
            log_dir,
            "analysis",
            agent_type,
            date,
            time,
        )

        # Create the output directory structure
        os.makedirs(output_dir, exist_ok=True)

        create_strategy_table_heatmap(log_data, output_dir)
        create_performance_summary(log_data, output_dir)
        create_action_distribution_chart(log_data, output_dir)
        create_state_value_analysis(log_data, output_dir)

        print(f"\n✅ Analysis complete! All visualizations saved to: {output_dir}")

        # Print summary
        summary = log_data["summary"]
        print(f"\n📋 QUICK SUMMARY:")
        print(f"Agent Type: {log_data['agent_type'].upper()}")
        print(f"Win Rate: {summary['win_rate']*100:.2f}%")
        print(f"Average Reward: {summary['avg_reward']:.3f}")
        print(
            f"Net Wins: {summary['game_outcome_percentages']['net_wins_percent']:.2f}%"
        )
        print(f"Bust Rate: {summary['game_outcome_percentages']['bust_percent']:.2f}%")

    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
