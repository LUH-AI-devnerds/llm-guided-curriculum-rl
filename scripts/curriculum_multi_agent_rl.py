import os
import argparse
from MultiAgentCurriculumSystem import (
    MultiAgentCurriculumSystem,
)
from MultiAgentStandardSystem import MultiAgentStandardSystem

"""
Enhanced Win Counting System

This module implements an enhanced win counting system that properly accounts for:
1. Split hands - Multiple hands can be won in a single episode
2. Double down scenarios - Bet amounts are doubled, affecting win value
3. Different bet amounts - Win counting is proportional to bet size

Win counting logic:
- Each winning hand counts as 1 win (or 2 wins if doubled down)
- Split scenarios can result in multiple wins per episode
- Win rate is calculated as: total_wins / total_episodes
- This provides more accurate training evaluation for RL agents
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-curriculum",
        action="store_true",
        help="Run standard RL without curriculum",
    )
    parser.add_argument("--num-agents", type=int, default=2, help="Number of agents")
    parser.add_argument(
        "--agent-types",
        nargs="*",
        default=["dqn", "tabular"],
        help="List of agent types (dqn/tabular)",
    )
    parser.add_argument(
        "--deck-type",
        type=str,
        default="infinite",
        help="Deck type (infinite, 1-deck, 6-deck, 8-deck)",
    )
    parser.add_argument(
        "--penetration",
        type=float,
        default=0.9,
        help="Deck penetration for reshuffling",
    )
    parser.add_argument(
        "--budget", type=int, default=10000, help="Starting budget for each agent"
    )
    parser.add_argument(
        "--no-dynamic-rewards",
        action="store_true",
        help="Disable dynamic reward scaling (use simple rewards)",
    )
    parser.add_argument(
        "--reward-type",
        type=str,
        default="simple",
        choices=["simple", "conservative_dynamic", "win_focused", "balanced"],
        help="Reward scheme to use",
    )
    parser.add_argument(
        "--episodes", type=int, default=100000, help="Total training episodes"
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=1000, help="Evaluation episodes"
    )
    args = parser.parse_args()

    API_KEY = os.getenv("GOOGLE_AI_API_KEY")

    if args.no_curriculum:
        print("\n🚀 STARTING STANDARD MULTI-AGENT RL TRAINING")
        system = MultiAgentStandardSystem(
            num_agents=args.num_agents,
            agent_types=args.agent_types,
            deck_type=args.deck_type,
            penetration=args.penetration,
            budget=args.budget,
            use_dynamic_rewards=not args.no_dynamic_rewards,
            reward_type=args.reward_type,
        )
        system.train(total_episodes=args.episodes, eval_episodes=args.eval_episodes)
    else:
        if API_KEY == "your_api_key_here" or not API_KEY:
            print(
                "⚠️  Please set your Google AI API key in the API_KEY variable or environment"
            )
            print("You can get an API key from: https://ai.google.dev/")
            exit(1)
        print("\n🚀 STARTING MULTI-AGENT CURRICULUM LEARNING")
        curriculum_system = MultiAgentCurriculumSystem(
            llm_api_key=API_KEY,
            num_agents=args.num_agents,
            agent_types=args.agent_types,
            deck_type=args.deck_type,
            penetration=args.penetration,
            budget=args.budget,
            use_dynamic_rewards=not args.no_dynamic_rewards,
            reward_type=args.reward_type,
        )
        final_report = curriculum_system.train_multi_agent_curriculum(
            total_episodes=args.episodes, eval_episodes=args.eval_episodes
        )
        curriculum_system.save_agents()
        curriculum_system.create_run_summary()
        print("\n✅ CURRICULUM LEARNING COMPLETE!")
        print(f"📁 All logs and models saved to: {curriculum_system.log_dir}")
        print("Check the generated JSON report for detailed results.")
