import os
import argparse
from MultiAgentCurriculumSystem import (
    MultiAgentCurriculumSystem,
)
from MultiAgentStandardSystem import MultiAgentStandardSystem

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
        "--episodes", type=int, default=100000, help="Total training episodes"
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=2000, help="Evaluation episodes"
    )
    parser.add_argument(
        "--max-episodes-per-stage",
        type=int,
        default=20000,
        help="Maximum episodes per stage before forcing advancement",
    )
    parser.add_argument(
        "--stage-epsilon-reset",
        action="store_true",
        help="Reset epsilon when advancing to new curriculum stages",
    )
    parser.add_argument(
        "--progressive-episodes",
        action="store_true",
        default=True,
        help="Use progressive episode allocation (more episodes for later stages)",
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
        )
        final_report = curriculum_system.train_multi_agent_curriculum(
            total_episodes=args.episodes,
            eval_episodes=args.eval_episodes,
            max_episodes_per_stage=args.max_episodes_per_stage,
        )
        curriculum_system.save_agents()
        curriculum_system.create_run_summary()
        print("\n✅ CURRICULUM LEARNING COMPLETE!")
        print(f"📁 All logs and models saved to: {curriculum_system.log_dir}")
        print("Check the generated JSON report for detailed results.")
