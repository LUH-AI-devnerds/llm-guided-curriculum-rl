#!/usr/bin/env python3
"""
Demo script for LLM-Guided Curriculum Learning in Blackjack RL

This script demonstrates:
1. Multi-agent curriculum learning
2. LLM-guided action selection
3. Progressive skill development
4. Performance comparison between agents
"""

import os
import sys
from datetime import datetime

# Add scripts directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from RLAgent import DQNAgent, QLearningAgent
from BlackJackENV import BlackjackEnv
from curriculum_multi_agent_rl import (
    MultiAgentCurriculumSystem,
    CurriculumStage,
    LLMGuidedCurriculum,
)


def run_demo_curriculum():
    """Run a demonstration of curriculum learning without LLM."""

    print("🎓 DEMO: Curriculum Learning in Blackjack RL")
    print("=" * 60)
    print("This demo shows how agents learn progressively through curriculum stages.")
    print()

    # Define curriculum stages manually (no LLM needed)
    stages = [
        CurriculumStage(
            1, "Basic Play", [0, 1], "Learn basic stand/hit decisions", 1, 0.35
        ),
        CurriculumStage(
            2, "Strategic Play", [0, 1], "Master timing of hits and stands", 2, 0.40
        ),
        CurriculumStage(
            3, "Advanced Betting", [0, 1, 2], "Learn when to double down", 3, 0.42
        ),
        CurriculumStage(
            4,
            "Expert Play",
            [0, 1, 2, 3],
            "Master all actions including splits",
            4,
            0.45,
        ),
    ]

    # Create agents
    agents = [
        DQNAgent(action_space=[0, 1, 2, 3], learning_rate=0.001),
        QLearningAgent(action_space=[0, 1, 2, 3], learning_rate=0.1),
    ]

    # Set agent properties
    for i, agent in enumerate(agents):
        agent.agent_id = i
        agent.agent_type = "dqn" if isinstance(agent, DQNAgent) else "tabular"
        agent.current_stage = 0
        agent.stage_performance = []

    print(f"Training {len(agents)} agents through {len(stages)} curriculum stages:")
    print("- Agent 0: DQN (Neural Network)")
    print("- Agent 1: Q-Learning (Tabular)")
    print()

    # Train through curriculum stages
    for stage_idx, stage in enumerate(stages):
        print(f"📚 STAGE {stage.stage_id}: {stage.name}")
        print(f"   Available Actions: {stage.available_actions}")
        print(f"   Learning Objective: {stage.description}")
        print(f"   Success Threshold: {stage.success_threshold}")
        print("-" * 50)

        for agent in agents:
            if agent.current_stage <= stage_idx:
                print(
                    f"🤖 Training Agent {agent.agent_id} ({agent.agent_type.upper()})"
                )

                # Create curriculum-constrained environment
                env = CurriculumBlackjackEnv(stage)

                # Train agent on current stage
                episodes = 8000  # Reduced for demo
                stage_performance = train_agent_on_stage(agent, env, stage, episodes)

                # Check advancement
                if stage_performance["win_rate"] >= stage.success_threshold:
                    agent.current_stage += 1
                    print(
                        f"   ✅ Advanced! Win Rate: {stage_performance['win_rate']:.3f}"
                    )
                else:
                    print(
                        f"   🔄 Needs more training. Win Rate: {stage_performance['win_rate']:.3f}"
                    )

                agent.stage_performance.append(stage_performance)

        print()

    # Final evaluation
    print("📊 FINAL RESULTS")
    print("=" * 40)

    final_env = BlackjackEnv(curriculum_stage=4)  # Full game

    for agent in agents:
        final_win_rate = agent.evaluate(final_env, episodes=5000)
        final_stage = agent.current_stage
        stages_completed = len(agent.stage_performance)

        print(f"Agent {agent.agent_id} ({agent.agent_type.upper()}):")
        print(f"  Final Win Rate: {final_win_rate:.2f}%")
        print(f"  Stages Completed: {stages_completed}/{len(stages)}")
        print(f"  Current Stage: {final_stage}")
        print()

    return agents


def train_agent_on_stage(agent, env, stage, episodes):
    """Train an agent on a specific curriculum stage."""

    wins = 0
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)

            # Update agent based on type
            if hasattr(agent, "remember"):  # DQN
                agent.remember(state, action, reward, next_state, done)
                agent.replay()
            else:  # Tabular
                agent.update(state, action, reward, next_state)

            state = next_state
            episode_reward += reward

        total_rewards.append(episode_reward)
        if episode_reward > 0:
            wins += 1

        agent.decay_epsilon()

        # Progress update
        if episode % 2000 == 0 and episode > 0:
            current_win_rate = wins / episode
            print(
                f"   Episode {episode}: Win Rate {current_win_rate:.3f}, Epsilon {agent.epsilon:.3f}"
            )

    # Evaluate performance
    win_rate = agent.evaluate(env, episodes=1000)

    return {
        "win_rate": win_rate / 100,  # Convert percentage to decimal
        "avg_reward": sum(total_rewards) / len(total_rewards),
        "episodes": episodes,
        "stage_id": stage.stage_id,
    }


class CurriculumBlackjackEnv(BlackjackEnv):
    """Environment that enforces curriculum stage constraints."""

    def __init__(self, curriculum_stage):
        super().__init__(curriculum_stage=curriculum_stage.stage_id)
        self.stage = curriculum_stage

    def step(self, action):
        """Override step to enforce curriculum constraints."""
        # Force action to be valid for current stage
        if action not in self.stage.available_actions:
            action = 0  # Default to stand

        return super().step(action)


def demonstrate_llm_guidance():
    """Show how LLM guidance would work (simulation)."""

    print("🧠 LLM GUIDANCE SIMULATION")
    print("=" * 40)
    print("This demonstrates how the LLM would guide action selection:")
    print()

    # Sample scenarios
    scenarios = [
        {
            "state": (15, 10, False, False, True, False),
            "stage": "Basic Play",
            "actions": [0, 1],
            "description": "Player: 15, Dealer: 10, can double",
        },
        {
            "state": (11, 6, False, False, True, False),
            "stage": "Advanced Betting",
            "actions": [0, 1, 2],
            "description": "Player: 11, Dealer: 6, can double",
        },
        {
            "state": (20, 9, False, True, False, False),
            "stage": "Expert Play",
            "actions": [0, 1, 2, 3],
            "description": "Player: 20 (pair), Dealer: 9, can split",
        },
    ]

    for i, scenario in enumerate(scenarios, 1):
        print(f"Scenario {i}: {scenario['description']}")
        print(f"Stage: {scenario['stage']}")
        print(f"Available Actions: {scenario['actions']}")

        # Simulate LLM recommendation
        if scenario["stage"] == "Basic Play":
            recommendation = 1 if scenario["state"][0] < 17 else 0
            reason = "Learn basic strategy - hit on soft totals"
        elif scenario["stage"] == "Advanced Betting":
            recommendation = (
                2 if scenario["state"][0] == 11 and scenario["state"][1] <= 10 else 1
            )
            reason = "Perfect double down opportunity - maximize value"
        else:  # Expert Play
            if scenario["state"][0] == 20:
                recommendation = 0
                reason = "Never split 10s - strong hand already"
            else:
                recommendation = 1
                reason = "Continue learning when to hit"

        action_names = {0: "Stand", 1: "Hit", 2: "Double", 3: "Split"}
        print(f"LLM Recommendation: {action_names[recommendation]} ({recommendation})")
        print(f"Reasoning: {reason}")
        print()


def show_curriculum_progression():
    """Visualize how curriculum stages build upon each other."""

    print("🎯 CURRICULUM PROGRESSION")
    print("=" * 50)
    print("Each stage builds upon the previous one:\n")

    progression = [
        {
            "stage": 1,
            "name": "Basic Play",
            "actions": ["Stand", "Hit"],
            "focus": "Learn fundamental decision making",
            "key_concepts": ["Card counting basics", "Risk assessment", "When to stop"],
        },
        {
            "stage": 2,
            "name": "Strategic Play",
            "actions": ["Stand", "Hit"],
            "focus": "Master timing and basic strategy",
            "key_concepts": [
                "Dealer up-card analysis",
                "Soft vs hard hands",
                "Probability awareness",
            ],
        },
        {
            "stage": 3,
            "name": "Advanced Betting",
            "actions": ["Stand", "Hit", "Double Down"],
            "focus": "Learn when to increase stakes",
            "key_concepts": [
                "Double down strategy",
                "Bankroll management",
                "Risk vs reward",
            ],
        },
        {
            "stage": 4,
            "name": "Expert Play",
            "actions": ["Stand", "Hit", "Double Down", "Split"],
            "focus": "Master all advanced techniques",
            "key_concepts": [
                "Split strategy",
                "Complex scenarios",
                "Multi-hand management",
            ],
        },
    ]

    for stage in progression:
        print(f"Stage {stage['stage']}: {stage['name']}")
        print(f"  Actions: {', '.join(stage['actions'])}")
        print(f"  Focus: {stage['focus']}")
        print(f"  Key Concepts: {', '.join(stage['key_concepts'])}")
        print()


if __name__ == "__main__":
    print("🎲 BLACKJACK RL CURRICULUM LEARNING DEMO")
    print("=" * 60)
    print("This demo showcases curriculum learning without requiring an API key.")
    print()

    # Show curriculum structure
    show_curriculum_progression()

    # Demonstrate LLM guidance concept
    demonstrate_llm_guidance()

    # Run actual curriculum training demo
    print("🚀 STARTING CURRICULUM TRAINING DEMO")
    print("=" * 50)
    trained_agents = run_demo_curriculum()

    print("✅ DEMO COMPLETE!")
    print()
    print("Key Features Demonstrated:")
    print("✓ Progressive curriculum stages")
    print("✓ Multi-agent learning")
    print("✓ Action constraint enforcement")
    print("✓ Performance-based advancement")
    print("✓ Agent type comparison (DQN vs Tabular)")
    print()
    print("With an LLM API key, the system would also provide:")
    print("• Dynamic curriculum generation")
    print("• Intelligent action recommendations")
    print("• Adaptive stage progression")
    print("• Real-time strategy guidance")
