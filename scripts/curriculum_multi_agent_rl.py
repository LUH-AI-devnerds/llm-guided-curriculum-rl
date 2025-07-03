import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import pickle
import json
import os
from datetime import datetime
from LLM import LLM
from BlackJackENV import BlackjackEnv
from RLAgent import DQNAgent, QLearningAgent


class CurriculumStage:
    """Defines a curriculum stage with available actions and learning objectives."""

    def __init__(
        self,
        stage_id,
        name,
        available_actions,
        description,
        difficulty,
        success_threshold=0.4,
    ):
        self.stage_id = stage_id
        self.name = name
        self.available_actions = available_actions  # List of action indices
        self.description = description
        self.difficulty = difficulty  # 1-5 scale
        self.success_threshold = success_threshold  # Win rate needed to advance

    def to_dict(self):
        return {
            "stage_id": self.stage_id,
            "name": self.name,
            "available_actions": self.available_actions,
            "description": self.description,
            "difficulty": self.difficulty,
            "success_threshold": self.success_threshold,
        }


class LLMGuidedCurriculum:
    """Uses LLM to design and adapt curriculum stages based on agent performance."""

    def __init__(self, api_key, action_descriptions=None):
        self.llm = LLM(api_key)
        self.action_descriptions = action_descriptions or {
            0: "Stand - Stay with current hand",
            1: "Hit - Draw another card",
            2: "Double Down - Double bet and draw one card",
            3: "Split - Split pair into two hands",
        }
        self.curriculum_history = []

    def generate_curriculum_stages(self, num_stages=4):
        """Generate curriculum stages using LLM guidance."""

        prompt = f"""
        You are an expert in reinforcement learning curriculum design for Blackjack. 
        
        Available actions:
        {self.action_descriptions}
        
        Design {num_stages} progressive curriculum stages for training RL agents in Blackjack.
        Each stage should gradually increase complexity and introduce new actions.
        
        For each stage, specify:
        1. Stage name
        2. Available actions (list of action indices: 0, 1, 2, 3)
        3. Description of learning objectives
        4. Difficulty level (1-5 scale)
        5. Success threshold (win rate 0.0-1.0 to advance)
        
        Respond in this exact JSON format:
        {{
            "stages": [
                {{
                    "stage_id": 1,
                    "name": "Stage Name",
                    "available_actions": [0, 1],
                    "description": "Learning objectives...",
                    "difficulty": 1,
                    "success_threshold": 0.35
                }},
                ...
            ],
            "rationale": "Explanation of curriculum design..."
        }}
        
        Focus on progressive skill building and realistic success thresholds.
        """

        try:
            response = self.llm.generate_response(prompt)
            # Extract JSON from response
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            json_str = response[start_idx:end_idx]

            curriculum_data = json.loads(json_str)
            stages = []

            for stage_data in curriculum_data["stages"]:
                stage = CurriculumStage(
                    stage_id=stage_data["stage_id"],
                    name=stage_data["name"],
                    available_actions=stage_data["available_actions"],
                    description=stage_data["description"],
                    difficulty=stage_data["difficulty"],
                    success_threshold=stage_data["success_threshold"],
                )
                stages.append(stage)

            print(f"LLM Generated Curriculum with {len(stages)} stages:")
            print(f"Rationale: {curriculum_data.get('rationale', 'Not provided')}")

            return stages

        except Exception as e:
            print(f"LLM curriculum generation failed: {e}")
            # Fallback to default curriculum
            return self._default_curriculum_stages()

    def adapt_curriculum(self, agent_performance, current_stage, stages):
        """Use LLM to adapt curriculum based on agent performance."""

        performance_summary = {
            "current_stage": current_stage.name,
            "win_rate": agent_performance.get("win_rate", 0),
            "average_reward": agent_performance.get("avg_reward", 0),
            "episodes_trained": agent_performance.get("episodes", 0),
            "struggling_actions": agent_performance.get("poor_actions", []),
        }

        prompt = f"""
        You are analyzing an RL agent's performance in Blackjack curriculum learning.
        
        Current Stage: {current_stage.name}
        Available Actions: {current_stage.available_actions}
        Stage Description: {current_stage.description}
        Success Threshold: {current_stage.success_threshold}
        
        Agent Performance:
        - Win Rate: {performance_summary['win_rate']:.3f}
        - Average Reward: {performance_summary['average_reward']:.3f}
        - Episodes Trained: {performance_summary['episodes_trained']}
        - Struggling with actions: {performance_summary['struggling_actions']}
        
        Action Descriptions:
        {self.action_descriptions}
        
        Based on this performance, recommend:
        1. Should the agent advance to next stage? (yes/no)
        2. What actions should be emphasized in next training episodes?
        3. Any curriculum modifications needed?
        
        Respond in JSON format:
        {{
            "advance_stage": true/false,
            "recommended_actions": [list of action indices to focus on],
            "curriculum_modifications": {{
                "adjust_threshold": 0.XX,
                "add_actions": [list],
                "remove_actions": [list]
            }},
            "reasoning": "Explanation of recommendations..."
        }}
        """

        try:
            response = self.llm.generate_response(prompt)
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            json_str = response[start_idx:end_idx]

            recommendations = json.loads(json_str)
            print(f"LLM Curriculum Adaptation: {recommendations.get('reasoning', '')}")

            return recommendations

        except Exception as e:
            print(f"LLM curriculum adaptation failed: {e}")
            # Fallback decision
            return {
                "advance_stage": agent_performance.get("win_rate", 0)
                >= current_stage.success_threshold,
                "recommended_actions": current_stage.available_actions,
                "curriculum_modifications": {},
                "reasoning": "Fallback decision due to LLM error",
            }

    def _default_curriculum_stages(self):
        """Fallback curriculum if LLM fails."""
        return [
            CurriculumStage(
                1, "Basic Play", [0, 1], "Learn basic stand/hit decisions", 1, 0.35
            ),
            CurriculumStage(
                2, "Strategic Play", [0, 1], "Master basic strategy", 2, 0.40
            ),
            CurriculumStage(
                3, "Advanced Betting", [0, 1, 2], "Learn double down strategy", 3, 0.42
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


class MultiAgentCurriculumSystem:
    """Manages multiple RL agents learning through curriculum stages."""

    def __init__(self, llm_api_key, num_agents=3, agent_types=None):
        self.llm_curriculum = LLMGuidedCurriculum(llm_api_key)
        self.num_agents = num_agents
        self.agent_types = agent_types or ["dqn", "tabular", "dqn"]

        # Initialize agents
        self.agents = []
        for i, agent_type in enumerate(self.agent_types):
            if agent_type == "dqn":
                agent = DQNAgent(
                    action_space=[0, 1, 2, 3],
                    learning_rate=0.001,
                    exploration_rate=1.0,
                    exploration_decay=0.9995,
                )
            else:  # tabular
                agent = QLearningAgent(
                    action_space=[0, 1, 2, 3],
                    learning_rate=0.1,
                    exploration_rate=1.0,
                    exploration_decay=0.999,
                )

            agent.agent_id = i
            agent.agent_type = agent_type
            agent.current_stage = 0
            agent.stage_performance = []
            self.agents.append(agent)

        # Generate curriculum
        self.curriculum_stages = self.llm_curriculum.generate_curriculum_stages()
        self.global_performance_log = []

    def train_multi_agent_curriculum(self, total_episodes=50000, eval_episodes=1000):
        """Train multiple agents through curriculum stages."""

        print(f"\n🎓 MULTI-AGENT CURRICULUM LEARNING")
        print("=" * 60)
        print(f"Agents: {self.num_agents} ({', '.join(self.agent_types)})")
        print(f"Curriculum Stages: {len(self.curriculum_stages)}")
        print(f"Total Episodes: {total_episodes}")

        for stage_idx, stage in enumerate(self.curriculum_stages):
            print(f"\n📚 STAGE {stage.stage_id}: {stage.name}")
            print(f"Available Actions: {stage.available_actions}")
            print(f"Description: {stage.description}")
            print(f"Success Threshold: {stage.success_threshold:.3f}")
            print("-" * 50)

            stage_results = {}

            for agent_idx, agent in enumerate(self.agents):
                if agent.current_stage <= stage_idx:
                    print(
                        f"\n🤖 Training Agent {agent_idx} ({agent.agent_type.upper()})"
                    )

                    # Create environment for current stage
                    env = CurriculumBlackjackEnv(stage)

                    # Train agent on current stage
                    stage_episodes = total_episodes // len(self.curriculum_stages)
                    agent_performance = self._train_agent_on_stage(
                        agent, env, stage, stage_episodes, eval_episodes
                    )

                    # Get LLM recommendations for this agent
                    recommendations = self.llm_curriculum.adapt_curriculum(
                        agent_performance, stage, self.curriculum_stages
                    )

                    # Apply LLM-guided action focus
                    if "recommended_actions" in recommendations:
                        self._apply_action_focus(
                            agent, recommendations["recommended_actions"]
                        )

                    # Check if agent should advance
                    if recommendations.get("advance_stage", False):
                        agent.current_stage += 1
                        print(f"✅ Agent {agent_idx} advanced to next stage!")
                    else:
                        print(
                            f"🔄 Agent {agent_idx} needs more training on current stage"
                        )

                    stage_results[f"agent_{agent_idx}"] = agent_performance
                    agent.stage_performance.append(agent_performance)

            self.global_performance_log.append(
                {
                    "stage": stage.to_dict(),
                    "results": stage_results,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        return self._generate_final_report()

    def _train_agent_on_stage(self, agent, env, stage, episodes, eval_episodes):
        """Train a single agent on a curriculum stage."""

        # Training phase
        episode_rewards = []
        wins = 0

        for episode in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0

            while not done:
                # Get LLM-guided action selection during training
                if random.random() < 0.1:  # 10% of time, use LLM guidance
                    action = self._get_llm_guided_action(agent, state, stage)
                else:
                    action = agent.get_action(state)

                next_state, reward, done = env.step(action)

                # Update agent (different for DQN vs tabular)
                if hasattr(agent, "remember"):  # DQN agent
                    agent.remember(state, action, reward, next_state, done)
                    agent.replay()
                else:  # Tabular agent
                    agent.update(state, action, reward, next_state)

                state = next_state
                total_reward += reward

            episode_rewards.append(total_reward)
            if total_reward > 0:
                wins += 1

            agent.decay_epsilon()

            if episode % 1000 == 0:
                recent_win_rate = wins / min(episode + 1, 1000)
                print(
                    f"  Episode {episode}: Win Rate: {recent_win_rate:.3f}, "
                    f"Epsilon: {agent.epsilon:.4f}"
                )
                wins = max(0, wins - (wins // 10))  # Decay win counter

        # Evaluation phase
        evaluation_results = self._evaluate_agent(agent, env, eval_episodes)

        return {
            "win_rate": evaluation_results["win_rate"],
            "avg_reward": evaluation_results["avg_reward"],
            "episodes": episodes,
            "stage_id": stage.stage_id,
            "agent_type": agent.agent_type,
            "poor_actions": evaluation_results.get("poor_actions", []),
        }

    def _get_llm_guided_action(self, agent, state, stage):
        """Get LLM guidance for action selection during training."""

        valid_actions = self._get_valid_actions_for_stage(state, stage)

        # Convert state to human-readable format
        player_sum, dealer_up, has_ace, can_split, can_double, is_blackjack = state

        state_description = f"Player sum: {player_sum}, Dealer up card: {dealer_up}"
        if has_ace:
            state_description += ", has usable ace"
        if can_split:
            state_description += ", can split"
        if can_double:
            state_description += ", can double down"
        if is_blackjack:
            state_description += ", has blackjack"

        prompt = f"""
        You are a Blackjack expert helping an RL agent learn optimal play.
        
        Current situation: {state_description}
        Available actions in this curriculum stage: {[self.llm_curriculum.action_descriptions[a] for a in valid_actions]}
        Stage objective: {stage.description}
        
        What is the best action for learning purposes in this curriculum stage?
        Consider both optimal play and educational value for the agent.
        
        Respond with just the action index (0, 1, 2, or 3):
        """

        try:
            response = self.llm_curriculum.llm.generate_response(prompt)
            # Extract number from response
            action = int("".join(filter(str.isdigit, response))[:1])
            if action in valid_actions:
                return action
        except:
            pass

        # Fallback to agent's own decision
        return agent.get_action(state)

    def _get_valid_actions_for_stage(self, state, stage):
        """Get valid actions for current state within curriculum stage constraints."""

        # Get all valid actions from environment
        player_sum, dealer_up, has_ace, can_split, can_double, is_blackjack = state
        valid_actions = [0]  # Stand always valid

        if player_sum < 21 and not is_blackjack:
            valid_actions.append(1)  # Hit
        if can_double:
            valid_actions.append(2)  # Double
        if can_split:
            valid_actions.append(3)  # Split

        # Filter by curriculum stage
        stage_valid_actions = [a for a in valid_actions if a in stage.available_actions]

        return stage_valid_actions if stage_valid_actions else [0]  # Always allow stand

    def _apply_action_focus(self, agent, recommended_actions):
        """Apply LLM recommendations to focus training on specific actions."""

        # Temporarily adjust exploration to favor recommended actions
        if hasattr(agent, "action_focus_weight"):
            agent.action_focus_weight = {}
        else:
            agent.action_focus_weight = {}

        for action in recommended_actions:
            agent.action_focus_weight[action] = 1.5  # Increase probability

        print(f"  🎯 Focusing on actions: {recommended_actions}")

    def _evaluate_agent(self, agent, env, episodes):
        """Evaluate agent performance and identify weak actions."""

        original_epsilon = agent.epsilon
        agent.epsilon = 0.0  # Pure exploitation

        total_rewards = []
        wins = 0
        action_rewards = {0: [], 1: [], 2: [], 3: []}

        for _ in range(episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            episode_actions = []

            while not done:
                action = agent.get_action(state)
                episode_actions.append(action)
                state, reward, done = env.step(action)
                episode_reward += reward

            total_rewards.append(episode_reward)
            if episode_reward > 0:
                wins += 1

            # Track action performance
            for action in episode_actions:
                action_rewards[action].append(episode_reward)

        agent.epsilon = original_epsilon

        # Identify poor performing actions
        poor_actions = []
        for action, rewards in action_rewards.items():
            if rewards and np.mean(rewards) < -0.2:  # Threshold for poor performance
                poor_actions.append(action)

        return {
            "win_rate": wins / episodes,
            "avg_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
            "poor_actions": poor_actions,
        }

    def _generate_final_report(self):
        """Generate comprehensive training report."""

        report = {
            "training_summary": {
                "total_agents": self.num_agents,
                "agent_types": self.agent_types,
                "curriculum_stages": len(self.curriculum_stages),
                "completion_time": datetime.now().isoformat(),
            },
            "curriculum_stages": [stage.to_dict() for stage in self.curriculum_stages],
            "agent_performance": [],
            "global_performance_log": self.global_performance_log,
        }

        # Summarize each agent's final performance
        for agent in self.agents:
            final_performance = (
                agent.stage_performance[-1] if agent.stage_performance else {}
            )
            report["agent_performance"].append(
                {
                    "agent_id": agent.agent_id,
                    "agent_type": agent.agent_type,
                    "final_stage": agent.current_stage,
                    "final_win_rate": final_performance.get("win_rate", 0),
                    "stage_progression": len(agent.stage_performance),
                }
            )

        # Save report
        with open(
            f'curriculum_training_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json',
            "w",
        ) as f:
            json.dump(report, f, indent=2)

        print(f"\n📊 FINAL TRAINING REPORT")
        print("=" * 50)
        for i, agent_perf in enumerate(report["agent_performance"]):
            print(
                f"Agent {i} ({agent_perf['agent_type'].upper()}): "
                f"Stage {agent_perf['final_stage']}, "
                f"Win Rate: {agent_perf['final_win_rate']:.3f}"
            )

        return report

    def save_agents(self, prefix="curriculum_agent"):
        """Save all trained agents."""
        for i, agent in enumerate(self.agents):
            filename = f"{prefix}_{agent.agent_type}_{i}"
            if agent.agent_type == "dqn":
                agent.save_model(f"{filename}.pth")
            else:
                agent.save_model(f"{filename}.pkl")
            print(f"Saved {filename}")


class CurriculumBlackjackEnv(BlackjackEnv):
    """Extended BlackjackEnv that enforces curriculum stage constraints."""

    def __init__(self, curriculum_stage):
        super().__init__(curriculum_stage=curriculum_stage.stage_id)
        self.stage = curriculum_stage

    def step(self, action):
        """Override step to enforce curriculum stage action constraints."""

        # Check if action is allowed in current curriculum stage
        if action not in self.stage.available_actions:
            # Force to stand if action not allowed
            action = 0

        return super().step(action)


if __name__ == "__main__":
    # Set your Google AI API key here
    API_KEY = "your_api_key_here"  # Replace with actual API key

    if API_KEY == "your_api_key_here":
        print("⚠️  Please set your Google AI API key in the API_KEY variable")
        print("You can get an API key from: https://ai.google.dev/")
        exit(1)

    print("🚀 STARTING MULTI-AGENT CURRICULUM LEARNING")
    print("=" * 60)

    # Initialize multi-agent curriculum system
    curriculum_system = MultiAgentCurriculumSystem(
        llm_api_key=API_KEY,
        num_agents=3,
        agent_types=["dqn", "tabular", "dqn"],  # Mix of agent types
    )

    # Train agents through curriculum
    final_report = curriculum_system.train_multi_agent_curriculum(
        total_episodes=40000, eval_episodes=1000  # Reduced for demo
    )

    # Save trained agents
    curriculum_system.save_agents()

    print("\n✅ CURRICULUM LEARNING COMPLETE!")
    print("Check the generated JSON report for detailed results.")
