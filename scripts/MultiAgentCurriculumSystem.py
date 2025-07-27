import os
import json
import time
from datetime import datetime
import numpy as np
from LLMGuidedCurriculum import LLMGuidedCurriculum
from RLAgent import DQNAgent, QLearningAgent
from BlackJackENV import BlackjackEnv


class CurriculumBlackjackEnv(BlackjackEnv):
    def __init__(
        self,
        curriculum_stage,
        deck_type="infinite",
        penetration=0.75,
    ):
        super().__init__(
            curriculum_stage=curriculum_stage.stage_id,
            deck_type=deck_type,
            penetration=penetration,
        )
        self.stage = curriculum_stage

    def step(self, action):
        if action not in self.stage.available_actions:
            print(
                f"⚠️  Curriculum constraint: Action {action} not allowed in stage {self.stage.stage_id}. Forcing stand."
            )
            action = 0
        elif not self._is_valid_action(action):
            action = 0

        return super().step(action)


class MultiAgentCurriculumSystem:
    def __init__(
        self,
        llm_api_key,
        num_agents=3,
        agent_types=None,
        deck_type="infinite",
        penetration=0.9,
    ):
        self.llm_curriculum = LLMGuidedCurriculum(llm_api_key)
        self.num_agents = num_agents
        self.agent_types = agent_types or ["dqn", "tabular", "dqn"]
        self.deck_type = deck_type
        self.penetration = penetration

        self.setup_logging_directory(deck_type, penetration)

        self.agents = []
        for i, agent_type in enumerate(self.agent_types):
            if agent_type == "dqn":
                agent = DQNAgent(
                    action_space=[0, 1, 2, 3, 4, 5],
                    learning_rate=0.0005,
                    exploration_rate=1.0,
                    exploration_decay=0.99995,
                    memory_size=100000,
                    batch_size=128,
                )
            else:
                agent = QLearningAgent(
                    action_space=[0, 1, 2, 3, 4, 5],
                    learning_rate=0.1,
                    exploration_rate=1.0,
                    exploration_decay=0.9999,
                )

            agent.agent_id = i
            agent.agent_type = agent_type
            agent.current_stage = 0
            agent.stage_performance = []
            self.agents.append(agent)

        self.curriculum_stages = self.llm_curriculum.generate_curriculum_stages()
        self.global_performance_log = []

    def setup_logging_directory(self, deck_type, penetration):
        if not os.path.exists("logs"):
            os.makedirs("logs")

        date_str = datetime.now().strftime("%Y%m%d")
        self.log_dir = f"logs/logs-{date_str}-{deck_type}-{penetration}"

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.eval_log_dir = os.path.join(self.log_dir, "evaluation")
        self.training_log_dir = os.path.join(self.log_dir, "training")
        self.report_log_dir = os.path.join(self.log_dir, "reports")

        for subdir in [self.eval_log_dir, self.training_log_dir, self.report_log_dir]:
            if not os.path.exists(subdir):
                os.makedirs(subdir)

        print(f"📁 Logging directory setup: {self.log_dir}")
        print(f"  - Evaluation logs: {self.eval_log_dir}")
        print(f"  - Training logs: {self.training_log_dir}")
        print(f"  - Report logs: {self.report_log_dir}")

    def train_multi_agent_curriculum(
        self, total_episodes=50000, eval_episodes=1000, max_episodes_per_stage=20000
    ):
        print(f"\n🎓 MULTI-AGENT CURRICULUM LEARNING")
        print("=" * 60)
        print(f"Agents: {self.num_agents} ({', '.join(self.agent_types)})")
        print(f"Curriculum Stages: {len(self.curriculum_stages)}")
        print(f"Total Episodes: {total_episodes}")
        print(f"Max Episodes per Stage: {max_episodes_per_stage}")

        stage_idx = 0
        agent_stage_attempts = {
            agent.agent_id: {stage.stage_id: 0 for stage in self.curriculum_stages}
            for agent in self.agents
        }

        while stage_idx < len(self.curriculum_stages):
            stage = self.curriculum_stages[stage_idx]
            print(f"\n📚 STAGE {stage.stage_id}: {stage.name}")
            print(f"Available Actions: {stage.available_actions}")
            print(f"Description: {stage.description}")
            print(f"Success Threshold: {stage.success_threshold:.3f}")
            print("-" * 50)

            stage_results = {}

            for agent_idx, agent in enumerate(self.agents):
                if agent.current_stage == stage_idx:
                    current_attempts = agent_stage_attempts[agent.agent_id][
                        stage.stage_id
                    ]
                    if current_attempts >= max_episodes_per_stage:
                        print(
                            f"⚠️  Agent {agent_idx} reached maximum training attempts ({max_episodes_per_stage}) for Stage {stage.stage_id}"
                        )
                        print(f"   📊 Forcing advancement to next stage...")
                        agent.current_stage += 1
                        continue
                    print(
                        f"\n🤖 Training Agent {agent_idx} ({agent.agent_type.upper()})"
                    )

                    env = CurriculumBlackjackEnv(
                        stage,
                        deck_type=self.deck_type,
                        penetration=self.penetration,
                    )

                    agent.set_curriculum_stage(stage)

                    if stage.stage_id > 1 and hasattr(agent, "stage_models"):
                        self._load_previous_stage_strategies(agent, stage)

                    print(
                        f"  🔍 Agent {agent_idx} curriculum stage set to: {agent.curriculum_stage.stage_id if agent.curriculum_stage else 'None'}"
                    )
                    print(
                        f"  🔍 Available actions for agent: {agent.curriculum_stage.available_actions if agent.curriculum_stage else 'None'}"
                    )

                    base_episodes = total_episodes // len(self.curriculum_stages)
                    stage_multiplier = 1.0 + (stage.stage_id - 1) * 0.2
                    stage_episodes = int(base_episodes * stage_multiplier)

                    print(
                        f"  📊 Stage {stage.stage_id} allocated {stage_episodes} episodes (multiplier: {stage_multiplier:.1f})"
                    )

                    agent_performance = self._train_agent_on_stage(
                        agent, env, stage, stage_episodes, eval_episodes
                    )

                    agent_stage_attempts[agent.agent_id][
                        stage.stage_id
                    ] += stage_episodes

                    recommendations = self.llm_curriculum.adapt_curriculum(
                        agent_performance, stage, self.curriculum_stages
                    )

                    if "recommended_actions" in recommendations:
                        self._apply_action_focus(
                            agent, recommendations["recommended_actions"]
                        )

                    episode_win_rate = agent_performance.get("win_rate", 0)
                    current_attempts = agent_stage_attempts[agent.agent_id][
                        stage.stage_id
                    ]

                    # Simplified advancement logic
                    threshold_met = episode_win_rate >= stage.success_threshold
                    max_attempts_reached = current_attempts >= max_episodes_per_stage

                    # Advance if threshold met OR max attempts reached
                    should_advance = threshold_met or max_attempts_reached

                    if should_advance:
                        agent.current_stage += 1
                        if threshold_met:
                            print(f"✅ Agent {agent_idx} advanced to next stage!")
                            print(
                                f"   📊 Win Rate: {episode_win_rate:.3f} >= {stage.success_threshold}"
                            )
                        else:
                            print(
                                f"⚠️  Agent {agent_idx} forced to advance (max attempts reached)"
                            )
                            print(
                                f"   📊 Win Rate: {episode_win_rate:.3f} < {stage.success_threshold}"
                            )
                        print(
                            f"   📊 Total Attempts: {current_attempts}/{max_episodes_per_stage}"
                        )

                        self._preserve_learned_strategies(
                            agent, stage, agent_performance
                        )
                    else:
                        print(
                            f"🔄 Agent {agent_idx} needs more training on current stage"
                        )
                        print(
                            f"   📊 Win Rate: {episode_win_rate:.3f} < {stage.success_threshold}"
                        )
                        print(
                            f"   📊 Total Attempts: {current_attempts}/{max_episodes_per_stage}"
                        )
                        print(f"   🎯 Success threshold not met")

                    stage_results[f"agent_{agent_idx}"] = agent_performance
                    agent.stage_performance.append(agent_performance)

                else:
                    print(
                        f"⏭️  Agent {agent_idx} ({agent.agent_type.upper()}) - Stage {agent.current_stage} not completed yet"
                    )

            self.global_performance_log.append(
                {
                    "stage": stage.to_dict(),
                    "results": stage_results,
                    "timestamp": datetime.now().isoformat(),
                }
            )

            all_agents_completed = all(
                agent.current_stage > stage_idx for agent in self.agents
            )

            if all_agents_completed:
                print(
                    f"\n🎉 All agents completed Stage {stage.stage_id}! Moving to next stage..."
                )
                stage_idx += 1
            else:
                print(
                    f"\n🔄 Some agents still need to complete Stage {stage.stage_id}. Continuing training..."
                )

        return self._generate_final_report()

    def _validate_curriculum_constraints(
        self, agent, stage, action, context="training"
    ):
        if action not in stage.available_actions:
            print(
                f"⚠️  {context.upper()}: Agent {agent.agent_id} ({agent.agent_type}) "
                f"tried action {action} in stage {stage.stage_id} '{stage.name}' "
                f"(available: {stage.available_actions})"
            )
            return False
        return True

    def _analyze_stage_performance(self, agent, stage, episode_rewards, wins, episodes):
        print(
            f"\n📊 STAGE {stage.stage_id} ANALYSIS for Agent {agent.agent_id} ({agent.agent_type}):"
        )
        print(f"  Stage: {stage.name}")
        print(f"  Available Actions: {stage.available_actions}")
        print(f"  Success Threshold: {stage.success_threshold}")
        print(f"  Win Rate: {wins/episodes:.4f}")
        print(f"  Average Reward: {np.mean(episode_rewards):.4f}")
        print(f"  Reward Std: {np.std(episode_rewards):.4f}")

        positive_rewards = [r for r in episode_rewards if r > 0]
        negative_rewards = [r for r in episode_rewards if r < 0]
        zero_rewards = [r for r in episode_rewards if r == 0]

        print(
            f"  Positive Rewards: {len(positive_rewards)} ({len(positive_rewards)/len(episode_rewards)*100:.1f}%)"
        )
        print(
            f"  Negative Rewards: {len(negative_rewards)} ({len(negative_rewards)/len(episode_rewards)*100:.1f}%)"
        )
        print(
            f"  Zero Rewards: {len(zero_rewards)} ({len(zero_rewards)/len(episode_rewards)*100:.1f}%)"
        )

        if positive_rewards:
            print(f"  Avg Positive Reward: {np.mean(positive_rewards):.4f}")
        if negative_rewards:
            print(f"  Avg Negative Reward: {np.mean(negative_rewards):.4f}")

        if stage.stage_id == 3:
            print(
                f"  🔍 Stage 3 (Double Available): Agents should be learning double strategy"
            )
        elif stage.stage_id == 4:
            print(
                f"  🔍 Stage 4 (All Actions): Agents now have splits - may affect double usage patterns"
            )
            print(
                f"  🔍 Expected: Double usage might decrease as agents explore splits"
            )

    def _train_agent_on_stage(self, agent, env, stage, episodes, eval_episodes):
        start_time = time.time()
        episode_rewards = []
        wins = 0
        window_wins = 0
        window_start = 0

        total_episodes_across_stages = episodes * len(self.curriculum_stages)
        every_n_episodes_to_log = total_episodes_across_stages // 100

        training_log = {
            "agent_id": agent.agent_id,
            "agent_type": agent.agent_type,
            "stage_id": stage.stage_id,
            "stage_name": stage.name,
            "total_episodes": episodes,
            "timestamp": datetime.now().isoformat(),
            "logged_episodes": [],
            "action_usage_summary": {
                0: 0,
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 0,
            },
            "action_rewards_summary": {
                0: [],
                1: [],
                2: [],
                3: [],
                4: [],
                5: [],
            },
        }

        for episode in range(episodes):
            state = env.reset()
            done = False
            total_reward = 0

            should_log_episode = episode % every_n_episodes_to_log == 0
            episode_log = None

            if should_log_episode:
                episode_log = {
                    "episode": episode,
                    "actions": [],
                    "states": [],
                    "rewards": [],
                    "game_info": [],
                    "epsilon": agent.epsilon,
                    "final_result": None,
                }

            while not done:
                action = agent.get_action(state)

                self._validate_curriculum_constraints(agent, stage, action, "training")

                if action in stage.available_actions:
                    training_log["action_usage_summary"][action] += 1

                next_state, reward, done = env.step(action)

                if action in stage.available_actions:
                    training_log["action_rewards_summary"][action].append(reward)

                if hasattr(agent, "remember"):
                    agent.remember(state, action, reward, next_state, done)

                    if episode < episodes // 2:
                        agent.replay()
                    else:
                        if episode % 3 == 0:
                            agent.replay()
                else:
                    agent.update(state, action, reward, next_state)

                state = next_state
                total_reward += reward

            episode_rewards.append(total_reward)

            detailed_stats = env.get_detailed_win_stats()
            if detailed_stats:
                episode_wins = 0
                episode_losses = 0
                for hand_detail in detailed_stats["hand_details"]:
                    bet_multiplier = 2 if hand_detail["doubled"] else 1
                    if (
                        hand_detail["result"] == "win"
                        or hand_detail["result"] == "blackjack"
                    ):
                        episode_wins += bet_multiplier
                    elif hand_detail["result"] in ["lose", "bust"]:
                        episode_losses += bet_multiplier

                if episode_wins > 0:
                    wins += episode_wins
                    window_wins += episode_wins

                if should_log_episode and detailed_stats["total_hands"] > 1:
                    print(
                        f"    Episode {episode} - Hands: {detailed_stats['total_hands']}, "
                        f"Won: {detailed_stats['hands_won']}, "
                        f"Double Downs: {detailed_stats['double_downs']}, "
                        f"Splits: {detailed_stats['splits']}, "
                        f"Episode Wins: {episode_wins}"
                    )

                if should_log_episode and episode_log:
                    episode_log["final_result"] = {
                        "episode_wins": episode_wins,
                        "total_hands": detailed_stats["total_hands"],
                        "hands_won": detailed_stats["hands_won"],
                        "hands_lost": detailed_stats["hands_lost"],
                        "double_downs": detailed_stats["double_downs"],
                        "splits": detailed_stats["splits"],
                    }

                    final_game_info = env.get_game_info()
                    episode_log["final_game_state"] = {
                        "player_hands": final_game_info["player_hands"].copy(),
                        "dealer_hand": final_game_info["dealer_hand"].copy(),
                        "detailed_stats": detailed_stats,
                    }

                    training_log["logged_episodes"].append(episode_log)

            agent.decay_epsilon()

            if should_log_episode:
                window_episodes = episode - window_start + 1
                recent_win_rate = (window_wins / window_episodes) * 100

                print(
                    f"  Episode {episode}: Win Rate: {recent_win_rate:.1f}%, "
                    f"Total Wins: {wins}, Epsilon: {agent.epsilon:.4f}"
                )

                window_wins = 0
                window_start = episode + 1

        training_log["summary"] = {
            "total_wins": wins,
            "win_rate": wins / episodes,
            "avg_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "action_usage": training_log["action_usage_summary"],
            "action_performance": {
                action: {
                    "count": len(rewards),
                    "avg_reward": np.mean(rewards) if rewards else 0,
                    "std_reward": np.std(rewards) if rewards else 0,
                }
                for action, rewards in training_log["action_rewards_summary"].items()
                if rewards
            },
        }

        self._analyze_stage_performance(agent, stage, episode_rewards, wins, episodes)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        training_filename = os.path.join(
            self.training_log_dir,
            f"training_log_agent_{agent.agent_id}_{agent.agent_type}_stage_{stage.stage_id}_{timestamp}.json",
        )
        with open(training_filename, "w") as f:
            json.dump(training_log, f, indent=2)

        print(f"  📊 Training log saved to: {training_filename}")

        evaluation_results = self._evaluate_agent(agent, env, eval_episodes, stage)
        end_time = time.time()
        time_taken = end_time - start_time
        print(f"Time taken: {time_taken:.2f} seconds")
        return {
            "win_rate": evaluation_results["win_rate"],
            "avg_reward": evaluation_results.get("avg_reward", 0),
            "total_wins": wins,
            "stage_id": stage.stage_id,
            "agent_type": agent.agent_type,
            "poor_actions": evaluation_results.get("poor_actions", []),
            "time_taken": time_taken,
        }

    def _get_llm_guided_action(self, agent, state, stage):
        valid_actions = self._get_valid_actions_for_stage(state, stage)

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
            action = int("".join(filter(str.isdigit, response))[:1])
            if action in valid_actions:
                return action
        except:
            pass

        return agent.get_action(state)

    def _get_valid_actions_for_stage(self, state, stage):
        (
            player_sum,
            dealer_up,
            has_ace,
            can_split,
            can_double,
            is_blackjack,
            can_surrender,
            can_insure,
        ) = state[:8]
        valid_actions = [0]

        if player_sum < 21 and not is_blackjack:
            valid_actions.append(1)
        if can_double:
            valid_actions.append(2)
        if can_split:
            valid_actions.append(3)
        if can_surrender:
            valid_actions.append(4)
        if can_insure:
            valid_actions.append(5)

        stage_valid_actions = [a for a in valid_actions if a in stage.available_actions]

        return stage_valid_actions if stage_valid_actions else [0]

    def _apply_action_focus(self, agent, recommended_actions):
        if hasattr(agent, "action_focus_weight"):
            agent.action_focus_weight = {}
        else:
            agent.action_focus_weight = {}

        for action in recommended_actions:
            agent.action_focus_weight[action] = 1.5

        print(f"  🎯 Focusing on actions: {recommended_actions}")

    def _preserve_learned_strategies(self, agent, stage, agent_performance):
        print(
            f"  💾 Preserving learned strategies for Agent {agent.agent_id} (Stage {stage.stage_id})"
        )

        models_dir = os.path.join(self.log_dir, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        if agent.agent_type == "dqn":
            model_path = os.path.join(
                models_dir, f"agent_{agent.agent_id}_stage_{stage.stage_id}.pth"
            )
            agent.save_model(model_path)
            print(f"    DQN model saved to: {model_path}")
        else:
            model_path = os.path.join(
                models_dir, f"agent_{agent.agent_id}_stage_{stage.stage_id}.pkl"
            )
            agent.save_model(model_path)
            print(f"    Q-table saved to: {model_path}")

        agent.stage_models = getattr(agent, "stage_models", {})
        agent.stage_models[stage.stage_id] = {
            "model_path": model_path,
            "performance": agent_performance,
            "available_actions": stage.available_actions,
        }

    def _load_previous_stage_strategies(self, agent, stage):
        print(
            f"  💾 Loading strategies from previous stages for Agent {agent.agent_id} (Stage {stage.stage_id})"
        )
        previous_stage_id = stage.stage_id - 1
        if previous_stage_id in agent.stage_models:
            previous_stage_data = agent.stage_models[previous_stage_id]
            print(f"    Loading strategies from Stage {previous_stage_id}")
            print(f"    Available Actions: {previous_stage_data['available_actions']}")

            model_path = previous_stage_data["model_path"]
            if agent.agent_type == "dqn":
                agent.load_model(model_path)
                print(f"    DQN model loaded from: {model_path}")
            else:
                agent.load_model(model_path)
                print(f"    Q-table loaded from: {model_path}")

            current_available_actions = set(stage.available_actions)
            previous_available_actions = set(previous_stage_data["available_actions"])
            new_available_actions = list(
                current_available_actions.union(previous_available_actions)
            )
            agent.action_space = new_available_actions
            print(f"    Merged available actions: {new_available_actions}")

            if hasattr(agent, "action_space"):
                agent.action_space = new_available_actions
                print(f"    Agent action_space updated to: {agent.action_space}")

            if hasattr(agent, "action_space"):
                agent.action_space = new_available_actions
                print(f"    Agent action_space updated to: {agent.action_space}")

    def _evaluate_agent(self, agent, env, episodes, stage=None):
        original_epsilon = agent.epsilon
        agent.epsilon = 0.05

        # Create a separate evaluation environment with different seed
        eval_env = (
            CurriculumBlackjackEnv(
                stage,
                deck_type=self.deck_type,
                penetration=self.penetration,
            )
            if stage
            else BlackjackEnv(
                deck_type=self.deck_type,
                penetration=self.penetration,
            )
        )

        total_rewards = []
        wins = 0
        action_rewards = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}

        state_action_stats = {}
        state_win_stats = {}
        state_reward_stats = {}
        game_outcomes = {
            "wins": 0,
            "losses": 0,
            "busts": 0,
            "pushes": 0,
            "blackjacks": 0,
        }

        evaluation_log = {
            "agent_id": agent.agent_id,
            "stage_id": stage.stage_id if stage else None,
            "agent_type": agent.agent_type,
            "evaluation_episodes": episodes,
            "timestamp": datetime.now().isoformat(),
        }

        for episode_idx in range(episodes):
            state = eval_env.reset()
            done = False
            episode_reward = 0
            episode_actions = []

            episode_log = {
                "episode": episode_idx,
                "actions": [],
                "states": [],
                "rewards": [],
                "game_info": [],
                "final_result": None,
                "detailed_stats": None,
            }

            while not done:
                action = agent.get_action(state)
                episode_actions.append(action)

                self._validate_curriculum_constraints(
                    agent, stage, action, "evaluation"
                )

                player_sum = state[0]
                dealer_up = state[1]
                has_ace = state[2]
                state_key = f"P{player_sum}_D{dealer_up}_A{has_ace}"

                if state_key not in state_action_stats:
                    state_action_stats[state_key] = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
                state_action_stats[state_key][action] += 1

                if state_key not in state_reward_stats:
                    state_reward_stats[state_key] = []
                if state_key not in state_win_stats:
                    state_win_stats[state_key] = {"wins": 0, "total": 0}

                episode_log["states"].append(
                    {
                        "player_sum": state[0],
                        "dealer_up": state[1],
                        "has_ace": state[2],
                        "can_split": state[3],
                        "can_double": state[4],
                        "is_blackjack": state[5],
                    }
                )
                episode_log["actions"].append(action)

                game_info = eval_env.get_game_info()
                episode_log["game_info"].append(
                    {
                        "player_hands": game_info["player_hands"].copy(),
                        "dealer_hand": game_info["dealer_hand"].copy(),
                        "current_hand_idx": game_info["current_hand_idx"],
                        "game_over": game_info["game_over"],
                    }
                )

                state, reward, done = eval_env.step(action)
                episode_reward += reward
                episode_log["rewards"].append(reward)

                state_reward_stats[state_key].append(reward)

            total_rewards.append(episode_reward)

            detailed_stats = eval_env.get_detailed_win_stats()
            if detailed_stats:
                episode_wins = 0
                episode_losses = 0
                for hand_detail in detailed_stats["hand_details"]:
                    bet_multiplier = 2 if hand_detail["doubled"] else 1
                    if (
                        hand_detail["result"] == "win"
                        or hand_detail["result"] == "blackjack"
                    ):
                        episode_wins += bet_multiplier
                    elif hand_detail["result"] in ["lose", "bust"]:
                        episode_losses += bet_multiplier

                if episode_wins > 0:
                    wins += episode_wins

                for hand_detail in detailed_stats["hand_details"]:
                    result = hand_detail["result"]
                    bet_multiplier = 2 if hand_detail["doubled"] else 1
                    if result == "win":
                        game_outcomes["wins"] += bet_multiplier
                    elif result == "lose":
                        game_outcomes["losses"] += bet_multiplier
                    elif result == "blackjack":
                        game_outcomes["blackjacks"] += bet_multiplier
                    elif result == "push":
                        game_outcomes["pushes"] += bet_multiplier
                    elif result == "bust":
                        game_outcomes["busts"] += bet_multiplier

                for state_key in state_win_stats:
                    state_win_stats[state_key]["total"] += 1
                    if episode_wins > 0:
                        state_win_stats[state_key]["wins"] += 1

                episode_log["detailed_stats"] = detailed_stats
                episode_log["final_result"] = {
                    "episode_wins": episode_wins,
                    "total_hands": detailed_stats["total_hands"],
                    "hands_won": detailed_stats["hands_won"],
                    "hands_lost": detailed_stats["hands_lost"],
                    "double_downs": detailed_stats["double_downs"],
                    "splits": detailed_stats["splits"],
                }

                final_game_info = eval_env.get_game_info()
                episode_log["final_game_state"] = {
                    "player_hands": final_game_info["player_hands"].copy(),
                    "dealer_hand": final_game_info["dealer_hand"].copy(),
                    "detailed_stats": detailed_stats,
                }

            for action in episode_actions:
                action_rewards[action].append(episode_reward)

        agent.epsilon = original_epsilon

        poor_actions = []
        for action, rewards in action_rewards.items():
            if rewards and np.mean(rewards) < -0.2:
                poor_actions.append(action)

        total_hands = sum(game_outcomes.values())
        game_outcome_percentages = {}
        if total_hands > 0:
            game_outcome_percentages = {
                "win_percent": (game_outcomes["wins"] / total_hands) * 100,
                "lose_percent": (game_outcomes["losses"] / total_hands) * 100,
                "bust_percent": (game_outcomes["busts"] / total_hands) * 100,
                "push_percent": (game_outcomes["pushes"] / total_hands) * 100,
                "blackjack_percent": (game_outcomes["blackjacks"] / total_hands) * 100,
                "win_loss_ratio": (
                    (
                        (game_outcomes["wins"] + game_outcomes["blackjacks"])
                        / (game_outcomes["losses"] + game_outcomes["busts"])
                    )
                    if (game_outcomes["losses"] + game_outcomes["busts"]) > 0
                    else float("inf")
                ),
            }

        strategy_table = {}
        for state_key, action_counts in state_action_stats.items():
            total_actions = sum(action_counts.values())
            if total_actions > 0:
                strategy_table[state_key] = {
                    "stand_percent": (action_counts[0] / total_actions) * 100,
                    "hit_percent": (action_counts[1] / total_actions) * 100,
                    "double_percent": (action_counts[2] / total_actions) * 100,
                    "split_percent": (action_counts[3] / total_actions) * 100,
                    "total_actions": total_actions,
                    "win_rate": (
                        (
                            state_win_stats[state_key]["wins"]
                            / state_win_stats[state_key]["total"]
                        )
                        * 100
                        if state_win_stats[state_key]["total"] > 0
                        else 0
                    ),
                    "avg_reward": (
                        np.mean(state_reward_stats[state_key])
                        if state_reward_stats[state_key]
                        else 0
                    ),
                }

        evaluation_log["summary"] = {
            "win_rate": wins / episodes,
            "avg_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
            "total_wins": wins,
            "poor_actions": poor_actions,
            "game_outcomes": game_outcomes,
            "game_outcome_percentages": game_outcome_percentages,
            "strategy_table": strategy_table,
            "state_action_stats": state_action_stats,
            "state_win_stats": state_win_stats,
            "state_reward_stats": {
                k: {"avg": np.mean(v), "std": np.std(v), "count": len(v)}
                for k, v in state_reward_stats.items()
                if v
            },
        }

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(
            self.eval_log_dir,
            f"evaluation_log_agent_{agent.agent_id}_{agent.agent_type}_{timestamp}.json",
        )
        with open(filename, "w") as f:
            json.dump(evaluation_log, f, indent=2)

        print(f"  📊 Evaluation log saved to: {filename}")

        return {
            "win_rate": wins / episodes,
            "avg_reward": np.mean(total_rewards),
            "std_reward": np.std(total_rewards),
            "total_wins": wins,
            "poor_actions": poor_actions,
            "game_outcomes": game_outcomes,
            "game_outcome_percentages": game_outcome_percentages,
            "strategy_table": strategy_table,
            "state_action_stats": state_action_stats,
            "state_win_stats": state_win_stats,
            "state_reward_stats": {
                k: {"avg": np.mean(v), "std": np.std(v), "count": len(v)}
                for k, v in state_reward_stats.items()
                if v
            },
        }

    def _generate_stage_comparison_summary(self):
        print(f"\n📊 STAGE 3 vs STAGE 4 COMPARISON SUMMARY")
        print("=" * 60)

        stage3_results = None
        stage4_results = None

        for stage_log in self.global_performance_log:
            if stage_log["stage"]["stage_id"] == 3:
                stage3_results = stage_log
            elif stage_log["stage"]["stage_id"] == 4:
                stage4_results = stage_log

        if stage3_results and stage4_results:
            print(f"🔍 Stage 3: {stage3_results['stage']['name']}")
            print(
                f"  Available Actions: {stage3_results['stage']['available_actions']}"
            )
            print(
                f"  Success Threshold: {stage3_results['stage']['success_threshold']}"
            )

            print(f"\n🔍 Stage 4: {stage4_results['stage']['name']}")
            print(
                f"  Available Actions: {stage4_results['stage']['available_actions']}"
            )
            print(
                f"  Success Threshold: {stage4_results['stage']['success_threshold']}"
            )

            print(f"\n📈 Performance Comparison:")
            for agent_key in stage3_results["results"]:
                if agent_key in stage4_results["results"]:
                    agent3 = stage3_results["results"][agent_key]
                    agent4 = stage4_results["results"][agent_key]

                    print(f"\n  {agent_key.upper()}:")
                    print(f"    Stage 3 Win Rate: {agent3['win_rate']:.4f}")
                    print(f"    Stage 4 Win Rate: {agent4['win_rate']:.4f}")
                    print(
                        f"    Win Rate Change: {agent4['win_rate'] - agent3['win_rate']:+.4f}"
                    )

                    print(f"    Stage 3 Avg Reward: {agent3['avg_reward']:.4f}")
                    print(f"    Stage 4 Avg Reward: {agent4['avg_reward']:.4f}")
                    print(
                        f"    Reward Change: {agent4['avg_reward'] - agent3['avg_reward']:+.4f}"
                    )

                    print(f"    Stage 3 Poor Actions: {agent3.get('poor_actions', [])}")
                    print(f"    Stage 4 Poor Actions: {agent4.get('poor_actions', [])}")

        print(f"\n💡 Key Insights:")
        print(f"  1. Stage 4 has ALL actions available (including double)")
        print(
            f"  2. Higher average rewards in Stage 4 are expected due to more options"
        )
        print(f"  3. Double usage patterns may change due to split availability")
        print(f"  4. This is normal curriculum learning behavior")

    def _generate_final_report(self):
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

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = os.path.join(
            self.report_log_dir, f"curriculum_training_report_{timestamp}.json"
        )
        with open(report_filename, "w") as f:
            json.dump(report, f, indent=2)

        print(f"📊 Final report saved to: {report_filename}")

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
        models_dir = os.path.join(self.log_dir, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        for i, agent in enumerate(self.agents):
            filename = f"{prefix}_{agent.agent_type}_{i}"
            if agent.agent_type == "dqn":
                model_path = os.path.join(models_dir, f"{filename}.pth")
                agent.save_model(model_path)
            else:
                model_path = os.path.join(models_dir, f"{filename}.pkl")
                agent.save_model(model_path)
            print(f"Saved {filename} to {model_path}")

    def create_run_summary(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary = {
            "run_timestamp": timestamp,
            "log_directory": self.log_dir,
            "training_config": {
                "num_agents": self.num_agents,
                "agent_types": self.agent_types,
                "deck_type": self.deck_type,
                "penetration": self.penetration,
                "curriculum_stages": len(self.curriculum_stages),
                "reward_type": "simplified",
            },
            "directory_structure": {
                "evaluation_logs": self.eval_log_dir,
                "training_logs": self.training_log_dir,
                "report_logs": self.report_log_dir,
                "models": os.path.join(self.log_dir, "models"),
            },
            "files_generated": {
                "evaluation_logs": [],
                "training_logs": [],
                "reports": [],
                "models": [],
            },
        }

        for log_type, dir_path in summary["directory_structure"].items():
            if os.path.exists(dir_path):
                files = [
                    f
                    for f in os.listdir(dir_path)
                    if f.endswith((".json", ".pth", ".pkl"))
                ]
                summary["files_generated"][
                    log_type.replace("_logs", "").replace("_log", "")
                ] = files

        summary_filename = os.path.join(self.log_dir, f"run_summary_{timestamp}.json")
        with open(summary_filename, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"📋 Run summary saved to: {summary_filename}")

        self._run_automatic_analysis(summary_filename)

        return summary

    def _run_automatic_analysis(self, summary_filename):
        try:
            import subprocess
            import sys

            print(f"\n🔍 Running automatic analysis...")

            script_dir = os.path.dirname(os.path.abspath(__file__))
            analyze_script = os.path.join(script_dir, "analyze_logs.py")

            cmd = [sys.executable, analyze_script, summary_filename]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(f"✅ Automatic analysis completed successfully!")
                if result.stdout:
                    print(result.stdout)
            else:
                print(f"⚠️  Analysis completed with warnings:")
                if result.stderr:
                    print(result.stderr)
                if result.stdout:
                    print(result.stdout)

        except Exception as e:
            print(f"⚠️  Could not run automatic analysis: {e}")
            print(
                f"   You can manually run: python scripts/analyze_logs.py {summary_filename}"
            )
