import json
from LLM import LLM

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
