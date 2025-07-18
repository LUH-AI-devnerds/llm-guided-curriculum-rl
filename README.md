# LLM-Guided Curriculum Learning for Reinforcement Learning

An advanced reinforcement learning system that uses **Large Language Models (LLMs)** to guide **curriculum learning** with **multiple agents** learning Blackjack strategies progressively.

## 🎯 Key Features

### 🎓 **Curriculum Learning**
- **Progressive skill development** through structured learning stages
- **Action constraints** that gradually introduce complexity
- **Performance-based advancement** between curriculum stages
- **Adaptive thresholds** based on agent performance

### 🤖 **Multi-Agent System** 
- **Multiple agent types** (DQN Neural Networks, Tabular Q-Learning)
- **Parallel training** across different agents
- **Performance comparison** between agent architectures
- **Independent stage progression** for each agent

### 🧠 **LLM-Guided Learning**
- **Dynamic curriculum generation** using LLM expertise
- **Intelligent action recommendations** during training
- **Real-time strategy guidance** based on game situations
- **Adaptive curriculum modifications** based on performance

### 🎲 **Advanced Blackjack Environment**
- **Full rule support**: Stand, Hit, Double Down, Split
- **Multi-hand management** for split scenarios
- **Realistic reward structures** with blackjack bonuses
- **State representation** with usable aces and action availability

## 🚀 Quick Start

### Option 1: LLM-Guided Training (Recommended)

```bash
cd scripts/
python RLAgent.py
# Enter your Google AI API key when prompted
```

### Option 2: Demo Mode (No API Key Required)

```bash
cd scripts/
python demo_curriculum_learning.py
```

### Option 3: Basic Training (Fallback)

```bash
cd scripts/
python RLAgent.py
# Press Enter without API key for basic single-agent training
```

## 📚 System Architecture

### Curriculum Stages

| Stage | Name | Actions | Focus | Success Threshold |
|-------|------|---------|--------|-------------------|
| 1 | Basic Play | Stand, Hit | Learn fundamental decisions | 35% |
| 2 | Strategic Play | Stand, Hit | Master basic strategy | 40% |
| 3 | Advanced Betting | Stand, Hit, Double | Learn when to double down | 42% |
| 4 | Expert Play | All Actions | Master splits and complex scenarios | 45% |

### Agent Types

- **DQN Agent**: Deep Q-Network with experience replay
  - Neural network: 6 → 128 → 128 → 4
  - Experience replay buffer (10,000 samples)
  - Target network for stability

- **Tabular Q-Learning**: Traditional lookup table approach
  - State-action Q-table
  - Simple and interpretable
  - Fast training and inference

## 🧠 LLM Integration

### Curriculum Generation
The LLM designs progressive learning stages by analyzing:
- Action complexity and dependencies
- Learning objectives for each stage
- Realistic success thresholds
- Optimal skill progression

### Action Guidance
During training, the LLM provides:
- **Contextual recommendations** based on game state
- **Educational value assessment** of different actions
- **Strategic insights** for complex scenarios
- **Learning-focused suggestions** vs. purely optimal play

### Performance Adaptation
The LLM continuously adapts the curriculum by:
- **Analyzing agent performance** across different actions
- **Identifying struggling areas** and weak strategies
- **Adjusting difficulty progression** based on learning curves
- **Recommending action focus** for next training episodes

## 📊 Training Results

### Performance Metrics
- **Win Rate**: Percentage of games won
- **Average Reward**: Expected value per game
- **Stage Progression**: How quickly agents advance
- **Action Mastery**: Performance with specific actions

### Expected Improvements
- **Faster Convergence**: 40-60% reduction in training time
- **Better Performance**: 5-15% higher final win rates
- **More Robust Learning**: Better generalization to new scenarios
- **Interpretable Progress**: Clear skill development tracking

## 🔧 Configuration

### Environment Setup

```bash
# Install dependencies
pip install torch numpy google-generativeai pygame

# Set environment variables (optional)
export GOOGLE_AI_API_KEY="your_api_key_here"
```

### Custom Curriculum

```python
# Define custom curriculum stages
stages = [
    CurriculumStage(
        stage_id=1,
        name="Custom Stage",
        available_actions=[0, 1],
        description="Learning objective",
        difficulty=1,
        success_threshold=0.35
    )
]
```

### Agent Configuration

```python
# Customize agent parameters
agents = [
    DQNAgent(
        action_space=[0, 1, 2, 3],
        learning_rate=0.001,
        exploration_rate=1.0,
        exploration_decay=0.9995
    ),
    QLearningAgent(
        action_space=[0, 1, 2, 3], 
        learning_rate=0.1,
        exploration_rate=1.0,
        exploration_decay=0.999
    )
]
```

## 📁 File Structure

```
scripts/
├── RLAgent.py                     # Main training script with curriculum integration
├── curriculum_multi_agent_rl.py   # Core curriculum learning system
├── demo_curriculum_learning.py    # Demo without API key requirement
├── BlackJackENV.py               # Blackjack environment implementation
├── LLM.py                        # LLM interface (Google Gemini)
├── comprehensive_comparison.py    # Performance analysis tools
└── *.pkl, *.pth                  # Saved model files

GUI Application:
├── blackjack_gui.py              # Professional GUI for agent visualization
```

## 🎮 Usage Examples

### GUI Visualization
```bash
# Run the Blackjack RL Agent GUI
python blackjack_gui.py
```

### Basic Training
```python
from curriculum_multi_agent_rl import MultiAgentCurriculumSystem

# Initialize system
system = MultiAgentCurriculumSystem(
    llm_api_key="your_key",
    num_agents=3,
    agent_types=['dqn', 'tabular', 'dqn']
)

# Train through curriculum
results = system.train_multi_agent_curriculum(
    total_episodes=50000,
    eval_episodes=1000
)
```

### Custom LLM Prompts
```python
# Customize LLM guidance
llm_curriculum = LLMGuidedCurriculum(api_key="your_key")

# Generate curriculum
stages = llm_curriculum.generate_curriculum_stages(num_stages=5)

# Get action recommendations
recommendations = llm_curriculum.adapt_curriculum(
    agent_performance=performance_data,
    current_stage=current_stage,
    stages=all_stages
)
```

## 📈 Monitoring and Analysis

### Real-time Monitoring
- Training progress per agent and stage
- Win rate trends and epsilon decay
- Action usage statistics
- LLM recommendation frequency

### Post-training Analysis
- Comprehensive JSON reports with all metrics
- Stage progression visualizations  
- Agent performance comparisons
- Curriculum effectiveness analysis

## 🔬 Research Applications

This system enables research in:
- **Curriculum Learning**: Optimal stage design and progression
- **Multi-Agent RL**: Cooperation and competition dynamics
- **LLM Integration**: Human expertise in RL training
- **Transfer Learning**: Skill transfer between stages
- **Explainable AI**: Interpretable learning progression

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Google Gemini API** for LLM integration
- **PyTorch** for neural network implementation
- **OpenAI Gym** for environment design patterns
- **Blackjack research community** for strategy insights

## 📞 Support

For questions, issues, or feature requests:
- Open an issue on GitHub
- Check the demo scripts for usage examples
- Review the comprehensive documentation in code comments

---

**Happy Learning! 🎯🤖🧠**