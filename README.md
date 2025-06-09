# 🐍 Snake Game AI (Deep Q-Learning)

This project implements an AI agent to play the classic Snake game using **Deep Q-Learning (DQN)** with **TensorFlow**. The agent learns to improve over time by interacting with the environment and storing its experiences.

## 📂 Structure

- `agent.py` – Core agent logic implementing Deep Q-Learning.
- `game.py` – Environment for the Snake game (based on Pygame).
- `model.py` – Builds the neural network using Keras.

## 🧠 Features

- DQN-based agent with:
  - Epsilon-greedy policy
  - Experience replay
  - Target Q-value updates
- Neural network with:
  - Input: 11-dimensional game state
  - Output: Q-values for 3 possible actions
- Reward shaping based on collision, apple distance, and game progress

## 🚀 How It Works

1. The agent observes the current game state.
2. Chooses an action via an epsilon-greedy strategy.
3. Plays a step and stores the experience `(state, action, reward, next_state, done)` in memory.
4. Trains the neural network on random batches from memory.

## 🛠️ Requirements

- Python 3.10+
- TensorFlow
- NumPy
- Pygame

Install dependencies:
```bash
pip install tensorflow numpy pygame
```

## ▶️ Run the Game

Make sure `game.py` and `model.py` exist and are implemented properly.

Then run:
```bash
python agent.py
```

## 📈 Training Parameters

| Parameter        | Value      |
|------------------|------------|
| Memory size      | 100,000    |
| Batch size       | 64         |
| Learning rate    | 0.001      |
| Gamma (discount) | 0.95       |
| Epsilon decay    | 0.998      |

## 📄 License

MIT License

---

Made with ❤️ by [YourName]
