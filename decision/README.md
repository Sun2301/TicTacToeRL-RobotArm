# Decision Module

This folder is the RL decision branch of the full project. It consumes a board state and outputs the next legal move.

## Scope
- game logic for Three Men's Morris
- action generation and validation
- policy training and policy inference
- terminal-based interactive play for local testing

It does not handle:
- camera detection (`perception/`)
- robot motion execution (`control/`)

## Board Encoding
Board indexing:
```
| 0 | 1 | 2 |
| 3 | 4 | 5 |
| 6 | 7 | 8 |
```

Cell symbols:
- `1` current agent piece
- `-1` opponent piece
- `0` empty

Phases:
- deployment: each side places 3 pieces
- movement: moves must respect `CONSTRAINT_DICT`

## Code Structure
- `core/board.py`: board transitions and legal moves
- `core/game.py`: win checks
- `core/agent.py`: train/play loops and epsilon-greedy action choice
- `algorithms/base.py`: algorithm interface
- `algorithms/qlearning.py`: tabular Q-learning implementation
- `scripts/train.py`: adversary + main training
- `scripts/play.py`: interactive play against model
- `scripts/quick_train_play.py`: short training followed by play
- `main.py`: CLI menu entry point

## Current Algorithm: Q-Learning
Update rule:
```
Q(s, a) <- Q(s, a) + alpha * (r + gamma * max_a' Q(s', a') - Q(s, a))
```

Current defaults:
- `alpha=0.5`
- `gamma=0.99`
- epsilon schedule during training: `0.9 -> 0.1` with decay `0.9995`
- rewards: win `+60`, loss `-50`, draw/stalemate `+1`

## Next Algorithm: DQN
The module is prepared for algorithm swapping through `LearningAlgorithm` in `algorithms/base.py`.

Planned DQN integration:
- add `algorithms/dqn.py` implementing the same interface
- keep `core/agent.py` unchanged (algorithm-agnostic use)
- add model checkpointing for neural network weights
- add replay buffer and target-network update policy

## Run
From repository root:
```bash
python3 decision/scripts/train.py
python3 decision/scripts/play.py
python3 -m decision.main
```

## Model Paths
Relative model paths default to `decision/models/` if not found in current working directory.
