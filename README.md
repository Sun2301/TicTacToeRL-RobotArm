# TicTacToeRL-RobotArm

A robotic arm that learns to play Three Men's Morris (Morpion) against a human opponent using Q-learning and computer vision.

## Overview

This project integrates reinforcement learning with computer vision to enable a DOFBOT robotic arm to play a strategic board game autonomously. The system perceives the board state in real time, decides the next move using a trained Q-learning agent, and executes the move physically via the robotic arm.

**Presented at:**
- Live Demo — Benin Workshop on AI (BWAI) 2025
- Poster — Deep Learning IndabaX Benin 2025
- Hands-on Demo — TEKBOT Robotics Challenge 2025

## Architecture

```
Camera Feed → [Perception] → Board State → [Decision (Q-Learning)] → Move → [Control] → Robot Arm
                (YOLOv8)                      (training in sim)                   (DOFBOT)
```

The system is split into three modules:

| Module | Folder | Description |
|--------|--------|-------------|
| Perception | `perception/` | Real-time board detection using YOLOv8 (88% accuracy) |
| Decision | `decision/` | Q-learning agent trained over 400,000 episodes |
| Control | `control/` | Motor control and trajectory planning for DOFBOT |




## Demo

https://github.com/user-attachments/assets/e8624f03-0c77-4c8b-ad30-adfe14059fc6


*DOFBOT playing Three Men's Morris against a human*


## Key Results

- **Detection accuracy:** 88% with YOLOv8 under variable lighting conditions
- **Training:** Q-learning agent converged after 400,000 episodes
- **Sim-to-real:** Calibrated motor precision and vision point estimates to bridge simulation-to-hardware gap

## Challenges Solved

- Handling variable lighting conditions for consistent board detection
- Calibrating robot arm movements to match vision-based position estimates
- Resolving placement errors where the arm targeted occupied squares
- Balancing exploration vs exploitation during Q-learning training

## Setup

Each module has its own dependencies. Start with:

```bash
git clone https://github.com/Sun2301/TicTacToeRL-RobotArm.git
cd TicTacToeRL-RobotArm
```

See `docs/` for detailed setup instructions per module.

## Status

- [x] Perception module (YOLOv8 detection)
- [x] Q-learning agent (trained, working)
- [x] Robot arm control (DOFBOT)
- [ ] Full pipeline integration refinement (in progress on feature branches)

## Publication

*Hounsinou, Fangnon, Kochoni, Kpokpo — "Intégration de l'apprentissage par renforcement et de la vision par ordinateur pour le jeu de Morpion avec un bras robotique"*

## License

MIT
