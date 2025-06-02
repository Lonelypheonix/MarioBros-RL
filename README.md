# MarioBros-RL
# RL Final Project
> Teaching an Agent to Play Mario Bros using Reinforcement Learning

## Table of contents
* [Introduction](#Introduction)
* [Model](#Model)
*  [Results](#Results)
*  [Training and Testing Video](# Training and Testing Video)
* [Code](#Code)

## Introduction

We are using OpenAI Gym environment for Super Mario Bros on The Nintendo Entertainment System (NES) using the nes-py emulator.

The goal of this game is to reach the end of each level while avoiding pitfalls and enemies.

## Model

We use two Algorithms:

1.   **DQN**  

2.   **DDQN:**  


## Results

Lets look at the graphs for a better understanding of the models.

### DQN
1.  Reward Vs Episode 
![Reward Vs Episode](https://github.com/Lonelypheonix/MarioBros-RL/blob/main/DQN/Results/Training_RewardvsEpisode_graph.png)

2. Exploration graph 
![Exploration](https://github.com/Lonelypheonix/MarioBros-RL/blob/main/DQN/Results/Exploration_graph.png)
3. Training 
![Training](https://github.com/Lonelypheonix/MarioBros-RL/blob/main/DQN/Results/model_Training.png)
### DDQN
1. Reward Vs Episode
![Reward Vs Episode](https://github.com/Lonelypheonix/MarioBros-RL/blob/main/DDQN/Results/Train_RewardvsEpisode_graph.png)

2. Accuracy and Loss Curve for SWIN Head only Fine-Tune  
![Swin_head](https://github.com/Lonelypheonix/MarioBros-RL/blob/main/DDQN/Results/Exploration_graph.png)
3. Training 
![Training](https://github.com/Lonelypheonix/MarioBros-RL/blob/main/DDQN/Results/model_training.png)

## Training and Testing Video

### DDQN
- [▶️ Play Training Video]([results/training.mp4](https://github.com/Lonelypheonix/MarioBros-RL/blob/main/DDQN/Results/Mario_train_DDQN.mp4
Testing))
<br>
https://github.com/Lonelypheonix/MarioBros-RL/blob/main/DDQN/Results/Mario_train_DDQN.mp4
Testing
https://github.com/Lonelypheonix/MarioBros-RL/blob/main/DDQN/Results/Mario_test_DDQN.mp4

### DQN
Training 
https://github.com/Lonelypheonix/MarioBros-RL/blob/main/DQN/Results/Mario_train_DQN.mp4
Testing
https://github.com/Lonelypheonix/MarioBros-RL/blob/main/DQN/Results/Mario_test_DQN.mp4

## Code 
You can check the respective folders for the code
