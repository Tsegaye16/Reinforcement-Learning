---

# WebGame AI Training with Stable-Baselines3

This project demonstrates how to create and train an AI agent to play a simple web-based game using reinforcement learning with **Stable-Baselines3**. The agent interacts with the game environment by taking actions, receiving rewards, and learning to optimize its performance through **Deep Q-Network (DQN)**.

## Project Overview

This project uses a **gym-like environment** to interact with a web game. The environment captures the game screen, processes the frames, sends actions, and receives feedback, including whether the game is over. Using **Stable-Baselines3**, a DQN model is trained to learn the best actions to take based on the observations it receives from the game environment.

The overall process involves:

1. **Defining a custom game environment**: Using Python libraries to capture and interpret the game’s state and control the game.
2. **Training the model**: Applying the DQN algorithm to train an agent to play the game by interacting with the environment.
3. **Evaluating and saving the model**: After training, evaluating the agent's performance and saving the best model during training.

## Key Features:

- **Custom Gym Environment**: The web game is wrapped as a custom environment class inheriting from `gym.Env`. The game state is captured using `mss` and processed using `OpenCV`.
- **Action Space**: The agent can choose from three actions: pressing 'space', pressing 'down', or performing no operation.
- **Observation Space**: The observation is a processed grayscale frame of the game environment, reshaped for the model input.
- **Training with Stable-Baselines3**: The agent is trained using the DQN algorithm with a CNN policy to handle image-based input.
- **Model Checkpoints and Video Capture**: Checkpoints are saved during training, and the agent's gameplay is captured as videos for evaluation.

## Prerequisites

### Required Libraries

You can install all the required dependencies by running:

```bash
pip install gym stable-baselines3 mss pydirectinput opencv-python numpy pytesseract matplotlib
```

These libraries provide functionalities for creating the game environment, interacting with it, and running reinforcement learning algorithms.

- `gym`: Provides the standard environment and structure for reinforcement learning.
- `stable-baselines3`: Contains implementations of RL algorithms like DQN.
- `mss`, `pydirectinput`, `opencv-python`: Used to capture the game screen and interact with the game.
- `numpy`, `matplotlib`: Used for data manipulation and visualization.
- `pytesseract`: Used to detect the "game over" screen from the game.

## Environment Setup

### Game Environment

A custom game environment is defined using **Gym's Env class**. The following details define the game interaction:

- **Observation Space**: The game screen is captured, converted to grayscale, resized, and reshaped as an observation.
- **Action Space**: The action space consists of three actions:
  - Action 0: Press the spacebar.
  - Action 1: Press the down arrow.
  - Action 2: No operation (do nothing).

### Action and Observation Processing

- **Actions**: The actions are mapped to corresponding keypresses using the `pydirectinput` library.
- **Observations**: The game screen is captured using `mss`, then processed with OpenCV to convert the image to grayscale and resized to match the input size required for the model.

### Game Over Detection

The game over condition is detected by capturing part of the screen and using **Tesseract OCR** to check for the presence of the words “GAME” or “GAHE”.

## Training the AI Agent

### Model: DQN with CNN Policy

The agent is trained using the **DQN (Deep Q-Network)** algorithm, which is suitable for environments with visual inputs (such as games). The `CnnPolicy` is used to handle image-based observations, and the model is trained to optimize the agent's actions to maximize its rewards.

```python
model = DQN('CnnPolicy', env, verbose=1, buffer_size=1200000, learning_starts=1000)
```

### Training Callbacks

A custom callback is defined to save the model at regular intervals:

```python
callback = TrainAndLoggingCallback(check_freq=1000, save_path='./train/')
```

This callback saves the model every 1000 steps to track its progress.

### Model Training Process

The model is trained for 10,000 time steps with the following process:

1. The agent interacts with the environment, taking actions based on the current state.
2. The agent receives rewards and observes new states.
3. The model updates its policy to improve performance.

After training, the model is saved to a checkpoint directory for future use or evaluation.

## Evaluating the Model

### Gameplay Recording

After training the model, we evaluate it over 5 episodes. The agent interacts with the environment, and the game frames are recorded as a video for analysis:

```python
video_path = f'gameplay_episode_{episode}.avi'
```

The video is saved using OpenCV’s `VideoWriter` functionality. The best 5 episodes are saved for visual inspection of the agent's behavior.

### Example of Running Evaluation:

```python
for episode in range(5):
    obs, _ = env.reset()  # Reset the environment
    done = False
    total_reward = 0
    frames = []  # Store frames for the video

    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, info = env.step(int(action))
        total_reward += reward

        # Capture the current frame for the video
        frames.append(env.current_frame)

    # Save the gameplay as a video
    if episode < 5:  # Save the best 5 episodes
        video_path = f'gameplay_episode_{episode}.avi'
        height, width, layers = frames[0].shape
        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))
        for frame in frames:
            video.write(frame)
        video.release()
```

This step allows you to visually evaluate how well the agent learned to play the game.

---

## Running the Project

1. **Clone the repository** or download the project files.
2. Install the required dependencies using `pip` as shown above.
3. Run the script to train the agent and record gameplay:

```bash
python train_and_evaluate.py
```

4. After training, check the `./train/` directory for model checkpoints and the `./logs/` directory for logs. The best episodes of gameplay will be saved as `.avi` videos in the project directory.

---

## Contributing

Feel free to fork the repository and submit issues or pull requests to improve the environment, training procedure, or any other aspect of the project.

---

## License

This project is open-source and available under the MIT license.

---
