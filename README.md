# 🎯 Inverted Pendulum Control System

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey)
![Status](https://img.shields.io/badge/status-Active-success)
![License](https://img.shields.io/badge/license-MIT-green)

A professional, real-time graphical control system for an inverted pendulum, integrating both classical control and modern Reinforcement Learning (RL) techniques.

This project provides a comprehensive framework to control, simulate, and train agents on an inverted pendulum, featuring a rich PyQt5 graphical user interface (GUI) for real-time interaction, live telemetry plotting, and seamless switching between control algorithms.

---

## 📖 Project Description

The Inverted Pendulum Control System is designed to bridge the gap between simulation and hardware deployment for complex control tasks. It implements robust real-time control logic utilizing both classical control theory (LQR) and advanced Reinforcement Learning algorithms (SAC, DDPG).

The system features a dynamic, responsive GUI built with PyQt5 that allows users to interact with the environment in real-time. Whether connecting to physical hardware via serial communication or running the embedded Pygame simulation, users can visualize telemetry data, tweak parameters, and trigger RL training pipelines directly from the dashboard.

## ✨ Key Features

- **🎛️ Controller Switching:** Switch seamlessly between LQR, SAC, and DDPG controllers in real-time.
- **📈 Real-time Telemetry & Live Plotting:** Visualize state variables (angle, position, velocities, rewards) dynamically using PyQtGraph.
- **🧠 GUI-Integrated RL Training:** Initiate, monitor, and manage the training of SAC and DDPG agents directly from the interface.
- **⚙️ Dynamic Parameter Tuning:** Edit the K gains of the LQR control.
- **💾 CSV Telemetry Export:** Export real-time operational data for offline analysis and academic review.
- **⚡ Multiprocessing Architecture:** Robust separation of GUI updates, control logic, and hardware communication to ensure strictly real-time performance.

---

## 🎥 Examples of the system

Here are demonstrations of the system running with different controllers:

### LQR Control

<video src="videos/lqr_pendulum_.mp4" width="300" controls>
  <a href="videos/lqr_pendulum_.mp4">View LQR Demo</a>
</video>

### Soft Actor-Critic (SAC) Control

<video src="videos/SAC_.mp4" width="300" controls>
  <a href="videos/SAC_.mp4">View SAC Demo</a>
</video>

### Deep Deterministic Policy Gradient (DDPG) Control

<video src="videos/DDPG_.mp4" width="300" controls>
  <a href="videos/DDPG_.mp4">View DDPG Demo</a>
</video>

---

## 🚀 GUI Usage Guide

The user interface is designed to be intuitive while exposing powerful control parameters.

1. **Selecting the Controller:** Use the main control panel dropdown to select your active controller (`LQR`, `SAC`, or `DDPG`).
2. **Running the System:**
   - Ensure your Arduino is connected (if using hardware mode) and select the correct COM port from the dropdown.
   - Click the **Start** button to initialize the control loop.
3. **Stopping Execution:** Click the **Stop** or **Emergency Stop** button to safely halt the pendulum and disable motor outputs.
4. **Editing LQR Gains:** When LQR is selected, use the dynamic input fields to adjust the $K$ penalty matrices to observe changes in system stability.
5. **Viewing Telemetry & Plots:** The dashboard displays live numerical telemetry. Switch to the plotting tabs to see scrolling, real-time graphs of the pendulum's angle, cart position, and control efforts.
6. **Training Agents:** Navigate to the RL Training tab, set your hyperparameters (episodes, batch size), and click **Train**. Progress and rewards are logged directly in the UI.

---

## 🛠️ Installation

Follow these steps to set up the project locally:

```bash
git clone https://github.com/primbiolab/IPFramework.git
cd IPFramework
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🏃 Running the System

To launch the primary graphical interface, run:

```bash
python main_GUI.py
```

The PyQt5 dashboard will open, initializing the multiprocessing queues. From here, you can select your control mode, load pre-trained models, or start a simulation session.

---

## 🔌 Hardware Requirements

To run this system on physical hardware, you need:

- An **Arduino** microcontroller flashed with `inverted_pendulum_arduino.ino`.
- Motor drivers and encoders wired to the specifications expected by the Arduino firmware.
- A stable USB Serial connection to the host PC.

---

## 🏗️ System Architecture

The software is engineered for strict real-time performance using a multi-layered architecture:

- **GUI (PyQt5):** Handles all user interactions, dynamic parameter updates, and live rendering of plots using `pyqtgraph`. It operates on the main thread to remain responsive.
- **Multiprocessing Engine:** To prevent GUI lag from interfering with control execution, the system uses Python's `multiprocessing` module. Control loops and serial communication run in isolated, high-priority processes.
- **Telemetry Queue:** A thread-safe, lock-free queue system transports state vectors and rewards from the control process back to the GUI for visualization and logging.
- **Control Logic:** Modularly designed to allow hot-swapping between the algebraic LQR solver and neural network inference engines (SAC/DDPG).
- **Environment Simulation:** A custom wrapper mimicking the hardware's serial API, allowing seamless transition between simulated Pygame physics and physical actuation.

---

## 📁 Project Structure

```text
Files/
├── agent_ddpg.py          # DDPG algorithm implementation and actor/critic logic
├── agent_sac.py           # SAC algorithm implementation with entropy tuning
├── buffer.py              # Replay buffer for RL off-policy training
├── controller_runner.py   # Multiprocessing wrapper for real-time control execution
├── environment.py         # Physics simulation and hardware interaction wrapper
├── LQR.py                 # Linear Quadratic Regulator matrix solvers and logic
├── main_ddpg.py           # Standalone script for training the DDPG agent
├── main_sac.py            # Standalone script for training the SAC agent
├── networks.py            # PyTorch neural network architectures for RL agents
├── plotting.py            # PyQtGraph real-time plotting utilities
├── test_ddpg.py           # Script to evaluate pre-trained DDPG models
├── test_sac.py            # Script to evaluate pre-trained SAC models
│
├── Models_agents/         # Directory containing trained PyTorch model weights (.pth)
│   ├── ddpg/
│   ├── Modelo_con_entropia_final/
│   └── Pendulo_Entrenamiento_Nuevo/
│
├── videos/                # Demonstration videos of controllers in action
│   ├── DDPG.mp4
│   ├── lqr_pendulum.mp4
│   └── SAC.mp4
│
├── inverted_pendulum_arduino.ino # C++ Firmware for the Arduino microcontroller
├── main_GUI.py            # Primary entry point launching the PyQt5 Dashboard
└── requirements.txt       # Python package dependencies
```

---

## 🏋️ Training RL Agents

The system supports end-to-end training of reinforcement learning models:

1. **SAC / DDPG Training:** You can train models using `main_sac.py` / `main_ddpg.py` or directly through the GUI.
2. **Episodes:** Training is structured in episodes. The environment automatically resets after 400 steps (aproximately 15 seconds).
3. **Model Saving:** Checkpoints are automatically saved periodically. Final weights are exported to the `Models_agents/` directory, allowing them to be instantly loaded into the GUI for real-time deployment.

---

## ⚠️ Notes & Troubleshooting

- **Serial Port Issues:** If the system fails to connect to the hardware, verify the correct COM port is selected in the GUI and ensure the Arduino IDE Serial Monitor is closed.
- **Missing Models:** If you receive an error when selecting SAC or DDPG in the GUI, ensure the pre-trained weights are correctly located in the `Models_agents/` subdirectories.

---

## 📜 Credits & Academic Context

This framework was developed in **hotbed PRIMBIO** with the context of advanced control systems, robotics, and reinforcement learning research . It serves as a comprehensive tool to evaluate, compare, and demonstrate the efficacy of classic algebraic control vs. modern, data-driven optimization strategies in highly unstable, non-linear environments.

## 🎓 University National of Colombia

**Sede de La Paz - hotbed PRIMBIO**
