import math
import time
import numpy as np
from Files.environment import InvertedPendulumEnv
from Files.agent_ddpg import DDPGAgent
from Files.plotting import plot_episode_variables

def preprocess(obs):
    return [obs[0]/5000,
            obs[2]/500,
            math.cos(math.radians(obs[1])),
            math.sin(math.radians(obs[1])),
            math.radians(obs[3])]

if __name__ == '__main__':
    env = InvertedPendulumEnv()
    chkpt_dir = 'Models_agents/ddpg'
    agent = DDPGAgent(input_dims=[5], n_actions=1, env=env, chkpt_dir=chkpt_dir)
    agent.actor.load_checkpoint()   # cargar pesos guardados
    agent.critic.load_checkpoint()

    time_list = []
    pos_list = []
    vel_list = []
    ang_list = []
    ang_vel_list = []
    accion = []

    pos_list.clear()
    vel_list.clear()
    ang_list.clear()
    ang_vel_list.clear()
    time_list.clear()
    start_time = time.time()
        
    obs = env.reset()
    state = preprocess(obs)
    done = False
    total_reward = 0
    step = 0
    while not done:
        current_time = time.time() - start_time - 2.0
        step += 1
        action = agent.choose_action(state, noise=0.0)  # sin exploración
        obs_, reward, done, _ = env.step(action)
        state = preprocess(obs_)
        total_reward += reward
        env.render()

        pos_list.append(obs_[0])         # posición del carrito
        vel_list.append(obs_[2])         # velocidad del carrito
        ang_list.append(obs_[1])         # ángulo del péndulo (grados)
        ang_vel_list.append(obs_[3])     # velocidad angular (grados/seg)
        accion.append(action)
        time_list.append(current_time)   

        if current_time >= 20:
            done = True
    env.center()

    t = np.array(time_list)
    plot_episode_variables(t, pos_list, vel_list, ang_list, ang_vel_list, accion, csv_file="ddpg_1_pendulo_2.csv")
    print(f'[TEST] Episode {1} → Reward: {total_reward:.2f}')
