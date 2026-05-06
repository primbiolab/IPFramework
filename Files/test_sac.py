from Files.environment import InvertedPendulumEnv
import numpy as np
from Files.agent_sac import Agent
from Files.plotting import plot_learning_curve, plot_episode_variables, plot_scores_scatter
import time
import math



if __name__ == '__main__':
    
    env = InvertedPendulumEnv()
    state = env.reset()
    chkpt_dir = 'Models_agents/Modelo_con_entropia_final'
    agent = Agent(input_dims=[5], env=env,
            n_actions=1, chkpt_dir=chkpt_dir)

    score_history = []
    load_checkpoint = True
    agent.load_models()
    env.render()
    
    time_list = []
    pos_list = []
    vel_list = []
    ang_list = []
    ang_vel_list = []
    accion = []

    env._get_state()
    observation = env.reset()  ## Jk
    observation = [observation[0]/5000 , observation[2]/500,math.cos(math.radians(observation[1])),math.sin(math.radians(observation[1])), math.radians(observation[3])]
    done = False
    steps = 0
    score = 0
        
    passed_vertical = False

    pos_list.clear()
    vel_list.clear()
    ang_list.clear()
    ang_vel_list.clear()
    time_list.clear()
    start_time = time.time()

    while not done:
        current_time = time.time() - start_time
        steps += 1
        action = agent.choose_action(observation,deterministic=True)
        observation_, reward, done, info = env.step(action)

        pos_list.append(observation_[0])         # posición del carrito
        vel_list.append(observation_[2])         # velocidad del carrito
        ang_list.append(observation_[1])         # ángulo del péndulo (grados)
        ang_vel_list.append(observation_[3])     # velocidad angular (grados/seg)
        accion.append(action)
        time_list.append(current_time)   
        if current_time >= 20:
            done = True

            
        observation_ = [observation_[0]/5000 , observation_[2]/500,math.cos(math.radians(observation_[1])),math.sin(math.radians(observation_[1])), math.radians(observation_[3])]
        score += reward
        agent.remember(observation, action, reward, observation_, done)
        env.render()
        observation = observation_
            
        
    score_history.append(score)
    avg_score = np.mean(score_history[-50:])
    env.center()

    t = np.array(time_list)

    plot_episode_variables(t, pos_list, vel_list, ang_list, ang_vel_list, accion, csv_file="sac_3_pendulo_2.csv")