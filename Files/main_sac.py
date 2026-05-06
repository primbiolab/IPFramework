from Files.environment import InvertedPendulumEnv, check_pause, wait_if_paused
import numpy as np
from Files.agent_sac import Agent
from Files.plotting import plot_learning_curve, plot_episode_variables, plot_scores_scatter
import time
import math
import threading


if __name__ == '__main__':
    
    threading.Thread(target=check_pause, daemon=True).start()

    env = InvertedPendulumEnv()
    state = env.reset()
    chkpt_dir = 'Modelos_Finales/Pendulo_1/Modelo_con_entropia_decreciente_07'
    agent = Agent(input_dims=[5], env=env,
            n_actions=1, chkpt_dir=chkpt_dir)
    n_games = 1200
    filename = 'Modelo_recompensa_con_condicion_.png'

    figure_file = filename

    best_score = -500
    print('Best score: ', best_score)
    score_history = []
    load_checkpoint = False
    #agent.load_models()
    env.render()

    for i in range(n_games):
        wait_if_paused()
        env._get_state()
        observation = env.reset()  ## Jk
        observation = [observation[0]/5000 , observation[2]/500,math.cos(math.radians(observation[1])),math.sin(math.radians(observation[1])), math.radians(observation[3])]
        done = False
        steps = 0
        score = 0
        
        passed_vertical = False

        if (i+1) % 100 == 0 and not load_checkpoint:
            agent.entropy = max(0.2, agent.entropy - 0.3)
            print(f"Reduciendo entropía, nuevo alpha: {agent.entropy}")
        
        while not done:
            #i_time = time.time()
            steps += 1
            if load_checkpoint:
                action = agent.choose_action(observation,deterministic=True)
            else:
                action = agent.choose_action(observation)

            observation_, reward, done, info = env.step(action)

            if (steps == (400) and not load_checkpoint):
                done = True
            
            observation_ = [observation_[0]/5000 , observation_[2]/500,math.cos(math.radians(observation_[1])),math.sin(math.radians(observation_[1])), math.radians(observation_[3])]
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            env.render()
            observation = observation_
            
        
        score_history.append(score)
        avg_score = np.mean(score_history[-50:])
        env.center()
        if (not load_checkpoint):
            for j in range(steps):
                agent.learn()
        
        
        
        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score, 'best_score %.1f' % best_score)

    if not load_checkpoint:
        x = [i+1 for i in range(len(score_history))]
        plot_learning_curve(x, score_history, figure_file)
        plot_scores_scatter(x, score_history, "scores_scatter.png")