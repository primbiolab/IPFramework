from Files.environment import InvertedPendulumEnv
import numpy as np
from Files.agent_ddpg import DDPGAgent
from Files.plotting import plot_learning_curve, plot_episode_variables, plot_scores_scatter
import time
import math



if __name__ == '__main__':
    
    env = InvertedPendulumEnv()
    state = env.reset()
    chkpt_dir = 'Modelos_Finales/Pendulo_1/DDPG'
    agent = DDPGAgent(input_dims=[5], env=env,
            n_actions=1, chkpt_dir=chkpt_dir)
    n_games = 2499
    filename = 'inverted_pendulum.png'

    

    figure_file = filename

    best_score = -1000
    print('Best score: ', best_score)
    score_history = []
    load_checkpoint = False
    #agent.load_models()
    env.render()
    
    pos_list = []
    vel_list = []
    ang_list = []
    ang_vel_list = []
    accion = []

    for i in range(n_games):
        env._get_state()
        observation = env.reset()  ## Jk
        observation = [observation[0]/5000 , observation[2]/500,math.cos(math.radians(observation[1])),math.sin(math.radians(observation[1])), math.radians(observation[3])]
        done = False
        steps = 0
        score = 0

        if load_checkpoint:
            pos_list.clear()
            vel_list.clear()
            ang_list.clear()
            ang_vel_list.clear()
            test_start_time = time.time()
    

        while not done:
            i_time = time.time()
            steps += 1

            action = agent.choose_action(observation)
            new_obs, reward, done, _ = env.step(action)
            new_state = [new_obs[0]/5000, new_obs[2]/500, math.cos(math.radians(new_obs[1])),
                         math.sin(math.radians(new_obs[1])), math.radians(new_obs[3])]
            agent.remember(observation, action, reward, new_state, done)
            agent.learn()
            env.render()
            observation = new_state
            score += reward

            if (steps == 400 and not load_checkpoint):
                done = True

            
            
            
        
        score_history.append(score)
        avg_score = np.mean(score_history[-50:])
        env.center()
        # if (not load_checkpoint):
        #     for j in range(steps):
        #         agent.learn()
        
        
        
        if avg_score > best_score:
            best_score = avg_score
            agent.actor.save_checkpoint()
            agent.critic.save_checkpoint()
        print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score, 'best_score %.1f' % best_score)

    # Plot training curve
    x = [i+1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, 'ddpg_training.png')