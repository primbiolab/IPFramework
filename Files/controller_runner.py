import time
import math
import pygame
import numpy as np
# Importar tus clases existentes
from Files.LQR import LQRVisualEnvironment
from Files.environment import InvertedPendulumEnv
from Files.agent_sac import Agent as SACAgent
from Files.agent_ddpg import DDPGAgent # Descomentar cuando tengas el archivo

def run_training_loop(agent_type, save_dir, n_episodes, stop_event, data_queue, port="COM4", win_id=None):
    """
    Bucle de entrenamiento de RL en un proceso separado.
    """
    print(f"Iniciando Entrenamiento de {agent_type} por {n_episodes} episodios en {save_dir} (Puerto: {port})")
    try:
        env = InvertedPendulumEnv(port=port, win_id=win_id)
        raw_obs = env.reset()

        # Instanciar el agente correspondiente
        if agent_type == "SAC":
            agent = SACAgent(input_dims=[5], env=env, n_actions=1, chkpt_dir=save_dir)
        elif agent_type == "DDPG":
            agent = DDPGAgent(input_dims=[5], env=env, n_actions=1, chkpt_dir=save_dir)
        else:
            raise ValueError("Agente no soportado")

        best_score = -500
        score_history = []

        for i in range(n_episodes):
            if stop_event.is_set():
                break # Salir si el usuario presiona "Detener"

            raw_obs = env.reset()
            obs = [raw_obs[0]/5000, raw_obs[2]/500, math.cos(math.radians(raw_obs[1])), 
                   math.sin(math.radians(raw_obs[1])), math.radians(raw_obs[3])]
            
            done = False
            steps = 0
            score = 0

            # Reducción de entropía (Solo SAC)
            if agent_type == "SAC" and (i + 1) % 100 == 0:
                agent.entropy = max(0.2, agent.entropy - 0.3)
                print(f"Reduciendo entropía, nuevo alpha: {agent.entropy}")

            # --- Bucle del Episodio ---
            while not done and not stop_event.is_set():
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        stop_event.set()

                steps += 1
                action = agent.choose_action(obs) # No determinístico para explorar

                new_raw_obs, reward, done, _ = env.step(action)
                
                if steps >= 400:
                    done = True

                new_obs = [new_raw_obs[0]/5000, new_raw_obs[2]/500, math.cos(math.radians(new_raw_obs[1])), 
                           math.sin(math.radians(new_raw_obs[1])), math.radians(new_raw_obs[3])]
                
                score += reward
                agent.remember(obs, action, reward, new_obs, done)
                
                # Enviar telemetría a la GUI
                data_queue.put({
                    'mode': 'train',
                    'episode': i + 1,
                    'score': score,
                    'pos': (2.0 * math.pi * new_raw_obs[0] / 2400) * 1.2,
                    'vel_pos': (2.0 * math.pi * new_raw_obs[2] / 2400) * 1.2,
                    'angle': new_raw_obs[1],
                    'vel_angle': (new_raw_obs[3] * (math.pi/180)),
                    'action': float(action[0]*10)
                })

                env.render()
                obs = new_obs
            
            # --- Fin del Episodio: Actualizar y Entrenar ---
            if not stop_event.is_set():
                score_history.append(score)
                avg_score = np.mean(score_history[-50:]) if len(score_history) > 0 else score
                env.center() # Centrar físicamente el péndulo (desde tu main_sac)

                # Notificar a la GUI la fase de entrenamiento
                data_queue.put({'status_msg': f"Entrenando redes (Episodio {i+1})..."})

                # Entrenamiento de las redes (es vital chequear stop_event aquí por si toma tiempo)
                for _ in range(steps):
                    if stop_event.is_set(): break
                    agent.learn()
                
                # Guardar el mejor modelo
                if avg_score > best_score and not stop_event.is_set():
                    best_score = avg_score
                    agent.save_models()
                    print(f"Mejor modelo guardado! Avg Score: {best_score:.1f}")

                data_queue.put({'status_msg': f"Episodio {i+1} completado. Score: {score:.1f}"})

    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")
    finally:
        if 'env' in locals():
            env.ser.write(b"00.00\n") 
            env.close() 
        print(f"Entrenamiento {agent_type} Finalizado y puerto cerrado.")

def run_controller(controller_name, stop_event, data_queue, model_path=None, command_queue=None, port="COM4", win_id=None):
    """
    Se añade el argumento model_path para cargar modelos específicos y port para puerto serial.
    """
    print(f"Iniciando proceso para: {controller_name} en {port}")

    if controller_name == "LQR Clásico":
        run_lqr_loop(stop_event, data_queue, command_queue, port, win_id)
    elif controller_name == "SAC (Soft Actor-Critic)":
        run_rl_loop("SAC", stop_event, data_queue, model_path, port, win_id) # Pasar ruta
    elif controller_name == "DDPG":
        run_rl_loop("DDPG", stop_event, data_queue, model_path, port, win_id) # Pasar ruta

def run_lqr_loop(stop_event, data_queue, command_queue=None, port="COM4", win_id=None):
    try:
        # Instanciar tu entorno LQR tal cual lo tienes
        env = LQRVisualEnvironment(port=port, win_id=win_id)

        env.render()
        pygame.event.pump()
        time.sleep(1.0)

        running = True

        while running and not stop_event.is_set():
            if command_queue and not command_queue.empty():
                try:
                    cmd = command_queue.get_nowait()
                    if cmd.get('type') == 'update_k':
                        env.K = cmd['K']
                        print(f"Matriz K actualizada a: {env.K}")
                except Exception:
                    pass

            # Manejar eventos de Pygame (para que no se congele la ventana)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Lógica de Control LQR
            if env.read_state():
                u = env.compute_control()
                env.send_voltage(u)
                
                # Enviar datos a la GUI de PyQt5
                data_queue.put({
                    'pos': env.state["pos_cm"],
                    'vel_pos': env.state["vel_cm_s"],
                    'angle': env.state["raw_angle_deg"],
                    'vel_angle': env.state["w_rad_s"],
                    'action': env.current_voltage
                })

            env.render()
            env.clock.tick(100) # 100Hz

    except Exception as e:
        print(f"Error en LQR: {e}")
    finally:
        # Cierre SEGURO obligatorio
        if 'env' in locals():
            env.send_voltage(0)
            env.ser.close()
            pygame.quit()
        print("LQR Finalizado y puerto cerrado.")

def run_rl_loop(agent_type, stop_event, data_queue, model_path, port="COM4", win_id=None):
    try:
        # 1. Instanciar el entorno de Gym
        env = InvertedPendulumEnv(port=port, win_id=win_id)
        raw_obs = env.reset()
        
        


        # Formatear la observación como la esperan tus agentes
        obs = [raw_obs[0]/5000 , raw_obs[2]/500, math.cos(math.radians(raw_obs[1])),
               math.sin(math.radians(raw_obs[1])), math.radians(raw_obs[3])]
        
        env.render()
        pygame.event.pump()
        time.sleep(1.0)
        
        # 2. Cargar el agente correspondiente
        if agent_type == "SAC":
            # Asumiendo que has entrenado y quieres cargar modelos
            agent = SACAgent(input_dims=[5], env=env, n_actions=1, 
                             chkpt_dir=model_path)
            agent.load_models() # <-- Descomenta para usar el modelo entrenado
            deterministic = True
            
        elif agent_type == "DDPG":
            # Instanciar DDPG aquí
            agent = DDPGAgent(input_dims=[5], env=env, n_actions=1, 
                              chkpt_dir=model_path)
            agent.load_models() # <-- Descomenta para usar el modelo entrenado
        # 3. Bucle de ejecución RL
        running = True
        while running and not stop_event.is_set():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Elegir acción (deterministica para pruebas físicas)
            if agent_type == "SAC":
                action = agent.choose_action(obs, deterministic=True)
            elif agent_type == "DDPG":
                action = agent.choose_action(obs, noise=0.0)
            else:
                action = [0] # Placeholder

            # Aplicar acción al Arduino
            new_raw_obs, reward, done, _ = env.step(action)
            
            # Actualizar observación
            obs = [new_raw_obs[0]/5000, new_raw_obs[2]/500, 
                   math.cos(math.radians(new_raw_obs[1])), math.sin(math.radians(new_raw_obs[1])), 
                   math.radians(new_raw_obs[3])]

            # Enviar datos a la GUI
            data_queue.put({
                'pos': (2.0 * math.pi * new_raw_obs[0] / 2400) * 1.2,
                'vel_pos': (2.0 * math.pi * new_raw_obs[2] / 2400) * 1.2,
                'angle': new_raw_obs[1],
                'vel_angle': (new_raw_obs[3]* (math.pi/180)) ,
                'action': float(action[0]*10) # Asumiendo que mapeas la acción de [-1,1] a [-10,10]V
            })

            env.render()
            
            # RL loops usualmente son más lentos que LQR, ajusta si es necesario

    except Exception as e:
        print(f"Error en {agent_type}: {e}")
    finally:
        # Cierre SEGURO obligatorio
        if 'env' in locals():
            env.ser.write(b"00.00\n") # Detener motor (Voltaje 0)
            env.close() # Tu función close() ya hace ser.close() y pygame.quit()
        print(f"{agent_type} Finalizado y puerto cerrado.")