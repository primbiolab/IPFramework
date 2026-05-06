import pygame
import math
import serial
import time
import numpy as np
from gym import spaces
import logging
import threading

pause_training = False

def check_pause():
    global pause_training
    while True:
        cmd = input()   
        if cmd.strip().lower() == 'p':
            pause_training = not pause_training

def wait_if_paused():
    global pause_training
    while pause_training:
        time.sleep(0.5)
        
class InvertedPendulumEnv:
    def __init__(self, port="COM4", win_id=None):
        if win_id is not None:
            import os
            os.environ['SDL_WINDOWID'] = str(win_id)
            os.environ['SDL_VIDEODRIVER'] = 'windows'

        pygame.init()

        self.WIDTH, self.HEIGHT = (400, 350) if win_id is not None else (918, 768)
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Péndulo Invertido")

        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)

        # Fondo personalizado
        try:
            self.background_img = pygame.image.load("images/918-768.png").convert()
            self.background_img = pygame.transform.scale(self.background_img, (self.WIDTH, self.HEIGHT))
        except pygame.error as e:
            print(f"No se pudo cargar la imagen de fondo: {e}")
            self.background_img = None

        # Imagen del carrito
        scale = 0.45 if win_id is not None else 1.0
        self.CART_WIDTH = int(70 * scale)
        self.CART_HEIGHT = int(110 * scale)
        try:
            self.cart_img = pygame.image.load("images/Carrito.png").convert_alpha()
            self.cart_img = pygame.transform.scale(self.cart_img, (self.CART_WIDTH, self.CART_HEIGHT))
        except pygame.error as e:
            print(f"No se pudo cargar la imagen del carrito: {e}")
            self.cart_img = None

        self.PENDULUM_LENGTH = int(200 * scale)
        self.RACK_LENGTH = int(600 * scale)

        # Serial communication
        self.arduino_port = port

        self.baud_rate = 115200
        self.ser = serial.Serial(self.arduino_port, self.baud_rate, timeout=1)
        time.sleep(2)

        
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.state = None
        self.reward_range = (-10, 10)
        self.clock = pygame.time.Clock()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def center(self):
        self.ser.write(b"10\n")  
    
    def reset(self):
        # Mueve el péndulo al centro
        self.ser.write(b"10\n")
        self.ser.write(b'R\n')
        self.state = self._get_state()

        # Espera hasta que el péndulo esté abajo
        start_time = None
        while True:
            self.ser.write(b'R\n')
            self.state = self._get_state()
            angle = self.state[1]

            if 178 <= angle <= 182 or -182 <= angle <= -178:
                if start_time is None:
                    start_time = time.time()
                elif time.time() - start_time >= 2.0:
                    break  
            else:
                start_time = None  
            time.sleep(0.01)  

        return self.state

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)*10
        action = np.clip(action, -10, 10)

    
        # Apply action
        self.ser.write(f"0{action[0]:.2f}\n".encode())  #'0' is voltage control mode
        # Wait for a short time to allow the system to respond
        time.sleep(0.027)
        
        for i in range(5): #try many times (some data was getting accumulated because of missed readings)
            if self.ser.in_waiting:
                try:
                    data_waste = self.ser.readline().decode().strip().split(',')    
                except:
                    pass
        self.ser.write(b'R\n')
        new_state = self._get_state()

        
        done = False
        reward = self._calculate_reward([new_state[0]/5000 , new_state[2]/500,math.cos(math.radians(new_state[1])),math.sin(math.radians(new_state[1])), math.radians(new_state[3])],done)
        
        self.state = new_state
        
        return new_state, reward, done, {}

    def render(self):
        if self.background_img:
            self.screen.blit(self.background_img, (0, 0))
        else:
            self.screen.fill(self.WHITE)
        self._draw_pendulum(self.state[0], self.state[1])
        font = pygame.font.Font(None, 36)
        info_text = font.render(f"Pos: {self.state[0]:.2f}, Angle: {self.state[1]:.2f}", True, self.BLACK)
        speed_text = font.render(f"Speed: {self.state[2]:.2f}, Angular Speed: {self.state[3]:.2f}", True, self.BLACK)
        self.screen.blit(info_text, (10, self.HEIGHT - 70))
        self.screen.blit(speed_text, (10, self.HEIGHT - 40))

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        self.ser.close()
        pygame.quit()

    def _get_state(self):
        for i in range(10): #try many times
            pygame.event.pump()  # Mantiene la ventana activa
            if self.ser.in_waiting:
                try:
                    data = self.ser.readline().decode().strip().split(',')
                    if len(data) == 4:                  
                        return list(map(float, data))
                except:
                    pass
            time.sleep(0.001)
        return [0, 0, 0, 0]  # Return a default state if unable to read

    def _calculate_reward(self, observation, done, angle_deg=None):
        x, vx, cos, sin, theta_dot = observation
        reward = cos
        reward -= 0.003* (theta_dot ** 2)
        reward -= 0.06 * abs(x)
        # Bonificación arriba
        if cos > 0.99 and abs(theta_dot) < 0.2 and abs(x) < 0.2:
            reward += 5
        return reward

    def _draw_pendulum(self, x, angle):
        # Escala la posición del carrito si la pantalla es menor
        cart_x = self.WIDTH // 2 + (x / 12000) * self.RACK_LENGTH
        cart_y = self.HEIGHT // 2 + int(100 * (0.45 if self.WIDTH == 400 else 1.0))
        pygame.draw.line(self.screen, self.BLACK,
                         (self.WIDTH // 2 - self.RACK_LENGTH // 2, cart_y + self.CART_HEIGHT // 2), #Dibuja el riel
                         (self.WIDTH // 2 + self.RACK_LENGTH // 2, cart_y + self.CART_HEIGHT // 2), 5)
        if self.cart_img:
            self.screen.blit(self.cart_img, (cart_x - self.CART_WIDTH // 2, cart_y-33)) #Cambia la posición del carrito
        else:
            pygame.draw.rect(self.screen, self.BLUE,
                             (cart_x - self.CART_WIDTH // 2, cart_y, self.CART_WIDTH, self.CART_HEIGHT))
        end_x = cart_x + self.PENDULUM_LENGTH * math.sin(math.radians(angle))
        end_y = cart_y - self.PENDULUM_LENGTH * math.cos(math.radians(angle))
        pygame.draw.line(self.screen, self.RED, (cart_x, cart_y), (end_x, end_y), 5)
        pygame.draw.circle(self.screen, self.RED, (int(end_x), int(end_y)), 10)
        
        


def manual_control(env):
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    input_box = pygame.Rect(10, 10, 140, 32)
    color_inactive = pygame.Color('lightskyblue3')
    color_active = pygame.Color('dodgerblue2')
    color = color_inactive
    active = False
    text = ''
    mode = '1'  # Start in position control mode

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if input_box.collidepoint(event.pos):
                    active = not active
                else:
                    active = False
                color = color_active if active else color_inactive
            if event.type == pygame.KEYDOWN:
                if active:
                    if event.key == pygame.K_RETURN:
                        env.ser.write(f"{mode}{text}\n".encode())
                        text = ''
                    elif event.key == pygame.K_BACKSPACE:
                        text = text[:-1]
                    else:
                        text += event.unicode
                if event.key == pygame.K_SPACE:
                    mode = '0' if mode == '1' else '1'
        if env.background_img:
            env.screen.blit(env.background_img, (0, 0))
        else:
            env.screen.fill(env.WHITE)

        # Read and display state
        env.ser.write(b'R\n')
        state = env._get_state()
        env._draw_pendulum(state[0], state[1])
        # Display data
        info_text = font.render(f"Pos: {state[0]:.2f}, Angle: {state[1]:.2f}", True, env.BLACK)
        speed_text = font.render(f"Speed: {state[2]:.2f}, Angular Speed: {state[3]:.2f}", True, env.BLACK)
        env.screen.blit(info_text, (10, env.HEIGHT - 70))
        env.screen.blit(speed_text, (10, env.HEIGHT - 40))

        # Draw input box
        txt_surface = font.render(text, True, color)
        width = max(200, txt_surface.get_width()+10)
        input_box.w = width
        env.screen.blit(txt_surface, (input_box.x+5, input_box.y+5))
        pygame.draw.rect(env.screen, color, input_box, 2)

        # Display current mode
        mode_text = font.render(f"Mode: {'Position' if mode == '1' else 'Voltage'}", True, env.BLACK)
        env.screen.blit(mode_text, (env.WIDTH - 200, 10))

        pygame.display.flip()
        clock.tick(60)

    env.close()


if __name__ == "__main__":
    env = InvertedPendulumEnv()
    manual_control(env)