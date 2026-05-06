import pygame
import math
import serial
import time
import sys

class LQRVisualEnvironment:
    def __init__(self, port='COM4', baudrate=115200, win_id=None):
        if win_id is not None:
            import os
            os.environ['SDL_WINDOWID'] = str(win_id)
            os.environ['SDL_VIDEODRIVER'] = 'windows'

        # --- Configuración de Pygame ---
        pygame.init()
        self.WIDTH, self.HEIGHT = (400, 350) if win_id is not None else (918, 768)
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Control LQR + Swing-up en Tiempo Real")
        self.clock = pygame.time.Clock()
        self.font_small = pygame.font.Font(None, int(36 * (0.6 if win_id is not None else 1.0)))
        self.font_large = pygame.font.Font(None, int(72 * (0.6 if win_id is not None else 1.0)))

        # Colores
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.BLUE = (0, 0, 255)
        self.GREEN = (0, 200, 0)

        # Cargar imágenes (Fondo y Carrito)
        try:
            self.background_img = pygame.image.load("images/918-768.png").convert()
            self.background_img = pygame.transform.scale(self.background_img, (self.WIDTH, self.HEIGHT))
        except pygame.error:
            print("Aviso: No se pudo cargar 'images/918-768.png'. Usando fondo blanco.")
            self.background_img = None

        scale = 0.45 if win_id is not None else 1.0
        self.CART_WIDTH = int(70 * scale)
        self.CART_HEIGHT = int(110 * scale)
        try:
            self.cart_img = pygame.image.load("images/Carrito.png").convert_alpha()
            self.cart_img = pygame.transform.scale(self.cart_img, (self.CART_WIDTH, self.CART_HEIGHT))
        except pygame.error:
            print("Aviso: No se pudo cargar 'images/Carrito.png'. Usando rectángulo azul.")
            self.cart_img = None

        self.PENDULUM_LENGTH = int(200 * scale)
        self.RACK_LENGTH = int(600 * scale)

        # --- Configuración Serial ---
        try:
            self.ser = serial.Serial(port, baudrate, timeout=0.1)
            print(f"Conectado a {port}")
        except Exception as e:
            print(f"Error crítico al conectar: {e}")
            sys.exit()
        time.sleep(2)  

        # --- Parámetros Físicos y Control LQR ---
        self.g = 9.81
        self.mp = 0.097
        self.lp = 0.2
        self.mplp = self.mp * self.lp
        self.Jp = 0.00517333
        self.INERTIA_EQ = self.Jp + self.mplp * self.lp
        self.MGL = self.mplp * self.g
        self.desired_energy = 2 * self.MGL
        
        self.MOTOR_PPR = 2400
        self.SHAFT_R = 1.2 
        self.K = [1600, 140, -13, -7.5]
        self.k_swingup = 1.5  
        self.theta_threshold = math.radians(12) 
        self.angle_setpoint = math.pi # En el .ino original, PI (180°) es arriba
        self.pos_limit_pulses = 5000
        self.startup_kick_voltage = 2.2 # V
        self.startup_kick_max_steps = 12 # 12 ciclos a 100 Hz = 120 ms
        self.startup_kick_steps = 0
        self.startup_window_steps = 200 # ventana de 2 s para permitir el impulso
        self.startup_w_threshold = 0.08 # rad/s
        
        self.state = {"pos_cm": 0, "angle_rad": 0, "vel_cm_s": 0, "w_rad_s": 0, "raw_pulses": 0, "raw_angle_deg": 0}
        self.is_paused = False
        self.current_voltage = 0.0

    def avoidStall(self, u):
        """Réplica exacta de la función del .ino en C++ para vencer la inercia"""
        MAX_STALL_U = 90.0
        if abs(u) < MAX_STALL_U:
            if u > 0:
                return 2.0 + MAX_STALL_U
            elif u < 0:
                return -2.0 - MAX_STALL_U
        return u

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.send_voltage(0)
            self.current_voltage = 0.0

    def pulses_to_meters(self, pulses):
        return 2.0 * math.pi * pulses / self.MOTOR_PPR * self.SHAFT_R

    def read_state(self):
        self.ser.reset_input_buffer() 
        self.ser.write(b'R\n')
        line = self.ser.readline().decode('utf-8').strip()
        
        if line:
            try:
                parts = line.split(',')
                if len(parts) == 4:
                    raw_pos = float(parts[0])
                    raw_angle_deg = float(parts[1]) # El esclavo envía 0° arriba
                    raw_vel = float(parts[2])
                    raw_angular_speed = float(parts[3])
                    
                    self.state["raw_pulses"] = raw_pos
                    self.state["raw_angle_deg"] = raw_angle_deg
                    
                    # 1. Posición y Velocidad en Centímetros
                    self.state["pos_cm"] = (2.0 * math.pi * raw_pos / self.MOTOR_PPR) * self.SHAFT_R
                    self.state["vel_cm_s"] = (2.0 * math.pi * raw_vel / self.MOTOR_PPR) * self.SHAFT_R
                    
                    # 2. Reconstruir el "theta" del .ino original (Donde PI es arriba y crece inverso)
                    theta_cpp = math.radians(-(raw_angle_deg - 180.0))
                    theta_cpp = math.fmod(theta_cpp, 2.0 * math.pi)
                    if theta_cpp < 0: 
                        theta_cpp += 2.0 * math.pi
                        
                    self.state["angle_rad"] = theta_cpp
                    self.state["w_rad_s"] = math.radians(-raw_angular_speed)
                    return True
            except ValueError:
                pass
        return False

    def send_voltage(self, voltage):
        self.current_voltage = voltage
        msg = f"0{voltage:.2f}\n"
        self.ser.write(msg.encode('utf-8'))

    def compute_control(self):
        theta = self.state["angle_rad"]
        w = self.state["w_rad_s"]
        x = self.state["pos_cm"]
        v = self.state["vel_cm_s"]
        raw_pulses = self.state["raw_pulses"]

        if self.startup_window_steps > 0:
            self.startup_window_steps -= 1

            near_bottom = (theta > (2 * math.pi - 0.35) or theta < 0.35)
            near_rest = abs(w) < self.startup_w_threshold

            if near_bottom and near_rest and self.startup_kick_steps < self.startup_kick_max_steps:
                self.startup_kick_steps += 1
                kick_dir = -1.0 if raw_pulses > 0 else 1.0
                return kick_dir * self.startup_kick_voltage

        

        if abs(raw_pulses) > self.pos_limit_pulses:
            return 12.0 if raw_pulses <= 0 else -12.0 

        # En el .ino original THETA_THRESHOLD es (PI / 12) que son 15 grados.
        # Evaluamos si estamos cerca del setpoint (PI)
        if abs(self.angle_setpoint - theta) < self.theta_threshold:
            # Ecuación LQR idéntica al C++
            u_lqr = (self.K[0] * (self.angle_setpoint - theta) - 
                     self.K[1] * w + 
                     self.K[2] * (0 - x) - 
                     self.K[3] * v)
            
            # Aplicar fricción y saturación (rango de PWM Arduino)
            u_pwm = self.avoidStall(u_lqr)
            u_pwm = max(min(u_pwm, 255.0), -255.0)
            
            # Convertir el PWM resultante al Voltaje que espera tu esclavo (-12 a 12)
            vol_u = u_pwm * (12.0 / 255.0) 

            # print(f"Theta={math.degrees(theta):.2f}°, W={w:.2f} rad/s, X={x:.3f} cm, V={v:.3f} cm/s, Pulses={raw_pulses:.0f}, voltaje={vol_u:.2f} V")
            
            return vol_u # Retornamos positivo. Los signos ya cuadran.
            
        else:
            # Lógica Swing-up idéntica al .ino original
            current_energy = 0.5 * self.INERTIA_EQ * (w**2) + self.MGL * (1 - math.cos(theta))
            
            if (theta > (2*math.pi - 0.28) or theta < 0.28) and current_energy < 0.85:
                accel = -300 * self.k_swingup * abs(current_energy - self.desired_energy) * w
                u_pwm = self.avoidStall(accel)
                u_pwm = max(min(u_pwm, 255.0), -255.0)
                vol_u = u_pwm * (12.0 / 255.0)
                return vol_u
            else:
                return 0.0


    def saturate(self, u):
        return max(min(u, 10.0), -10.0)

    def render(self):
        # Dibujar Fondo
        if self.background_img:
            self.screen.blit(self.background_img, (0, 0))
        else:
            self.screen.fill(self.WHITE)

        # Dibujar Péndulo (Usando los valores extraídos en raw_pulses y angle_rad)
        raw_angle_rad = math.radians(self.state["raw_angle_deg"])
        self._draw_pendulum(self.state["raw_pulses"], raw_angle_rad)

        # Renderizar Texto de Datos
        # info_text = self.font_small.render(
        #     f"Pos (Pulsos): {self.state['raw_pulses']:.0f} | Ángulo (Deg): {self.state['raw_angle_deg']:.2f}°", 
        #     True, self.BLACK)
        # speed_text = self.font_small.render(
        #     f"Vel (cm/s): {self.state['vel_cm_s']:.3f} | W (rad/s): {self.state['w_rad_s']:.2f}",
        #     True, self.BLACK)
        # volt_text = self.font_small.render(
        #     f"Voltaje Motor: {self.current_voltage:.2f} V", 
        #     True, self.BLUE if not self.is_paused else self.RED)

        # self.screen.blit(info_text, (10, self.HEIGHT - 100))
        # self.screen.blit(speed_text, (10, self.HEIGHT - 70))
        # self.screen.blit(volt_text, (10, self.HEIGHT - 40))

        # Indicador de Estado LQR
        modo_actual = "LQR ESTABILIZANDO" if abs(self.angle_setpoint - self.state["angle_rad"]) < self.theta_threshold else "SWING-UP"
        modo_color = self.GREEN if abs(self.state["angle_rad"]) < self.theta_threshold else self.BLACK
        modo_text = self.font_small.render(f"Estado Lógica: {modo_actual}", True, modo_color)
        self.screen.blit(modo_text, (self.WIDTH - 450, 10))

        # Renderizar Estado de Pausa
        if self.is_paused:
            pause_text = self.font_large.render("SISTEMA PAUSADO", True, self.RED)
            text_rect = pause_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/4))
            self.screen.blit(pause_text, text_rect)
            
            resume_text = self.font_small.render("Presiona ESPACIO para continuar", True, self.BLACK)
            resume_rect = resume_text.get_rect(center=(self.WIDTH/2, self.HEIGHT/4 + 50))
            self.screen.blit(resume_text, resume_rect)

        pygame.display.flip()

    def _draw_pendulum(self, raw_pulses, angle_rad):
        # Mapear posición (Ajustado para coincidir con tu environment.py original)
        cart_x = self.WIDTH // 2 + (raw_pulses / 12000) * self.RACK_LENGTH
        cart_y = self.HEIGHT // 2 + int(100 * (0.45 if self.WIDTH == 400 else 1.0))

        # Riel
        pygame.draw.line(self.screen, self.BLACK,
                         (self.WIDTH // 2 - self.RACK_LENGTH // 2, cart_y + self.CART_HEIGHT // 2), 
                         (self.WIDTH // 2 + self.RACK_LENGTH // 2, cart_y + self.CART_HEIGHT // 2), 5)
        
        # Carrito
        if self.cart_img:
            self.screen.blit(self.cart_img, (cart_x - self.CART_WIDTH // 2, cart_y - 33)) 
        else:
            pygame.draw.rect(self.screen, self.BLUE,
                             (cart_x - self.CART_WIDTH // 2, cart_y, self.CART_WIDTH, self.CART_HEIGHT))
        
        # Vara del péndulo (Usando Sen y Cos con el ángulo en radianes)
        end_x = cart_x + self.PENDULUM_LENGTH * math.sin(angle_rad)
        end_y = cart_y - self.PENDULUM_LENGTH * math.cos(angle_rad)
        
        pygame.draw.line(self.screen, self.RED, (cart_x, cart_y), (end_x, end_y), 5)
        pygame.draw.circle(self.screen, self.RED, (int(end_x), int(end_y)), 10)

    def run(self):
        running = True
        while running:
            # 1. Captura de Eventos UI
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.toggle_pause()

            # 2. Bucle de Control Físico
            if not self.is_paused:
                if self.read_state():
                    u = self.compute_control()
                    self.send_voltage(u)
            else:
                self.ser.reset_input_buffer() # Evita acumular latencia mientras está pausado

            # 3. Renderizado Visual
            self.render()

            # Forzar el bucle a 100Hz (Coincidente con la resolución del diferencial de tiempo dt)
            self.clock.tick(100)

        # 4. Apagado Seguro
        self.send_voltage(0)
        self.ser.close()
        pygame.quit()
        print("Programa terminado correctamente.")

if __name__ == "__main__":
    env = LQRVisualEnvironment()
    env.run()