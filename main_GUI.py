import sys
import os
import multiprocessing
import time
from collections import deque
import pyqtgraph as pg
import csv

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QComboBox, QLabel, QGroupBox, 
                             QFormLayout, QLineEdit, QSpinBox, QDoubleSpinBox, QGridLayout,
                             QFrame, QSizePolicy, QFileDialog, QMessageBox)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QFont, QColor, QPalette

from Files.controller_runner import run_controller, run_training_loop

class PendulumGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Control Dashboard - Inverted Pendulum")
        self.setGeometry(50, 50, 1400, 950)
        
        self.apply_dark_theme()

        # Multiprocessing variables
        self.current_process = None
        self.stop_event = multiprocessing.Event()
        self.data_queue = multiprocessing.Queue()
        self.command_queue = multiprocessing.Queue()

        # Plotting variables
        self.max_plot_points = 200  # About 20 seconds of data at 10Hz
        self.time_data = deque(maxlen=self.max_plot_points)
        self.pos_data = deque(maxlen=self.max_plot_points)
        self.angle_data = deque(maxlen=self.max_plot_points)
        self.vel_pos_data = deque(maxlen=self.max_plot_points)
        self.vel_angle_data = deque(maxlen=self.max_plot_points)
        self.action_data = deque(maxlen=self.max_plot_points)
        
        self.run_start_time = 0
        self.data_log = {
            "time": [], "pos": [], "angle": [], "vel_pos": [], "vel_angle": [], "action": []
        }

        self.initUI()

        # Timer to update GUI with data from queue (10Hz)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_telemetry)
        self.timer.start(100)

    def apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow {
                background-color: #121212;
            }
            QLabel {
                color: #e0e0e0;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 14px;
            }
            QGroupBox {
                border: 1px solid #333344;
                border-radius: 8px;
                margin-top: 15px;
                font-weight: bold;
                color: #ffffff;
                background-color: #1e1e24;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 10px;
                color: #4CAF50;
                font-size: 14px;
            }
            QPushButton {
                background-color: #3e3e5e;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px;
                font-weight: bold;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #4e4e70;
            }
            QPushButton:pressed {
                background-color: #2e2e4e;
            }
            QPushButton:disabled {
                background-color: #2a2a35;
                color: #555555;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #2a2a35;
                color: #ffffff;
                border: 1px solid #444455;
                border-radius: 4px;
                padding: 6px;
                font-size: 13px;
            }
            QComboBox QAbstractItemView {
                background-color: #2a2a35;
                color: #ffffff;
                selection-background-color: #4CAF50;
                selection-color: #ffffff;
                border: 1px solid #444455;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {
                border: 1px solid #4CAF50;
            }
            QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
                background-color: #1a1a25;
                color: #555555;
            }
            QFrame#TelemetryPanel {
                background-color: #1e1e24;
                border-radius: 12px;
                padding: 15px;
                border: 1px solid #333344;
            }
            QFrame#DataCard {
                background-color: #252530;
                border-radius: 8px;
                padding: 10px;
                border: 1px solid #333344;
            }
        """)

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(25)

        # ==========================================
        # LEFT PANEL: CONTROLS & SIMULATION
        # ==========================================
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(15)
        
        # 1. Controller Selection
        group_ctrl = QGroupBox("⚙️ Main Controller")
        layout_ctrl = QVBoxLayout()
        layout_ctrl.setSpacing(12)
        
        # Serial Port Selection
        layout_port = QHBoxLayout()
        lbl_port = QLabel("Serial Port:")
        self.combo_port = QComboBox()
        self.combo_port.addItems(["COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", "COM10", "/dev/ttyUSB0", "/dev/ttyACM0"])
        self.combo_port.setCurrentText("COM4")
        self.combo_port.setEditable(True)
        self.combo_port.setToolTip("Select or type the serial port")
        layout_port.addWidget(lbl_port)
        layout_port.addWidget(self.combo_port)
        layout_ctrl.addLayout(layout_port)
        
        self.combo_controllers = QComboBox()
        self.combo_controllers.addItems(["Classic LQR", "SAC (Soft Actor-Critic)", "DDPG"])
        self.combo_controllers.setToolTip("Select the control algorithm to execute")
        layout_ctrl.addWidget(self.combo_controllers)

        self.widget_model_path = QWidget()
        layout_model_path = QHBoxLayout(self.widget_model_path)
        layout_model_path.setContentsMargins(0, 0, 0, 0)
        lbl_model = QLabel("Model Path:")
        self.line_load_path = QLineEdit('Models_agents/Modelo_con_entropia_final')
        layout_model_path.addWidget(lbl_model)
        layout_model_path.addWidget(self.line_load_path)
        layout_ctrl.addWidget(self.widget_model_path)
        
        layout_btns = QHBoxLayout()
        self.btn_start = QPushButton("▶ Start System")
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white;")
        self.btn_start.setToolTip("Start the selected controller")
        self.btn_start.clicked.connect(self.start_controller)
        
        self.btn_stop = QPushButton("⏹ Stop")
        self.btn_stop.setStyleSheet("background-color: #f44336; color: white;")
        self.btn_stop.setToolTip("Stop the current execution")
        self.btn_stop.clicked.connect(self.stop_controller)
        self.btn_stop.setEnabled(False)

        layout_btns.addWidget(self.btn_start)
        layout_btns.addWidget(self.btn_stop)
        layout_ctrl.addLayout(layout_btns)
        group_ctrl.setLayout(layout_ctrl)
        left_layout.addWidget(group_ctrl)

        # 2. LQR Parameters
        self.group_lqr = QGroupBox("🎛️ LQR Parameters")
        layout_lqr = QGridLayout()
        layout_lqr.setSpacing(10)
        
        self.spin_k1 = QDoubleSpinBox()
        self.spin_k1.setRange(-20000, 20000)
        self.spin_k1.setDecimals(2)
        self.spin_k1.setValue(1600)
        
        self.spin_k2 = QDoubleSpinBox()
        self.spin_k2.setRange(-20000, 20000)
        self.spin_k2.setDecimals(2)
        self.spin_k2.setValue(140)
        
        self.spin_k3 = QDoubleSpinBox()
        self.spin_k3.setRange(-20000, 20000)
        self.spin_k3.setDecimals(2)
        self.spin_k3.setValue(-13)
        
        self.spin_k4 = QDoubleSpinBox()
        self.spin_k4.setRange(-20000, 20000)
        self.spin_k4.setDecimals(2)
        self.spin_k4.setValue(-7.5)

        layout_lqr.addWidget(QLabel("K1 (Angle):"), 0, 0)
        layout_lqr.addWidget(self.spin_k1, 0, 1)
        layout_lqr.addWidget(QLabel("K2 (Vel angle):"), 1, 0)
        layout_lqr.addWidget(self.spin_k2, 1, 1)
        layout_lqr.addWidget(QLabel("K3 (Position):"), 0, 2)
        layout_lqr.addWidget(self.spin_k3, 0, 3)
        layout_lqr.addWidget(QLabel("K4 (Vel Pos):"), 1, 2)
        layout_lqr.addWidget(self.spin_k4, 1, 3)

        self.btn_apply_k = QPushButton("✓ Apply K")
        self.btn_apply_k.setStyleSheet("background-color: #FF9800; color: white;")
        self.btn_apply_k.clicked.connect(self.apply_k)
        layout_lqr.addWidget(self.btn_apply_k, 2, 0, 1, 4)
        
        self.group_lqr.setLayout(layout_lqr)
        left_layout.addWidget(self.group_lqr)

        # 3. RL Training
        self.group_train = QGroupBox("🧠 RL Training")
        layout_train = QFormLayout()
        layout_train.setSpacing(12)

        self.combo_train_agent = QComboBox()
        self.combo_train_agent.addItems(["SAC", "DDPG"])
        
        self.spin_episodes = QSpinBox()
        self.spin_episodes.setRange(1, 10000)
        self.spin_episodes.setValue(1200)

        self.line_path = QLineEdit('Models/Folder')

        self.btn_train = QPushButton("⚡ Start Training")
        self.btn_train.setStyleSheet("background-color: #2196F3; color: white;")
        self.btn_train.clicked.connect(self.start_training)

        layout_train.addRow("Agent:", self.combo_train_agent)
        layout_train.addRow("Episodes:", self.spin_episodes)
        layout_train.addRow("Save Path:", self.line_path)
        layout_train.addRow(self.btn_train)
        self.group_train.setLayout(layout_train)
        left_layout.addWidget(self.group_train)

        # 4. Simulation Viewer (Embedded Pygame)
        self.group_sim = QGroupBox("🎮 Live Simulation")
        layout_sim = QVBoxLayout()
        layout_sim.setContentsMargins(5, 15, 5, 5)
        
        self.sim_widget = QWidget()
        self.sim_widget.setMinimumHeight(350)
        self.sim_widget.setMinimumWidth(400)
        self.sim_widget.setStyleSheet("background-color: #000000; border-radius: 4px; border: 1px solid #444455;")
        
        layout_sim.addWidget(self.sim_widget)
        self.group_sim.setLayout(layout_sim)
        left_layout.addWidget(self.group_sim)
        
        left_layout.addStretch()

        # Connect signals for dynamic UI visibility
        self.combo_controllers.currentTextChanged.connect(self.update_ui_visibility)
        self.update_ui_visibility(self.combo_controllers.currentText())

        # Gather controls to block them during execution
        self.ui_controls = [
            self.btn_start, self.btn_train, self.combo_controllers, 
            self.combo_train_agent, self.spin_episodes, self.line_path,
            self.line_load_path, self.spin_k1, self.spin_k2, self.spin_k3,
            self.spin_k4, self.btn_apply_k, self.combo_port
        ]

        # ==========================================
        # RIGHT PANEL: TELEMETRY & PLOTS
        # ==========================================
        right_panel = QFrame()
        right_panel.setObjectName("TelemetryPanel")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(15)
        
        # --- Telemetry Header ---
        lbl_title_telemetry = QLabel("REAL-TIME TELEMETRY")
        lbl_title_telemetry.setAlignment(Qt.AlignCenter)
        lbl_title_telemetry.setStyleSheet("font-size: 20px; font-weight: bold; color: #4CAF50; letter-spacing: 1px;")
        right_layout.addWidget(lbl_title_telemetry)

        self.lbl_status = QLabel("INACTIVE")
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setStyleSheet("""
            background-color: #2a2a35; 
            color: #9e9e9e; 
            padding: 10px; 
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            border: 2px solid #444455;
        """)
        right_layout.addWidget(self.lbl_status)

        self.lbl_train_info = QLabel("Episode: -- | Reward: --")
        self.lbl_train_info.setAlignment(Qt.AlignCenter)
        self.lbl_train_info.setStyleSheet("color: #64b5f6; font-size: 14px; font-weight: bold;")
        right_layout.addWidget(self.lbl_train_info)
        
        # --- Telemetry Data Grid ---
        grid_telemetry = QGridLayout()
        grid_telemetry.setSpacing(10)
        
        def create_data_card(title, value):
            frame = QFrame()
            frame.setObjectName("DataCard")
            l = QVBoxLayout(frame)
            l.setContentsMargins(10, 10, 10, 10)
            lbl_title = QLabel(title)
            lbl_title.setStyleSheet("color: #a0a0d0; font-size: 12px; font-weight: bold; text-transform: uppercase;")
            lbl_title.setAlignment(Qt.AlignCenter)
            lbl_val = QLabel(value)
            lbl_val.setStyleSheet("color: #ffffff; font-size: 22px; font-weight: bold;")
            lbl_val.setAlignment(Qt.AlignCenter)
            l.addWidget(lbl_title)
            l.addWidget(lbl_val)
            return frame, lbl_val

        card_pos, self.lbl_pos = create_data_card("Cart Position", "0.00 cm")
        card_angle, self.lbl_angle = create_data_card("Pendulum Angle", "0.00°")
        card_vpos, self.lbl_vel_pos = create_data_card("Cart Velocity", "0.00 cm/s")
        card_vangle, self.lbl_vel_angle = create_data_card("Angular Velocity", "0.00 °/s")
        card_action, self.lbl_action = create_data_card("Action (Voltage)", "0.00 V")

        self.lbl_angle.setStyleSheet("color: #ff5252; font-size: 26px; font-weight: bold;")
        self.lbl_action.setStyleSheet("color: #ffb300; font-size: 24px; font-weight: bold;")
        self.lbl_pos.setStyleSheet("color: #4dd0e1; font-size: 24px; font-weight: bold;")

        grid_telemetry.addWidget(card_pos, 0, 0)
        grid_telemetry.addWidget(card_angle, 0, 1)
        grid_telemetry.addWidget(card_vpos, 1, 0)
        grid_telemetry.addWidget(card_vangle, 1, 1)
        grid_telemetry.addWidget(card_action, 2, 0, 1, 2)
        
        right_layout.addLayout(grid_telemetry)

        # --- Real-Time Plots Section Header ---
        header_layout = QHBoxLayout()
        lbl_plots = QLabel("REAL-TIME PLOTS")
        lbl_plots.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        lbl_plots.setStyleSheet("font-size: 16px; font-weight: bold; color: #64b5f6;")
        
        self.btn_export = QPushButton("💾 Export CSV")
        self.btn_export.setStyleSheet("background-color: #3f51b5; color: white; padding: 5px 15px; font-size: 12px;")
        self.btn_export.clicked.connect(self.export_csv)
        
        header_layout.addWidget(lbl_plots)
        header_layout.addStretch()
        header_layout.addWidget(self.btn_export)
        right_layout.addLayout(header_layout)

        # Global PyQtGraph Settings
        pg.setConfigOptions(antialias=True)
        
        self.graph_widget = pg.GraphicsLayoutWidget()
        self.graph_widget.setBackground('#1e1e24')
        right_layout.addWidget(self.graph_widget, stretch=1)

        # Plot 1: Position & Velocity
        self.plot_pos = self.graph_widget.addPlot(title="Car state")
        self.plot_pos.addLegend(offset=(10, 10))
        self.plot_pos.showGrid(x=True, y=True, alpha=0.3)
        self.plot_pos.setLabel('bottom', "Time", units='s')
        self.curve_pos = self.plot_pos.plot(pen=pg.mkPen('#4dd0e1', width=2), name="Pos (cm)")
        self.curve_vel_pos = self.plot_pos.plot(pen=pg.mkPen('#81c784', width=2), name="Vel (cm/s)")

        self.graph_widget.nextRow()

        # Plot 2: Angle & Angular Velocity
        self.plot_angle = self.graph_widget.addPlot(title="Pendulum state")
        self.plot_angle.addLegend(offset=(10, 10))
        self.plot_angle.showGrid(x=True, y=True, alpha=0.3)
        self.plot_angle.setLabel('bottom', "Time", units='s')
        # Link X axis to first plot so they zoom/pan together
        self.plot_angle.setXLink(self.plot_pos)
        self.curve_angle = self.plot_angle.plot(pen=pg.mkPen('#ff5252', width=2), name="Angle (°)")
        self.curve_vel_angle = self.plot_angle.plot(pen=pg.mkPen('#ba68c8', width=2), name="Ang Vel (°/s)")

        self.graph_widget.nextRow()

        # Plot 3: Control Action
        self.plot_action = self.graph_widget.addPlot(title="Control Action")
        self.plot_action.addLegend(offset=(10, 10))
        self.plot_action.showGrid(x=True, y=True, alpha=0.3)
        self.plot_action.setLabel('bottom', "Time", units='s')
        self.plot_action.setLabel('left', "Voltage", units='V')
        self.plot_action.setXLink(self.plot_pos)
        self.curve_action = self.plot_action.plot(pen=pg.mkPen('#ffb300', width=2), name="Action (V)")

        # Add panels to main layout
        main_layout.addWidget(left_panel, 3) 
        main_layout.addWidget(right_panel, 6) 

    def update_ui_visibility(self, controller_text):
        if controller_text == "Classic LQR":
            self.group_lqr.setVisible(True)
            self.widget_model_path.setVisible(False)
        else:
            self.group_lqr.setVisible(False)
            self.widget_model_path.setVisible(True)

    def apply_k(self):
        k1 = self.spin_k1.value()
        k2 = self.spin_k2.value()
        k3 = self.spin_k3.value()
        k4 = self.spin_k4.value()
        new_k = [k1, k2, k3, k4]
        
        # Send to controller if running
        self.command_queue.put({'type': 'update_k', 'K': new_k})
        print(f"Applying new K parameters: {new_k}")

    def reset_plots(self):
        # Clear data deques for plots
        self.time_data.clear()
        self.pos_data.clear()
        self.angle_data.clear()
        self.vel_pos_data.clear()
        self.vel_angle_data.clear()
        self.action_data.clear()

        # Clear complete data log
        self.data_log = {
            "time": [], "pos": [], "angle": [], "vel_pos": [], "vel_angle": [], "action": []
        }

        # Reset time
        self.run_start_time = time.time()

        # Clear curves
        self.curve_pos.setData([], [])
        self.curve_vel_pos.setData([], [])
        self.curve_angle.setData([], [])
        self.curve_vel_angle.setData([], [])
        self.curve_action.setData([], [])

    def start_training(self):
        if self.current_process is not None and self.current_process.is_alive():
            return

        agent_type = self.combo_train_agent.currentText()
        n_episodes = self.spin_episodes.value()
        save_dir = self.line_path.text()

        self.stop_event.clear()
        while not self.data_queue.empty():
            self.data_queue.get()

        self.reset_plots()
        
        # Ensure widget is fully rendered before grabbing winId
        QApplication.processEvents()
        win_id = int(self.sim_widget.winId())

        self.current_process = multiprocessing.Process(
            target=run_training_loop,
            args=(agent_type, save_dir, n_episodes, self.stop_event, self.data_queue, self.combo_port.currentText(), win_id)
        )
        self.current_process.start()

        self.set_status(f"TRAINING {agent_type}", "#2196F3", "#0d47a1") 
        
        for control in self.ui_controls:
            control.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def start_controller(self):
        if self.current_process is not None and self.current_process.is_alive():
            return

        selected_ctrl = self.combo_controllers.currentText()
        
        mapped_ctrl_name = selected_ctrl
        if selected_ctrl == "Classic LQR":
            mapped_ctrl_name = "LQR Clásico"
            
        model_to_load = self.line_load_path.text()
        self.stop_event.clear()

        if mapped_ctrl_name == "LQR Clásico":
            self.apply_k()

        while not self.data_queue.empty():
            self.data_queue.get()

        self.reset_plots()

        # Ensure widget is fully rendered before grabbing winId
        QApplication.processEvents()
        win_id = int(self.sim_widget.winId())

        self.current_process = multiprocessing.Process(
            target=run_controller,
            args=(mapped_ctrl_name, self.stop_event, self.data_queue, model_to_load, self.command_queue, self.combo_port.currentText(), win_id)
        )
        self.current_process.start()

        self.set_status(f"RUNNING {selected_ctrl}", "#4CAF50", "#1b5e20") 
        
        for control in self.ui_controls:
            control.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stop_controller(self):
        if self.current_process is not None:
            self.stop_event.set()
            self.current_process.join(timeout=2.0)
            if self.current_process.is_alive():
                self.current_process.terminate()
            self.current_process = None

        self.set_status("INACTIVE", "#9e9e9e", "#444455")
        
        for control in self.ui_controls:
            control.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.lbl_train_info.setText("Episode: -- | Reward: --")

    def set_status(self, text, color, border_color):
        self.lbl_status.setText(f"{text}")
        self.lbl_status.setStyleSheet(f"""
            background-color: #252530; 
            color: {color}; 
            padding: 10px; 
            border-radius: 8px;
            font-size: 16px;
            font-weight: bold;
            border: 2px solid {border_color};
            text-transform: uppercase;
            letter-spacing: 1px;
        """)

    def update_telemetry(self):
        latest_data = None
        while not self.data_queue.empty():
            try:
                latest_data = self.data_queue.get_nowait()
            except Exception:
                break

        if latest_data:
            if 'status_msg' in latest_data:
                self.lbl_train_info.setText(f"{latest_data['status_msg']}")
                return

            if 'episode' in latest_data:
                self.lbl_train_info.setText(f"Episode: {latest_data['episode']} | Accumulated Reward: {latest_data['score']:.1f}")

            if 'pos' in latest_data:
                pos = latest_data.get('pos', 0)
                angle = latest_data.get('angle', 0)
                vel_pos = latest_data.get('vel_pos', 0)
                vel_angle = latest_data.get('vel_angle', 0)
                action = latest_data.get('action', 0)

                # Update Text UI
                self.lbl_pos.setText(f"{pos:.3f} cm")
                self.lbl_angle.setText(f"{angle:.2f}°")
                self.lbl_vel_pos.setText(f"{vel_pos:.2f} cm/s")
                self.lbl_vel_angle.setText(f"{vel_angle:.2f} °/s")
                self.lbl_action.setText(f"{action:.2f} V")

                current_time = time.time() - self.run_start_time

                # Update Plot Data (Maxlen limited deque)
                self.time_data.append(current_time)
                self.pos_data.append(pos)
                self.angle_data.append(angle)
                self.vel_pos_data.append(vel_pos)
                self.vel_angle_data.append(vel_angle)
                self.action_data.append(action)

                # Update Continuous Log Data (Unlimited)
                self.data_log["time"].append(current_time)
                self.data_log["pos"].append(pos)
                self.data_log["angle"].append(angle)
                self.data_log["vel_pos"].append(vel_pos)
                self.data_log["vel_angle"].append(vel_angle)
                self.data_log["action"].append(action)

                # Refresh Plot Curves
                t_list = list(self.time_data)
                self.curve_pos.setData(t_list, list(self.pos_data))
                self.curve_vel_pos.setData(t_list, list(self.vel_pos_data))
                self.curve_angle.setData(t_list, list(self.angle_data))
                self.curve_vel_angle.setData(t_list, list(self.vel_angle_data))
                self.curve_action.setData(t_list, list(self.action_data))

    def export_csv(self):
        if not hasattr(self, 'data_log') or len(self.data_log["time"]) == 0:
            QMessageBox.warning(self, "No Data", "There is no telemetry data to export.\nPlease run a controller first.")
            return
            
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Telemetry Data", "telemetry_data.csv", "CSV Files (*.csv)", options=options)
        
        if file_path:
            try:
                with open(file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["time", "pos", "angle", "vel_pos", "vel_angle", "action"])
                    for i in range(len(self.data_log["time"])):
                        writer.writerow([
                            self.data_log["time"][i],
                            self.data_log["pos"][i],
                            self.data_log["angle"][i],
                            self.data_log["vel_pos"][i],
                            self.data_log["vel_angle"][i],
                            self.data_log["action"][i]
                        ])
                QMessageBox.information(self, "Export Successful", f"Data exported successfully to:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export data:\n{e}")

    def closeEvent(self, event):
        self.stop_controller()
        event.accept()

if __name__ == '__main__':
    multiprocessing.freeze_support() 
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    ex = PendulumGUI()
    ex.show()
    sys.exit(app.exec_())