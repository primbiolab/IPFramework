import numpy as np
import matplotlib.pyplot as plt
import csv

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

def plot_scores_scatter(x, scores, figure_file="scores_scatter.png"):
    plt.figure()
    plt.scatter(x, scores, color='blue', s=10)
    plt.title('Scores por episodio')
    plt.xlabel('Episodio')
    plt.ylabel('Score')
    plt.tight_layout()
    plt.savefig(figure_file)
    plt.show()

def save_episode_variables_csv(filename, t, pos_list, vel_list, ang_list, ang_vel_list, accion):
    # Guarda los datos en el mismo orden en columnas, sin corchetes en 'Acción'
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Tiempo', 'Posición', 'Velocidad', 'Ángulo', 'Velocidad angular', 'Acción'])
        for i in range(len(t)):
            # Si accion[i] es una lista, toma el primer elemento
            accion_val = accion[i][0] if isinstance(accion[i], (list, np.ndarray)) else accion[i]
            writer.writerow([
                t[i],
                pos_list[i],
                vel_list[i],
                ang_list[i],
                ang_vel_list[i],
                accion_val
            ])

def plot_episode_variables(t, pos_list, vel_list, ang_list, ang_vel_list, accion, figure_file="test", csv_file="None"):
    plt.figure(figsize=(12,8))
    plt.subplot(3,2,1)
    plt.plot(t, pos_list)
    plt.title("Posición del carrito")
    plt.xlabel("Tiempo (steps)")
    plt.ylabel("Posición")
    plt.subplot(3,2,2)
    plt.plot(t, vel_list)
    plt.title("Velocidad del carrito")
    plt.xlabel("Tiempo (steps)")
    plt.ylabel("Velocidad")
    plt.subplot(3,2,3)
    plt.plot(t, ang_list)
    plt.title("Ángulo del péndulo")
    plt.xlabel("Tiempo (steps)")
    plt.ylabel("Ángulo (grados)")
    plt.subplot(3,2,4)
    plt.plot(t, ang_vel_list)
    plt.title("Velocidad angular del péndulo")
    plt.xlabel("Tiempo (steps)")
    plt.ylabel("Velocidad angular")
    plt.subplot(3,2,5)
    plt.plot(t, accion)
    plt.title("Señal de control")
    plt.xlabel("Tiempo (steps)")
    plt.ylabel("PWM")
    plt.tight_layout()
    # if figure_file:
    #     plt.savefig("Modelos_Finales_Figuras/Pendulo_2/test_de_entrenamiento_csv/prueba.png")
    plt.show()
    if csv_file:
        ruta_csv = f"{csv_file}"
        save_episode_variables_csv(ruta_csv, t, pos_list, vel_list, ang_list, ang_vel_list, accion)