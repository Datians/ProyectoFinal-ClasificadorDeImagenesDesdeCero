import matplotlib.pyplot as plt
import re

def generar_grafica(archivo='log_entrenamiento.txt'):
    with open(archivo, 'r') as f:
        contenido = f.read()
    
    # Extraer los datos usando expresiones regulares
    losses = [float(x) for x in re.findall(r"Loss:\s*([\d\.]+)", contenido)]
    accs = [float(x) for x in re.findall(r"Accuracy:\s*([\d\.]+)%", contenido)]
    epocas = range(1, len(losses) + 1)
    
    # Crear gráfica de dos paneles
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(epocas, losses, 'r-o')
    ax1.set_title('Evolución de la Pérdida (Loss)')
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Loss')
    
    ax2.plot(epocas, accs, 'g-o')
    ax2.set_title('Evolución de la Exactitud (Accuracy)')
    ax2.set_xlabel('Época')
    ax2.set_ylabel('Accuracy (%)')
    
    plt.tight_layout()
    plt.savefig('curvas_entrenamiento.png') # Guarda la imagen
    plt.show()

generar_grafica()