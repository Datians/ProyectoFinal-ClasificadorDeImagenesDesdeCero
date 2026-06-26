import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, classification_report

# ==========================================
# 1. GRAFICAR EVOLUCIÓN DEL LOSS Y ACCURACY
# ==========================================
try:
    df_historial = pd.read_csv("historial_entrenamiento.txt")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Gráfica de Pérdida (Loss)
    ax1.plot(df_historial['epoch'], df_historial['loss'], color='darkred', lw=2)
    ax1.set_title('Evolución del Loss por Época', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Épocas')
    ax1.set_ylabel('Loss Promedio')
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Gráfica de Precisión (Accuracy)
    ax2.plot(df_historial['epoch'], df_historial['accuracy'], color='darkgreen', lw=2)
    ax2.set_title('Evolución del Accuracy por Época', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Épocas')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig('grafica_rendimiento.png', dpi=300)
    print("Gráfica de rendimiento guardada como 'grafica_rendimiento.png'")
    plt.close()
except Exception as e:
    print("No se pudo generar la gráfica de historial:", e)


# ==========================================
# 2. GENERAR MATRIZ DE CONFUSIÓN Y RECALL
# ==========================================
try:
    df_metrica = pd.read_csv("metricas_finales.txt")
    y_true = df_metrica['real']
    y_pred = df_metrica['predicho']
    
    # Obtener dinámicamente las clases únicas
    clases_unicas = np.unique(np.concatenate([y_true, y_pred]))
    num_clases = len(clases_unicas)
    
    # Calcular Matriz de Confusión
    cm = confusion_matrix(y_true, y_pred, labels=clases_unicas)
    
    # --- NUEVO: CALCULAR RECALL Y OTRAS MÉTRICAS ---
    # Calculamos el recall por clase (devolverá un array, ej: [recall_clase0, recall_clase1])
    recalls = recall_score(y_true, y_pred, average=None, labels=clases_unicas)
    recall_global = recall_score(y_true, y_pred, average='macro')
    precision_global = precision_score(y_true, y_pred, average='macro')
    f1_global = f1_score(y_true, y_pred, average='macro')
    acc_global = (y_true == y_pred).mean()

    print("\n==================================================")
    print("      REPORTE DE RENDIMIENTO FINAL (PYTHON)       ")
    print("==================================================")
    for idx, clase in enumerate(clases_unicas):
        print(f"Clase {clase} -> Recall (Sensibilidad): {recalls[idx]*100:.2f}%")
    print(f"Recall Promedio General: {recall_global*100:.2f}%")
    print("==================================================\n")
    
    # Imprimir el reporte detallado clásico de scikit-learn en consola
    print(classification_report(y_true, y_pred, labels=clases_unicas))

    # Graficar la Matriz de Confusión corregida
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap='Blues')
    fig.colorbar(cax)

    ax.set_xticks(np.arange(num_clases))
    ax.set_yticks(np.arange(num_clases))
    ax.set_xticklabels([str(c) for c in clases_unicas])
    ax.set_yticklabels([str(c) for c in clases_unicas])
    ax.xaxis.set_ticks_position('bottom')

    for i in range(num_clases):
        for j in range(num_clases):
            color = "white" if cm[i, j] > (cm.max() / 2) else "black"
            ax.text(j, i, str(cm[i, j]), va='center', ha='center', color=color, fontsize=12, fontweight='bold')

    plt.title('Matriz de Confusión - Red Neuronal CUDA', fontsize=12, fontweight='bold', pad=15)
    plt.ylabel('Clase Real (Ground Truth)', fontsize=11)
    plt.xlabel('Clase Predicha', fontsize=11)
    
    plt.tight_layout()
    plt.savefig('matriz_confusion_final.png', dpi=300)
    print("Matriz de confusión guardada correctamente como 'matriz_confusion_final.png'")
    plt.close()

    # --- NUEVA GRÁFICA: COMPARATIVA DE MÉTRICAS GLOBALES ---
    metricas_nombres = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    metricas_valores = [acc_global * 100, precision_global * 100, recall_global * 100, f1_global * 100]
    
    plt.figure(figsize=(7, 4))
    barras = plt.bar(metricas_nombres, metricas_valores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'], width=0.5)
    plt.title('Métricas Globales del Modelo', fontsize=12, fontweight='bold', pad=15)
    plt.ylabel('Porcentaje (%)')
    plt.ylim(0, 110)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Añadir el valor numérico arriba de cada barra
    for barra in barras:
        height = barra.get_height()
        plt.text(barra.get_x() + barra.get_width()/2.0, height + 2, f'{height:.2f}%', ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('metricas_globales.png', dpi=300)
    print("Gráfica de métricas globales guardada como 'metricas_globales.png'")
    plt.show()

except Exception as e:
    print("Error al procesar las métricas o la matriz de confusión:", e)