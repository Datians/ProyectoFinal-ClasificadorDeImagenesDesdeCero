import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def analizar_solo_csv(csv_file='predicciones_test.csv'):
    try:
        # Leer el archivo CSV
        df = pd.read_csv(csv_file)
        
        # Calcular métricas
        reporte = classification_report(df['TrueLabel'], df['PredictedLabel'])
        print("\n--- INFORME DE CLASIFICACIÓN (Métricas Finales) ---")
        print(reporte)
        
        # Generar Matriz de Confusión
        cm = confusion_matrix(df['TrueLabel'], df['PredictedLabel'])
        
        # Visualización
        fig, ax = plt.subplots(figsize=(8, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title('Matriz de Confusión')
        plt.show()
        
        print("[ÉXITO] Análisis completado.")
        
    except FileNotFoundError:
        print(f"[ERROR] No se encuentra el archivo '{csv_file}'. Asegúrate de que esté en la misma carpeta.")
    except Exception as e:
        print(f"[ERROR] Ocurrió un problema: {e}")

if __name__ == "__main__":
    analizar_solo_csv()