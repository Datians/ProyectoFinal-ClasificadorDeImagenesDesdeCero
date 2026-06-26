import streamlit as st
import subprocess
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title="CUDA Mask Detector", layout="centered")

st.title("🧠 Detector de Mascarillas - Acelerado por GPU")
st.write("Sube una imagen para clasificarla en tiempo real usando tu red neuronal en CUDA.")

# Diccionario de clases (Ajusta el orden según tu dataset)
CLASES = {
    0: "🔓 SIN MASCARILLA",
    1: "😷 CON MASCARILLA"
}

uploaded_file = st.file_uploader("Elige una imagen...", type=["bin", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    if uploaded_file.name.endswith(('png', 'jpg', 'jpeg')):
        image = Image.open(uploaded_file).convert('L') # Escala de grises
        image_resized = image.resize((64, 64)) # 64x64 = 4096
        st.image(image_resized, caption="Imagen procesada para la Red (64x64)", width=200)
        
        pixel_data = np.array(image_resized, dtype=np.float32) / 255.0
        flat_data = pixel_data.flatten()
    else:
        bytes_data = uploaded_file.read()
        flat_data = np.frombuffer(bytes_data, dtype=np.float32)
        st.success(f"Archivo binario cargado. Tamaño: {len(flat_data)} características.")

    if len(flat_data) == 4096:
        if st.button("🚀 Clasificar con la GPU (CUDA)"):
            with st.spinner("Procesando en los Tensor Cores / CUDA Cores..."):
                temp_filename = "temp_streamlit_sample.bin"
                flat_data.tofile(temp_filename)
                
                try:
                    result = subprocess.run(
                        ["./neural_net", temp_filename], 
                        capture_output=True, 
                        text=True, 
                        check=True
                    )
                    
                    output_console = result.stdout
                    
                    # Buscamos la línea mágica que imprimió CUDA
                    prediccion_detectada = None
                    for line in output_console.split("\n"):
                        if "PRED_CLASS:" in line:
                            clase_num = int(line.split(":")[1].strip())
                            prediccion_detectada = CLASES.get(clase_num, f"Clase desconocida ({clase_num})")
                    
                    # --- DISEÑO DEL RESULTADO EN LA INTERFAZ ---
                    st.markdown("---")
                    st.subheader("🎯 Resultado de la Clasificación:")
                    
                    if prediccion_detectada:
                        if "CON MASCARILLA" in prediccion_detectada:
                            st.success(f"### **Predicción:** {prediccion_detectada}")
                        else:
                            st.error(f"### **Predicción:** {prediccion_detectada}")
                    else:
                        st.warning("Se ejecutó bien, pero no se detectó la etiqueta PRED_CLASS en el código C++.")
                    
                    # Colapsable para ver la consola de CUDA (Modo desarrollador)
                    with st.expander("🛠️ Ver logs técnicos de la GPU (Consola CUDA)"):
                        st.code(output_console)
                        
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"Hubo un error al ejecutar el código CUDA: {e}")
                finally:
                    if os.path.exists(temp_filename):
                        os.remove(temp_filename)
    else:
        st.error(f"Error: La red espera 4096 características, recibidas {len(flat_data)}.")