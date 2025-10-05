import pandas as pd
import os

# --- Configuración ---
input_csv = "kepler.csv"       # Tu archivo original
output_folder = "excel_splits" # Carpeta donde se guardarán los Excel
rows_per_file = 100            # Filas por archivo Excel

# --- Crear carpeta de salida si no existe ---
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# --- Leer CSV y limpiar columnas ---
df = pd.read_csv(input_csv, encoding='utf-8-sig', comment='#')
df.columns = df.columns.str.strip().str.replace('"','')  # Quitar espacios y comillas

# --- Subdividir y guardar ---
total_rows = len(df)
num_files = (total_rows // rows_per_file) + 1

for i in range(num_files):
    start = i * rows_per_file
    end = start + rows_per_file
    df_chunk = df.iloc[start:end]
    output_file = os.path.join(output_folder, f"kepler_part_{i+1}.xlsx")
    df_chunk.to_excel(output_file, index=False)
    print(f"✅ Guardado: {output_file} ({len(df_chunk)} filas)")

print(f"🎉 Completado. {num_files} archivos Excel generados en '{output_folder}'")
