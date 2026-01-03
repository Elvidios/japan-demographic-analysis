import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

# --- 1. CARGA DE DATOS ---
print("Cargando datos...")
df = pd.read_csv(
    'population_per_year.csv', 
    encoding='shift_jis', 
    sep=';', 
    header=13,
    index_col=0,
    na_values=['-', ' ', ''] # Ayuda extra para valores vacíos simples
)

# --- 2. LIMPIEZA INICIAL ---
df.index.name = 'Prefectura'

# Eliminar fila Total
if 'Total' in df.index:
    df_total = df.loc['Total']
    df = df.drop('Total')

# Limpiar nombres de las prefecturas (Quitar '01 ', '02 ', etc.)
# Esto es crucial: Dejamos los nombres limpios para todo el proceso
df.index = df.index.str.replace(r'^\d+\s+', '', regex=True)

# --- 3. ANÁLISIS ESTADÍSTICO (Variación 2000-2024) ---
# Verificamos que las columnas existan antes de calcular
if '2000' in df.columns and '2024' in df.columns:
    # Cálculo de variación porcentual
    df['Variacion_2000_2024'] = ((df['2024'] - df['2000']) / df['2000']) * 100

    print("\n--- Top 10 Prefecturas con mayor despoblación (2000-2024) ---")
    print(df['Variacion_2000_2024'].sort_values().head(10))

    # ¡IMPORTANTE! Eliminamos la columna calculada
    # Si no la borramos, el paso siguiente (melt) fallará mezclando números con texto
    df = df.drop(columns=['Variacion_2000_2024'])
else:
    print("No se encontraron las columnas 2000 y 2024 para el cálculo de variación.")

# --- 4. TRANSFORMACIÓN DE DATOS (MELT) ---
df_reset = df.reset_index()

df_largo = df_reset.melt(
    id_vars=['Prefectura'], 
    var_name='Año', 
    value_name='Poblacion'
)

# --- 5. LIMPIEZA PROFUNDA (Arreglo de errores "..." y tipos) ---

# Convertir Año a numérico
df_largo['Año'] = pd.to_numeric(df_largo['Año'])

# Convertir Población a numérico (CORRECCIÓN CLAVE)
# 'errors=coerce' transforma los "…" y textos raros en NaN (vacío) automáticamente
df_largo['Poblacion'] = pd.to_numeric(df_largo['Poblacion'], errors='coerce')

# Eliminar filas donde la población sea NaN (los años con "…")
df_largo = df_largo.dropna(subset=['Poblacion'])

print(f"\nDatos listos para graficar: {len(df_largo)} registros procesados.")

# --- 6. GRAFICAR ---

# Lista de prefecturas a comparar
# IMPORTANTE: Como arriba limpiamos los números, aquí usa solo los nombres exactos
# Si en tu CSV dice "Tokyo-to", pon "Tokyo-to". Si dice "Tokyo", pon "Tokyo".
objetivos = ['Tokyo', 'Akita', 'Okinawa', 'Aomori'] 

# Filtrar
df_grafico = df_largo[df_largo['Prefectura'].isin(objetivos)]

if not df_grafico.empty:
    plt.figure(figsize=(12, 6))

    sns.lineplot(
        data=df_grafico, 
        x='Año', 
        y='Poblacion', 
        hue='Prefectura', 
        marker='o', 
        linewidth=2.5
    )

    # Estética
    plt.title('Evolución Demográfica: Centro vs Periferia en Japón', fontsize=16)
    plt.ylabel('Población (Millones)', fontsize=12)
    plt.xlabel('Año', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)

    # Formato de millones en eje Y (ej: 1.5 M)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.1f} M'.format(x/1000000)))

    plt.tight_layout()
    plt.show()
else:
    print("\nERROR: El gráfico está vacío.")
    print(f"Estás buscando: {objetivos}")
    print("Pero en tus datos tienes nombres como:", df_largo['Prefectura'].unique()[:5])
    print("Revisa mayúsculas/minúsculas o si faltan sufijos como '-ken' o '-to'.")