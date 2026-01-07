import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import scipy.stats as stats

#----------------------------------------------------

#FUNCIONES DE PROCESAMIENTO DE DATOS
def demographic_dev(): 

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



    return df_largo  # Retornar el DataFrame para inspección externa
    # --- FIN DEL SCRIPT ---

def tasa_de_extranjeros_por_prefectura(archivo_csv):
    print(f"--- PROCESANDO DATOS: {archivo_csv} ---")
    
    datos_limpios = []
    
    # 1. EXTRACCIÓN MANUAL (MODO CIRUGÍA)
    try:
        with open(archivo_csv, 'r', encoding='shift_jis') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            # Filtro para detectar líneas con datos de prefecturas
            if len(line) > 20 and ("-ken" in line.lower() or "-to" in line.lower() or "-fu" in line.lower() or "hokkaido" in line.lower() or " ken" in line.lower()):
                # Normalizar separadores a pipe '|'
                temp_line = line.replace('","', '|').replace(',"', '|').replace('",', '|')
                columnas = temp_line.split('|')
                
                if len(columnas) > 5:
                    # Limpiar comillas
                    columnas = [c.replace('"', '').strip() for c in columnas]
                    datos_limpios.append(columnas)
    except Exception as e:
        print(f"❌ Error leyendo archivo: {e}")
        return None, None

    if not datos_limpios:
        print("Error: No se encontraron datos válidos.")
        return None, None

    df = pd.DataFrame(datos_limpios)

    # 2. DETECCIÓN DE COLUMNAS
    idx_anio, idx_pref = -1, -1
    # Analizamos la primera fila
    for idx, val in enumerate(df.iloc[0]):
        val_str = str(val)
        if val_str.isdigit() and len(val_str) == 4 and val_str.startswith(('19', '20')):
            if idx_anio == -1: idx_anio = idx
        if "ken" in val_str.lower() or "to" in val_str.lower() or "hokkaido" in val_str.lower():
            idx_pref = idx
            
    # Fallback si falla la detección automática
    if idx_pref == -1: 
        idx_anio, idx_pref = 2, 5
    
    # Índices relativos (Total suele estar +2 y Japoneses +5 desde la Prefectura)
    idx_total = idx_pref + 2
    idx_jap = idx_pref + 5

    try:
        df_final = df.iloc[:, [idx_anio, idx_pref, idx_total, idx_jap]].copy()
        df_final.columns = ['Año', 'Prefectura', 'Poblacion_Total', 'Poblacion_Japonesa']
    except:
        print("⚠️ Usando índices fijos de emergencia (2,5,7,10)")
        df_final = df.iloc[:, [2, 5, 7, 10]].copy()
        df_final.columns = ['Año', 'Prefectura', 'Poblacion_Total', 'Poblacion_Japonesa']

    # 3. LIMPIEZA Y CÁLCULOS
    df_final['Año'] = pd.to_numeric(df_final['Año'], errors='coerce')
    
    # Obtener el año más reciente disponible
    anio_obj = sorted(df_final['Año'].dropna().unique())[-1]
    df_final = df_final[df_final['Año'] == anio_obj].copy()

    # Limpiar números (quitar comas y guiones)
    for col in ['Poblacion_Total', 'Poblacion_Japonesa']:
        df_final[col] = df_final[col].astype(str).str.replace(',', '', regex=False).str.replace('-', '0', regex=False)
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce')

    # Calcular Tasa
    df_final['Extranjeros'] = df_final['Poblacion_Total'] - df_final['Poblacion_Japonesa']
    df_final = df_final[df_final['Poblacion_Total'] > 0] # Evitar división por cero
    df_final['Tasa'] = (df_final['Extranjeros'] / df_final['Poblacion_Total']) * 100

    print(f"✅ Datos procesados correctamente para el año: {anio_obj}")
    
    # RETORNAMOS EL DATAFRAME Y EL AÑO (Necesario para el título del gráfico)
    return df_final, anio_obj

#----------------------------------------------------

#VARIABLES GLOBALES
df_historico = demographic_dev()

df_final, anio_objetivo = tasa_de_extranjeros_por_prefectura('population_per_prefecture_2024.csv')

#----------------------------------------------------

#FUNCIONES DE GRÁFICOS
def graph_demografia_historica(df_largo):

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

def graph_demografia_historica_periferia(df_largo):

    # Lista de prefecturas a comparar
    # IMPORTANTE: Como arriba limpiamos los números, aquí usa solo los nombres exactos
    objetivos = ['Akita', 'Aomori', 'kochi', 'Iwate', 'Yamagata', 'Fukushima', 'Wakayama', 'Nagasaki'] 

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
        plt.title('Periferia en Japón: Mayores Desviaciones Demográficas', fontsize=16)
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

def graph_proporciones_demograficas(df_largo, anio_objetivo=2024):
    """
    Crea un gráfico circular de la distribución de población para un año específico.
    Agrupa las prefecturas menores para que el gráfico sea legible.
    """
    
    # 1. Filtrar solo los datos del año que nos interesa
    df_anio = df_largo[df_largo['Año'] == anio_objetivo].copy()
    
    # Verificar si hay datos para ese año
    if df_anio.empty:
        print(f"No hay datos para el año {anio_objetivo}")
        return

    # 2. Ordenar de mayor a menor población
    df_anio = df_anio.sort_values(by='Poblacion', ascending=False)

    # 3. Lógica "Top N + Otros" (Para evitar 47 rebanadas)
    top_n = 7  # Mostraremos las 7 más grandes individualmente
    
    # Sepamos las Top
    df_top = df_anio.head(top_n)
    
    # Sumamos todas las demás en una categoría "Otros"
    otros_poblacion = df_anio.iloc[top_n:]['Poblacion'].sum()
    df_otros = pd.DataFrame({'Prefectura': ['Otras (Prefecturas)'], 'Poblacion': [otros_poblacion]})
    
    # Unimos para el gráfico
    df_grafico = pd.concat([df_top[['Prefectura', 'Poblacion']], df_otros])

    # 4. Crear el Gráfico Circular
    plt.figure(figsize=(7, 4))
    
    # Colores: Usamos una paleta pastel para que sea agradable
    colores = sns.color_palette('pastel')[0:len(df_grafico)]

    plt.pie(
        df_grafico['Poblacion'], 
        labels=df_grafico['Prefectura'], 
        autopct='%1.1f%%',       # Muestra el porcentaje con 1 decimal
        startangle=140,          # Rota el gráfico para que el #1 quede arriba
        colors=colores,
        pctdistance=0.85,        # Distancia de los números del centro
        explode=[0.1] + [0]*(len(df_grafico)-1) # "Saca" la rebanada más grande (Tokyo) para destacarla
    )

    plt.title(f'Distribución de la Población en Japón ({anio_objetivo})', fontsize=16, y=0.95)
    plt.axis('equal')  # Asegura que sea un círculo perfecto y no un óvalo
    plt.tight_layout()
    plt.show()

def graph_mapas_calor_japon_final(df_largo):
    print("1. Descargando mapa...")
    url_mapa = "https://raw.githubusercontent.com/dataofjapan/land/master/japan.geojson"
    japan_map = gpd.read_file(url_mapa)
    
    # --- CORRECCIÓN DE NOMBRES EN EL MAPA (CRÍTICO) ---
    # El mapa trae 'Akita Ken', 'Hokkai Do'. Tus datos traen 'Akita', 'Hokkaido'.
    # Limpiamos la columna 'nam' del mapa para igualarla a tu CSV.
    
    # 1. Caso especial: Hokkaido (en el mapa suele venir como 'Hokkai Do')
    japan_map['nam'] = japan_map['nam'].str.replace('Hokkai Do', 'Hokkaido', regex=False)
    
    # 2. Quitar sufijos comunes con espacio (Ken, Fu, To)
    # Reemplazamos " Ken" por nada, " Fu" por nada, etc.
    eliminar = [' Ken', ' Fu', ' To']
    for sufijo in eliminar:
        japan_map['nam'] = japan_map['nam'].str.replace(sufijo, '', regex=False)
    
    # 3. Quitar espacios extra por seguridad
    japan_map['nam'] = japan_map['nam'].str.strip()
    
    # ----------------------------------------------------

    print("2. Cruzando datos y generando mapas...")
    anios_clave = [1995, 2005, 2015, 2024] 
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    # Base año 2000 para comparar (índice normalizado)
    df_base = df_largo[df_largo['Año'] == 2000].set_index('Prefectura')['Poblacion']

    for i, anio in enumerate(anios_clave):
        ax = axes[i]
        
        # Filtramos datos del año
        df_anio = df_largo[df_largo['Año'] == anio].copy()
        
        # Calculamos variación porcentual vs año 2000
        # Formula: ((Poblacion_Actual - Poblacion_2000) / Poblacion_2000) * 100
        # Para cada fila (apply axis=1), toma la población de este año (x['Poblacion']), réstale la población que tenía esa misma prefectura en el año 2000 (df_base.get), y divide el resultado por la población del 2000. Luego multiplica por 100 para hacerlo porcentaje.
        df_anio['Variacion'] = df_anio.apply(
            lambda x: ((x['Poblacion'] - df_base.get(x['Prefectura'], x['Poblacion'])) / df_base.get(x['Prefectura'], 1)) * 100, 
            axis=1
        )
        
        # UNIÓN (MERGE): Ahora sí coinciden 'nam' (mapa) y 'Prefectura' (datos)
        mapa_data = japan_map.merge(df_anio, left_on='nam', right_on='Prefectura', how='left')
        
        # Graficar
        mapa_data.plot(
            column='Variacion',
            cmap='RdBu_r',    # Rojo = Crece, Azul = Decrece
            linewidth=0.5,
            ax=ax,
            edgecolor='0.6',
            legend=True,
            legend_kwds={'label': "Variación vs Año 2000 (%)", 'shrink': 0.5},
            missing_kwds={'color': 'lightgrey'} # Gris si falta el dato
        )
        
# 1. Título más abajo (como pediste)
        ax.set_title(f'Año {anio}', fontsize=14, y=0.85)
        
        # 2. Quitar ejes
        ax.set_axis_off()

        # 3. ENFOQUE (ZOOM): Recortamos coordenadas para ver solo la isla principal
        # Longitud (Este-Oeste): 128° a 146°
        # Latitud (Norte-Sur): 30° a 46°
        ax.set_xlim(128, 146)
        ax.set_ylim(30, 46)

    plt.suptitle('Evolución Demográfica Japón: Variación respecto al año 2000', fontsize=20, y=1)
    plt.tight_layout()
    plt.show()

def graph_tasa_de_extranjeros_por_prefectura(df, anio):
    if df is None or df.empty:
        print("No hay datos para graficar.")
        return

    print("--- GENERANDO GRÁFICO ---")

    # Configuración del Lienzo
    plt.figure(figsize=(12, 8))
    
    # Dibujar Scatter Plot
    sns.scatterplot(
        data=df, 
        x='Poblacion_Total', 
        y='Tasa', 
        hue='Prefectura', 
        legend=False, 
        s=150, 
        alpha=0.8, 
        edgecolor='black'
    )
    
    # Configuración de Ejes
    plt.xscale('linear')
    plt.ticklabel_format(style='plain', axis='x') # Números completos (no científicos)
    plt.title(f'Tasa de población extranjera por prefectura ({anio})', fontsize=16)
    plt.xlabel('Población Total', fontsize=12)
    plt.ylabel('% de Extranjeros', fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)

    # --- ETIQUETADO INTELIGENTE (LIMPIEZA DE NOMBRES) ---
    a_eliminar = [
        '-to', '-To', '-TO',   # Caso Tokyo
        '-fu', '-Fu', '-FU',   # Caso Osaka/Kyoto
        '-ken', '-Ken', '-KEN', # Caso general
        ' To', ' Fu', ' Ken',   # Caso con espacio
        ' TO', ' FU', ' KEN'
    ]

    for i in range(df.shape[0]):
        row = df.iloc[i]
        
        # Solo etiquetamos si la tasa es alta (>2%) o la población es grande (>5M)
        if row['Tasa'] > 2.0 or row['Poblacion_Total'] > 5000000:
            nombre = str(row['Prefectura'])
            
            # Limpieza del nombre
            for basura in a_eliminar:
                if basura in nombre: # Si encuentra el sufijo, lo borra
                    nombre = nombre.replace(basura, "")
            
            nombre = nombre.strip()
            
            # Escribir texto en el gráfico
            plt.text(row['Poblacion_Total'], row['Tasa'] + 0.1, nombre, fontsize=9, weight='bold')

    plt.tight_layout()
    plt.show()
    
    # Imprimir tabla de control en consola
    print("\n--- TOP 5 PREFECTURAS (CONTROL) ---")
    df_temp = df.copy()
    df_temp['Nombre_Limpio'] = df_temp['Prefectura']
    for basura in a_eliminar:
        df_temp['Nombre_Limpio'] = df_temp['Nombre_Limpio'].str.replace(basura, '', regex=False)
        
    print(df_temp[['Nombre_Limpio', 'Tasa']].sort_values('Tasa', ascending=False).head())

#----------------------------------------------------

#Generar Gráficos Históricos (Descomenta el que quieras ver)

#graph_demografia_historica(df_historico)
#graph_demografia_historica_periferia(df_historico)
#graph_proporciones_demograficas(df_historico, anio_objetivo=2024)
#graph_mapas_calor_japon_final(df_historico)  # Requiere GeoPandas
#graph_tasa_de_extranjeros_por_prefectura(df_final, anio_objetivo)


#----------------------------------------------------
# 3. ANÁLISIS ESTADÍSTICO (LA SOLUCIÓN AL ERROR)
#----------------------------------------------------

def probar_hipotesis_correlacion(df_historico, df_extranjeros):
    print("\n--- PRUEBA DE HIPÓTESIS H1: CORRELACIÓN ---")

    # A) PREPARAR DATOS HISTÓRICOS (Variación 2000 - Actualidad)
    try:
        # Filtrar año 2000 y el último disponible
        anio_fin = df_historico['Año'].max()
        
        # Crear series indexadas por Prefectura
        s_2000 = df_historico[df_historico['Año'] == 2000].set_index('Prefectura')['Poblacion']
        s_fin = df_historico[df_historico['Año'] == anio_fin].set_index('Prefectura')['Poblacion']
        
        # Calcular Variación Porcentual
        df_variacion = pd.DataFrame(((s_fin - s_2000) / s_2000) * 100)
        df_variacion.columns = ['Variacion_Demografica_Pct']
        
        # Limpieza básica de espacios
        df_variacion.index = df_variacion.index.str.strip()
        
    except Exception as e:
        print(f"Error calculando variaciones históricas: {e}")
        return

    # B) PREPARAR DATOS DE EXTRANJEROS Y LIMPIAR NOMBRES
    # Aquí está la magia para que el cruce funcione
    df_tasa_ext = df_extranjeros.set_index('Prefectura')[['Tasa']]
    df_tasa_ext.columns = ['Tasa_Extranjeros_Pct']

    # --- CORRECCIÓN DE NOMBRES (Estandarización) ---
    print("Normalizando nombres para el cruce (quitando sufijos -to, -ken)...")
    
    basura_a_eliminar = [
        '-to', '-To', '-TO', 
        '-fu', '-Fu', '-FU', 
        '-ken', '-Ken', '-KEN', 
        ' To', ' Fu', ' Ken',
        ' TO', ' FU', ' KEN'
    ]
    
    # Creamos una copia del índice para limpiarlo
    index_limpio = df_tasa_ext.index.astype(str)
    for basura in basura_a_eliminar:
        index_limpio = index_limpio.str.replace(basura, '', regex=False)
    
    df_tasa_ext.index = index_limpio.str.strip()

    # C) UNIÓN DE TABLAS (CRUCE)
    # Ahora "Tokyo" (Histórico) coincidirá con "Tokyo" (Extranjeros limpio)
    df_analisis = df_variacion.join(df_tasa_ext, how='inner')
    
    print(f"✅ Cruce exitoso: {len(df_analisis)} prefecturas coincidieron.")
    
    if len(df_analisis) < 5:
        print("❌ ERROR: Pocos datos cruzados. Verifica manualmente los nombres.")
        print("Histórico ejemplo:", df_variacion.index[:3].tolist())
        print("Extranjeros ejemplo:", df_tasa_ext.index[:3].tolist())
        return

    # D) CÁLCULO DE PEARSON
    df_analisis = df_analisis.dropna()
    r, p_value = stats.pearsonr(df_analisis['Tasa_Extranjeros_Pct'], df_analisis['Variacion_Demografica_Pct'])
    
    print(f"\nResultados Estadísticos:")
    print(f" -> Coeficiente de Correlación (r): {r:.4f}")
    print(f" -> Valor P (Significancia): {p_value:.4f}")
    
    interpretation = ""
    if p_value < 0.05:
        interpretation = "RELACIÓN SIGNIFICATIVA (Rechazamos H0)"
        if r > 0:
            interpretation += "\nConfirmado: A mayor inmigración, mejor desempeño demográfico."
    else:
        interpretation = "NO SIGNIFICATIVO (No se rechaza H0)"
        
    print(f"CONCLUSIÓN: {interpretation}")

    # E) GRÁFICO DE REGRESIÓN
    plt.figure(figsize=(10, 7))
    sns.regplot(
        data=df_analisis,
        x='Tasa_Extranjeros_Pct',
        y='Variacion_Demografica_Pct',
        scatter_kws={'s': 100, 'alpha': 0.7, 'edgecolor': 'black'},
        line_kws={'color': 'red', 'label': f'r={r:.2f} ({interpretation.split()[0]})'}
    )
    
    plt.title(f'Prueba de Hipótesis: Inmigración vs Variación Poblacional (2000-{anio_fin})', fontsize=14)
    plt.xlabel('Tasa de Extranjeros 2024 (%)', fontsize=12)
    plt.ylabel(f'Crecimiento/Decrecimiento 2000-{anio_fin} (%)', fontsize=12)
    plt.axhline(0, color='gray', linestyle='--')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Etiquetar outliers
    for i in range(len(df_analisis)):
        row = df_analisis.iloc[i]
        if abs(row['Variacion_Demografica_Pct']) > 5 or row['Tasa_Extranjeros_Pct'] > 2.5:
            plt.text(row['Tasa_Extranjeros_Pct']+0.05, row['Variacion_Demografica_Pct'], df_analisis.index[i], fontsize=9)

    plt.tight_layout()
    plt.show()

#----------------------------------------------------
# 4. EJECUCIÓN PRINCIPAL
#----------------------------------------------------
if __name__ == "__main__":
    # 1. Cargar Histórico
    df_historico = demographic_dev()
    
    # 2. Cargar Extranjeros 2024
    df_final_ext, anio_ext = tasa_de_extranjeros_por_prefectura('population_per_prefecture_2024.csv')
    
    # 3. EJECUTAR LA PRUEBA FINAL
    if df_historico is not None and df_final_ext is not None:
        probar_hipotesis_correlacion(df_historico, df_final_ext)



#FIN DEL SCRIPT