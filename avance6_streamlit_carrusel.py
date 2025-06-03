# AVANCE6 en Streamlit con carrusel de gráficas

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import zipfile
import os
from io import BytesIO
from collections import Counter

# --- FUNCIONES DE CARGA ---
def make_unique(headers):
    counts = Counter()
    new_headers = []
    for h in headers:
        counts[h] += 1
        if counts[h] > 1:
            new_headers.append(f"{h}_{counts[h]-1}")
        else:
            new_headers.append(h)
    return new_headers

def read_imotions_csv(file, participant_name, header_index, tipo):
    content = file.read().decode('utf-8')
    lines = content.splitlines()
    headers = make_unique(lines[header_index].strip().split(","))
    data = "\n".join(lines[header_index + 1:])
    df = pd.read_csv(StringIO(data), names=headers)
    df["Participant"] = participant_name
    df["Tipo"] = tipo
    return df

def upload_and_concat(tipo, header_index):
    uploaded_files = st.file_uploader(f"Sube archivos de {tipo} (CSV)", accept_multiple_files=True, type="csv")
    dfs = []
    if uploaded_files:
        for file in uploaded_files:
            participant = file.name.replace(".csv", "").strip()
            df = read_imotions_csv(file, participant, header_index, tipo)
            dfs.append(df)
        df_merged = pd.concat(dfs, ignore_index=True)
        st.success(f"{tipo} fusionado con {len(dfs)} archivo(s).")
        st.download_button(f"Descargar {tipo} mergeado", df_merged.to_csv(index=False).encode(), file_name=f"{tipo.lower()}_merged.csv", mime='text/csv')
        return df_merged
    return pd.DataFrame()

# --- APP STREAMLIT ---
st.set_page_config(layout="wide")
st.title("AVANCE6 - Análisis de Eyetracking con carrusel")

with st.sidebar:
    st.header("Carga de archivos")
    df_et = upload_and_concat("Eyetracking", 25)
    df_fea = upload_and_concat("FEA", 25)
    df_gsr = upload_and_concat("GSR", 27)

# --- ANÁLISIS EYETRACKING ---
if not df_et.empty:
    st.header("Análisis de Eyetracking")

    df_et["ET_TimeSignal"] = pd.to_numeric(df_et["ET_TimeSignal"], errors="coerce")
    df_et = df_et.dropna(subset=["ET_TimeSignal", "SourceStimuliName"])

    tabla_et = df_et.groupby("SourceStimuliName").agg(
        Tiempo_Medio=("ET_TimeSignal", "mean"),
        Desviacion_Estandar=("ET_TimeSignal", "std"),
        Conteo=("ET_TimeSignal", "count")
    ).reset_index()

    st.subheader("Tabla de estadísticos por Estímulo")
    st.dataframe(tabla_et)
    st.download_button("Descargar tabla estadística", tabla_et.to_csv(index=False).encode(), file_name="tabla_eyetracking.csv", mime='text/csv')

    # ANOVA
    estimulos = df_et["SourceStimuliName"].unique()
    data_por_estimulo = [df_et[df_et["SourceStimuliName"] == stim]["ET_TimeSignal"] for stim in estimulos]
    anova_result = stats.f_oneway(*data_por_estimulo)
    f_stat = anova_result.statistic
    p_value = anova_result.pvalue
    f_squared = (f_stat * (len(estimulos) - 1)) / (len(df_et) - len(estimulos)) if len(estimulos) > 1 else None

    estad_txt = f"ANOVA F-statistic: {f_stat:.4f}\\n"
    estad_txt += f"p-value: {p_value:.4e}\\n"
    
    
    if f_squared:
        estad_txt += f"F-squared: {f_squared:.4f}
"
    st.text_area("Estadísticos", estad_txt, height=100)
    st.download_button("Descargar estadísticos", estad_txt, file_name="estadisticos_eyetracking.txt")

    # GRAFICAS
    st.subheader("Gráficas de Eyetracking")
    sns.set(style="whitegrid")
    imagenes = []

    fig1, ax1 = plt.subplots(figsize=(10,6))
    sns.barplot(data=df_et, x="SourceStimuliName", y="ET_TimeSignal", ci="sd", capsize=0.1, ax=ax1)
    ax1.set_title("Tiempo promedio por Estímulo")
    ax1.set_ylabel("Tiempo de permanencia")
    ax1.tick_params(axis='x', rotation=45)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(10,6))
    sns.violinplot(data=df_et, x="SourceStimuliName", y="ET_TimeSignal", ax=ax2)
    ax2.set_title("Distribución del Tiempo por Estímulo")
    ax2.tick_params(axis='x', rotation=45)
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(10,6))
    sns.boxplot(data=df_et, x="SourceStimuliName", y="ET_TimeSignal", ax=ax3)
    ax3.set_title("Boxplot por Estímulo")
    ax3.tick_params(axis='x', rotation=45)
    st.pyplot(fig3)

    colores = sns.color_palette("tab10", n_colors=len(estimulos))
    for idx, stim in enumerate(estimulos):
        subset = df_et[df_et["SourceStimuliName"] == stim]
        fig, ax = plt.subplots(figsize=(8,5))
        sns.histplot(subset["ET_TimeSignal"], kde=True, color=colores[idx], ax=ax)
        ax.set_title(f"Histograma - {stim}")
        ax.set_xlabel("Tiempo de permanencia")
        st.pyplot(fig)
