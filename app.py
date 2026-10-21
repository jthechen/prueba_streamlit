"""
Dashboard Interactivo — Modelo LightGBM · Porto Seguro
=====================================================

Entregable gerencial para simular escenarios financieros iterando sobre el
umbral de decisión y minimizar el costo total de clasificación de siniestros
automotrices.

Autor : Equipo de Data Science
Stack : Python · Streamlit · Plotly · Pandas · NumPy
Datos : resultados_test.csv  (y_test, probabilidad_siniestro)

Ejecución local
---------------
    streamlit run app.py

Despliegue
----------
    Streamlit Community Cloud — apuntar al repositorio de GitHub.
"""

# ──────────────────────────────────────────────────────────────────────────────
# 0 · IMPORTACIONES
# ──────────────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ──────────────────────────────────────────────────────────────────────────────
# 1 · CONFIGURACIÓN DE PÁGINA (debe ir ANTES de cualquier otro st.*)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="Dashboard Porto Seguro",
    page_icon="🚗",
)

# ──────────────────────────────────────────────────────────────────────────────
# 2 · PALETA DE COLORES Y VARIABLES DE DISEÑO
# ──────────────────────────────────────────────────────────────────────────────
# Colores primarios
AZUL_MARINO   = "#1A365D"   # No Siniestro / base
AZUL_PIZARRA  = "#2B6CB0"   # Variante del primario

# Colores de alerta
NARANJA_OXIDO = "#DD6B20"   # Siniestro / riesgo
ROJO_CARMESI  = "#C53030"   # Variante de alerta

# Contraste
AMARILLO_AMBAR = "#D69E2E"  # Falsos Positivos

# Neutros
FONDO_CLARO    = "#FFFFFF"
GRIS_SUAVE     = "#F7FAFC"
TEXTO_OSCURO   = "#2D3748"

# Escala para la matriz de confusión (de blanco a azul marino)
ESCALA_CONFUSION = [
    [0.0, "#EBF8FF"],
    [0.25, "#BEE3F8"],
    [0.5, "#63B3ED"],
    [0.75, "#3182CE"],
    [1.0, AZUL_MARINO],
]

# Plantilla Plotly común — fondo blanco, tipografía limpia
PLOTLY_LAYOUT = dict(
    paper_bgcolor=FONDO_CLARO,
    plot_bgcolor=GRIS_SUAVE,
    font=dict(family="Inter, Segoe UI, Roboto, sans-serif", color=TEXTO_OSCURO),
    margin=dict(l=50, r=30, t=50, b=50),
)

# ──────────────────────────────────────────────────────────────────────────────
# 3 · CSS MÍNIMO PARA PULIR LA UI
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <style>
        /* KPI cards */
        div[data-testid="stMetric"] {{
            background: {GRIS_SUAVE};
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,.06);
        }}
        div[data-testid="stMetric"] label {{
            color: {AZUL_PIZARRA};
            font-weight: 600;
        }}
        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background: {GRIS_SUAVE};
        }}
        /* Título principal */
        .main-title {{
            color: {AZUL_MARINO};
            font-size: 2rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }}
        .sub-title {{
            color: {AZUL_PIZARRA};
            font-size: 1rem;
            margin-bottom: 1.5rem;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# 4 · CARGA DE DATOS  (con manejo de error amigable)
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH = "resultados_test.csv"

try:
    df = pd.read_csv(DATA_PATH)
    # Validación rápida de columnas esperadas
    assert {"y_test", "probabilidad_siniestro"}.issubset(df.columns), (
        "El archivo debe contener las columnas 'y_test' y 'probabilidad_siniestro'."
    )
except FileNotFoundError:
    st.error(
        f"⚠️ **Archivo no encontrado:** `{DATA_PATH}` no existe en el directorio "
        "de trabajo. Coloca el CSV con las columnas `y_test` y "
        "`probabilidad_siniestro` en la misma carpeta que `app.py` y recarga."
    )
    st.stop()
except AssertionError as e:
    st.error(f"⚠️ **Error de esquema:** {e}")
    st.stop()

# Vectores NumPy para cálculos rápidos
y_true = df["y_test"].values
y_prob = df["probabilidad_siniestro"].values

# ──────────────────────────────────────────────────────────────────────────────
# 5 · BARRA LATERAL — PARÁMETROS DE NEGOCIO
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown(f"### 🎛️ Parámetros de Negocio")

cfp = st.sidebar.number_input(
    "Costo por Falso Positivo (CFP)",
    min_value=0,
    value=10,
    step=1,
    help="Costo de investigar a un conductor que NO tendrá un siniestro.",
)

cfn = st.sidebar.number_input(
    "Costo por Falso Negativo (CFN)",
    min_value=0,
    value=500,
    step=10,
    help="Costo de pagar un siniestro que el modelo NO predijo.",
)

umbral_actual = st.sidebar.slider(
    "Umbral de Decisión Actual",
    min_value=0.00,
    max_value=1.00,
    value=0.50,
    step=0.01,
    help="Probabilidad a partir de la cual se clasifica como siniestro.",
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "💡 Ajusta CFP y CFN según la realidad de tu negocio. "
    "El dashboard recalculará el umbral óptimo automáticamente."
)

# ──────────────────────────────────────────────────────────────────────────────
# 6 · LÓGICA DE NEGOCIO — CÁLCULO VECTORIZADO DE COSTOS
# ──────────────────────────────────────────────────────────────────────────────

def calcular_metricas(y_true: np.ndarray, y_prob: np.ndarray, umbral: float):
    """
    Dado un umbral, calcula:
      - VP  (Verdaderos Positivos)
      - VN  (Verdaderos Negativos)
      - FP  (Falsos Positivos)
      - FN  (Falsos Negativos)

    Returns
    -------
    dict con VP, VN, FP, FN (enteros).
    """
    y_pred = (y_prob >= umbral).astype(int)
    vp = int(np.sum((y_pred == 1) & (y_true == 1)))
    vn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    return {"VP": vp, "VN": vn, "FP": fp, "FN": fn}


def calcular_costo(fp: int, fn: int, cfp: float, cfn: float) -> float:
    """Costo Total = FP × CFP  +  FN × CFN."""
    return fp * cfp + fn * cfn


# --- 6a. Iteración sobre todos los umbrales (0.00 → 1.00) ---
umbrales = np.arange(0.00, 1.01, 0.01)
costos = []
fps_lista = []
fns_lista = []

for u in umbrales:
    m = calcular_metricas(y_true, y_prob, u)
    costos.append(calcular_costo(m["FP"], m["FN"], cfp, cfn))
    fps_lista.append(m["FP"])
    fns_lista.append(m["FN"])

costos = np.array(costos)
fps_lista = np.array(fps_lista)
fns_lista = np.array(fns_lista)

# --- 6b. Umbral óptimo (mínimo costo) ---
idx_optimo = int(np.argmin(costos))
umbral_optimo = round(umbrales[idx_optimo], 2)
costo_optimo = costos[idx_optimo]

# --- 6c. Métricas del umbral actual del slider ---
metricas_actual = calcular_metricas(y_true, y_prob, umbral_actual)
costo_actual = calcular_costo(
    metricas_actual["FP"], metricas_actual["FN"], cfp, cfn
)

# --- 6d. Costo base (umbral 0.50) para calcular ahorro ---
metricas_base = calcular_metricas(y_true, y_prob, 0.50)
costo_base = calcular_costo(metricas_base["FP"], metricas_base["FN"], cfp, cfn)
ahorro_proyectado = costo_base - costo_optimo

# ──────────────────────────────────────────────────────────────────────────────
# 7 · ENCABEZADO DEL DASHBOARD
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<p class="main-title">🚗 Dashboard de Siniestros · Porto Seguro</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-title">'
    "Simulación financiera del modelo LightGBM — "
    "optimización del umbral de decisión"
    "</p>",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# 8 · SECCIÓN SUPERIOR — KPIs
# ──────────────────────────────────────────────────────────────────────────────
kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.metric(
        label="🎯 Umbral Óptimo Matemático",
        value=f"{umbral_optimo:.2f}",
        help="Umbral que minimiza el Costo Total dados los CFP y CFN actuales.",
    )

with kpi2:
    # Delta respecto al óptimo — positivo = estás gastando de más
    delta_costo = costo_actual - costo_optimo
    st.metric(
        label="💰 Costo Total Actual (slider)",
        value=f"${costo_actual:,.0f}",
        delta=f"+${delta_costo:,.0f} vs. óptimo" if delta_costo > 0 else "Óptimo ✓",
        delta_color="inverse",  # rojo = malo (más costo)
        help="Costo total con el umbral seleccionado en el slider.",
    )

with kpi3:
    st.metric(
        label="📉 Ahorro Proyectado",
        value=f"${ahorro_proyectado:,.0f}",
        delta="vs. umbral por defecto 0.50",
        delta_color="off",
        help="Diferencia entre operar con umbral 0.50 y operar con el óptimo.",
    )

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# 9 · GRÁFICO 1 — CURVA DE COSTO VS. UMBRAL
# ──────────────────────────────────────────────────────────────────────────────
st.subheader("📊 Curva de Costo Total vs. Umbral de Decisión")

fig_costo = go.Figure()

# Línea principal de costo
fig_costo.add_trace(
    go.Scatter(
        x=umbrales,
        y=costos,
        mode="lines",
        line=dict(color=AZUL_PIZARRA, width=3),
        name="Costo Total",
        hovertemplate="Umbral: %{x:.2f}<br>Costo: $%{y:,.0f}<extra></extra>",
    )
)

# Punto destacado: Umbral Óptimo (valle de la curva)
fig_costo.add_trace(
    go.Scatter(
        x=[umbral_optimo],
        y=[costo_optimo],
        mode="markers+text",
        marker=dict(color=ROJO_CARMESI, size=14, symbol="star",
                    line=dict(width=2, color=FONDO_CLARO)),
        text=[f"Óptimo: {umbral_optimo:.2f}"],
        textposition="top center",
        textfont=dict(color=ROJO_CARMESI, size=13, family="Inter, sans-serif"),
        name="Umbral Óptimo",
        hovertemplate=(
            f"<b>Umbral Óptimo: {umbral_optimo:.2f}</b><br>"
            f"Costo Mínimo: ${costo_optimo:,.0f}<extra></extra>"
        ),
    )
)

# Línea vertical dinámica: Umbral del slider
fig_costo.add_vline(
    x=umbral_actual,
    line_dash="dash",
    line_color=NARANJA_OXIDO,
    line_width=2,
    annotation_text=f"Actual: {umbral_actual:.2f}",
    annotation_position="top right",
    annotation_font=dict(color=NARANJA_OXIDO, size=12),
)

fig_costo.update_layout(
    **PLOTLY_LAYOUT,
    xaxis_title="Umbral de Decisión",
    yaxis_title="Costo Total ($)",
    showlegend=True,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    height=420,
)

st.plotly_chart(fig_costo, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
# 10 · GRÁFICOS 2 Y 3 — MATRIZ DE CONFUSIÓN + DISTRIBUCIÓN (lado a lado)
# ──────────────────────────────────────────────────────────────────────────────
col_izq, col_der = st.columns(2)

# ---------- GRÁFICO 2: MATRIZ DE CONFUSIÓN DINÁMICA ----------
with col_izq:
    st.subheader("🔲 Matriz de Confusión Dinámica")

    # Armar la matriz 2×2 en orden [[VN, FP], [FN, VP]]
    cm = np.array([
        [metricas_actual["VN"], metricas_actual["FP"]],
        [metricas_actual["FN"], metricas_actual["VP"]],
    ])

    # Etiquetas legibles para cada celda
    etiquetas_texto = np.array([
        [
            f"VN\n{metricas_actual['VN']:,}",
            f"FP\n{metricas_actual['FP']:,}",
        ],
        [
            f"FN\n{metricas_actual['FN']:,}",
            f"VP\n{metricas_actual['VP']:,}",
        ],
    ])

    fig_cm = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Pred: No Siniestro", "Pred: Siniestro"],
            y=["Real: Siniestro", "Real: No Siniestro"],
            text=etiquetas_texto,
            texttemplate="%{text}",
            textfont=dict(size=15, color=FONDO_CLARO),
            colorscale=ESCALA_CONFUSION,
            showscale=False,
            hovertemplate=(
                "Predicho: %{x}<br>Real: %{y}<br>"
                "Cantidad: %{z:,}<extra></extra>"
            ),
            # Invertir eje Y para que Real: No Siniestro quede arriba
            # (convención estándar scikit-learn)
        )
    )

    # Invertir eje Y para convención estándar
    fig_cm.update_yaxes(autorange="reversed")

    fig_cm.update_layout(
        **PLOTLY_LAYOUT,
        xaxis_title="Predicción",
        yaxis_title="Valor Real",
        height=420,
    )

    st.plotly_chart(fig_cm, use_container_width=True)

    # Detalle de costos debajo de la matriz
    st.caption(
        f"FP × CFP = {metricas_actual['FP']:,} × \${cfp} "
        f"= **\${metricas_actual['FP'] * cfp:,.0f}** · "
        f"FN × CFN = {metricas_actual['FN']:,} × \${cfn} "
        f"= **\${metricas_actual['FN'] * cfn:,.0f}**"
    )

# ---------- GRÁFICO 3: DISTRIBUCIÓN DE PROBABILIDADES ----------
with col_der:
    st.subheader("📈 Distribución de Probabilidades")

    # Separar probabilidades por clase
    prob_clase_0 = y_prob[y_true == 0]
    prob_clase_1 = y_prob[y_true == 1]

    fig_dist = go.Figure()

    # Histograma — Clase 0 (No Siniestro)
    fig_dist.add_trace(
        go.Histogram(
            x=prob_clase_0,
            nbinsx=60,
            name="No Siniestro (y=0)",
            marker_color=AZUL_PIZARRA,
            opacity=0.65,
            hovertemplate="Prob: %{x:.2f}<br>Frecuencia: %{y:,}<extra></extra>",
        )
    )

    # Histograma — Clase 1 (Siniestro)
    fig_dist.add_trace(
        go.Histogram(
            x=prob_clase_1,
            nbinsx=60,
            name="Siniestro (y=1)",
            marker_color=NARANJA_OXIDO,
            opacity=0.70,
            hovertemplate="Prob: %{x:.2f}<br>Frecuencia: %{y:,}<extra></extra>",
        )
    )

    # Línea vertical interactiva — Umbral actual
    fig_dist.add_vline(
        x=umbral_actual,
        line_dash="dash",
        line_color=ROJO_CARMESI,
        line_width=2.5,
        annotation_text=f"Umbral: {umbral_actual:.2f}",
        annotation_position="top right",
        annotation_font=dict(color=ROJO_CARMESI, size=12),
    )

    fig_dist.update_layout(
        **PLOTLY_LAYOUT,
        barmode="overlay",
        xaxis_title="Probabilidad de Siniestro",
        yaxis_title="Frecuencia",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
    )

    st.plotly_chart(fig_dist, use_container_width=True)

    st.caption(
        "Cuanto más separadas estén las distribuciones, mayor poder "
        "discriminativo tiene el modelo."
    )

# ──────────────────────────────────────────────────────────────────────────────
# 11 · SECCIÓN ADICIONAL — TABLA RESUMEN DE SENSIBILIDAD
# ──────────────────────────────────────────────────────────────────────────────
with st.expander("📋 Tabla de Sensibilidad (Costo por Umbral)", expanded=False):
    df_resumen = pd.DataFrame({
        "Umbral": np.round(umbrales, 2),
        "FP": fps_lista,
        "FN": fns_lista,
        "Costo Total ($)": costos,
    })
    # Resaltar la fila del óptimo
    st.dataframe(
        df_resumen.style.format({
            "Umbral": "{:.2f}",
            "FP": "{:,.0f}",
            "FN": "{:,.0f}",
            "Costo Total ($)": "${:,.0f}",
        }).highlight_min(subset=["Costo Total ($)"], color="#C6F6D5"),
        use_container_width=True,
        height=350,
    )

# ──────────────────────────────────────────────────────────────────────────────
# 12 · PIE DE PÁGINA
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Dashboard generado con Streamlit & Plotly · "
    f"Dataset: {len(df):,} observaciones · "
    f"Tasa de siniestros: {y_true.mean():.2%} · "
    "Modelo: LightGBM · Porto Seguro"
)
