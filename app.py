"""
Dashboard Interactivo — Porto Seguro
=====================================

Entregable para simular escenarios financieros modificando
los costos de clasificación y el umbral de decisión.

Datos requeridos:
    resultados_test.csv

Columnas requeridas:
    - y_test
    - probabilidad_siniestro

Ejecución local:
    streamlit run app.py
"""

# =============================================================================
# 0. IMPORTACIONES
# =============================================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =============================================================================
# 1. CONFIGURACIÓN DE PÁGINA
# =============================================================================

st.set_page_config(
    layout="wide",
    page_title="Dashboard Porto Seguro",
    page_icon="🚗",
)


# =============================================================================
# 2. PALETA DE COLORES
# =============================================================================

AZUL_MARINO = "#1A365D"
AZUL_PIZARRA = "#2B6CB0"

NARANJA_OXIDO = "#DD6B20"
ROJO_CARMESI = "#C53030"

AMARILLO_AMBAR = "#D69E2E"

FONDO_CLARO = "#FFFFFF"
GRIS_SUAVE = "#F7FAFC"
TEXTO_OSCURO = "#2D3748"

ESCALA_CONFUSION = [
    [0.0, "#EBF8FF"],
    [0.25, "#BEE3F8"],
    [0.5, "#63B3ED"],
    [0.75, "#3182CE"],
    [1.0, AZUL_MARINO],
]

PLOTLY_LAYOUT = dict(
    paper_bgcolor=FONDO_CLARO,
    plot_bgcolor=GRIS_SUAVE,
    font=dict(
        family="Inter, Segoe UI, Roboto, sans-serif",
        color=TEXTO_OSCURO
    ),
    margin=dict(
        l=50,
        r=30,
        t=50,
        b=50
    ),
)


# =============================================================================
# 3. CSS
# =============================================================================

st.markdown(
    f"""
    <style>

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

        section[data-testid="stSidebar"] {{
            background: {GRIS_SUAVE};
        }}

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


# =============================================================================
# 4. CARGA DE DATOS
# =============================================================================

DATA_PATH = "resultados_test.csv"

try:

    df = pd.read_csv(DATA_PATH)

    columnas_requeridas = {
        "y_test",
        "probabilidad_siniestro"
    }

    assert columnas_requeridas.issubset(df.columns), (
        "El archivo debe contener las columnas "
        "'y_test' y 'probabilidad_siniestro'."
    )

except FileNotFoundError:

    st.error(
        f"⚠️ Archivo no encontrado: `{DATA_PATH}`. "
        "Asegúrate de que esté en la misma carpeta que app.py."
    )

    st.stop()

except AssertionError as e:

    st.error(
        f"⚠️ Error en la estructura del archivo: {e}"
    )

    st.stop()


y_true = df["y_test"].to_numpy()

y_prob = df[
    "probabilidad_siniestro"
].to_numpy()


# =============================================================================
# 5. BARRA LATERAL
# =============================================================================

st.sidebar.markdown(
    "### 🎛️ Parámetros de Negocio"
)

# Valores coherentes con el análisis del notebook
cfp = st.sidebar.number_input(
    "Costo por Falso Positivo (CFP)",
    min_value=0,
    value=20000,
    step=5000,
    help=(
        "Costo asociado a clasificar como riesgoso "
        "a un asegurado que realmente no tendrá siniestro."
    ),
)

cfn = st.sidebar.number_input(
    "Costo por Falso Negativo (CFN)",
    min_value=0,
    value=100000,
    step=10000,
    help=(
        "Costo asociado a no detectar un asegurado "
        "que realmente tendrá un siniestro."
    ),
)

umbral_actual = st.sidebar.slider(
    "Umbral de Decisión Manual",
    min_value=0.00,
    max_value=1.00,
    value=0.50,
    step=0.01,
    help=(
        "Probabilidad mínima para clasificar "
        "un caso como siniestro."
    ),
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "💡 Al modificar CFP o CFN se actualizan "
    "automáticamente el umbral teórico, los costos "
    "y el óptimo de simulación."
)


# =============================================================================
# 6. FUNCIONES
# =============================================================================

def calcular_metricas(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    umbral: float
):
    """
    Calcula VP, VN, FP y FN para un umbral dado.
    """

    y_pred = (
        y_prob >= umbral
    ).astype(int)

    vp = int(
        np.sum(
            (y_pred == 1) &
            (y_true == 1)
        )
    )

    vn = int(
        np.sum(
            (y_pred == 0) &
            (y_true == 0)
        )
    )

    fp = int(
        np.sum(
            (y_pred == 1) &
            (y_true == 0)
        )
    )

    fn = int(
        np.sum(
            (y_pred == 0) &
            (y_true == 1)
        )
    )

    return {
        "VP": vp,
        "VN": vn,
        "FP": fp,
        "FN": fn
    }


def calcular_costo(
    fp: int,
    fn: int,
    cfp: float,
    cfn: float
):
    """
    Costo Total =
    FP × CFP + FN × CFN
    """

    return (
        fp * cfp +
        fn * cfn
    )


# =============================================================================
# 7. CURVA DE COSTOS
# =============================================================================

umbrales = np.arange(
    0.00,
    1.01,
    0.01
)

costos = []

fps_lista = []

fns_lista = []


for u in umbrales:

    m = calcular_metricas(
        y_true,
        y_prob,
        u
    )

    costos.append(
        calcular_costo(
            m["FP"],
            m["FN"],
            cfp,
            cfn
        )
    )

    fps_lista.append(
        m["FP"]
    )

    fns_lista.append(
        m["FN"]
    )


costos = np.array(costos)

fps_lista = np.array(
    fps_lista
)

fns_lista = np.array(
    fns_lista
)


# =============================================================================
# 8. ÓPTIMO DE SIMULACIÓN
# =============================================================================

idx_optimo = int(
    np.argmin(costos)
)

umbral_optimo = float(
    umbrales[idx_optimo]
)

costo_optimo = float(
    costos[idx_optimo]
)


# =============================================================================
# 9. UMBRAL TEÓRICO
# =============================================================================

if (cfp + cfn) > 0:

    p_star = (
        cfp /
        (cfp + cfn)
    )

else:

    p_star = 0.50


# =============================================================================
# 10. MÉTRICAS DEL UMBRAL MANUAL
# =============================================================================

metricas_actual = calcular_metricas(
    y_true,
    y_prob,
    umbral_actual
)

costo_actual = calcular_costo(
    metricas_actual["FP"],
    metricas_actual["FN"],
    cfp,
    cfn
)


vp = metricas_actual["VP"]

fp = metricas_actual["FP"]

fn = metricas_actual["FN"]


recall_actual = (
    vp / (vp + fn)
    if (vp + fn) > 0
    else 0
)


precision_actual = (
    vp / (vp + fp)
    if (vp + fp) > 0
    else 0
)


# =============================================================================
# 11. ESCENARIO BASE 0.50
# =============================================================================

metricas_base = calcular_metricas(
    y_true,
    y_prob,
    0.50
)

costo_base = calcular_costo(
    metricas_base["FP"],
    metricas_base["FN"],
    cfp,
    cfn
)


ahorro_proyectado = (
    costo_base -
    costo_optimo
)


# =============================================================================
# 12. ENCABEZADO
# =============================================================================

st.markdown(
    '<p class="main-title">'
    "🚗 Dashboard de Siniestros · Porto Seguro"
    "</p>",
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="sub-title">'
    "Simulación financiera del modelo seleccionado — "
    "optimización del umbral de decisión"
    "</p>",
    unsafe_allow_html=True,
)


# =============================================================================
# 13. KPIs PRINCIPALES
# =============================================================================

kpi1, kpi2, kpi3, kpi4, kpi5 = (
    st.columns(5)
)


with kpi1:

    st.metric(
        label="🎯 Óptimo de Simulación",
        value=f"{umbral_optimo:.2f}",
        help=(
            "Umbral que minimiza el costo total "
            "sobre los datos utilizados en el dashboard."
        ),
    )


with kpi2:

    st.metric(
        label="📐 Umbral Teórico p*",
        value=f"{p_star:.3f}",
        help=(
            "p* = CFP / (CFP + CFN)"
        ),
    )


with kpi3:

    st.metric(
        label="🔎 Recall",
        value=f"{recall_actual:.3f}",
        help=(
            "Proporción de siniestros reales "
            "correctamente detectados."
        ),
    )


with kpi4:

    st.metric(
        label="✅ Precisión",
        value=f"{precision_actual:.3f}",
        help=(
            "Proporción de predicciones positivas "
            "que efectivamente corresponden a siniestros."
        ),
    )


with kpi5:

    st.metric(
        label="📉 Ahorro vs. 0.50",
        value=f"${ahorro_proyectado:,.0f}",
        help=(
            "Diferencia entre el costo del umbral "
            "0.50 y el costo mínimo de simulación."
        ),
    )


# KPIs secundarios

kpi6, kpi7 = st.columns(2)


with kpi6:

    st.metric(
        label="💰 Costo con Umbral Manual",
        value=f"${costo_actual:,.0f}",
        help=(
            "Costo correspondiente al umbral "
            "seleccionado mediante el slider."
        ),
    )


with kpi7:

    st.metric(
        label="💵 Costo Mínimo Simulado",
        value=f"${costo_optimo:,.0f}",
    )


st.markdown("---")


# =============================================================================
# 14. CURVA DE COSTO VS UMBRAL
# =============================================================================

st.subheader(
    "📊 Curva de Costo Total vs. Umbral de Decisión"
)

fig_costo = go.Figure()


# Curva principal

fig_costo.add_trace(
    go.Scatter(
        x=umbrales,
        y=costos,
        mode="lines",
        line=dict(
            color=AZUL_PIZARRA,
            width=3
        ),
        name="Costo Total",
        hovertemplate=(
            "Umbral: %{x:.2f}"
            "<br>Costo: $%{y:,.0f}"
            "<extra></extra>"
        ),
    )
)


# Óptimo de simulación

fig_costo.add_trace(
    go.Scatter(
        x=[umbral_optimo],
        y=[costo_optimo],
        mode="markers+text",
        marker=dict(
            color=ROJO_CARMESI,
            size=14,
            symbol="star",
            line=dict(
                width=2,
                color=FONDO_CLARO
            ),
        ),
        text=[
            f"Óptimo simulación: "
            f"{umbral_optimo:.2f}"
        ],
        textposition="top center",
        textfont=dict(
            color=ROJO_CARMESI,
            size=13
        ),
        name="Óptimo de Simulación",
        hovertemplate=(
            f"<b>Óptimo de simulación: "
            f"{umbral_optimo:.2f}</b>"
            f"<br>Costo mínimo: "
            f"${costo_optimo:,.0f}"
            "<extra></extra>"
        ),
    )
)


# Umbral manual

fig_costo.add_vline(
    x=umbral_actual,
    line_dash="dash",
    line_color=NARANJA_OXIDO,
    line_width=2,
    annotation_text=(
        f"Manual: "
        f"{umbral_actual:.2f}"
    ),
    annotation_position="top right",
)


# Umbral estándar 0.50

fig_costo.add_vline(
    x=0.50,
    line_dash="dot",
    line_color="gray",
    line_width=2,
    annotation_text="Umbral 0.50",
    annotation_position="bottom right",
)


# Umbral teórico

fig_costo.add_vline(
    x=p_star,
    line_dash="dashdot",
    line_color=ROJO_CARMESI,
    line_width=2,
    annotation_text=(
        f"p*: {p_star:.3f}"
    ),
    annotation_position="top left",
)


fig_costo.update_layout(
    **PLOTLY_LAYOUT,
    xaxis_title=(
        "Umbral de Decisión"
    ),
    yaxis_title=(
        "Costo Total ($)"
    ),
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    height=450,
)


st.plotly_chart(
    fig_costo,
    use_container_width=True
)


# =============================================================================
# 15. MATRIZ DE CONFUSIÓN + DISTRIBUCIÓN
# =============================================================================

col_izq, col_der = (
    st.columns(2)
)


# -----------------------------------------------------------------------------
# MATRIZ DE CONFUSIÓN
# -----------------------------------------------------------------------------

with col_izq:

    st.subheader(
        "🔲 Matriz de Confusión Dinámica"
    )

    # Orden estándar:
    #
    # [[VN, FP],
    #  [FN, VP]]

    cm = np.array([
        [
            metricas_actual["VN"],
            metricas_actual["FP"]
        ],
        [
            metricas_actual["FN"],
            metricas_actual["VP"]
        ],
    ])


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

            x=[
                "Pred: No Siniestro",
                "Pred: Siniestro"
            ],

            y=[
                "Real: No Siniestro",
                "Real: Siniestro"
            ],

            text=etiquetas_texto,

            texttemplate="%{text}",

            textfont=dict(
                size=15,
                color=FONDO_CLARO
            ),

            colorscale=ESCALA_CONFUSION,

            showscale=False,

            hovertemplate=(
                "Predicho: %{x}"
                "<br>Real: %{y}"
                "<br>Cantidad: %{z:,}"
                "<extra></extra>"
            ),
        )
    )


    fig_cm.update_layout(
        **PLOTLY_LAYOUT,
        xaxis_title="Predicción",
        yaxis_title="Valor Real",
        height=420,
    )


    st.plotly_chart(
        fig_cm,
        use_container_width=True
    )


    st.caption(
        f"FP × CFP = "
        f"{metricas_actual['FP']:,} × "
        f"${cfp:,.0f} = "
        f"**${metricas_actual['FP'] * cfp:,.0f}** · "
        f"FN × CFN = "
        f"{metricas_actual['FN']:,} × "
        f"${cfn:,.0f} = "
        f"**${metricas_actual['FN'] * cfn:,.0f}**"
    )


# -----------------------------------------------------------------------------
# DISTRIBUCIÓN DE PROBABILIDADES
# -----------------------------------------------------------------------------

with col_der:

    st.subheader(
        "📈 Distribución de Probabilidades"
    )


    prob_clase_0 = (
        y_prob[y_true == 0]
    )

    prob_clase_1 = (
        y_prob[y_true == 1]
    )


    fig_dist = go.Figure()


    fig_dist.add_trace(
        go.Histogram(
            x=prob_clase_0,
            nbinsx=60,
            name="No Siniestro (y=0)",
            marker_color=AZUL_PIZARRA,
            opacity=0.65,
            hovertemplate=(
                "Prob: %{x:.2f}"
                "<br>Frecuencia: %{y:,}"
                "<extra></extra>"
            ),
        )
    )


    fig_dist.add_trace(
        go.Histogram(
            x=prob_clase_1,
            nbinsx=60,
            name="Siniestro (y=1)",
            marker_color=NARANJA_OXIDO,
            opacity=0.70,
            hovertemplate=(
                "Prob: %{x:.2f}"
                "<br>Frecuencia: %{y:,}"
                "<extra></extra>"
            ),
        )
    )


    # Umbral manual

    fig_dist.add_vline(
        x=umbral_actual,
        line_dash="dash",
        line_color=ROJO_CARMESI,
        line_width=2.5,
        annotation_text=(
            f"Manual: "
            f"{umbral_actual:.2f}"
        ),
        annotation_position="top right",
    )


    # Umbral teórico

    fig_dist.add_vline(
        x=p_star,
        line_dash="dashdot",
        line_color=AMARILLO_AMBAR,
        line_width=2,
        annotation_text=(
            f"p*: "
            f"{p_star:.3f}"
        ),
        annotation_position="top left",
    )


    # Umbral 0.50

    fig_dist.add_vline(
        x=0.50,
        line_dash="dot",
        line_color="gray",
        line_width=2,
        annotation_text="0.50",
        annotation_position="bottom right",
    )


    fig_dist.update_layout(
        **PLOTLY_LAYOUT,
        barmode="overlay",
        xaxis_title=(
            "Probabilidad de Siniestro"
        ),
        yaxis_title=(
            "Frecuencia"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=420,
    )


    st.plotly_chart(
        fig_dist,
        use_container_width=True
    )


    st.caption(
        "Cuanto mayor sea la separación entre las "
        "distribuciones de ambas clases, mayor es el "
        "poder discriminativo del modelo."
    )


# =============================================================================
# 16. TABLA DE SENSIBILIDAD
# =============================================================================

with st.expander(
    "📋 Tabla de Sensibilidad (Costo por Umbral)",
    expanded=False
):

    df_resumen = pd.DataFrame({
        "Umbral":
            np.round(
                umbrales,
                2
            ),

        "FP":
            fps_lista,

        "FN":
            fns_lista,

        "Costo Total ($)":
            costos,
    })


    st.dataframe(

        df_resumen.style.format({

            "Umbral":
                "{:.2f}",

            "FP":
                "{:,.0f}",

            "FN":
                "{:,.0f}",

            "Costo Total ($)":
                "${:,.0f}",

        }).highlight_min(
            subset=[
                "Costo Total ($)"
            ],
            color="#C6F6D5"
        ),

        use_container_width=True,

        height=350,
    )


# =============================================================================
# 17. PIE DE PÁGINA
# =============================================================================

st.markdown("---")


st.caption(
    "Dashboard generado con Streamlit & Plotly · "
    f"Dataset evaluado: {len(df):,} observaciones · "
    f"Tasa de siniestros: {y_true.mean():.2%} · "
    "Porto Seguro"
)
