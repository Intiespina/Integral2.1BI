"""
ConecTel SA — Simulador de Riesgo de Morosidad
Aplicación Streamlit para predecir la probabilidad de mora >90 días.

Modelo: Regresión Logística (mejor ROC-AUC en validación cruzada 5-fold: 0.6821)
Pipeline: SimpleImputer (mediana) → StandardScaler → LogisticRegression
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ─── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="ConecTel SA – Simulador de Riesgo de Morosidad",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Estilos personalizados ────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Header principal */
    .main-header {
        background: linear-gradient(135deg, #1a3a6b 0%, #2563eb 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .main-header h1 { color: white; margin: 0; font-size: 1.8rem; }
    .main-header p  { color: #cbd5e1; margin: 0.3rem 0 0 0; font-size: 0.95rem; }

    /* Tarjetas de resultado */
    .risk-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-weight: bold;
    }
    .risk-high   { background: #fef2f2; border: 2px solid #ef4444; color: #991b1b; }
    .risk-medium { background: #fffbeb; border: 2px solid #f59e0b; color: #92400e; }
    .risk-low    { background: #f0fdf4; border: 2px solid #22c55e; color: #166534; }

    /* Indicador de probabilidad */
    .prob-label { font-size: 3rem; font-weight: 800; margin: 0.5rem 0; }
    .prob-desc  { font-size: 1rem; opacity: 0.85; }

    /* Sección de factores */
    .factor-box {
        background: #f8fafc;
        border-left: 4px solid #2563eb;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin: 0.4rem 0;
        font-size: 0.9rem;
    }

    /* Sidebar secciones */
    .sidebar-section {
        background: #f1f5f9;
        padding: 0.5rem 0.8rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        font-weight: 600;
        color: #1e40af;
        font-size: 0.85rem;
    }

    /* Eliminar padding extra */
    .block-container { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)


# ─── Carga del pipeline (cacheado para no recargar en cada interacción) ────────
@st.cache_resource(show_spinner="Cargando modelo…")
def cargar_pipeline():
    """Carga y devuelve el modelo, scaler, imputer y lista de features desde disco."""
    base = os.path.dirname(os.path.abspath(__file__))
    modelo     = joblib.load(os.path.join(base, "modelo.joblib"))
    scaler     = joblib.load(os.path.join(base, "scaler.joblib"))
    imputer    = joblib.load(os.path.join(base, "imputer.joblib"))
    top_feats  = joblib.load(os.path.join(base, "top_features.joblib"))
    return modelo, scaler, imputer, top_feats


# ─── Función de preprocesamiento ──────────────────────────────────────────────
def construir_features(datos: dict, top_features: list) -> pd.DataFrame:
    """
    Replica exactamente el pipeline de preprocesamiento del notebook:
      1. Codificación ordinal (tipo_contrato, descuento_activo)
      2. One-Hot Encoding (metodo_pago, genero, region)
      3. Feature derivada: mora_por_antiguedad
    Devuelve un DataFrame con las columnas en el orden exacto de top_features.
    """
    row = {f: 0 for f in top_features}

    # ── Numéricas directas ─────────────────────────────────────────────────────
    numericas = {
        "cambios_plan_12m":     datos["cambios_plan_12m"],
        "dias_mora_hist":       datos["dias_mora_hist"],
        "velocidad_mbps":       datos["velocidad_mbps"] if datos["tiene_internet"] else 0,
        "llamadas_soporte_6m":  datos["llamadas_soporte_6m"],
        "ingreso_estimado_clp": datos["ingreso_estimado_clp"],
        "antiguedad_meses":     datos["antiguedad_meses"],
        "tiene_tv":             int(datos["tiene_tv"]),
        "tiene_internet":       int(datos["tiene_internet"]),
        "num_servicios":        datos.get("num_servicios", 1),
        "nps":                  datos.get("nps", 5.0),
        "edad":                 datos.get("edad", 35),
    }
    for col, val in numericas.items():
        if col in row:
            row[col] = val

    # ── Feature derivada ───────────────────────────────────────────────────────
    if "mora_por_antiguedad" in row:
        row["mora_por_antiguedad"] = datos["dias_mora_hist"] / (datos["antiguedad_meses"] + 1)

    if "ratio_factura_ingreso" in row and "factura_mensual_clp" in datos:
        row["ratio_factura_ingreso"] = (
            datos["factura_mensual_clp"] / (datos["ingreso_estimado_clp"] + 1)
        )

    if "indice_conflictividad" in row and "reclamos_12m" in datos:
        row["indice_conflictividad"] = datos["reclamos_12m"] + datos["llamadas_soporte_6m"]

    # ── Ordinal ────────────────────────────────────────────────────────────────
    if "tipo_contrato_cod" in row:
        row["tipo_contrato_cod"] = {"Mensual": 1, "Anual": 2, "Bianual": 3}[datos["tipo_contrato"]]
    if "plan_cod" in row:
        row["plan_cod"] = {"Básico": 1, "Estándar": 2, "Premium": 3}[datos.get("plan", "Estándar")]
    if "descuento_activo_cod" in row:
        row["descuento_activo_cod"] = 1 if datos["descuento_activo"] == "Sí" else 0

    # ── OHE: metodo_pago (referencia = Cheque) ─────────────────────────────────
    mp = datos["metodo_pago"]
    for ohe in ["metodo_pago_Débito automático", "metodo_pago_Efectivo",
                "metodo_pago_Transferencia", "metodo_pago_WebPay"]:
        if ohe in row:
            row[ohe] = 1 if ohe == f"metodo_pago_{mp}" else 0

    # ── OHE: genero (referencia = Femenino) ────────────────────────────────────
    g = datos["genero"]
    for ohe in ["genero_Masculino", "genero_No binario",
                "genero_No especificado", "genero_Prefiero no decir"]:
        if ohe in row:
            row[ohe] = 1 if ohe == f"genero_{g}" else 0

    # ── OHE: region (referencia = Antofagasta) ─────────────────────────────────
    r = datos["region"]
    for ohe in [c for c in top_features if c.startswith("region_")]:
        nombre_region = ohe.replace("region_", "")
        row[ohe] = 1 if r == nombre_region else 0

    # DataFrame en el orden exacto del pipeline
    return pd.DataFrame([row])[top_features]


# ─── Función de visualización del resultado ───────────────────────────────────
def mostrar_resultado(proba: float, datos: dict):
    """Renderiza la tarjeta de riesgo con probabilidad, nivel y factores clave."""

    # Umbrales calibrados para ~11% de mora en la población
    if proba >= 0.45:
        nivel, clase, icono, msg = (
            "RIESGO ALTO",
            "risk-high",
            "⚠️",
            "Alta probabilidad de mora. Se recomienda **acción preventiva inmediata**: "
            "contacto del equipo de cobranza, oferta de plan de pago o renegociación.",
        )
    elif proba >= 0.20:
        nivel, clase, icono, msg = (
            "RIESGO MEDIO",
            "risk-medium",
            "🔶",
            "Probabilidad moderada. Se recomienda **monitoreo activo**: campaña de "
            "satisfacción y verificación de método de pago.",
        )
    else:
        nivel, clase, icono, msg = (
            "RIESGO BAJO",
            "risk-low",
            "✅",
            "Baja probabilidad de mora. **Sin acción urgente requerida**. "
            "Continuar monitoreo periódico.",
        )

    # Tarjeta principal
    st.markdown(
        f"""
        <div class="risk-card {clase}">
            <div style="font-size:2rem;">{icono}</div>
            <div style="font-size:1.4rem; font-weight:800; margin:0.3rem 0;">{nivel}</div>
            <div class="prob-label">{proba*100:.1f}%</div>
            <div class="prob-desc">probabilidad estimada de mora a 90 días</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Barra de probabilidad
    st.markdown("<br>", unsafe_allow_html=True)
    col_bar1, col_bar2 = st.columns([9, 1])
    with col_bar1:
        st.progress(float(proba), text=f"Probabilidad: {proba*100:.1f}%")
    st.caption(msg)

    st.divider()

    # ── Factores de riesgo clave ───────────────────────────────────────────────
    st.markdown("#### 🔍 Factores de Riesgo Detectados")

    factores = []
    if datos["tipo_contrato"] == "Mensual":
        factores.append("📋 **Contrato Mensual** — menor compromiso, mayor riesgo de abandono.")
    if datos["metodo_pago"] in ("Efectivo", "Cheque"):
        factores.append(f"💵 **Pago en {datos['metodo_pago']}** — método manual con mayor tasa histórica de mora.")
    if datos["dias_mora_hist"] > 15:
        factores.append(f"📅 **Mora histórica: {datos['dias_mora_hist']} días** — predictor más fuerte de mora futura.")
    if datos.get("nps") and datos["nps"] < 5:
        factores.append(f"😞 **NPS bajo: {datos['nps']:.0f}/10** — insatisfacción correlacionada con impago.")
    if datos["llamadas_soporte_6m"] >= 4:
        factores.append(f"📞 **{datos['llamadas_soporte_6m']} llamadas al soporte** — alta conflictividad del cliente.")
    if datos["cambios_plan_12m"] >= 2:
        factores.append(f"🔄 **{datos['cambios_plan_12m']} cambios de plan** — comportamiento inestable.")
    if datos["antiguedad_meses"] < 12:
        factores.append(f"🆕 **Antigüedad: {datos['antiguedad_meses']} meses** — clientes nuevos tienen mayor riesgo.")
    if datos["ingreso_estimado_clp"] < 400_000:
        factores.append("💰 **Ingreso estimado bajo** — mayor presión financiera.")

    if factores:
        for f in factores:
            st.markdown(
                f'<div class="factor-box">{f}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.info("Sin factores de riesgo significativos detectados para este cliente.")

    # ── Recomendación comercial ────────────────────────────────────────────────
    st.divider()
    st.markdown("#### 💼 Recomendación para el Equipo Comercial")

    if proba >= 0.45:
        st.error(
            "**Prioridad ALTA.** Activar protocolo de cobranza preventiva en los próximos 7 días. "
            "Ofrecer plan de pago a cuotas o migración a débito automático."
        )
    elif proba >= 0.20:
        st.warning(
            "**Prioridad MEDIA.** Incluir en campaña de satisfacción y revisar método de pago. "
            "Re-evaluar en 30 días."
        )
    else:
        st.success(
            "**Sin urgencia.** Mantener en seguimiento regular. "
            "Considerar oferta de upgrade de plan para aumentar fidelización."
        )


# ═══════════════════════════════════════════════════════════════════════════════
# INTERFAZ PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="main-header">
        <h1>📡 ConecTel SA — Simulador de Riesgo de Morosidad</h1>
        <p>Sistema de alerta temprana · Predicción de mora a 90 días · Regresión Logística (ROC-AUC CV = 0.68)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Cargar pipeline ────────────────────────────────────────────────────────────
try:
    modelo, scaler, imputer, top_features = cargar_pipeline()
except FileNotFoundError:
    st.error(
        "No se encontraron los archivos del modelo (`modelo.joblib`, `scaler.joblib`, "
        "`imputer.joblib`, `top_features.joblib`). "
        "Asegúrate de que estén en el mismo directorio que `app.py` y que el notebook "
        "haya sido ejecutado completamente."
    )
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — FORMULARIO DE ENTRADA
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📝 Datos del Cliente")
    st.caption("Completa todos los campos y presiona **Predecir** para evaluar el riesgo.")

    # ── PERFIL DEL CLIENTE ─────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">👤 Perfil del Cliente</div>', unsafe_allow_html=True)

    region = st.selectbox(
        "Región",
        ["Antofagasta", "Araucanía", "Atacama", "Biobío", "Coquimbo",
         "Los Lagos", "Maule", "Metropolitana", "O'Higgins", "Valparaíso"],
        index=7,  # default: Metropolitana
    )
    edad = st.slider("Edad (años)", min_value=18, max_value=90, value=35)
    genero = st.selectbox(
        "Género",
        ["Femenino", "Masculino", "No binario", "Prefiero no decir", "No especificado"],
    )

    # ── CONTRATO Y SERVICIOS ───────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">📋 Contrato y Servicios</div>', unsafe_allow_html=True)

    tipo_contrato = st.radio(
        "Tipo de contrato",
        ["Mensual", "Anual", "Bianual"],
        horizontal=True,
    )
    plan = st.radio("Plan", ["Básico", "Estándar", "Premium"], index=1, horizontal=True)
    antiguedad_meses = st.slider("Antigüedad (meses)", min_value=1, max_value=120, value=24)
    descuento_activo = st.radio("¿Tiene descuento activo?", ["No", "Sí"], horizontal=True)

    col_sv1, col_sv2 = st.columns(2)
    with col_sv1:
        tiene_tv = st.checkbox("TV incluida", value=True)
    with col_sv2:
        tiene_internet = st.checkbox("Internet incluido", value=True)

    num_servicios = st.selectbox("N° de servicios", [1, 2, 3], index=1)

    if tiene_internet:
        velocidad_mbps = st.selectbox(
            "Velocidad internet (Mbps)",
            [50, 100, 200, 500, 1000],
            index=1,
        )
    else:
        velocidad_mbps = 0
        st.caption("Sin servicio de internet contratado.")

    # ── DATOS FINANCIEROS ──────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">💰 Datos Financieros</div>', unsafe_allow_html=True)

    ingreso_estimado_clp = st.number_input(
        "Ingreso estimado mensual (CLP)",
        min_value=200_000,
        max_value=4_000_000,
        value=600_000,
        step=50_000,
        format="%d",
    )
    factura_mensual_clp = st.number_input(
        "Factura mensual (CLP)",
        min_value=5_000,
        max_value=200_000,
        value=35_000,
        step=1_000,
        format="%d",
    )
    metodo_pago = st.selectbox(
        "Método de pago",
        ["Débito automático", "WebPay", "Transferencia", "Efectivo", "Cheque"],
    )

    # ── HISTORIAL Y COMPORTAMIENTO ─────────────────────────────────────────────
    st.markdown('<div class="sidebar-section">📊 Historial y Comportamiento</div>', unsafe_allow_html=True)

    nps = st.slider(
        "Satisfacción del cliente (NPS 1–10)",
        min_value=1.0,
        max_value=10.0,
        value=7.0,
        step=0.5,
    )
    dias_mora_hist = st.number_input(
        "Días de mora histórico",
        min_value=0,
        max_value=90,
        value=0,
        step=1,
    )
    llamadas_soporte_6m = st.number_input(
        "Llamadas al soporte (últimos 6 meses)",
        min_value=0,
        max_value=15,
        value=1,
        step=1,
    )
    reclamos_12m = st.number_input(
        "Reclamos formales (últimos 12 meses)",
        min_value=0,
        max_value=10,
        value=0,
        step=1,
    )
    cambios_plan_12m = st.number_input(
        "Cambios de plan (últimos 12 meses)",
        min_value=0,
        max_value=5,
        value=0,
        step=1,
    )

    st.markdown("---")

    # ── BOTÓN DE PREDICCIÓN ────────────────────────────────────────────────────
    predecir = st.button(
        "🔍  Evaluar Riesgo de Morosidad",
        type="primary",
        use_container_width=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# ÁREA PRINCIPAL — RESULTADO
# ═══════════════════════════════════════════════════════════════════════════════

if predecir:
    # ── Construir vector de features ──────────────────────────────────────────
    datos_cliente = {
        "region":               region,
        "edad":                 edad,
        "genero":               genero,
        "tipo_contrato":        tipo_contrato,
        "plan":                 plan,
        "antiguedad_meses":     int(antiguedad_meses),
        "descuento_activo":     descuento_activo,
        "tiene_tv":             tiene_tv,
        "tiene_internet":       tiene_internet,
        "num_servicios":        num_servicios,
        "velocidad_mbps":       int(velocidad_mbps),
        "ingreso_estimado_clp": float(ingreso_estimado_clp),
        "factura_mensual_clp":  float(factura_mensual_clp),
        "metodo_pago":          metodo_pago,
        "nps":                  float(nps),
        "dias_mora_hist":       int(dias_mora_hist),
        "llamadas_soporte_6m":  int(llamadas_soporte_6m),
        "reclamos_12m":         int(reclamos_12m),
        "cambios_plan_12m":     int(cambios_plan_12m),
    }

    X_raw  = construir_features(datos_cliente, top_features)
    X_imp  = pd.DataFrame(imputer.transform(X_raw),  columns=top_features)
    X_sc   = pd.DataFrame(scaler.transform(X_imp),   columns=top_features)

    proba  = float(modelo.predict_proba(X_sc)[0, 1])

    # ── Layout resultado ──────────────────────────────────────────────────────
    col_res, col_det = st.columns([1, 1], gap="large")

    with col_res:
        st.markdown("### Resultado de la Evaluación")
        mostrar_resultado(proba, datos_cliente)

    with col_det:
        st.markdown("### Detalle del Vector de Features")
        st.caption("Valores procesados por el pipeline (tras imputer + scaler).")

        df_display = pd.DataFrame({
            "Feature":          top_features,
            "Valor procesado":  X_sc.values[0].round(4),
        })
        st.dataframe(df_display, use_container_width=True, height=480)

        st.markdown("---")
        st.markdown("#### ℹ️ Modelo utilizado")
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Algoritmo", "Reg. Logística")
        col_m2.metric("ROC-AUC (CV)", "0.6821")
        col_m1.metric("Regularización C", "1.0")
        col_m2.metric("class_weight", "balanced")

else:
    # ── Estado inicial: instrucciones ─────────────────────────────────────────
    col_info1, col_info2, col_info3 = st.columns(3)

    with col_info1:
        st.info(
            "**📋 Paso 1**\n\n"
            "Completa el formulario en la barra lateral izquierda con los datos del cliente "
            "que deseas evaluar.",
        )
    with col_info2:
        st.info(
            "**🔍 Paso 2**\n\n"
            "Haz clic en **Evaluar Riesgo de Morosidad** para que el modelo procese "
            "los datos y genere la predicción.",
        )
    with col_info3:
        st.info(
            "**📊 Paso 3**\n\n"
            "Revisa la probabilidad estimada, el nivel de riesgo y los factores clave "
            "detectados para tomar una decisión informada.",
        )

    st.markdown("---")

    # ── Tabla de umbrales de decisión ─────────────────────────────────────────
    st.markdown("#### 🎯 Umbrales de Decisión y Acciones Recomendadas")

    df_umbrales = pd.DataFrame({
        "Nivel de Riesgo":  ["🟢 Bajo",     "🟡 Medio",    "🔴 Alto"],
        "Probabilidad":     ["< 20%",       "20% – 45%",   "≥ 45%"],
        "Acción":           [
            "Sin urgencia – monitoreo periódico",
            "Campaña de satisfacción y revisión del método de pago",
            "Protocolo de cobranza preventiva inmediata",
        ],
        "Costo estimado FN": ["$120.000 CLP", "$120.000 CLP", "$120.000 CLP"],
        "Costo campaña FP":  ["$5.000 CLP",   "$5.000 CLP",  "$5.000 CLP"],
    })
    st.dataframe(df_umbrales, use_container_width=True, hide_index=True)

    st.caption(
        "Modelo entrenado con 6.400 registros históricos de ConecTel SA (80% train / 20% test). "
        "Los umbrales están calibrados para minimizar Falsos Negativos (clientes morosos no detectados), "
        "dado que su costo es 24× mayor que el de un Falso Positivo."
    )


# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small>ConecTel SA · IICG 514 Business Intelligence · "
    "Inti Espina y Bastián Morales · 2026 · "
    "Modelo: RandomForestClassifier (scikit-learn)</small>",
    unsafe_allow_html=True,
)
