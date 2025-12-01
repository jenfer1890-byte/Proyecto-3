# simulador.py
# Simulador: Masa de un material con densidad dependiente de temperatura
# Ejecutar: streamlit run simulador.py

import streamlit as st
import numpy as np
import plotly.graph_objects as go
import math


st.set_page_config(page_title="Simulador: Masa con densidad dependiente de temperatura", layout="wide")

# Metadatos requeridos por la entrega
PROYECTO_TITULO = "PROYECTO 3: MASA DE UN MATERIAL CON DENSIDAD Y DEPENDIENTE DE TEMPERATURA"
AUTORES = "Autor: Jennifer Rivera Andrade y Denisse Valeria Guzman Barragan"
MATERIA = "Materia: Cálculo de Varias Variables"
DOCENTE = "Docente: Javier Garcia Lara"

# Mostrar encabezado y metadatos
st.title(PROYECTO_TITULO)
st.markdown(f"{AUTORES}**  \n*{MATERIA}*  \n*{DOCENTE}*")
st.markdown("---")
st.markdown(
    "Descripción breve: Simulador de un cilindro cuya densidad depende linealmente de la temperatura. "
    "La densidad está dada por:  \n"
    r"$$\rho(z)=\rho_0\big(1+\alpha(T_0 + k z)\big)$$"
    "  \nDonde \(T(z)=T_0 + k z\). El simulador permite cambiar parámetros y ver la distribución de densidad, la masa total y el centroide."
)


st.sidebar.header("Controles del simulador (sliders)")
# valores por defecto (guardados para reset)
DEFAULTS = dict(rho0=7800.0, alpha=-0.0005, T0=20.0, deltaT=0.5, R=0.25, H=1.2, slices=80)

# Inicializar session_state con defaults la primera vez
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Sliders con valores visibles
st.sidebar.markdown("Ajusta los parámetros y la figura se actualizará automáticamente.")
st.session_state['R'] = st.sidebar.number_input(
    "Radio R (m)", min_value=0.01, value=float(st.session_state['R']), step=0.01, format="%.3f"
)
st.session_state['H'] = st.sidebar.number_input(
    "Altura H (m)", min_value=0.01, value=float(st.session_state['H']), step=0.01, format="%.3f"
)
st.session_state['rho0'] = st.sidebar.number_input(
    "ρ₀ (kg/m³)", min_value=0.1, value=float(st.session_state['rho0']), step=1.0, format="%.1f"
)
st.session_state['alpha'] = st.sidebar.number_input(
    "α (1/°C)", value=float(st.session_state['alpha']), step=1e-6, format="%.6f"
)
st.session_state['T0'] = st.sidebar.number_input(
    "T₀ (°C) — temp base", value=float(st.session_state['T0']), step=0.1, format="%.2f"
)
st.session_state['deltaT'] = st.sidebar.number_input(
    "ΔT (°C) (cima - base)", value=float(st.session_state['deltaT']), step=0.1, format="%.3f"
)
st.session_state['slices'] = st.sidebar.slider(
    "Segmentos visuales", min_value=8, max_value=300, value=int(st.session_state['slices']), step=1
)

# Sólo botón RESTABLECER (resetea a valores por defecto)
if st.sidebar.button("🔁 Restablecer valores"):
    # reescribir session_state con DEFAULTS y forzar recarga
    st.session_state['rho0'] = DEFAULTS['rho0']
    st.session_state['alpha'] = DEFAULTS['alpha']
    st.session_state['T0'] = DEFAULTS['T0']
    st.session_state['deltaT'] = DEFAULTS['deltaT']
    st.session_state['R'] = DEFAULTS['R']
    st.session_state['H'] = DEFAULTS['H']
    st.session_state['slices'] = DEFAULTS['slices']
    st.rerun()


R = float(st.session_state['R'])
H = float(st.session_state['H'])
rho0 = float(st.session_state['rho0'])
alpha = float(st.session_state['alpha'])
T0 = float(st.session_state['T0'])
deltaT = float(st.session_state['deltaT'])
slices = int(st.session_state['slices'])
k = deltaT / H if H != 0 else 0.0

def rho_z(z, rho0, alpha, T0, k):
    return rho0 * (1 + alpha * (T0 + k * z))

def compute_mass_analytical(rho0, alpha, T0, k, R, H):
    A = math.pi * R**2
    integral = H + alpha * (T0 * H + k * H**2 / 2)
    return A * rho0 * integral

def compute_centroid_z(rho0, alpha, T0, k, R, H):
    # Formula para el centroide en z: z̄ = (1/M) * ∫ z * ρ(z) dV
    # Para cilindro con sección A: numerator = A * ρ0 * [ H^2/2 + α ( T0 * H^2/2 + k * H^3/3 ) ]
    A = math.pi * R**2
    numerator = (H*2 / 2) + alpha * (T0 * H**2 / 2 + k * H*3 / 3)
    numerator *= A * rho0
    M = compute_mass_analytical(rho0, alpha, T0, k, R, H)
    return numerator / M if M != 0 else 0.0

def compute_volume(R, H):
    return math.pi * R**2 * H


def build_cylinder_mesh(R, H, slices, rho0, alpha, T0, k):
    n_height = max(8, min(slices, 300))
    n_radial = max(16, min(120, int(round(2 * math.pi * R * 24))))
    theta = np.linspace(0, 2 * np.pi, n_radial, endpoint=False)
    z = np.linspace(0, H, n_height)
    TH, Z = np.meshgrid(theta, z)
    X = R * np.cos(TH)
    Y = R * np.sin(TH)
    x = X.ravel(); y = Y.ravel(); zf = Z.ravel()
    rho_vals = rho_z(zf, rho0, alpha, T0, k)
    rho_min = float(np.min(rho_vals)); rho_max = float(np.max(rho_vals))
    if abs(rho_max - rho_min) < 1e-12:
        t_vals = np.full_like(rho_vals, 0.5)
    else:
        t_vals = (rho_vals - rho_min) / (rho_max - rho_min)
    verts = np.arange(n_radial * n_height).reshape((n_height, n_radial))
    i=[]; j=[]; k_idx=[]
    for row in range(n_height - 1):
        for col in range(n_radial - 1):
            v0 = int(verts[row, col]); v1 = int(verts[row, col + 1])
            v2 = int(verts[row + 1, col]); v3 = int(verts[row + 1, col + 1])
            i.append(v0); j.append(v1); k_idx.append(v2)
            i.append(v1); j.append(v3); k_idx.append(v2)
        v0 = int(verts[row, -1]); v1 = int(verts[row, 0]); v2 = int(verts[row + 1, -1]); v3 = int(verts[row + 1, 0])
        i.append(v0); j.append(v1); k_idx.append(v2)
        i.append(v1); j.append(v3); k_idx.append(v2)
    # top & bottom centers
    top_idx = len(x)
    x = np.append(x, 0.0); y = np.append(y, 0.0); zf = np.append(zf, H)
    t_vals = np.append(t_vals, (rho_z(H, rho0, alpha, T0, k) - rho_min) / (rho_max - rho_min + 1e-12))
    bot_idx = len(x)
    x = np.append(x, 0.0); y = np.append(y, 0.0); zf = np.append(zf, 0.0)
    t_vals = np.append(t_vals, (rho_z(0.0, rho0, alpha, T0, k) - rho_min) / (rho_max - rho_min + 1e-12))
    for col in range(n_radial - 1):
        v = int(verts[-1, col]); vnext = int(verts[-1, col + 1])
        i.append(top_idx); j.append(v); k_idx.append(vnext)
    v = int(verts[-1, -1]); vnext = int(verts[-1, 0])
    i.append(top_idx); j.append(v); k_idx.append(vnext)
    for col in range(n_radial - 1):
        v = int(verts[0, col]); vnext = int(verts[0, col + 1])
        i.append(bot_idx); j.append(vnext); k_idx.append(v)
    v = int(verts[0, -1]); vnext = int(verts[0, 0])
    i.append(bot_idx); j.append(vnext); k_idx.append(v)
    return dict(
        x=x, y=y, z=zf, i=np.array(i), j=np.array(j), k=np.array(k_idx),
        intensity=t_vals, rho_min=rho_min, rho_max=rho_max,
        n_radial=n_radial, n_height=n_height
    )


mesh = build_cylinder_mesh(R, H, slices, rho0, alpha, T0, k)
fig = go.Figure()
fig.add_trace(go.Mesh3d(
    x=mesh['x'], y=mesh['y'], z=mesh['z'],
    i=mesh['i'], j=mesh['j'], k=mesh['k'],
    intensity=mesh['intensity'],
    colorscale='Viridis',
    showscale=True,
    flatshading=False,
    name='cilindro'
))
zbar = compute_centroid_z(rho0, alpha, T0, k, R, H)
fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[zbar], mode='markers', marker=dict(size=5, color='red'), name='centroide'))
fig.update_layout(
    scene=dict(aspectmode='auto',
               xaxis=dict(visible=False), yaxis=dict(visible=False),
               zaxis=dict(visible=True, title='z (m)')),
    margin=dict(l=10, r=10, t=30, b=10)
)


M = compute_mass_analytical(rho0, alpha, T0, k, R, H)
V = compute_volume(R, H)
rho_avg = M / V if V != 0 else 0.0

left, right = st.columns([1, 2])
with left:
    st.header("Resultados numéricos")
    st.markdown(f"- Masa total: {M:.6f} kg")
    st.markdown(f"- Volumen: {V:.6f} m³")
    st.markdown(f"- Densidad promedio: {rho_avg:.3f} kg/m³")
    st.markdown(f"- Gradiente k: {k:.4f} °C/m")
    st.markdown(f"- Centroide z̄: {zbar:.4f} m")
    st.markdown("---")
with right:
    # mostrar figura (stretch para ancho)
    st.plotly_chart(fig, width='stretch', theme="streamlit")

