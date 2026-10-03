import sys
import types

import joblib
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Función usada dentro del pipeline entrenado (paso 'conteos').
# El modelo se entrenó importándola desde un módulo llamado 'utilidades';
# se registra aquí con ese mismo nombre para que el .pkl cargue sin archivos extra.
# ---------------------------------------------------------------------------
SINTOMAS = ['cefalea', 'dolrretroo', 'malgias', 'artralgia', 'erupcionr', 'dolor_abdo', 'vomito']
ALARMA = ['somnolenci', 'hipotensio', 'hepatomeg', 'hem_mucosa', 'hipotermia',
          'aum_hemato', 'caida_plaq', 'acum_liquievento']


def agregar_conteos(X):
    '''Agrega el número de signos de alarma y de síntomas generales presentes.'''
    X = X.copy()
    X['n_alarma'] = X[ALARMA].fillna(0).sum(axis=1)
    X['n_sintomas'] = X[SINTOMAS].fillna(0).sum(axis=1)
    return X


_mod = types.ModuleType('utilidades')
_mod.agregar_conteos, _mod.SINTOMAS, _mod.ALARMA = agregar_conteos, SINTOMAS, ALARMA
sys.modules.setdefault('utilidades', _mod)
sys.modules['__main__'].agregar_conteos = agregar_conteos   # por si el modelo se guardó desde __main__

st.set_page_config(page_title='Riesgo de hospitalización por dengue', page_icon='🦟', layout='centered')


@st.cache_resource
def cargar_modelo():
    modelo = joblib.load('modelo_dengue.pkl')
    umbral = float(getattr(modelo, 'best_threshold_', 0.5))
    pipe = getattr(modelo, 'estimator_', modelo)
    comunas = list(pipe.named_steps['prep'].named_transformers_['cat']
                   .named_steps['onehot'].categories_[0])
    return modelo, umbral, comunas


modelo, UMBRAL, COMUNAS = cargar_modelo()

st.title('Riesgo de hospitalización por dengue')
st.caption('Medellín · Gradient Boosting entrenado con casos SIVIGILA 2015–2021')
st.info('Herramienta de apoyo analítico. No es un diagnóstico y no reemplaza el criterio médico.')

REGIMEN = {'Contributivo': 'C', 'Subsidiado': 'S', 'Especial': 'E', 'Excepción': 'P',
           'No asegurado': 'N', 'Indeterminado': 'I'}
NOMBRES_S = {'cefalea': 'Cefalea', 'dolrretroo': 'Dolor retroocular', 'malgias': 'Mialgias',
             'artralgia': 'Artralgias', 'erupcionr': 'Erupción cutánea', 'dolor_abdo': 'Dolor abdominal',
             'vomito': 'Vómito'}
NOMBRES_A = {'somnolenci': 'Somnolencia', 'hipotensio': 'Hipotensión', 'hepatomeg': 'Hepatomegalia',
             'hem_mucosa': 'Hemorragia en mucosas', 'hipotermia': 'Hipotermia',
             'aum_hemato': 'Aumento del hematocrito', 'caida_plaq': 'Caída de plaquetas',
             'acum_liquievento': 'Acumulación de líquidos'}

with st.form('paciente'):
    st.subheader('Datos del paciente')
    c1, c2 = st.columns(2)
    edad = c1.number_input('Edad (años)', 0.0, 100.0, 30.0, step=1.0)
    sexo = c2.radio('Sexo', ['Femenino', 'Masculino'], horizontal=True)
    dias = c1.number_input('Días desde el inicio de síntomas', 0, 30, 3)
    semana = c2.number_input('Semana epidemiológica', 1, 53, 20)
    comuna = c1.selectbox('Comuna o corregimiento', COMUNAS,
                          index=COMUNAS.index('Belen') if 'Belen' in COMUNAS else 0)
    regimen = c2.selectbox('Régimen de salud', list(REGIMEN))
    despl = st.checkbox('Se desplazó a otro municipio en los últimos 15 días')

    st.subheader('Síntomas')
    cols = st.columns(3)
    sintomas = {k: cols[i % 3].checkbox(v) for i, (k, v) in enumerate(NOMBRES_S.items())}
    st.subheader('Signos de alarma')
    cols = st.columns(3)
    alarma = {k: cols[i % 3].checkbox(v) for i, (k, v) in enumerate(NOMBRES_A.items())}
    enviar = st.form_submit_button('Calcular riesgo', type='primary')

if enviar:
    fila = {'edad_anios': edad, 'semana': semana, 'dias_sintomas': dias,
            'sexo_': 1 if sexo == 'Masculino' else 0, 'desplazami': int(despl),
            **{k: int(v) for k, v in sintomas.items()}, **{k: int(v) for k, v in alarma.items()},
            'comuna': comuna, 'tipo_ss_': REGIMEN[regimen]}
    prob = float(modelo.predict_proba(pd.DataFrame([fila]))[0, 1])
    st.metric('Probabilidad estimada de hospitalización', f'{prob:.0%}')
    st.progress(min(prob, 1.0))
    if prob >= UMBRAL:
        st.error(f'Riesgo ALTO: el modelo sugiere hospitalización (umbral de decisión = {UMBRAL:.0%}).')
    else:
        st.success(f'Riesgo BAJO: el modelo no sugiere hospitalización (umbral de decisión = {UMBRAL:.0%}).')
    st.caption(f'Signos de alarma marcados: {sum(alarma.values())} de 8 · Días con síntomas: {dias}')
