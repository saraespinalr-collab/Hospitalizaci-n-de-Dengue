import json
import joblib
import pandas as pd
import streamlit as st
from utilidades import SINTOMAS, ALARMA, agregar_conteos  # noqa: F401 (necesario para cargar el pipeline)

st.set_page_config(page_title='Riesgo de hospitalización por dengue', page_icon='🦟', layout='centered')


@st.cache_resource
def cargar():
    return joblib.load('modelo_dengue.joblib'), json.load(open('metadata.json', encoding='utf-8'))


modelo, meta = cargar()

st.title('Riesgo de hospitalización por dengue')
st.caption('Medellín · Modelo entrenado con casos SIVIGILA 2015–2021 · '
           f"{meta['modelo']} (F1 = {meta['f1_test']:.2f}, recall = {meta['recall_test']:.2f}, ROC-AUC = {meta['roc_auc_test']:.2f})")
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
    comuna = c1.selectbox('Comuna o corregimiento', meta['comunas'],
                          index=meta['comunas'].index('Belen') if 'Belen' in meta['comunas'] else 0)
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
    X = pd.DataFrame([fila])
    prob = float(modelo.predict_proba(X)[0, 1])
    hosp = prob >= meta['umbral']
    st.metric('Probabilidad estimada de hospitalización', f'{prob:.0%}')
    st.progress(min(prob, 1.0))
    if hosp:
        st.error(f"Riesgo ALTO: el modelo sugiere hospitalización (umbral de decisión = {meta['umbral']:.0%}).")
    else:
        st.success(f"Riesgo BAJO: el modelo no sugiere hospitalización (umbral de decisión = {meta['umbral']:.0%}).")
    n_al = sum(alarma.values())
    st.caption(f'Signos de alarma marcados: {n_al} de 8 · Días con síntomas: {dias}')
