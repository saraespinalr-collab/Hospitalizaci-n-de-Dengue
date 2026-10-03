# Riesgo de hospitalización por dengue – Medellín

Proyecto integrador del curso Aprendizaje de Máquina (UPB), desarrollado con la metodología **CRISP-DM**.
La aplicación estima la probabilidad de que un caso de dengue requiera hospitalización a partir de la
información disponible en la consulta: edad, sexo, comuna, régimen de salud, días con síntomas,
síntomas generales y signos de alarma.

## Datos
- **Fuente:** SIVIGILA – Dengue, Medellín (Medata / Datos Abiertos Medellín).
- **Periodo:** 2015–2021 (antes de 2015 la ficha no registraba los signos de alarma).
- **Registros tras la limpieza:** 27.069 (70 % entrenamiento, 30 % prueba, partición estratificada).
- **Variable objetivo:** `pac_hos_` → hospitalizado (1 = Sí, 0 = No). ≈ 25 % de los casos fueron hospitalizados.
- Se excluyeron `clas_dengue` y `tip_cas_` para evitar fuga de información.

## Modelo
- **Algoritmo:** Gradient Boosting (`HistGradientBoostingClassifier`) con `class_weight='balanced'`.
- **Hiperparámetros:** learning_rate = 0,092 · max_iter = 500 · max_leaf_nodes = 15 · min_samples_leaf = 20 · l2_regularization = 10.
- **Umbral de decisión:** 0,565 (ajustado con validación cruzada para maximizar F1).
- **Pipeline serializado:** conteo de signos/síntomas → imputación → escalamiento y one-hot → modelo → umbral.

## Desempeño en el conjunto de prueba (30 %)
| Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| 0,823 | 0,623 | 0,744 | 0,678 | 0,864 |

## Archivos
| Archivo | Contenido |
|---|---|
| `app.py` | Aplicación web en Streamlit |
| `modelo_dengue.pkl` | Pipeline completo entrenado |
| `requirements.txt` | Librerías necesarias |
| `README.md` | Este documento |

## Ejecutar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```

> Herramienta de apoyo analítico. No es un diagnóstico y no reemplaza el criterio médico.
