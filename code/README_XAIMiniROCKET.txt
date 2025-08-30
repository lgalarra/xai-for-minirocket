XAI4MiniROCKET — README (para revisión de código)
=================================================

Este repositorio contiene:
- Una extensión de MiniROCKET para series multivariadas y longitudes variables, con extracción de trazas (convoluciones, PPV, umbrales) para poder redistribuir atribuciones al dominio temporal.
- Un notebook de demostración con el pipeline completo de entrenamiento y explicación.

Archivos principales:
- minirocket_multivariate_variable.py  ← implementación y API de uso
- XAIMiniROCKET.ipynb  ← ejemplo de punta a punta

-------------------------------------------------
1) Requisitos
-------------------------------------------------
python >= 3.9
pip install numpy numba scikit-learn shap
# opcional (para StratoSHAP):
# pip install stratoshap

Notas:
- Numba compila en el primer uso; el primer llamado a algunas funciones puede tardar más.
- Si no usará StratoSHAP, puede ignorar esa dependencia (use base_explanation_method='shap').

-------------------------------------------------
2) Formatos de datos
-------------------------------------------------
- Entradas de series: X con forma (n, C, L)
  - n: número de instancias  
  - C: canales  
  - L: longitud (para variable-length se maneja internamente con L_i por instancia)
- Longitudes: L_train (opcional) vector de enteros con longitudes por instancia.
- MiniROCKET produce características φ(X) de tamaño (n, F).

-------------------------------------------------
3) Flujo mínimo (entrenar + explicar)
-------------------------------------------------
import numpy as np
from sklearn.linear_model import LogisticRegression
from minirocket_multivariate_variable import (
    fit_minirocket_parameters, _transform_batch,
    shap_explain_on_timeseries
)

# 1) Datos de ejemplo (n, C, L)
X_train = np.random.randn(100, 3, 500).astype(np.float32)
y_train = np.random.randint(0, 2, size=100)

X_test  = np.random.randn(10, 3, 500).astype(np.float32)

# 2) Ajustar parámetros de MiniROCKET
params = fit_minirocket_parameters(X_train)

# 3) Transformar a espacio φ
phi_train = _transform_batch(X_train, params)

# 4) Clasificador lineal sobre φ
clf = LogisticRegression(max_iter=1000)
clf.fit(phi_train, y_train)

# 5) Explicar una instancia (β_j por timestamp/canal)
x0 = X_test[:1]
background = X_train[:50]

beta = shap_explain_on_timeseries(
    X_to_explain=x0,
    background_X=background,
    clf=clf,
    transform_function=lambda X: _transform_batch(X, params)
)

-------------------------------------------------
4) Funciones clave
-------------------------------------------------
A. Ajuste y transformación
- fit_minirocket_parameters(X_train, ...): ajusta parámetros de MiniROCKET.
- _transform_batch(X, parameters): transforma X a φ(X).

B. Explicaciones (nivel feature → tiempo)
- shap_explain_on_timeseries(...): redistribuye atribuciones de φ(X) al tiempo/canales.
- shap_explain_on_minirocket_classifier(...): atribuciones en el espacio φ(X).
- minirocket_explain(...): versión avanzada con trazas y políticas de baseline.

-------------------------------------------------
5) Políticas de referencia (baseline)
-------------------------------------------------
- 'shap': promedio del background.
- 'centroid': centroide (media/mediana).
- 'baricenter_opposite_class': requiere labels (pendiente).

-------------------------------------------------
6) Buenas prácticas y tips
-------------------------------------------------
- Fijar semilla np.random.seed(...) para reproducibilidad.
- El primer llamado a Numba compila (latencia inicial).
- Usar un background representativo.
- Normalizar β por canal al graficar.

-------------------------------------------------
7) Notebook
-------------------------------------------------
El notebook XAIMiniROCKET.ipynb muestra:
- Entrenamiento MiniROCKET → φ(X).
- Clasificador (Ridge/Logistic).
- Cálculo de α (SHAP) y β (redistribución).
- Métricas de fidelidad, MSE vs baseline, estabilidad.
- Visualización de saliency maps.

-------------------------------------------------
8) Preguntas para probar
-------------------------------------------------
- ¿Qué pasa si cambio num_features o max_dilations_per_kernel?
- ¿Qué tan sensible es β al background?
- ¿Qué clase se explica?
- ¿StratoSHAP mejora tiempos?

-------------------------------------------------
9) Limitaciones
-------------------------------------------------
- La versión multivar/variable-length es experimental.
- Política 'baricenter_opposite_class' aún no implementada.
- Primera ejecución compila Numba.
