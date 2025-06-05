Funciones Principales: 

transform_prime(X, X_bar)

1- Transforma un conjunto de series de tiempo usando MiniRocket y extrae trazas para cada característica generada:

* φ_k(x): proporción de ventanas activadas por cada kernel.

* Trazas: información detallada de activaciones, canales, pesos, ventanas y dilataciones.


2- explain(X, X_bar, alphas, transform_function)

Calcula las contribuciones β_j punto a punto, aplicando la fórmula de Luis:

- Utiliza las trazas y las diferencias Δφ_k.

- Redistribuye la importancia de cada φ_k(x) en base a su impacto convolucional.


3. Funciones Encapsuladas

minirocket_explain(...)

Realiza el flujo completo:

1. Transforma las entradas y el background con transform_prime.

2. Aplica SHAP o StratoSHAP para obtener α_k.

3. Calcula el baricentro de referencia según una política.

4. Aplica explain para calcular β_j.


Parámetros importantes:

*baseline_policy: define el tipo de referencia ("shap", "baricenter_opposite_class", etc.)

*base_explanation_method: "shap" o "strato"


shap_explain_on_minirocket_classifier(...)

Extrae directamente los α_k desde un modelo entrenado en φ(x), usando SHAP o StratoSHAP.


shap_explain_on_timeseries(...)

Aplica explicabilidad punto a punto directamente sobre la serie temporal original, sin necesidad de usar φ(x), usando StratoSHAP con presupuesto k=1.


4. Flujo Recomendado de Uso

Transformar dataset de entrenamiento y prueba con transform_prime.

Entrenar clasificador con phi_X_train.

Usar shap_explain_on_minirocket_classifier o minirocket_explain para obtener α_k.

Calcular β_j con explain.

Comparar contra StratoSHAP aplicado directamente en entrada (función shap_explain_on_timeseries).

Evaluar similitud de atribuciones con r2_score().


5. Recomendaciones

Usar presupuesto k=1 para StratoSHAP.

Validar que las sumas de β_j se aproximen a α^T Δφ(x).

Verificar consistencia usando métricas como r2_score() entre métodos.