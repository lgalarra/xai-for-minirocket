# Angus Dempster, Daniel F Schmidt, Geoffrey I Webb

# MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series
# Classification

# https://arxiv.org/abs/2012.08791

# ** This is an experimental extension of MiniRocket to variable-length,
#    multivariate input.  It is untested, may contain errors, and may be
#    inefficient in terms of both storage and computation. **

from numba import njit, prange, vectorize
import numpy as np

@njit("float32[:](float32[:,:],int32[:],int32[:],int32[:],int32[:],int32[:],float32[:])", fastmath = True, parallel = False, cache = True)
def _fit_biases(X, L, num_channels_per_combination, channel_indices, dilations, num_features_per_dilation, quantiles):

    num_examples = len(L)

    num_channels, _ = X.shape

    # equivalent to:
    # >>> from itertools import combinations
    # >>> indices = np.array([_ for _ in combinations(np.arange(9), 3)], dtype = np.int32)
    indices = np.array((
        0,1,2,0,1,3,0,1,4,0,1,5,0,1,6,0,1,7,0,1,8,
        0,2,3,0,2,4,0,2,5,0,2,6,0,2,7,0,2,8,0,3,4,
        0,3,5,0,3,6,0,3,7,0,3,8,0,4,5,0,4,6,0,4,7,
        0,4,8,0,5,6,0,5,7,0,5,8,0,6,7,0,6,8,0,7,8,
        1,2,3,1,2,4,1,2,5,1,2,6,1,2,7,1,2,8,1,3,4,
        1,3,5,1,3,6,1,3,7,1,3,8,1,4,5,1,4,6,1,4,7,
        1,4,8,1,5,6,1,5,7,1,5,8,1,6,7,1,6,8,1,7,8,
        2,3,4,2,3,5,2,3,6,2,3,7,2,3,8,2,4,5,2,4,6,
        2,4,7,2,4,8,2,5,6,2,5,7,2,5,8,2,6,7,2,6,8,
        2,7,8,3,4,5,3,4,6,3,4,7,3,4,8,3,5,6,3,5,7,
        3,5,8,3,6,7,3,6,8,3,7,8,4,5,6,4,5,7,4,5,8,
        4,6,7,4,6,8,4,7,8,5,6,7,5,6,8,5,7,8,6,7,8
    ), dtype = np.int32).reshape(84, 3)

    num_kernels = len(indices)
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)

    biases = np.zeros(num_features, dtype = np.float32)

    feature_index_start = 0

    combination_index = 0
    num_channels_start = 0

    for dilation_index in range(num_dilations):

        dilation = dilations[dilation_index]
        padding = ((9 - 1) * dilation) // 2

        num_features_this_dilation = num_features_per_dilation[dilation_index]

        for kernel_index in range(num_kernels):

            feature_index_end = feature_index_start + num_features_this_dilation

            num_channels_this_combination = num_channels_per_combination[combination_index]

            num_channels_end = num_channels_start + num_channels_this_combination

            channels_this_combination = channel_indices[num_channels_start:num_channels_end]

            example_index = np.random.randint(num_examples)

            input_length = np.int64(L[example_index])

            b = np.sum(L[0:example_index + 1])
            a = b - input_length

            _X = X[channels_this_combination, a:b]

            A = -_X          # A = alpha * X = -X
            G = _X + _X + _X # G = gamma * X = 3X

            C_alpha = np.zeros((num_channels_this_combination, input_length), dtype = np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, num_channels_this_combination, input_length), dtype = np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):

                # thanks to Murtaza Jafferji @murtazajafferji for suggesting this fix
                if end > 0:

                    C_alpha[:, -end:] = C_alpha[:, -end:] + A[:, :end]
                    C_gamma[gamma_index, :, -end:] = G[:, :end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):

                if start < input_length:

                    C_alpha[:, :-start] = C_alpha[:, :-start] + A[:, start:]
                    C_gamma[gamma_index, :, :-start] = G[:, start:]

                start += dilation

            index_0, index_1, index_2 = indices[kernel_index]

            C = C_alpha + C_gamma[index_0] + C_gamma[index_1] + C_gamma[index_2]
            C = np.sum(C, axis = 0)

            biases[feature_index_start:feature_index_end] = np.quantile(C, quantiles[feature_index_start:feature_index_end])

            feature_index_start = feature_index_end

            combination_index += 1
            num_channels_start = num_channels_end

    return biases

def _fit_dilations(reference_length, num_features, max_dilations_per_kernel):

    num_kernels = 84

    num_features_per_kernel = num_features // num_kernels
    true_max_dilations_per_kernel = min(num_features_per_kernel, max_dilations_per_kernel)
    multiplier = num_features_per_kernel / true_max_dilations_per_kernel

    max_exponent = np.log2((reference_length - 1) / (9 - 1))
    dilations, num_features_per_dilation = \
    np.unique(np.logspace(0, max_exponent, true_max_dilations_per_kernel, base = 2).astype(np.int32), return_counts = True)
    num_features_per_dilation = (num_features_per_dilation * multiplier).astype(np.int32) # this is a vector

    remainder = num_features_per_kernel - np.sum(num_features_per_dilation)
    i = 0
    while remainder > 0:
        num_features_per_dilation[i] += 1
        remainder -= 1
        i = (i + 1) % len(num_features_per_dilation)

    return dilations, num_features_per_dilation

# low-discrepancy sequence to assign quantiles to kernel/dilation combinations
def _quantiles(n):
    return np.array([(_ * ((np.sqrt(5) + 1) / 2)) % 1 for _ in range(1, n + 1)], dtype = np.float32)

def fit(X, L, reference_length = None, num_features = 10_000, max_dilations_per_kernel = 32):

    # note in relation to dilation:
    # * change *reference_length* according to what is appropriate for your
    #   application, e.g., L.max(), L.mean(), np.median(L)
    # * use fit(...) with an appropriate subset of time series, e.g., for
    #   reference_length = L.mean(), call fit(...) using only time series of at
    #   least length L.mean() [see filter_by_length(...)]
    if reference_length == None:
        reference_length = L.max()

    num_channels, _ = X.shape

    num_kernels = 84

    dilations, num_features_per_dilation = _fit_dilations(reference_length, num_features, max_dilations_per_kernel)

    num_features_per_kernel = np.sum(num_features_per_dilation)

    quantiles = _quantiles(num_kernels * num_features_per_kernel)

    num_dilations = len(dilations)
    num_combinations = num_kernels * num_dilations

    max_num_channels = min(num_channels, 9)
    max_exponent = np.log2(max_num_channels + 1)

    num_channels_per_combination = (2 ** np.random.uniform(0, max_exponent, num_combinations)).astype(np.int32)

    channel_indices = np.zeros(num_channels_per_combination.sum(), dtype = np.int32)

    num_channels_start = 0
    for combination_index in range(num_combinations):
        num_channels_this_combination = num_channels_per_combination[combination_index]
        num_channels_end = num_channels_start + num_channels_this_combination
        channel_indices[num_channels_start:num_channels_end] = np.random.choice(num_channels, num_channels_this_combination, replace = False)

        num_channels_start = num_channels_end

    biases = _fit_biases(X, L, num_channels_per_combination, channel_indices, dilations, num_features_per_dilation, quantiles)

    return num_channels_per_combination, channel_indices, dilations, num_features_per_dilation, biases

# _PPV(C, b).mean() returns PPV for vector C (convolution output) and scalar b (bias)
@vectorize("float32(float32,float32)", nopython = True, cache = True)
def _PPV(a, b):
    if a > b:
        return 1
    else:
        return 0

@njit("float32[:,:](float32[:,:],int32[:],Tuple((int32[:],int32[:],int32[:],int32[:],float32[:])))", fastmath = True, parallel = True, cache = True)
def transform(X, L, parameters):

    num_examples = len(L)

    num_channels, _ = X.shape

    num_channels_per_combination, channel_indices, dilations, num_features_per_dilation, biases = parameters

    # equivalent to:
    # >>> from itertools import combinations
    # >>> indices = np.array([_ for _ in combinations(np.arange(9), 3)], dtype = np.int32)
    indices = np.array((
        0,1,2,0,1,3,0,1,4,0,1,5,0,1,6,0,1,7,0,1,8,
        0,2,3,0,2,4,0,2,5,0,2,6,0,2,7,0,2,8,0,3,4,
        0,3,5,0,3,6,0,3,7,0,3,8,0,4,5,0,4,6,0,4,7,
        0,4,8,0,5,6,0,5,7,0,5,8,0,6,7,0,6,8,0,7,8,
        1,2,3,1,2,4,1,2,5,1,2,6,1,2,7,1,2,8,1,3,4,
        1,3,5,1,3,6,1,3,7,1,3,8,1,4,5,1,4,6,1,4,7,
        1,4,8,1,5,6,1,5,7,1,5,8,1,6,7,1,6,8,1,7,8,
        2,3,4,2,3,5,2,3,6,2,3,7,2,3,8,2,4,5,2,4,6,
        2,4,7,2,4,8,2,5,6,2,5,7,2,5,8,2,6,7,2,6,8,
        2,7,8,3,4,5,3,4,6,3,4,7,3,4,8,3,5,6,3,5,7,
        3,5,8,3,6,7,3,6,8,3,7,8,4,5,6,4,5,7,4,5,8,
        4,6,7,4,6,8,4,7,8,5,6,7,5,6,8,5,7,8,6,7,8
    ), dtype = np.int32).reshape(84, 3)

    num_kernels = len(indices)
    num_dilations = len(dilations)

    num_features = num_kernels * np.sum(num_features_per_dilation)

    features = np.zeros((num_examples, num_features), dtype = np.float32)

    for example_index in prange(num_examples):

        input_length = np.int64(L[example_index])

        b = np.sum(L[0:example_index + 1])
        a = b - input_length

        _X = X[:, a:b]

        A = -_X          # A = alpha * X = -X
        G = _X + _X + _X # G = gamma * X = 3X

        feature_index_start = 0

        combination_index = 0
        num_channels_start = 0

        for dilation_index in range(num_dilations):

            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2

            num_features_this_dilation = num_features_per_dilation[dilation_index]

            C_alpha = np.zeros((num_channels, input_length), dtype = np.float32)
            C_alpha[:] = A

            C_gamma = np.zeros((9, num_channels, input_length), dtype = np.float32)
            C_gamma[9 // 2] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(9 // 2):

                # thanks to Murtaza Jafferji @murtazajafferji for suggesting this fix
                if end > 0:

                    C_alpha[:, -end:] = C_alpha[:, -end:] + A[:, :end]
                    C_gamma[gamma_index, :, -end:] = G[:, :end]

                end += dilation

            for gamma_index in range(9 // 2 + 1, 9):

                if start < input_length:

                    C_alpha[:, :-start] = C_alpha[:, :-start] + A[:, start:]
                    C_gamma[gamma_index, :, :-start] = G[:, start:]

                start += dilation

            for kernel_index in range(num_kernels):

                feature_index_end = feature_index_start + num_features_this_dilation

                num_channels_this_combination = num_channels_per_combination[combination_index]

                num_channels_end = num_channels_start + num_channels_this_combination

                channels_this_combination = channel_indices[num_channels_start:num_channels_end]

                index_0, index_1, index_2 = indices[kernel_index]

                C = C_alpha[channels_this_combination] + \
                    C_gamma[index_0][channels_this_combination] + \
                    C_gamma[index_1][channels_this_combination] + \
                    C_gamma[index_2][channels_this_combination]
                C = np.sum(C, axis = 0)

                for feature_count in range(num_features_this_dilation):
                    features[example_index, feature_index_start + feature_count] = _PPV(C, biases[feature_index_start + feature_count]).mean()

                feature_index_start = feature_index_end

                combination_index += 1
                num_channels_start = num_channels_end

    return features
import numpy as np


# Nuevas funciones:

def minirocket_explain(
    X_timeseries_to_explain,           # shape: (1, C, L)
    classifier_phi,                    # modelo entrenado sobre características φ(x)
    background_X,                      # shape: (N, C, L)
    transform_function,                # como transform_prime
    parameters,                        # parámetros de MiniRocket (kernels, biases, etc.)
    base_explanation_method='shap',   # 'shap' o 'stratoshap'
    baseline_policy='shap',           # 'shap', 'baricenter_opposite_class', etc.
    k=1                                # presupuesto para StratoSHAP
):
    # 1. Transformar entrada real y background
    phi_x, traces = transform_function(X_timeseries_to_explain, X_timeseries_to_explain, parameters)
    phi_background, _ = transform_function(background_X, background_X, parameters)

    # 2. Calcular alphas
    if base_explanation_method.lower() == 'shap':
        import shap
        explainer = shap.Explainer(classifier_phi, phi_background)
        shap_values = explainer(phi_x)
        class_index = classifier_phi.predict_proba(phi_x)[0].argmax()
        alphas = shap_values.values[0][:, class_index]  # (500,)
    
    elif base_explanation_method.lower() in ['stratoshap', 'st-shap']:
        from stratoshap import SHAPStratum  # Asegúrate de tenerlo
        class_index = classifier_phi.predict_proba(phi_x)[0].argmax()

        # Aplanar φ(x) y φ(background)
        x_flat = phi_x[0]
        x_bar = phi_background.mean(axis=0)
        
        class Game:
            def __init__(self, model, x, x_baseline):
                self.model = model
                self.x = x
                self.baseline = x_baseline
                self.feature_count = len(x)

            def compute_value(self, coalition):
                mask = np.zeros_like(self.x)
                mask[coalition] = 1
                x_masked = self.baseline * (1 - mask) + self.x * mask
                return self.model(x_masked.reshape(1, -1))[0]
        
        game = Game(classifier_phi.predict_proba, x_flat, x_bar)
        shap_stratum = SHAPStratum()
        shap_stratum.game = game
        shap_stratum.n = len(x_flat)
        shap_stratum.budget = k
        shap_stratum.idx_dims = class_index
        shap_stratum.dim = 1

        shap_values, _ = shap_stratum.approximate_shapley_values()
        alphas = shap_values[0]  # (500,)
    
    else:
        raise ValueError(f"Unsupported explanation method: {base_explanation_method}")

    # 3. Calcular X_bar de acuerdo con la baseline_policy
    if baseline_policy == 'shap':
        X_bar = background_X.mean(axis=0, keepdims=True)

    elif baseline_policy == 'baricenter_opposite_class':
        # Asumimos que tienes y_train o etiquetas del background
        raise NotImplementedError("Necesitas proporcionar y_train para usar esta política")

    elif baseline_policy == 'centroid':
        X_bar = np.median(background_X, axis=0, keepdims=True)

    else:
        raise ValueError(f"Unsupported baseline policy: {baseline_policy}")

    # 4. Aplicar fórmula de Luis (explain) con trazas y alphas
    contribuciones = explain(X_timeseries_to_explain, X_bar, alphas, 
                             lambda X, X_bar: transform_function(X, X_bar, parameters))

    return contribuciones, classifier_phi.predict(phi_x)

def shap_explain_on_minirocket_classifier(
    clf, phi_X, phi_X_background,
    class_of_interest=None,
    base_explanation_method='shap',
    budget=1  # solo aplica a StratoSHAP
):
    """
    Explicación de φ(X) mediante SHAP o StratoSHAP para un clasificador entrenado en φ.

    Parámetros:
    - clf: clasificador entrenado en φ(X)
    - phi_X: ndarray (1, num_features) → instancia a explicar
    - phi_X_background: ndarray (n_background, num_features)
    - class_of_interest: int (índice de clase a explicar)
    - base_explanation_method: 'shap' o 'strato'
    - budget: int (solo para StratoSHAP, por defecto k=1)

    Retorna:
    - alphas: ndarray de forma (num_features,)
    """
    if base_explanation_method == 'shap':
        explainer = shap.Explainer(clf, phi_X_background)
        shap_values = explainer(phi_X)  # (1, num_features, num_classes)
        class_idx = class_of_interest if class_of_interest is not None else clf.predict_proba(phi_X)[0].argmax()
        alphas = shap_values.values[0][:, class_idx]  # (num_features,)

    elif base_explanation_method == 'strato':
        x_flat = phi_X.flatten()
        x_bar_flat = phi_X_background.mean(axis=0).flatten()

        class_idx = class_of_interest if class_of_interest is not None else clf.predict_proba(phi_X)[0].argmax()

        def predict_phi_flat(X_flat_input):
            return clf.predict_proba(X_flat_input.reshape(1, -1))

        game = TimeSeriesGame(predict_phi_flat, x_flat, x_bar_flat)
        shap_stratum = SHAPStratum()
        shap_stratum.game = game
        shap_stratum.n = len(x_flat)
        shap_stratum.budget = budget
        shap_stratum.idx_dims = class_idx
        shap_stratum.dim = 1
        shap_values, _ = shap_stratum.approximate_shapley_values()
        alphas = shap_values[0]

    else:
        raise ValueError("base_explanation_method debe ser 'shap' o 'strato'.")

    return alphas

def shap_explain_on_timeseries(
    X_to_explain,                   # (1, C, L)
    background_X,                   # (n, C, L)
    clf,                            # Clasificador entrenado en φ
    transform_function,             # MiniRocket modificado (e.g., transform_prime)
    X_bar_policy='shap',            # 'shap', 'baricenter_opposite_class', etc.
    base_explanation_method='shap', # 'shap' o 'strato'
    class_of_interest=None,
    budget=1                        # Presupuesto k (solo aplica a StratoSHAP)
):
    """
    Aplica MiniRocket + SHAP o StratoSHAP desde la serie original,
    y obtiene contribuciones β_j punto a punto.
    """
    # Paso 1: Obtener φ(X) y φ(background)
    phi_X, traces = transform_function(X_to_explain)
    phi_background, _ = transform_function(background_X)

    # Paso 2: Calcular la referencia φ(X̄)
    if X_bar_policy == 'shap' or X_bar_policy == 'centroid':
        X_bar = np.mean(background_X, axis=0, keepdims=True)  # (1, C, L)
        phi_X_bar = np.mean(phi_background, axis=0, keepdims=True)

    elif X_bar_policy == 'baricenter_opposite_class':
        preds = clf.predict(transform_function(background_X)[0])
        pred_class = clf.predict(phi_X)[0]
        opposite_class = 1 - pred_class  # Solo si es binario
        mask = preds == opposite_class
        X_bar = np.mean(background_X[mask], axis=0, keepdims=True)
        phi_X_bar = np.mean(phi_background[mask], axis=0, keepdims=True)

    else:
        raise ValueError("Política de referencia no implementada")

    # Paso 3: Calcular alphas en φ(X)
    alphas = shap_explain_on_minirocket_classifier(
        clf=clf,
        phi_X=phi_X,
        phi_X_background=phi_background,
        class_of_interest=class_of_interest,
        base_explanation_method=base_explanation_method,
        budget=budget
    )

    # Paso 4: Aplicar fórmula de Luis con trazas
    beta_j = explain(X_to_explain, X_bar, alphas, transform_function)

    return beta_j



