# Angus Dempster, Daniel F Schmidt, Geoffrey I Webb

# MiniRocket: A Very Fast (Almost) Deterministic Transform for Time Series
# Classification

# https://arxiv.org/abs/2012.08791

# ** This is an experimental extension of MiniRocket to variable-length,
#    multivariate input.  It is untested, may contain errors, and may be
#    inefficient in terms of both storage and computation. **

from numba import njit, prange, vectorize
import numpy as np
def _as_TC(x):
    """Devuelve (T, C) dada una entrada (1,C,L) o (C,T) o (T,C)."""
    x = np.asarray(x)
    if x.ndim == 3:  # (1, C, L)
        assert x.shape[0] == 1
        return x[0].T  # -> (L, C)
    if x.ndim == 2:
        return x if x.shape[0] >= x.shape[1] else x.T
    raise ValueError(f"Forma no soportada para _as_TC: {x.shape}")

def _ensure_sigma_in_traces(traces):
    """
    A cada traza k le añade 'sigma' si no existe:
      sigma_k[j] = 1{ conv_sum_k[j] > bias_b_k }
    Requiere que 'conv_sum' y 'bias_b' estén presentes en cada traza k.
    """
    for tr in traces:
        if "sigma" not in tr:
            conv = np.asarray(tr["conv_sum"], dtype=np.float64)  # (d,)
            b    = float(tr["bias_b"])
            tr["sigma"] = (conv > b).astype(np.int8)
    return traces

def compute_sigma_ref_from_x0(x0_raw, transform_prime):
    """
    x0_raw: (1, C, L) o (T, C) o (C, T).
    Devuelve lista de arrays sigma_ref[k] (cada uno de longitud d_k) alineada con las firmas de transform_prime(x0).
    """
    # Asegura batch para transform_prime
    if np.asarray(x0_raw).ndim == 2:      # (T,C) o (C,T)
        x0_batch = _as_TC(x0_raw)[None, ...].transpose(0, 2, 1)  # -> (1, C, T) == (1,C,L)
    else:
        x0_batch = x0_raw  # ya (1,C,L)

    out0 = transform_prime(x0_batch)
    traces0 = _ensure_sigma_in_traces(out0["traces"])
    sigma_ref = [tr0["sigma"] for tr0 in traces0]
    return sigma_ref




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

def _compute_sigma_traces_one(Xi_CL, parameters):
    """
    Xi_CL: (C, L)  (una sola instancia)
    Devuelve: lista de dicts (len = num_features), cada dict tiene:
      - 'sigma'    : ndarray (L,) con {0,1}
      - 'bias_b'   : float
      - 'dilation' : int
      - 'channels' : list[int]
      - 'kernel'   : ndarray (9,) con taps reales de MiniRocket (-1 en todas, +2 en 3 pos)
      - 'conv_sum' : ndarray (L,) suma convolutiva (útil si luego quieres recomputar σ)
    Replica la construcción de C de transform(...), pero sin promediar .mean().
    """
    import numpy as np

    (num_channels_per_combination,
     channel_indices,
     dilations,
     num_features_per_dilation,
     biases) = parameters

    # mismos índices de tripletas que usa transform(...)
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
    ), dtype=np.int32).reshape(84, 3)

    C_count, L = Xi_CL.shape
    num_kernels  = len(indices)
    num_dilations = len(dilations)

    # A y G como en transform(...)
    A = -Xi_CL.astype(np.float32)
    G = (Xi_CL + Xi_CL + Xi_CL).astype(np.float32)

    traces_list = []
    feature_index_start = 0
    combination_index = 0
    num_channels_start = 0

    for dilation_index in range(num_dilations):
        dilation  = int(dilations[dilation_index])
        padding   = ((9 - 1) * dilation) // 2
        num_features_this_dil = int(num_features_per_dilation[dilation_index])

        # Construcciones como en transform(...)
        C_alpha = np.zeros((C_count, L), dtype=np.float32); C_alpha[:] = A
        C_gamma = np.zeros((9, C_count, L), dtype=np.float32); C_gamma[9 // 2] = G

        start = dilation
        end   = L - padding

        for gamma_index in range(9 // 2):
            if end > 0:
                C_alpha[:, -end:] += A[:, :end]
                C_gamma[gamma_index, :, -end:] = G[:, :end]
            end += dilation

        for gamma_index in range(9 // 2 + 1, 9):
            if start < L:
                C_alpha[:, :-start] += A[:, start:]
                C_gamma[gamma_index, :, :-start] = G[:, start:]
            start += dilation

        for kernel_index in range(num_kernels):
            feature_index_end = feature_index_start + num_features_this_dil

            num_chs_this_comb = int(num_channels_per_combination[combination_index])
            num_channels_end = num_channels_start + num_chs_this_comb
            channels_this_comb = channel_indices[num_channels_start:num_channels_end]

            idx0, idx1, idx2 = indices[kernel_index]

            # Serie convolutiva (suma sobre canales seleccionados)
            C_series = (
                C_alpha[channels_this_comb] +
                C_gamma[idx0][channels_this_comb] +
                C_gamma[idx1][channels_this_comb] +
                C_gamma[idx2][channels_this_comb]
            )
            C_series = np.sum(C_series, axis=0).astype(np.float32)  # (L,)

            # Kernel real MiniRocket: -1 en todas, +2 en tres posiciones
            kappa = np.full(9, -1.0, dtype=np.float64)
            kappa[idx0] += 3.0
            kappa[idx1] += 3.0
            kappa[idx2] += 3.0
            # (Esto equivale a +2 en idx0/idx1/idx2 y -1 en el resto)

            # Para cada feature de esta dilatación (misma C_series, distinto bias)
            for fidx in range(feature_index_start, feature_index_end):
                b = float(biases[fidx])
                sigma = (C_series > b).astype(np.int8)
                traces_list.append(dict(
                    sigma=sigma,
                    bias_b=b,
                    dilation=dilation,
                    channels=list(map(int, channels_this_comb)),
                    kernel=kappa,
                    conv_sum=C_series.copy()
                ))

            feature_index_start = feature_index_end
            combination_index   += 1
            num_channels_start  = num_channels_end

    return traces_list


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

# =========================
# A) GLOBALS & HELPERS
# =========================
import numpy as np

MINIROCKET_PARAMETERS = None  # parámetros de MiniRocket tras fit(...)

def fit_minirocket_parameters(X_train, L_train=None, reference_length=None,
                              num_features=10_000, max_dilations_per_kernel=32):
    """
    Ajusta MiniRocket y guarda los parámetros globalmente.
    X_train: (n, C, L)  |  L_train: (n,)
    Usa tu fit(...) original del módulo.
    """
    global MINIROCKET_PARAMETERS
    assert X_train.ndim == 3, f"Esperaba (n,C,L), recibí {X_train.shape}"
    n, C, L = X_train.shape
    if L_train is None:
        L_train = np.full(n, L, dtype=np.int32)

    # Apilar para tu fit original (C, sum(L))
    X_stack = X_train.transpose(1,0,2).reshape(C, -1).astype(np.float32)
    MINIROCKET_PARAMETERS = fit(X_stack, L_train, reference_length,
                                num_features, max_dilations_per_kernel)
    return MINIROCKET_PARAMETERS

def _transform_batch(X, parameters):
    """
    X: (n, C, L) -> (n, F) usando tu transform(...) original.
    """
    assert X.ndim == 3, f"Esperaba (n,C,L), recibí {X.shape}"
    n, C, L = X.shape
    feats = []
    for i in range(n):
        Xi = X[i].astype(np.float32)           # (C, L)
        Li = np.array([L], dtype=np.int32)
        phi_i = transform(Xi, Li, parameters)  # tu transform numba
        feats.append(phi_i[0])                 # (1,F) -> (F,)
    return np.vstack(feats).astype(np.float32)

# =========================
# B) transform_prime: ϕ + traces
# =========================
def transform_prime(X_in, parameters=None):
    """
    X_in: (n,C,L) o (1,C,L)
    return:
      - 'phi'    : (n,F)
      - 'traces' : si n==1 -> LISTA por-feature con 'sigma', 'bias_b', 'dilation', 'channels', 'kernel'
                   si n>1  -> dict de metadatos (como antes) para no penalizar rendimiento en batch
    """
    global MINIROCKET_PARAMETERS
    if parameters is None:
        if MINIROCKET_PARAMETERS is None:
            raise RuntimeError("MINIROCKET_PARAMETERS es None. Llama fit_minirocket_parameters(...) primero.")
        parameters = MINIROCKET_PARAMETERS

    # Normaliza a (n,C,L)
    X = X_in if X_in.ndim == 3 else X_in.reshape(1, *X_in.shape)
    n, C, L = X.shape

    # Características φ con la ruta rápida numba (como ya hacías)
    phi = _transform_batch(X, parameters)

    # Si es una sola instancia, devolvemos trazas “ricas” por feature (lo que necesita propagate_luis)
    if n == 1:
        traces = _compute_sigma_traces_one(X[0].astype(np.float32), parameters)
        return {'phi': phi, 'traces': traces}

    # Si hay batch, mantenemos trazas ligeras como antes (metadatos globales)
    num_channels_per_combination, channel_indices, dilations, num_features_per_dilation, biases = parameters
    traces_meta = dict(
        num_channels_per_combination = num_channels_per_combination,
        channel_indices              = channel_indices,
        dilations                    = dilations,
        num_features_per_dilation    = num_features_per_dilation,
        biases                       = biases,
        input_length                 = int(L),
        num_channels                 = int(C),
    )
    return {'phi': phi, 'traces': traces_meta}


# =========================
# C) Calibración α→Δf y explain wrapper
# =========================
def calibrate_alphas_to_delta_f(alphas, delta_f, phi_x, phi_x0, ridge=1e-8, eps=1e-12):
    s = float(np.sum(alphas))
    if abs(s) > eps:
        return alphas * (delta_f / s)
    dphi = (phi_x - phi_x0).reshape(-1)
    denom = float(np.dot(dphi, dphi) + ridge)
    return (delta_f / denom) * dphi

def _ensure_TC(x):
    x = np.asarray(x)
    if x.ndim == 3:
        assert x.shape[0] == 1, f"Se espera batch=1; recibido {x.shape}"
        x = x[0]
    if x.ndim == 2:
        return x.T if x.shape[0] < x.shape[1] else x
    raise ValueError(f"Forma no soportada: {x.shape}")

def explain(X, X_bar, alphas, transform_func=transform_prime, parameters=None, propagate_luis_fn=None):
    out = transform_func(X, parameters)
    traces = out['traces']
    x_tc  = _ensure_TC(X)
    x0_tc = _ensure_TC(X_bar)
    if propagate_luis_fn is None:
        if 'propagate_luis' not in globals():
            raise RuntimeError("propagate_luis no está definido.")
        propagate_luis_fn = propagate_luis
    beta = propagate_luis_fn(np.asarray(alphas).reshape(-1), traces, x_tc, x0_tc)
    return beta

# =========================
# D) Logits desde φ (para usar LogisticRegression)
# =========================
_CLF_PHI_FOR_LOGITS = None
_CLASS_IDX_FOR_LOGITS = None  # opcional: fija clase objetivo

def set_phi_classifier_for_logits(clf_phi, class_idx=None):
    global _CLF_PHI_FOR_LOGITS, _CLASS_IDX_FOR_LOGITS
    _CLF_PHI_FOR_LOGITS = clf_phi
    _CLASS_IDX_FOR_LOGITS = class_idx

def _to_phi(x_tc):
    x_raw = np.transpose(x_tc, (1,0))[None, ...]  # (1,C,L)
    out = transform_prime(x_raw)
    return out["phi"]

def model_logit(x_tc, eps=1e-9):
    """
    Escoge la clase objetivo y devuelve:
      - binario: log(p/(1-p))
      - multiclase: log p_c
    """
    global _CLF_PHI_FOR_LOGITS, _CLASS_IDX_FOR_LOGITS
    if _CLF_PHI_FOR_LOGITS is None:
        raise RuntimeError("Llama antes a set_phi_classifier_for_logits(clf_phi, class_idx=None).")
    phi_x = _to_phi(x_tc)                           # (1,F)
    proba = _CLF_PHI_FOR_LOGITS.predict_proba(phi_x)[0]
    cls = int(np.argmax(proba)) if _CLASS_IDX_FOR_LOGITS is None else int(_CLASS_IDX_FOR_LOGITS)
    p = proba[cls] if 0 <= cls < len(proba) else proba.max()
    if len(proba) == 2:
        p = float(np.clip(p, eps, 1.0 - eps))
        return float(np.log(p / (1.0 - p)))
    else:
        return float(np.log(float(np.clip(p, eps, 1.0))))  # log-prob

# =========================
# E) Propagación de Luis (base, preserva eficiencia)
# =========================
def propagate_luis(alphas, traces_x, x_tc, x0_tc, *, sigma_ref=None, mode="channel_energy", dt=None):
    """
    Propagación fiel a la fórmula corregida:
      - Usa Δσ_{b,k}^j ∈ {-1,0,1} como compuerta y signo local
      - Pondera por taps de κ y dilatación (ventana efectiva)
      - Normaliza UNA sola vez al final para cerrar Σβ = Σα (= Δf)
      - Soporta Δt opcional (dt); si None, se asume rejilla uniforme (dt=1)

    Devuelve: beta (T, 1)
    """
    x_tc  = _as_TC(x_tc)
    x0_tc = _as_TC(x0_tc)

    # Asegura sigma en trazas de x y obtiene sigma de referencia
    traces_x = _ensure_sigma_in_traces(traces_x)
    if sigma_ref is None:
        sigma_ref = compute_sigma_ref_from_x0(x0_tc, transform_prime)

    T, C = x_tc.shape
    beta = np.zeros((T, 1), dtype=np.float64)

    # Δt (opcional). Si no se da, rejilla uniforme (1)
    if dt is None:
        dt_vec = np.ones(T, dtype=np.float64)
    else:
        dt_vec = np.asarray(dt, dtype=np.float64).reshape(-1)
        if dt_vec.shape[0] != T:
            raise ValueError(f"dt debe tener longitud T={T}, recibido {dt_vec.shape}")

    total_alpha = float(np.sum(alphas))
    if total_alpha == 0.0:
        return beta  # nada que propagar

    for k, tr in enumerate(traces_x):
        alpha_k = float(alphas[k])
        if alpha_k == 0.0:
            continue

        sigma_x  = np.asarray(tr["sigma"], dtype=np.int8)
        sigma_0  = np.asarray(sigma_ref[k], dtype=np.int8)
        delta_sig = sigma_x - sigma_0  # {-1,0,1}
        if not np.any(delta_sig):
            continue  # sin cambio de estado, esta firma no aporta

        kappa = np.asarray(tr["kernel"], dtype=np.float64)  # (q,), taps reales
        delta = int(tr["dilation"])
        q = len(kappa)
        half = q // 2

        # Peso por ventana/taps κ con dilatación (en el eje de tiempo T)
        w = np.zeros(T, dtype=np.float64)
        idx = np.arange(T)
        for m in range(-half, half + 1):
            j = idx + m * delta
            valid = (j >= 0) & (j < T)
            w[valid] += abs(kappa[m + half])
        w[w == 0.0] = 1.0  # evita división por cero

        # Gate Δσ mapeado a T (en tu trazado d_k == T; si no, mapear aquí)
        gate = np.zeros(T, dtype=np.float64)
        dk = min(len(delta_sig), T)
        gate[:dk] = delta_sig[:dk].astype(np.float64)

        mask = (gate != 0.0)
        if not np.any(mask):
            continue

        # Normalización local de pesos sólo en los j activos por Δσ
        w_eff = np.zeros_like(w)
        w_eff[mask] = w[mask]
        Z = w_eff[mask].sum()
        if Z <= 0:
            w_eff[mask] = 1.0
            Z = np.count_nonzero(mask)

        contrib = np.zeros(T, dtype=np.float64)
        # SIGNO LOCAL = sign(Δσ_j). Esto es la corrección clave.
        contrib[mask] = alpha_k * np.sign(gate[mask]) * (w_eff[mask] / Z)

        beta[:, 0] += contrib

    # Factor Δt (si uniforme, es 1 y no cambia)
    beta[:, 0] *= dt_vec

    # Cierre global: Σβ = Σα (= Δf)
    S = float(beta.sum())
    if S != 0.0:
        beta *= (total_alpha / S)

    return beta






