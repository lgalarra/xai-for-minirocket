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
# Vamos a preparar una versión completamente modificada de minirocket_transform_with_traces
# que guarda toda la información relevante en la traza para aplicabilidad directa de la fórmula de Luis.
# Mantendremos los comentarios originales del usuario y añadiremos los nuevos con "# NUEVO: ..."


def transform_prime(X, parameters):
    """
    MiniRocket transform (multivariate) with trace information for each feature φ_i(x),
    adapted for integration with attribution methods like the one proposed by Luis.

    Parameters:
    - X: ndarray of shape (num_examples, num_channels, input_length)
    - X_bar: ndarray of shape (num_examples, num_channels, input_length), reference input
    - parameters: tuple containing:
        * num_channels_per_combination
        * channel_indices
        * dilations
        * num_features_per_dilation
        * biases

    Returns:
    - features: ndarray of shape (num_examples, num_features)
    - all_traces: list of trace dictionaries for each feature φ_i
    """
    #Preparación de entradas
    #num_channels_per_combination: cuántos canales se usan por kernel (usualmente 3)
    #channel_indices: índices concatenados de los canales usados por cada kernel
    #dilations: diferentes factores de dilatación
    #num_features_per_dilation: cuántas φ_i genera cada dilatación
    #biases: el umbral de activación para cada φ_i
    num_examples, num_channels, input_length = X.shape
    num_channels_per_combination, channel_indices, dilations, num_features_per_dilation, biases = parameters

    #84 Kernels de tamaño fijo
    indices = np.array(( # Comentario original mantenido
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

    num_kernels = len(indices)
    num_dilations = len(dilations)
    num_features = num_kernels * np.sum(num_features_per_dilation)

    #features: matriz donde guardaremos los φ(x)
    #all_traces: trazas para cada φᵢ(x)
    features = np.zeros((num_examples, num_features), dtype=np.float32)
    all_traces = []

    for example_index in range(num_examples):
        _X = X[example_index]
        _X_bar = X_bar[example_index]

        A = -_X
        G = 3 * _X

        feature_index_start = 0
        combination_index = 0
        num_channels_start = 0

        for dilation_index in range(num_dilations):
            dilation = dilations[dilation_index]
            padding = ((9 - 1) * dilation) // 2
            num_features_this_dilation = num_features_per_dilation[dilation_index]

            C_alpha = np.copy(A)
            C_gamma = np.zeros((9, num_channels, input_length), dtype=np.float32)
            C_gamma[4] = G

            start = dilation
            end = input_length - padding

            for gamma_index in range(4):
                C_alpha[:, -end:] += A[:, :end]
                C_gamma[gamma_index, :, -end:] = G[:, :end]
                end += dilation

            for gamma_index in range(5, 9):
                C_alpha[:, :-start] += A[:, start:]
                C_gamma[gamma_index, :, :-start] = G[:, start:]
                start += dilation

            for kernel_index in range(num_kernels):
                feature_index_end = feature_index_start + num_features_this_dilation
                num_channels_this_combination = num_channels_per_combination[combination_index]
                num_channels_end = num_channels_start + num_channels_this_combination
                channels_this_combination = channel_indices[num_channels_start:num_channels_end]
                index_0, index_1, index_2 = indices[kernel_index]
                kernel_weights = [-1] * 9
                for i in [index_0, index_1, index_2]:
                    kernel_weights[i] = 2

                C = (
                    C_alpha[channels_this_combination]
                    + C_gamma[index_0][channels_this_combination]
                    + C_gamma[index_1][channels_this_combination]
                    + C_gamma[index_2][channels_this_combination]
                )
                C = np.sum(C, axis=0)

                for feature_count in range(num_features_this_dilation):
                    feature_index = feature_index_start + feature_count
                    bias = biases[feature_index]

                    if (dilation_index + kernel_index) % 2 == 0:
                        mask = C > bias
                        trace_indices = np.where(mask)[0]
                        ppv = np.mean(mask)
                    else:
                        trimmed = C[padding:-padding]
                        mask = trimmed > bias
                        trace_indices = np.where(mask)[0] + padding
                        ppv = np.mean(mask)

                    features[example_index, feature_index] = ppv

                    trace = {
                        'example': example_index,
                        'feature_index': feature_index,
                        'kernel_index': kernel_index,
                        'bias': bias,
                        'dilation': dilation,
                        'channels': channels_this_combination.tolist(),
                        'ppv': ppv,
                        'activated_indices': trace_indices.tolist(),
                        'weights_indices': [index_0, index_1, index_2],
                        'kernel_weights': kernel_weights
                    }

                    # Para cada característica, ver en qué posición j se activó.
                    input_segments = []
                    for j in trace_indices:
                        for c in channels_this_combination:
                            segment = []
                            for m in range(-4, 5):
                                j_m = j + m * dilation
                                if j_m < 0 or j_m >= input_length:
                                    x_val = 0.0
                                    #x_bar_val = 0.0
                                else:
                                    x_val = _X[c, j_m]
                                    #x_bar_val = _X_bar[c, j_m]
                                w = kernel_weights[m + 4]
                                segment.append((int(j_m), float(x_val), int(w)))
                            input_segments.append({
                                'j': int(j),
                                'canal': int(c),
                                'valores': segment
                            })
                    trace['input_segments'] = input_segments
                    all_traces.append(trace)

                feature_index_start = feature_index_end
                combination_index += 1
                num_channels_start = num_channels_end

    return features, all_traces


def explain(X, X_bar, alphas, transform_function, d=1.0):
    """
    Implementa la fórmula completa de Luis con división recíproca y factores de convolución:
    β_j = Δt_j * m_Δt_j Δf(x), con multiplicador completo.

    Parámetros:
    - X: ndarray (1, canales, longitud) — muestra real
    - X_bar: ndarray — muestra de referencia
    - alphas: vector α_k de pesos del modelo
    - transform_function: función de transformación con trazas enriquecidas
    - d: escalar divisor (por defecto 1.0)

    Retorna:
    - contribuciones_totales: dict {j_m: β_j_m}
    """
    phi_x, trazas = transform_function(X, X_bar) #Se obtiene el valor de la atribución y las trazas para mi entrada real
    phi_x_bar, _ = transform_function(X_bar, X_bar) #Se obtiene el valor de la atribución y las trazas para mi entrada de referencia.
    contribuciones_totales = {} #Suma de todas las contribuciones

    for traza in trazas: #Se procesa cada característica generada por miniRocket
        k = traza['feature_index']  #Indica que característica de miniRocket se está procesando. K es el índice del vector de características phi_x y phi_x
        alpha_k = alphas[k] #Se obtiene el alpha de k del peso que asignó SHAP a la característica
        delta_phi_k = phi_x[k] - phi_x_bar[k] #Diferencia entra la característica phi, tanto para mi entrada real como mi entrada baselina, para notar cuánto cambió la característica al pasar por el baseline.
        if abs(delta_phi_k) < 1e-10:
            continue

        input_segments = traza['input_segments']  #Contiene todas las ventanas activadas. Organizadas por canal involucrado i, centro de la ventana j y las posiciones desplazadas
        # Acumuladores para convoluciones totales
        chi_ki_j = {}      #  #convoulción del canal i, centro j, para X
        chi_ki_j_bar = {}
        chi_k_j = {}       #  convolución total del kernel φₖ en j
        chi_k_j_bar = {}

        for segment in input_segments: #Recorre cada combinación (canal, j) donde se activó phi en la entrada real.
            canal = segment['canal'] #índice del canal
            j = segment['j'] #centro de la ventana que activó. 
            valores = segment['valores']

            chi_total = sum(weight * float(x_val) for (j_m, x_val, _, weight) in valores) #Salida de la convolución del valor real de x en la posición j_m con el peso del kernel en esa posición
            chi_bar_total = sum(weight * float(x_bar_val) for (j_m, _, x_bar_val, weight) in valores)

            chi_ki_j[(canal, j)] = chi_total #Guardar los valores de convolución por canal 
            chi_ki_j_bar[(canal, j)] = chi_bar_total

            chi_k_j[j] = chi_k_j.get(j, 0.0) + chi_total #Se acumula el total del kernek en la posición j, es decir, el chi total de cada canal para la misma ventana
            chi_k_j_bar[j] = chi_k_j_bar.get(j, 0.0) + chi_bar_total

        # Recorremos nuevamente para calcular contribuciones
        for segment in input_segments: #Para cada segmento donde ocurrió la activación 
            canal = segment['canal']
            j = segment['j']
            valores = segment['valores'] #valores de la ventana dispera

            delta_chi_ki_j = chi_ki_j[(canal, j)] - chi_ki_j_bar[(canal, j)] 
            delta_chi_k_j = chi_k_j[j] - chi_k_j_bar[j]

            for (j_m, x_val, x_bar_val, weight) in valores: #Se recorren los 9 puntos de la ventana
                x_val = float(x_val)
                x_bar_val = float(x_bar_val)
                delta_t = x_val - x_bar_val
                chi = weight * x_val
                chi_bar = weight * x_bar_val
                delta_chi = chi - chi_bar

                if abs(delta_t * delta_chi) > 1e-10:
                    dz = weight / (delta_t * delta_chi)
                else:
                    dz = 0.0

                contrib = (alpha_k * delta_phi_k * delta_chi_ki_j * delta_chi_k_j / d) * dz
                contribuciones_totales[j_m] = contribuciones_totales.get(j_m, 0.0) + contrib

    return contribuciones_totales


