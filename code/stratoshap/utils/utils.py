import warnings
warnings.filterwarnings('ignore')
import numpy as np
import scipy.special
import itertools
from datetime import datetime
import os
import json
import math
import matplotlib.pyplot as plt

def layer_instance_number(layer,nbF):
    """
    Number of coalitions given a layer and the number of features.
    
    Args:
        layer (int): desired layer.
        nbF (int): Total number of features.

    Returns:
        int: total number of coalitions in the layer.
    """
    total_layers = np.int64(np.ceil((nbF - 1) / 2.0))
    layer = max(1, min(layer, total_layers))

    num = scipy.special.binom(nbF, layer)

    if layer <= (nbF - 1) // 2:
        num *= 2

    return np.int64(num)

def stratum_instance_number(stratum, nbF):
    """
    Number of coalitions given a stratum and the number of features.
    
    Args:
        stratum (int): desired stratum.
        nbF (int): Total number of features.

    Returns:
        int: number of coalitions within the stratum.
    """
    total_stratums = int(np.ceil((nbF - 1) / 2.0))
    stratum = max(1, stratum)
    stratum = min(stratum, total_stratums)
    num = 0
    if stratum == 1 :
        num = 2 * nbF

    elif stratum == total_stratums :
        num = 2**nbF - 2
    else:
        for k in range(1, stratum + 1):
            num += layer_instance_number(k, nbF)
          
    return int(num)

def budget_to_stratum(num, nbF, strict=True):
    """
    Find the stratum corresponding to a given budget (num)
    and the number of features (nbF).

    Args:
        num (int): The given number of coalitions.
        nbF (int): Total number of features.

    Returns:
        int: The corresponding stratum.
    """
    total_stratums = int(np.ceil((nbF - 1) / 2.0))
    cumulative_coalitions = 0


    for stratum in range(1, total_stratums + 1):
        coalitions_in_stratum = layer_instance_number(stratum, nbF)
        cumulative_coalitions += coalitions_in_stratum

        if num:
            if cumulative_coalitions == num:
                return stratum
            elif cumulative_coalitions < num:
                continue
            else:
                return 0 if strict else stratum - 1
        else:
            return 0

def all_coalitions(nbF):
    """
    Generate all possible coalitions given the number of features.

    Args:
        nbF (int): Total number of features.

    Returns:
        All possible coalitions based on the number of features.
    """
    s = list(nbF)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1)) 

def layer_coalitions(layer, nbF):
    """
    Generate all coalitions of a given layer.

    Args:
        layer (int): desired layer.
        nbF (int): Total number of features.
    
    Returns:
        All coalitions of this layer.
    """
    s = range(nbF)
    coalitions = []
    total_layers = np.int64(np.ceil((nbF - 1) / 2.0))
    num_paired_subset_sizes = np.int64(np.floor((nbF - 1) / 2.0))
    layer = max(1,layer)
    layer = min(layer, total_layers)
    
    presence = itertools.combinations(s, layer)
    coalitions = itertools.chain(coalitions, presence)
    if layer<= num_paired_subset_sizes:
        occlusion = itertools.combinations(s,nbF-layer)
        coalitions= itertools.chain(coalitions,occlusion)

    return list(coalitions)

def stratum_coalitions(stratum, nbF):
    """
    Generate all coalitions of a given stratum.

    Args:
        stratum (int): desired stratum.
        nbF (int): Total number of features.
    
    Returns:
        All coalitions of this stratum.
    """
    total_stratums = np.int64(np.ceil((nbF - 1) / 2.0))
    stratum = max(1, stratum)
    stratum = min(stratum, total_stratums)
    coalitions = []
    for layer in range(1, stratum + 1):
        coalitions += layer_coalitions(layer, nbF)
    return list(coalitions)

def is_binary_list(lst):
    """
    Check if a list contains only binary elements (0 or 1).
    
    Args:
        lst (list): The list to check.
    
    Returns:
        bool: True if all elements are 0 or 1, False otherwise.
    """
    return all(x == 0 or x == 1 for x in lst)

def shapley_kernel(layer, nbF):
    """
    Calculate the weight of a layer's instances.
    Performed using the formula
    w = (nbF-1)/[(nbF choose layer) * layer * (nbF-layer)]
    
    Args:
        layer (int, tuple, list): Desired layer.
            If int: use the value as is. It corresponds to the desired layer (coalitions with the number of features present and features absent).
            If tuple: use the length of the tuple. It corresponds to indexes of features in coaliton.
            If binary vector (list): use the count of 1s in the list. It corresponds to the number of features present in the coalition.
        nbF (int): Total number of features.
    
    Returns:
        float: Weight of layer instances.
    """
    if isinstance(layer, int):
        layer_length=layer   
    elif isinstance(layer, (tuple, list)):
        if isinstance(layer, list) and is_binary_list(layer) and len(layer) == nbF:
            layer_length = len([x for x in layer if x == 1])      
        else:
            layer_length = len(layer)
    else:
        raise ValueError("Unsupported type for 'layer'. Must be an integer, tuple or list.")
    
    if layer_length == 0 or layer_length == nbF:
        return 10_000_000_000
    else:
        return (nbF - 1) / (scipy.special.binom(nbF, layer_length) * layer_length * (nbF - layer_length))
    

def vsi_and_jaccard_indices(data, explanation_size):
    """
        Function to compute the Variable Stability Index and the Jaccard Index.

        Args:
            data (list): list of different explanations (list) representing different executions
            explanation_size (int): Size of the explanation, i.e., the number of non-zero coefficients in each of the explanations.
        Returns:
            vsi (float): Variable Stability Index.
            jaccard (float): Jaccard Index.
    """

    nb_combs = (len(data) * (len(data) - 1)) // 2 
    concordance = 0
    jaccard_coefficients  = np.zeros(nb_combs)

    for i,pair in enumerate(itertools.combinations(range(len(data)), 2)):
        i1, i2 = pair
        intersection = np.sum(data[i1] & data[i2])
        union = np.sum(data[i1] | data[i2])
        if union > 0:
            jaccard_coefficients[i] = intersection/union

        concordance += intersection

    jaccard = round(np.mean(jaccard_coefficients ), 3)
    vsi = round((concordance / (nb_combs * explanation_size)), 3)

    return vsi, jaccard

# @njit
def reconstruct_coalitions(coalitions, explained_instance, nbF, background_dataset):
    """
    Reconstruct coalitions from the binary masks to the original space of the instance to explain.
    
    Args:
        masks (numpy.ndarray): coalitions.
        explained_instance (numpy.ndarray): instance to explain.
        nbF (int): number of features.
        reference (numpy.ndarray): reference instance. Since most models aren't designed to handle arbitrary missing data at test time, we simulate "missing" by replacing the feature a baselinbe value.

    """

    # print(f"Coalitions: {coalitions}")
    coalitions = np.array(coalitions)

    # Convert a single coalition to a list of coalitions if necessary
    if isinstance(coalitions, np.ndarray) and coalitions.ndim == 1:
        coalitions = [coalitions]


    num_bg_instances = background_dataset.shape[0]
    synth_data = np.zeros((len(coalitions)*num_bg_instances, nbF))
    masks = np.zeros((len(coalitions), nbF))
    weights = np.zeros(len(coalitions))

    synth_data= np.tile(background_dataset, (len(coalitions), 1))
    
    for i, s in enumerate(coalitions):
        s = list(s)
        masks[i, s] = 1
        weights[i] = shapley_kernel(len(s), nbF)

        synth_data[i*num_bg_instances:(i+1)*num_bg_instances, s] = explained_instance[s]
    
    
    synth_data = synth_data.reshape((len(coalitions), num_bg_instances, nbF))
    return masks, synth_data, weights

def get_masks_and_weights(coalitions, nbF):
    """
    Function to get the masks and weights for each coalition.

    Args:
        coalitions (List of lists): Each sublist represents the active features for a coalition.
        nbF (int): Number of features.

    Returns:
        masks (numpy.ndarray): Masks of the coalitions.
        weights (numpy.ndarray): Weights for each coalition based on the Shapley kernel.
    """
    
    coalitions = np.array(coalitions)
    masks = np.zeros((len(coalitions), nbF))
    weights = np.zeros(len(coalitions))

    for i, s in enumerate(coalitions):
        s = list(s)
        masks[i, s] = 1
        weights[i] = shapley_kernel(len(s), nbF)

    return masks, weights

def segment_image(image_array, tile_size):
    """
     Function to segment image into tiles
    """
    
    img_dim = image_array.shape[1]
    tiles = []
    for i in range(0, img_dim, tile_size):
        for j in range(0, img_dim, tile_size):
            tile = image_array[0, i:i+tile_size, j:j+tile_size, :]
            tiles.append(tile)
    return tiles


def construct_disturbed_images(coalitions, img_dim, all_tiles):
    """
        Function to construct disturbed images with tiles that are not part of the coalition set to black.

    Args:
        coalitions (List of lists ):Each sublist represents the active tiles for a coalition.
        all_tiles (list): List of all tiles from the segmented image.

    Returns:
        disturb_imgs (numpy.ndarray): Disturbed images with inactive tiles set to black.
        masks (numpy.ndarray): Masks of the coalitions.
        weights (numpy.ndarray): Weights for each coalition based on the Shapley kernel.
    """
    nb_superpixels = len(all_tiles)
    coalitions = np.array(coalitions)

    tile_size = int(img_dim//math.sqrt(nb_superpixels))
    tile_per_dim = int(math.sqrt(nb_superpixels))

    if isinstance(coalitions, np.ndarray) and coalitions.ndim == 1:
        coalitions = [coalitions]

    disturb_imgs = np.zeros((len(coalitions),img_dim,img_dim,3))

    masks = np.zeros((len(coalitions), nb_superpixels))
    weights = np.zeros(len(coalitions))

    for i,s in enumerate(coalitions):
        s = list(s)
        masks[i, s] = 1
        weights[i] = shapley_kernel(len(s), nb_superpixels)

        #Construct disturbed image

        disturb_image = np.zeros((img_dim,img_dim,3))

        for idx, tile in enumerate(all_tiles):
            if idx in s:
                row = (idx // tile_per_dim) * tile_size
                col = (idx % tile_per_dim) * tile_size
                disturb_image[row:row+tile_size, col:col+tile_size] = tile
                
        disturb_imgs[i] = disturb_image


    return masks, disturb_imgs, weights

        
def visualize_tiles(tiles):
    num_tiles = len(tiles)
    grid_size = int(np.ceil(np.sqrt(num_tiles)))  # Determine the grid size

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
    axes = axes.flatten()

    for i, tile in enumerate(tiles):
        if tile.shape[0] > 0 and tile.shape[1] > 0:
            axes[i].imshow((tile / 255))
            axes[i].axis('off')
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()


def reverse_preprocess_input(img_array):
    """
    Reverses the preprocessing of preprocess_input for ResNet, VGG16, and VGG19 models.
    The goal is to restore the original pixel values before normalization.

    Args:
    - img_array: Normalized image (shape: (1, H, W, 3) or (H, W, 3))

    Returns:
    - img_original: Restored image (pixel values between 0 and 255))
    """
    imagenet_mean = np.array([103.939, 116.779, 123.68], dtype=np.float32)

    if img_array.ndim == 4:
        img_array = img_array[0]

    img_original = img_array + imagenet_mean

    img_original = np.clip(img_original, 0, 255).astype(np.uint8)

    return img_original

def build_image_from_tiles(tiles, img_dim):
    nb_superpixels = len(tiles)

    tile_per_dim = int(math.sqrt(nb_superpixels))
    tile_size = img_dim // tile_per_dim

    img = np.zeros((img_dim, img_dim, 3))

    for i in range(tile_per_dim):
        for j in range(tile_per_dim):
            idx = i * tile_per_dim + j
            img[i*tile_size:(i+1)*tile_size, j*tile_size:(j+1)*tile_size, :] = tiles[idx]
    return img


def calculate_mse(estimate, actual):
    return np.square(np.subtract(estimate, actual)).mean()


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
def save_experiment_to_file(data):
    timestamp = datetime.now().strftime("%m-%d-%Y_%H%M%S")
    if not os.path.exists('Data'):
        os.mkdir('Data')
    with open(os.path.join('Data', f'evaluation__{timestamp}.json'), mode='w') as file:
        json.dump(data, file, cls=NumpyArrayEncoder, indent=2)


# uniform distribution over sizes between 2 and n-2
def generateUniformDistribution(n):
    dist = [0 for i in range(n+1)]
    for s in range(2, n-1):
        dist[s] = 1 / (n-3)
    return dist


# probability distribution over sizes for sampling according to paper
def generatePaperDistribution(n):
    dist = [0 for i in range(n+1)]

    if n % 2 == 0:
        nlogn = n * math.log(n)
        H = sum([1 / s for s in range(1, int(n / 2))])
        nominator = nlogn - 1
        denominator = 2 * nlogn * (H - 1)
        frac = nominator / denominator
        for s in range(2, int(n / 2)):
            dist[s] = frac / s
            dist[n - s] = frac / s
        dist[int(n / 2)] = 1 / nlogn
    else:
        H = sum([1 / s for s in range(1, int((n - 1) / 2 + 1))])
        frac = 1 / (2 * (H - 1))
        for s in range(2, int((n - 1) / 2 + 1)):
            dist[s] = frac / s
            dist[n - s] = frac / s

    return dist


def generateLinearlyDescendingDistribution(n, factor):
    dist = [0 for i in range(n + 1)]
    m = 1.0 / ((n-3) * (factor*(n-2)-2)/(factor-1) - (n-2)*(n-1)/2 + 1)
    b = m * (factor * (n-2) - 2) / (factor-1)
    for s in range(2, n-1):
        dist[s] = b - m * s
    return dist

def generateDistribution(name, n):
    if name == "paper":
        return generatePaperDistribution(n)
    elif name == "uniform":
        return generateUniformDistribution(n)
    elif name == "descending":
        return generateLinearlyDescendingDistribution(n, 3.0)
    else:
        return []