import PIL
import cv2
import PIL.Image
import numpy as np
import tensorflow as tf
from copy import deepcopy
from tqdm.auto import tqdm

from typing import Tuple, List
import matplotlib.pyplot as plt
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import load_img

from utils.cam import grad_cam, grad_cam_plusplus


def load_network(network_name: str) -> Tuple[
    tf.keras.Model,
    Tuple[int, int],
    tf.keras.layers.Layer,
    imagenet_utils.preprocess_input,
    imagenet_utils.decode_predictions,
]:
    """Function that loads the respective network from tensorflow.keras.applications

    Arguments:
        network_name (str): The name of the network to load

     Returns:
        model (tf.keras.Model): The respective network instance
        size (Tuple[int, int]): The required input size for the respective network
        last_conv_layer (tf.keras.layers.Layer): The last convolutional layer of the respective network
        preprocess_input (imagenet_utils.preprocess_input): The input preprocessing function of the respective network
        decode_predictions (imagenet_utils.decode_predictions): A function to decode the respective network's preidctions
    """

    match network_name:
        case "ResNet50":
            from tensorflow.keras.applications.resnet50 import (
                ResNet50,
                preprocess_input,
                decode_predictions,
            )

            # Load the model and remove the softmax activation of the last layer
            # as the gradients need to be computed for the non-softmax score of the class
            model = ResNet50(weights="imagenet", classifier_activation=None)

            # Extract the last convolutional layer of the model for which we are going to compute the heatmap
            last_conv_layer = model.get_layer("conv5_block3_out")

        case "VGG16":

            from tensorflow.keras.applications.vgg16 import (
                VGG16,
                preprocess_input,
                decode_predictions,
            )

            model = VGG16(weights="imagenet", classifier_activation=None)

            last_conv_layer = model.get_layer("block5_conv3")

        case "DenseNet121":

            from tensorflow.keras.applications.densenet import (
                DenseNet121,
                preprocess_input,
                decode_predictions,
            )

            model = DenseNet121(weights="imagenet", classifier_activation=None)

            last_conv_layer = model.get_layer("relu")

        case "MobileNetV2":

            from tensorflow.keras.applications.mobilenet_v2 import (
                MobileNetV2,
                preprocess_input,
                decode_predictions,
            )

            model = MobileNetV2(weights="imagenet", classifier_activation=None)

            last_conv_layer = model.get_layer("out_relu")

        case "Xception":

            from tensorflow.keras.applications.xception import (
                Xception,
                preprocess_input,
                decode_predictions,
            )

            model = Xception(weights="imagenet", classifier_activation=None)

            last_conv_layer = model.get_layer("block14_sepconv2_act")

    # Size for the input images of the network
    size = model.get_config()["layers"][0]["config"]["batch_shape"][1:3]

    return model, size, last_conv_layer, preprocess_input, decode_predictions


def load_and_preprocess_img(
    path: str,
    size: Tuple[int, int],
    preprocess_input: imagenet_utils.preprocess_input = None,
) -> Tuple[PIL.Image.Image, np.array]:
    """Function that loads an image and applys preprocessing

    Arguments:
        path (str): The path to the image
        size (Tuple[int, int]): The required input size for the respective network, used for scaling the image when loading
        preprocess_input (imagenet_utils.preprocess_input): The preprocessing function of the respective network

    Returns:
        image (PIL.Image.Image): The scaled image without further preprocessing
        image_arr (np.array): The preprocessed image as an array
    """

    # First we load the image from the path with the respective target size
    image = load_img(path=path, target_size=size)

    # 1. Convert to array
    image_arr = np.array(image)

    # 2. Add a batch dimension
    image_arr = np.expand_dims(image_arr, axis=0)

    # 3. Apply the preprocessing scheme of the respective model
    image_arr = preprocess_input(image_arr)

    return image, image_arr


def draw_heatmap(
    input_image: PIL.Image.Image,
    heatmaps: List[Tuple[Tuple[str, str, float], tf.Tensor]],
    counterfactual: bool = False,
) -> None:
    """Function that visualizes the heatmap on top of the given input image

    Arguments:
        input_image (PIL.Image.Image): The input image
        heatmaps (List[Tuple[Tuple[str, str, float], tf.Tensor]]): The computed class activation maps
            with their class labels and probabilities
        counterfactual (bool): Whether the heatmaps contain a counterfactual explanation or not
    """

    # Get dims of the input image
    size = input_image.size

    # Calculate the number of rows and columns for the subplots depending on the number of heatmaps
    # there will be at least always two images, the input image and the heatmap
    rows = (len(heatmaps) + 1) // 3

    if (len(heatmaps) + 1) % 3 != 0:
        rows += 1

    cols = 3

    heatmaps.insert(0, ("Input Image", input_image))

    fig, ax = plt.subplots(rows, cols, figsize=(20, 10))

    # Deactive axis for all subplots
    for a in ax.flatten():
        a.axis("off")

    # Plot the heatmaps
    for ax, (category, heatmap) in zip(ax.flatten()[: len(heatmaps) + 1], heatmaps):

        if category == "Input Image":
            ax.imshow(heatmap)
            ax.set_title(category)

        else:
            heatmap = cv2.resize(heatmap.numpy(), size)
            ax.imshow(input_image)
            ax.imshow(heatmap, cmap="rainbow", alpha=0.5)

            if counterfactual:
                ax.set_title(f"Counterfactual Heatmap for {category[1]}")
            else:
                ax.set_title(
                    f"Heatmap for {category[1]} with probability {round(category[2]*100, 2)}%"
                )


def guided_bp_map_postprocessing(guided_bp_map: np.array) -> np.array:

    # Center on 0 with std 0.25
    guided_bp_map -= guided_bp_map.mean()
    guided_bp_map /= guided_bp_map.std() + tf.keras.backend.epsilon()
    guided_bp_map *= 0.25

    # Clip to [0, 1]
    guided_bp_map += 0.5
    guided_bp_map = np.clip(guided_bp_map, 0, 1)

    # Convert to 0-255
    guided_bp_map *= 255
    guided_bp_map = guided_bp_map.astype(np.uint8)

    return guided_bp_map


def draw_guided_heatmap(
    input_image: PIL.Image.Image,
    guided_bp_map: tf.Tensor,
    heatmap: tf.Tensor,
) -> None:
    """Function that visualizes guided backpropagation, the corresponding heatmap and their combination

    Arguments:
        input_image (PIL.Image.Image): The input image
        guided_bp_map (nd
        heatmap (List[Tuple[Tuple[str, str, float], tf.Tensor]]): The computed class activation map
            with their class labels and probabilities
    """

    # Get size of image
    size = input_image.size

    # Get predicted class
    class_name = heatmap[0][0][1]

    # Upscale the heatmap to the size of the input image
    heatmap = cv2.resize(heatmap[0][1].numpy(), size)

    # Combine both the guided bp and grad cam images
    # As the CAM lacks color channels we repeat it 3 times to match the saliency map
    guided_heatmap = guided_bp_map * np.repeat(heatmap[..., np.newaxis], 3, axis=2)

    guided_bp_map_copy = deepcopy(guided_bp_map)
    guided_heatmap_copy = deepcopy(guided_heatmap)

    guided_bp_map_processed = guided_bp_map_postprocessing(guided_bp_map_copy)
    guided_heatmap_processed = guided_bp_map_postprocessing(guided_heatmap_copy)

    fig, ax = plt.subplots(1, 4, figsize=(20, 10))

    # Deactive axis for all subplots
    for a in ax.flatten():
        a.axis("off")

    # Plot the maps
    ax[0].imshow(input_image)
    ax[1].set_title("Input Image")

    ax[1].imshow(input_image)
    ax[1].imshow(heatmap, cmap="rainbow", alpha=0.5)
    ax[1].set_title(f"Heatmap for {class_name}")

    ax[2].imshow(guided_bp_map_processed)
    ax[2].set_title(f"Guided BP Map for {class_name}")

    ax[3].imshow(guided_heatmap_processed)
    ax[3].set_title(f"Guided Heatmap for {class_name}")


def draw_conv_layer_comparision_heatmap(
    input_image: PIL.Image.Image,
    heatmaps: List[Tuple[str, tf.Tensor]],
) -> None:
    """Function that visualizes the heatmap on top of the given input image for various convolutional layers of ResNet50

    Arguments:
        input_image (PIL.Image.Image): The input image
        heatmaps (List[Tuple[str, tf.Tensor]]): The computed class activation maps
            with their layer names
    """

    # Get dims of the input image
    size = input_image.size

    fig, ax = plt.subplots(5, 4, figsize=(30, 30))

    # Deactive axis for all subplots
    for a in ax.flatten():
        a.axis("off")

    ax[0, 0].imshow(input_image)
    ax[0, 0].set_title("Input Image")

    # Plot the heatmaps
    for ax, (layer_name, heatmap) in zip(ax.flatten()[1 : len(heatmaps) + 1], heatmaps):

        heatmap = cv2.resize(heatmap.numpy(), size)
        ax.imshow(input_image)
        ax.imshow(heatmap, cmap="rainbow", alpha=0.5)
        ax.set_title(f"Heatmap for  layer {layer_name}")


def conv_layer_comparision(image_path: str, method: str = "grad_cam"):
    """Function that draws the heatmaps for various convolutional layers of ResNet50

    Arguments:
        image_path (str): The path to the image
        method (str): The method to compute the heatmap (grad_cam or grad_cam_plusplus)
    """

    model, size, _, preprocess_input, decode_predictions = load_network("ResNet50")
    image, image_arr = load_and_preprocess_img(
        path=image_path, size=size, preprocess_input=preprocess_input
    )

    # Extract activated convolutional layers from the model that are the output of a ResNet block
    conv_layers = [
        layer
        for layer in model.layers
        if "conv" in layer.name
        and "out" in layer.name
        and isinstance(layer, tf.keras.layers.Activation)
    ]

    heatmaps = []

    # Compute the heatmaps for each convolutional layer
    for layer in tqdm(conv_layers, desc=f"Computing Heatmaps for {method}"):

        match method:
            case "grad_cam":
                heatmap = grad_cam(
                    model=model,
                    last_conv_layer=layer,
                    decode_predictions=decode_predictions,
                    image_arr=image_arr,
                    top_k=1,
                    counterfactual=False,
                )[0]
            case "grad_cam_plusplus":
                heatmap = grad_cam_plusplus(
                    model=model,
                    last_conv_layer=layer,
                    decode_predictions=decode_predictions,
                    image_arr=image_arr,
                    top_k=1,
                    counterfactual=False,
                )[0]

        heatmaps.append((layer.name, heatmap[1]))

    draw_conv_layer_comparision_heatmap(input_image=image, heatmaps=heatmaps)
