import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm
from typing import Tuple, List
from tensorflow.keras.applications import imagenet_utils


def cam(
    model: tf.keras.Model,
    last_conv_layer: tf.keras.Model,
    decode_predictions: imagenet_utils.decode_predictions,
    image_arr: np.array,
    top_k: int,
) -> List[Tuple[Tuple[str, str, float], tf.Tensor]]:
    """Function that implements the Class Activation Mapping (CAM) Algorithm

    Arguments:
        model (tf.keras.Model): The model for which the CAM heatmap should be computed
        last_conv_layer (tf.keras.layers): The last convolutional layer of the model
        decode_predictions (imagenet_utils.decode_predictions): The function to decode the model predictions
        image_arr (np.array): The preprocessed image as an array
        top_k (int): The number of top predictions to consider

    Returns:
        heatmaps (List[Tuple[Tuple[str, str, float], tf.Tensor]]): The computed class activation maps
            with their class labels and probabilities
    """

    # Create a model that returns both the output of the last convolutional layer as well as the model predictions
    cam_model = tf.keras.Model(model.inputs, [last_conv_layer.output, model.output])

    last_conv_layer_output, preds = cam_model(image_arr)

    # Decode the model predictions and get the top category index
    output_probabalities = tf.nn.softmax(preds).numpy()
    top_prediction = decode_predictions(output_probabalities, top=top_k)

    # Retrieve class indices of the given top k categories
    class_indices = tf.math.top_k(preds[0], k=top_k)[1]

    heatmaps = []

    for i in tqdm(range(top_k), desc="Computing CAM Heatmaps"):
        
        # Get the weights of the last dense layer
        cam_weights = model.weights[-2][:, class_indices[i]]

        # Compute the class activation map
        heatmap = tf.tensordot(last_conv_layer_output[0], cam_weights, axes=1)
        heatmap = tf.squeeze(heatmap)

        heatmaps.append((top_prediction[0][i], heatmap))

    return heatmaps


# The following implementation is inspired by the tutorial from the keras documentation
# https://keras.io/examples/vision/grad_cam/#gradcam-class-activation-visualization.
def grad_cam(
    model: tf.keras.Model,
    last_conv_layer: tf.keras.Model,
    decode_predictions: imagenet_utils.decode_predictions,
    image_arr: np.array,
    top_k: int,
    counterfactual: bool = False,
) -> List[Tuple[Tuple[str, str, float], tf.Tensor]]:
    """Function that implements the Grad-Cam Algorithm

    Arguments:
        model (tf.keras.Model): The model for which the Grad-CAM heatmap should be computed
        last_conv_layer (tf.keras.layers): The last convolutional layer of the model
        decode_predictions (imagenet_utils.decode_predictions): The function to decode the model predictions
        image_arr (np.array): The preprocessed image as an array
        top_k (int): The number of top predictions to consider
        counterfactual (bool): Whether to compute the counterfactual Grad-CAM heatmap

    Returns:
        heatmaps (List[Tuple[Tuple[str, str, float], tf.Tensor]]): The computed class activation maps
            with their class labels and probabilities
    """

    # Create a model that returns both the output of the last convolutional layer as well as the model predictions
    grad_model = tf.keras.Model(model.inputs, [last_conv_layer.output, model.output])

    heatmaps = []

    for i in tqdm(range(top_k), desc="Computing Grad-CAM Heatmaps"):

        # Here we compute the gradients of the output neuron (of the top predicted class) w.r.t.
        # the feature map of the last conv layer
        with tf.GradientTape() as tape:

            last_conv_layer_output, preds = grad_model(image_arr)

            # Retrieve class index of the given top_category
            class_index = tf.math.top_k(preds[0], k=top_k)[1][i - 1]

            # Extract logit value for the respective class
            y_class = preds[:, class_index]

            # The gradients have the same dimensionality as the feature maps of the extracted convolutional layer
            # If counterfactual is set to True, we negate the gradients
            if counterfactual:
                gradients = -tape.gradient(y_class, last_conv_layer_output)

            else:
                gradients = tape.gradient(y_class, last_conv_layer_output)

        # Compute the mean of each gradient map (GAP)
        # --> We are left with one mean gradient value per feature map
        alphas = tf.reduce_mean(gradients, axis=(0, 1, 2))

        # Multiply each feature map with the respective alpha weight and
        # sum across the different feature maps to obtain a single heatmap
        heatmap = last_conv_layer_output[0] @ alphas[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        # Apply ReLU
        heatmap = tf.maximum(heatmap, 0)

        # Decode the model predictions and get the top category index
        output_probabalities = tf.nn.softmax(preds).numpy()
        top_prediction = decode_predictions(output_probabalities, top=top_k)

        heatmaps.append((top_prediction[0][i], heatmap))

    return heatmaps


# Implementation for Grad-CAM++ is based on following sources as well as on the original paper:
# - https://www.kaggle.com/itoeiji/visual-explanations-gradcam-gradcam-scorecam
# - https://github.com/adityac94/Grad_CAM_plus_plus/blob/4a9faf6ac61ef0c56e19b88d8560b81cd62c5017/misc/utils.py#L51
# - https://github.com/totti0223/gradcamplusplus/blob/master/gradcamutils.py
# - https://github.com/samson6460/tf_keras_gradcamplusplus/blob/master/gradcam.py
def grad_cam_plusplus(
    model: tf.keras.Model,
    last_conv_layer: tf.keras.Model,
    decode_predictions: imagenet_utils.decode_predictions,
    image_arr: np.array,
    top_k: int,
) -> List[Tuple[Tuple[str, str, float], tf.Tensor]]:
    """Function that implements the Grad-Cam++ Algorithm

    Arguments:
        model (tf.keras.Model): The model for which the Grad-CAM++ heatmap should be computed
        last_conv_layer (tf.keras.layers): The last convolutional layer of the model
        decode_predictions (imagenet_utils.decode_predictions): The function to decode the model predictions
        image_arr (np.array): The preprocessed image as an array
        top_k (int): The number of top predictions to consider

    Returns:
        heatmaps (List[Tuple[Tuple[str, str, float], tf.Tensor]]): The computed class activation maps
            with their class labels and probabilities
    """

    # Create a model that returns both the output of the last convolutional layer as well as the model predictions
    grad_model = tf.keras.Model(model.inputs, [last_conv_layer.output, model.output])

    heatmaps = []

    for i in tqdm(range(top_k), desc="Computing Grad-CAM++ Heatmaps"):

        # Here we compute the first-, second- and third-order derivatives of the exponentiated
        # output neuron w.r.t. the feature map of the last conv layer
        with tf.GradientTape() as tape1:
            with tf.GradientTape() as tape2:
                with tf.GradientTape() as tape3:

                    last_conv_layer_output, preds = grad_model(image_arr)

                    # Retrieve class index of the given top_category
                    class_index = tf.math.top_k(preds[0], k=top_k)[1][i - 1]

                    # Extract logit value for the respective class
                    y_class = preds[:, class_index]

                    first_order_gradients = tape3.gradient(
                        tf.exp(y_class), last_conv_layer_output
                    )
                second_order_gradients = tape2.gradient(
                    first_order_gradients, last_conv_layer_output
                )
            third_order_gradients = tape1.gradient(
                second_order_gradients, last_conv_layer_output
            )

        # Sum up each feature map of the last conv layer
        # --> Results in one value per feature map
        sum_of_feature_maps = tf.reduce_sum(last_conv_layer_output, axis=(0, 1, 2))

        # Here we set up the numerator and denominator for the beta values
        beta_numerator = second_order_gradients[0]
        beta_denominator = (
            2.0 * second_order_gradients[0]
            + sum_of_feature_maps * third_order_gradients[0]
        )
        beta_denominator = np.where(beta_denominator != 0.0, beta_denominator, 1e-10)

        # Calculating the beta values and normalizing them
        betas = beta_numerator / beta_denominator
        beta_normalization_constant = tf.reduce_sum(betas, axis=(0, 1))
        betas /= beta_normalization_constant

        # Produce the final heatmap
        gradients = tf.maximum(first_order_gradients[0], 0.0)
        alphas = tf.reduce_sum(gradients * betas, axis=(0, 1))
        heatmap = tf.reduce_sum(alphas * last_conv_layer_output[0], axis=2)

        # Apply ReLU
        heatmap = tf.maximum(heatmap, 0)

        # Decode the model predictions and get the top category index
        output_probabalities = tf.nn.softmax(preds).numpy()
        top_prediction = decode_predictions(output_probabalities, top=top_k)

        heatmaps.append((top_prediction[0][i], heatmap))

    return heatmaps
