import numpy as np
import tensorflow as tf
from typing import Tuple
from copy import deepcopy


# Code inspired by https://github.com/ismailuddin/gradcam-tensorflow-2
@tf.custom_gradient
def guided_relu(x: tf.Tensor) -> Tuple[tf.Tensor, callable]:
    """Applies the guided ReLU activation function to a tensor.

    For the foward pass, the guided ReLU function is identical to the ReLU function:
        guided_relu(x) = max(0, x)

    Args:
        x (tf.Tensor): The input tensor.

    Returns:
        (Tuple[tf.Tensor, callable]): The output tensor after applying the guided ReLU
            activation function and the gradient function.

    """

    def grad(dy: tf.Tensor) -> tf.Tensor:
        """Computes the gradient of the guided ReLU function.

        The gradient of the guided ReLU function is defined as:
            grad(dy) = dy if dy > 0 and x > 0, otherwise 0

        Args:
            dy (tf.Tensor): The gradient tensor.

        Returns:
            (tf.Tensor): The computed gradient tensor.

        """
        return tf.cast(dy > 0, "float32") * tf.cast(x > 0, "float32") * dy

    return tf.nn.relu(x), grad


def build_guided_model(model) -> tf.keras.Model:
    """Builds a new model by replacing all ReLU activation function
    with the guided ReLU activation function.

    Arguments:
        model (tf.keras.Model): Tensorflow model.

    Returns:
        guided_bp_model (tf.keras.Model): Guided backpropagation model.
    """

    # Deep copy the model
    guided_bp_model = deepcopy(model)

    # Replace ReLU activation with guided ReLU activation
    layers = [
        layer for layer in guided_bp_model.layers[1:] if hasattr(layer, "activation")
    ]
    for layer in layers:
        if layer.activation == tf.keras.activations.relu:
            layer.activation = guided_relu

    return guided_bp_model


def guided_backpropagation(model, image: np.ndarray, top_k_index: int = 1) -> tf.Tensor:
    """Guided backpropagation algorithm. This function computes the gradients of a given top
    category with respect to the input image.

    Arguments:
        model (tf.keras.Model): Tensorflow model.
        image (np.ndarray): Input image.
        top_k_index (int): Index of the top category.

    Returns:
        gradients (tf.Tensor): Gradients of the chosen top category with respect to the input image.
    """

    # Build guided backpropagation model
    guided_bp_model = build_guided_model(model)

    # Compute gradients
    with tf.GradientTape() as tape:
        inputs = tf.cast(image, tf.float32)
        tape.watch(inputs)
        preds = guided_bp_model(inputs)

        # Retrieve class index of the given top_category
        class_index = tf.math.top_k(preds[0], k=top_k_index)[1][top_k_index - 1]

        # Extract logit value for the respective class
        y_class = preds[:, class_index]

    gradients = tape.gradient(y_class, inputs)[0]

    return gradients
