import numpy as np


def fedavg(
    client_weights,
    client_sizes
):
    

    total_samples = sum(
        client_sizes
    )

    averaged_weights = []

    for layer_weights in zip(
        *client_weights
    ):

        weighted_average = np.zeros_like(
            layer_weights[0],
            dtype=np.float32
        )

        for weights, size in zip(
            layer_weights,
            client_sizes
        ):

            weighted_average += (
                weights *
                (size / total_samples)
            )

        averaged_weights.append(
            weighted_average
        )

    return averaged_weights