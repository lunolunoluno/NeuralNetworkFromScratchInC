#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "dataset.h"
#include "layer.h"
#include "utils.h"

#define BATCH_SIZE 64
#define INPUT_SIZE 2
#define NB_OUTPUT 3

int main()
{
    srand((unsigned int)time(NULL));
    int nb_layers = 4;
    int layers_shape[] = {64, 64, 64, NB_OUTPUT};

    // INIT VARIABLES
    dataset train_dataset;

    layer_params *layers = malloc(nb_layers * sizeof(layer_params));
    activation_params *activations = malloc(nb_layers * sizeof(activation_params));

    // GIVE VARIABLES INITIAL VALUES
    int nb_epochs = 1000;
    float learning_rate = 0.01;
    float epsilon = 1e-7;

    create_dataset(&train_dataset, "./100000_samples_3_cat_shuffled.csv", INPUT_SIZE, BATCH_SIZE, NB_OUTPUT, ',');

    init_layer(INPUT_SIZE, layers_shape[0], BATCH_SIZE, &layers[0]);
    init_activation(layers_shape[0], BATCH_SIZE, &activations[0]);
    for (int i = 1; i < nb_layers; i++)
    {
        init_layer(layers_shape[i - 1], layers_shape[i], BATCH_SIZE, &layers[i]);
        init_activation(layers_shape[i], BATCH_SIZE, &activations[i]);
    }

    // TRAIN NETWORK
    for (int epoch = 0; epoch < nb_epochs; epoch++)
    {
        for (int batch_index = 0; batch_index < train_dataset.nb_batches; batch_index++)
        {
            for (int layer_idx = 0; layer_idx < nb_layers; layer_idx++)
            {
                // FEED FORWARD
                if (layer_idx == 0)
                {
                    layer_forward(&layers[layer_idx], train_dataset.batches[batch_index].data);
                } else {
                    layer_forward(&layers[layer_idx], activations[layer_idx - 1].outputs);
                }

                // ACTIVATION FUNCTION
                if (layer_idx == nb_layers - 1)
                {
                    softmax_forward(&activations[layer_idx], layers[layer_idx].outputs);
                }
                else
                {
                    relu_forward(&activations[layer_idx], layers[layer_idx].outputs);
                }
            }

            // CALCULATE CATEGORICAL CROSS-ENTROPY LOSS
            float loss = calculate_crossentropy_loss(&activations[nb_layers - 1], train_dataset.batches[batch_index].labels);
            if (epoch % 1 == 0)
            {
                printf("epoch: %d, batch %d, loss: %f\n", epoch, batch_index, loss);
            }

            if (loss <= epsilon)
            {
                printf("LOSS <= %f at epoch %d, batch %d!\n", epsilon, epoch, batch_index);
                break;
            }

            for (int layer_idx = nb_layers - 1; layer_idx >= 0; layer_idx--)
            {
                // BACKPROPAGATION OF ACTIVATION
                if (layer_idx == nb_layers - 1)
                {
                    softmax_crossentropy_backward(&activations[layer_idx], train_dataset.batches[batch_index].labels_one_hot);
                }
                else
                {
                    relu_backward(&activations[layer_idx], layers[layer_idx].outputs, layers[layer_idx + 1].dinputs);
                }
                // BACKPROPAGATION LAYER
                if (layer_idx > 0)
                {
                    layer_backward(&layers[layer_idx], activations[layer_idx - 1].outputs, activations[layer_idx].dinputs);
                }
                else
                {
                    layer_backward(&layers[layer_idx], train_dataset.batches[batch_index].data, activations[layer_idx].dinputs);
                }

                // UPDATE PARAMETERS
                update_layer_params(&layers[layer_idx], learning_rate);
            }
        }
    }

    // FREE VARIABLES
    for (int layer_idx = 0; layer_idx < nb_layers; layer_idx++)
    {
        destroy_activation(&activations[layer_idx]);
        destroy_layer(&layers[layer_idx]);
    }
    free(layers);
    free(activations);

    destroy_dataset(&train_dataset);
    return 0;
}