#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "dataset.h"
#include "layer.h"
#include "utils.h"

#define BATCH_SIZE 7
#define INPUT_SIZE 2
#define LAYER1_NB_NEURONS 3
#define LAYER2_NB_NEURONS 3 // this is also the output

int main()
{
    dataset train_dataset;
    create_dataset(&train_dataset, "./100_samples_3_cat.csv", INPUT_SIZE, BATCH_SIZE, ',');

    for (int i = 0; i < train_dataset.nb_batches; i++)
    {
        printf("BATCH %d\n", i);
        for (int j = 0; j < train_dataset.batches[i].batch_size; j++)
        {
            printf("sample %d: [", j);
            for (int k = 0; k < train_dataset.batches[i].nb_inputs; k++)
            {
                printf("%f, ", train_dataset.batches[i].data[j][k]);
            }
            printf("] => %d\n", train_dataset.batches[i].labels[j]);
        }
    }

    // INIT VARIABLES

    // float **batch = malloc(BATCH_SIZE * sizeof(float *));
    // float **label_one_hot = malloc(BATCH_SIZE * sizeof(float *));
    // for (int i = 0; i < BATCH_SIZE; i++)
    // {
    //     batch[i] = calloc(INPUT_SIZE, sizeof(float));
    //     label_one_hot[i] = calloc(LAYER2_NB_NEURONS, sizeof(float));
    // }
    // int *labels = calloc(BATCH_SIZE, sizeof(int));

    // layer_params layer1;
    // activation_params layer1_relu;

    // layer_params layer2;
    // activation_params layer2_softmax;

    // // GIVE VARIABLES INITIAL VALUES

    // int nb_epochs = 100000;
    // float learning_rate = 1.0;
    // float epsilon = 1e-7;
    // batch[0][0] = -0.8326189893369458;
    // batch[0][1] = -0.5538462048218106;
    // batch[1][0] = 0.9989373586840176;
    // batch[1][1] = 0.046088538980948564;
    // batch[2][0] = -0.6360447021800988;
    // batch[2][1] = 0.7716522123525789;

    // labels[0] = 0;
    // labels[1] = 1;
    // labels[2] = 2;

    // for (int i = 0; i < BATCH_SIZE; i++)
    // {
    //     label_one_hot[i][labels[i]] = 1.0;
    // }

    // init_layer(INPUT_SIZE, LAYER1_NB_NEURONS, BATCH_SIZE, &layer1);
    // init_activation(LAYER1_NB_NEURONS, BATCH_SIZE, &layer1_relu);

    // init_layer(LAYER1_NB_NEURONS, LAYER2_NB_NEURONS, BATCH_SIZE, &layer2);
    // init_activation(LAYER2_NB_NEURONS, BATCH_SIZE, &layer2_softmax);

    // printf("BATCH:\n");
    // for (int b = 0; b < BATCH_SIZE; b++)
    // {
    //     for (int i = 0; i < INPUT_SIZE; i++)
    //     {
    //         printf("%f, ", batch[b][i]);
    //     }
    //     printf("\n");
    // }

    // printf("ONE-HOT LABEL:\n");
    // for (int b = 0; b < BATCH_SIZE; b++)
    // {
    //     for (int i = 0; i < LAYER2_NB_NEURONS; i++)
    //     {
    //         printf("%f, ", label_one_hot[b][i]);
    //     }
    //     printf("\n");
    // }

    // for (int epoch = 0; epoch < nb_epochs; epoch++)
    // {
    //     // FEED FORWARD LAYER 1
    //     layer_forward(&layer1, batch);

    //     // LAYER 1 ReLU
    //     relu_forward(&layer1_relu, layer1.outputs);

    //     // FEED FORWARD LAYER 2
    //     layer_forward(&layer2, layer1_relu.outputs);

    //     // LAYER 2 SOFTMAX
    //     softmax_forward(&layer2_softmax, layer2.outputs);

    //     // CALCULATE CATEGORICAL CROSS-ENTROPY LOSS
    //     float loss = calculate_crossentropy_loss(&layer2_softmax, labels);
    //     if (epoch % 100 == 0)
    //     {
    //         printf("epoch: %d, loss: %f\n", epoch, loss);
    //     }

    //     if (loss <= epsilon)
    //     {
    //         printf("LOSS <= %f at epoch %d!\n", epsilon, epoch);
    //         break;
    //     }

    //     // BACKPROPAGATION OF SOFTMAX + CROSS-ENTROPY LOSS (easier to implement)
    //     softmax_crossentropy_backward(&layer2_softmax, label_one_hot);

    //     // BACKPROPAGATION LAYER 2
    //     layer_backward(&layer2, layer1_relu.outputs, layer2_softmax.dinputs);

    //     // BACKPROPAGATION LAYER 1 RELU
    //     relu_backward(&layer1_relu, layer1.outputs, layer2.dinputs);

    //     // BACKPROPAGATION LAYER 1
    //     layer_backward(&layer1, batch, layer1_relu.dinputs);

    //     // UPDATE PARAMETERS
    //     // Layer 1
    //     update_layer_params(&layer1, learning_rate);
    //     // Layer 2
    //     update_layer_params(&layer2, learning_rate);
    // }

    // layer_forward(&layer1, batch);

    // relu_forward(&layer1_relu, layer1.outputs);

    // layer_forward(&layer2, layer1_relu.outputs);

    // softmax_forward(&layer2_softmax, layer2.outputs);
    // printf("NETWORK PREDICTION:\n");
    // for (int b = 0; b < BATCH_SIZE; b++)
    // {
    //     for (int i = 0; i < LAYER2_NB_NEURONS; i++)
    //     {
    //         printf("%f,", layer2_softmax.outputs[b][i]);
    //     }
    //     printf("\n");
    // }

    // float loss = calculate_crossentropy_loss(&layer2_softmax, labels);
    // printf("FINAL LOSS: %f\n", loss);

    // // FREE VARIABLES

    // destroy_activation(&layer2_softmax);
    // destroy_layer(&layer2);

    // destroy_activation(&layer1_relu);
    // destroy_layer(&layer1);

    // free(labels);
    // for (int i = 0; i < BATCH_SIZE; i++)
    // {
    //     free(batch[i]);
    //     free(label_one_hot[i]);
    // }
    // free(label_one_hot);
    // free(batch);
    destroy_dataset(&train_dataset);
    return 0;
}