#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "dataset.h"
#include "layer.h"
#include "utils.h"

#define BATCH_SIZE 7
#define INPUT_SIZE 2
#define LAYER1_NB_NEURONS 3
#define LAYER2_NB_NEURONS 3 // this is also the output

int main()
{
    srand((unsigned int)time(NULL));
    // INIT VARIABLES

    dataset train_dataset;

    layer_params layer1;
    activation_params layer1_relu;

    layer_params layer2;
    activation_params layer2_softmax;

    // GIVE VARIABLES INITIAL VALUES
    int nb_epochs = 100;
    float learning_rate = 0.01;
    float epsilon = 1e-7;

    create_dataset(&train_dataset, "./100_samples_3_cat_shuffled.csv", INPUT_SIZE, BATCH_SIZE, LAYER2_NB_NEURONS, ',');

    // for (int i = 0; i < train_dataset.nb_batches; i++)
    // {
    //     printf("BATCH %d\n", i);
    //     for (int j = 0; j < train_dataset.batches[i].batch_size; j++)
    //     {
    //         printf("sample %d: [", j);
    //         for (int k = 0; k < train_dataset.batches[i].nb_inputs; k++)
    //         {
    //             printf("%f, ", train_dataset.batches[i].data[j][k]);
    //         }
    //         printf("] => %d\n", train_dataset.batches[i].labels[j]);
    //     }
    // }

    init_layer(INPUT_SIZE, LAYER1_NB_NEURONS, BATCH_SIZE, &layer1);
    init_activation(LAYER1_NB_NEURONS, BATCH_SIZE, &layer1_relu);

    // printf("LAYER 1:\nWeights:\n");
    // for (int i = 0; i < layer1.nb_neurons; i++)
    // {
    //     for (int j = 0; j < layer1.nb_inputs; j++)
    //     {
    //         printf("%f,", layer1.weights[i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("Biases:\n");
    // for (int i = 0; i < layer1.nb_neurons; i++)
    // {
    //     printf("%f,", layer1.biases[i]);
    // }
    // printf("\n");    

    init_layer(LAYER1_NB_NEURONS, LAYER2_NB_NEURONS, BATCH_SIZE, &layer2);
    init_activation(LAYER2_NB_NEURONS, BATCH_SIZE, &layer2_softmax);

    // printf("LAYER 2:\nWeights:\n");
    // for (int i = 0; i < layer2.nb_neurons; i++)
    // {
    //     for (int j = 0; j < layer2.nb_inputs; j++)
    //     {
    //         printf("%f,", layer2.weights[i][j]);
    //     }
    //     printf("\n");
    // }
    // printf("Biases:\n");
    // for (int i = 0; i < layer2.nb_neurons; i++)
    // {
    //     printf("%f,", layer2.biases[i]);
    // }
    // printf("\n");    


    // TRAIN NETWORK
    for (int epoch = 0; epoch < nb_epochs; epoch++)
    {
        for (int batch_index = 0; batch_index < train_dataset.nb_batches; batch_index++)
        {
            // printf("BATCH %d\n", batch_index);
            // for (int j = 0; j < train_dataset.batches[batch_index].batch_size; j++)
            // {
            //     printf("sample %d: inputs [", j);
            //     for (int k = 0; k < train_dataset.batches[batch_index].nb_inputs; k++)
            //     {
            //         printf("%f, ", train_dataset.batches[batch_index].data[j][k]);
            //     }
            //     printf("], label: %d\n", train_dataset.batches[batch_index].labels[j]);
            // }

            // FEED FORWARD LAYER 1
            layer_forward(&layer1, train_dataset.batches[batch_index].data);

            // LAYER 1 ReLU
            relu_forward(&layer1_relu, layer1.outputs);

            // FEED FORWARD LAYER 2
            layer_forward(&layer2, layer1_relu.outputs);

            // LAYER 2 SOFTMAX
            softmax_forward(&layer2_softmax, layer2.outputs);

            // printf("OUTPUT:\n"); 
            // for (int i = 0; i < train_dataset.batches[batch_index].batch_size; i++)
            // {
            //     printf("sample %d, softmax output: [%f, %f, %f]\n", i, layer2_softmax.outputs[i][0], layer2_softmax.outputs[i][1], layer2_softmax.outputs[i][2]);
            // }

            // CALCULATE CATEGORICAL CROSS-ENTROPY LOSS
            float loss = calculate_crossentropy_loss(&layer2_softmax, train_dataset.batches[batch_index].labels);
            if (epoch % 1 == 0)
            {
                printf("epoch: %d, batch %d, loss: %f\n", epoch, batch_index, loss);
            }

            if (loss <= epsilon)
            {
                printf("LOSS <= %f at epoch %d, batch %d!\n", epsilon, epoch, batch_index);
                break;
            }

            // BACKPROPAGATION OF SOFTMAX + CROSS-ENTROPY LOSS (easier to implement)
            softmax_crossentropy_backward(&layer2_softmax, train_dataset.batches[batch_index].labels_one_hot);

            // BACKPROPAGATION LAYER 2
            layer_backward(&layer2, layer1_relu.outputs, layer2_softmax.dinputs);

            // BACKPROPAGATION LAYER 1 RELU
            relu_backward(&layer1_relu, layer1.outputs, layer2.dinputs);

            // BACKPROPAGATION LAYER 1
            layer_backward(&layer1, train_dataset.batches[batch_index].data, layer1_relu.dinputs);

            // UPDATE PARAMETERS
            // Layer 1
            update_layer_params(&layer1, learning_rate);
            // Layer 2
            update_layer_params(&layer2, learning_rate);
            
            // printf("UPDATED LAYER 1:\nWeights:\n");
            // for (int i = 0; i < layer1.nb_neurons; i++)
            // {
            //     for (int j = 0; j < layer1.nb_inputs; j++)
            //     {
            //         printf("%f,", layer1.weights[i][j]);
            //     }
            //     printf("\n");
            // }
            // printf("Biases:\n");
            // for (int i = 0; i < layer1.nb_neurons; i++)
            // {
            //     printf("%f,", layer1.biases[i]);
            // }
            // printf("\n");   
            // printf("UPDATED LAYER 2:\nWeights:\n");
            // for (int i = 0; i < layer2.nb_neurons; i++)
            // {
            //     for (int j = 0; j < layer2.nb_inputs; j++)
            //     {
            //         printf("%f,", layer2.weights[i][j]);
            //     }
            //     printf("\n");
            // }
            // printf("Biases:\n");
            // for (int i = 0; i < layer2.nb_neurons; i++)
            // {
            //     printf("%f,", layer2.biases[i]);
            // }
            // printf("\n");   
            // break;
        }
    }

    // FREE VARIABLES

    destroy_activation(&layer2_softmax);
    destroy_layer(&layer2);

    destroy_activation(&layer1_relu);
    destroy_layer(&layer1);

    destroy_dataset(&train_dataset);
    return 0;
}