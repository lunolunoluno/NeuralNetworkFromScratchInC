#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "array_utils.h"
#include "layer.h"
#include "dataset.h"
#include "function.h"

int main(){

    int batch_size = 2;
    dataset dataset;
    char csv_path[] = "./test.csv";
    init_dataset_from_csv(&dataset, csv_path, batch_size);

    int n_inputs = dataset.train.shape[2];
    int n_batches = dataset.train.shape[0];
    int n_neurons_l1 = 3;
    int n_neurons_l2 = 3;

    // create layer 1
    layer_dense layer1;
    layer_init(&layer1, n_neurons_l1, n_inputs); 

    // create layer 2
        layer_dense layer2;
        layer_init(&layer2, n_neurons_l2, n_neurons_l1); 

    for (int i = 0; i < n_batches; i++)
    {  
        // get batch
        ndarray batch;
        extract_subarray(&batch, &dataset.train, i);
        printf("===============================\n");
        printf("Batch %d/%d:",i+1, n_batches);
        print_ndarray(batch);

        // calculate output for layer 1   
        layer_forward(&layer1, batch);

        printf("Layer 1 output:\n");
        print_ndarray(layer1.outputs);

        ndarray layer1_outputs_relu = relu_forward_ndarray(layer1.outputs);
        printf("Layer 1 ReLU:\n");
        print_ndarray(layer1_outputs_relu);

        // calculate output for layer 2   
        layer_forward(&layer2, copy_ndarray(layer1_outputs_relu)); // important to make a copy to avoid double-free

        printf("Layer 2 output:\n");
        print_ndarray(layer2.outputs);

        ndarray layer2_outputs_relu = relu_forward_ndarray(layer2.outputs);
        printf("Layer 2 ReLU:\n");
        print_ndarray(layer2_outputs_relu);

        destroy_ndarray(&layer1_outputs_relu);
        destroy_ndarray(&layer2_outputs_relu);
    }

    print_ndarray(dataset.train_labels);

    // destroy allocated variables
    destroy_layer(&layer1);
    destroy_layer(&layer2);
    destroy_dataset(&dataset);

    return 0;
}