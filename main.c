#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
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

        ndarray layer2_outputs_softmax = copy_ndarray(layer2.outputs);
        float *softmax_values = (float *)calloc(layer2.outputs.total_size, sizeof(float));
        for (int j = 0; j < layer2.outputs.shape[0]; j++)
        {
            ndarray batch_row;
            extract_subarray(&batch_row, &layer2.outputs, j);
            ndarray batch_row_softmax = softmax_forward_vector(batch_row);
            
            memcpy(&softmax_values[j * batch_row_softmax.total_size], batch_row_softmax.data, batch_row_softmax.total_size * sizeof(float));
            
            destroy_ndarray(&batch_row);
            destroy_ndarray(&batch_row_softmax);
        }

        set_data_ndarray(&layer2_outputs_softmax, softmax_values);
        free(softmax_values);
        
        printf("Layer 2 Softmax:\n");
        print_ndarray(layer2_outputs_softmax);

        destroy_ndarray(&layer1_outputs_relu);
        destroy_ndarray(&layer2_outputs_softmax);
    }

    print_ndarray(dataset.train_labels);

    // destroy allocated variables
    destroy_layer(&layer1);
    destroy_layer(&layer2);
    destroy_dataset(&dataset);

    return 0;
}