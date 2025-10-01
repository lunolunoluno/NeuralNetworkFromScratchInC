#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "array_utils.h"

int main(){
    srand((unsigned int)time(NULL));

    int n_inputs = 4;
    int n_batch = 3;
    int n_neurons = 3;

    ndarray inputs;
    int inputs_shape[2] = {n_batch, n_inputs};
    init_ndarray(&inputs, 2, inputs_shape, 0.0);
    float inputs_values[12] = {1.0, 2.0, 3.0, 2.5, 
                               2.0, 5.0, -1.0, 2.0, 
                               -1.5, 2.7, 3.3, -0.8};
    set_data_ndarray(&inputs, inputs_values);

    ndarray weights;
    int weights_shape[2] = {n_neurons, n_inputs};
    init_ndarray(&weights, 2, weights_shape, 0.0);
    float weights_values[12] = {0.2, 0.8, -0.5, 1.0,
                                0.5, -0.91, 0.26, -0.5,
                                -0.26, -0.27, 0.17, 0.87};
    set_data_ndarray(&weights, weights_values);
    float biases[3] = {2.0, 3.0, 0.5};

    ndarray weights_T = transpose_copy_ndarray(weights);
    ndarray outputs = matrix_product(inputs, weights_T);//X * W
    add_vec_to_matrix(&outputs, biases);//+ biases

    print_ndarray(outputs);

    destroy_ndarray(&weights);
    destroy_ndarray(&weights_T);
    destroy_ndarray(&inputs);
    destroy_ndarray(&outputs);

    return 0;
}