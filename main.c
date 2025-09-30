#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "array_utils.h"

int main(){
    srand((unsigned int)time(NULL));

    float inputs[4] = {1.0, 2.0, 3.0, 2.5};
    int n_inputs = 4;
    int n_neurons = 3;
    ndarray weights;
    int weights_shape[2] = {n_neurons, n_inputs};
    init_ndarray(&weights, 2, weights_shape, 0.0);
    float weights_values[12] = {0.2, 0.8, -0.5, 1.0,
                                0.5, -0.91, 0.26, -0.5,
                                -0.26, -0.27, 0.17, 0.87};
    set_data_ndarray(&weights, weights_values);
    float biases[3] = {2.0, 3.0, 0.5};

    float* layer_outputs = init_array(n_neurons, 0.0);

    for (int n = 0; n < n_neurons; n++)
    {
        for (int i = 0; i < n_inputs; i++)
        {
            layer_outputs[n] += inputs[i]*weights.data[n*n_inputs+i];
        }
        layer_outputs[n] += biases[n];
    }
    
    // [ 4.800000, 1.210000, 2.385000,]
    print_float_array(n_neurons, layer_outputs);

    destroy_ndarray(&weights);
    free(layer_outputs);
    return 0;
}