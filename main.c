#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "array_utils.h"

int main(){
    srand((unsigned int)time(NULL));

    // float inputs[4] = {1.0, 2.0, 3.0, 2.5};
    
    // float weights1[4] = {0.2, 0.8, -0.5, 1};
    // float weights2[4] = {0.5, -0.91, 0.26, -0.5};
    // float weights3[4] = {-0.26, -0.27, 0.17, 0.87};

    // float bias1 = 2.0;
    // float bias2 = 3.0;
    // float bias3 = 0.5;

    // float output[3] = {
    //     // Neuron 1
    //     inputs[0]*weights1[0] +
    //     inputs[1]*weights1[1] +
    //     inputs[2]*weights1[2] +
    //     inputs[3]*weights1[3] + bias1,
    //     // Neuron 2
    //     inputs[0]*weights2[0] +
    //     inputs[1]*weights2[1] +
    //     inputs[2]*weights2[2] +
    //     inputs[3]*weights2[3] + bias2,
    //     // Neuron 3
    //     inputs[0]*weights3[0] +
    //     inputs[1]*weights3[1] +
    //     inputs[2]*weights3[2] +
    //     inputs[3]*weights3[3] + bias3
    // }; 

    // // [ 4.800000, 1.210000, 2.385000,]
    // print_float_array(3, output);


    ndarray t;
    int shape[3] = {2, 3, 2}; 
    init_ndarray(&t, 3, shape, 0.0);
    
    float *values = get_random_array(t.total_size, 0.0, 1.0);
    set_data_ndarray(&t, values);

    print_ndarray(t);

    free(values);
    destroy_ndarray(&t);
    return 0;
}