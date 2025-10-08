#include "function.h"

float relu_forward(float input){
    return (input > 0)?input:0;
}

ndarray relu_forward_ndarray(const ndarray arr){
    ndarray res;
    init_ndarray(&res, arr.dimension, arr.shape, 0.0);
    float *relu_values = (float*)malloc(arr.total_size * sizeof(float));
    for (int i = 0; i < arr.total_size; i++)
    {
        relu_values[i] = relu_forward(arr.data[i]);
    }
    set_data_ndarray(&res, relu_values);

    free(relu_values);
    return res;
}
