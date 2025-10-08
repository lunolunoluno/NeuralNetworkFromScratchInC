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

ndarray softmax_forward_vector(const ndarray arr){
    assert(arr.dimension == 1);
    // get the max value and sum
    float max_value = arr.data[0];
    for (int i = 1; i < arr.total_size; i++)
    {
        if (arr.data[i] > max_value)
            max_value = arr.data[i];
    }
    
    // Compute exponentials and their sum
    float *exp_values = (float *)malloc(arr.total_size * sizeof(float));
    float sum_exp = 0.0;
    for (int i = 0; i < arr.total_size; i++)
    {
        exp_values[i] = exp(arr.data[i] - max_value);
        sum_exp += exp_values[i];
    }

    // normalize probabilities
    float *probabilities = (float *)malloc(arr.total_size * sizeof(float));
    for (int i = 0; i < arr.total_size; i++)
    {
        probabilities[i] = exp_values[i] / sum_exp;
    }

    ndarray out;
    init_ndarray(&out, 1, arr.shape, 0.0);
    set_data_ndarray(&out, probabilities);
    free(exp_values);
    free(probabilities);
    return out;
}