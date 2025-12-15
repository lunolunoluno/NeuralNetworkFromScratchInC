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


float calculate_loss_categorical_crossentropy(const ndarray pred, const ndarray label){
    assert(pred.dimension == 2 && label.dimension == 2);
    assert(pred.shape[0] == label.shape[0] && pred.shape[1] == label.shape[1]);
    int n_samples = pred.shape[0];
    int n_classes = pred.shape[1];
    float negative_log_likelihoods = 0.0f;

    for (int i = 0; i < n_samples; i++) {
        float correct_confidences = 0.0f;
        int offset = i * n_classes;  

        for (int j = 0; j < n_classes; j++) {
            correct_confidences += pred.data[offset + j] * label.data[offset + j];
        }

        if (correct_confidences < 1e-15f){
            correct_confidences = 1e-15f; 
        }

        negative_log_likelihoods += -logf(correct_confidences);
    }
    float average_loss = negative_log_likelihoods / n_samples;
    return average_loss;
}

float calculate_accuracy(const ndarray pred, const ndarray label){
    assert(pred.dimension == 2 && label.dimension == 2);
    assert(pred.shape[0] == label.shape[0] && pred.shape[1] == label.shape[1]);
    int n_samples = pred.shape[0];
    int n_classes = pred.shape[1];
    int correct_count = 0;

    for (int i = 0; i < n_samples; i++) {
        int offset = i * n_classes;

        int predicted_class = 0;
        int target_class = 0; 

        float max_prob = pred.data[offset];
        for (int j = 1; j < n_classes; j++) {
            if (pred.data[offset + j] > max_prob) {
                max_prob = pred.data[offset + j];
                predicted_class = j;
            }
            if (label.data[offset + j] > label.data[offset + target_class]) {
                target_class = j;
            }
        }

        if (predicted_class == target_class) {
            correct_count++;
        }
    }

    return (float)correct_count / n_samples;
}
