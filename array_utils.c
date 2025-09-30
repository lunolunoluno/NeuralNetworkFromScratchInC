#include <stdlib.h>
#include <stdio.h>
#include "array_utils.h"

void print_float_array(int size, float array[]){
    printf("[");
    for(int i = 0;i < size; i++){
        printf(" %f,",array[i]);
    }
    printf("]\n");
}

float get_random_float(float min, float max){
    return min + (float)((double)rand() / (double)RAND_MAX) * (max - min);
}

float* get_random_array(int size, float min_value, float max_value){
    float *array = (float*)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++)
    {
        array[i] = get_random_float(min_value, max_value);
    }    
    return array;
}

float* init_array(int size, float value){
    float *array = (float*)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++)
    {
        array[i] = value;
    }    
    return array;
}

float dot_product(float a[], float b[], int size){
    float res = 0;
    for (int i = 0; i < size; i++)
    {
        res += a[i] * b[i];
    }
    return res;
}



error_code init_ndarray(ndarray *t, int dimension, int shape[], float value){
    t->dimension = dimension;

    t->shape = (int*)malloc(dimension * sizeof(int));
    if (t->shape == NULL) {
        return err;
    }
    int total_size = 1;
    for (int i = 0; i < dimension; i++) {
        t->shape[i] = shape[i];
        total_size *= shape[i];
    }
    t->total_size = total_size;

    t->data = (float*)malloc(total_size * sizeof(float));
    if (t->data == NULL) {
        free(t->shape); 
        t->shape = NULL;
        return err;
    }    

    for (int i = 0; i < total_size; i++) {
        t->data[i] = value;
    }
    return ok;
}

void set_data_ndarray(ndarray *t, float values[]){
    for (int i = 0; i < t->total_size; i++)
    {
        t->data[i] = values[i];
    }
}

void destroy_ndarray(ndarray *t){
    free(t->shape);
    free(t->data);
}







void print_ndarray_recursive(ndarray t, int dim, int* indices, int offset) {
    if (dim == t.dimension - 1) {
        printf("[");
        for (int i = 0; i < t.shape[dim]; i++) {
            int index = offset + i;
            printf("%.2f", t.data[index]);
            if (i < t.shape[dim] - 1) {
                printf(", ");
            }
        }
        printf("]");
    } else {
        printf("[");
        int stride = 1;
        for (int i = dim + 1; i < t.dimension; i++) {
            stride *= t.shape[i];
        }
        for (int i = 0; i < t.shape[dim]; i++) {
            print_ndarray_recursive(t, dim + 1, indices, offset + i * stride);
            if (i < t.shape[dim] - 1) {
                printf(", ");
            }
        }
        printf("]");
    }
}

void print_ndarray(const ndarray t) {
    print_ndarray_recursive(t, 0, NULL, 0);
    printf("\n");
}

