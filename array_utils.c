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



error_code init_ndarray(ndarray *arr, int dimension, int shape[], float value){
    arr->dimension = dimension;

    arr->shape = (int*)malloc(dimension * sizeof(int));
    if (arr->shape == NULL) {
        return err;
    }
    int total_size = 1;
    for (int i = 0; i < dimension; i++) {
        arr->shape[i] = shape[i];
        total_size *= shape[i];
    }
    arr->total_size = total_size;

    arr->data = (float*)malloc(total_size * sizeof(float));
    if (arr->data == NULL) {
        free(arr->shape); 
        arr->shape = NULL;
        return err;
    }    

    for (int i = 0; i < total_size; i++) {
        arr->data[i] = value;
    }
    return ok;
}

void set_data_ndarray(ndarray *arr, float values[]){
    for (int i = 0; i < arr->total_size; i++)
    {
        arr->data[i] = values[i];
    }
}

ndarray copy_ndarray(ndarray arr){
    ndarray a;
    init_ndarray(&a, arr.dimension, arr.shape, 0.0);
    set_data_ndarray(&a, arr.data);
    return a;
}

/*
! Designed to work only with 2D array !
*/
ndarray transpose_copy_ndarray(ndarray arr){
    ndarray a;

    int shape[2] = {arr.shape[1], arr.shape[0]};
    init_ndarray(&a, 2, shape, 0.0);

    float* a_values = (float*)malloc(arr.total_size * sizeof(float));
    for (int i = 0; i < arr.shape[0]; i++)
    {
        for (int j = 0; j <  arr.shape[1]; j++)
        {
            a_values[j*arr.shape[0]+i] = arr.data[i*arr.shape[1]+j];
        }
    }
    set_data_ndarray(&a, a_values);
    free(a_values);
    return a;
}

void get_matrix_column(ndarray arr, int col, float* buffer) {
    if (arr.dimension != 2){
        fprintf(stderr, "Error: get_matrix_column only works on 2D arrays.\n");
        exit(1);
    }
    for (int i = 0; i < arr.shape[0]; i++) {
        buffer[i] = arr.data[i * arr.shape[1] + col];
    }
}

float* get_matrix_row(ndarray arr, int row) {
    return arr.data + row * arr.shape[1];
}

ndarray matrix_product(ndarray arr1, ndarray arr2){
    if (arr1.dimension != 2 || arr2.dimension != 2) {
        fprintf(stderr, "Error: matrix product only works on 2D arrays.\n");
        exit(1);
    }

    int m = arr1.shape[0];
    int n = arr1.shape[1];
    int n2 = arr2.shape[0];
    int p = arr2.shape[1];

    if (n != n2) {
        fprintf(stderr, "Error: incompatible shapes [%d x %d] and [%d x %d]\n",
                m, n, n2, p);
        exit(1);
    }

    // Allocate result matrix
    ndarray result;
    result.dimension = 2;
    result.shape = (int*)malloc(2 * sizeof(int));
    result.shape[0] = m;
    result.shape[1] = p;
    result.total_size = m * p;
    result.data = (float*)malloc(result.total_size * sizeof(float));

    // Temporary buffer for columns
    float* col_buffer = (float*)malloc(n * sizeof(float));

    // Compute result
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < p; j++) {
            float* row = get_matrix_row(arr1, i);
            get_matrix_column(arr2, j, col_buffer);
            result.data[i * p + j] = dot_product(row, col_buffer, n);
        }
    }

    free(col_buffer);
    return result;
}

void add_vec_to_matrix(ndarray *arr, float vec[]){
    for (int i = 0; i < arr->shape[0]; i++) {
        for (int j = 0; j < arr->shape[1]; j++) {
            arr->data[i * arr->shape[0] + j] += vec[j];
        }
    }
}

void destroy_ndarray(ndarray *arr){
    free(arr->shape);
    free(arr->data);
}


void print_ndarray_recursive(ndarray arr, int dim, int* indices, int offset) {
    if (dim == arr.dimension - 1) {
        printf("[");
        for (int i = 0; i < arr.shape[dim]; i++) {
            int index = offset + i;
            printf("%.2f", arr.data[index]);
            if (i < arr.shape[dim] - 1) {
                printf(", ");
            }
        }
        printf("]");
    } else {
        printf("[");
        int stride = 1;
        for (int i = dim + 1; i < arr.dimension; i++) {
            stride *= arr.shape[i];
        }
        for (int i = 0; i < arr.shape[dim]; i++) {
            print_ndarray_recursive(arr, dim + 1, indices, offset + i * stride);
            if (i < arr.shape[dim] - 1) {
                printf(", ");
            }
        }
        printf("]");
    }
}

void print_ndarray(const ndarray arr) {
    print_ndarray_recursive(arr, 0, NULL, 0);
    printf("\n");
}

