#ifndef _ARRAYUTILS_
#define _ARRAYUTILS_

typedef enum _error_code {
    ok = 0,
    err = 1
} error_code;

typedef struct ndarray
{
    int dimension; // dimension of the data
    int* shape;    // shape of the data
    int total_size; //aka. product of all elem in shape
    float* data;
}ndarray;


void print_float_array(int size, float array[]);
float get_random_float(float min, float max);
float* get_random_array(int size, float min_value, float max_value);
float* init_array(int size, float value);
float dot_product(float a[], float b[], int size);

error_code init_ndarray(ndarray *arr, int dimension, int shape[], float value);
void set_data_ndarray(ndarray *arr, float values[]);
ndarray copy_ndarray(ndarray arr);
ndarray transpose_copy_ndarray(ndarray arr);
ndarray matrix_product(ndarray arr1, ndarray arr2);
void add_vec_to_matrix(ndarray *arr, float vec[]);
void destroy_ndarray(ndarray *arr);
void print_ndarray(const ndarray arr);
#endif