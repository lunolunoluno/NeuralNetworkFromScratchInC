#include <stdlib.h>
#include "utils.h"

void set_2darray_value(float **array2d, int nb_col, int nb_row, float *values)
{
    for (int i = 0; i < nb_row; i++)
    {
        for (int j = 0; j < nb_col; j++)
        {
            array2d[i][j] = values[(i * nb_col) + j];
        }
    }
}

float get_random_float(float min, float max){
    return min + (float)((double)rand() / (double)RAND_MAX) * (max - min);
}

float *get_random_array(int size, float min_value, float max_value)
{
    float *array = calloc(size, sizeof(float));
    for (int i = 0; i < size; i++)
    {
        array[i] = get_random_float(min_value, max_value);
    }
    return array;
}