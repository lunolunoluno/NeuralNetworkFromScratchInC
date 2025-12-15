#ifndef _DATASET_
#define _DATASET_
#include "array_utils.h"

typedef struct dataset
{
    int distribution[3]; // unused so far
    int nb_inputs;
    int nb_data;
    ndarray train;
    ndarray train_labels;
    ndarray test; // unused so far
    ndarray test_labels; // unused so far
    ndarray validate; // unused so far
    ndarray validate_labels; // unused so far
}dataset;

void init_dataset_from_csv(dataset *dataset, char csv_path[], int batch_size, int n_categories);
void destroy_dataset(dataset *dataset);

#endif