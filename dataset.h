#ifndef _DATASET_
#define _DATASET_

typedef struct dataset_batch
{
    int nb_inputs;
    int batch_size;
    float **data;
    int *labels;
} dataset_batch;

typedef struct dataset
{
    int nb_batches;
    dataset_batch *batches;
} dataset;

void create_dataset(dataset *ds, char *csv_path, int nb_inputs, int batch_size, char csv_separator);
void destroy_dataset(dataset *ds);

#endif