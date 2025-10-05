#include "dataset.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <sys/types.h>

int _countlines_in_file(char *filename){                                    
    FILE *fp = fopen(filename,"r");
    int ch=0;
    int lines=0;

    if (fp == NULL){
        return 0;
    }
    lines++;
    while ((ch = fgetc(fp)) != EOF){
        if (ch == '\n')
        lines++;
    }
    fclose(fp);
    return lines;
}

/*
Will load the data from a CSV into a dataset struct.
The last column of the CSV will be used as label while the other ones will serve as inputs.
This code will assume that the first line of the CSV file are the name of the columns.
*/
void init_dataset_from_csv(dataset *dataset, char csv_path[], int batch_size){
    // get the number of lines in the csv file
    int nb_lines = _countlines_in_file(csv_path);
    dataset->nb_data = nb_lines - 1; // -1 because of the header

    // go through the csv file
    FILE* fp = fopen(csv_path,"r"); 
    assert(NULL != fp);
    size_t len = 0;
    ssize_t read; 
    char * line = NULL;
    int line_index = 0;
    float *dataset_values;
    float *label_values;
    int dataset_values_index = 0;
    int num_columns = 1;
    while ((read = getline(&line, &len, fp)) != -1) {
        if (line_index == 0)
        {
            // get the number of column in the CSV
            if (line[read - 1] == '\n'){
                line[read - 1] = '\0';
            }
            for (char *p = line; *p; p++) {
                if (*p == ',')
                    num_columns++;
            }
            assert(num_columns > 1); // need at least 1 input and 1 label for the code to work
            dataset->nb_inputs = num_columns-1;

            // init the train dataset with default values
            int dataset_shape[3] = {
                (dataset->nb_data + batch_size - 1) / batch_size, // number of batches
                batch_size, // size of one batch
                dataset->nb_inputs // all the inputs 
            };
            init_ndarray(&dataset->train, 3, dataset_shape, 0.0);
            dataset_values = (float*)malloc(dataset->train.total_size * sizeof(float));

            int train_label_shape[2] = {
                (dataset->nb_data + batch_size - 1) / batch_size, // number of batches
                batch_size
            };
            init_ndarray(&dataset->train_labels, 2, train_label_shape, 0.0);
            label_values = (float*)malloc(dataset->train_labels.total_size * sizeof(float));
        }else{
            // read each column and add the value to the train dataset
            char *token = strtok(line, ",");
            int col_index = 0;
            while (token != NULL && col_index < num_columns) {
                float value = atof(token); 
                if (col_index < num_columns - 1)
                {
                    dataset_values[dataset_values_index] = value;
                    dataset_values_index++;
                }else{
                    label_values[line_index-1] = value;
                }
                
                token = strtok(NULL, ",");
                col_index++;
            }
        }
        line_index++;
    }   

    // set the value of the train dataset
    set_data_ndarray(&dataset->train, dataset_values);
    set_data_ndarray(&dataset->train_labels, label_values);

    // cleanup
    fclose(fp);
    if (line){
        free(line);
    }
    free(dataset_values);
    free(label_values);

    // init test and validate dataset with default values
    int shape[2] = {1,1};
    init_ndarray(&dataset->test, 2, shape, 0.0);
    init_ndarray(&dataset->test_labels, 2, shape, 0.0);
    init_ndarray(&dataset->validate, 2, shape, 0.0);
    init_ndarray(&dataset->validate_labels, 2, shape, 0.0);
}

void destroy_dataset(dataset *dataset){
    destroy_ndarray(&dataset->train);
    destroy_ndarray(&dataset->train_labels);
    destroy_ndarray(&dataset->test);
    destroy_ndarray(&dataset->test_labels);
    destroy_ndarray(&dataset->validate);
    destroy_ndarray(&dataset->validate_labels);
}
