#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "dataset.h"

int count_lines_in_file(char *filename)
{
    FILE *fp = fopen(filename, "r");
    int ch = 0;
    int lines = 0;

    if (fp == NULL)
    {
        return 0;
    }
    lines++;
    while ((ch = fgetc(fp)) != EOF)
    {
        if (ch == '\n')
            lines++;
    }
    fclose(fp);
    return lines;
}

// this code will assume that the last column of the csv is the label and the the label is an integer
void create_dataset(dataset *ds, char *csv_path, int nb_inputs, int batch_size, int nb_outputs, char csv_separator)
{
    int nb_lines = count_lines_in_file(csv_path);
    int nb_batches = (nb_lines + batch_size - 1) / batch_size;

    // Init dataset
    ds->nb_batches = nb_batches;
    ds->batches = malloc(nb_batches * sizeof(dataset_batch));
    for (int i = 0; i < nb_batches; i++)
    {
        ds->batches[i].batch_size = batch_size;
        ds->batches[i].nb_inputs = nb_inputs;
        ds->batches[i].labels = calloc(batch_size, sizeof(int));
        if (i == nb_batches - 1)
        {
            for (int j = 0; j < batch_size; j++)
            {
                // set labels of last batch to -1 in case nb_lines and batch_size are not perfectly divisible
                // a sample with a label of -1 will be ignored during processing
                ds->batches[i].labels[j] = -1; 
            }
        }
        ds->batches[i].labels_one_hot = malloc(batch_size * sizeof(int *));
        ds->batches[i].data = malloc(batch_size * sizeof(float *));
        for (int j = 0; j < batch_size; j++)
        {
            ds->batches[i].labels_one_hot[j] = calloc(nb_outputs, sizeof(int));
            ds->batches[i].data[j] = calloc(nb_inputs, sizeof(float));
        }
    }

    FILE *file = fopen(csv_path, "r");
    if (file == NULL)
    {
        fprintf(stderr, "Error: Could not open file %s\n", csv_path);
        return;
    }

    char buffer[2048];
    char sep_str[2] = {csv_separator, '\0'};

    // Read line by line
    int batch_index = 0;
    int sample_index = 0;
    while (fgets(buffer, sizeof(buffer), file))
    {
        buffer[strcspn(buffer, "\n")] = 0;
        buffer[strcspn(buffer, "\r")] = 0;
        if (strlen(buffer) == 0)
            continue;
        
        // Parse the line
        char *token = strtok(buffer, sep_str);
        int col = 0;

        // Extract the inputs
        while (token != NULL && col < nb_inputs) {
            ds->batches[batch_index].data[sample_index][col] = strtof(token, NULL);
            token = strtok(NULL, sep_str);
            col++;
        }

        // Extract the integer label
        if (token != NULL) {
            int label = atoi(token);
            ds->batches[batch_index].labels[sample_index] = label;
            ds->batches[batch_index].labels_one_hot[sample_index][label] = 1;
        } else {
            fprintf(stderr, "Warning: Row %d is missing the label column.\n", (batch_index * batch_size) + sample_index + 1);
            ds->batches[batch_index].labels[sample_index] = -1; // Default fallback
        }
        sample_index++;
        if (sample_index >= batch_size) {
            sample_index = 0;
            batch_index++;
        }
    }
}


void destroy_dataset(dataset *ds)
{
    for (int i = 0; i < ds->nb_batches; i++)
    {
        for (int j = 0; j < ds->batches[i].batch_size; j++)
        {
            free(ds->batches[i].data[j]);
            free(ds->batches[i].labels_one_hot[j]);
        }
        free(ds->batches[i].data);
        free(ds->batches[i].labels_one_hot);
        free(ds->batches[i].labels); 
    }
    free(ds->batches);
}