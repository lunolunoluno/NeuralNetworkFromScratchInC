#ifndef _FUNCTION_
#define _FUNCTION_
#include "array_utils.h"
#include <stdlib.h>
#include <math.h>
#include <assert.h>

float relu_forward(float input);
ndarray relu_forward_ndarray(const ndarray arr);

ndarray softmax_forward_vector(const ndarray arr);

#endif