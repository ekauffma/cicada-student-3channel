#ifndef CICADA_H_
#define CICADA_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void cicada(
    input_t input_main[N_INPUT_1_1], input3_t input_bin[N_INPUT_1_3*N_INPUT_2_3],
    result_t layer15_out[N_LAYER_13]
);

#endif
