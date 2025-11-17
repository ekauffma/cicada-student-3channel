#ifndef CICADA_H_
#define CICADA_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "hls_stream.h"

#include "defines.h"

// Prototype of top level function for C-synthesis
void cicada(
    input_t input_main[N_INPUT_1_1],
    result_t layer12_out[N_LAYER_10]
);

#endif
