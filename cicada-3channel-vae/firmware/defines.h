#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 252
#define N_SIZE_0_2 18
#define N_SIZE_1_2 14
#define N_SIZE_2_2 1
#define OUT_HEIGHT_3 9
#define OUT_WIDTH_3 7
#define N_FILT_3 4
#define OUT_HEIGHT_3 9
#define OUT_WIDTH_3 7
#define N_FILT_3 4
#define OUT_HEIGHT_3 9
#define OUT_WIDTH_3 7
#define N_FILT_3 4
#define N_SIZE_0_6 252
#define N_LAYER_7 16
#define N_LAYER_7 16
#define N_LAYER_7 16
#define N_LAYER_10 1
#define N_LAYER_10 1
#define N_LAYER_10 1

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<10,6> input_t;
typedef ap_fixed<30,22> conv_accum_t;
typedef ap_fixed<30,22> layer3_t;
typedef ap_fixed<12,4> weight3_t;
typedef ap_uint<1> bias3_t;
typedef ap_fixed<16,6> layer4_t;
typedef ap_fixed<18,8> conv_linear_table_t;
typedef ap_ufixed<10,6,AP_RND_CONV,AP_SAT> layer5_t;
typedef ap_fixed<18,8> relu0_table_t;
typedef ap_fixed<26,14> dense1_accum_t;
typedef ap_fixed<26,14> layer7_t;
typedef ap_fixed<8,2> weight7_t;
typedef ap_fixed<8,4> bias7_t;
typedef ap_uint<1> layer7_index;
typedef ap_fixed<16,6> layer8_t;
typedef ap_fixed<18,8> dense1_linear_table_t;
typedef ap_ufixed<10,6,AP_RND_CONV,AP_SAT> layer9_t;
typedef ap_fixed<18,8> relu1_table_t;
typedef ap_fixed<26,14> dense2_accum_t;
typedef ap_fixed<26,14> layer10_t;
typedef ap_fixed<12,4> weight10_t;
typedef ap_uint<1> bias10_t;
typedef ap_uint<1> layer10_index;
typedef ap_fixed<16,6> layer11_t;
typedef ap_fixed<18,8> dense2_linear_table_t;
typedef ap_ufixed<16,8,AP_RND_CONV,AP_SAT> result_t;
typedef ap_fixed<18,8> outputs_table_t;

#endif
