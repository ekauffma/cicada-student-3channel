#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <cstddef>
#include <cstdio>

// hls-fpga-machine-learning insert numbers
#define N_INPUT_1_1 252
#define N_SIZE_0_2 252
#define N_SIZE_1_2 1
#define N_INPUT_1_3 252
#define N_INPUT_2_3 2
#define OUT_CONCAT_0_4 252
#define OUT_CONCAT_1_4 3
#define N_SIZE_0_5 18
#define N_SIZE_1_5 14
#define N_SIZE_2_5 3
#define OUT_HEIGHT_6 9
#define OUT_WIDTH_6 7
#define N_FILT_6 4
#define OUT_HEIGHT_6 9
#define OUT_WIDTH_6 7
#define N_FILT_6 4
#define OUT_HEIGHT_6 9
#define OUT_WIDTH_6 7
#define N_FILT_6 4
#define N_SIZE_0_9 252
#define N_LAYER_10 16
#define N_LAYER_10 16
#define N_LAYER_10 16
#define N_LAYER_10 16
#define N_LAYER_14 1
#define N_LAYER_14 1
#define N_LAYER_14 1

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<10,6> input_t;
typedef ap_fixed<1,1> input3_t;
typedef ap_fixed<16,6> layer4_t;
typedef ap_fixed<30,22> conv_replacement_accum_t;
typedef ap_fixed<30,22> layer6_t;
typedef ap_fixed<12,4> weight6_t;
typedef ap_uint<1> bias6_t;
typedef ap_fixed<16,6> layer7_t;
typedef ap_fixed<18,8> conv_replacement_linear_table_t;
typedef ap_ufixed<10,6,AP_RND_CONV,AP_SAT> layer8_t;
typedef ap_fixed<18,8> relu0_table_t;
typedef ap_fixed<26,14> dense1_accum_t;
typedef ap_fixed<26,14> layer10_t;
typedef ap_fixed<8,2> weight10_t;
typedef ap_fixed<8,4> bias10_t;
typedef ap_uint<1> layer10_index;
typedef ap_fixed<16,6> layer11_t;
typedef ap_fixed<18,8> dense1_linear_table_t;
typedef ap_fixed<16,6> layer12_t;
typedef ap_fixed<16,6> bn1_scale_t;
typedef ap_fixed<16,6> bn1_bias_t;
typedef ap_ufixed<10,6,AP_RND_CONV,AP_SAT> layer13_t;
typedef ap_fixed<18,8> relu1_table_t;
typedef ap_fixed<26,14> dense2_accum_t;
typedef ap_fixed<26,14> layer14_t;
typedef ap_fixed<12,4> weight14_t;
typedef ap_uint<1> bias14_t;
typedef ap_uint<1> layer14_index;
typedef ap_fixed<16,6> layer15_t;
typedef ap_fixed<18,8> dense2_linear_table_t;
typedef ap_ufixed<16,8,AP_RND_CONV,AP_SAT> result_t;
typedef ap_fixed<18,8> outputs_table_t;

#endif
