#include <iostream>

#include "cicada.h"
#include "parameters.h"

void cicada(
    input_t input_main[N_INPUT_1_1], input3_t input_bin[N_INPUT_1_3*N_INPUT_2_3],
    result_t layer15_out[N_LAYER_13]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=input_main complete dim=0
    #pragma HLS ARRAY_RESHAPE variable=input_bin complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer15_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=input_main,input_bin,layer15_out 
    #pragma HLS DATAFLOW 

#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        // hls-fpga-machine-learning insert load weights
        nnet::load_weights_from_txt<weight6_t, 48>(w6, "w6.txt");
        nnet::load_weights_from_txt<bias6_t, 4>(b6, "b6.txt");
        nnet::load_weights_from_txt<weight10_t, 4032>(w10, "w10.txt");
        nnet::load_weights_from_txt<bias10_t, 16>(b10, "b10.txt");
        nnet::load_weights_from_txt<weight13_t, 16>(w13, "w13.txt");
        nnet::load_weights_from_txt<bias13_t, 1>(b13, "b13.txt");
        loaded_weights = true;
    }
#endif

    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    auto& layer2_out = input_main;
    layer4_t layer4_out[OUT_CONCAT_0_4*OUT_CONCAT_1_4];
    #pragma HLS ARRAY_PARTITION variable=layer4_out complete dim=0
    nnet::concatenate2d<input_t, input3_t, layer4_t, config4>(layer2_out, input_bin, layer4_out); // concat_inputs

    auto& layer5_out = layer4_out;
    layer6_t layer6_out[OUT_HEIGHT_6*OUT_WIDTH_6*N_FILT_6];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0
    nnet::conv_2d_cl<layer4_t, layer6_t, config6>(layer5_out, layer6_out, w6, b6); // conv

    layer7_t layer7_out[OUT_HEIGHT_6*OUT_WIDTH_6*N_FILT_6];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0
    nnet::linear<layer6_t, layer7_t, linear_config7>(layer6_out, layer7_out); // conv_linear

    layer8_t layer8_out[OUT_HEIGHT_6*OUT_WIDTH_6*N_FILT_6];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0
    nnet::relu<layer7_t, layer8_t, relu_config8>(layer7_out, layer8_out); // relu0

    auto& layer9_out = layer8_out;
    layer10_t layer10_out[N_LAYER_10];
    #pragma HLS ARRAY_PARTITION variable=layer10_out complete dim=0
    nnet::dense<layer8_t, layer10_t, config10>(layer9_out, layer10_out, w10, b10); // dense1

    layer11_t layer11_out[N_LAYER_10];
    #pragma HLS ARRAY_PARTITION variable=layer11_out complete dim=0
    nnet::linear<layer10_t, layer11_t, linear_config11>(layer10_out, layer11_out); // dense1_linear

    layer12_t layer12_out[N_LAYER_10];
    #pragma HLS ARRAY_PARTITION variable=layer12_out complete dim=0
    nnet::relu<layer11_t, layer12_t, relu_config12>(layer11_out, layer12_out); // relu1

    layer13_t layer13_out[N_LAYER_13];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0
    nnet::dense<layer12_t, layer13_t, config13>(layer12_out, layer13_out, w13, b13); // dense2

    layer14_t layer14_out[N_LAYER_13];
    #pragma HLS ARRAY_PARTITION variable=layer14_out complete dim=0
    nnet::linear<layer13_t, layer14_t, linear_config14>(layer13_out, layer14_out); // dense2_linear

    nnet::relu<layer14_t, result_t, relu_config15>(layer14_out, layer15_out); // outputs

}
