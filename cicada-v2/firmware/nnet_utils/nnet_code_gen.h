#ifndef NNET_INSTR_GEN_H_
#define NNET_INSTR_GEN_H_

#include "nnet_helpers.h"
#include <iostream>

namespace nnet {

template <class data_T, typename CONFIG_T> class FillConv1DBuffer {
  public:
    static void fill_buffer(data_T data[CONFIG_T::in_width * CONFIG_T::n_chan],
                            data_T buffer[CONFIG_T::n_pixels][CONFIG_T::filt_width * CONFIG_T::n_chan],
                            const unsigned partition) {
        // To be implemented in subclasses
    }
};

template <class data_T, typename CONFIG_T> class FillConv2DBuffer {
  public:
    static void
    fill_buffer(data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
                data_T buffer[CONFIG_T::n_pixels][CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan],
                const unsigned partition) {
        // To be implemented in subclasses
    }
};

// hls4ml insert code
template<class data_T, typename CONFIG_T>
class fill_buffer_3 : public FillConv2DBuffer<data_T, CONFIG_T> {
    public:
    static void fill_buffer(
        data_T data[CONFIG_T::in_height * CONFIG_T::in_width * CONFIG_T::n_chan],
        data_T buffer[CONFIG_T::n_pixels][CONFIG_T::filt_height * CONFIG_T::filt_width * CONFIG_T::n_chan],
        const unsigned partition
    ) {
        if (partition ==   0) {
            buffer[0][0] =    data[0]; buffer[0][1] =    data[1]; buffer[0][2] =   data[14]; buffer[0][3] =   data[15];
            buffer[1][0] =    data[2]; buffer[1][1] =    data[3]; buffer[1][2] =   data[16]; buffer[1][3] =   data[17];
            buffer[2][0] =    data[4]; buffer[2][1] =    data[5]; buffer[2][2] =   data[18]; buffer[2][3] =   data[19];
            buffer[3][0] =    data[6]; buffer[3][1] =    data[7]; buffer[3][2] =   data[20]; buffer[3][3] =   data[21];
            buffer[4][0] =    data[8]; buffer[4][1] =    data[9]; buffer[4][2] =   data[22]; buffer[4][3] =   data[23];
            buffer[5][0] =   data[10]; buffer[5][1] =   data[11]; buffer[5][2] =   data[24]; buffer[5][3] =   data[25];
            buffer[6][0] =   data[12]; buffer[6][1] =   data[13]; buffer[6][2] =   data[26]; buffer[6][3] =   data[27];
            buffer[7][0] =   data[28]; buffer[7][1] =   data[29]; buffer[7][2] =   data[42]; buffer[7][3] =   data[43];
            buffer[8][0] =   data[30]; buffer[8][1] =   data[31]; buffer[8][2] =   data[44]; buffer[8][3] =   data[45];
            buffer[9][0] =   data[32]; buffer[9][1] =   data[33]; buffer[9][2] =   data[46]; buffer[9][3] =   data[47];
            buffer[10][0] =   data[34]; buffer[10][1] =   data[35]; buffer[10][2] =   data[48]; buffer[10][3] =   data[49];
            buffer[11][0] =   data[36]; buffer[11][1] =   data[37]; buffer[11][2] =   data[50]; buffer[11][3] =   data[51];
            buffer[12][0] =   data[38]; buffer[12][1] =   data[39]; buffer[12][2] =   data[52]; buffer[12][3] =   data[53];
            buffer[13][0] =   data[40]; buffer[13][1] =   data[41]; buffer[13][2] =   data[54]; buffer[13][3] =   data[55];
            buffer[14][0] =   data[56]; buffer[14][1] =   data[57]; buffer[14][2] =   data[70]; buffer[14][3] =   data[71];
            buffer[15][0] =   data[58]; buffer[15][1] =   data[59]; buffer[15][2] =   data[72]; buffer[15][3] =   data[73];
            buffer[16][0] =   data[60]; buffer[16][1] =   data[61]; buffer[16][2] =   data[74]; buffer[16][3] =   data[75];
            buffer[17][0] =   data[62]; buffer[17][1] =   data[63]; buffer[17][2] =   data[76]; buffer[17][3] =   data[77];
            buffer[18][0] =   data[64]; buffer[18][1] =   data[65]; buffer[18][2] =   data[78]; buffer[18][3] =   data[79];
            buffer[19][0] =   data[66]; buffer[19][1] =   data[67]; buffer[19][2] =   data[80]; buffer[19][3] =   data[81];
            buffer[20][0] =   data[68]; buffer[20][1] =   data[69]; buffer[20][2] =   data[82]; buffer[20][3] =   data[83];

        }
        if (partition ==   1) {
            buffer[0][0] =   data[84]; buffer[0][1] =   data[85]; buffer[0][2] =   data[98]; buffer[0][3] =   data[99];
            buffer[1][0] =   data[86]; buffer[1][1] =   data[87]; buffer[1][2] =  data[100]; buffer[1][3] =  data[101];
            buffer[2][0] =   data[88]; buffer[2][1] =   data[89]; buffer[2][2] =  data[102]; buffer[2][3] =  data[103];
            buffer[3][0] =   data[90]; buffer[3][1] =   data[91]; buffer[3][2] =  data[104]; buffer[3][3] =  data[105];
            buffer[4][0] =   data[92]; buffer[4][1] =   data[93]; buffer[4][2] =  data[106]; buffer[4][3] =  data[107];
            buffer[5][0] =   data[94]; buffer[5][1] =   data[95]; buffer[5][2] =  data[108]; buffer[5][3] =  data[109];
            buffer[6][0] =   data[96]; buffer[6][1] =   data[97]; buffer[6][2] =  data[110]; buffer[6][3] =  data[111];
            buffer[7][0] =  data[112]; buffer[7][1] =  data[113]; buffer[7][2] =  data[126]; buffer[7][3] =  data[127];
            buffer[8][0] =  data[114]; buffer[8][1] =  data[115]; buffer[8][2] =  data[128]; buffer[8][3] =  data[129];
            buffer[9][0] =  data[116]; buffer[9][1] =  data[117]; buffer[9][2] =  data[130]; buffer[9][3] =  data[131];
            buffer[10][0] =  data[118]; buffer[10][1] =  data[119]; buffer[10][2] =  data[132]; buffer[10][3] =  data[133];
            buffer[11][0] =  data[120]; buffer[11][1] =  data[121]; buffer[11][2] =  data[134]; buffer[11][3] =  data[135];
            buffer[12][0] =  data[122]; buffer[12][1] =  data[123]; buffer[12][2] =  data[136]; buffer[12][3] =  data[137];
            buffer[13][0] =  data[124]; buffer[13][1] =  data[125]; buffer[13][2] =  data[138]; buffer[13][3] =  data[139];
            buffer[14][0] =  data[140]; buffer[14][1] =  data[141]; buffer[14][2] =  data[154]; buffer[14][3] =  data[155];
            buffer[15][0] =  data[142]; buffer[15][1] =  data[143]; buffer[15][2] =  data[156]; buffer[15][3] =  data[157];
            buffer[16][0] =  data[144]; buffer[16][1] =  data[145]; buffer[16][2] =  data[158]; buffer[16][3] =  data[159];
            buffer[17][0] =  data[146]; buffer[17][1] =  data[147]; buffer[17][2] =  data[160]; buffer[17][3] =  data[161];
            buffer[18][0] =  data[148]; buffer[18][1] =  data[149]; buffer[18][2] =  data[162]; buffer[18][3] =  data[163];
            buffer[19][0] =  data[150]; buffer[19][1] =  data[151]; buffer[19][2] =  data[164]; buffer[19][3] =  data[165];
            buffer[20][0] =  data[152]; buffer[20][1] =  data[153]; buffer[20][2] =  data[166]; buffer[20][3] =  data[167];

        }
        if (partition ==   2) {
            buffer[0][0] =  data[168]; buffer[0][1] =  data[169]; buffer[0][2] =  data[182]; buffer[0][3] =  data[183];
            buffer[1][0] =  data[170]; buffer[1][1] =  data[171]; buffer[1][2] =  data[184]; buffer[1][3] =  data[185];
            buffer[2][0] =  data[172]; buffer[2][1] =  data[173]; buffer[2][2] =  data[186]; buffer[2][3] =  data[187];
            buffer[3][0] =  data[174]; buffer[3][1] =  data[175]; buffer[3][2] =  data[188]; buffer[3][3] =  data[189];
            buffer[4][0] =  data[176]; buffer[4][1] =  data[177]; buffer[4][2] =  data[190]; buffer[4][3] =  data[191];
            buffer[5][0] =  data[178]; buffer[5][1] =  data[179]; buffer[5][2] =  data[192]; buffer[5][3] =  data[193];
            buffer[6][0] =  data[180]; buffer[6][1] =  data[181]; buffer[6][2] =  data[194]; buffer[6][3] =  data[195];
            buffer[7][0] =  data[196]; buffer[7][1] =  data[197]; buffer[7][2] =  data[210]; buffer[7][3] =  data[211];
            buffer[8][0] =  data[198]; buffer[8][1] =  data[199]; buffer[8][2] =  data[212]; buffer[8][3] =  data[213];
            buffer[9][0] =  data[200]; buffer[9][1] =  data[201]; buffer[9][2] =  data[214]; buffer[9][3] =  data[215];
            buffer[10][0] =  data[202]; buffer[10][1] =  data[203]; buffer[10][2] =  data[216]; buffer[10][3] =  data[217];
            buffer[11][0] =  data[204]; buffer[11][1] =  data[205]; buffer[11][2] =  data[218]; buffer[11][3] =  data[219];
            buffer[12][0] =  data[206]; buffer[12][1] =  data[207]; buffer[12][2] =  data[220]; buffer[12][3] =  data[221];
            buffer[13][0] =  data[208]; buffer[13][1] =  data[209]; buffer[13][2] =  data[222]; buffer[13][3] =  data[223];
            buffer[14][0] =  data[224]; buffer[14][1] =  data[225]; buffer[14][2] =  data[238]; buffer[14][3] =  data[239];
            buffer[15][0] =  data[226]; buffer[15][1] =  data[227]; buffer[15][2] =  data[240]; buffer[15][3] =  data[241];
            buffer[16][0] =  data[228]; buffer[16][1] =  data[229]; buffer[16][2] =  data[242]; buffer[16][3] =  data[243];
            buffer[17][0] =  data[230]; buffer[17][1] =  data[231]; buffer[17][2] =  data[244]; buffer[17][3] =  data[245];
            buffer[18][0] =  data[232]; buffer[18][1] =  data[233]; buffer[18][2] =  data[246]; buffer[18][3] =  data[247];
            buffer[19][0] =  data[234]; buffer[19][1] =  data[235]; buffer[19][2] =  data[248]; buffer[19][3] =  data[249];
            buffer[20][0] =  data[236]; buffer[20][1] =  data[237]; buffer[20][2] =  data[250]; buffer[20][3] =  data[251];

        }
    }
};

} // namespace nnet

#endif
