import os

os.environ["XILINX_VIVADO"] = "/afs/hep.wisc.edu/cms/sw/Xilinx/Vivado/2021.1"
os.environ["PATH"] = "/afs/hep.wisc.edu/cms/sw/Xilinx/Vivado/2021.1/bin:" + os.environ.get("PATH", "")

os.environ["XILINX_VITIS"] = "/afs/hep.wisc.edu/cms/sw/Xilinx/Vitis/2021.1"
os.environ["PATH"] = "/afs/hep.wisc.edu/cms/sw/Xilinx/Vitis/2021.1/bin:/afs/hep.wisc.edu/cms/sw/Xilinx/Vitis/2021.1/gnu/microblaze/lin/bin:/afs/hep.wisc.edu/cms/sw/Xilinx/Vitis/2021.1/gnu/arm/lin/bin:/afs/hep.wisc.edu/cms/sw/Xilinx/Vitis/2021.1/gnu/microblaze/linux_toolchain/lin64_le/bin:/afs/hep.wisc.edu/cms/sw/Xilinx/Vitis/2021.1/gnu/aarch32/lin/gcc-arm-linux-gnueabi/bin:/afs/hep.wisc.edu/cms/sw/Xilinx/Vitis/2021.1/gnu/aarch32/lin/gcc-arm-none-eabi/bin:/afs/hep.wisc.edu/cms/sw/Xilinx/Vitis/2021.1/gnu/aarch64/lin/aarch64-linux/bin:/afs/hep.wisc.edu/cms/sw/Xilinx/Vitis/2021.1/gnu/aarch64/lin/aarch64-none/bin:/afs/hep.wisc.edu/cms/sw/Xilinx/Vitis/2021.1/gnu/armr5/lin/gcc-arm-none-eabi/bin:/afs/hep.wisc.edu/cms/sw/Xilinx/Vitis/2021.1/tps/lnx64/cmake-3.3.2/bin:/afs/hep.wisc.edu/cms/sw/Xilinx/Vitis/2021.1/aietools/bin:" + os.environ.get("PATH", "")

os.environ["PATH"] = "/afs/hep.wisc.edu/cms/sw/Xilinx/Model_Composer/2021.1/bin:" + os.environ.get("PATH", "")

os.environ["XILINX_HLS"] = "/afs/hep.wisc.edu/cms/sw/Xilinx/Vitis_HLS/2021.1"
os.environ["PATH"] = "/afs/hep.wisc.edu/cms/sw/Xilinx/Vitis_HLS/2021.1/bin:" + os.environ.get("PATH", "")

import numpy as np
from pathlib import Path
from tensorflow import data
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.optimizers import Adam

from qkeras import (
    QConv2D,
    QDepthwiseConv2D,
    QActivation,
    QDense,
    QBatchNormalization,
    QDenseBatchnorm,
    quantized_bits,
)
from tensorflow.keras.layers import Input, Flatten, Dropout, Reshape, Concatenate
from tensorflow.keras.models import Model

import hls4ml
from hls4ml.model.layers import Activation as ActivationHLS
from hls4ml.model.optimizer import OptimizerPass, register_pass

import pprint

cicada = load_model("/hdfs/store/user/aloeliger/CICADA2026Models/cicadaStudent_classic", compile=False)
for i, layer in enumerate(cicada.layers):
    print(i, layer.name, layer.__class__.__name__)

def get_hls_config(keras_model):
    hls_config = hls4ml.utils.config_from_keras_model(keras_model, granularity="name")

    hls_config["Model"]["Strategy"] = "Latency"

    for layer_name in hls_config['LayerName']:
        if 'conv' in layer_name:
            hls_config['LayerName'][layer_name]['StreamOutputs'] = False
            hls_config['LayerName'][layer_name]['implementation'] = 'array'

    # # Default reuse factor for all layers
    for layer in hls_config["LayerName"].keys():
        hls_config["LayerName"][layer]["ReuseFactor"] = 2


    hls_config["LayerName"]["student_input"]["Precision"]["result"] = "fixed<10,6>"


    hls_config["LayerName"]["conv"]["Strategy"] = "Resource"
    hls_config["LayerName"]["conv"]["ReuseFactor"] = 1
    hls_config["LayerName"]["conv"]["ParallelizationFactor"] = 21
    hls_config["LayerName"]["conv"]["Precision"]["result"] = "fixed<30,22>"
    hls_config["LayerName"]["conv"]["Precision"]["accum"] = "fixed<30,22>"

    # Dense1 precision (v2)
    hls_config["LayerName"]["dense1"]["Precision"]["result"] = "fixed<26,14>"
    hls_config["LayerName"]["dense1"]["Precision"]["accum"] = "fixed<26,14>"

    # ---- Dense2 output precision ----
    hls_config["LayerName"]["dense2"]["Precision"]["result"] = "fixed<26,14>"
    hls_config["LayerName"]["dense2"]["Precision"]["accum"] = "fixed<26,14>"

    return hls_config

hls_config = get_hls_config(cicada)

def convert_to_hls4ml_model(keras_model, hls_config):
    hls_model = hls4ml.converters.convert_from_keras_model(
        keras_model,
        clock_period=6.25,
        backend="Vitis",
        hls_config=hls_config,
        io_type="io_parallel",
        output_dir="cicada-classic",
        part="xc7vx690tffg1927-2",
        project_name="cicada",
        version=3,
    )
    hls_model.compile()
    return hls_model

hls_model = convert_to_hls4ml_model(cicada, hls_config)

# hls_model.build(csim=True, synth=True, vsynth=False)
