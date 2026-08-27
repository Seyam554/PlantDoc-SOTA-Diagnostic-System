"""
hls4ml build template for PlantEdgeNet INT8 on Artix-7 XC7A200T
(part xc7a200tfbg484-2). Cross-check / fallback to the FINN flow.

    pip install hls4ml
    # needs Vitis HLS 2023.x / Vivado HLS on PATH for csynth + export_ip

Feeds the standard ONNX (fpga/export_onnx.py --format onnx). Uses io_stream +
Resource strategy for a CNN, 8-bit fixed point, and a modest reuse factor so
DSP fits. Then export_ip -> add the IP in Vivado, wire AXI-Stream to a
MicroBlaze + AXI-DMA (no PS on this device).
"""

import os
import hls4ml
import onnx

HERE = os.path.dirname(os.path.abspath(__file__))
ONNX_PATH = os.path.join(HERE, "export", "plantedgenet_int8.onnx")
OUT = os.path.join(HERE, "hls4ml_prj")

model = onnx.load(ONNX_PATH)

cfg = hls4ml.utils.config_from_onnx_model(
    model, granularity="name", default_precision="ap_fixed<8,4>", backend="Vitis"
)
cfg["Model"]["Strategy"] = "Resource"
cfg["Model"]["ReuseFactor"] = 64          # raise to cut DSP, lowers throughput
cfg["Model"]["Precision"] = "ap_fixed<8,4>"
for layer in cfg.get("LayerName", {}).values():
    layer["Strategy"] = "Resource"
    layer.setdefault("Precision", {})
    # keep accumulators wide enough (see QNNs low-precision accumulation paper)
    layer["Precision"]["accum"] = "ap_fixed<24,12>"

hls_model = hls4ml.converters.convert_from_onnx_model(
    model,
    output_dir=OUT,
    io_type="io_stream",
    backend="Vitis",
    part="xc7a200tfbg484-2",
    clock_period=10,                       # 100 MHz
    hls_config=cfg,
)

if __name__ == "__main__":
    assert os.path.exists(ONNX_PATH), f"missing {ONNX_PATH} - run fpga/export_onnx.py --format onnx"
    hls_model.compile()
    # numerical check against ONNX Runtime before synthesis:
    #   import numpy as np, onnxruntime as ort
    #   x = np.random.rand(16,3,64,64).astype('float32')
    #   ...compare hls_model.predict(x) vs ort output, argmax agreement...
    hls_model.build(csim=False, synth=True, export=True)   # csynth + export_ip
    hls4ml.report.read_vivado_report(OUT)
    print(f"\nhls4ml project + IP in {OUT}")
