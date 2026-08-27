"""
FINN dataflow build template for PlantEdgeNet INT8 on the Puzhi PA200T-StarLite
(AMD Artix-7 XC7A200T, part xc7a200tfbg484-2).

RUN THIS INSIDE THE FINN DOCKER CONTAINER:
    git clone https://github.com/Xilinx/finn && cd finn
    ./run-docker.sh
    # then, inside:  python /path/to/ICCIT_Paper/fpga/finn_build.py

FINN turns the QONNX (from fpga/export_onnx.py --format qonnx) into a
per-layer HLS dataflow accelerator + a stitched IP + a Vivado project.
You then open the Vivado project, run synth/impl, and close timing.

Notes for this device (no Zynq PS):
  * target board is NOT in FINN's pynq board list -> use fpga_part + a custom
    shell / manual AXI-Stream wiring (MicroBlaze + AXI-DMA, or JTAG-to-AXI for
    bench tests). Set generate_outputs to STITCHED_IP + OOC_SYNTH, not BITFILE,
    then integrate + implement in Vivado yourself.
  * tune `folding_config_file` (PE/SIMD per layer) so DSP <= ~600 and
    BRAM <= ~300 with margin. Start from auto-folding, then hand-tune.
"""

import os
from finn.builder.build_dataflow import build_dataflow_cfg
from finn.builder.build_dataflow_config import (
    DataflowBuildConfig, DataflowOutputType, ShellFlowType, VerificationStepType,
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "export", "plantedgenet_int8.qonnx")
OUTPUT = os.path.join(HERE, "finn_out")

cfg = DataflowBuildConfig(
    output_dir=OUTPUT,
    synth_clk_period_ns=10.0,                 # 100 MHz; relax to 13.3 (75 MHz) if timing fails
    fpga_part="xc7a200tfbg484-2",
    shell_flow_type=ShellFlowType.VIVADO_ZYNQ if False else None,  # bare Artix -> no shell
    board=None,
    steps=None,                              # default step list
    generate_outputs=[
        DataflowOutputType.ESTIMATE_REPORTS,
        DataflowOutputType.STITCHED_IP,
        DataflowOutputType.OOC_SYNTH,          # out-of-context synth for real resource/timing
        DataflowOutputType.PYNQ_DRIVER,        # reference driver; adapt for MicroBlaze/AXI-DMA
    ],
    verify_steps=[
        VerificationStepType.QONNX_TO_FINN_PYTHON,
        VerificationStepType.STREAMLINED_PYTHON,
        VerificationStepType.FOLDED_HLS_CPPSIM,
    ],
    verify_input_npy=os.path.join(HERE, "export", "verify_input.npy"),   # (N,3,64,64) uint8/int8
    verify_expected_output_npy=os.path.join(HERE, "export", "verify_output.npy"),
    # folding_config_file=os.path.join(HERE, "folding_config.json"),     # hand-tuned PE/SIMD
    auto_fifo_depths=True,
    target_fps=200,                            # drives auto-folding; raise for more parallelism
    standalone_thresholds=True,
    force_python_rtlsim=False,
)

if __name__ == "__main__":
    assert os.path.exists(MODEL), f"missing {MODEL} - run fpga/export_onnx.py --format qonnx first"
    build_dataflow_cfg(MODEL, cfg)
    print(f"\nFINN outputs in {OUTPUT}")
    print("Next: open the stitched-IP Vivado project, add MicroBlaze+AXI-DMA (or JTAG-to-AXI),")
    print("      run implementation, check utilization/timing, generate bitstream.")
