
WE TALK ABOUT IMPROVING SPEED

## Notes
1. OpenVINO required lower numpy version, so it lower the numpy version to numpy-1.26.4 from 2.4.3
   also with the networkx to 3.1 from 3.6.1.
2. on the other side, opencv and imagecodecs required numpy version>=2.0 (update it again if u want to use opencv with latest opencv versions)
3. u can downgrade the opencv anad imagecodecs if needed
   "pip install "opencv-python<4.10" "opencv-contrib-python<4.10" "imagecodecs<2024""
4. Script 1 output
   Available OpenVINO devices on this machine: ['CPU']
  (On Apple Silicon, expect: ['CPU'] only -- no Intel GPU/NPU)
  (On Intel hardware, you might see: ['CPU', 'GPU', 'NPU'])
  CPU: Apple M1 Pro
5. Deployment matrix results
    
2x2 DEPLOYMENT MATRIX RESULTS

Precision      Runtime          Mean ms   Std ms      FPS

FP32           ONNX Runtime      50.40    20.20     19.8
FP32           OpenVINO          15.94     2.91     62.7
Static INT8    ONNX Runtime      14.93     0.71     67.0
Static INT8    OpenVINO          19.51     2.21     51.2


ISOLATING EACH OPTIMIZATION (controlling for the other variable)

Quantization effect (FP32 -> Static INT8):
  Within ONNX Runtime: 50.40ms -> 14.93ms (3.38x speedup)  <- this is Day 15's number
  Within OpenVINO:     15.94ms -> 19.51ms (0.82x speedup)

Runtime effect (ONNX Runtime -> OpenVINO):
  At FP32:        50.40ms -> 15.94ms (3.16x speedup)
  At Static INT8: 14.93ms -> 19.51ms (0.76x speedup)

Best overall: Static INT8 + ONNX Runtime = 14.93ms (67.0 FPS)

-> i found that with the FP32, the conversion to openvino make it faster
   but with the static int8, i become slower a bit.
6. openpino accuracy

ONNX RUNTIME vs OPENVINO IR -- ACCURACY CHECK

Mean IoU (ONNX Runtime vs OpenVINO): 0.7400
Images with identical detection count: 29/48

-> lesson learned. the cross implementation can make numerical differences can flip detections. -> improve recall is the solution
   ONNX Runtime and OpenVINO: different software,
   tiny FP32 numerical differences (mean=0.14)
   -- UNAVOIDABLE, present in ANY model on ANY 2 runtimes

   Static INT8 quantization adds a ROUNDING STAIRCASE.
   When the two runtimes' tiny FP differences straddle
   a rounding boundary, the difference AMPLIFIES
   (mean=1.64, max=109.26)
   -- INT8-SPECIFIC, but still "normal" for INT8

   THIS MODEL (recall=0.66) has MANY true detections
   with confidence scores clustered near conf_thresh=0.25
   -- a property of THIS TRAINED MODEL specifically
   (Day 15's finding)

   When amplified noise happens to land on
   confidence channels for step 3's near-threshold
   anchors -> some anchors flip from "detected" to
   "not detected" (or vice versa) BETWEEN the two
   runtimes -> detection-level IoU=0.74

FP32 (raw recipe):
  ONNX Runtime cooks it as-written: 50.40ms (slow, unoptimized)
  OpenVINO REWRITES the recipe first (combines steps,
  reorders ingredients for ITS oven): 15.94ms (much faster)
  -> OpenVINO's rewrite was VERY valuable here

Static INT8 (Day 15 already rewrote the recipe AND
switched to metric units):
  ONNX Runtime cooks this pre-rewritten recipe: 14.93ms
  (already fast -- Day 15 did the rewriting AND unit
  conversion)

  OpenVINO takes this ALREADY-REWRITTEN recipe and
  insists on rewriting it AGAIN into ITS OWN preferred
  format: 19.51ms
  -> The SECOND rewrite found nothing left to improve,
     but STILL COST TIME to perform the rewrite itself
  -> Net: slower than just using the already-good recipe
     directly

optimization A and optimization B both target the same bottleneck, so A+B < max(A, B) gain, sometimes A+B < A alone

## Analogy
1. 
ONNX Runtime  = a generic kitchen that works in ANY house
                (your Mac, a generic Linux PC, anything)
                Good, but not hyper-optimized for any one
                specific oven brand.

OpenVINO      = Intel's PREMIUM kitchen, built specifically
                for Intel CPUs/iGPUs. Knows every trick for
                Intel hardware. If your factory PC has an
                Intel chip, OpenVINO often runs the SAME
                model 20-50% faster than generic ONNX Runtime.

TensorRT      = NVIDIA's PREMIUM kitchen, built specifically
                for NVIDIA GPUs (Jetson, RTX, etc.). Same idea
                -- if you have an NVIDIA GPU, TensorRT extracts
                performance ONNX Runtime can't.
## Questions
1. What does the OpenVINO IR format (.xml + .bin) represent, conceptually, vs ONNX's single .onnx file?
   Open VINO IR format represents the graph structure (.xml) and the weights (.bin). when single.onnx file contains all of it/
2. On Apple Silicon, which OpenVINO "device" will actually be available — CPU, GPU, or both? (hint: this connects to the supported devices page)
  only CPU
3. Given Day 15's finding that static INT8 was already 2.1x faster than FP32 via ONNX Runtime, what's the BEST CASE for OpenVINO today — multiply that further, or reach the same floor faster?
   i think it would be faster if we run it on intel's hardware. since i dont use intel's hardware, i think it would be the same as the static INT8 via ONNX runtime
4. Given everything from Days 14-16, what is your FINAL Stage 1 deployment recommendation, and what ONE caveat would you give to someone deploying thison  real factory hardware?
   ONNX runtime + INT8 quantization. recall is the main thing. try to make it as high as possible.
   