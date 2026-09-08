# Dataset

Power traces and derived statistics for 21 deep-learning models, collected on a single
NVIDIA H100 PCIe GPU via NVML.

- `traces/` — full sequential power traces with ground-truth layer labels. One row per
  power sample: model, layer name, layer type, layer start/end time, sample index, watts.
- `layer_power/` — per-layer aggregates: average power, standard deviation, energy, time.
- `fingerprints/` — one row per model: 31 power/energy/throughput/structure features.

Layer types: Conv, Norm, Activation, Pool, Linear, Dropout, Embedding, Other.

Absolute wattage is specific to the H100 used here (idle ~53 W). The recovery pipeline
normalizes each segment in amplitude and time, so layer *shapes* are the transferable part.

License: CC BY 4.0.
