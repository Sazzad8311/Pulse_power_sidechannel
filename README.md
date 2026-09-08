# PULSE: Recovering Neural Network Architectures from Power Telemetry

PULSE (*Power-based Unmasking of Layer SEquences*) recovers the ordered sequence of
**layer types** executed by a deep-learning model from GPU board power telemetry alone —
no access to the model's weights, inputs, activations, or computational graph, and no
physical probe, oscilloscope, or shared-cache co-location.

This repository contains the artifacts for the PULSE paper: the power-profiling harness,
the layer-recovery pipeline, and the collected power-trace dataset for 21 models.

## Headline result

Across 21 networks (11 vision CNNs, 10 transformer/NLP models) and 8 layer classes,
PULSE recovers layer-type sequences at **60.1% mean accuracy** under leave-one-model-out
validation — about **5x** the 12.5% eight-class chance baseline — and up to **92%**
(BERT-Base). The HMM ordering prior improves over per-segment k-NN on **17 of 20** models
(+7.5 points on average).

## Repository layout

```
notebooks/            profiling + recovery notebooks
  <Model>_power_profiling.ipynb    per-model profiling (21 notebooks)
  Template_Bag_Power_Curves.ipynb  builds the per-layer-type template library
  Layer_Recovery_From_Power_Trace.ipynb  segmentation + k-NN + HMM/Viterbi recovery
data/
  traces/             <Model>_sequential_power_trace.csv   full power traces w/ layer labels
  layer_power/        <Model>_layer_power.csv              per-layer power/energy/time
  fingerprints/       <Model>_fingerprint.csv              31-feature per-model fingerprint
results/              layer_recovery_holdout.csv           per-model k-NN vs HMM accuracy
                      template_bag_corr_*.csv              template separability
figures/              figures used in the paper
```

### Data schema

`data/traces/<Model>_sequential_power_trace.csv`

| column | meaning |
|---|---|
| `model` | model name |
| `layer_name` | framework layer identifier |
| `layer_type` | one of Conv, Norm, Activation, Pool, Linear, Dropout, Embedding, Other |
| `t_start_s`, `t_end_s` | layer execution window (seconds) |
| `sample_idx` | power sample index |
| `power_W` | GPU board power (watts) |

`results/layer_recovery_holdout.csv` — `held_out`, `n_segs`, `knn_accuracy`, `hmm_accuracy`.

## Models profiled (21)

**Vision (11, on MNIST resized to 224x224x3):** AlexNet, VGG-16, VGG-19, ResNet-18,
ResNet-20, ResNet-50, GoogLeNet, DenseNet-121, MobileNetV2, EfficientNet-B0, ConvNeXt-Tiny.

**Transformer/NLP (10, on SST-2 from GLUE):** BERT-Base, RoBERTa-Base, DistilBERT,
TinyBERT, ALBERT-Base, ELECTRA-Small, BART-Base, T5-Small, GPT2-Small, DistilGPT2.

## Setup

```bash
git clone https://github.com/<USER>/pulse-power-sidechannel.git
cd pulse-power-sidechannel
pip install -r requirements.txt
```

Profiling requires an NVIDIA GPU exposing power via NVML. Re-running the **recovery**
experiments needs no GPU — the released traces are sufficient.

## Reproducing the results

1. **Recovery only (no GPU needed).** Open `notebooks/Layer_Recovery_From_Power_Trace.ipynb`
   and run against `data/traces/`. It rebuilds the template library and the layer-type
   transition matrix from the non-held-out models, then reports per-model k-NN and HMM
   accuracy, reproducing `results/layer_recovery_holdout.csv` and the paper's headline
   60.1% figure.
2. **Collect new traces (GPU needed).** Run any `notebooks/<Model>_power_profiling.ipynb`.
   Power is sampled from NVML in a separate process and timestamped, then aligned with
   layer execution windows. Absolute wattage is hardware-specific; the pipeline's features
   are amplitude- and time-normalized, so shapes transfer across devices.

## Method

- **Offline.** Profile known models; pool each layer type's amplitude/time-normalized power
  shapes (resampled to L=100) into a *template bag*; estimate the layer-type transition
  matrix.
- **Online.** Smooth and segment an unknown trace at power change-points; score each segment
  against the template bags with k-NN (distance = 1 - Pearson r, plus a magnitude penalty)
  to get HMM emissions; decode the full sequence with Viterbi.

k-NN and Viterbi are implemented directly in NumPy/SciPy (no scikit-learn).

## Environment

Traces were collected on a single NVIDIA H100 PCIe (80 GB) node of TACC Lonestar6
(Rocky Linux 8.4, Slurm), host AMD EPYC 7763. GPU idle draw is ~53 W.

## Ethics

All experiments were run on models and hardware controlled by the authors. There was no
real victim, no third-party model, and no personal or user data at any stage. The artifacts
are released to support reproducibility and defensive research.

## Citation

See `CITATION.cff`. <!-- Add the Zenodo DOI badge here after the first release. -->

## License

Code: MIT (see `LICENSE`). Data: CC BY 4.0 (see `data/LICENSE`).
