# Comprehensive Neural Network Statistics Summary

## Overview

This document summarizes the comprehensive statistical analysis performed on three types of neural network models:
1. **Artificial Neural Networks (ANNs)** - Traditional deep learning models
2. **Rate-encoded Spiking Neural Networks (Rate SNNs)** - ANN-to-SNN converted models using rate encoding
3. **Temporal Spiking Neural Networks (Temporal SNNs)** - Native SNN models using temporal encoding

## Analysis Scope

The analysis calculated and visualized five key metrics:
1. **Number of neurons** across all model types
2. **Number of synapses** (parameters) across all model types  
3. **Inference time** for batch sizes [16, 32, 64, 128]
4. **Time per spike** for SNN models
5. **Total spike count** for SNN models

## Key Findings

### 1. Model Architecture Statistics

#### Neurons Count:
- **ANN Models**: Average 1,869 neurons (Range: 1,176 - 2,736)
- **Rate SNN Models**: Average 5,608 neurons (Range: 3,528 - 8,208) 
- **Temporal SNN Models**: Average 1,869 neurons (Range: 1,176 - 2,736)

*Key Insight*: Rate-encoded SNN models have ~3x more neurons due to the conversion process that expands the architecture.

#### Synapses Count:
- **ANN Models**: Average 620,024 synapses (Range: 219,288 - 1,153,712)
- **Rate SNN Models**: Average 1,240,062 synapses (Range: 438,586 - 2,307,442)
- **Temporal SNN Models**: Average 620,024 synapses (Range: 219,288 - 1,153,712)

*Key Insight*: Rate-encoded SNNs have ~2x more synapses, while temporal SNNs maintain the same synapse count as their ANN counterparts.

### 2. Performance Analysis (Batch Size 64)

#### Inference Times:
- **ANN**: 0.0020s (baseline)
- **Rate SNN**: 0.2579s (131.12x slower than ANN)
- **Temporal SNN**: 0.0738s (37.52x slower than ANN)

*Key Insight*: Temporal SNNs are significantly more efficient than rate-encoded SNNs, though both are slower than ANNs.

### 3. Spike Statistics (Batch Size 64)

#### Total Spikes Generated:
- **Rate SNN**: 11,712,472 spikes on average
- **Temporal SNN**: 1,190,050 spikes on average

#### Time per Spike:
- **Rate SNN**: 2.2 × 10⁻⁸ seconds per spike
- **Temporal SNN**: 6.2 × 10⁻⁸ seconds per spike

*Key Insight*: Rate-encoded SNNs generate ~10x more spikes but process each spike faster, while temporal SNNs are more spike-efficient.

### 4. Scaling Analysis

#### Batch Size Scaling:
- **ANN models**: Excellent scaling with batch size
- **Rate SNN models**: Poor scaling, high overhead
- **Temporal SNN models**: Good scaling, moderate overhead

#### Architecture Size Impact:
- **Small models** (1,176-1,200 neurons): Fastest across all types
- **Medium models** (1,688-1,712 neurons): Moderate performance
- **Big models** (2,712-2,736 neurons): Slowest but highest capacity

## Model Categories Analyzed

### ANN Models (9 total):
- SmallAutoencoder (32, 16, 8 latent dimensions)
- Autoencoder (32, 16, 8 latent dimensions) 
- BigAutoencoder (32, 16, 8 latent dimensions)

### Rate SNN Models (9 total):
- Converted from corresponding ANN models using sinabs framework
- Use rate encoding with 100 timesteps
- Significantly larger architecture due to conversion overhead

### Temporal SNN Models (9 total):
- SmallSNNAutoencoder (32, 16, 8 latent dimensions)
- MediumSNNAutoencoder (32, 16, 8 latent dimensions)
- BigSNNAutoencoder (32, 16, 8 latent dimensions)
- Native snntorch implementation with temporal dynamics

## Efficiency Metrics

### Computational Efficiency:
1. **ANN**: Highest efficiency (baseline)
2. **Temporal SNN**: 37.5x computational overhead
3. **Rate SNN**: 131x computational overhead

### Spike Efficiency:
1. **Temporal SNN**: 1.2M spikes per inference
2. **Rate SNN**: 11.7M spikes per inference

### Memory Efficiency:
1. **ANN/Temporal SNN**: Same parameter count
2. **Rate SNN**: 2x parameter overhead

## Practical Implications

### When to Use Each Model Type:

#### ANN Models:
- **Best for**: High-throughput applications requiring maximum speed
- **Trade-offs**: Highest power consumption, no neuromorphic compatibility

#### Temporal SNN Models:
- **Best for**: Edge computing, neuromorphic hardware deployment
- **Trade-offs**: Moderate computational overhead but excellent spike efficiency

#### Rate SNN Models:
- **Best for**: Rapid prototyping, ANN-to-SNN conversion workflows
- **Trade-offs**: High computational overhead, inefficient spike generation

## Generated Visualizations

The analysis produced comprehensive visualizations saved in `../plots/`:

1. **comprehensive_statistics.png**: 
   - Neuron and synapse counts by model type
   - Inference time vs batch size comparison
   - Total spikes vs batch size analysis

2. **spike_analysis.png**:
   - Time per spike analysis
   - Individual model size breakdowns
   - Spike processing efficiency metrics

## Data Files

All numerical results are saved in:
- **comprehensive_statistics.json**: Complete dataset with all metrics
- **Organized by**: Model type, batch size, and metric category

## Conclusion

This comprehensive analysis demonstrates that while SNN models require more computational resources than traditional ANNs, temporal SNNs offer a reasonable trade-off between biological plausibility and computational efficiency. The spike counting and timing analysis provides crucial insights for optimizing SNN deployment in practical applications.

Rate-encoded SNNs, while useful for conversion workflows, show significant computational overhead and should be considered primarily for prototyping rather than production deployment. 