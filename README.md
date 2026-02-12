# Scaling Capability in Token Space: An Analysis of Large Vision Language Model

This repository contains the implementation and experiments for the paper "Scaling Capability in Token Space: An Analysis of Large Vision Language Model", published in the Journal of Machine Learning Research. This research investigates the scaling relationship in vision-language models with respect to the number of vision tokens, revealing theoretical foundations for token-efficient model design.

## Table of Contents
- [Scaling Capability in Token Space: An Analysis of Large Vision Language Model](#scaling-capability-in-token-space-an-analysis-of-large-vision-language-model)
  - [Table of Contents](#table-of-contents)
  - [Abstract](#abstract)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Basic Demo](#basic-demo)
    - [Experiment with Different Vision Token Counts](#experiment-with-different-vision-token-counts)
  - [Requirements](#requirements)
  - [File Descriptions](#file-descriptions)
  - [Results](#results)
  - [Citation](#citation)
  - [License](#license)

## Abstract

Large language models have demonstrated predictable scaling behaviors with respect to model parameters and training data. This study investigates whether a similar scaling relationship exists for vision-language models with respect to the number of vision tokens. A mathematical framework is developed to characterize a relationship between vision token number and the expected divergence of distance between vision-referencing sequences. The theoretical analysis reveals two distinct scaling regimes: sublinear scaling for fewer vision tokens and linear scaling for more vision tokens. This aligns with model performance relationships of the form S(n) ≈ c / n^(α(n)), where the scaling exponent relates to the correlation structure between vision token representations. Empirical validations across multiple vision-language benchmarks show that model performance matches the prediction from scaling relationship. The findings contribute to understanding vision token scaling in transformers through a theoretical framework that complements empirical observations.

**Keywords**: large language models, vision large language models, token-efficiency scaling capability, vision tokens, theoretical analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-repo/LLaVA.git
cd LLaVA/ScalingCapFusedVisionLM
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

Or install packages individually:
```bash
pip install transformers==4.44.2
pip install torch>=2.4.0
pip install fastchat==0.2.36
pip install thop
```

Note: For VLMEvalKit, install from the specific commit:
```bash
pip install git+https://github.com/open-compass/VLMEvalKit.git@0ff0c3e
```

## Usage

### Basic Demo
Run the basic demonstration of the proposed model:
```bash
python demo.py
```

### Experiment with Different Vision Token Counts
Evaluate models with various numbers of vision tokens:
```bash
python experiment_inference.py
```

See individual demo files for specific usage instructions and parameters.

## Requirements

- Python >= 3.8
- transformers==4.44.2
- pytorch >= 2.4.0
- fastchat==0.2.36
- VLMEvalKit (commit id: 0ff0c3e); see [VLMEvalKit](https://github.com/open-compass/VLMEvalKit.git) for more details.
- thop (to compute FLOPs and params)

## File Descriptions

### Main Scripts
- `demo.py`: A demo for the proposed model, demonstrating the core functionality (moved to `demos/`)
- `experiment_inference.py`: A demo script for inference of models with various number of vision tokens (moved to `demos/`)
- `_update_ckpt.py`: Utility script for updating checkpoints
- `inference_time_estimate.py`: Tools for estimating inference time
- `init_imgque_with_new_backbone.py`: Initialization utilities for new backbone architectures

### Demo Scripts
All demo scripts have been moved to the `demos/` directory:
- `demos/demo_clip_text_fusion.py`: Example of CLIP-text fusion techniques
- `demos/demo_eval_vlmeval.py`: Evaluation script using VLMEvalKit
- `demos/demo_llava1.5-7B_hf_baseline.py`: Baseline implementation using LLaVA 1.5-7B HuggingFace model
- `demos/demo_modeling_img_multi_query.py`: Implementation of multi-query image modeling
- `demos/demo_modeling_query_adapt_clip*.py`: Various implementations for query adaptation with CLIP
- `demos/demo_phi3.py`: Example using Phi3 model integration
- `demos/demo_visionzip.py`: Implementation of vision compression techniques

### Figure Generation Utilities
All figure generation utilities have been moved to the `figures/` directory:
- `figures/fig_utils.py`: Utilities for figure generation and visualization
- `figures/expectation_bound_visualize_main.py`: Visualization tools for expectation bounds

### Other Components
- `datasets_share/`: Shared dataset utilities
- `scripts/`: Additional utility scripts
- `tensorfusionvlm/`: Tensor fusion VLM implementation

## Results

Our theoretical analysis reveals two distinct scaling regimes:
- **Sublinear scaling**: For models with fewer vision tokens, performance scales sublinearly with the number of tokens
- **Linear scaling**: For models with more vision tokens, performance scales approximately linearly with the number of tokens

These findings provide insights into optimal vision token allocation for vision-language models and suggest strategies for developing more efficient architectures.

## Citation

If you use this code in your research, please cite:

```
@article{Li2025scalingcapabilitytokenspace,
  title = {Scaling Capability in Token Space: An Analysis of Large Vision Language Model},
  author = {Tenghui Li and Guoxu Zhou and Xuyang Zhao and Qibin Zhao},
  journal = {Journal of Machine Learning Research},
  year = {2025},
  month = {10},
  url = {http://jmlr.org/papers/v26/24-2443.html},
}
```

## License

This project is licensed under the terms of the license specified in the repository.

For any questions or issues, please open an issue in the GitHub repository.