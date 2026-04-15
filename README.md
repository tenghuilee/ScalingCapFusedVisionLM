# Scaling Capability in Token Space: An Analysis of Large Vision Language Model

[paper](https://www.jmlr.org/papers/v26/24-2243.html)

This repository contains the implementation and experiments for the paper "Scaling Capability in Token Space: An Analysis of Large Vision Language Model", published in the Journal of Machine Learning Research. This research investigates the scaling relationship in vision-language models with respect to the number of vision tokens, revealing theoretical foundations for token-efficient model design.


## Abstract

Large language models have demonstrated predictable scaling behaviors with respect to model parameters and training data.
This study investigates whether a similar scaling relationship exist for vision-language models with respect to the number of vision tokens.
A mathematical framework is developed to characterize a relationship between vision token number and the expected divergence of distance between vision-referencing sequences.
The theoretical analysis reveals two distinct scaling regimes: sublinear scaling for less vision tokens and linear scaling for more vision tokens.
This aligns with model performance relationships of the form $S(n) \approx c / n^{\alpha(n)}$, where the scaling exponent relates to the correlation structure between vision token representations.
Empirical validations across multiple vision-language benchmarks show that model performance matches the prediction from scaling relationship.
The findings contribute to understanding vision token scaling in transformers through a theoretical framework that complements empirical observations.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/tenghuilee/ScalingCapFusedVisionLM.git
cd ScalingCapFusedVisionLM
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

Checkpoints: [modelscope](https://modelscope.cn/models/LiTenghui/scalingcapabilitytokenspace)

## Usage

### Basic Demo

Some demos are provided under folder [demos](./demos).

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


## Citation

```
@article{Li2025scalingcapabilitytokenspace,
  title = {Scaling Capability in Token Space: An Analysis of Large Vision Language Model},
  author = {Tenghui Li and Guoxu Zhou and Xuyang Zhao and Qibin Zhao},
  journal = {Journal of Machine Learning Research},
  year = {2025},
  month = {10},
  url = {https://www.jmlr.org/papers/v26/24-2243.html},
}
```

## License

MIT License.

For any questions or issues, please open an issue in the GitHub repository.
