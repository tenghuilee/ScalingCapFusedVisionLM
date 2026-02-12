# Demo Scripts

This directory contains demonstration scripts for the paper "Scaling Capability in Token Space: An Analysis of Large Vision Language Model".

## Files

- `demo.py`: Basic demonstration of the proposed model
- `demo_clip_text_fusion.py`: Example of CLIP-text fusion techniques
- `demo_eval_vlmeval.py`: Evaluation script using VLMEvalKit
- `demo_llava1.5-7B_hf_baseline.py`: Baseline implementation using LLaVA 1.5-7B HuggingFace model
- `demo_modeling_img_multi_query.py`: Implementation of multi-query image modeling
- `demo_modeling_query_adapt_clip.py`: Query adaptation with CLIP implementation
- `demo_modeling_query_adapt_clip_lowrank.py`: Low-rank approximation for query adaptation
- `demo_modeling_query_adapt_clip_search.py`: Search-based optimization for query adaptation
- `demo_phi3.py`: Example using Phi3 model integration
- `demo_visionzip.py`: Implementation of vision compression techniques
- `experiment_inference.py`: Demo script for inference of models with various number of vision tokens

## Usage

Each demo script can be run independently to reproduce specific aspects of the research:

```bash
python demo.py
```

For specific usage instructions, please refer to the individual script headers.

## Dependencies

- transformers==4.44.2
- pytorch >= 2.4.0
- fastchat==0.2.36
- VLMEvalKit (commit id: 0ff0c3e)
- thop (for FLOPs computation)