import torch
import os

template = """
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  gradient_accumulation_steps: 'auto'
  gradient_clipping: 1.0
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: false
  zero_stage: 2
distributed_type: DEEPSPEED
downcast_bf16: 'no'
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: {num_processes}
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
"""

count = torch.cuda.device_count()

fname = f"./scripts/default_config_g{count}.yaml"

if not os.path.exists(fname):
    with open(fname, "w") as f:
        f.write(template.format(num_processes=count))

print(fname)
