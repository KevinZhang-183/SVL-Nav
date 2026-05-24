# Visualization eval: overhead RGB + mosaic frames (uses run_OpenNav_vis.yaml).
# Quick test: add  EVAL.EPISODE_COUNT 1  to the flag block below.

flag="--exp_name cont-cwp-opennav-vis
      --exp-config run_OpenNav_vis.yaml
      --llm ollama-llms
      --api_key 123456
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_ID 0
      TORCH_GPU_IDS [0]
      EVAL.SPLIT val_unseen
      "
CUDA_VISIBLE_DEVICES=0 python run.py $flag
