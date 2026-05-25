# Visualization eval: enable overhead RGB + top-down via NAV_VIS in run_OpenNav_vis.yaml.
# Example CLI overrides:
#   NAV_VIS.ENABLE_OVERHEAD_RGB False
#   NAV_VIS.ENABLE_TOPDOWN_MAP True
#   NAV_VIS.SAVE_MOSAIC False
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
