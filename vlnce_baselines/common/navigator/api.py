from openai import OpenAI
import torch
import numpy as np

import sys
import os

from tenacity import retry, wait_random_exponential, stop_after_attempt

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings

from transformers import AutoConfig, AutoModelForCausalLM
from SpatialBot3B.configuration_bunny_phi import *
from SpatialBot3B.modeling_bunny_phi import *

AutoConfig.register("bunny-phi", BunnyPhiConfig)
AutoModelForCausalLM.register(BunnyPhiConfig, BunnyPhiForCausalLM)
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()
warnings.filterwarnings('ignore')

from recognize_anything.ram.models import ram
from recognize_anything.ram import inference_ram
from recognize_anything.ram import get_transform


def check_model_all_on_gpu(model, name="model", verbose=True):
    """
    检查模型是否全部在 GPU 上（没有部分在 CPU 上）。
    适用于 Hugging Face 用 device_map='auto' 加载的模型。
    可在加载处与调用大模型的入口处复用。

    Args:
        model: 待检查的 PyTorch/HF 模型
        name: 模型显示名称
        verbose: 是否打印检查结果

    Returns:
        bool: True 表示全部在 GPU 上；False 表示有参数在 CPU 上。
    """
    def _log(msg):
        if verbose:
            print(msg)
    # 方式1: 有 hf_device_map 时（device_map='auto' 会写入）
    if hasattr(model, "hf_device_map") and model.hf_device_map is not None:
        cpu_modules = [k for k, v in model.hf_device_map.items() if v == "cpu"]
        if cpu_modules:
            _log(f"[{name}] 部分在 CPU 上, 在 CPU 的模块: {cpu_modules[:5]}{'...' if len(cpu_modules) > 5 else ''}")
            return False
        _log(f"[{name}] 全部在 GPU 上 (hf_device_map 均为 cuda)")
        return True
    # 方式2: 遍历所有参数检查 .device
    devices = set()
    for p in model.parameters():
        devices.add(str(p.device))
    on_cpu = any("cpu" in d for d in devices)
    if on_cpu:
        _log(f"[{name}] 部分在 CPU 上, 涉及设备: {devices}")
        return False
    _log(f"[{name}] 全部在 GPU 上, 设备: {devices}")
    return True


def _force_device(module, device):
    """把模型所有子模块移到 device，并统一各层上的 .device 属性（modeling_bunny_phi 内部 .to(self.device) 依赖此属性）。"""
    module.to(device)
    for _name, child in module.named_modules():
        child.to(device)
        if hasattr(child, "device"):
            try:
                child.device = device
            except (AttributeError, TypeError):
                pass
    if hasattr(module, "device"):
        try:
            module.device = device
        except (AttributeError, TypeError):
            pass


def _patch_spatialbot_device(model, device):
    """Patch mm_projector.forward：在计算前把输入挪到与权重同一 device，避免 cached modeling_bunny_phi 里 vision 在 CPU、projector 在 GPU 报错。"""
    inner = getattr(model, "get_model", lambda: None)()
    if inner is None or not hasattr(inner, "mm_projector"):
        return
    proj = inner.mm_projector
    target_device = next(proj.parameters()).device
    _orig_forward = proj.forward

    def _forward_with_device(x):
        if hasattr(x, "to") and str(x.device) != str(target_device):
            x = x.to(target_device)
        return _orig_forward(x)

    proj.forward = _forward_with_device


class llmClient:
    def __init__(self, model_type = '', api_key=None, base_url=None):
        '''
        Initialize LLM client based on model type and API key.
        
        Args:
            model_type (str): Either "gpt" or "opensource"
            api_key (str): API key for OpenAI (if using GPT)
        '''
        # Configure based on model type
        if model_type == "gpt-4o-2024-08-06":
            self.model = model_type
            self.client = OpenAI(api_key=api_key)
            
        elif model_type == "Qwen/Qwen2-72B":
            self.model = model_type
            self.client = OpenAI(
                api_key="not-needed",  # This value doesn't matter for local deployment
                base_url="http://0.0.0.0:23333/v1"
            )

        ##### Ollama 部署的分支选择 #############################################
        elif model_type == "ollama-llms":
            # 这里的 "qwen2:7b" 必须和你在 Ollama 里拉取/运行的模型名一致
            # self.model = "qwen2:7b" # 神
            # self.model = "qwen2.5:32b" # 跑的效果很差，乱跑
            # self.model = "qwen2.5:14b" # 反复横跳
            # self.model = "llama3.1:70b" # 累死的骆驼
            # self.model = "llama3-70b-nav"   # num_ctx=2048, num_batch=4
            self.model = "llama370b_ctx_batch"   # num_ctx=4096, num_batch=64
    
            # self.model = "phi3:14b"
            self.client = OpenAI(
                api_key="not-needed",  # Ollama 不校验这个
                base_url="http://127.0.0.1:11434/v1"
            )
        ##### Ollama 部署的分支选择 #############################################
        else:
            raise ValueError(f"Unknown model type: {model_type}. Use 'gpt' or 'opensource'.")
        
        print(f"Initialized LLM client with model: {self.model}")

    def set_model(self, model):
        self.model = model

    # Long timeout for local LLM (e.g. Ollama): default client timeout is 10 min, which can trigger retries on slow inference
    DEFAULT_REQUEST_TIMEOUT = 1800  # 30 minutes for local models

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def _completion_with_backoff(self, **kwargs):
        kwargs.setdefault("timeout", self.DEFAULT_REQUEST_TIMEOUT)
        return self.client.chat.completions.create(**kwargs)

    def gpt_infer(self, system_prompt, user_prompt, num_output=1, max_tokens=None):
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        request_params = {
            "model": self.model,
            "messages": messages,
            "temperature": 0
        }
        if max_tokens is not None:
            request_params["max_tokens"] = max_tokens
        
        if num_output == 1:
            chat_response = self._completion_with_backoff(**request_params)
            answer = chat_response.choices[0].message.content
            return answer
        else:
            responses = []
            for _ in range(num_output):
                chat_response = self._completion_with_backoff(**request_params)
                responses.append(chat_response.choices[0].message.content)
            return responses

    
class spatialClient:
    def __init__(self, device):
        self.device = device
        self.ram_path = "./recognize_anything/pretrained/ram_swin_large_14m.pth"
        self.spatialbot_path = "./SpatialBot3B"
        view_record_path = "cache_files/view_cache.json"
        self.ram_transform = None
        self.ram_model = None
        self.spatialbot_model = None
        self.spatialbot_tokenizer = None
        self._init_error = None
        try:
            # 整模型放在同一 device，避免 vision/mm_projector 分在 CPU/GPU 引发 RuntimeError
            self.spatialbot_model = AutoModelForCausalLM.from_pretrained(
                self.spatialbot_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            ).to(self.device)
            # 强制所有参数、buffer 和子模块到同一 device，并统一 model 内部使用的 .device
            _force_device(self.spatialbot_model, self.device)
            # Monkey-patch: 保证 encode_images 输出与 mm_projector 同 device，避免 cached modeling 里 cpu/cuda 混用
            _patch_spatialbot_device(self.spatialbot_model, self.device)
            self.spatialbot_tokenizer = AutoTokenizer.from_pretrained(
                self.spatialbot_path,
                trust_remote_code=True)
            self.ram_transform = get_transform(image_size=224)
            self.ram_model = ram(pretrained=self.ram_path, image_size=224, vit='swin_l').eval().to(self.device)
        except Exception as e:
            self._init_error = e
            print(f"Error in loading RAM or SpatialBot: {e}")
            raise RuntimeError("spatialClient 初始化失败，无法加载 RAM 或 SpatialBot。请检查权重路径与依赖。") from e
            
    def ram_img_tagging(self, image):
        if self.ram_transform is None or self.ram_model is None:
            raise RuntimeError("RAM 未成功加载，无法进行 ram_img_tagging。请检查 recognize_anything 权重路径及上述报错。") from self._init_error
        ram_img = self.ram_transform(image).unsqueeze(0).to(self.device)
        img_tags = inference_ram(ram_img, self.ram_model)[0]
        return img_tags
    
    def spatialbot_description(self, image_dict, prompt):
        offset_bos = 0
        text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image 1>\n<image 2>\n{prompt} ASSISTANT:"
        text_chunks = [self.spatialbot_tokenizer(chunk).input_ids for chunk in text.split('<image 1>\n<image 2>\n')]
        input_ids = torch.tensor(text_chunks[0] + [-201] + [-202] + text_chunks[1][offset_bos:], dtype=torch.long).unsqueeze(0).to(self.device)
        image1 = image_dict['rgb']
        image2 = image_dict['depth']
        channels = len(image2.getbands())
        if channels == 1:
            img = np.array(image2)
            height, width = img.shape
            three_channel_array = np.zeros((height, width, 3), dtype=np.uint8)
            three_channel_array[:, :, 0] = (img // 1024) * 4
            three_channel_array[:, :, 1] = (img // 32) * 8
            three_channel_array[:, :, 2] = (img % 32) * 8
            image2 = Image.fromarray(three_channel_array, 'RGB')
        image_tensor = self.spatialbot_model.process_images([image1,image2], self.spatialbot_model.config).to(dtype=self.spatialbot_model.dtype, device=self.device)
        # Do not move vision_tower here: model is already on device via device_map='auto'; repeated .to(device) can trigger OOM when GPU is full.
        output_ids = self.spatialbot_model.generate(
            input_ids,
            images=image_tensor,
            max_new_tokens=200, 
            use_cache=True,
            repetition_penalty=1.0 
        )[0]
        return self.spatialbot_tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
    
    def observe_view(self, logger, current_step, direction_idx, direction_image):
        img_tags = self.ram_img_tagging(direction_image['rgb'])
        spatial_scene_description_prompt = "What objects are in the image, and how far are these objects from the camera, calculate the result in meter."
        spatial_scene_description = self.spatialbot_description(direction_image, spatial_scene_description_prompt)
        view_observation = f"Scene Description: {spatial_scene_description} Scene Objects: {img_tags}; "
        observe_result = f"Direction {direction_idx} Direction Viewpoint ID: {direction_idx} in Step ID: {current_step} Elevation: Eye Level "  + view_observation
        return observe_result