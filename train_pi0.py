from lerobot.policies.pi0.modeling_pi0 import PI0Policy
from lerobot.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy

# 将模型ID改为本地目录的绝对路径
local_model_path = r"/media/xlab/My Book/dataset/pi0fast_ckpt"

policy = PI0FASTPolicy.from_pretrained(
    local_model_path,  # 关键：直接传本地路径
    local_files_only=True  # 确保不联网
)



print("Model loaded successfully from local path.")

