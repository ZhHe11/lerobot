import gymnasium as gym
import metaworld
import numpy as np
from collections import defaultdict
from lerobot.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy
from pathlib import Path
from PIL import Image
import os
import torch
from torchvision.transforms import ToTensor
import json

class TaskDescriptionLoader:
    def __init__(self, json_path):
        self.task_dict = self._load_json(json_path)
    
    def _load_json(self, path):
        """加载JSON任务描述文件"""
        task_dict = {}
        with open(path, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    task_dict[data['task_index']] = data['task']
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {line}\nError: {e}")
        return task_dict
    
    def get_description(self, task_index):
        """获取指定任务索引的描述"""
        return self.task_dict.get(task_index, f"Task {task_index}")


device = "cuda" 
pretrained_policy_path = "/home/xlab/Code/lerobot/lerobot/ckpt/pretrained_model"

# Load the Pi0Fast policy
policy = PI0FASTPolicy.from_pretrained(pretrained_policy_path)
policy.eval()
policy.to(device)


def evaluate_mt50(seed=42, num_episodes=10, save_images=True):
    """
    评估 MT50 环境中所有任务的准确率
    
    参数:
        seed: 随机种子
        num_episodes: 每个任务评估的episode数量
        render: 是否渲染环境
    """
    # 创建同步向量环境
    envs = gym.make_vec('Meta-World/MT50', vector_strategy='sync', seed=seed, camera_id=2, render_mode='rgb_array')
    
    # 获取所有任务名称
    mt50 = metaworld.MT50(seed=seed)
    task_names = [name for name, _ in mt50.train_classes.items()]
    task_loader = TaskDescriptionLoader("/home/xlab/Code/lerobot/lerobot/eval_env/tasks.jsonl")

    # 存储每个任务的评估结果
    task_results = defaultdict(list)
    
    # 重置所有环境
    # obs, info = envs.reset()
    
    # 为每个环境分配一个特定任务
    env_indices = np.arange(50)
    np.random.shuffle(env_indices)
    
    for task_idx, task_name in enumerate(task_names):
        # 为当前任务选择环境索引
        env_idx = env_indices[task_idx % 50]
        # 获取当前任务的env
        env = envs.envs[env_idx]
        task_desc = task_loader.get_description(env_idx)
        print(f"Evaluating task: {task_name} - {task_desc} (Index: {env_idx})")

        # 获取当前任务的专家演示策略
        # expert_policy = mt50.train_classes[task_name]().get_policy(render=render)
        
        # 评估当前任务
        successes = []
        for ep in range(num_episodes):
            obs, _ = env.reset()
            img_array = env.render()
            img = Image.fromarray(img_array)
            img = img.rotate(180)
            img_array = np.array(img)
            # todo: input the action chunk size
            action_chunk = torch.from_numpy(env.action_space.sample()).expand(10, -1).to(device)


            # Create the data format for the policy
            episode_data = {
                'observation.state': [],
                'action': [],
                'next.reward': [],
                'next.success': [],
                'observation.environment_state': [],
                'observation.image': [],
                'task_id': env_idx,
                'timestamp': 0.0,
                'frame_index': 0,
                'episode_index': ep,
                'index': 0,
                'task_index': env_idx,
                'action_is_pad': [],
                'task': task_name
            }

            terminated = False
            truncated = False
            success = False

            # convert array to image and save the img
            img_path = Path(f"results/{task_name}_episode_{ep}.png")
            if save_images:
                # to do: input the path to save the image
                if not os.path.exists("results"):
                    os.makedirs("results", exist_ok=True)
                img_path = Path(f"results/{task_name}_episode_{ep}_init.png")
                img.save(img_path)
                print(f"Initial observation saved to {img_path}")            
            
            # while not (terminated or truncated):
            for i in range(100):  # Limit to 100 steps per episode
                # Load data into the episode data
                img_array = env.render()
                img = Image.fromarray(img_array).rotate(180)
                img_tensor = ToTensor()(img).to(device)
                episode_data['observation.state'] = torch.tensor(obs[:4])
                episode_data['observation.environment_state'] = torch.tensor(obs)
                episode_data['observation.image'] = img_tensor.squeeze(0).cpu()
                episode_data['action_is_pad'] = torch.tensor([False, False, False, False, False, False, False, False, False, False])
                val_input = {
                    'observation.state': episode_data['observation.state'].to(device),
                    'observation.environment_state': episode_data['observation.environment_state'].to(device),
                    'observation.image': img_tensor,
                    'action': action_chunk,
                    'action_is_pad': episode_data['action_is_pad'].to(device),
                    'task': task_name  # 保持为列表
                }

                # 使用策略生成动作
                with torch.no_grad():
                    action_chunk = policy(val_input)

                action = action_chunk[0].cpu().numpy()
                
                # 执行动作
                obs, reward, terminated, truncated, info = env.step(action)

                # 记录结果
                episode_data['action'].append(torch.tensor(action))
                episode_data['next.reward'].append(torch.tensor(reward))
                episode_data['next.success'].append(torch.tensor(info.get('success', False)))
                
                # 更新时间步
                episode_data['timestamp'] += 1.0
                episode_data['frame_index'] += 1
                episode_data['index'] += 1
                
                if info.get('success', False):
                    success = True
            
            successes.append(success)
            print(f"Episode {ep+1}/{num_episodes} - Success: {success}")
        
        # 计算成功率
        success_rate = np.mean(successes)
        task_results[task_name] = success_rate
        print(f"Task {task_name} success rate: {success_rate:.2f}")

    
    # 打印汇总结果
    print("\n=== Final Evaluation Results ===")
    for task_name, success_rate in task_results.items():
        print(f"{task_name}: {success_rate:.2f}")
    
    avg_success_rate = np.mean(list(task_results.values()))
    print(f"\nAverage success rate across all tasks: {avg_success_rate:.2f}")
    
    return task_results

if __name__ == "__main__":
    results = evaluate_mt50(seed=42, num_episodes=3, save_images=True)

