import gymnasium as gym
import metaworld
import time
import numpy as np

env = gym.make('Meta-World/MT1', 
               env_name='reach-v3',
               render_mode='human')

obs = env.reset()

for step in range(500):  # 限制500步
    # 生成更大幅度的随机动作
    a = np.random.uniform(-1, 1, size=env.action_space.shape) 
    
    obs, reward, truncate, terminate, info = env.step(a)

    print(a)

    env.render()
    
    # 添加延迟让运动可见
    time.sleep(0.05)  
    
    if truncate or terminate:
        print(f"Episode ended at step {step}")
        obs, info = env.reset()

env.close()