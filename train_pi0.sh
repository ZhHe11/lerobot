## 训练指令
python -m lerobot.scripts.train \
    --dataset.repo_id=metaworld \
    --dataset.root="/home/xlab/Dataset/metaworld_mt50" \
    --policy.type=pi0fast \
    --policy.checkpoint_path="/home/xlab/Checkpoint/pi0fast_ckpt" \
    --policy.push_to_hub=false \
    --policy.repo_id="/home/xlab/Dataset/metaworld_mt50" \


