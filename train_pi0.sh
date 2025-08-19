torchrun --nproc_per_node=4 -m lerobot.scripts.train \
    --dataset.repo_id=metaworld \
    --dataset.root="/workspace/Dataset/metaworld_mt50_A100" \
    --policy.type=pi0fast \
    --policy.checkpoint_path="/workspace/Checkpoint/pi0fast/pi0fast_ckpt" \
    --policy.push_to_hub=false \
    --policy.repo_id="/workspace/Dataset/metaworld_mt50_A100"

