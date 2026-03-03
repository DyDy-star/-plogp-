# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
自定义Trainer，在reward计算前先计算完整词表的熵
"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_reward
import numpy as np


class EntropyRewardTrainer(RayPPOTrainer):
    """
    扩展RayPPOTrainer，在reward计算前先计算完整词表的熵，
    并将熵值传递给reward函数
    """
    
    def _compute_reward_with_entropy(self, batch, reward_fn):
        """
        计算奖励，但首先计算完整词表的熵并传递给reward函数
        
        Args:
            batch: DataProto对象
            reward_fn: 奖励函数
            
        Returns:
            reward_tensor, reward_extra_infos_dict
        """
        # 首先计算完整词表的熵
        # 设置calculate_entropy=True以请求熵计算
        batch.meta_info["calculate_entropy"] = True
        
        # 调用actor的compute_log_prob来获取熵
        # 这会使用完整词表的logits计算精确的Shannon熵
        entropy_result = self.actor_rollout_wg.compute_log_prob(batch)
        
        # 提取token级别的熵（基于完整词表logits计算）
        if "entropys" in entropy_result.batch:
            token_entropies = entropy_result.batch["entropys"]  # shape: (batch_size, response_length)
            
            # 将熵值传递到non_tensor_batch中，供reward函数使用
            # 注意：需要考虑response_mask，只保留有效的token熵
            response_mask = batch.batch.get("response_mask")
            if response_mask is None:
                from verl.trainer.ppo.ray_trainer import compute_response_mask
                response_mask = compute_response_mask(batch)
            
            # 对每个样本，提取有效的token熵
            batch_size = token_entropies.shape[0]
            token_entropies_list = []
            
            for i in range(batch_size):
                sample_entropies = token_entropies[i]  # (response_length,)
                sample_mask = response_mask[i]  # (response_length,)
                
                # 只保留有效的token
                valid_entropies = sample_entropies[sample_mask.bool()]
                token_entropies_list.append(valid_entropies.cpu().numpy())
            
            # 存储到non_tensor_batch中
            batch.non_tensor_batch["token_entropies"] = np.array(token_entropies_list, dtype=object)
            
            print(f"[EntropyReward] 已为 {batch_size} 个样本计算完整词表熵")
        else:
            print(f"[EntropyReward] 警告：compute_log_prob未返回熵值")
        
        # 现在计算reward，reward函数可以通过extra_info访问token_entropies
        reward_tensor, reward_extra_infos_dict = compute_reward(batch, reward_fn)
        
        return reward_tensor, reward_extra_infos_dict


