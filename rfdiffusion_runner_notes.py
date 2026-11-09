import torch 
import torch.nn as nn

sampler = Sampler()
sampler.sample_step()
# 谁在预测？ 谁在更新坐标？ 不是一起的吗？ 


with torch.no_grad():
    msa_prev, pair_prev, px0, state_prev, alpha, logits, plddt = self.model(
        msa_masked,
        msa_full,
        seq_in,
        xt_in,
        idx_pdb,
        t1d=t1d,
        t2d=t2d,
        xyz_t=xyz_t,
        alpha_t=alpha_t,
        msa_prev=msa_prev,
        pair_prev=pair_prev,
        state_prev=state_prev,
        t=torch.tensor(t),
        return_infer=True,
        motif_mask=self.diffusion_mask.squeeze().to(self.device), # （1，L） -》 （L, ）
    )


# 在反向去噪声的过程中只要大于最后一步就不断预测上一步的三维坐标
if t > final_step:
    seq_t_1 = nn.one_hot(seq_init, num_classes=22).to(self.device)
    x_t_1, px0 = self.denoiser.get_next_pose(
        xt=x_t,
        px0=px0,
        t=t,
        diffusion_mask=self.mask_str.squeeze(),
        align_motif=self.inf_conf.align_motif,
    )
else:
    x_t_1 = torch.clone(px0).to(x_t.device)
    seq_t_1 = torch.clone(seq_init)
    px0 = px0.to(x_t.device)



# 这里的model就是Rosstamodel 最重要的作用用于预测 Px0
# 为什么也预测seq？


 # RoseTTAFoldModel.py       # forward 总控 （在前面提到的每一步预测px0的步骤中都会走这个前向forward 总控）
 #           │
 #         ├─ Embeddings.py          # embedding
 #         ├─ Track_module.py        # attention tracks
  #        └─ SE3_network.py 等      # SE(3) 结构更新