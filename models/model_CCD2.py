import torch
import numpy as np 
import torch.nn as nn
from torch import nn
from einops import reduce
from torch.nn import ReLU
from models.layers.cross_attention import FeedForward, MMAttentionLayer
import pdb
import math
import pandas as pd
import torch.nn.functional as F

def exists(val):
    return val is not None

def SNN_Block(dim1, dim2, dropout=0.25):
    r"""
    Multilayer Reception Block w/ Self-Normalization (Linear + ELU + Alpha Dropout)
    """
    import torch.nn as nn
    return nn.Sequential(
            nn.Linear(dim1, dim2),
            nn.ELU(),
            nn.AlphaDropout(p=dropout, inplace=False))

class CCD_SurvPath(nn.Module):
    def __init__(
        self, 
        omic_sizes=[100, 200, 300, 400, 500, 600],  # 每个通路的基因数量
        raw_gene_dim=4999,  # 原始基因表达数据维度
        wsi_embedding_dim=1024,
        dropout=0.1,
        num_classes=4,
        wsi_projection_dim=256,
        omic_names = [],
        pathway_gene_matrix=None,  # 通路-基因关系矩阵 [num_pathways, raw_gene_dim]
        ):
        super(CCD_SurvPath, self).__init__()

        #---> general props
        self.num_pathways = len(omic_sizes)
        self.raw_gene_dim = raw_gene_dim
        self.dropout = dropout

        #---> 通路-基因关系矩阵（二进制矩阵，1表示基因属于该通路）
        if pathway_gene_matrix is not None:
            self.register_buffer('pathway_gene_matrix', pathway_gene_matrix)  # [num_pathways, raw_gene_dim]
        else:
            # 如果没有提供，创建一个随机的关系矩阵作为示例
            self.register_buffer('pathway_gene_matrix', torch.randint(0, 2, (self.num_pathways, raw_gene_dim)).float())

        #---> omics preprocessing for captum
        if omic_names != []:
            self.omic_names = omic_names
            all_gene_names = []
            for group in omic_names:
                all_gene_names.append(group)
            all_gene_names = np.asarray(all_gene_names)
            all_gene_names = np.concatenate(all_gene_names)
            all_gene_names = np.unique(all_gene_names)
            all_gene_names = list(all_gene_names)
            self.all_gene_names = all_gene_names

        #---> wsi props
        self.wsi_embedding_dim = wsi_embedding_dim 
        self.wsi_projection_dim = wsi_projection_dim

        self.wsi_projection_net = nn.Sequential(
            nn.Linear(self.wsi_embedding_dim, self.wsi_projection_dim),
        )

        #---> 基因分支的因果干预参数
        self.init_causal_gene_branch()

        #---> 通路分支：处理已分组的通路特征
        self.init_per_path_model(omic_sizes)

        #---> cross attention props
        self.identity = nn.Identity() 
        self.cross_attender = MMAttentionLayer(
            dim=self.wsi_projection_dim,
            dim_head=self.wsi_projection_dim // 2,
            heads=1,
            residual=False,
            dropout=0.1,
            num_pathways = self.num_pathways
        )

        #---> logits props 
        self.num_classes = num_classes
        self.feed_forward = FeedForward(self.wsi_projection_dim // 2, dropout=dropout)
        self.layer_norm = nn.LayerNorm(self.wsi_projection_dim // 2)

        # 各分支的生存预测头
        hidden_dim = 256
        self.gene_survival_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.num_classes)
        )
        
        self.pathway_survival_head = nn.Sequential(
            nn.Linear(self.wsi_projection_dim // 2, int(self.wsi_projection_dim/4)),
            nn.ReLU(),
            nn.Linear(int(self.wsi_projection_dim/4), self.num_classes)
        )
        
        self.wsi_survival_head = nn.Sequential(
            nn.Linear(self.wsi_projection_dim, int(self.wsi_projection_dim/4)),
            nn.ReLU(),
            nn.Linear(int(self.wsi_projection_dim/4), self.num_classes)
        )
        
        # 最终融合头
        self.to_logits = nn.Sequential(
                nn.Linear(self.wsi_projection_dim, int(self.wsi_projection_dim/4)),
                nn.ReLU(),
                nn.Linear(int(self.wsi_projection_dim/4), self.num_classes)
            )
    
    def init_causal_gene_branch(self):
        """
        初始化基因分支的因果干预参数
        """
        hidden_dim = 256
        
        # 基因特征编码器
        self.gene_encoder = nn.Sequential(
            nn.Linear(self.raw_gene_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 因果干预的核心参数
        # 1. 基因-通路交互计算
        self.gene_pathway_interaction0 = nn.Sequential(
            nn.Linear(self.raw_gene_dim + self.num_pathways, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.gene_pathway_interaction = nn.Sequential(
            nn.Linear(self.raw_gene_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 2. 通路先验概率编码
        self.pathway_prior_encoder = nn.Sequential(
            nn.Linear(self.num_pathways, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.num_pathways)
        )
        
        # 3. 因果效应计算网络
        self.causal_effect_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 4. 最终融合参数
        self.Wt_direct = nn.Linear(self.raw_gene_dim, hidden_dim)  # 直接基因效应
        self.Wu_confound = nn.Linear(hidden_dim, hidden_dim)      # 混杂效应调整
    def compute_pathway_activations0(self, raw_gene_expr):
        """
        计算基因表达在各通路上的激活强度 - 使用ssGSEA方法
        """
        batch_size, num_genes = raw_gene_expr.shape
        num_pathways = self.pathway_gene_matrix.shape[0]
        
        # 对每个样本的基因表达进行排序
        sorted_expr, sorted_indices = torch.sort(raw_gene_expr, dim=1, descending=True)
        
        pathway_activations = torch.zeros(batch_size, num_pathways, device=raw_gene_expr.device)
        
        for pathway_idx in range(num_pathways):
            # 获取当前通路的基因mask
            pathway_genes = self.pathway_gene_matrix[pathway_idx] > 0  # [num_genes]
            pathway_size = torch.sum(pathway_genes).float()
            
            if pathway_size == 0:
                continue
                
            for sample_idx in range(batch_size):
                # 获取当前样本的排序基因索引
                sample_sorted_indices = sorted_indices[sample_idx]
                sample_sorted_expr = sorted_expr[sample_idx]
                
                # 计算通路基因在排序列表中的位置和表达值
                pathway_positions = []
                pathway_values = []
                
                for pos in range(num_genes):
                    gene_idx = sample_sorted_indices[pos]
                    if pathway_genes[gene_idx]:
                        pathway_positions.append(pos)
                        pathway_values.append(sample_sorted_expr[pos])
                
                if len(pathway_positions) == 0:
                    continue
                    
                pathway_positions = torch.tensor(pathway_positions, device=raw_gene_expr.device)
                pathway_values = torch.tensor(pathway_values, device=raw_gene_expr.device)
                
                # ssGSEA权重计算 (alpha=0.25)
                weights = torch.abs(pathway_values) ** 0.25
                if torch.sum(weights) > 0:
                    weights = weights / torch.sum(weights)
                
                # 计算累积富集分数
                cumulative_hit = torch.zeros(num_genes + 1, device=raw_gene_expr.device)
                cumulative_miss = torch.zeros(num_genes + 1, device=raw_gene_expr.device)
                
                # 标记通路基因位置
                hit_mask = torch.zeros(num_genes, dtype=torch.bool, device=raw_gene_expr.device)
                hit_mask[pathway_positions] = True
                
                weight_idx = 0
                for i in range(num_genes):
                    cumulative_hit[i + 1] = cumulative_hit[i]
                    cumulative_miss[i + 1] = cumulative_miss[i]
                    
                    if hit_mask[i]:
                        cumulative_hit[i + 1] += weights[weight_idx]
                        weight_idx += 1
                    else:
                        cumulative_miss[i + 1] += 1.0 / (num_genes - pathway_size)
                
                # 计算富集分数 (最大偏差)
                running_sum = cumulative_hit[1:] - cumulative_miss[1:]
                max_deviation = torch.max(torch.abs(running_sum))
                
                # 保持符号
                max_pos = torch.argmax(torch.abs(running_sum))
                if running_sum[max_pos] < 0:
                    max_deviation *= -1
                    
                pathway_activations[sample_idx, pathway_idx] = max_deviation
        
        return pathway_activations

    def compute_pathway_activations(self, raw_gene_expr):
        """
        计算基因表达在各通路上的激活强度
        """
        # 使用通路-基因矩阵计算每个通路的激活强度
        pathway_activations = torch.matmul(raw_gene_expr, self.pathway_gene_matrix.T)  # [batch_size, num_pathways]
        
        # 归一化：每个通路除以该通路包含的基因数量
        pathway_gene_counts = torch.sum(self.pathway_gene_matrix, dim=1, keepdim=True)  # [num_pathways, 1]
        pathway_activations = pathway_activations / (pathway_gene_counts.T + 1e-8)  # [batch_size, num_pathways]
        
        return pathway_activations

    def causal_intervention_gene_branch(self, raw_gene_expr):
        """
        对原始基因表达数据应用因果干预 - 使用后门调整消除偏差
        
        理论基础：
        在基因表达与生存分析中，通路激活往往是基因表达和生存结果的共同原因（混杂因子）。
        为了估计基因表达对生存的因果效应，我们需要消除通路混杂带来的偏差。
        
        因果图：Pathway → Gene → Survival
                    ↓           ↗
                    ——————————————
        
        后门调整公式: P(Y|do(Gene)) = Σ_pathway P(Y|Gene,pathway) * P(pathway)
        
        其中：
        - Gene: 基因表达数据（治疗变量）
        - Pathway: 通路激活（混杂因子，满足后门准则）
        - Y: 生存结果（结果变量）
        
        实现步骤：
        1. 识别混杂因子：计算通路激活强度
        2. 估计P(pathway)：通路的边际概率分布
        3. 计算P(Y|Gene,pathway=k)：在每个通路条件下的基因效应
        4. 按概率加权平均：得到去偏的因果效应
        
        Args:
            raw_gene_expr: [batch_size, raw_gene_dim] 原始基因表达数据
        
        Returns:
            final_causal_features: [batch_size, hidden_dim] 去偏后的基因特征
        """
        batch_size = raw_gene_expr.shape[0]
        
        # Step 1: 计算通路激活强度（识别混杂因子）
        # 这些通路激活代表了可能影响基因表达和生存结果的共同原因
        pathway_activations = self.compute_pathway_activations(raw_gene_expr)  # [batch_size, num_pathways]
        
        # Step 2: 估计通路的边际概率 P(pathway)
        # 这是后门调整中的权重，代表每个通路在当前样本中的重要性
        pathway_marginal_probs = torch.softmax(
            self.pathway_prior_encoder(pathway_activations), dim=-1
        )  # [batch_size, num_pathways]
        
        # Step 3: 计算条件效应 P(Y|Gene, pathway=k) for each pathway k
        # 这是后门调整的核心：分别计算基因在每个通路条件下的效应
        conditional_effects = []
        
        for pathway_idx in range(self.num_pathways):
            # 获取当前通路的激活值作为条件
            current_pathway_activation = pathway_activations[:, pathway_idx:pathway_idx+1]  # [batch_size, 1]
            
            # 构建条件输入：基因表达 + 当前通路条件
            # 这确保我们计算的是 P(Y|Gene, Pathway=pathway_idx)
            conditional_input = torch.cat([
                raw_gene_expr,  # 基因表达（治疗变量）
                current_pathway_activation  # 当前通路作为条件（混杂因子的特定值）
            ], dim=1)  # [batch_size, raw_gene_dim + 1]
            
            # 计算在当前通路条件下的基因效应
            conditional_gene_effect = self.gene_pathway_interaction(conditional_input)
            conditional_effects.append(conditional_gene_effect)
        
        # 将所有条件效应堆叠
        conditional_effects = torch.stack(conditional_effects, dim=1)  # [batch_size, num_pathways, hidden_dim]
        
        # Step 4: 应用后门调整公式
        # P(Y|do(Gene)) = Σ_k P(Y|Gene, Pathway=k) * P(Pathway=k)
        pathway_weights = pathway_marginal_probs.unsqueeze(-1)  # [batch_size, num_pathways, 1]
        causal_gene_effect = torch.sum(
            pathway_weights * conditional_effects, dim=1
        )  # [batch_size, hidden_dim]
        
        # Step 5: 结合直接基因效应和调整后的因果效应
        # 直接效应：基因对结果的直接影响（不通过通路）
        direct_gene_effect = self.Wt_direct(raw_gene_expr)
        
        # 混杂调整：通过后门调整得到的去偏效应
        adjusted_causal_effect = self.Wu_confound(causal_gene_effect)
        
        # 最终的去偏基因特征
        debiased_gene_features = direct_gene_effect + adjusted_causal_effect
        
        # 通过因果效应网络进一步处理
        final_causal_features = self.causal_effect_network(debiased_gene_features)
        
        return final_causal_features
    
    def causal_intervention_gene_branch2(self, raw_gene_expr):
        # 直接使用注意力机制处理所有通路
        pathway_activations = self.compute_pathway_activations(raw_gene_expr)
        
        # 使用scaled dot-product attention (类似原文)
        Q = self.Wq(raw_gene_expr)  # [batch, hidden]
        K = self.Wk(pathway_activations)  # [batch, num_pathways, hidden]
        V = self.Wv(pathway_activations)
        
        attention_weights = F.softmax(Q @ K.T / np.sqrt(self.wsi_projection_dim), dim=-1)
        confounding_effect = attention_weights @ V
        
        # 组合直接和混杂效应
        debiased_features = self.Wt_direct(raw_gene_expr) + self.Wu_confound(confounding_effect)
    def init_per_path_model(self, omic_sizes):
        """
        初始化每个通路的编码网络
        """
        hidden = [256, 256]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)    

    def forward0(self, return_branches=False, use_causal_intervention=True, **kwargs):
        """
        前向传播
        """
        wsi = kwargs['x_path']
        
        # 获取原始基因表达数据
        raw_gene_expr = kwargs['raw_gene_expr']  # [batch_size, raw_gene_dim]
        
        # 获取已分组的通路特征
        x_omic = [kwargs['x_omic%d' % i] for i in range(1,self.num_pathways+1)]
        
        mask = None
        return_attn = kwargs.get("return_attn", False)
        
        # === 基因分支：对原始基因表达应用因果干预 ===
        if use_causal_intervention:
            causal_gene_features = self.causal_intervention_gene_branch(raw_gene_expr)
        else:
            causal_gene_features = self.gene_encoder(raw_gene_expr)
        
        gene_logits = self.gene_survival_head(causal_gene_features)
        
        # === 通路分支：处理已分组的通路特征 ===
        h_omic = [self.sig_networks[idx].forward(sig_feat.float()) for idx, sig_feat in enumerate(x_omic)]
        h_omic_bag = torch.stack(h_omic, dim=0).unsqueeze(0)  # [batch_size, num_pathways, feature_dim]

        # === 病理分支 ===
        wsi_embed = self.wsi_projection_net(wsi) # [batch_size, 4090, wsi_projection_dim]

        # === 多模态融合 ===
        tokens = torch.cat([h_omic_bag, wsi_embed], dim=1)
        tokens = self.identity(tokens)
        
        if return_attn:
            mm_embed, attn_pathways, cross_attn_pathways, cross_attn_histology = self.cross_attender(x=tokens, mask=mask if mask is not None else None, return_attention=True)
        else:
            mm_embed = self.cross_attender(x=tokens, mask=mask if mask is not None else None, return_attention=False)

        # feedforward and layer norm 
        mm_embed = self.feed_forward(mm_embed)
        mm_embed = self.layer_norm(mm_embed)
        
        # 获取各分支特征
        paths_postSA_embed = mm_embed[:, :self.num_pathways, :]
        paths_postSA_embed = torch.mean(paths_postSA_embed, dim=1)

        wsi_postSA_embed = mm_embed[:, self.num_pathways:, :]
        wsi_postSA_embed = torch.mean(wsi_postSA_embed, dim=1)

        # 各分支的预测
        pathway_logits = self.pathway_survival_head(paths_postSA_embed)
        wsi_logits = self.wsi_survival_head(wsi_postSA_embed)
        
        if return_branches:
            return gene_logits, pathway_logits, wsi_logits
        
        # 最终融合预测
        embedding = torch.cat([paths_postSA_embed, wsi_postSA_embed], dim=1)
        fusion_logits = self.to_logits(embedding)

        if return_attn:
            return fusion_logits, attn_pathways, cross_attn_pathways, cross_attn_histology
        else:
            return fusion_logits
        
    def forward(self, return_branches=False, use_causal_intervention=True, **kwargs):
        """
        前向传播
        """
        wsi = kwargs['x_path']
        
        # 获取原始基因表达数据
        raw_gene_expr = kwargs['raw_gene_expr']  # [batch_size, raw_gene_dim]
        
        # 获取已分组的通路特征
        x_omic = [kwargs['x_omic%d' % i] for i in range(1,self.num_pathways+1)]
        
        mask = None
        return_attn = kwargs.get("return_attn", False)
        
        # === 基因分支：对原始基因表达应用因果干预 ===
        if use_causal_intervention:
            causal_gene_features = self.causal_intervention_gene_branch(raw_gene_expr)
        else:
            causal_gene_features = self.gene_encoder(raw_gene_expr)
        
        gene_logits = self.gene_survival_head(causal_gene_features)
        
        # === 通路分支：处理已分组的通路特征 ===
        h_omic = [self.sig_networks[idx].forward(sig_feat.float()) for idx, sig_feat in enumerate(x_omic)]
        h_omic_bag = torch.stack(h_omic, dim=0).unsqueeze(0)  # [batch_size, num_pathways, feature_dim]

        # === 病理分支 ===
        wsi_embed = self.wsi_projection_net(wsi) # [batch_size, 4090, wsi_projection_dim]

        # === 多模态融合 ===
        tokens = torch.cat([h_omic_bag, wsi_embed], dim=1)
        tokens = self.identity(tokens)
        
        if return_attn:
            mm_embed, attn_pathways, cross_attn_pathways, cross_attn_histology = self.cross_attender(x=tokens, mask=mask if mask is not None else None, return_attention=True)
        else:
            mm_embed = self.cross_attender(x=tokens, mask=mask if mask is not None else None, return_attention=False)

        # feedforward and layer norm 
        mm_embed = self.feed_forward(mm_embed)
        mm_embed = self.layer_norm(mm_embed)
        
        # 获取各分支特征
        paths_postSA_embed = mm_embed[:, :self.num_pathways, :]
        paths_postSA_embed = torch.mean(paths_postSA_embed, dim=1)

        wsi_postSA_embed = mm_embed[:, self.num_pathways:, :]
        wsi_postSA_embed = torch.mean(wsi_postSA_embed, dim=1)

        # 各分支的预测
        # pathway_logits = self.pathway_survival_head(paths_postSA_embed)
        # wsi_logits = self.wsi_survival_head(wsi_postSA_embed)
        wsi_logits = self.wsi_survival_head(wsi_embed.mean(dim=1).squeeze(1))
        # 最终融合预测
        embedding = torch.cat([paths_postSA_embed, wsi_postSA_embed], dim=1)
        fusion_logits = self.to_logits(embedding)
        
        if return_branches:
            return gene_logits, wsi_logits, fusion_logits

        if return_attn:
            return fusion_logits, attn_pathways, cross_attn_pathways, cross_attn_histology
        else:
            return fusion_logits
    
    def counterfactual_inference(self, **kwargs):
        """
        反事实推理：去除病理图像的直接偏差效应
        """
        wsi = kwargs['x_path']
        
        # === 现实世界的预测 ===
        gene_logits, pathway_logits, wsi_logits = self.forward(return_branches=True, use_causal_intervention=True, **kwargs)
        fusion_logits = self.forward(return_branches=False, use_causal_intervention=True, **kwargs)
        
        # 计算总效应（类似原文的融合函数F(Yt, Yi, Yc)）
        total_effect = fusion_logits + torch.tanh(gene_logits) + torch.tanh(pathway_logits) + torch.tanh(wsi_logits)
        
        # === 反事实世界：没有病理信息 ===
        zero_wsi = torch.zeros_like(wsi)
        counterfactual_kwargs = kwargs.copy()
        counterfactual_kwargs['x_path'] = zero_wsi
        
        gene_ref, pathway_ref, wsi_ref = self.forward(return_branches=True, use_causal_intervention=True, **counterfactual_kwargs)
        fusion_ref = self.forward(return_branches=False, use_causal_intervention=True, **counterfactual_kwargs)
        
        reference_effect = fusion_ref + torch.tanh(gene_ref) + torch.tanh(pathway_ref) + torch.tanh(wsi_ref)
        
        # === 反事实：只有病理信息 ===
        wsi_only_kwargs = kwargs.copy()
        # 将基因和通路特征设为零
        wsi_only_kwargs['raw_gene_expr'] = torch.zeros_like(kwargs['raw_gene_expr'])
        for i in range(1, self.num_pathways+1):
            wsi_only_kwargs['x_omic%d' % i] = torch.zeros_like(kwargs['x_omic%d' % i])
        
        gene_wsi_only, pathway_wsi_only, wsi_wsi_only = self.forward(return_branches=True, use_causal_intervention=True, **wsi_only_kwargs)
        fusion_wsi_only = self.forward(return_branches=False, use_causal_intervention=True, **wsi_only_kwargs)
        
        wsi_only_effect = fusion_wsi_only + torch.tanh(gene_wsi_only) + torch.tanh(pathway_wsi_only) + torch.tanh(wsi_wsi_only)
        
        # 计算病理图像的直接偏差效应（NDE）
        wsi_direct_effect = wsi_only_effect - reference_effect
        
        # 去偏后的最终预测（TIE = TE - NDE）
        debiased_logits = total_effect - wsi_direct_effect
        
        return debiased_logits