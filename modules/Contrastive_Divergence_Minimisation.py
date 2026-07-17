import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations

class InteractionSufficientStatistics:
    def __init__(self, max_order=5):
        """
        max_order: 相互作用の最大次数（デフォルトで5に制限）
        """
        self.max_order = max_order
        self._cached_num_sources = None
        self._combinations_by_order = {}

    def _update_combinations(self, num_sources):
        if self._cached_num_sources != num_sources:
            # ソース変数の数と max_order の小さい方を上限とする
            limit_order = min(self.max_order, num_sources)
            self._combinations_by_order = {
                k: list(combinations(range(num_sources), k)) 
                for k in range(1, limit_order + 1)
            }
            self._cached_num_sources = num_sources

    def __call__(self, T, S):
        """
        T: 予測結果のワンホットテンソル (batch_size, num_classes)
        S: ソース変数のテンソル (batch_size, num_sources)
        """
        num_sources = S.shape[1]
        self._update_combinations(num_sources)
        
        cumulative_stats = []
        stats_for_models = {}
        
        # キャッシュされた次数（最大 max_order まで）の範囲でループ
        for k in self._combinations_by_order.keys():
            order_k_stats = []
            for idx in self._combinations_by_order[k]:
                # ソース変数の積を計算 (batch_size, 1)
                interaction_term = S[:, idx].prod(dim=1, keepdim=True)
                
                # ワンホットT と掛け合わせる
                # ブロードキャストにより (batch_size, num_classes) に拡張される
                order_k_stats.append(T * interaction_term)
            
            # k次のみの組み合わせテンソルを結合 (batch_size, num_classes * k次の組み合わせ数)
            cumulative_stats.append(torch.cat(order_k_stats, dim=1))
            
            # 1次からk次までを累積的に結合して辞書に格納
            stats_for_models[k] = torch.cat(cumulative_stats, dim=1)
            
        return stats_for_models

class ExponentialFamilyModel(nn.Module):
    def __init__(self, stats_dim):
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(stats_dim))

    def sample_T_given_S(self, S_batch, calc_stats_fn, k):
        """
        現在のパラメータ θ に基づき、条件付き分布 p(T|S; θ) から T をサンプリングする。
        """
        batch_size = S_batch.shape[0]
        num_classes = 10
        device = S_batch.device
        
        energies = []
        
        # MCMCのサンプリング時は勾配計算をオフにし、メモリを節約
        with torch.no_grad():
            for c in range(num_classes):
                T_dummy = torch.zeros(batch_size, num_classes, device=device)
                T_dummy[:, c] = 1.0
                
                stats_c = calc_stats_fn(T_dummy, S_batch)[k]
                energy_c = torch.matmul(stats_c, self.theta)
                energies.append(energy_c)
                
            energies_tensor = torch.stack(energies, dim=1) # (batch, 10)
            
            # 確率 p(T|S) を計算
            p_T_given_S = F.softmax(energies_tensor, dim=1)
            
            # カテゴリカル分布から T をサンプリング (MCMCステップ)
            sampled_class_indices = torch.multinomial(p_T_given_S, num_samples=1).squeeze()
            
            # サンプルされた T をワンホットベクトルに変換
            T_sampled = F.one_hot(sampled_class_indices, num_classes=10).float()
            
        return T_sampled

    def compute_negative_stats_mean(self, S_batch, calc_stats_fn, k):
        """
        [完全解析版]
        10クラスすべての十分統計量を計算し、Softmax確率で重み付けた期待値を厳密に算出する。
        """
        batch_size = S_batch.shape[0]
        num_classes = 10
        device = S_batch.device
        
        stats_all_classes = []
        energies = []
        
        # 1. 10クラスすべての統計量とエネルギー E_k(T=c, S) を計算
        for c in range(num_classes):
            T_dummy = torch.zeros(batch_size, num_classes, device=device)
            T_dummy[:, c] = 1.0
            
            stats_c = calc_stats_fn(T_dummy, S_batch)[k]
            stats_all_classes.append(stats_c)
            
            energy_c = torch.matmul(stats_c, self.theta)
            energies.append(energy_c)
            
        energies_tensor = torch.stack(energies, dim=1) # (batch, 10)
        
        # 2. 条件付き確率 p(T|S) をSoftmaxで計算
        p_T_given_S = F.softmax(energies_tensor, dim=1)
        
        # 3. 確率で重み付けして期待値 E_{p(T|S)}[\phi(T, S)] を計算
        expected_stats = torch.zeros_like(stats_all_classes[0])
        for c in range(num_classes):
            prob_c = p_T_given_S[:, c].unsqueeze(1) # ブロードキャスト用の次元追加
            expected_stats += prob_c * stats_all_classes[c]
            
        # 最後にミニバッチ方向での平均をとる E_{p_data(S)}[...]
        return expected_stats.mean(dim=0)

    def contrastive_divergence_step(self, optimizer, pos_mean, neg_mean):
        """
        実データのサンプル平均と、MCMCサンプルの平均の差からパラメータを更新する
        """
        # 勾配 = <φ>_data - <φ>_model
        grad_constant = (pos_mean - neg_mean).detach()
        surrogate_loss = -torch.dot(self.theta, grad_constant)
        
        optimizer.zero_grad()
        surrogate_loss.backward()
        optimizer.step()
        
        return surrogate_loss.item()