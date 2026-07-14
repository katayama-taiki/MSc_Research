import torch
import torch.nn as nn
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
        # パラメータ Θ は十分統計量と全く同じ次元のベクトル
        self.theta = nn.Parameter(torch.zeros(stats_dim))

    def forward(self, stats):
        # 統計量とパラメータの内積を計算（エネルギーの計算などに使用）
        return torch.matmul(stats, self.theta)

def contrastive_divergence_step(model, optimizer, pos_stats, neg_stats):
    """
    1ステップ分のCDMを実行する関数
    pos_stats: 実データから計算した十分統計量 X^0
    neg_stats: MCMC等でモデルからサンプリングしたデータから計算した十分統計量 X^1
    """
    # 1. 期待値（バッチ方向の平均）を計算
    # スライドの <d log f(x) / d theta>_X^0 と _X^1 に相当
    pos_mean = pos_stats.mean(dim=0)
    neg_mean = neg_stats.mean(dim=0)
    
    # 2. 勾配の固定（定数化）
    # .detach() をつけることで、PyTorchの計算グラフから切り離し、単なる定数ベクトルにする
    grad_constant = (pos_mean - neg_mean).detach()
    
    # 3. 疑似ロス (Surrogate Loss) の定義
    # L = - (pos_mean - neg_mean) * Θ
    # この L を Θ で微分すると、ちょうど -(pos_mean - neg_mean) になる
    surrogate_loss = -torch.dot(model.theta, grad_constant)
    
    # 4. パラメータの更新
    # PyTorchは Θ_t - η * (勾配) を実行するので、結果的にスライドの式と完全に一致する
    optimizer.zero_grad()
    surrogate_loss.backward()
    optimizer.step()
    
    return surrogate_loss.item()