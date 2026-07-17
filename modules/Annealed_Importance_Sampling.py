import torch
import torch.nn.functional as F

def compute_exact_kl_divergence(model_k, model_k_minus_1, S_batch, calc_stats_fn, k, num_classes=10):
    """
    [完全解析版] Tが有限の離散クラス(10クラス)であることを利用し、
    AISによるMCMCサンプリングを回避して条件付き分布のD_KL(P_k || P_{k-1})の厳密解を計算する。
    """
    batch_size = S_batch.shape[0]
    device = S_batch.device
    
    energies_k = []
    energies_k_minus_1 = []
    
    with torch.no_grad():
        for c in range(num_classes):
            T_dummy = torch.zeros(batch_size, num_classes, device=device)
            T_dummy[:, c] = 1.0
            
            # 関数呼び出しは1回にまとめる（計算効率化）
            stats_dict = calc_stats_fn(T_dummy, S_batch)
            
            # k次モデルの統計量とエネルギー
            stats_k = stats_dict[k]
            e_k = torch.matmul(stats_k, model_k.theta)
            energies_k.append(e_k)
            
            # (k-1)次モデルの統計量とエネルギー
            if k - 1 == 0:
                stats_k_minus_1 = T_dummy # 0次項はT_dummyそのものを使用
            else:
                stats_k_minus_1 = stats_dict[k - 1]
                
            e_k_minus_1 = torch.matmul(stats_k_minus_1, model_k_minus_1.theta)
            energies_k_minus_1.append(e_k_minus_1)
            
        E_k_tensor = torch.stack(energies_k, dim=1)
        E_k_minus_1_tensor = torch.stack(energies_k_minus_1, dim=1)
        
        log_Z_k = torch.logsumexp(E_k_tensor, dim=1)
        log_Z_k_minus_1 = torch.logsumexp(E_k_minus_1_tensor, dim=1)
        
        p_k_given_S = F.softmax(E_k_tensor, dim=1)
        
        energy_diff = E_k_tensor - E_k_minus_1_tensor
        expected_energy_diff = torch.sum(p_k_given_S * energy_diff, dim=1)
        
        kl_S = expected_energy_diff - log_Z_k + log_Z_k_minus_1
        
        return kl_S.mean().item()