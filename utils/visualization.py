import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_ga1_convergence(ga_p50s, rs_p50s, bo_p50, results_ga1, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    axes[0].boxplot([ga_p50s, rs_p50s], tick_labels=['GA', 'RS'], patch_artist=True, boxprops=dict(facecolor='steelblue', alpha=0.7))
    axes[0].axhline(bo_p50, color='red', linestyle='--', linewidth=2, label=f'BO = {bo_p50:.4f}')
    axes[0].set_ylabel('Precision@50')
    axes[0].set_title('Precision@50: GA vs RS vs BO\n(calibración de pesos, 100 eval.)')
    axes[0].legend()
    axes[0].set_ylim(0, 1)

    all_hists = np.array([r['history'] for r in results_ga1])
    mean_hist = all_hists.mean(axis=0)
    std_hist = all_hists.std(axis=0)
    gens = range(len(mean_hist))

    axes[1].plot(gens, mean_hist, 'o-', color='steelblue', linewidth=2, label='GA (media)')
    axes[1].fill_between(gens, mean_hist - std_hist, mean_hist + std_hist, alpha=0.2, color='steelblue')
    axes[1].axhline(bo_p50, color='red', linestyle='--', label=f'BO = {bo_p50:.4f}')
    axes[1].axhline(np.mean(rs_p50s), color='gray', linestyle=':', label=f'RS media = {np.mean(rs_p50s):.4f}')
    axes[1].set_xlabel('Generación')
    axes[1].set_ylabel('Precision@50')
    axes[1].set_title('Convergencia GA Nivel 1\n(media ± std de 10 corridas)')
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)

def plot_ga2_convergence(results_ga2, rs_info, greedy_info, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['steelblue','darkorange','green']
    
    for i, r in enumerate(results_ga2):
        hist = pd.DataFrame(r['history'])
        axes[0].plot(hist['gen'], hist['rank_real'], color=colors[i], alpha=0.8, label=f'GA seed={r["seed"]}')
    axes[0].axhline(int(rs_info.get('rank_real',0)), color='red', linestyle='--', label=f'RS rank={int(rs_info.get("rank_real",0))}')
    axes[0].axhline(greedy_info['rank_real'], color='gold', linestyle='--', linewidth=2, label=f'Greedy K=20 rank={greedy_info["rank_real"]}')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Generación')
    axes[0].set_ylabel('Rank real (log, menor=mejor)')
    axes[0].set_title('Convergencia GA Nivel 2\nvs RS y Greedy')
    axes[0].legend(fontsize=9)
    axes[0].invert_yaxis()

    for i, r in enumerate(results_ga2):
        hist = pd.DataFrame(r['history'])
        axes[1].plot(hist['gen'], hist['acc_valid'], color=colors[i], alpha=0.8, label=f'GA seed={r["seed"]}')
    axes[1].axhline(rs_info.get('acc_valid',0), color='red', linestyle='--', label='RS')
    axes[1].axhline(greedy_info['acc_valid'], color='gold', linestyle='--', linewidth=2, label='Greedy K=20')
    axes[1].set_xlabel('Generación')
    axes[1].set_ylabel('acc_valid real (%)')
    axes[1].set_title('Accuracy real de la mejor arq.\npor generación')
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)

def plot_nsga2_pareto(df_pareto_real, df_pareto_nsga2, df_pareto_rs, hist_fronts, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(df_pareto_real['params_k'], df_pareto_real['acc_valid'], c='gold', s=40, zorder=3, label=f'Pareto real ({len(df_pareto_real)} arqs)', edgecolors='black', linewidths=0.5)
    ax.plot(df_pareto_real.sort_values('params_k')['params_k'], df_pareto_real.sort_values('params_k')['acc_valid'], 'gold', alpha=0.5, linewidth=1)
    ax.scatter(df_pareto_nsga2['params_k'], df_pareto_nsga2['acc_valid'], c='steelblue', s=60, zorder=4, label=f'NSGA-II ({len(df_pareto_nsga2)} arqs)', edgecolors='black', linewidths=0.5)
    ax.scatter(df_pareto_rs['params_k'], df_pareto_rs['acc_valid'], c='tomato', s=40, zorder=2, alpha=0.7, label=f'RS Pareto ({len(df_pareto_rs)} arqs)', marker='^')
    ax.set_xlabel('Parámetros (K)')
    ax.set_ylabel('acc_valid real (%)')
    ax.set_title('Frentes de Pareto: NSGA-II vs RS vs Ground Truth')
    ax.legend(fontsize=9)

    ax2 = axes[1]
    best_acc_by_gen = [max(p[0] for p in front) for front in hist_fronts]
    front_size = [len(front) for front in hist_fronts]
    ax2.plot(best_acc_by_gen, color='steelblue', label='Mejor proxy en frente')
    ax2b = ax2.twinx()
    ax2b.plot(front_size, color='darkorange', linestyle='--', label='Tamaño del frente')
    ax2.set_xlabel('Generación')
    ax2.set_ylabel('Mejor proxy score', color='steelblue')
    ax2b.set_ylabel('Soluciones en frente de Pareto', color='darkorange')
    ax2.set_title('Convergencia NSGA-II')
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1+lines2, labels1+labels2, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)

def plot_final_results(final_results, df_pareto_ext, benchmark_best, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['steelblue', 'darkorange']
    
    for i, r in enumerate(final_results):
        axes[0].plot(r['history_200'], color=colors[i], linewidth=2, label=f'C={r["channel"]}, nc={r["num_cells"]}, {r["params_k"]:.0f}K params')
    axes[0].axhline(benchmark_best, color='red', linestyle='--', linewidth=2, label=f'Benchmark best ({benchmark_best}%)')
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Entrenamiento arquitecturas extendidas vs benchmark')
    axes[0].legend(fontsize=9)

    axes[1].scatter(df_pareto_ext['params_k'], df_pareto_ext['proxy_score'], c='steelblue', s=50, alpha=0.7, label='Frente Pareto extendido (proxy)')
    for r in final_results:
        axes[1].scatter(r['params_k'], r['proxy'], c='gold', s=200, zorder=5, edgecolors='black', linewidths=2)
        axes[1].annotate(f'{r["acc_200ep"]:.2f}%', (r['params_k'], r['proxy']), textcoords='offset points', xytext=(5,5), fontsize=9, color='red', fontweight='bold')
    axes[1].set_xlabel('Parámetros (K)')
    axes[1].set_ylabel('Proxy score')
    axes[1].set_title('Frente de Pareto extendido\n(oro = entrenadas, etiqueta = acc real)')
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
