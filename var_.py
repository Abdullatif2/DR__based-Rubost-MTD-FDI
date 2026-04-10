# # # # save as compare_score_history_variance.py
# # # import os
# # # import numpy as np
# # # import pandas as pd

# # # # ROOT = "Robust_MTD-main_BigBus_2"
# # # ROOT = "."
# # # PATH_ALL = os.path.join(
# # #     ROOT,
# # #     "td3_reward_variance_comparison",
# # #     "perturb_all_lines_ratio_0.3",
# # #     "with_ep_schdlr_ptienc_2_0.001_0.001_dropout_03_first_2_layers_old_noise",
# # #     "score_plot",
# # # )
# # # PATH_SELECTED = os.path.join(
# # #     ROOT,
# # #     "td3_reward_variance_comparison",
# # #     "minimum_full_rank_ratio_0.3",
# # #     "with_ep_schdlr_ptienc_2_0.001_0.001_dropout_03_first_2_layers_old_noise",
# # #     "score_plot",
# # # )

# # # def load_scores(score_dir: str) -> np.ndarray:
# # #     csv_path = os.path.join(score_dir, "score_history.csv")
# # #     npy_path = os.path.join(score_dir, "score_history.npy")

# # #     if os.path.isfile(csv_path):
# # #         s = pd.read_csv(csv_path, header=None).squeeze("columns")
# # #         return s.to_numpy(dtype=float).ravel()
# # #     if os.path.isfile(npy_path):
# # #         return np.load(npy_path).astype(float).ravel()

# # #     raise FileNotFoundError(
# # #         f"Neither score_history.csv nor score_history.npy found in: {score_dir}"
# # #     )

# # # def basic_stats(x: np.ndarray, name: str):
# # #     x = np.asarray(x, dtype=float).ravel()
# # #     return {
# # #         "name": name,
# # #         "count": x.size,
# # #         "mean": np.mean(x),
# # #         "var_pop": np.var(x, ddof=0),
# # #         "var_sample": np.var(x, ddof=1) if x.size > 1 else float("nan"),
# # #         "std_pop": np.std(x, ddof=0),
# # #         "std_sample": np.std(x, ddof=1) if x.size > 1 else float("nan"),
# # #         "min": np.min(x),
# # #         "max": np.max(x),
# # #         "p05": np.percentile(x, 5),
# # #         "median": np.median(x),
# # #         "p95": np.percentile(x, 95),
# # #     }

# # # def print_stats(stats):
# # #     print(f"\n== {stats['name']} ==")
# # #     print(f"n={stats['count']}")
# # #     print(f"mean={stats['mean']:.6f}")
# # #     print(f"var(pop)={stats['var_pop']:.6f}, var(sample)={stats['var_sample']:.6f}")
# # #     print(f"std(pop)={stats['std_pop']:.6f}, std(sample)={stats['std_sample']:.6f}")
# # #     print(f"min={stats['min']:.6f}, p05={stats['p05']:.6f}, "
# # #           f"median={stats['median']:.6f}, p95={stats['p95']:.6f}, max={stats['max']:.6f}")

# # # def try_levene(x, y):
# # #     try:
# # #         from scipy.stats import levene
# # #         stat, p = levene(x, y, center="median")
# # #         print(f"\nLevene test (equal variances, Brown–Forsythe): stat={stat:.6f}, p={p:.6g}")
# # #     except Exception as e:
# # #         print(f"\n[Info] Skipping Levene test (scipy not available): {e}")

# # # def main():
# # #     scores_all = load_scores(PATH_ALL)
# # #     scores_sel = load_scores(PATH_SELECTED)

# # #     # Full stats
# # #     s_all = basic_stats(scores_all, "All branches (perturb_all_lines)")
# # #     s_sel = basic_stats(scores_sel, "Selected branches (min_full_rank)")

# # #     print_stats(s_all)
# # #     print_stats(s_sel)

# # #     # Variance ratio
# # #     if s_sel["var_sample"] > 0:
# # #         print(f"\nVariance ratio (sample): {s_all['var_sample'] / s_sel['var_sample']:.4f}")
# # #     else:
# # #         print("\nVariance ratio (sample): undefined")

# # #     # Aligned comparison (equal length)
# # #     n = min(scores_all.size, scores_sel.size)
# # #     if scores_all.size != scores_sel.size:
# # #         print(f"\n[Note] Length mismatch (all={scores_all.size}, sel={scores_sel.size}), comparing first {n}.")
# # #     try_levene(scores_all[:n], scores_sel[:n])

# # # if __name__ == "__main__":
# # #     main()












# # # var_steady.py
# # import os
# # import argparse
# # import numpy as np
# # import pandas as pd

# # ROOT = "."  # run from inside Robust_MTD-main_BigBus_2

# # PATH_ALL = os.path.join(
# #     ROOT, "td3_reward_variance_comparison",
# #     "perturb_all_lines_ratio_0.3",
# #     "with_ep_schdlr_ptienc_2_0.001_0.001_dropout_03_first_2_layers_old_noise",
# #     "score_plot",
# # )
# # PATH_SEL = os.path.join(
# #     ROOT, "td3_reward_variance_comparison",
# #     "minimum_full_rank_ratio_0.3",
# #     "with_ep_schdlr_ptienc_2_0.001_0.001_dropout_03_first_2_layers_old_noise",
# #     "score_plot",
# # )

# # def load_scores(path: str) -> np.ndarray:
# #     csv = os.path.join(path, "score_history.csv")
# #     npy = os.path.join(path, "score_history.npy")

# #     if os.path.isfile(csv):
# #         df = pd.read_csv(csv, header=None)
# #         # Keep only numeric columns, flatten to 1-D
# #         arr = df.select_dtypes(include=[np.number]).to_numpy(dtype=float).ravel()
# #         if arr.size == 0:
# #             raise ValueError(f"No numeric data in {csv}")
# #         return arr

# #     if os.path.isfile(npy):
# #         return np.load(npy).astype(float).ravel()

# #     raise FileNotFoundError(f"Missing score_history.csv / .npy in {path}")

# # def stats_1d(x: np.ndarray):
# #     x = np.asarray(x, dtype=float).ravel()
# #     mean = np.mean(x)
# #     std = np.std(x, ddof=1) if x.size > 1 else np.nan
# #     var = np.var(x, ddof=1) if x.size > 1 else np.nan
# #     cv = (std / mean) if mean != 0 else np.nan
# #     return {
# #         "n": x.size, "mean": mean, "std": std, "var": var, "cv": cv,
# #         "min": np.min(x), "max": np.max(x)
# #     }

# # def print_stats(title: str, s: dict):
# #     print(f"\n== {title} ==")
# #     print(f"n={s['n']}, mean={s['mean']:.6f}, var={s['var']:.6e}, std={s['std']:.6f}, cv={s['cv']:.4%}")
# #     print(f"min={s['min']:.6f}, max={s['max']:.6f}")

# # def levene_equal_var(x: np.ndarray, y: np.ndarray, where: str):
# #     try:
# #         from scipy.stats import levene
# #         # Ensure 1-D
# #         x = np.asarray(x, dtype=float).ravel()
# #         y = np.asarray(y, dtype=float).ravel()
# #         stat, p = levene(x, y, center="median")
# #         print(f"Levene (Brown–Forsythe) {where}: stat={stat:.6f}, p={p:.6g}")
# #     except Exception as e:
# #         print(f"[Info] Skipping Levene {where}: {e}")

# # def main():
# #     ap = argparse.ArgumentParser()
# #     ap.add_argument("--last", type=int, default=200, help="episodes for steady-state slice (default: 200)")
# #     args = ap.parse_args()

# #     all_scores = load_scores(PATH_ALL)
# #     sel_scores = load_scores(PATH_SEL)

# #     # Full stats
# #     s_all_full = stats_1d(all_scores)
# #     s_sel_full = stats_1d(sel_scores)
# #     print_stats("All branches (full)", s_all_full)
# #     print_stats("Selected branches (full)", s_sel_full)

# #     # Full Levene + variance ratio
# #     if np.isfinite(s_all_full["var"]) and np.isfinite(s_sel_full["var"]) and s_sel_full["var"] > 0:
# #         print(f"\nVariance ratio (full, sample): {s_all_full['var'] / s_sel_full['var']:.4f}")
# #     else:
# #         print("\nVariance ratio (full): undefined")
# #     levene_equal_var(all_scores, sel_scores, "(full)")

# #     # Steady-state stats: use the same last-N for both (fair comparison)
# #     last_n = min(args.last, all_scores.size, sel_scores.size)
# #     all_last = all_scores[-last_n:].ravel()
# #     sel_last = sel_scores[-last_n:].ravel()

# #     s_all_last = stats_1d(all_last)
# #     s_sel_last = stats_1d(sel_last)
# #     print_stats(f"All branches (last {last_n})", s_all_last)
# #     print_stats(f"Selected branches (last {last_n})", s_sel_last)

# #     if np.isfinite(s_all_last["var"]) and np.isfinite(s_sel_last["var"]) and s_sel_last["var"] > 0:
# #         print(f"\nVariance ratio (last {last_n}, sample): {s_all_last['var'] / s_sel_last['var']:.4f}")
# #     else:
# #         print(f"\nVariance ratio (last {last_n}): undefined")
# #     levene_equal_var(all_last, sel_last, f"(last {last_n})")

# #     # Compact line for rebuttal paste
# #     print("\n--- Rebuttal summary line ---")
# #     print(
# #         f"Full: var_all={s_all_full['var']:.3e}, var_sel={s_sel_full['var']:.3e}, "
# #         f"ratio={s_all_full['var']/s_sel_full['var']:.2f} | "
# #         f"Last {last_n}: var_all={s_all_last['var']:.3e}, var_sel={s_sel_last['var']:.3e}, "
# #         f"ratio={s_all_last['var']/s_sel_last['var']:.2f}"
# #     )

# # if __name__ == "__main__":
# #     main()





# # plot_score_history.py
# import os
# import argparse
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# ROOT = "."  # run from inside Robust_MTD-main_BigBus_2

# PATH_ALL = os.path.join(
#     ROOT, "td3_reward_variance_comparison",
#     "perturb_all_lines_ratio_0.3",
#     "with_ep_schdlr_ptienc_2_0.001_0.001_dropout_03_first_2_layers_old_noise",
#     "score_plot",
# )
# PATH_SEL = os.path.join(
#     ROOT, "td3_reward_variance_comparison",
#     "minimum_full_rank_ratio_0.3",
#     "with_ep_schdlr_ptienc_2_0.001_0.001_dropout_03_first_2_layers_old_noise",
#     "score_plot",
# )

# def load_scores(path: str) -> np.ndarray:
#     csv = os.path.join(path, "score_history.csv")
#     npy = os.path.join(path, "score_history.npy")
#     if os.path.isfile(csv):
#         # read numeric cols only, flatten to 1D
#         arr = pd.read_csv(csv, header=None).select_dtypes(include=[np.number]).to_numpy(dtype=float).ravel()
#         if arr.size == 0:
#             raise ValueError(f"No numeric data in {csv}")
#         return arr
#     if os.path.isfile(npy):
#         return np.load(npy).astype(float).ravel()
#     raise FileNotFoundError(f"Missing score_history.csv / .npy in {path}")

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--last", type=int, default=200, help="episodes for steady-state plot (default: 200)")
#     ap.add_argument("--outdir", type=str, default=".", help="where to save PNGs")
#     args = ap.parse_args()

#     # Load
#     all_scores = load_scores(PATH_ALL)
#     sel_scores = load_scores(PATH_SEL)

#     # Align to same length for fair plotting
#     n = min(all_scores.size, sel_scores.size)
#     all_al = all_scores[:n]
#     sel_al = sel_scores[:n]
#     x = np.arange(1, n + 1)

#     # Figure 1: full aligned series
#     plt.figure()
#     plt.plot(x, all_al, label="All branches")
#     plt.plot(x, sel_al, label="Selected branches")
#     plt.xlabel("Episode")
#     plt.ylabel("Score")
#     plt.title("Score History (Aligned)")
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     plt.tight_layout()
#     f1 = os.path.join(args.outdir, "score_history_aligned.png")
#     plt.savefig(f1, dpi=220)
#     plt.close()

#     # Figure 2: steady-state (last N)
#     last_n = min(args.last, n)
#     x_last = np.arange(n - last_n + 1, n + 1)
#     all_last = all_al[-last_n:]
#     sel_last = sel_al[-last_n:]

#     plt.figure()
#     plt.plot(x_last, all_last, label=f"All branches (last {last_n})")
#     plt.plot(x_last, sel_last, label=f"Selected branches (last {last_n})")
#     plt.xlabel("Episode")
#     plt.ylabel("Score")
#     plt.title(f"Score History — Steady State (Last {last_n})")
#     plt.grid(True, alpha=0.3)
#     plt.legend()
#     plt.tight_layout()
#     f2 = os.path.join(args.outdir, f"score_history_last_{last_n}.png")
#     plt.savefig(f2, dpi=220)
#     plt.close()

#     print(f"Saved:\n  {f1}\n  {f2}")

# if __name__ == "__main__":
#     main()
    






# plot_score_history.py
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = "."  # run from inside Robust_MTD-main_BigBus_2

# check_npy_size.py
import numpy as np

# file_path = "/home/nima/Downloads/Robust_MTD-main_BigBus_2/final_simulation_data/case57_ac_opf_50/ac/TD3_ratio_0.3/with_ep_schdlr_ptienc_2_0.001_0.001_dropout_03_first_2_layers/choice_0_ratio_1_column_0_TIME_ROBUST.npy"
file_path = "/home/nima/Downloads/Robust_MTD-main_BigBus_3/final_simulation_data/case57/ac/DDPG_ratio_0.3/with_ep_schdlr_ptienc_2_0.001_0.001_dropout_03_first_2_layers/choice_0_ratio_1_column_0_TIME_ROBUST.npy"

try:
    arr = np.load(file_path, allow_pickle=True)
    result = {
        "shape": arr.shape,
        "size": arr.size,
        "dtype": str(arr.dtype)
    }
except Exception as e:
    result = {"error": str(e)}

print(result)
