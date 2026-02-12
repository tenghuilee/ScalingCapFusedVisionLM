#%%

import os
from tkinter import font

import matplotlib
import matplotlib.colors
from torch import _functional_sym_constrain_range_for_size

os.environ["LD_LIBRARY_PATH"] = f"/usr/lib/x86_64-linux-gnu:{os.environ['LD_LIBRARY_PATH']}"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

#%%
from expectation_bound_visualize.base import *
import expectation_bound_visualize.zoo_config as zoo_config
from expectation_bound_visualize.psi import *

#%%
zoo = ModelZoo(
    dpo_dataset_path="./datasets/dpo/orca_rlhf.jsonl",
    batch_size=2,
    device="cuda:0",
    fig_dir="./cached_expectation_bound_files/imgs-gd",
    show_fig=True,
).fast_register_from_dict(
    llm_name_dict={
        "vicuna_v1.5-7b": "lmsys/vicuna-7b-v1.5",
        "microsoft-phi-3-mini": "microsoft/Phi-3-mini-4k-instruct",
        "llama-2-7b": "./llama_hf_mirror/llama_hf/llama-2-7b-chat", 
        "llama-3-8b-instruct": "./llama_hf_mirror/llama_hf/llama-3-8b-instruct",
        "qwen2-7b-instruct": "Qwen/Qwen2-7B-Instruct",
        "qwen2-0.5b-instruct": "Qwen/Qwen2-0.5B-Instruct",
        "qwen2-1.5b-instruct": "Qwen/Qwen2-1.5B-Instruct",
        # "qwen2.5-1.5b-instruct": "Qwen/Qwen2.5-1.5B-Instruct",
        # "qwen2.5-3b-instruct": "Qwen/Qwen2.5-3B-Instruct",
        # "qwen2.5-7b-instruct": "Qwen/Qwen2.5-7B-Instruct",
        "qwen2.5-14b-instruct": "Qwen/Qwen2.5-14B-Instruct",
        # "qwen2.5-32b-instruct": "Qwen/Qwen2.5-32B-Instruct",
    },
    cached_dir="./cached_expectation_bound_files"
)


#%%

from expectation_bound_visualize import histE_CosTheta
# histE_CosTheta.collector.run_all(zoo, color="#66b0e9", figsize = (20, 9), fontsize=14)

#%%

color_1 = "#042940"
color_2 = "#C77401"
color_3 = "#49DB00"

color_skyblue = "#74bff9"
color_deepskyblue = "#023059"

color_map_sky = matplotlib.colors.LinearSegmentedColormap.from_list(
    "sky", [color_skyblue, color_deepskyblue]
)

def color_map_sky_discrete(n: int):
    return color_map_sky(np.linspace(0, 1, n))

#%%
with zoo.fig_util(
    fname="compare_psi_aa_bb_aabb.pdf",
    with_legned=False,
    fontsize=14,
    figsize=(20, 9),
) as futil:
    nrows = 2
    ncols = np.ceil(len(zoo) / nrows).astype(int)

    color_bar = color_map_sky_discrete(3)

    for i, (name, factory) in enumerate(zoo.items()):
        plt.subplot(nrows, ncols, i + 1)

        psi_N = factory.load_reduce_cache(
            EvalItemThetaCrossAA(), max_seq_len=801).value()
        xx = np.arange(1, psi_N.shape[0])
        plt.plot(xx, psi_N[1:], label=r"$\psi_{\text{cross}}^{(++)}(n)$", color=color_bar[0])

        psi_N = factory.load_reduce_cache(
            EvalItemThetaCrossAABB(), max_seq_len=801).value()
        xx = np.arange(1, psi_N.shape[0])
        plt.plot(
            xx,
            psi_N[1:],
            label=r"$0.5 (\psi_{\text{cross}}^{(++)}(n) + \psi_{\text{cross}}^{(--)}(n))$",
            color=color_bar[1],
        )

        psi_N = factory.load_reduce_cache(
            EvalItemThetaCrossBB(), max_seq_len=801).value()
        xx = np.arange(1, psi_N.shape[0])
        plt.plot(xx, psi_N[1:], label=r"$\psi_{\text{cross}}^{(--)}(n)$", color=color_bar[2])


        plt.xticks(fontsize=futil.fontsize)
        plt.yticks(fontsize=futil.fontsize)
        plt.legend(fontsize=futil.fontsize, loc="lower right", framealpha=0.5)
        plt.xlabel(name, fontsize=futil.fontsize+5)


#%%
with zoo.fig_util(
    fname="compare_psi_aabb_gt_ab.pdf",
    with_legned=False,
    fontsize=14,
    figsize=(20,9),
) as futil:

    nrows = 2
    ncols = np.ceil(len(zoo) / nrows).astype(int)

    for i, (name, factory) in enumerate(zoo.items()):
        plt.subplot(nrows, ncols, i + 1)

        psi_N = factory.load_reduce_cache(
            EvalItemThetaCrossAABB(), max_seq_len=801).value()
        xx = np.arange(1, psi_N.shape[0])
        plt.plot(
            xx,
            psi_N[1:],
            label=r"$\psi_{\text{cross}}^{(AA)}(n)$",
            color=color_skyblue,
        )

        psi_N = factory.load_reduce_cache(
            EvalItemThetaCrossAB(), max_seq_len=801).value()
        xx = np.arange(1, psi_N.shape[0])
        plt.plot(xx, psi_N[1:], label=r"$\psi_{\text{cross}}^{(AB)}(n)$", color=color_deepskyblue)

        plt.xticks(fontsize=futil.fontsize)
        plt.yticks(fontsize=futil.fontsize)
        plt.legend(fontsize=futil.fontsize, loc="lower right", framealpha=0.5)
        plt.xlabel(name, fontsize=futil.fontsize+5)

# %%

with zoo.fig_util(
    fname="compare_psi_cross_ab_aabb_equal_ab.pdf",
    with_legned=False,
    fontsize=14,
    figsize=(20,9),
) as futil:

    nrows = 2
    ncols = np.ceil(len(zoo) / nrows).astype(int)

    colorskys = color_map_sky_discrete(3)

    for i, (name, factory) in enumerate(zoo.items()):
        plt.subplot(nrows, ncols, i + 1)

        psi_N = factory.load_reduce_cache(
            EvalItemThetaCrossAABB(), max_seq_len=801).value()
        xx = np.arange(1, psi_N.shape[0])
        plt.plot(
            xx,
            psi_N[1:],
            label=r"$\psi_{\text{cross}}^{(AA)}(n)$",
            color=colorskys[0],
        )

        psi_N = factory.load_reduce_cache(
            EvalItemThetaEqualAB(), max_seq_len=801).value()
        xx = np.arange(1, psi_N.shape[0])
        plt.plot(xx, psi_N[1:], label=r"$\psi_{\text{equal}}^{(AB)}(n)$", color=colorskys[1])

        psi_N = factory.load_reduce_cache(
            EvalItemThetaCrossAB(), max_seq_len=801).value()
        xx = np.arange(1, psi_N.shape[0])
        plt.plot(xx, psi_N[1:], label=r"$\psi_{\text{cross}}^{(AB)}(n)$", color=colorskys[2])

        plt.xticks(fontsize=futil.fontsize)
        plt.yticks(fontsize=futil.fontsize)
        plt.legend(fontsize=futil.fontsize, loc="lower right", framealpha=0.5)
        plt.xlabel(name, fontsize=futil.fontsize+5)

#%%

def curve_func(x, scale, peab, pcaa, pcab):
    return scale * np.sqrt(x * (1 - peab) + x*(x-1)*(pcaa - pcab))

with zoo.fig_util(
    fname="various_E_mean_std_psi_N.pdf",
    with_legned=False,
    fontsize=11,
    figsize=(19,8),
) as futil:

    nrows = 2
    ncols = np.ceil(len(zoo) / nrows).astype(int)
    for i, (name, factory) in enumerate(zoo.items()):
        plt.subplot(nrows, ncols, i + 1)

        _cache = factory.load_reduce_cache(
            EvalItemDiffAccumulateTokens(),
            max_seq_len=801
        ) # type: ReduceMeanStd

        _mean, _std = _cache.mean_std()

        psi_equal_ab = factory.load_reduce_cache(
            EvalItemThetaEqualAB(), max_seq_len=801).value()
        psi_cross_aabb = factory.load_reduce_cache(
            EvalItemThetaCrossAABB(), max_seq_len=801).value()
        psi_cross_ab = factory.load_reduce_cache(
            EvalItemThetaCrossAB(), max_seq_len=801).value()

        xx = np.arange(1, psi_equal_ab.shape[0]+1)
        
        def object_func(scale, x, y):
            return np.mean((curve_func(
                x, scale, psi_equal_ab, psi_cross_aabb, psi_cross_ab) - y)**2)
        
        opt_res = optimize.minimize(object_func, 1, args=(xx, _mean))

        yy = curve_func(xx, opt_res.x, psi_equal_ab, psi_cross_aabb, psi_cross_ab)

        ele1 = plt.plot(xx, _mean, "-", linewidth=4, label=r"$\mathbb{E}_{\mathcal{V}}\left[\mathcal{D}(n)\right]$")
        ele2 = plt.fill_between(xx, _mean - _std, _mean + _std,
                     alpha=0.08, color="blue", label=r"$\mathbb{E}_{\mathcal{V}}\left[\mathcal{D}(n)\right] \pm \sigma$")
        ele3 = plt.fill_between(xx, _mean - 2*_std, _mean + 2*_std,
                     alpha=0.08, color="blue", label=r"$\mathbb{E}_{\mathcal{V}}\left[\mathcal{D}(n)\right] \pm 2 \sigma$")
        ele4 = plt.plot(xx, yy, '-', linewidth=2,
                 label=r"$\lambda \sqrt{\Upsilon(n)}, \lambda=%.2f$" % (opt_res.x[0]))

        # manual handlers for covering in figures; color not correct in legend
        handlers = [
            *ele1,
            plt.Rectangle(
                (0, 0), 1, 1,
                color=ele2.get_facecolor(),
                alpha=ele2.get_alpha()+ele3.get_alpha(), # manual correct the alpha
                label=ele2.get_label(),
            ),
            ele3,
            *ele4,
        ]

        plt.xticks(fontsize=futil.fontsize)
        plt.yticks(fontsize=futil.fontsize)
        plt.legend(
            handles=handlers,
            loc="upper left",
            fontsize=futil.fontsize,
            framealpha=0.3,
            fancybox=True,
        )
        plt.xlabel(name, fontsize=futil.fontsize+5)

#%%
# compute function of alpha(n)

with zoo.fig_util(
    fname=None,
    with_legned=True,
) as futil:

    max_seq_len = 100
    for i, (name, factory) in enumerate(zoo.items()):

        psi_equal_ab = factory.load_reduce_cache(
            EvalItemThetaEqualAB(), max_seq_len=max_seq_len).value()
        psi_cross_aabb = factory.load_reduce_cache(
            EvalItemThetaCrossAABB(), max_seq_len=max_seq_len).value()
        psi_cross_ab = factory.load_reduce_cache(
            EvalItemThetaCrossAB(), max_seq_len=max_seq_len).value()

        xx = np.arange(1, psi_equal_ab.shape[0]+1)

        c = 1 - psi_equal_ab[0]


#%%

with zoo.fig_util(
    fname=None,
    with_legned=True,
) as futil:

    max_seq_len = 100
    for i, (name, factory) in enumerate(zoo.items()):
        # plt.subplot(1, len(zoo), i + 1)

        psi_equal_ab = factory.load_reduce_cache(
            EvalItemThetaEqualAB(), max_seq_len=max_seq_len).value()
        psi_cross_aabb = factory.load_reduce_cache(
            EvalItemThetaCrossAABB(), max_seq_len=max_seq_len).value()
        psi_cross_ab = factory.load_reduce_cache(
            EvalItemThetaCrossAB(), max_seq_len=max_seq_len).value()

        xx = np.arange(1, psi_equal_ab.shape[0]+1)

        yy = (1 - psi_equal_ab) / (psi_cross_aabb - psi_cross_ab) - 1

        plt.plot(
            xx,
            yy,
            label=name,
            # marker=futil.m
            markersize=10,
        )
    
    plt.plot(xx, xx, color="black", linestyle="--", alpha=0.8)

# %%
