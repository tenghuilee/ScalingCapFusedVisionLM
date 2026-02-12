
import io
import os
from typing import Optional, Tuple

import matplotlib
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt


class FigUtilBase:

    seq__marks: str = "^ov^<>spPhHxXDd+*12348"
    
    def __init__(
        self,
        fig_dir: str = None,
        show_fig: bool = False,
        fontsize: int = 24,
        figsize: Tuple[int, int] = (16, 9),
        **kwargs,
    ):
        self.fig_dir = fig_dir
        os.makedirs(fig_dir, exist_ok=True)

        self.show_fig = show_fig
        self.fontsize = fontsize
        self.figsize = figsize

        self.kwargs = kwargs

        self.seq1_marks = [c+"-" for c in self.seq__marks]
        self.seq2_marks = [c+":" for c in self.seq__marks]
        self.seq3_marks = [c+"--" for c in self.seq__marks]

        self.seq1_color_cmp = matplotlib.colors.LinearSegmentedColormap.from_list(
            "s1_linear", ["#042940", "#D6D58E"])
        self.seq2_color_cmp = matplotlib.colors.LinearSegmentedColormap.from_list(
            "s2_linear", ["#C77401", "#DB8F8A"])
        self.seq3_color_cmp = matplotlib.colors.LinearSegmentedColormap.from_list(
            "s3_linear", ["#49DB00", "#937EB7"])

        self.color_skyblue = "#74bff9"
        self.color_deepskyblue = "#023059"

        self.color_map_sky = matplotlib.colors.LinearSegmentedColormap.from_list(
            "sky", [self.color_skyblue, self.color_deepskyblue]
        )
    
    @staticmethod
    def seq_color(N: int, begin: str, end: str):
        cmp = matplotlib.colors.LinearSegmentedColormap.from_list(
            "s_linear", [begin, end])
        return cmp(np.linspace(0, 1, N))
    
    def seq1_colors(self, N: int):
        return self.seq1_color_cmp(np.linspace(0, 1, N))

    def seq2_colors(self, N: int):
        return self.seq2_color_cmp(np.linspace(0, 1, N))

    def seq3_colors(self, N: int):
        return self.seq3_color_cmp(np.linspace(0, 1, N))

    def save_fig(self, fig: plt.Figure, fname: str, bbox_inches='tight', padd_inches=0.1, **kwargs):
        fpath = os.path.join(self.fig_dir, fname)
        fig.savefig(fpath, bbox_inches=bbox_inches,
                    pad_inches=padd_inches, **kwargs)

    def fig_util(
        self,
        fname: Optional[str] = None,
        image_dir: Optional[str] = None,
        fontsize: Optional[int] = None,
        figsize: Optional[Tuple[int, int]] = None,
        with_legned: bool = False,
        show_fig: bool = None,
        dump_to_disk: bool = True,
        subplots_config: Optional[dict] = None,
        **kwargs,
    ):
        class _Wrapper:
            def __init__(
                self,
                fname: str,
                fontsize: int,
                figsize: Tuple[int, int],
                with_legned: bool,
                show_fig: bool,
                dump_to_disk: bool,
                subplots_config: Optional[dict],
                **kwargs,
            ):
                self.fname = fname
                self.fontsize = fontsize
                self.figsize = figsize
                self.with_legned = with_legned
                self.show_fig = show_fig
                self.kwargs = kwargs
                self.dump_to_disk = dump_to_disk
                self.subplots_config = subplots_config
                self.fig = None
                self.axs = None

                if subplots_config is not None:
                    if not "figsize" in subplots_config:
                        subplots_config["figsize"] = figsize

            def __str__(self):
                ans = io.StringIO("FigUtilWrapper(\n")
                for k, v in self.__dict__.items():
                    ans.write(f"    {k} = {v}\n")
                ans.write(")\n")
                return ans.getvalue()

            def __enter__(self):
                # do somthing
                if self.subplots_config is not None:
                    self.fig, self.axs = plt.subplots(**self.subplots_config)
                else:
                    self.fig = plt.figure(figsize=figsize)
                    self.axs = None
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                # do something
                if exc_type is not None:
                    print(f"Error occurred while saving figure: {exc_value}")
                    print(traceback)
                    return False  # return False to propagate the exception

                if self.with_legned:
                    plt.legend(fontsize=self.fontsize)
                plt.xticks(fontsize=self.fontsize)
                plt.yticks(fontsize=self.fontsize)
                if (self.fname is not None) and self.dump_to_disk:
                    plt.savefig(self.fname, bbox_inches='tight',
                                pad_inches=0.01)

                if self.show_fig:
                    plt.show()
                plt.close(self.fig)
                return True  # return True to indicate that the exception has been handled

            def __call__(self, func):
                def __inner(*args, **kwargs):
                    with self:
                        func(*args, **kwargs)
                return __inner

        image_dir = image_dir if image_dir is not None else self.fig_dir
        show_fig = show_fig if show_fig is not None else self.show_fig
        fontsize = fontsize if fontsize is not None else self.fontsize
        figsize = figsize if figsize is not None else self.figsize

        if fname is not None:
            fname = os.path.join(image_dir, fname)

        return _Wrapper(
            fname, fontsize, figsize, with_legned, show_fig,
            dump_to_disk=dump_to_disk, subplots_config=subplots_config,
            **kwargs,
        )
