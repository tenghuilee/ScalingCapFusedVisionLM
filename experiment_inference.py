"""
This is a demo for evaluating the models with different lengths of vision tokens.
"""

import io
import json

import torch

from tensorfusionvlm.vlmeval_extend import supported_VLM

model_name_list = []

for flen in [1, 8, 16, 32, 64, 128, 256, 384, 512, 768]:
    model_name = f"ImageMultiQueryCLIPLLMModel_llama-2_{flen}"
    model_name_list.append(model_name)

model_name_list.extend([
    "LLaVA-v1.5-7B-hf",
    "InstructBLIP-7B-hf",
])

tag = "i01"
image_path = "./__hidden/i01.png"
ans_dict = dict()

for model_name in model_name_list:
    model = supported_VLM[model_name]()

    # ans = model.generate([
    #     "./__hidden/i00.jpeg",
    #     "Please describe the image in full detail."
    # ])
    ans = model.generate([
        image_path,
        "Please read all text from the image."
    ])

    ans_dict[model_name] = ans

    print("\033[32m", ans, "\033[0m")
    del model
    torch.cuda.empty_cache()


with open(f"./vlmeval_results_ft/generate_caption_{tag}.json", "w") as f:
    json.dump(ans_dict, f, indent=4)


# write to latex table format
out_str = io.StringIO()

out_str.write(r"""
\begin{table}[ht]
\centering
\begin{tabular}{c|p{0.75\textwidth}}
\hline
Model & Caption \\
\hline
""")

for model_name, ans in ans_dict.items():
    # finter out latex special characters
    if model_name.startswith("ImageMultiQueryCLIPLLMModel"):
        model_name = model_name.split("_")[2]

        out_str.write(f"\(N_l = {model_name}\) & \\begin{{lstlisting}}\n{ans}\n\end{{lstlisting}}")
    else:
        out_str.write(f"{model_name} & \\begin{{lstlisting}}\n{ans}\n\end{{lstlisting}}")
    out_str.write(r"\\ \hline" + "\n")

out_str.write("\hline\n")
out_str.write("\end{tabular}\n")
out_str.write("\end{table}\n")

with open(f"./vlmeval_results_ft/generate_caption_{tag}.tex", "w") as f:
    f.write(out_str.getvalue())


out_str = io.StringIO()

for model_name, ans in ans_dict.items():
    if model_name.startswith("ImageMultiQueryCLIPLLMModel"):
        model_name = model_name.split("_")[2]

        out_str.write(f"\(N_l = {model_name}\):\n")
        out_str.write(r"\begin{lstlisting}" + "\n")
        out_str.write(ans + "\n")
        out_str.write(r"\end{lstlisting}" + "\n")
    else:
        out_str.write(f"{model_name}:\n")
        out_str.write(r"\begin{lstlisting}" + "\n")
        out_str.write(ans + "\n")
        out_str.write(r"\end{lstlisting}" + "\n")

with open(f"./vlmeval_results_ft/generate_caption_{tag}.tex", "w") as f:
    f.write(out_str.getvalue())
    

