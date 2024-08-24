import subprocess
import numpy as np
import matplotlib.pyplot as plt
import argparse
import csv
from tqdm import tqdm
import os


def get_args():
    # Programのコマンドの引数を取る。descriptionにProgramの内容を記述
    # parserについては[https://qiita.com/kzkadc/items/e4fc7bc9c003de1eb6d0]を参考に
    parser = argparse.ArgumentParser(description="Debiasing Vision-Language Models")

    # Dataset
    parser.add_argument("--dataset", type=str, default="")

    # Zero-shot Model
    parser.add_argument("--model", type=str, default="")

    # CUDA
    parser.add_argument("--CUDANum", type=int, default="")

    # How to measure distance
    parser.add_argument("--dist_method", type=str, default="")

    # parameter: the value of coefficient of regularization term
    parser.add_argument("--lam", default=1000, type=float)

    # # parameter: the number of estimated keywords
    # parser.add_argument("-s", type=int, default="")

    # # parameter: the number of cluster
    # parser.add_argument("-cluster_num", type=int, default="")

    parser.add_argument("--seed", default=0, type=int)

    # base method
    parser.add_argument("--base", type=str, default="")

    args = parser.parse_args()  # 引数を解析

    return args


def main():
    args = get_args()  # コマンド実行時の引数をgetする

    subprocess.run(
        f'echo "" > ./nohup.out',
        shell=True,
    )
    subprocess.run(
        f'echo "" > ./output1.log',
        shell=True,
    )
    subprocess.run(
        f'echo "" > ./output2.log',
        shell=True,
    )

    if not (os.path.exists(f"./Result_wiki/{args.dataset}/{args.dist_method}/")):
        os.makedirs(f"./Result_wiki/{args.dataset}/{args.dist_method}")

    with open(f"./Result/{args.dataset}/{args.dist_method}/result_SSC.csv", "a") as f:
        writer = csv.writer(f)
        writer.writerow(["V", "K", "Avg", "WG", "Gap", "Distance", "Similarity"])

    # for K in range(5, 6, 1):
    for K in range(3, 13, 1):
        for V in range(130, 131, 10):
            # for V in range(80, 171, 10):
            for itr in range(1):
                subprocess.run(
                    f"CUDA_VISIBLE_DEVICES={args.CUDANum} python main_exp.py --dataset {args.dataset} --load_base_model {args.model} --dist_method {args.dist_method} --base {args.base} --V {V} --K {K} --debias",
                    shell=True,
                )

    # for K in range(2, 3, 1):
    # for K in range(7, 8, 1):
    # for K in range(2, 21, 1):
    # for K in range(5, 6, 1):
    #     # for K in range(8, 9, 1):
    #     # for K in range(3, 4, 1):
    #     for V in range(130, 131, 10):
    #         # for V in range(60, 61, 10):
    #         # for V in range(200, 201, 10):
    #         # for V in range(20, 201, 10):
    #         # for V in range(40, 41, 10):
    #         for itr in range(1):
    #             if args.base == "b2t":
    #                 subprocess.run(
    #                     f"CUDA_VISIBLE_DEVICES={args.CUDANum} python main_SSC.py --dataset {args.dataset} --load_base_model {model} -s {V} -cluster {K} --lam {args.lam} --dist_method {args.dist_method} --seed {args.seed} --base {args.base} --nas --debias --prob_skip",
    #                     shell=True,
    #                 )
    #             elif args.base == "vocab":
    #                 subprocess.run(
    #                     f"CUDA_VISIBLE_DEVICES={args.CUDANum} python main_SSC.py --dataset {args.dataset} --load_base_model {model} -s {V} -cluster {K} --dist_method {args.dist_method} --base {args.base} --nas --debias --prob_skip --csv vocab_wiki_frequency.csv",
    #                     shell=True,
    #                 )


if __name__ == "__main__":
    main()  # main関数の呼び出し
