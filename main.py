import os
import numpy as np
import argparse
import torch
import torch.nn.functional as F
from clip import clip
from datasets import initialize_data
from network import load_base_model
from network.clip import evaluate_clip
from network.clip import make_S
from utils import initialize_experiment
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import csv
from tqdm import tqdm
from ssc.DataProjection import *
from ssc.BuildAdjacency import *
from ssc.OutlierDetection import *
from ssc.SparseCoefRecovery import *
from scipy import linalg
from PIL import Image
from sklearn.cluster import SpectralClustering as sklearnSSC
import warnings


def get_args():
    parser = argparse.ArgumentParser(description="Debiasing Vision-Language Models")

    # Method
    parser.add_argument("--debias", default=False, action="store_true")
    parser.add_argument("--eta", default=1000, type=float)

    # Dataset
    parser.add_argument("--dataset", type=str, default="waterbirds")
    parser.add_argument("--num_workers", default=2, type=int)

    # Zero-shot Model
    parser.add_argument("--load_base_model", type=str, default="")

    # Hyperparams
    parser.add_argument("--bs_trn", default=128, type=int)
    parser.add_argument("--bs_val", default=128, type=int)

    # Misc.
    parser.add_argument("--no_cuda", default=False, action="store_true")
    parser.add_argument("--verbose", default=False, action="store_true")

    # Experiment's Parameter
    parser.add_argument("--V", default=5, type=int)
    parser.add_argument("--K", default=10, type=int)
    parser.add_argument("--base", type=str, default="")
    parser.add_argument("--dist_method", type=str, default="")
    parser.add_argument("--seed", default=0, type=int)

    # For loading Hugging Face models
    parser.add_argument("--cache_dir", default="./models/pretrained_models", type=str)
    args = parser.parse_args()  # 引数を解析

    args.arch = args.load_base_model.split("_")[-1].replace("/", "_").replace("-", "_")
    args.directory_name = "debias_vl"

    if torch.cuda.is_available() and args.no_cuda is False:
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    return args


# Helper functions for debiasing
def get_proj_matrix(embeddings, seed_value):
    tSVD = TruncatedSVD(n_components=len(embeddings), random_state=seed_value)
    embeddings_ = tSVD.fit_transform(embeddings)
    basis = tSVD.components_.T

    proj = np.linalg.inv(np.matmul(basis.T, basis))
    proj = np.matmul(basis, proj)
    proj = np.matmul(proj, basis.T)
    proj = np.eye(proj.shape[0]) - proj
    return proj


def get_A(z_i, z_j):
    z_i = z_i[:, None]
    z_j = z_j[:, None]
    return (
        np.matmul(z_i, z_i.T)
        + np.matmul(z_j, z_j.T)
        - np.matmul(z_i, z_j.T)
        - np.matmul(z_j, z_i.T)
    )


def get_M(embeddings, S):
    d = embeddings.shape[1]
    M = np.zeros((d, d))
    for s in S:
        M += get_A(embeddings[s[0]], embeddings[s[1]])
    return M / len(S)


def schmidt(arr):
    arr = np.array(arr, dtype=np.float64)

    k = arr.shape[1]

    u = arr[:, [0]]

    q = u / linalg.norm(u)

    for j in range(1, k):
        u = arr[:, [j]]
        for i in range(j):
            u -= np.dot(q[:, i], arr[:, j]) * q[:, [i]]
        qi = u / linalg.norm(u)
        q = np.append(q, qi, axis=1)
    return q


def print_info(args):
    print("------------------------------------")
    print(f"Model: {args.load_base_model.split('_')[1]}")
    print(f"Dataset: {args.dataset}")
    print(f"Initial Vocabulary: {args.base}")
    print(f"V: {args.V}")
    print(f"K: {args.K}")
    print(f"η: {args.eta}")
    print("------------------------------------")


def main():
    # warnings.filterwarnings("ignore")
    args = get_args()
    print_info(args)

    base_model_args = args.load_base_model.split("_")

    base_model_components = load_base_model(base_model_args, args, clip=clip)
    (
        base_model,
        base_transform,
        get_embeddings,
        get_dataset_embeddings,
        get_zeroshot_predictions,
    ) = base_model_components

    load_dataloaders, visualize_dataset = initialize_data(args)
    dataloaders_base = load_dataloaders(
        args, train_shuffle=False, transform=base_transform
    )
    train_loader_base, val_loader_base, test_loader_base = dataloaders_base
    splits = ["train", "val", "test"]

    # Initialize other parts of experiment
    initialize_experiment(args)

    # Get pretrained model dataset embeddings
    dataset_embeddings = {}
    for dix, split in enumerate(splits):
        dataset_embeddings[split] = get_dataset_embeddings(
            base_model, dataloaders_base[dix], args, split=split
        ).to(device=args.device)
    # Get embedding dimensions
    print(dataset_embeddings["train"].shape)
    args.base_model_dim = dataset_embeddings["train"].shape[1]
    print(f"-> Embedding dimensions: {args.base_model_dim}")
    print("------------------------------------")
    ######################################################################################
    V = args.V
    subspace_num = args.K
    # vocabulary_name = "wiki"
    ######################################################################################
    # load initial vocabulary
    if args.base == "Captioning-based":
        if args.dataset == "waterbirds":
            df = pd.read_csv(
                f"./vocabulary_file/captioning_based/Waterbird/keyword_{args.V}.csv",
                index_col=0,
            )
        elif args.dataset == "celebA":
            df = pd.read_csv(
                f"./vocabulary_file/captioning_based/CelebA/keyword_{args.V}.csv",
                index_col=0,
            )
        df = df.drop_duplicates(subset="Keyword", keep="first")
        vocab_selected = df["Keyword"].values
        if len(vocab_selected) <= 1:
            return 0
    elif args.base == "Retrieval-based":
        if args.dataset == "waterbirds":
            if args.load_base_model == "clip_ViTL14":
                csv_file_path = "./vocabulary_file/retrieval_based/Waterbird/ViT-L14/vocab_wiki_frequency.csv"
            elif args.load_base_model == "clip_RN50":
                csv_file_path = "./vocabulary_file/retrieval_based/Waterbird/ResNet-50/vocab_wiki_frequency.csv"
        elif args.dataset == "celebA":
            if args.load_base_model == "clip_ViTL14":
                csv_file_path = "./vocabulary_file/retrieval_based/CelebA/ViT-L14/vocab_wiki_frequency.csv"
            elif args.load_base_model == "clip_RN50":
                csv_file_path = "./vocabulary_file/retrieval_based/CelebA/ResNet-50/vocab_wiki_frequency.csv"
        df = pd.read_csv(csv_file_path, index_col=0)
        df = df.head(V)
        vocab_selected = df.index
    ######################################################################################
    # Sparse Subspace Clustering (SSC)
    ssc_prompt = []
    for attribute in vocab_selected:
        ssc_prompt.append(f"This is a picture of a {attribute}.")
    ssc_embeddings = get_embeddings(
        ssc_prompt, base_model, args, normalize=True, verbose=True
    )
    ssc_embeddings = ssc_embeddings.T
    K = 0

    r = 0  # Enter the projection dimension e.g. r = d*n, enter r = 0 to not project
    Cst = 0  # Enter 1 to use the additional affine constraint sum(c) == 1
    OptM = "Lasso"  # OptM can be {'L1Perfect','L1Noise','Lasso','L1ED'}
    lmbda = 1.0  # Regularization parameter in 'Lasso' or the noise level for 'L1Noise'
    Xp = DataProjection(ssc_embeddings, r, "PCA")
    CMat = SparseCoefRecovery(Xp, Cst, OptM, lmbda)
    # Make small values 0
    eps = np.finfo(float).eps
    CMat[np.abs(CMat) < eps] = 0

    CMatC, OutlierIndx, Fail = OutlierDetection(CMat, subspace_num)

    if Fail == False:
        CKSym = BuildAdjacency(CMatC, K)
        spectral_clustering = sklearnSSC(
            n_clusters=subspace_num,
            affinity="nearest_neighbors",
            n_neighbors=10,
            assign_labels="kmeans",
            random_state=args.seed,
        )
        Grps = spectral_clustering.fit_predict(CKSym)
    else:
        print("Something failed")
    ######################################################################################
    # Distance-based Subspace Filtering
    if args.dataset == "waterbirds":
        text_descriptions = [
            "This is a picture of a landbird.",
            "This is a picture of a waterbird.",
        ]
    if args.dataset == "celebA":
        text_descriptions = [
            "A photo of a celebrity with dark hair.",
            "A photo of a celebrity with blond hair.",
        ]
    query_embeddings = get_embeddings(
        text_descriptions, base_model, args, normalize=True, verbose=True
    )

    SS_embeddings_list = []
    SS_word_list = []

    for subspace_id in range(subspace_num):
        vocab_selected_idx = []
        subspace_vocab = []
        spurious_prompt = []
        for idx, grp_name in enumerate(Grps):
            if grp_name == subspace_id:
                vocab_selected_idx.append(idx)
        for idx in vocab_selected_idx:
            subspace_vocab.append(vocab_selected[idx])
        for attribute in subspace_vocab:
            if args.dataset == "waterbirds":
                spurious_prompt.append(f"This is a picture of a {attribute}.")
            elif args.dataset == "celebA":
                spurious_prompt.append(f"A photo of a {attribute}.")
        spurious_embeddings = get_embeddings(
            spurious_prompt, base_model, args, normalize=True, verbose=True
        )
        SS_embeddings_list.append(spurious_embeddings.numpy())
        SS_word_list.append(subspace_vocab)

    if args.dist_method == "Similarity-based":
        cossim_list = []
        for tmp_ss_embedding in SS_embeddings_list:
            cos_sim = cosine_similarity(tmp_ss_embedding, query_embeddings)
            cos_sim = cos_sim.mean(axis=1)
            cos_sim = cos_sim.mean(axis=0)
            cossim_list.append(cos_sim)

        max_value = max(cossim_list)
        max_index = cossim_list.index(max_value)
        subspace_vocab = SS_word_list[max_index]
    elif args.dist_method == "Distance-based":
        subspace_vocab = []
        for class_idx in range(query_embeddings.shape[0]):
            class_dist = []
            for tmp_ss_embedding in SS_embeddings_list:
                A = np.array(tmp_ss_embedding)
                Q = schmidt(A)
                f_v = np.zeros(Q.shape[1])

                for idx in range(Q.shape[0]):
                    f_v += np.dot(query_embeddings[class_idx, :], Q[idx, :]) * Q[idx, :]

                class_dist.append(np.linalg.norm(query_embeddings[class_idx, :] - f_v))

            best_ss_id = class_dist.index(min(class_dist))
            subspace_vocab.append(SS_word_list[best_ss_id])
        combined = set()
        for sublist in subspace_vocab:
            for item in sublist:
                combined.add(item)
        subspace_vocab = list(combined)

    ######################################################################################
    # Calibration
    if args.dataset == "waterbirds":
        text_descriptions = [
            "This is a picture of a landbird.",
            "This is a picture of a waterbird.",
        ]
    if args.dataset == "celebA":
        text_descriptions = [
            "A photo of a celebrity with dark hair.",
            "A photo of a celebrity with blond hair.",
        ]
    query_embeddings = get_embeddings(
        text_descriptions, base_model, args, normalize=True, verbose=True
    )

    S = make_S(subspace_vocab)
    spurious_prompt = []

    for attribute in subspace_vocab:
        spurious_prompt.append(f"This is a picture of a {attribute}.")

    candidate_prompt_1 = []
    candidate_prompt_2 = []

    if args.dataset == "waterbirds":
        for attribute in subspace_vocab:
            candidate_prompt_1.append(
                f"This is a picture of a landbird with {attribute}."
            )
            candidate_prompt_2.append(
                f"This is a picture of a waterbird with {attribute}."
            )

    elif args.dataset == "celebA":
        for attribute in subspace_vocab:
            candidate_prompt_1.append(
                f"A photo of a celebrity with dark hair with {attribute}."
            )
            candidate_prompt_2.append(
                f"A photo of a celebrity with blond hair with {attribute}."
            )

    candidate_prompt_1.extend(candidate_prompt_2)
    candidate_prompt = candidate_prompt_1

    if args.debias:
        spurious_embeddings = get_embeddings(
            spurious_prompt, base_model, args, normalize=True, verbose=True
        )

        spurious_embeddings = spurious_embeddings.numpy()
        P0 = get_proj_matrix(spurious_embeddings, seed_value=args.seed)

        # Calculate Embedding of Positive Pairs
        candidate_embeddings = get_embeddings(
            candidate_prompt, base_model, args, normalize=True, verbose=True
        )
        candidate_embeddings = candidate_embeddings.numpy()

        # Closed Form Optimum
        print("Solve Closed Form Optimum")
        M = get_M(candidate_embeddings, S)
        G = args.eta * M + np.eye(M.shape[0])
        P = np.matmul(P0, np.linalg.inv(G))
        text_embeddings = np.matmul(query_embeddings, P.T)

        text_embeddings = F.normalize(text_embeddings, dim=-1)
        text_embeddings = torch.tensor(text_embeddings).float()

    else:  # Zero-shot CLIP
        text_embeddings = query_embeddings
        text_embeddings = torch.tensor(text_embeddings).float()

    # Evaluate
    dataset_predictions = {}
    for dix, split in enumerate(splits):
        dataset_predictions[split] = get_zeroshot_predictions(
            dataset_embeddings[split],
            text_embeddings.to(device=args.device),
            args,
            temperature=100.0,
        )
    for ix, split in enumerate(splits):
        avg_acc, robust_acc, gap, adj_avg_acc = evaluate_clip(
            dataset_predictions[split],
            dataloaders_base[ix],
            ix,
            args.dataset,
            verbose=True,
        )

    avg_acc = round(avg_acc, 3)
    robust_acc = round(robust_acc, 3)
    gap = round(gap, 3)
    adj_avg_acc = round(adj_avg_acc, 3)

    result_dir_path = f"./Result"
    if not (os.path.exists(result_dir_path)):
        os.makedirs(result_dir_path)
    result_dir_path = os.path.join(result_dir_path, args.dataset)
    if not (os.path.exists(result_dir_path)):
        os.makedirs(result_dir_path)
    result_dir_path = os.path.join(result_dir_path, args.dist_method)
    if not (os.path.exists(result_dir_path)):
        os.makedirs(result_dir_path)
    with open(
        f"./Result/{args.dataset}/{args.dist_method}/result_SSC.csv",
        "a",
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["V", "K", "Avg", "WG", "Gap", "Spurious Subspace"])
        writer.writerow([V, subspace_num, adj_avg_acc, robust_acc, gap, subspace_vocab])
    ######################################################################################


if __name__ == "__main__":
    main()  # main関数の呼び出し
