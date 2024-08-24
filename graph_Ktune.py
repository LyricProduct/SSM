import pandas as pd
import matplotlib.pyplot as plt


def main():
    ############################################################################
    # Parameters
    csv_file_path = "./Result/celebA/Distance-based/result_SSC.csv"
    dataset = "CelebA"
    vocab_method = "Retrieval-based"
    K_list = [3, 4, 5, 6, 8, 10, 12]
    ############################################################################
    df = pd.read_csv(csv_file_path)

    # Kのリストに基づいてフィルタリング
    df_filtered = df[df["K"].isin(K_list)]

    # 各Kの値において、最もWGの値が大きい行を取得
    upper_bound = df_filtered.loc[df_filtered.groupby("K")["WG"].idxmax()]

    # 各Kの値において、最もDistanceが小さい行を取得
    ours = df_filtered.loc[df_filtered.groupby("K")["Distance"].idxmin()]

    # Upper boundのグラフ (破線)
    plt.plot(
        upper_bound["K"],
        upper_bound["WG"],
        "r--",
        color="red",
        label="Upper bound",
        marker="o",
    )

    # Oursのグラフ (実線)
    plt.plot(
        ours["K"],
        ours["WG"],
        "b-",
        color="red",
        label="Ours",
        marker="x",
    )

    # グラフのタイトルとラベル
    plt.title(f"Worst Group accuracy for {dataset}, {vocab_method}")
    plt.xlabel(r"$K$")
    plt.ylabel("Worst Group Accuracy(%)")
    plt.ylim(0, 100)

    # 凡例の表示
    plt.legend()

    # 後処理
    # plt.legend()
    plt.savefig(f"./WG_{dataset}_{vocab_method}_K_tuning.jpeg")
    plt.figure()

    # CSVファイルにupper_boundのデータを出力
    output_csv_path = "upper_bound_output_Ktune.csv"
    upper_bound.to_csv(output_csv_path, index=False)


if __name__ == "__main__":
    main()  # main関数の呼び出し
