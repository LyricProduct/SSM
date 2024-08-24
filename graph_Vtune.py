import pandas as pd
import matplotlib.pyplot as plt


def main():
    ############################################################################
    # Parameters
    csv_file_path = "./Result/celebA/Distance-based/result_SSC_Vtune.csv"
    dataset = "CelebA"
    vocab_method = "Retrieval-based"

    ############################################################################
    df = pd.read_csv(csv_file_path)

    # 各Vの値において、最もWGの値が大きい行を取得
    upper_bound = df.loc[df.groupby("V")["WG"].idxmax()]

    # 各Vの値において、最もDistanceが小さい行を取得
    ours = df.loc[df.groupby("V")["Distance"].idxmin()]

    # Upper boundのグラフ (破線)
    plt.plot(
        upper_bound["V"],
        upper_bound["WG"],
        "r--",
        color="blue",
        label="Upper bound",
        # marker="o",
    )

    # Oursのグラフ (実線)
    plt.plot(
        ours["V"],
        ours["WG"],
        "b-",
        color="blue",
        label="Ours",
        marker="o",
    )

    # グラフのタイトルとラベル
    plt.title(f"Worst Group accuracy for {dataset}, {vocab_method}")
    plt.xlabel(r"$V$")
    plt.ylabel("Worst Group Accuracy(%)")
    plt.ylim(0, 100)

    # 凡例の表示
    plt.legend()

    # 後処理
    # plt.legend()
    plt.savefig(f"./WG_{dataset}_{vocab_method}_V_tuning.jpeg")
    plt.figure()

    # CSVファイルにupper_boundのデータを出力
    output_csv_path = "upper_bound_output_Vtune.csv"
    upper_bound.to_csv(output_csv_path, index=False)


if __name__ == "__main__":
    main()  # main関数の呼び出し
