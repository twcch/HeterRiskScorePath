import matplotlib.pyplot as plt


def plot_summary_figure(
    summary_obj, filename="tobit_model_summary.png", figsize=(8, 6)
):
    summary_text = str(summary_obj)

    # 建立圖表
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    # 將 summary 文字放入圖表
    ax.text(
        0.01,
        0.99,
        summary_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        fontfamily="Monospace",
        wrap=True,
    )

    plt.tight_layout()

    # 儲存圖檔
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"img save: {filename}")
