import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import os

if __name__=='__main__':
    alphas_dir = sys.argv[1]
    # Find all matching files
    files = glob.glob(f"{alphas_dir}/alphas_instance_*.csv")
    metadata_file = sys.argv[2]   # second CSV file
    index_values_file = sys.argv[3]  # third file

    # Load metadata
    meta = pd.read_csv(metadata_file)

    # Split instance IDs by label
    ids_label1 = meta[meta["label"] == 1]["instance_id"].tolist()
    ids_label0 = meta[meta["label"] == 0]["instance_id"].tolist()

    # Load third file: index → integer
    index_df = pd.read_csv(index_values_file, header=None)
    idx3 = index_df.iloc[:, 0]
    val3 = index_df.iloc[:, 1]
    dilation_dict = dict(zip(idx3.values, val3.values))

    def accumulate(ids):
        cumulative = defaultdict(float)

        for instance_id in ids:
            fname = os.path.join(alphas_dir, f"alphas_instance_{instance_id}.csv")
            if not os.path.exists(fname):
                continue

            df = pd.read_csv(fname, header=None, names=["index", "alpha"])

            for idx, val in zip(df["index"], df["alpha"]):
                cumulative[idx] += val

        pairs = sorted(cumulative.items())
        indices = [i for i, _ in pairs]
        values = [v for _, v in pairs]

        return indices, values

    # Compute cumulative attributions
    idx1, val1 = accumulate(ids_label1)
    idx0, val0 = accumulate(ids_label0)

    # Plot
    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=False)

    pairs1 = sorted(zip(idx1, val1), key=lambda x: x[1], reverse=True)[:20]
    idx1_sorted = [p[0] for p in pairs1]
    val1_sorted = [p[1] for p in pairs1]
    x_rank1 = range(len(val1_sorted))

    pairs0 = sorted(zip(idx0, val0), key=lambda x: x[1], reverse=True)[:20]
    idx0_sorted = [p[0] for p in pairs0]
    val0_sorted = [p[1] for p in pairs0]
    x_rank0 = range(len(val0_sorted))

    # Plot for label = 1
    axes[0].bar(x_rank1, np.array(val1_sorted))
    axes[0].set_title("Cumulative attribution (label = 1)")
    axes[0].set_ylabel("Cumulative attribution")
    axes[0].set_xticks(x_rank1, idx1_sorted, rotation=90)


    # Plot for label = 0
    axes[1].bar(x_rank0, np.array(val0_sorted))
    axes[1].set_title("Cumulative attribution (label = 0)")
    axes[1].set_xlabel("Observation index")
    axes[1].set_ylabel("Cumulative attribution")
    axes[1].set_xticks(x_rank0, idx0_sorted, rotation=90)

    axes[3].bar(x_rank0, [dilation_dict[np.int64(x)] for x in idx0_sorted])
    axes[3].set_yscale('log', base=2)
    axes[3].set_title("Dilations (label = 0)")
    axes[3].set_xlabel("Observation index")
    axes[3].set_ylabel("Dilation")
    axes[3].set_xticks(x_rank0, idx0_sorted, rotation=90)


    axes[2].bar(x_rank1, [dilation_dict[np.int64(x)] for x in idx1_sorted])
    axes[2].set_yscale('log', base=2)
    axes[2].set_title("Dilations (label = 1)")
    axes[2].set_xlabel("Observation index")
    axes[2].set_ylabel("Dilation")
    axes[2].set_xticks(x_rank1, idx1_sorted, rotation=90)



    plt.tight_layout()
    plt.savefig("histogram_alphas.png")
