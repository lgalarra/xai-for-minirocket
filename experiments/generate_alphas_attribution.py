import glob
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import sys


if __name__=='__main__':
    folder = sys.argv[1]
    # Find all matching files
    files = glob.glob(f"{folder}/alphas_instance_*.csv")
    metadata_file = sys.argv[2]   # second CSV file

    # Load metadata
    meta = pd.read_csv(metadata_file)

    # Split instance IDs by label
    ids_label1 = meta[meta["label"] == 1]["instance_id"].tolist()
    ids_label0 = meta[meta["label"] == 0]["instance_id"].tolist()

    def accumulate(ids):
        cumulative = defaultdict(float)

        for instance_id in ids:
            fname = os.path.join(alphas_dir, f"alphas_instance_{instance_id}.csv")
            if not os.path.exists(fname):
                continue

            df = pd.read_csv(fname, header=None, names=["index", "alpha"])

            for idx, val in zip(df["index"], df["alpha"]):
                cumulative[idx] += val

        indices = sorted(cumulative.keys())
        values = [cumulative[i] for i in indices]

        return indices, values

    # Compute cumulative attributions
    idx1, val1 = accumulate(ids_label1)
    idx0, val0 = accumulate(ids_label0)

    # Plot
    plt.figure(figsize=(10,5))

    plt.plot(idx1, val1, label="label = 1")
    plt.plot(idx0, val0, label="label = 0")

    plt.xlabel("Observation index")
    plt.ylabel("Cumulative attribution")
    plt.title("Cumulative attribution per observation index by label")
    plt.legend()

    plt.tight_layout()
    plt.savefig("histogram_alphas.png")
