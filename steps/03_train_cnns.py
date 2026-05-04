# Run all CNN experiments by shelling out to lib/train_cnn.py.

import argparse
import json
import os
import subprocess
import sys

TRAIN_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "lib", "train_cnn.py")

DEFAULT_HPARAMS = {
    "simple": {"lr": 1e-3, "base_ch": 32, "dropout": 0.3},
    "resnet": {"lr": 5e-4, "base_ch": 32, "dropout": 0.3},
    "tcn":    {"lr": 1e-3, "base_ch": 32, "dropout": 0.2},
}


def load_hparams(hparam_file):
    if not hparam_file or not os.path.exists(hparam_file):
        return DEFAULT_HPARAMS
    with open(hparam_file) as f:
        hpo = json.load(f)
    out = {}
    for arch, records in hpo.items():
        best = max(records, key=lambda r: r["val_auc_pr"])
        out[arch] = {"lr": best["lr"], "base_ch": best["base_ch"],
                     "dropout": best["dropout"]}
        print(f"  HPO {arch}: lr={best['lr']:.0e} ch={best['base_ch']} "
              f"do={best['dropout']} (val AUC-PR={best['val_auc_pr']:.4f})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all",
                    choices=["all", "arch", "ablation", "extended"])
    ap.add_argument("--arch", nargs="+", default=["simple", "resnet", "tcn"],
                    choices=["simple", "resnet", "tcn"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[3, 33, 333])
    ap.add_argument("--results_dir", default="experiments")
    ap.add_argument("--hparam_file", default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    hparams = load_hparams(args.hparam_file)

    # build the run list. each row = (label_or_None, extra_flags_dict, base_ch_override)
    # for the train_cnn.py CLI. label=None is the plain architecture comparison.
    runs = []

    if args.mode in ("all", "arch"):
        runs.append((None, {}, None))

    if args.mode in ("all", "ablation"):
        runs += [
            ("no_sampler",            {"no_sampler": True}, None),
            ("no_augment",            {"no_augment": True}, None),
            ("no_sampler_no_augment", {"no_sampler": True, "no_augment": True}, None),
        ]

    if args.mode in ("all", "extended"):
        runs += [
            ("pos_weight",       {"pos_weight": -1},   None),
            ("focal_loss",       {"focal_loss": True}, None),
            ("reduced_capacity", {},                   16),
            ("window_168",       {"window_size": 168}, None),
            ("window_84",        {"window_size": 84},  None),
        ]

    n_done = n_skip = n_fail = 0
    for label, extra, base_ch_over in runs:
        for arch in args.arch:
            hp = hparams.get(arch, DEFAULT_HPARAMS[arch])
            base_ch = base_ch_over if base_ch_over is not None else hp["base_ch"]
            for seed in args.seeds:
                tag = (f"{arch}_seed{seed}" if label is None
                       else f"{arch}_{label}_seed{seed}")
                out_path = os.path.join(args.results_dir, f"results_{tag}.json")
                if args.resume and os.path.exists(out_path):
                    print(f"  skip {tag}")
                    n_skip += 1
                    continue

                cmd = [
                    sys.executable, TRAIN_SCRIPT,
                    "--model", arch,
                    "--lr", str(hp["lr"]),
                    "--base_ch", str(base_ch),
                    "--dropout", str(hp["dropout"]),
                    "--seed", str(seed),
                    "--results_dir", args.results_dir,
                    "--tag", tag,
                ]
                for k, v in extra.items():
                    if v is True:
                        cmd.append(f"--{k}")
                    else:
                        cmd.extend([f"--{k}", str(v)])

                print(f"  run {tag}")
                rc = subprocess.run(cmd).returncode
                if rc == 0:
                    n_done += 1
                else:
                    print(f"  FAILED {tag} (rc={rc})")
                    n_fail += 1

    print(f"\n{n_done} done, {n_skip} skipped, {n_fail} failed -> {args.results_dir}/")


if __name__ == "__main__":
    main()
