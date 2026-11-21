#!/usr/bin/env python3
"""Fine-tune the upstream Sign-MNIST CNN on our own ESP32 crops, quantize to
int8 and write firmware/asl_sign/model_data.cpp.

Colab (GPU runtime), after uploading this repo's training/ + tools/ folders and data/crops.csv:
  !pip -q install kagglehub
  !python training/train_asl.py --crops data/crops.csv --holdout carol   # honest number for judges
  !python training/train_asl.py --crops data/crops.csv                   # final model, all signers
  # then download firmware/asl_sign/model_data.cpp and re-flash

Architecture and normalisation (x/127.5 - 1) are the same as the upstream
notebook: J. Randall, "American-Sign-language", GitHub, 2023.
"""
import argparse
import os
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ------------------------------------------------------------------ data
def load_mnist(path):
    """Sign MNIST CSVs: label,pixel1..pixel784 with labels 0..24 (A..Y, no J=9)."""
    if path is None:
        try:
            import kagglehub
            path = kagglehub.dataset_download("datamunge/sign-language-mnist")
        except Exception as e:  # noqa: BLE001
            print(f"[mnist] kagglehub failed ({e}); pass --mnist-dir or --no-mnist")
            raise SystemExit(1)
    files = [p for p in pathlib.Path(path).rglob("sign_mnist_*.csv")]
    if not files:
        raise SystemExit(f"no sign_mnist_*.csv under {path}")
    df = pd.concat([pd.read_csv(f) for f in files])
    y = df.pop("label").to_numpy()
    x = df.to_numpy().astype(np.uint8)
    print(f"[mnist] {len(x)} images from {[f.name for f in files]}")
    return x, y


def load_crops(paths):
    dfs = [pd.read_csv(p) for p in paths]
    df = pd.concat(dfs)
    signers = df.pop("signer").astype(str).to_numpy()
    y = df.pop("label").to_numpy()
    df.pop("letter")
    x = df.to_numpy().astype(np.uint8)
    print(f"[crops] {len(x)} crops; per signer:",
          dict(zip(*np.unique(signers, return_counts=True))))
    return x, y, signers


def norm(x):
    return (x.astype(np.float32) / 127.5 - 1.0).reshape(-1, 28, 28, 1)


# ------------------------------------------------------------------ model
def build_model(tf, n_classes):
    L = tf.keras.layers
    m = tf.keras.Sequential([
        L.Input((28, 28, 1)),
        L.Conv2D(65, 3, padding="same", activation="relu"),
        L.MaxPool2D(2, 2, padding="same"),
        L.Conv2D(40, 3, padding="same", activation="relu"),
        L.Dropout(0.2),
        L.MaxPool2D(2, 2, padding="same"),
        L.Conv2D(25, 3, padding="same", activation="relu"),
        L.MaxPool2D(2, 2, padding="same"),
        L.Flatten(),
        L.Dense(256, activation="relu"),
        L.Dropout(0.3),
        L.Dense(n_classes, activation="softmax"),
    ])
    return m


def dataset(tf, x, y, batch, augment, flip):
    ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(len(x), reshuffle_each_iteration=True)
    if augment:
        aug = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(10 / 360, fill_mode="nearest"),
            tf.keras.layers.RandomZoom(0.12, fill_mode="nearest"),
            tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="nearest"),
        ])

        def f(img, lab):
            img = aug(img, training=True)
            if flip:  # left-handed signers
                img = tf.cond(tf.random.uniform(()) < 0.5,
                              lambda: tf.image.flip_left_right(img), lambda: img)
            img = img * tf.random.uniform((), 0.85, 1.15) + tf.random.uniform((), -0.1, 0.1)
            return tf.clip_by_value(img, -1.0, 1.0), lab
        ds = ds.batch(batch).map(f, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = ds.batch(batch)
    return ds.prefetch(tf.data.AUTOTUNE)


# ------------------------------------------------------------------ export
def to_tflite(tf, model, rep_x, out_path):
    def rep():
        for i in np.random.permutation(len(rep_x))[:400]:
            yield [rep_x[i:i + 1]]

    try:
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
    except Exception:  # Keras 3 fallback
        sm = str(out_path) + "_savedmodel"
        model.export(sm)
        conv = tf.lite.TFLiteConverter.from_saved_model(sm)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = rep
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    data = conv.convert()
    pathlib.Path(out_path).write_bytes(data)
    return data


def tflite_predict(tf, data, x):
    it = tf.lite.Interpreter(model_content=data)
    it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    s, zp = i["quantization"]
    ops = {d["op_name"] for d in it._get_ops_details()}
    preds = []
    for k in range(len(x)):
        q = np.clip(np.round(x[k:k + 1] / s + zp), -128, 127).astype(np.int8)
        it.set_tensor(i["index"], q)
        it.invoke()
        preds.append(int(np.argmax(it.get_tensor(o["index"])[0])))
    return np.array(preds), ops


def report(name, y_true, y_pred, labels):
    acc = (y_true == y_pred).mean()
    print(f"\n== {name}: {100 * acc:.1f}% on {len(y_true)} crops")
    per = []
    for c in np.unique(y_true):
        m = y_true == c
        per.append((labels[c], (y_pred[m] == c).mean(), m.sum()))
    print("   per letter:", " ".join(f"{l}:{100 * a:.0f}" for l, a, _ in per))
    pairs = {}
    for t, p in zip(y_true, y_pred):
        if t != p:
            pairs[(labels[t], labels[p])] = pairs.get((labels[t], labels[p]), 0) + 1
    top = sorted(pairs.items(), key=lambda kv: -kv[1])[:8]
    print("   top confusions:", ", ".join(f"{t}->{p} x{n}" for (t, p), n in top) or "none")
    return acc


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", nargs="*", default=[str(ROOT / "data" / "crops.csv")])
    ap.add_argument("--mnist-dir", help="folder containing sign_mnist_train.csv / _test.csv")
    ap.add_argument("--no-mnist", action="store_true", help="train on own crops only")
    ap.add_argument("--holdout", help="signer to hold out for an honest accuracy number")
    ap.add_argument("--epochs-pre", type=int, default=8, help="epochs on Sign MNIST")
    ap.add_argument("--epochs-ft", type=int, default=25, help="epochs on own crops (+ MNIST mix)")
    ap.add_argument("--no-flip", action="store_true", help="disable left/right flip augmentation")
    ap.add_argument("--out-dir", default=str(ROOT / "models"))
    ap.add_argument("--c-out", default=str(ROOT / "firmware" / "asl_sign" / "model_data.cpp"))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import tensorflow as tf
    tf.keras.utils.set_random_seed(a.seed)
    flip = not a.no_flip

    xc, yc, sc = load_crops(a.crops)
    if a.no_mnist:
        xm, ym = np.zeros((0, 784), np.uint8), np.zeros((0,), int)
    else:
        xm, ym = load_mnist(a.mnist_dir)

    # Classes = every letter that has data (Sign MNIST gives 24, our crops add J/Z).
    present = sorted(set(ym.tolist()) | set(yc.tolist()))
    labels = "".join(ALPHA[c] for c in present)
    remap = {c: i for i, c in enumerate(present)}
    ym = np.array([remap[v] for v in ym], dtype=np.int64)
    yc = np.array([remap[v] for v in yc], dtype=np.int64)
    n = len(labels)
    print(f"[classes] {n}: {labels}")
    missing = set(ALPHA) - set(labels)
    if missing:
        print(f"[classes] WARNING no data for {''.join(sorted(missing))} - collect them!")

    if a.holdout:
        te = sc == a.holdout
        if not te.any():
            raise SystemExit(f"no crops for signer {a.holdout!r}")
        xc_tr, yc_tr, xc_te, yc_te = xc[~te], yc[~te], xc[te], yc[te]
    else:
        # small random split just to watch training; final model uses it all anyway
        idx = np.random.RandomState(a.seed).permutation(len(xc))
        cut = max(1, len(xc) // 10)
        xc_te, yc_te = xc[idx[:cut]], yc[idx[:cut]]
        xc_tr, yc_tr = xc, yc

    Xm, Xc_tr, Xc_te = norm(xm), norm(xc_tr), norm(xc_te)
    model = build_model(tf, n)
    model.summary()
    lr_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy", patience=3, factor=0.5, min_lr=1e-5)

    # Phase 1: learn hand shapes from Sign MNIST (as upstream did).
    if len(Xm) and a.epochs_pre:
        model.compile(tf.keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", ["accuracy"])
        model.fit(dataset(tf, Xm, ym, 128, True, flip), epochs=a.epochs_pre,
                  validation_data=dataset(tf, Xc_te, yc_te, 256, False, False), verbose=2)

    # Phase 2: adapt to our camera. Own crops repeated so they are ~half of each epoch.
    reps = max(1, int(round(min(len(Xm), 20000) / max(1, len(Xc_tr)))))
    keep = np.random.RandomState(a.seed).permutation(len(Xm))[:min(len(Xm), 20000)]
    Xmix = np.concatenate([np.repeat(Xc_tr, reps, 0), Xm[keep]])
    ymix = np.concatenate([np.repeat(yc_tr, reps, 0), ym[keep]])
    print(f"[finetune] {len(Xc_tr)} own crops x{reps} + {len(keep)} MNIST")
    model.compile(tf.keras.optimizers.Adam(5e-4), "sparse_categorical_crossentropy", ["accuracy"])
    model.fit(dataset(tf, Xmix, ymix, 128, True, flip), epochs=a.epochs_ft,
              validation_data=dataset(tf, Xc_te, yc_te, 256, False, False),
              callbacks=[lr_cb], verbose=2)

    float_pred = np.argmax(model.predict(Xc_te, verbose=0), 1)
    title = f"held-out signer '{a.holdout}'" if a.holdout else "random 10% of own crops (seen in training)"
    report(f"float model, {title}", yc_te, float_pred, labels)

    out_dir = pathlib.Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"holdout_{a.holdout}" if a.holdout else "all"
    tfl_path = out_dir / f"asl{n}_{tag}.tflite"
    rep_x = np.concatenate([Xc_tr, Xm[keep[:2000]]]) if len(Xm) else Xc_tr
    data = to_tflite(tf, model, rep_x, tfl_path)
    q_pred, ops = tflite_predict(tf, data, Xc_te)
    acc = report(f"INT8 model, {title}", yc_te, q_pred, labels)
    print(f"\n[export] {tfl_path} ({len(data)} bytes), ops: {sorted(ops)}")
    ok_ops = {"CONV_2D", "MAX_POOL_2D", "RESHAPE", "FULLY_CONNECTED", "SOFTMAX",
              "DEPTHWISE_CONV_2D", "AVERAGE_POOL_2D", "QUANTIZE", "DEQUANTIZE", "DELEGATE"}
    if ops - ok_ops:
        print(f"[export] WARNING ops not registered in firmware: {ops - ok_ops}")

    if a.holdout:
        print(f"\nJudge number: {100 * acc:.0f}% per-frame accuracy on a signer the model never saw.")
        print("(Not writing model_data.cpp for a hold-out run; rerun without --holdout for the final model.)")
        return
    subprocess.check_call([sys.executable, str(ROOT / "tools" / "tflite_to_c.py"), str(tfl_path),
                           "--labels", labels, "--out", a.c_out,
                           "--note", f"fine-tuned on {len(xc)} own crops + Sign MNIST"])
    print("\nNext: copy model_data.cpp into firmware/asl_sign/ and re-flash.")


if __name__ == "__main__":
    main()
