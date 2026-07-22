OSCL MATLAB Demo

This folder contains a compact MATLAB implementation of **Online Semantic
Correlation Learning (OSCL)** for streaming image--text retrieval. The code
uses the manuscript notation: `X` and `Y` are image/text features, `Z` is the
annotation matrix, and `W` and `G` are the learned projections.

## Datasets and data splits

The online training chunks and query data for the four evaluated datasets—
IAPR TC-12, MS-COCO, NUS-WIDE, and MIRFlickr—are available from the
[OSCL dataset folder on Google Drive](https://drive.google.com/drive/folders/1q-LamOdA_M3J0hnLrjm03qF08ZvmnhBe).

## Running the demo

Place a prepared dataset (for example, `MIRFlickr.mat` or
`NUSWIDE21.mat`) beside the MATLAB files. Set `dataFile` in
`main_demo.m`, and then run the following command in MATLAB:

```matlab
main_demo
```

The demo evaluates OSCL after every incoming data chunk and reports
image-to-text (I-to-T) mAP, text-to-image (T-to-I) mAP, per-round training
time, and accumulated training time. Metrics, timing results, and projection
matrices are saved automatically in the `results` folder.

### Example output: MIRFlickr

The following table shows representative output for MIRFlickr. Each round
introduces 2,000 additional training pairs.

| Round | Training pairs | I-to-T mAP | T-to-I mAP | Accumulated training time (s) |
|------:|---------------:|-----------:|-----------:|------------------------------:|
| 1 | 2,000 | 0.6719 | 0.7649 | 5.95 |
| 2 | 4,000 | 0.6937 | 0.7556 | 18.07 |
| 3 | 6,000 | 0.7054 | 0.7497 | 36.70 |
| 4 | 8,000 | 0.7105 | 0.7432 | 56.94 |
| 5 | 10,000 | 0.7163 | 0.7415 | 81.45 |
| 6 | 12,000 | 0.7211 | 0.7418 | 106.45 |
| 7 | 14,000 | 0.7239 | 0.7413 | 133.09 |
| 8 | 16,000 | 0.7256 | 0.7418 | 163.73 |
| 9 | 18,000 | 0.7257 | 0.7404 | 207.39 |

The accumulated training time at round *t* includes the initialization time
and all online updates through round *t*. It excludes feature encoding and
retrieval evaluation. Timing values depend on the hardware and MATLAB
environment.

## Files

- `main_demo.m` — settings, data loading, result summary, and saving.
- `evaluate_OSCL.m` — round-by-round training and evaluation workflow.
- `train_OSCL.m` — first-round initialization and later-round dispatch.
- `update_OSCL_svd.m` — mean-corrected incremental SVD update.
- `solve_OSCL_projection.m` — closed-form computation of `W` and `G`.
- `compute_retrieval_metrics.m` — mAP, precision-at-K, and P--R metrics.

The implementation is self-contained and uses standard MATLAB functions.
No external evaluation utilities or Statistics and Machine Learning Toolbox
functions are required.

## Compared algorithms

The following open-source implementations were used as comparison methods:

- [Multi-view Multi-label Canonical Correlation Analysis for Cross-modal Multimedia Retrieval](https://github.com/asharani97/Fast-MVMLCCA)
- [Multi-modal Semantic Autoencoder for Cross-modal Retrieval](https://github.com/yiling2018/mmsae)
- [Discrete Online Cross-modal Hashing with Consistency Preservation](https://github.com/XX-kang/DOCMH)
- [OH-CMH: Towards Cross-modal Hashing for Streaming Data with Hierarchical Labels and Label Increment Scenario](https://github.com/pjunjie/OH-CMH)
- [Multiple Information Embedded Hashing for Large-Scale Cross-modal Retrieval](https://github.com/yxinwang/MIEH)
- [One for More: Structured Multi-modal Hashing for Multiple Multimedia Retrieval Tasks](https://github.com/ChaoqunZheng/SMMH)
- [Discrete Online Cross-modal Hashing](https://github.com/yw-zhan/DOCH)

## Dataset format

A prepared dataset MAT file must contain the following variables:

```matlab
XChunk   % T-by-1 cell; XChunk{t} is n_t-by-d_x image features
YChunk   % T-by-1 cell; YChunk{t} is n_t-by-d_y text features
LChunk   % T-by-1 cell; LChunk{t} is n_t-by-c labels
XTest    % n_q-by-d_x image queries
YTest    % n_q-by-d_y text queries
LTest    % n_q-by-c query labels
```

`main_demo.m` aliases `LChunk/LTest` to `ZChunk/ZQuery` and
`XTest/YTest` to `XQuery/YQuery`, so the algorithm code remains consistent
with the manuscript. Rows must represent aligned image--text pairs, and all
chunks must use the same label vocabulary.

