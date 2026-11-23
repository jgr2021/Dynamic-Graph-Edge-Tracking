# Dynamic Graph Edge Tracking

Research code for **Recursive Probabilistic Estimation of Dynamic Graph Edge Weights from Node Signals** by Gurui Jin and Ercan E. Kuruoğlu.

## Problem and method

Given synchronized scalar signals at the nodes of a network, the estimator tracks which candidate edges are present and how strongly they couple their endpoints. It assumes known first-order diffusion dynamics and permits edge weights to change between observation blocks.

A node–edge incidence matrix and the observed endpoint differences give a regression in the unknown edge weights. The filter predicts a discrete weight distribution for each edge and updates it using mean-field inference. The mass at zero determines the support decision. For edges declared present, the conditional mean gives the weight. These point estimates determine the graph Laplacian and its spectrum.

## Reproduce the experiments

From a fresh clone, install the dependencies and run the synthetic graph benchmark:

```sh
python -m pip install -r requirements.txt
cd code
python theta_u_independent_20261027/run.py
python table1_probssm.py
```

| Experiment | Question | Entry point (relative to code/) |
| --- | --- | --- |
| Synthetic graph tracking | How accurately are weights, eigenvalues and connected components recovered? | [Graph benchmark](code/theta_u_independent_20261027/run.py) |
| Binary-state baseline | How do posterior-average and local-MAP readouts compare on the same observations? | [ProbSSM comparison](code/table1_probssm.py) |
| Synthetic altitude coordination | How does estimation affect a one-step decision at a shared state? | [Coordination benchmark](code/theta_u_spl_review_round2_20261102/run_uav.py) |
| Measured MILUV flights | Does estimated coupling improve one-step altitude prediction over persistence? | [Flight prediction](code/miluv_prediction.py) |

See the [reproduction guide](code/README.md) for data preparation, fixed settings, output files and baseline attribution.

## Results in the manuscript

Across 24 synthetic trajectories, mean relative edge-weight error was 0.12163 for the proposed estimator and 0.15134 for the adapted sparse EKF. False splits occurred in 8 and 76 of 1,068 connected blocks, respectively. The proposed estimator incurred higher computational cost and greater edge-increment error at abrupt changes.

On ten measured three-UAV flights, one-step altitude RMSE was 18.04 mm for the proposed method, 18.43 mm for the adapted sparse EKF and 18.01 mm for persistence. These flight data evaluate short-horizon prediction. Edge-weight recovery is assessed on the synthetic trajectories with known ground truth.
