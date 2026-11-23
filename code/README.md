# Recursive probabilistic estimation of dynamic graph edge weights

This directory contains the numerical experiment drivers, graph generator, frozen 65-state transition model, baseline adaptations and scoring code for the paper by Gurui Jin and Ercan E. Kuruoğlu. Run the commands below from this directory. Python 3.9 with NumPy 1.26.4, SciPy 1.13.1 and Numba 0.60 was used to check the release. Install dependencies with `python -m pip install -r requirements.txt`. On Windows use a short checkout path, or set the environment variable NUMBA_CACHE_DIR to a short writable directory before running.

## Synthetic graph benchmark (Table I)

```sh
python theta_u_independent_20261027/run.py
python table1_probssm.py
```

The first command generates all 24 trajectories from fixed seeds. Each trajectory has 72 blocks of 48 transitions. It writes observations, truth, estimates, and SUMMARY.json under theta_u_independent_20261027. The reported proposed, sparse EKF adaptation, and Gaussian filter with sparse readout correspond to old_support, hard_box, and gaussian. Other archived outputs are controls. The second command computes average and local-MAP ProbSSM readouts on exactly these observations. Both use the same local binary-state posterior. The stable log-domain scorer was checked against the archived upstream implementation across all 1,728 blocks. Their maximum absolute point differences were below 7e-15 (average) and zero (local MAP). Published timing is machine-dependent.

The original supplementary runner in theta_u_probssm_comparison_20261027 used lital-dgold/PolyEKF at commit cbb1a0a0bd911ecc7c5ff3d034e8336f31e1dfea. These optional upstream files are not redistributed. The archived URL returned HTTP 404 during release verification. Install networkx, matplotlib and psutil, and use prepare_probssm.py --source /path/to/original/modules with a copy of the two modules, or try its default upstream URL, then run run.py and analyze.py in that directory, in order. The Table I commands above work independently of the upstream repository.

## Synthetic altitude-coordination experiment (Table II)

```sh
python theta_u_spl_review_round2_20261102/run_uav.py
```

This generates the six geometries, both excitation levels, 432 observation blocks, both filters and matched-state one-step decision scores. Results appear under uav_run/SUMMARY.json. The simulated actions are scored at a shared terminal state. The experiment does not feed them back into the trajectories. The synthetic drivers create new output files exclusively. Re-run in a fresh checkout or move a previous output directory and generated summary files aside.

## Measured-flight prediction (MILUV)

Download the original ROS bags for the ten recording identifiers in miluv_recordings.json from the [MILUV dataset](https://www.decar.ca/miluv/). For each recording and each of ifo001, ifo002 and ifo003, extract the pose topic using the separate optional dependency `rosbags` in a compatible Python environment. For example:

```sh
python export_mocap.py /path/to/ifo001.bag ifo001 data/default_3_random_0/ifo001/mocap.csv
python miluv_prediction.py data my_miluv_results.json
```

The second command expects all thirty CSVs under data/recording/robot/. Each uses integer header timestamps in nanoseconds and vertical position in metres. The inference and prediction code uses the last available altitude on each 10 Hz grid point, restarts after gaps and scores the next block before updating on it. miluv_results.json contains recording-level squared errors and counts. These evaluate altitude prediction; reference inter-vehicle feedback weights are unavailable.

## Attribution

The hard-threshold EKF adaptation follows Dabush, Shlezinger and Routtenberg, [Sparsity-Aware Extended Kalman Filter for Tracking Dynamic Graphs](https://arxiv.org/abs/2507.09999). The binary ProbSSM comparison uses the separate repository baseline described above. MILUV is the multi-UAV indoor localization dataset of Shalaby et al., [dataset paper](https://doi.org/10.1177/02783649251405898). Flight logs are obtained separately from the dataset authors.

