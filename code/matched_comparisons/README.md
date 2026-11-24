# Matched graph-tracking comparisons

From the parent code directory, first generate the 24 archived observation trajectories with the synthetic benchmark command in its README. Then run

    python matched_comparisons/matched.py . matched_results

The driver uses the original filter modules and checks the hard-threshold replay against every archived weight vector. The weighted panel contains 24 trajectories. The paired binary panel contains the 12 six-node trajectories with the same support, input and process innovations and unit active weights. Reconstructed innovations are used only to generate this paired simulation. Estimators receive only the generated observations, known parameters, and complete candidate list.

The binary tracker enumerates local binary neighborhoods and applies independent flips (probability 0.01) exactly along each bit axis. It uses the block diffusion likelihood and averages endpoint marginals before thresholding at 0.5 to obtain an undirected binary edge. This is an adaptation of Tenorio et al. (ICASSP 2025, doi:10.1109/ICASSP49660.2025.10890647 and arXiv:2609.11263), with approximate undirected reconciliation. Its estimates differ from the older repository ProbSSM controls in transition and readout.

The soft KF follows the one-step shrinkage configuration of Dabush et al. (arXiv:2507.09999) with the same bounded weights and blockwise Gaussian update as the existing hard-threshold control. The paper setting is 0.25. The development-selected setting minimizes average relative weight error on eight separate trajectories across node counts, graph generators and input levels. A threshold of 0.1 on the proposed posterior mean is an additional readout control. All comparisons on the original observations are retrospective. Event RMSE is the square root of mean squared increment error over all candidate edges and support-change blocks, weighting each block equally.

The reference scores and tuning results are included here. The script also writes per-block values to the selected output folder.
