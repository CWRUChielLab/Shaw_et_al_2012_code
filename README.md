Code for the simulations, figures, and movies from "Phase Resetting in an Asymptotically Phaseless System"
----------------------------------------------------------------------------------------------------------

This repository contains the code used to generate the figures and movies for the following
paper:

Shaw KM, Park Y-M, Chiel HJ, Thomas PJ. Phase Resetting in an Asymptotically
Phaseless System: On the Phase Response of Limit Cycles Verging on a
Heteroclinic Orbit. SIAM Journal on Applied Dynamical Systems 2012;11:350–391.

Published version: http://epubs.siam.org/siads/resource/1/sjaday/v11/i1/p350_s1

Preprint (freely downloadable): http://arxiv.org/abs/1103.5647

To generate the various figures from the paper, you will need to run the following commands:

    python generate_figures.py
    python generate_figures_ml.py
    octave irisprc_coupling_check.m

To generate the movies from the paper, you will need to run

    python generate_movies.py

Note that these scripts require the user to have installed python 2.6 or above,
scipy, matplotlib, mencoder, and mplex.
