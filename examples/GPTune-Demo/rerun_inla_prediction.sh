#!/bin/bash
# Prediction of a timed-out 3D INLA training of model_comparison_inla_wendland.py at its final hyperparameters:
# reads the last row of the training history and submits the harness with the hyperparameter ranges pinned to
# it (noise and amplitude within 0.01%, the length scales within their fitted span) and the initial guess set to
# them; no optimization is done (GP_LBFGS_MAXITER=0) and the one history point is replayed for the test metrics.
# Usage: ./rerun_inla_prediction.sh <results dir> <objtype> <N> <nu> <isotropic 0|1> [walltime] [suffix]
# STAGE=1 appends the submission line to spatial_h2/stage_premium.queue (the premium stager) instead of submitting.
set -e
DIR=$1; OBJ=$2; N=$3; NU=$4; ISO=$5; WALL=${6:-03:00:00}; SUFFIX=${7:-_pred}
CSV=$DIR/inla/training_iterations_obj${OBJ}_N${N}_gradient.csv
read NOISE AMP LMIN LMAX LGEO LS <<< $(python3 - "$CSV" <<'PY'
import sys, numpy as np
r = np.genfromtxt(sys.argv[1], delimiter=',', names=True)
r = r[-1] if r.ndim else r
hp = [r[k] for k in r.dtype.names if k.startswith('hp_')]
ls = np.sqrt(np.exp(np.array(hp[2:])))
print(np.exp(hp[0]), np.exp(hp[1]), ls.min(), ls.max(), np.exp(np.mean(np.log(ls))), ','.join('%.8g' % v for v in ls))
PY
)
echo "final hyperparameters: noise $NOISE amplitude $AMP length scales $LMIN..$LMAX"
SHAPE=${GP_INLA_SHAPE:+GP_INLA_SHAPE=$GP_INLA_SHAPE }
LINE="GP_LBFGS_MAXITER=0 GP_REPLAY_POINTS=2 OBJ_CHOICE=$OBJ GP_OBJECTIVE_SET=bpack GP_NS=$N GP_MODELS=inla GP_OPTIMIZERS=gradient GP_INLA_NU=$NU GP_ISOTROPIC=$ISO GP_INLA_BUFFER=0.5 ${SHAPE}\
GP_NOISEVAR=\"$(python3 -c "print($NOISE*0.9999, $NOISE*1.0001, $NOISE)")\" \
GP_AMPLITUDE=\"$(python3 -c "print($AMP*0.9999, $AMP*1.0001, $AMP)")\" \
GP_LENGTHSCALE=\"$(python3 -c "print($LMIN*0.9999, $LMAX*1.0001, $LGEO)")\" GP_INLA_INITIAL_LENGTHSCALE=\"${LS//,/ }\" \
RESULTS_DIR=\$PWD/${DIR#$PWD/}${SUFFIX} sbatch -t $WALL run_inla_wendland_gpu.sbatch"
if [ "${STAGE:-0}" = 1 ]; then
    echo "$LINE" >> $(dirname $0)/spatial_h2/stage_premium.queue
    echo "staged: $LINE"
else
    cd $(dirname $0) && eval "$LINE"
fi
