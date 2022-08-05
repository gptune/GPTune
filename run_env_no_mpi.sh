export GPTUNEROOT=$PWD
export PATH=$BREWPATH/python@3.9/$pythonversion/bin/:$PATH
export PATH=$GPTUNEROOT/env/bin/:$PATH
export PYTHONPATH=$GPTUNEROOT/GPTune:$PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/autotune/
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/scikit-optimize/
export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/GPy/
if [[ $(uname -s) == "Darwin" ]]; then
    export PYTHONPATH=$PYTHONPATH:$GPTUNEROOT/pygmo2/build/
fi
export PYTHONWARNINGS=ignore
