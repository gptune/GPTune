# set -x 
module load python

appstr=nimrod
ntask=1
bmin=1
bmax=8
eta=2
Nloop=2

# parse results for each run
for expid in 1 2 3 
do
    python parse_results_history.py -appstr ${appstr} -ntask ${ntask} -bmin ${bmin} -bmax ${bmax} -eta ${eta} -Nloop ${Nloop} -expid ${expid}
done

# plot for all runs
python plot_history.py -explist 1 2 3 -deleted_tuners None -appstr ${appstr} -ntask ${ntask} -bmin ${bmin} -bmax ${bmax} -eta ${eta} -Nloop ${Nloop}


