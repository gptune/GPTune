# set -x 

dataset="susy_10Kn"
ntask=1
bmin=2
bmax=8
eta=2
Nloop=1

# parse results for each run
for expid in TEST1 TEST2 TEST3
do
    python krr_parse_results_history.py -dataset ${dataset} -ntask ${ntask} -bmin ${bmin} -bmax ${bmax} -eta ${eta} -Nloop ${Nloop} -expid ${expid}
done

# plot for all runs
python krr_plot_history.py -explist TEST1 TEST2 TEST3 -deleted_tuners None -dataset ${dataset} -ntask ${ntask} -bmin ${bmin} -bmax ${bmax} -eta ${eta} -Nloop ${Nloop}


