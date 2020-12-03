set -x 


ntask=20
nrun1=20
nrun2=40
nrun3=80
nodes=1
exp=models
# read results
python demo_parse_results.py --ntask ${ntask} --nodes ${nodes} --exp ${exp}
# plot
python demo_plot_performancemodel.py --ntask ${ntask} --nodes ${nodes} --nrun1 ${nrun1} --nrun2 ${nrun2} --nrun3 ${nrun3} --exp ${exp}
