set -x 

mmax=20000
nmax=20000
ntask=10
nrun=10
nodes=16
exp=tuners
# read results
python scalapack_parse_results.py --mmax ${mmax} --nmax ${nmax} --ntask ${ntask} --nodes ${nodes} --exp ${exp}
# plot
python scalapack_plot_tunercompare.py --mmax ${mmax} --nmax ${nmax} --ntask ${ntask} --nodes ${nodes} --nrun ${nrun} --exp ${exp}


# mmax=20000
# nmax=20000
# ntask=5
# nrun1=10
# nrun2=20
# nrun3=40
# nodes=16
# exp=models
# # read results
# python scalapack_parse_results.py --mmax ${mmax} --nmax ${nmax} --ntask ${ntask} --nodes ${nodes} --exp ${exp}
# # plot
# python scalapack_plot_performancemodel.py --mmax ${mmax} --nmax ${nmax} --ntask ${ntask} --nodes ${nodes} --nrun1 ${nrun1} --nrun2 ${nrun2} --nrun3 ${nrun3} --exp ${exp}
