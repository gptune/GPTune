import numpy as np
import os, sys, re
import mpi4py
from mpi4py import MPI
import time

# Paths
MACHINE_NAME = 'tmp'
TUNER_NAME = 'tmp'
ROOTDIR = os.path.abspath(os.path.join(os.path.realpath(__file__), os.pardir, os.pardir))
EXPDIR = os.path.abspath(os.path.join(ROOTDIR, "hypre-driver/exp", MACHINE_NAME + '/' + TUNER_NAME))
EXCUDIR = os.path.abspath(os.path.join(ROOTDIR, "hypre/src/test/ij"))
# print(EXPDIR)
# print(EXCUDIR)

max_setup_time = 1000
max_solve_time = 1000

def execute(params, RUNDIR, niter = 1):
    # extract arguments
    Problem = params['problem_name']; solver = params['solver']
    coeffs_c = params['coeffs_c']; coeffs_a = params['coeffs_a']
    nx = params['nx']; ny = params['ny']; nz = params['nz']
    Px = params['Px']; Py = params['Py']; Pz = params['Pz']
    strong_threshold = params['strong_threshold']
    trunc_factor = params['trunc_factor']
    P_max_elmts = params['P_max_elmts']
    coarsen_type = params['coarsen_type']
    relax_type = params['relax_type']
    smooth_type = params['smooth_type']
    smooth_num_levels = params['smooth_num_levels']
    interp_type = params['interp_type']
    agg_num_levels = params['agg_num_levels']
    nthreads = params['nthreads']
    npernode = params['npernode']

    
    # reshape for args
    NProc = Px*Py*Pz
    Size = "-n %d %d %d " % (nx, ny, nz)
    ProcTopo = "-P %d %d %d " % (Px, Py, Pz)
    StrThr = f"-th {strong_threshold} "
    TrunFac = f"-tr {trunc_factor} "
    PMax = "-Pmx %d " % P_max_elmts
    RelType = "-rlx %d " % relax_type
    SmooType = "-smtype %d " % smooth_type
    SmooLev = "-smlv %d " % smooth_num_levels 
    InterType = "-interptype %d " % interp_type 
    AggLev = "-agg_nl %d " % agg_num_levels
    CoarsTypes = {0:"-cljp", 1:"-ruge", 2:"-ruge2b", 3:"-ruge2b", 4:"-ruge3c", 6:"-falgout", 8:"-pmis", 10:"-hmis"}
    CoarsType = CoarsTypes[coarsen_type]

    outputfilename = os.path.abspath(os.path.join(RUNDIR,f"ijoutput_{nx}_{ny}_{nz}_{Px}_{Py}_{Pz}_{strong_threshold}_{trunc_factor}_{P_max_elmts}_{coarsen_type}_{relax_type}_{smooth_type}_{smooth_num_levels}_{interp_type}_{agg_num_levels}"))
    myargs = Problem + Size + coeffs_c + coeffs_a + f"-solver {solver} " + ProcTopo + StrThr + TrunFac + PMax + RelType + SmooType + SmooLev + InterType + AggLev + CoarsType
    myargslist = [Problem, '-n', f'{nx}', f'{ny}', f'{nz}', coeffs_c, coeffs_a, '-solver', f'{solver}', '-P', f'{Px}', f'{Py}', f'{Pz}', '-th', f'{strong_threshold}', '-tr', f'{trunc_factor}', 
                  '-Pmx', f'{P_max_elmts}', '-rlx', f'{relax_type}', '-smtype', f'{smooth_type}', '-smlv', f'{smooth_num_levels}', '-interptype', f'{interp_type}', '-agg_nl', f'{agg_num_levels}', CoarsType, '-logfile', outputfilename]
    
    def read_output(outputfilename):
        setup_time = max_setup_time
        solve_time = max_solve_time
        with open(outputfilename,'r') as outputfile:
            while True:
                line = outputfile.readline()
                if not line:
                    break
                if 'ERROR' in line:
                    break
                if 'Setup phase times' in line:
                    outputfile.readline()
                    outputfile.readline()
                    setup_wallclocktime_str = outputfile.readline()
                    time_str = re.findall("\d+\.\d+", setup_wallclocktime_str)
                    if time_str:
                        setup_time = float(time_str[0])
                if 'Solve phase times' in line:
                    outputfile.readline()
                    outputfile.readline()
                    solve_wallclocktime_str = outputfile.readline()
                    time_str = re.findall("\d+\.\d+", solve_wallclocktime_str)
                    if time_str:
                        solve_time = float(time_str[0])
        runtime = setup_time + solve_time
        print("[----- runtime = %f -----]\n" % runtime)
        return runtime

    def v_parallel():

        info = MPI.Info.Create()
        info.Set('env', 'OMP_NUM_THREADS=%d\n' %(nthreads))
        info.Set('npernode','%d'%(npernode))  # YL: npernode is deprecated in openmpi 4.0, but no other parameter (e.g. 'map-by') works
        

        print('exec ', EXCUDIR, 'args: ', myargslist, 'nproc', NProc)
        runtimes = []
        for i in range(niter):
            os.system("rm -rf %s"%(outputfilename))
            comm = MPI.COMM_SELF.Spawn(EXCUDIR, args=myargslist, maxprocs=NProc,info=info)
            comm.Disconnect()
            time.sleep(2.0) # this gives new MPI_spawn more time to find the resource
            runtime = read_output(outputfilename)
            runtimes.append(runtime)
        return np.mean(runtimes)             
    
    runtime = v_parallel()
    return runtime

def hypredriver(params, niter = 3, JOBID: int=-1):
    global EXPDIR 
    global ROOTDIR

    MACHINE_NAME = os.environ['MACHINE_NAME']
    TUNER_NAME = os.environ['TUNER_NAME']
    EXPDIR = os.path.abspath(os.path.join(ROOTDIR, "hypre-driver/exp", MACHINE_NAME + '/' + TUNER_NAME))

    if (JOBID==-1):  # -1 is the default value if jobid is not set from command line
        JOBID = os.getpid()
    RUNDIR = os.path.abspath(os.path.join(EXPDIR, str(JOBID)))
    os.makedirs("%s"%(RUNDIR),exist_ok=True)
    dtype = [("nx", int), ("ny", int), ("nz", int), ("coeffs_a", 'U10'), ("coeffs_c", 'U10'), ("problem_name", 'U10'), ("solver", int), 
            ("Px", int), ("Py", int), ("Pz", int), ("strong_threshold", float), 
            ("trunc_factor", float), ("P_max_elmts", int), ("coarsen_type", int), ("relax_type", int),
            ("smooth_type", int), ("smooth_num_levels", int), ("interp_type", int), ("agg_num_levels", int), ("nthreads", int), ("npernode", int)]
    params = np.array(params, dtype=dtype)
    times = []
    for param in params:
        print(f"Current param {param}")
        time_cur = execute(param, RUNDIR, niter=niter)
        times.append(time_cur)
    # os.system('rm -fr %s'%(RUNDIR))
    return times


if __name__ == "__main__":
    os.environ['MACHINE_NAME'] = 'cori'
    os.environ['TUNER_NAME'] = 'GPTune'
    params = [(60, 50, 80, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian ', 3, 2, 2, 2, 0.25, 0, 4, 10, 8, 6, 0, 6, 0, 1, 1),\
              (60, 50, 80, '-a 0 0 0 ', '-c 1 1 1 ', '-laplacian ', 3, 2, 2, 2, 0.3, 0.2, 5, 10, 8, 6, 1, 6, 1, 1, 1)
              ]
    times = hypredriver(params, niter=1)
    
    print(times)
