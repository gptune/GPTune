# GPTune Copyright (c) 2019, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of any
# required approvals from the U.S.Dept. of Energy) and the University of
# California, Berkeley.  All rights reserved.
#
# If you have questions about your rights to use or distribute this software, 
# please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.
#
# NOTICE. This Software was developed under funding from the U.S. Department 
# of Energy and the U.S. Government consequently retains certain rights.
# As such, the U.S. Government has been granted for itself and others acting
# on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in
# the Software to reproduce, distribute copies to the public, prepare
# derivative works, and perform publicly and display publicly, and to permit
# other to do so.
#

class GPTune(object):

    def __init__(self, problem : Problem, computer = Computer() : Computer, data = Data() : Data, options = Options() : Options, **kwargs):

        self.problem  = problem
        self.computer = computer
        self.data     = data
        self.options  = options

        if (self.options['distributed_memory_parallelism']\
            and\
            ('mpi4py' in sys.modules)): # make sure that the mpi4py has been loaded successfully
            if ('mpi_comm' in kwargs):
                self.mpi_comm = kwargs['mpi_comm']
            else:
                self.mpi_comm = mpi4py.MPI.COMM_WORLD
            self.mpi_rank = self.mpi_comm.Get_rank()
            self.mpi_size = self.mpi_comm.Get_size()
        else: # fall back to sequential tuning (MPI wise, but still multithreaded)
            self.mpi_comm = None
            self.mpi_rank = 0
            self.mpi_size = 1

    def MLA(NS, NS1 = None, NI = None, **kwargs):

        kwargs = (copy.deepcopy(self.options).update(kwargs))
        kwargs = kwargs.update({'mpi_comm' : self.mpi_comm})

        """ Multi-task Learning Autotuning """

        if (self.mpi_rank == 0):

            sampler = eval(f'{kwargs["sample"]}()')

            if (self.data.T is None):

                check_constraints = functools.partial(\
                        self.computer.evaluate_constraints,\
                        constraints = self.problem.constraints
                        inputs_only = True,\
                        kwargs = kwargs)
                self.data.T = sampler.sample_inputs(n_samples = NI, IS = self.problem.IS, check_constraints = check_constraints, **kwargs)

            if (self.data.X is None):

                if (NS1 is None):
                    NS1 = min(NS - 1, 3 * self.problem.DP)
    
                check_constraints = functools.partial(\
                        self.computer.evaluate_constraints,\
                        constraints = self.problem.constraints
                        inputs_only = False,\
                        kwargs = kwargs)
                self.data.X = sampler.sample_parameters(n_samples = NS1, T = self.data.T, IS = self.problem.IS, PS = self.problem.PS, check_constraints = check_constraints, **kwargs)

            if (self.data.Y is None):
                self.Y = self.computer.evaluate_objective(\
                        fun = self.problem.objective,\
                        T = self.data.T,\
                        X = self.data.X,\
                        kwargs = kwargs)

            if ((self.mpi_comm is not None) and (self.mpi_size > 1)):
                mpi_comm.bcast(self.data, root=0)

        else:

            self.data = mpi_comm.bcast(None, root=0)

        NS2 = NS - len(self.X[0])

        modeler  = eval(f'{kwargs["model"]} (problem = self.problem)')
        searcher = eval(f'{kwargs["search"]}(problem = self.problem, computer = self.computer)')

        for optiter in range(NS2):

            newdata = Data(T = self.data.T)

            modeler.train(data = self.data, **kwargs)
            newdata.X = searcher.search_multitask(data = self.data, model = modeler, **kwargs)

            if (self.mpi_rank == 0):

                newdata.Y = self.computer.evaluate_objective(\
                        fun = self.problem.objective,\
                        T = newdata.T,\
                        X = newdata.X,\
                        kwargs = kwargs)

                if ((self.mpi_comm is not None) and (self.mpi_size > 1)):
                    mpi_comm.bcast(newdata.Y, root=0)

            else:

                newdata.Y = mpi_comm.bcast(None, root=0)

            self.data.merge(newdata)

        return (copy.deepcopy(self.data), modeler)

    def TLA1():

        pass

    def TLA2(): # co-Kriging

        pass

