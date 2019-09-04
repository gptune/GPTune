class GPTune(object):

    def __init__(self, problem : Problem, computer = None : Computer, data = None : Data, options = None : Options):

        self.problem = problem

        if (computer is None):
            self.computer = Computer()
        else:
            self.computer = computer

        if (data is None):
            self.data = Data()
        else:
            self.data = data

        if (options is None):
            self.options = options()
        else:
            self.options = options

        if (self.options['parallel_tuning']\
            and\
            ('mpi4py' in sys.modules)): # make sure that the mpi4py has been loaded successfully
            if (self.options['mpi_comm'] is None):
                self.mpi_comm = mpi4py.MPI.COMM_WORLD
            else:
                self.mpi_comm = self.options['mpi_comm']
            self.mpi_rank = self.mpi_comm.Get_rank()
            self.mpi_size = self.mpi_comm.Get_size()
        else: # fall back to sequential (MPI wise, but still multithreaded) tuning
            self.mpi_comm = None
            self.mpi_rank = 0
            self.mpi_size = 1

#        self.sampler   : Sampler
#        self.models    : Dict[Model]
#        self.searcher  : Searcher
#        self.evaluator : Evaluator

    def MLA():

        pass

    def TLA1():

        pass

    def TLA2(): # co-Kriging

        pass

