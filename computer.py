class Computer(object):

    def __init__(self, nodes = 1 : int, cores = 1 : int, number_of_processes_and_threads = None : Callable):

        self.nodes = nodes
        self.cores = cores

        self.number_of_processes_and_threads = number_of_processes_and_threads

    def evaluate(fun: Callable, args, parallel = False : bool):

        pass

#print(MPI.COMM_WORLD.Get_rank(), MPI.Get_processor_name())
