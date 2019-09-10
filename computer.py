# GPTune Copyright (c) 2019, The Regents of the University of California, through Lawrence Berkeley National Laboratory
# (subject to receipt of any required approvals from the U.S.Dept. of Energy) and the University of California, Berkeley.
# All rights reserved.
#
# If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's 
# Intellectual Property Office at IPO@lbl.gov.
#
# NOTICE. This Software was developed under funding from the U.S. Department of Energy and the U.S. Government consequently
# retains certain rights. As such, the U.S. Government has been granted for itself and others acting on its behalf a paid-up,
# nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the public,
# prepare derivative works, and perform publicly and display publicly, and to permit other to do so.
#

class Computer(object):

    def __init__(self, nodes = 1 : int, cores = 1 : int, number_of_processes_and_threads = None : Callable):

        self.nodes = nodes
        self.cores = cores

        if (number_of_processes_and_threads is not None):
            self.number_of_processes_and_threads = number_of_processes_and_threads
        else:
            self.number_of_processes_and_threads = lambda point: (1, 1) # Fall back to sequential evaluation by default

    # TODO
    def evaluate_constraints(constraints : Dict, points : Dict, inputs_only = True : bool, **kwargs):

#       kwargs['constraints_evaluation_parallelism']

        # points can be either a dict or a list of dicts on which to iterate

        cond = True
        for cstname, cst in self.cstrs.items():
            if (isinstance(cst, str)):
                try:
                    # {} has to be the global argument to eval
                    # and kwargs the local one, otherwise,
                    # kwargs will be corrupted / updated by eval
                    cond = eval(cst, {}, kwargs)
                except Exception as inst:
                    if (on_task_parameters_only and isinstance(inst, NameError)):
                        pass
                    else:
                        raise Exception(f"Unexpected exception '{inst}' was raised while evaluating constraint '{cstname}'. Correct this constraint before calling the tuner again.")
            else:
                try:
                    kwargs2 = {}
                    sig = inspect.signature(cst)
                    for varname in kwargs:
                        if (varname in sig.parameters):
                            kwargs2[varname] = kwargs[varname]
                    cond = cst(**kwargs2)
                except Exception as inst:
                    if (isinstance(inst, TypeError)):
                        lst = inst.__str__().split()
                        if (len(lst) >= 5 and lst[1] == 'missing' and lst[3] == 'required' and lst[4] == 'positional'):
                            pass
                        else:
                            raise Exception(f"Unexpected exception '{inst}' was raised while evaluating constraint '{cstname}'. Correct this constraint before calling the tuner again.")
                    else:
                        raise Exception(f"Unexpected exception '{inst}' was raised while evaluating constraint '{cstname}'. Correct this constraint before calling the tuner again.")
            if (not cond):
                break

        return cond


    # TODO
    def evaluate_objective(fun : Callable, T = None : np.ndarray, X = None : Collection[np.ndarray], **kwargs):

#        kwargs['objective_evaluation_parallelism'])

        Y = []
        for i in range(len(T)):
            t = T[i]
            kwargst = {self.TSorig[k].name: t[k] for k in range(self.DT)}
            X2 = X[i]
            Y2 = []
            for j in range(len(X2)):
                x = X2[j]
                kwargs = {self.ISorig[k].name: x[k] for k in range(self.DI)}
                kwargs.update(kwargst)
                y = self.objfun(**kwargs)
                Y2.append(y)
            Y.append(np.array(Y2).reshape((len(Y2), self.DO)))

        return Y

#print(MPI.COMM_WORLD.Get_rank(), MPI.Get_processor_name())
