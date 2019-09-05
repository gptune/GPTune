class Data(object):

    def __init__(self, problem : Problem, T = None : np.ndarray, X = None : Collection[np.ndarray], Y = None : Collection[np.ndarray]):

        self.problem = problem

        if (not self.check_inputs(T)):
            raise Exception("")

        self.T = T

        if (not self.check_parameters(X)):
            raise Exception("")

        self.X = X

        if (not self.check_outputs(Y)):
            raise Exception("")

        self.Y = Y

    @property
    def NI(self):

        if (self.T is None):
            return 0
        else:
            return len(self.T)

    def check_inputs(self, T: np.ndarray) -> bool:

        cond = True
        if (T is not None):
            if (not (T.ndim == 2 and T.shape[1] == problem.DI)):
                cond = False

        return cond

    def check_parameters(self, X: Collection[np.ndarray]) -> bool:

        cond = True
        if (X is not None):
            for x in X:
                if (x is not None and len(x) > 0):
                    if not (x.ndim == 2 and x.shape[1] == problem.DP):
                        cond = False
                        break

        return cond

    def check_outputs(self, T: Collection[np.ndarray]) -> bool:

        cond = True
        if (Y is not None):
            for y in Y:
                if (y is not None and len(y) > 0):
                    if not (y.ndim == 2 and y.shape[1] == problem.DO)
                        cond = False
                        break

        return cond

    def points2kwargs(self):

        # transform the self.T and self.X into a list of dictionaries

        pass

    def merge(self, newdata):

        # merge the newdata with self, making sure that the Ts coincide

        pass

#    def insert(T = None: np.ndarray, X = None : Collection[np.ndarray], Y = None : Collection[np.ndarray]):
#
#        if (T is not None):
#            if (T.ndim == 1):
#                assert(T.shape[0] == self.problem.DI)
#            elif (T.ndim == 2):
#                assert(T.shape[1] == self.problem.DI)
#            else:
#                raise Exception("")
#            self.T.append(T)

