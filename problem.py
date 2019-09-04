class Problem(object):

    def __init__(self, tp : TuningProblem):

        self.name = tp.name

        self.IS = tp.input_space
        self.PS = tp.parameter_space
        self.OS = tp.output_space

        self.objective   = tp.objective
        self.constraints = tp.constraints
        self.models      = tp.models

    @property
    def DI(self):

        return self.IS.n_dim

    @property
    def DP(self):

        return self.PS.n_dim)

    @property
    def DO(self):

        return self.OS.n_dim

