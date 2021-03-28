from autotune.space import Space
from typing import Callable, List

# Custom types

class TuningProblem:
    """

    Args:
        input_space (Space): the input space represents the set of problems you want to solve.
        parameter_space (Space): the parameter space represents a space of parameters you want to optimize for a particular problem. Given a choice in the input space you want to find the optimal set of parameters in the parameter space.
        objective (Callable): a function wich returns a single scalar or a tuple of scalar values.
        constraints (str, Callable, optional): if str then it is ... if a Callable then it is .... Defaults to None.
        models (...): analytical models. Defaults to None.
        name (str, optional): A name corresponding to the TuningProblem. Defaults to None.
        constants (Dict): A dictionary defining global constants. Defaults to None. 

    >>> from autotune import TuningProblem
    >>> from autotune.space import *
    >>> input_space = Space([
    ...     Categorical(["boyd1.mtx"], name="matrix")
    ... ])
    >>> parameter_space = Space([
    ...     Integer(10, 100, name="m"),
    ...     Integer(10, 100, name="n")
    ... ])
    >>> output_space = Space([
    ...     Real(0.0, inf, name="time")
    ... ])
    >>> def myobj(point):
    ...     return point['m']
    >>> cst = "m > n & m-n > 10"
    >>> csts = {'cst1': cst}
    >>> def model1(point):
    ...     from numpy import log
    ...     return log(point['m']) + log(point['n'] + point['m']*point['n'])
    >>> models = {'model1': model1}
    >>> constants = {'const1': const1}
    >>> problem = TuningProblem(input_space, parameter_space, output_space, myobj, csts, models, constants)
    """

    def __init__(self,
        input_space: Space,
        parameter_space: Space,
        output_space: Space,
        objective: Callable,
        constraints=None,
        models=None,
        name=None,
        constants=None,
        **kwargs):

        self.name = name
        self.input_space = input_space
        self.parameter_space = parameter_space
        self.output_space = output_space
        self.objective = objective
        self.constraints = constraints
        self.models = models
        self.constants = constants
