GPTune Copyright (c) 2019, The Regents of the University of California, through 
Lawrence Berkeley National Laboratory (subject to receipt of any required approvals
from the U.S.Dept. of Energy) and the University of California, Berkeley.
All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative
works, and perform publicly and display publicly, and to permit other to do so.

# GPTune

*GPTune* is an autotuning framework that relies on multitask and transfer learnings to help solve the underlying black-box optimization problem.
GPTune is part of the xSDK effort supported by the Exascale Computing Project (ECP).

## Install

GPTune relies on external Python libraries as listed in the `requirements.txt` file.
They can all be installed through the standard Python repository through the pip tool.

```
pip install --upgrade --user -r requirements.txt
```
Besides the basic requirements, a common autotuning interface has to be installed manually as follows:

```
git clone https://github.com/ytopt-team/autotune.git
cd autotune/
pip install -e .
```

The library can be run either sequentially or in parallel.  In the latter case, the MPI library should be installed.
#XXX MPICC
## Examples

The file `demo.py` in the `examples` folder shows how to describe the autotuning problem and how to invoke GPTune.

```
python examples/demo.py
```

## Usage

### Problem description

#### Spaces

In order to autotune a certain application, three spaces have to be defined through an instance of the **Space** class.
1. Input Space (IS): this space defines the problems that the application targets.
Every point in this space represents one instance of a problem.
In the context of GPTune, the word *task* means application *problem*.
2. Parameter Space (PS): this space defines the application parameters to be tuned.
A point in this space represents a combination of the parameters.
The goal of the tuner is to find the best possible combination that minimizes the objective function of the application.
3. Output Space (OS): this space defines the result(s) of the application, i.e., the objective of the application to be optimized.
For examples, this can be runtime, memory or energy consumption in HPC applications or prediction accuracy in machine learning applications.
The current version of GPTune supports only single dimensional output spaces.
However, future developments intend to support multi-dimensional output spaces, i.e. multi-objective tuning.

#### Parameters

Every dimension of the above mentioned spaces is defined by a **Parameter** object.
Every parameter i defined by its name, type and range or set of values.
Three types of parameters can be defined:
1. Real: defines floating point parameters.
The range of values that the parameter spans should be defined in the *range* argument.
2. Integer: defines integer parameters.
The range of values that the parameter spans should be defined in the *range* argument.
3. Categorical: defines parameters that take their values in a set or list of values.
The list of valid values defining the parameter should be defined in the *values* argument.

**_Note_**
```
If the problems the application targets cannot be defined in a cartesian space, the user can simply give a list of problems (as a Categorical parameter) in the definition of the task space.
```
#### Constraints

Not all points in the task or input spaces correspond to valid problems or parameter configurations.
Constraints might exist that define the validity of a given combination of input parameters and problem description parameters results.
Two ways exist to define constraints in GPTune:
1. Strings: the user can define a Python statement in a string.
The evaluation of that statement should be a boolean.
2. Functions: the user can define a Python function that returns a boolean.  The parameters of the function should have the same name as the parameters defining the problem.
*TODO*: Extra parameters can be passed as a \*\*kwargs argument.

#### Models

The user having additional knowledge about the application can help speed up or improve the result of the tuning process by passing a model(s) of the objective function to be optimized.

These models are defined through Python functions following similarly to the constraints definition.

### GPTune invocation

Once the parameters and spaces (and optionally constraints and models) are defined, an object of the **GPTune** class has to be instantiated.
Then, the different kinds of tuning techniques (*MLA, ...*) can be called through it.

## REFERENCES

