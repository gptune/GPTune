#! /usr/bin/env python


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

import os, ctypes
import numpy as np
import GPy
import mpi4py
import itertools
import scipy
import sys
from sys import platform
import time

ROOTDIR = os.path.abspath(__file__ + "/../../build")

if platform == "linux" or platform == "linux2":
    cliblcm = ctypes.cdll.LoadLibrary(ROOTDIR + '/lib_gptuneclcm.so')
elif platform == "darwin":
    cliblcm = ctypes.cdll.LoadLibrary(ROOTDIR + '/lib_gptuneclcm.dylib')
elif platform == "win32":
    raise Exception(f"Windows is not yet supported")





####################################################################################################

class LCM(GPy.kern.Kern):

    """
    LCM kernel:

    .. math::

    """

    def __init__(self, input_dim, num_outputs, Q, name='LCM'):  # self and input_dim are required for GPy

        super(LCM, self).__init__(input_dim + 1, active_dims=None, name=name)

        self.num_outputs = num_outputs
        self.Q = Q

        self.theta =       np.power(10,np.random.randn(Q * input_dim))
        self.var   =       np.power(10,np.random.randn(Q))
        self.kappa =       np.power(10,np.random.randn(Q * num_outputs))
        self.sigma =       np.power(10,np.random.randn(num_outputs))
        self.WS    =       np.power(10,np.random.randn(Q * num_outputs))
        # print('why????',self.theta,self.var,self.kappa,self.sigma,self.WS)

    #     self.theta =  0.54132485 * np.ones(Q * input_dim)
    #     self.var   =  0.54132485 * np.ones(Q)
    #     self.kappa = -0.43275213 * np.ones(Q * num_outputs)
    #     self.sigma =  0.54132485 * np.ones(num_outputs)
    # #        np.random.seed(0)
    #     self.WS    =   .5 * np.random.randn(Q * num_outputs)

        self.BS    = np.empty(Q * self.num_outputs ** 2)

    def get_param_array(self):

        x = np.concatenate([self.theta, self.var, self.kappa, self.sigma, self.WS])

        return x


    def get_correlation_metric(self):
        # self.kappa =  b_{1,1}, ..., b_{delta,1}, ..., b_{1,Q}, ..., b_{\delta,Q}
        # self.sigma = d_1, ..., d_delta
        # self.WS = a_{1,1}, ..., a_{delta,1}, ..., a_{1,Q}, ..., a_{delta,Q}
        kappa = self.kappa
        sigma = self.sigma
        WS = self.WS
        delta = len(sigma)
        Q = int(len(WS)/delta)
        # print('NI = ', delta)
        # print('Q = ', Q)
        B = np.zeros((delta, delta, Q))
        for i in range(Q):
            Wq = WS[i*delta : (i+1)*delta]
            Kappa_q = kappa[i*delta : (i+1)*delta]
            B[:, :, i] = np.outer(Wq, Wq) + np.diag(Kappa_q)
            # print("In model.py, i = ", i)
            # print(B[:, :, i])

        # return C_{i, i'}
        C = np.zeros((delta, delta))
        for i in range(delta):
            for ip in range(i, delta):
                C[i, ip] = np.linalg.norm(B[i, ip, :]) / np.sqrt(np.linalg.norm(B[i, i, :]) * np.linalg.norm(B[ip, ip, :]))
        return C


    def set_param_array(self, x):

        cpt = 0
        for i in range(len(self.theta)):
            self.theta[i] = x[cpt]
            cpt += 1
        for i in range(len(self.var)):
            self.var[i] = x[cpt]
            cpt += 1
        for i in range(len(self.kappa)):
            self.kappa[i] = x[cpt]
            cpt += 1
        for i in range(len(self.sigma)):
            self.sigma[i] = x[cpt]
            cpt += 1
        for i in range(len(self.WS)):
            self.WS[i] = x[cpt]
            cpt += 1

        self.parameters_changed()

        # print(self.theta)
        # print(self.var)
        # print(self.kappa)
        # print(self.sigma)
        # print(self.WS)
        # print(self.BS)

    def parameters_changed(self):

        for q in range(self.Q):

            ws = self.WS[q * self.num_outputs : (q + 1) * self.num_outputs].reshape(1, self.num_outputs)
            a = np.dot(ws.T, ws) + np.diag(self.kappa[q * self.num_outputs : (q + 1) * self.num_outputs])
            self.BS[q * self.num_outputs ** 2 : (q + 1) * self.num_outputs ** 2] = a.flatten()

    def K(self, X1, X2=None):   # Required for GPy, X1 and X2 are ndarray stored in row major

        if X2 is None: X2 = X1
        # print("cao",X1)
        K = np.empty((X1.shape[0], X2.shape[0]))

        try:
            cliblcm.K(ctypes.c_int(self.input_dim - 1),\
                    ctypes.c_int(self.num_outputs),\
                    ctypes.c_int(self.Q),\
                    ctypes.c_int(X1.shape[0]),\
                    ctypes.c_int(X2.shape[0]),\
                    self.theta.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                    self.var.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                    self.BS.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                    X1.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                    X2.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),\
                    K.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        except Exception as inst:
            print(inst)

        # print("cadfdfo",X1)
        return K

    def Kdiag(self, X):   # Required for GPy

        return np.diag(self.K(X, X2=X))

    def update_gradients_full(self, dL_dK, X1, X2=None):

        pass

    def update_gradients_diag(self, dL_dKdiag, X):

        pass

    def gradients_X(self, dL_dK, X1, X2):

        raise("Not implemented")

    def gradients_X_diag(self,dL_dKdiag,X):

        raise("Not implemented")

    def train_kernel(self, X, Y, computer, kwargs):
        npernode = int(computer.cores/kwargs['model_threads'])
        maxtries = kwargs['model_max_jitter_try']
        jitter = kwargs['model_jitter']
        mpi_size=kwargs['model_processes']  # this makes sure every rank belongs to the blacs grid
        nprow = int(np.sqrt(mpi_size))
        npcol = mpi_size // nprow
        mpi_size = nprow * npcol

        t1 = time.time_ns()
        mpi_comm = computer.spawn(__file__, nproc=mpi_size, nthreads=kwargs['model_threads'], npernode=npernode, kwargs = kwargs)
        t2 = time.time_ns()
        if (kwargs['verbose']):
            print('LCM spawn time: ',(t2-t1)/1e9)

        X = np.concatenate([np.concatenate([X[i], np.ones((len(X[i]), 1)) * i], axis=1) for i in range(len(X))])
        Y = np.array(list(itertools.chain.from_iterable(Y)))

        _ = mpi_comm.bcast(("init", (self, X, Y, maxtries,jitter)), root=mpi4py.MPI.ROOT)

        _log_lim_val = np.log(np.finfo(np.float64).max)
        _exp_lim_val = np.finfo(np.float64).max
        _lim_val = 36.0
        epsilon = np.finfo(np.float64).resolution

        def transform_x(x):  # YL: Why is this needed?

            x2 = np.power(10,x.copy())

            # x2[list(range(len(self.theta)+len(self.var)+len(self.kappa)+len(self.sigma),len(x0)))] = np.log(x2[list(range(len(self.theta)+len(self.var)+len(self.kappa)+len(self.sigma),len(x)))])

            # for i in range(len(self.theta) + len(self.var) + len(self.kappa) + len(self.sigma)):
            #     x2[i] = np.where(x[i]>_lim_val, x[i], np.log1p(np.exp(np.clip(x[i], -_log_lim_val, _lim_val)))) #+ epsilon
            #     #x2[i] = np.where(x[i]>_lim_val, x[i], np.log(np.expm1(x[i]))) #+ epsilon

            return x2

        def inverse_transform_x(x):  # YL: Why is this needed?

            x0 = x.copy()
            ws = x0[list(range(len(self.theta)+len(self.var)+len(self.kappa)+len(self.sigma),len(x0)))]
            x2 = np.log10(x0)
            # x2[list(range(len(self.theta)+len(self.var)+len(self.kappa)+len(self.sigma),len(x0)))] = ws
            return x2



        def transform_gradient(x, grad):  # YL: Why is this needed?

            grad2 = grad.copy()
            # x2 = transform_x(x)
            # for i in range(len(self.theta) + len(self.var) + len(self.kappa) + len(self.sigma)):
            #     grad2[i] = grad[i]*np.where(x2[i]>_lim_val, 1., - np.expm1(-x2[i]))

            return grad2

        # Gradient-based optimization

        gradients = np.zeros(len(self.theta) + len(self.var) + len(self.kappa) + len(self.sigma) + len(self.WS))
        iteration = [0] #np.array([0])

        history_xs = [None]
        history_fs = [float('Inf')]

        def fun(x, *args):

            # print(np.power(10,x),'hp')
            t3 = time.time_ns()
            x2 = transform_x(x)
            # x2 = np.insert(x2,len(self.theta), np.ones(len(self.var)))  # fix self.var to 1
            _ = mpi_comm.bcast(("fun_jac", x2), root=mpi4py.MPI.ROOT)
    #            gradients[:] = 0.
            # print("~~~~")
            (neg_log_marginal_likelihood, g) = mpi_comm.recv(source = 0)
            # print("@@@@")
            # print(x2,neg_log_marginal_likelihood)
            #print ("g: ", g)
            #print ("iteration: " + str(iteration[0]))

            iteration[0] += 1

            gradients[:] = g[:]
            if (kwargs['verbose']):
                sys.stdout.flush()
            if (neg_log_marginal_likelihood < min(history_fs)):
                history_xs.append(x2)
                history_fs.append(neg_log_marginal_likelihood)
            t4 = time.time_ns()
            # print('fun_jac py: ',(t4-t3)/1e9)

            return (neg_log_marginal_likelihood)

        def grad(x, *args):
            # x = np.insert(x,len(self.theta), np.ones(len(self.var))) # fix self.var to 1
            grad = - gradients
            grad = transform_gradient(x, grad)

            # grad = np.delete(grad,list(range(len(self.theta),len(self.theta)+len(self.var)))) # fix self.var to 1
            return (grad)

        x0 = self.get_param_array()
        x0_log = inverse_transform_x(x0)

        # x0_log[0]=0
        x0_log[list(range(len(self.theta),len(self.theta)+len(self.var)))]=0
        # x0_log[2]=0
        # x0_log[3]=-10
        # x0_log[4]=-10
        # print(x0_log,'before')
        # sol = scipy.optimize.show_options(method='L-BFGS-B', disp=True, solver='minimize')
        t3 = time.time_ns()

        # bounds = [(-10, 10)] * len(x0_log)
        bounds = [(-10, 8)] * len(self.theta) + [(None, None)] * len(self.var) + [(-10, 8)] * len(self.kappa)+ [(-10, -5)] * len(self.sigma)+ [(-10, 6)] * len(self.WS)
        # print(bounds)

        # sol = scipy.optimize.minimize(fun, x0_log, args=(), method='L-BFGS-B', jac=grad)
        sol = scipy.optimize.minimize(fun, x0_log, args=(), method='L-BFGS-B', jac=grad, bounds=bounds, tol=None, callback=None, options={'disp': None, 'maxcor': 10, 'ftol': 1e-32, 'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 1000, 'maxiter': 1000, 'iprint': -1, 'maxls': 100})

        # print(sol.x,'after')
        # print(transform_x(sol.x),'after exp')  # sol.x is not yet transformed

        t4 = time.time_ns()
        if (kwargs['verbose']):
            print('L-BFGS time: ',(t4-t3)/1e9)

        if (kwargs['verbose']):
            print('fun      : ', sol.fun)
            #print('hess_inv : ', sol.hess_inv)
            #print('jac      : ', jac)
            print('message  : ', sol.message)
            print('nfev     : ', sol.nfev)
            print('nit      : ', sol.nit)
            print('status   : ', sol.status)
            print('success  : ', sol.success)
            #print('x        : ', x)
    #            xopt = transform_x(sol.x)
    #            fopt = sol.fun
        xopt = history_xs[history_fs.index(min(history_fs))] # history_xs is already transformed
        fopt = min(history_fs)
        #print ("gradients: ", str(gradients))
        #print ("iteration: " + str(iteration[0]))

        if(xopt is None):
            raise Exception(f"L-BFGS failed: consider reducing options['model_latent'] !")


    #        # Particle Swarm Optimization
    #
    #        import pyswarms as ps
    #        min_bound = np.array([self.bounds[i][0] for i in range(len(self.bounds))], dtype='float64')
    #        max_bound = np.array([self.bounds[i][1] for i in range(len(self.bounds))], dtype='float64')
    #        bounds = (min_bound, max_bound)
    #        optimizer = ps.single.GlobalBestPSO(n_particles=100, dimensions=len(self.bounds), options={'c1': 0.5, 'c2': 0.3, 'w': 0.9}, bounds=bounds)
    #        fopt, xopt = optimizer.optimize(fun, iters=100)
    #        xopt = transform_x(xopt)
    #
    #        import pyswarm
    #        min_bound = np.array([-20 for i in range(len(self.bounds))], dtype='float64')
    #        max_bound = np.array([ 20 for i in range(len(self.bounds))], dtype='float64')
    #        xopt, fopt = pyswarm.pso(fun, min_bound, max_bound, ieqcons=[], f_ieqcons=None, args=(), kwargs={}, swarmsize=100, omega=0.5, phip=0.5, phig=0.5, maxiter=100, minstep=1e-8, minfunc=1e-8, debug=False)
    #        xopt = transform_x(xopt)

        self.set_param_array(xopt)
        _ = mpi_comm.bcast(("end", None), root=mpi4py.MPI.ROOT)

        mpi_comm.Disconnect()

        return (xopt, fopt, gradients, iteration[0])

if __name__ == "__main__":

    from ctypes import Structure, c_int, c_double, c_void_p, POINTER
    from mpi4py import MPI
    if mpi4py.MPI._sizeof(mpi4py.MPI.Comm) == ctypes.sizeof(ctypes.c_int):
        c_mpi_comm_t = c_int
    else:
        c_mpi_comm_t = c_void_p

    class fun_jac_struct(Structure):
        _fields_ = [("DI", c_int),\
                    ("NT", c_int),\
                    ("NL", c_int),\
                    ("nparam", c_int),\
                    ("m" , c_int),\
                    ("X", POINTER(c_double)),\
                    ("Y", POINTER(c_double)),\
                    ("dists", POINTER(c_double)),\
                    ("exps", POINTER(c_double)),\
                    ("alpha", POINTER(c_double)),\
                    ("K", POINTER(c_double)),\
                    ("gradients_TPS", POINTER(POINTER(c_double))),\
                    ("mb", c_int),\
                    ("lr", c_int),\
                    ("lc", c_int),\
                    ("maxtries", c_int),\
                    ("nprow", c_int),\
                    ("npcol", c_int),\
                    ("pid", c_int),\
                    ("prowid", c_int),\
                    ("pcolid", c_int),\
                    ("context", c_int),\
                    ("Kdesc", POINTER(c_int)),\
                    ("alphadesc", POINTER(c_int)),\
                    ("jitter", c_double),\
                    ("distY", POINTER(c_double)),\
                    ("buffer", POINTER(c_double)),\
                    ("mpi_comm", POINTER(c_mpi_comm_t))]

    mpi_comm = mpi4py.MPI.Comm.Get_parent()
    #    mpi_comm.Merge()

    #    color = self.mpi_rank // (self.mpi_size // num_subgroups)
    #    key   = self.mpi_rank %  (self.mpi_size // num_subgroups)
    #
    mpi_size = mpi_comm.Get_size()
    mpi_rank = mpi_comm.Get_rank()
    nprow = int(np.sqrt(mpi_size))
    npcol = mpi_size // nprow
    #    assert(nprow * npcol == mpi_size)
    mb = 32

    cond = True
    while (cond):

        res = mpi_comm.bcast(None, root=0)
        # if (mpi_rank == 0 ):
        #     print(res)

        if (res[0] == "init"):

            (ker_lcm, X, Y, maxtries,jitter) = res[1]
            mb = min(mb, max(1,min(X.shape[0]//nprow, X.shape[0]//npcol)))   # YL: mb <=32 doesn't seem reasonable, comment this line out ?
            # # print('mb',mb,'nprow',nprow,'npcol',npcol)
            cliblcm.initialize.restype = POINTER(fun_jac_struct)
            z = cliblcm.initialize (\
                    c_int(ker_lcm.input_dim - 1),\
                    c_int(ker_lcm.num_outputs),\
                    c_int(ker_lcm.Q),\
                    c_int(X.shape[0]),\
                    X.ctypes.data_as(POINTER(c_double)),\
                    Y.ctypes.data_as(POINTER(c_double)),\
                    c_int(mb),\
                    c_int(maxtries),\
                    c_double(jitter),\
                    c_int(nprow),\
                    c_int(npcol),\
                    c_mpi_comm_t.from_address(mpi4py.MPI._addressof(mpi4py.MPI.COMM_WORLD)))

        elif (res[0] == "fun_jac"):
            x2 = res[1]
            gradients = np.zeros(len(ker_lcm.theta) + len(ker_lcm.var) + len(ker_lcm.kappa) + len(ker_lcm.sigma) + len(ker_lcm.WS))
            cliblcm.fun_jac.restype = c_double

            # res = mpi_comm.bcast(None, root=mpi4py.MPI.ROOT)
            # print('check',res)

            neg_log_marginal_likelihood = cliblcm.fun_jac ( x2.ctypes.data_as(POINTER(c_double)), z, gradients.ctypes.data_as(POINTER(c_double)) )
            if (mpi_rank == 0):
                mpi_comm.send((neg_log_marginal_likelihood, gradients), dest=0)

        elif (res[0] == "end"):

            cond = False
            cliblcm.finalize(z)
            mpi_comm.Disconnect()

