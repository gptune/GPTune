import numpy as np
import ctypes
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.multiclass import unique_labels
sp = ctypes.cdll.LoadLibrary(
    '/project/projectdirs/sparse/younghyun/GPTune/examples/STRUMPACK-KRR-HSS/STRUMPACK/install/lib/libstrumpack.so')


class STRUMPACKKernel(BaseEstimator, ClassifierMixin):

    # kernel can be 'rbf'/'Gauss', 'Laplace' or 'ANOVA'
    def __init__(self, h=1., lam=4.,
                 p=10, vann=64, epow=-2,
                 degree=1, kernel='rbf',
                 approximation='HSS', mpi=False, argv=None):
        self.h = h
        self.lam = lam
        self.p = p
        self.vann = vann
        self.epow = epow
        self.degree = int(degree)
        self.kernel = kernel
        self.approximation = approximation
        self.mpi = mpi
        self.argv = argv

    def __del__(self):
        try:
            sp.STRUMPACK_destroy_kernel_double(self.K_)
        except:
            pass

    def fit(self, X, y):
        if X.dtype != np.float32 and \
           X.dtype != np.float64:
            print(X.dtype)
            raise ValueError("precision", X.dtype, "not supported")

        if self.approximation == 'HODLR' and self.mpi is False:
            raise ValueError("HODLR approximation requires mpi=True")
        ktype = 0
        if self.kernel == 'rbf' or self.kernel == 'Gauss':
            ktype = 0
        elif self.kernel == 'Laplace':
            ktype = 1
        elif self.kernel == 'ANOVA':
            ktype = 2
        else:
            raise ValueError("Kernel type", self.kernel, "not recognized")
        if self.approximation != 'HSS' and \
           self.approximation != 'HODLR':
            raise ValueError("Approximation type not recognized,"
                             "should be 'HSS' or 'HODLR'")

        # check that X and y have correct shape
        X, y = check_X_y(X, y)
        # store the classes seen during fit
        self.classes_ = unique_labels(y)

        if X.dtype == np.float64:
            sp.STRUMPACK_create_kernel_double.restype = \
                ctypes.POINTER(ctypes.c_void_p)
            self.K_ = sp.STRUMPACK_create_kernel_double(
                ctypes.c_int(X.shape[0]), ctypes.c_int(X.shape[1]),
                ctypes.c_void_p(X.ctypes.data),
                ctypes.c_double(self.h), ctypes.c_double(self.lam),
                ctypes.c_int(self.degree), ctypes.c_int(ktype))
        elif X.dtype == np.float32:
            sp.STRUMPACK_create_kernel_float.restype = \
                ctypes.POINTER(ctypes.c_void_p)
            self.K_ = sp.STRUMPACK_create_kernel_float(
                ctypes.c_int(X.shape[0]), ctypes.c_int(X.shape[1]),
                ctypes.c_void_p(X.ctypes.data),
                ctypes.c_float(self.h), ctypes.c_float(self.lam),
                ctypes.c_int(self.degree), ctypes.c_int(ktype))

        if self.argv is None:
            self.argv = []
        LP_c_char = ctypes.POINTER(ctypes.c_char)
        argc = len(self.argv)
        argv = (LP_c_char * (argc + 1))()
        for i, arg in enumerate(self.argv):
            enc_arg = arg.encode('utf-8')
            argv[i] = ctypes.create_string_buffer(enc_arg)

        if self.approximation == 'HSS':
            if self.mpi:
                if X.dtype == np.float64:
                    sp.STRUMPACK_kernel_fit_HSS_MPI_double(
                        self.K_, ctypes.c_void_p(y.ctypes.data),
                        ctypes.c_int(self.p), ctypes.c_int(self.vann), ctypes.c_int(self.epow),
                        ctypes.c_int(argc), argv)
                elif X.dtype == np.float32:
                    sp.STRUMPACK_kernel_fit_HSS_MPI_float(
                        self.K_, ctypes.c_void_p(y.ctypes.data),
                        ctypes.c_int(self.p), ctypes.c_int(self.vann), ctypes.c_int(self.epow),
                        ctypes.c_int(argc), argv)
            else:
                if X.dtype == np.float64:
                    sp.STRUMPACK_kernel_fit_HSS_double(
                        self.K_, ctypes.c_void_p(y.ctypes.data),
                        ctypes.c_int(self.p), ctypes.c_int(self.vann), ctypes.c_int(self.epow),
                        ctypes.c_int(argc), argv)
                elif X.dtype == np.float32:
                    sp.STRUMPACK_kernel_fit_HSS_float(
                        self.K_, ctypes.c_void_p(y.ctypes.data),
                        ctypes.c_int(self.p), ctypes.c_int(self.vann), ctypes.c_int(self.epow),
                        ctypes.c_int(argc), argv)
        elif self.approximation == 'HODLR':
            if X.dtype == np.float64:
                sp.STRUMPACK_kernel_fit_HODLR_MPI_double(
                    self.K_, ctypes.c_void_p(y.ctypes.data),
                    ctypes.c_int(self.p), ctypes.c_int(self.vann), ctypes.c_int(self.epow),
                    ctypes.c_int(argc), argv)
            elif X.dtype == np.float32:
                sp.STRUMPACK_kernel_fit_HODLR_MPI_float(
                    self.K_, ctypes.c_void_p(y.ctypes.data),
                    ctypes.c_int(self.p), ctypes.c_int(self.vann), ctypes.c_int(self.epow),
                    ctypes.c_int(argc), argv)
        # return the classifier
        return self

    def predict(self, X):
        # TODO make sure there are only 2 classes?
        check_is_fitted(self, 'K_')
        prediction = np.zeros((X.shape[0], 1), dtype=X.dtype)
        if X.dtype == np.float64:
            sp.STRUMPACK_kernel_predict_double(
                self.K_, ctypes.c_int(X.shape[0]),
                ctypes.c_void_p(X.ctypes.data),
                ctypes.c_void_p(prediction.ctypes.data))
        elif X.dtype == np.float32:
            sp.STRUMPACK_kernel_predict_float(
                self.K_, ctypes.c_int(X.shape[0]),
                ctypes.c_void_p(X.ctypes.data),
                ctypes.c_void_p(prediction.ctypes.data))
        return [self.classes_[0] if prediction[i] < 0.0 else self.classes_[1]
                for i in range(X.shape[0])]

    def decision_function(self, X):
        check_is_fitted(self, 'K_')
        prediction = np.zeros((X.shape[0], 1), dtype=X.dtype)
        if X.dtype == np.float64:
            sp.STRUMPACK_kernel_predict_double(
                self.K_, ctypes.c_int(X.shape[0]),
                ctypes.c_void_p(X.ctypes.data),
                ctypes.c_void_p(prediction.ctypes.data))
        elif X.dtype == np.float32:
            sp.STRUMPACK_kernel_predict_float(
                self.K_, ctypes.c_int(X.shape[0]),
                ctypes.c_void_p(X.ctypes.data),
                ctypes.c_void_p(prediction.ctypes.data))
        return prediction
