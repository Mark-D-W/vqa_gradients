import numpy as np
from scipy.optimize import minimize
from scipy.optimize import basinhopping
import numdifftools as nd


class QAD():
    def __init__(self, tol=1e-6, num=20, optimiser=basinhopping, optimiser_args={}, logfile="qad_log.txt"):
        self.tol = tol
        self.num = num
        self.optimiser = optimiser
        self.optimiser_args = optimiser_args
        self.logfile = logfile


    def __call__(self, E, par):
        self.E = E
        num_models = 0
        params = par
        prev_min = E(params)
        min_found = prev_min
        min_params = params
        min_optim = None
        break_reason = ""
        while True:
            self.get_model_data(params)
            num_models += 1

            optim_res = self.optimiser(self.model, np.zeros_like(params), **self.optimiser_args)
            params = params + optim_res["x"]
            minima = self.E(params)
            if min_optim==None:
                min_found = minima
                min_params = params
                min_optim = optim_res
            elif minima <= min_found:
                min_found = minima
                min_params = params
                min_optim = optim_res

            tol = np.abs(prev_min-self.E(params))
            self.__log(f"Iteration={num_models}, tolerence={tol}, params={params}, min_found={min_found}")
            if tol < self.tol:
                break_reason = "tol"
                break
            elif num_models > self.num:
                break_reason = "num"
                params, optim_res = min_params, min_optim
                break
            prev_min = self.E(params)

        optim_res["x"] = params
        optim_res["fun"] = self.E(params)
        if self.optimiser == basinhopping:
            optim_res.success = optim_res.lowest_optimization_result.success
        self.__log(f"Final optimisation results: min={optim_res.fun}")
        return optim_res



    def __log(self, msg):
        with open(self.logfile, "a") as infile:
            infile.write(f"{msg}\n")



    def get_model_data(self, params):
        jacobian = nd.Jacobian(self.E)
        hessian = nd.Hessian(self.E)

        self.E_A = self.E(params)
        self.E_B = jacobian(params)
        self.E_C = np.diag(hessian(params)) + self.E_A / 2
        self.E_D = np.triu(hessian(params), 1)


    def model(self, params):
        A = np.prod(np.cos(0.5 * params) ** 2)
        B_over_A = 2 * np.tan(0.5 * params)
        C_over_A = B_over_A ** 2 / 2
        D_over_A = np.outer(B_over_A, B_over_A)

        all_terms_over_A = [
            self.E_A,
            np.dot(self.E_B, B_over_A),
            np.dot(self.E_C, C_over_A),
            np.dot(B_over_A, self.E_D @ B_over_A),
        ]

        cost = A * np.sum(all_terms_over_A)

        return cost




    def gen_E_approx(self, params):
        m = len(params)
        k_range = np.int64(np.linspace(0,m-1,m))

        E_A = self.E(params)
        E_B = self.jacobian(params)
        E_C = np.diag(self.hessian(params)) + E_A / 2
        E_D = self.hessian(params)

        A = lambda x: np.prod( np.cos(x[k_range]/2)**2 )
        B = lambda x: 2*np.tan(x/2) * A(x)
        C = lambda x: 2*np.tan(x/2)**2 * A(x)
        D = lambda x: 4*np.outer(np.tan(x/2), np.tan(x/2)) * A(x)

        if self.DEBUG:
            print(f"""
            E_A={E_A}
            E_B={E_B}
            E_C={E_C}
            E_D={E_D}
            A={A(params)}
            B={B(params)}
            C={C(params)}
            D={D(params)}
            """)

        E_approx = lambda theta: (
                A(theta) * E_A +
                np.sum( B(theta) * E_B + C(theta) * E_C ) +
                np.sum( np.triu(D(theta)*E_D, 1) )
            )

        return E_approx
