import torch
import math
import matplotlib.pyplot as plt
import numpy as np
import Create_P
import Create_b
import GaussMixClass
from logger_config import logger

def solve_Pb(n_nodes, mix_start, mix_product, gamma, noise):
    b_vec = Create_b.b(n_nodes, mix_start, mix_product, gamma, noise)
    # b_vec[0] = b_vec[0] * (-1)
    #b_vec[1] = b_vec[1] * (-1)
    #b_vec[2] = b_vec[2] * (-1)  
    P_mat = Create_P.P(n_nodes, mix_start, mix_product, gamma, noise)
    sol = torch.linalg.solve(P_mat, b_vec.unsqueeze(-1)).squeeze(-1)

    cond_num = torch.linalg.cond(P_mat)
    if cond_num > 1e8:  # High condition number = unstable
        logger.error(f"🔥 P_mat is ill-conditioned! κ(P) = {cond_num:.4e}")
        print("Hi")
    #elif cond_num > 1e4:
    #    logger.warning(f"⚠️ P_mat moderately ill-conditioned: κ(P) = {cond_num:.4e}")
    #    print("Hi")
    symmetry_error = torch.norm(P_mat - P_mat.T)
    if symmetry_error > 1e-6:
        logger.warning(f"⚠️ P_mat not symmetric! Error: {symmetry_error:.4f}")


    logger.debug(f"P is: {P_mat}.")
    logger.debug(f"b is: {b_vec}.")

    n = int(sol.size(0)/3)
    w = torch.zeros(n)
    m = torch.zeros(n)
    s = torch.zeros(n)

    for i in range(0, n):
        w[i] = sol[3*i    ]
        m[i] = sol[3*i + 1]
        s[i] = sol[3*i + 2]

    return GaussMixClass.GaussMix(w, m, s)

def backward_euler(n_nodes, n_steps, mix_start, mix_product, noise, margin):
    dgamma = 1 / n_steps
    gammas = np.linspace(0, 1, n_steps + 1)

    mix = mix_start

    for n in range(n_steps):
        gamma_next = gammas[n+1]
        # Predictor: use forward Euler as initial guess
        rhs_n = solve_Pb(n_nodes, mix, mix_product, gammas[n], noise)
        mix_next = mix + dgamma * rhs_n

        # Newton iteration for implicit correction
        max_iter = 20
        for _ in range(max_iter):
            rhs_next = solve_Pb(n_nodes, mix_next, mix_product, gamma_next, noise)
            F = mix_next + (-1)*mix + (-1)*dgamma * rhs_next  # residual
            if np.linalg.norm(F.w) < 1e-8 and np.linalg.norm(F.m) < 1e-8 and np.linalg.norm(F.s) < 1e-8:
                break
            # Simple fixed-point iteration (or approximate Newton)
            mix_next = mix_next + (-1)*F  # can replace by smarter update if you can compute Jacobian

        mix = mix_next

    return mix

def rk4_Pb(n_nodes, n_steps, mix_start, mix_product, noise, margin):
    h = 1 / n_steps
        
    gammas = torch.linspace(0, 1, n_steps+1)
    mix = mix_start

    for i in range(n_steps):
        logger.debug(f"Loop number {i}:")
        g = gammas[i]
        k1 = solve_Pb(n_nodes, mix,              mix_product, g,       noise) * 1
        logger.debug(f"k1 is w={k1.w}, m={k1.m}, s={k1.s}.")
        # quit()
        k2 = solve_Pb(n_nodes, mix + (h/2) * k1, mix_product, g + h/2, noise) * 1
        logger.debug(f"k2 is w={k2.w}, m={k2.m}, s={k2.s}.")
        k3 = solve_Pb(n_nodes, mix + (h/2) * k2, mix_product, g + h/2, noise) * 1
        logger.debug(f"k3 is w={k3.w}, m={k3.m}, s={k3.s}.")
        k4 = solve_Pb(n_nodes, mix +  h    * k3, mix_product, g + h,   noise) * 1
        logger.debug(f"k4 is w={k4.w}, m={k4.m}, s={k4.s}.")
            
        change = (h/6)*(k1 + 2*k2 + 2*k3 + k4)

        
        logger.debug(f"Loop number {i}:")
        logger.debug(f"k3 is w={k3.w}, m={k3.m}, s={k3.s}.")
        logger.debug(f"k4 is w={k4.w}, m={k4.m}, s={k4.s}.")

        logger.debug(f"Change is w={change.w}, m={change.m}, s={change.s}.")
        
        mix = mix + change 

        # mix.s = torch.clamp(mix.s, min = 0.01)

        #if in_margin(n_nodes, eta, eta1, eta2, margin) == False:
        #    eta = add_component(n_nodes, eta, eta1, eta2)
        # TODO

    return mix


if __name__ == "__main__":
    """
        w1 = torch.tensor([0.4, 0.2, 0.4])
        m1 = torch.tensor([-3.0, 0.0, 3.0])
        s1 = torch.tensor([1., 1., 1.])

        w2 = torch.tensor([0.25, 0.25, 0.25, 0.25])
        m2 = torch.tensor([-2.0, -1.0, 1.0, 2.0])
        s2 = torch.tensor([1.5, 1.0, 1.0, 1.5])
    """

    w1 = torch.tensor([1.])
    m1 = torch.tensor([1.])
    s1 = torch.tensor([1.])

    w2 = torch.tensor([1.])
    m2 = torch.tensor([-1.])
    s2 = torch.tensor([1.])

    nnodes = 20
    nsteps = 10
    gamma = 1.0
    noise = 0.01
    margin = 0.1

    
    logger.debug(f"Input Gauss mix f_1 is w={w1}, m={m1}, s={s1}.")
    logger.debug(f"Input Gauss mix f_2 is w={w2}, m={m2}, s={s2}.")

    mix_start = GaussMixClass.GaussMix(w2, m2, s2)
    mix_product = GaussMixClass.GaussMixProduct(w1, m1, s1, w2, m2, s2)

    new_mix = rk4_Pb(nnodes, nsteps, mix_start, mix_product, noise, margin)


    logger.debug(f"Output Gauss mix is w={new_mix.w}, m={new_mix.m}, s={new_mix.s}.")

    x = torch.linspace(-30, 30, 400)
    f_0 = new_mix.eval(x)
    f_1 = mix_product.mix1.eval(x)
    f_2 = mix_product.mix2.eval(x)
    f_tild = mix_product.eval_tilde(x, gamma, noise)
    f_p = f_1*f_2

    plt.plot(x, f_0, label="approximation")
    plt.plot(x, f_1, label="f_1")
    plt.plot(x, f_2, label="f_2")
    plt.plot(x, f_tild, label="f_tilde")
    plt.plot(x, f_p, label="f_p")
        
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True)
    plt.ylim(-0.1, 0.5)
    plt.show()