#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 14:56:03 2021
Pitting corrosion in FEniCs
@author: Sergio Lucarini
"""

import os
import math
import fenics as fn
import numpy as np

save_dir = './results-1d-active'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


# Length and discretization of mesh
L = [-50e-6, 50e-6]
n = 100
# Create 1D mesh with n0 nodes
mesh = fn.IntervalMesh(n, L[0], L[1])
# export mesh point
# mesh_points = mesh.coordinates()
# print(mesh_points)
# np.save('mesh_points', mesh_points)
# mesh = fn.RectangleMesh(fn.Point(0, 0), fn.Point(
#     L[0], L[1]), n[0], n[1], "crossed")

# Define Space: Lagrange 1-linear 2-quadratic elements
Fun = fn.FiniteElement('CG', mesh.ufl_cell(), 1)
pcFun = fn.FunctionSpace(mesh, Fun * Fun)

mesh_points = pcFun.tabulate_dof_coordinates()[::2]
print(mesh_points.shape)
np.save(f'{save_dir}/mesh_points.npy', mesh_points)


# solution variables, funtion and test functions
pc_sol = fn.Function(pcFun)
p_sol, c_sol = fn.split(pc_sol)
tpc = fn.TrialFunction(pcFun)
tp, tc = fn.split(tpc)
dpc = fn.TestFunction(pcFun)
dp, dc = fn.split(dpc)
# auxiliary variables for saving previous solutions
pc_t = fn.Function(pcFun)
p_t, c_t = fn.split(pc_t)

# Introduce manually the material parameters
alphap = 1.03e-4
omegap = 1.76e7
DD = 8.5e-10
AA = 5.35e7
Lp = 1e-11 # activation control
# Lp = 2.0 # diffusion control
cse = 1.
cle = 5100/1.43e5

# Boundary conditions


# def ConcentrationBoundary(x):
#     return fn.near(x[1], 140.0e-6) and x[0] <= 100.0e-6+5e-6 and x[0] >= 100.0e-6-5e-6

def ConcentrationBoundary_left(x):
    return fn.near(x[0], L[0])


def ConcentrationBoundary_right(x):
    return fn.near(x[0], L[1])


bc_pc = [fn.DirichletBC(pcFun.sub(0), fn.Constant(1.0), ConcentrationBoundary_left),
         fn.DirichletBC(pcFun.sub(1), fn.Constant(
             1.0), ConcentrationBoundary_left),
         fn.DirichletBC(pcFun.sub(0), fn.Constant(
             0.0), ConcentrationBoundary_right),
         fn.DirichletBC(pcFun.sub(1), fn.Constant(0.0), ConcentrationBoundary_right)]

# Initial conditions as expression


class InitialConditions(fn.UserExpression):
    def eval_cell(self, values, x, ufc_cell):
        values[0] = (1 - math.tanh(math.sqrt(omegap) /
                                   math.sqrt(2 * alphap) * x[0])) / 2
        hp0 = -2*values[0]**3 + 3*values[0]**2
        values[1] = hp0*cse + (1-hp0)*0 # create a gradient in the initial concentration field

    def value_shape(self):
        return (2,)


# Apply initial conditions
pc_sol.interpolate(InitialConditions())
pc_t.interpolate(InitialConditions())

# Weak Form # Constituive functions
dx = fn.dx()  # define the volume integral
DT = fn.Expression("dtime", dtime=0, degree=0)  # difine time incrementation
E_pc = (c_sol-c_t)/DT*dc*dx +\
    -fn.inner(-DD*fn.grad(c_sol)+DD*(cse-cle)*fn.grad(-2*p_sol**3 + 3*p_sol**2), fn.grad(dc))*dx +\
    (p_sol-p_t)/DT/Lp*dp*dx +\
    -2*AA*(c_sol-(-2*p_sol**3 + 3*p_sol**2)*(cse-cle)-cle)*(cse-cle)*(-6*p_sol**2 + 6*p_sol)*dp*dx +\
    omegap*(4*p_sol**3-6*p_sol**2+2*p_sol)*dp*dx +\
    fn.inner(alphap*fn.grad(p_sol), fn.grad(dp))*dx

# Automatically calculate jacobian
Jpc = fn.derivative(E_pc, pc_sol, tpc)
# Define the non linear problem and solver parameters
p_pc = fn.NonlinearVariationalProblem(E_pc, pc_sol, bc_pc, J=Jpc)
solver_pc = fn.NonlinearVariationalSolver(p_pc)
solver_pc.parameters['newton_solver']['absolute_tolerance'] = 1E-8
solver_pc.parameters['newton_solver']['linear_solver'] = 'mumps'
solver_pc.parameters['newton_solver']["convergence_criterion"] = "incremental"
solver_pc.parameters['newton_solver']["relative_tolerance"] = 1e-6
solver_pc.parameters['newton_solver']["maximum_iterations"] = 10

# Initialization of the time incremental procedure and output requests
t = 0
DeltaT = 1000
DT.dtime = 1.*DeltaT
conc_f = fn.File(f"{save_dir}/fields.pvd")
conc_f << pc_sol

np.save(f"./{save_dir}/sol-{t:.2f}",
        pc_sol.vector().get_local().reshape(-1, 2))
# Time incremental loop
while t <= 100000:
    # Update time

    DT.dtime = DeltaT
    print("Time: ", t)

    # Newton solver with updated time increment
    try:
        info = solver_pc.solve()
    # If not solved, reduce time increment and start again or stop
    except:
        print("Solver not converged")
        t -= DeltaT
        DeltaT /= 2.
        print('red dt', DeltaT)
        if DeltaT < 1e-4:
            break
        continue

    # save ouput
    print('Iterations:', info[0], ', Total time:', t)
    conc_f << pc_sol

    # save state in t+dt for next increment
    pc_t.vector()[:] = pc_sol.vector()
    pc_val = pc_sol.vector().get_local().reshape(-1, 2)

    t += DeltaT
    np.save(f"./{save_dir}/sol-{t:.2f}", pc_val)


print('Simu complete')
