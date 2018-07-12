import ufl
import numpy.linalg
from petsc4py import PETSc
#from matplotlib import pyplot, tri

import timeit
import math

from minidolfin.meshing import build_unit_square_mesh
from minidolfin.dofmap import build_dofmap
from minidolfin.dofmap import build_sparsity_pattern
from minidolfin.dofmap import pattern_to_csr
from minidolfin.dofmap import interpolate_vertex_values
from minidolfin.petsc import create_matrix_from_csr
from minidolfin.assembling import assemble, assemble_vectorized
from minidolfin.bcs import build_dirichlet_dofs
from minidolfin.petsc import set_solver_package


# Plane wave
omega2 = 1.5**2 + 1**2
u_exact = lambda x: math.cos(-1.5*x[0] + x[1])

# UFL form
element = ufl.FiniteElement("P", ufl.triangle, 3)
u, v = ufl.TrialFunction(element), ufl.TestFunction(element)
a = (ufl.inner(ufl.grad(u), ufl.grad(v)) - omega2*ufl.dot(u, v))*ufl.dx

# Build mesh
mesh = build_unit_square_mesh(2, 2)
tdim = mesh.reference_cell.get_dimension()
print('Number cells: {}'.format(mesh.num_entities(tdim)))

# Build dofmap
dofmap = build_dofmap(element, mesh)
print('Number dofs: {}'.format(dofmap.dim))

def assemble_and_solve(assemble_fun):
    # Build sparsity pattern and create matrix
    pattern = build_sparsity_pattern(dofmap)
    i, j = pattern_to_csr(pattern)
    A = create_matrix_from_csr((i, j))

    # Run and time assembly
    t_ass = -timeit.default_timer()
    assemble_fun(A, dofmap, a, form_compiler="ffc")
    t_ass += timeit.default_timer()
    print('Assembly time a: {}'.format(t_ass))

    # Prepare solution and rhs vectors and apply boundary conditions
    x, b = A.createVecs()
    bc_dofs, bc_vals = build_dirichlet_dofs(dofmap, u_exact)
    x.setValues(bc_dofs, bc_vals)
    A.zeroRowsColumns(bc_dofs, diag=1, x=x, b=b)

    # Solve linear system
    ksp = PETSc.KSP().create(A.comm)
    ksp.setType(PETSc.KSP.Type.PREONLY)
    ksp.pc.setType(PETSc.PC.Type.CHOLESKY)
    set_solver_package(ksp.pc, "mumps")
    #A.setOption(PETSc.Mat.Option.SPD, True)  # FIXME: Is that true?
    t = -timeit.default_timer()
    ksp.setOperators(A)
    ksp.setUp()
    t += timeit.default_timer()
    print('Setup linear solver time: {}'.format(t))
    t = -timeit.default_timer()
    ksp.solve(b, x)
    t += timeit.default_timer()
    print('Solve linear system time: {}'.format(t))


    # Plot solution
    vertex_values = interpolate_vertex_values(dofmap, x)

    return t_ass, vertex_values

t_ass_ref, vertex_values_ref = assemble_and_solve(assemble)
t_ass_vec, vertex_values_vec = assemble_and_solve(assemble_vectorized)

norm_ref = numpy.linalg.norm(vertex_values_ref)
norm_vec = numpy.linalg.norm(vertex_values_vec)
error = numpy.abs(norm_ref-norm_vec)

print("")
print("norm_ref: {:.6f}, norm_vec: {:.6f}, err: {:.6f}".format(norm_ref,
                                                               norm_vec,
                                                               error))
print("assembly t_ref: {:.2f}ms, t_vec: {:.2f}ms, speedup: {:.3f}x".format(t_ass_ref*1000,
                                                                           t_ass_vec*1000,
                                                                           t_ass_ref/t_ass_vec))
