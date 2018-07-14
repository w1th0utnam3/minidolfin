from ufl import *
from petsc4py import PETSc

import timeit

from minidolfin.meshing import build_unit_square_mesh
from minidolfin.dofmap import build_dofmap
from minidolfin.dofmap import build_sparsity_pattern
from minidolfin.dofmap import pattern_to_csr
from minidolfin.petsc import create_matrix_from_csr
from minidolfin.assembling import assemble

element = FiniteElement("P", triangle, 3)
u, v = TrialFunction(element), TestFunction(element)
c = Constant(element.cell())
lmbda = Coefficient(FiniteElement("P", triangle, 1))
#a = c*inner(lmbda*grad(u), grad(v))*dx
a = inner(grad(u), grad(v))*dx

# Build mesh
mesh = build_unit_square_mesh(2, 2)
tdim = mesh.reference_cell.get_dimension()
print('Number cells: {}'.format(mesh.num_entities(tdim)))

# Build dofmap
dofmap = build_dofmap(element, mesh)
print('Number dofs: {}'.format(dofmap.dim))

def test_assembly(compiler):
    # Build sparsity pattern and create matrix
    pattern = build_sparsity_pattern(dofmap)
    i, j = pattern_to_csr(pattern)
    A = create_matrix_from_csr((i, j))

    # Run and time assembly
    t = -timeit.default_timer()
    assemble(A, dofmap, a, form_compiler_parameters={"compiler": compiler})
    t += timeit.default_timer()
    print('Assembly time a: {}'.format(t))

    norm = A.norm(PETSc.NormType.NORM_FROBENIUS)
    print("A norm: {:.6e}".format(norm))

    return norm

norm_ffc = test_assembly("ffc")
norm_tsfc = test_assembly("tsfc")

print("")
print("Difference: {:.6e}".format(abs(norm_ffc-norm_tsfc)))
