import tsfc
import tsfc.fiatinterface
import dijitso
import ffc.compiler
from coffee.plan import ASTKernel

import numpy
import numba
from petsc4py import PETSc

import ctypes
import hashlib


def tsfc_compile_wrapper(form, extra_parameters=None, **kwargs):
    """Compiles form with TSFC and returns source code."""

    parameters = {'mode': 'spectral'}
    parameters.update({} if extra_parameters is None else extra_parameters)

    kernel, = tsfc.compile_form(form, parameters=parameters)

    k = ASTKernel(kernel.ast)
    k.plan_cpu(dict(optlevel='Ov'))

    code = kernel.ast.gencode()

    code = code.replace('static inline', '')
    code = "#include <math.h>\n\n" + code

    return code


def ffc_compile_wrapper(form, extra_parameters=None, cross_element_width=0, gcc_vec_ext=False, **kwargs):
    """Compiles form with FFC and returns source code."""

    parameters = ffc.parameters.default_parameters()
    parameters["cross_element_width"] = cross_element_width
    parameters["enable_cross_element_gcc_ext"] = gcc_vec_ext

    parameters.update({} if extra_parameters is None else extra_parameters)

    # Call FFC
    code_h, code_c = ffc.compiler.compile_form(form, parameters=parameters)

    prefix = "form"
    form_index = 0

    # Extract tabulate_tensor definition
    function_name = "tabulate_tensor_{}_cell_integral_{}_otherwise".format(prefix, form_index)
    index_start = code_c.index("void {}(".format(function_name))
    index_end = code_c.index("ufc_cell_integral* create_{}_cell_integral_{}_otherwise(void)".format(prefix, form_index),
                             index_start)
    tabulate_tensor_code = code_c[index_start:index_end].strip()

    # Extract tabulate_tensor body
    body_start = tabulate_tensor_code.index("{")
    tabulate_tensor_body = tabulate_tensor_code[body_start:].strip()

    tabulate_tensor_signature = "void form_cell_integral_otherwise (double* restrict A, const double *restrict coordinate_dofs)"

    preamble = [
        "#include <math.h>",
        "#include <stdalign.h>\n"
    ]

    if gcc_vec_ext:
        tabulate_tensor_signature = tabulate_tensor_signature.replace("double", "double4")
        preamble.append("typedef double double4 __attribute__ ((vector_size (32)));\n")

    code = "\n".join(preamble + [
        tabulate_tensor_signature,
        tabulate_tensor_body
    ])

    return code


def compile_form(a, form_compiler=None, form_compiler_parameters=None, **kwargs):
    """Compiles form with the specified compiler and returns a ctypes function ptr."""

    # Use tsfc as default form compiler
    form_compiler = "tsfc" if form_compiler is None else form_compiler

    form_compilers = {
        "ffc": lambda form: ffc_compile_wrapper(form, form_compiler_parameters, **kwargs),
        "tsfc": lambda form: tsfc_compile_wrapper(form, form_compiler_parameters, **kwargs)
    }

    run_form_compiler = form_compilers[form_compiler]

    # Define generation function executed on cache miss
    def generate(form, name, signature, params):
        code = run_form_compiler(form)
        return None, code, ()

    # Compute unique name
    name = "mfc_{}_{}_{}".format(form_compiler, str(form_compiler_parameters), a.signature())
    hashed_name = hashlib.sha1(name.encode()).hexdigest()

    # Set dijitso into C mode
    params = {
         'build': {
             'cxx': 'cc',
            'cxxflags': (
                '-O2', '-Wall', '-shared', '-fPIC', '-std=c11',
                '-march=skylake', '-mtune=skylake', '-ftree-vectorize', '-funroll-loops'
            ),
         },
         'cache': {'src_postfix': '.c'},
    }

    # Do JIT compilation
    module, name = dijitso.jit(a, hashed_name, params, generate=generate)

    # Grab assembly kernel from ctypes module and set its arguments
    func = getattr(module, 'form_cell_integral_otherwise')
    func.argtypes = (ctypes.c_void_p, ctypes.c_void_p)

    return func


# Get C MatSetValues function from PETSc because can't call
# petsc4py.PETSc.Mat.setValues() with numba.jit(nopython=True)
petsc = ctypes.CDLL('libpetsc.so')
MatSetValues = petsc.MatSetValues
MatSetValues.argtypes = 7 * (ctypes.c_void_p,)
ADD_VALUES = PETSc.InsertMode.ADD_VALUES
del petsc


def assemble(petsc_tensor, dofmap, form, **kwargs):
    assert len(form.arguments()) == 2, "Now only bilinear forms"

    # JIT compile UFL form into ctypes function
    assembly_kernel = compile_form(form, **kwargs)

    # Fetch data
    tdim = dofmap.mesh.reference_cell.get_dimension()
    cells = dofmap.mesh.get_connectivity(tdim, 0)
    vertices = dofmap.mesh.vertices
    cell_dofs = dofmap.cell_dofs
    mat = petsc_tensor.handle

    # Prepare cell tensor temporary
    elements = tuple(arg.ufl_element() for arg in form.arguments())
    fiat_elements = map(tsfc.fiatinterface.create_element, elements)
    element_dims = tuple(fe.space_dimension() for fe in fiat_elements)
    _A = numpy.zeros(element_dims, dtype=numpy.double)

    # Prepare coordinates temporary
    num_vertices_per_cell = cells.shape[1]
    gdim = vertices.shape[1]
    _coords = numpy.zeros((num_vertices_per_cell, gdim), dtype=numpy.double)

    @numba.jit(nopython=True)
    def _assemble(assembly_kernel, cells, vertices, cell_dofs, mat, _coords, _A):
        coords_ptr = _coords.ctypes.data
        A_ptr = _A.ctypes.data
        nrows = ncols = cell_dofs.shape[1]

        # Loop over cells
        for i in range(cells.shape[0]):
            # Update temporaries
            _coords[:] = vertices[cells[i]]
            _A[:] = 0

            # Assemble cell tensor
            assembly_kernel(A_ptr, coords_ptr)

            # Add to global tensor
            rows = cols = cell_dofs[i].ctypes.data
            ierr = MatSetValues(mat, nrows, rows, ncols, cols, A_ptr, ADD_VALUES)
            assert ierr == 0

    # Call jitted hot loop
    _assemble(assembly_kernel, cells, vertices, cell_dofs, mat, _coords, _A)

    petsc_tensor.assemble()


def empty_aligned(shape, dtype, align):
    """
    Allocate numpy array with a specified memory alignment.

    :param shape: Shape of the requested array.
    :param dtype: Data-type of the requested array.
    :param align: Memory alignment requirement of the array in bytes.
    :return: Array with the specified properties.
    """

    # Aligned memory is required for AVX2 but not supported natively by NumPy
    # See: https://github.com/numpy/numpy/issues/5312

    # Calculate the required amount of bytes for the array
    n = numpy.dtype(dtype).itemsize * numpy.prod(shape)

    # Allocate array with possible offset for alignment
    a = numpy.empty(n + (align - 1), dtype=numpy.uint8)
    data_align = a.ctypes.data % align
    offset = 0 if data_align == 0 else (align - data_align)
    a_aligned = a[offset: offset + n]

    # View with requested data-type
    a_casted = a_aligned.view(dtype=dtype)
    # Assign new shape (guarantees no copy)
    a_casted.shape = shape

    return a_casted


def assemble_vectorized(petsc_tensor, dofmap, form, **kwargs):
    assert len(form.arguments()) == 2, "Now only bilinear forms"

    # Size of cross-element batch
    vec_width = 4

    # JIT compile UFL form into ctypes function
    assembly_kernel = compile_form(form,
                                   form_compiler="ffc",
                                   cross_element_width=vec_width,
                                   gcc_vec_ext=True)

    # Fetch data
    tdim = dofmap.mesh.reference_cell.get_dimension()
    cells = dofmap.mesh.get_connectivity(tdim, 0)
    vertices = dofmap.mesh.vertices
    cell_dofs = dofmap.cell_dofs
    mat = petsc_tensor.handle

    # Prepare cell tensor temporary
    elements = tuple(arg.ufl_element() for arg in form.arguments())
    fiat_elements = map(tsfc.fiatinterface.create_element, elements)
    element_dims = tuple(fe.space_dimension() for fe in fiat_elements)

    # Aligned storage for strided element tensor
    _A = empty_aligned((*element_dims, vec_width), dtype=numpy.float64, align=32)
    # Aligned storage for transposed element tensor
    _A_t = empty_aligned((vec_width, *element_dims), dtype=numpy.float64, align=32)

    # Prepare coordinates temporary
    num_vertices_per_cell = cells.shape[1]
    gdim = vertices.shape[1]

    # Aligned storage for strided coordinates array
    _coords = empty_aligned((num_vertices_per_cell, gdim, vec_width), dtype=numpy.float64, align=32)
    # Aligned storage for transposed coordinates array
    _coords_t = empty_aligned((vec_width, num_vertices_per_cell, gdim), dtype=numpy.float64, align=32)

    # Tuple that can be used to transpose cross-element dimension from inner-most to outer-most
    identity_transpose = tuple(range(len(element_dims) + 1))
    unstride_A = (identity_transpose[-1], *identity_transpose[:-1])

    @numba.jit(nopython=True)
    def _assemble(assembly_kernel, cells, vertices, cell_dofs, mat, _coords, _A, _coords_t, _A_t):
        A_ptr = _A.ctypes.data
        coords_ptr = _coords.ctypes.data
        nrows = ncols = cell_dofs.shape[1]

        # Currently no residual loop implemented
        assert cells.shape[0] % vec_width == 0, "Number of cells not divisible by vectorization width"

        # Loop over cells
        for i in range(0, cells.shape[0], vec_width):
            # Collect vertex coordinates for each element
            for j in range(vec_width):
                _coords_t[j, :] = vertices[cells[i + j]]

            # Make coordinates strided
            _coords[:] = numpy.transpose(_coords_t, (1, 2, 0))

            # Assemble cell tensor
            assembly_kernel(A_ptr, coords_ptr)

            # "Unstride" element matrix
            _A_t[:] = numpy.transpose(_A, unstride_A)

            # Add to global tensor
            for j in range(vec_width):
                rows = cols = cell_dofs[i + j].ctypes.data
                ierr = MatSetValues(mat, nrows, rows, ncols, cols, _A_t[j, :].ctypes.data, ADD_VALUES)
                assert ierr == 0, "MatSetValues error!"

    # Call jitted hot loop
    _assemble(assembly_kernel, cells, vertices, cell_dofs, mat, _coords, _A, _coords_t, _A_t)

    petsc_tensor.assemble()
