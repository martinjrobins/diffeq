use core::panic;
use std::cell::RefCell;
use std::ops::MulAssign;

use diffsl::{execution::module::CodegenModule, Compiler};

use crate::{
    error::DiffsolError, find_jacobian_non_zeros, find_matrix_non_zeros,
    jacobian::JacobianColoring, matrix::sparsity::MatrixSparsity,
    op::nonlinear_op::NonLinearOpJacobian, ConstantOp, ConstantOpSens, ConstantOpSensAdjoint,
    LinearOp, LinearOpTranspose, Matrix, NonLinearOp, NonLinearOpAdjoint, NonLinearOpSens,
    NonLinearOpSensAdjoint, OdeEquations, OdeEquationsRef, Op, Scale, Vector,
};

pub type T = f64;

/// Context for the ODE equations specified using the [DiffSL language](https://martinjrobins.github.io/diffsl/).
///
/// This contains the compiled code and the data structures needed to evaluate the ODE equations.
pub struct DiffSlContext<M: Matrix<T = T>, CG: CodegenModule> {
    compiler: Compiler<CG>,
    data: RefCell<Vec<M::T>>,
    ddata: RefCell<Vec<M::T>>,
    sens_data: RefCell<Vec<M::T>>,
    tmp: RefCell<M::V>,
    tmp2: RefCell<M::V>,
    tmp_out: RefCell<M::V>,
    tmp2_out: RefCell<M::V>,
    nstates: usize,
    nroots: usize,
    nparams: usize,
    has_mass: bool,
    has_root: bool,
    has_out: bool,
    nout: usize,
    nthreads: usize,
}

impl<M: Matrix<T = T>, CG: CodegenModule> DiffSlContext<M, CG> {
    /// Create a new context for the ODE equations specified using the [DiffSL language](https://martinjrobins.github.io/diffsl/).
    /// The input parameters are not initialized and must be set using the [OdeEquations::set_params] function before solving the ODE.
    ///
    /// # Arguments
    ///
    /// * `text` - The text of the ODE equations in the DiffSL language.
    /// * `nthreads` - The number of threads to use for code generation (0 for automatic, 1 for single-threaded).
    ///
    pub fn new(text: &str, nthreads: usize) -> Result<Self, DiffsolError> {
        let mode = match nthreads {
            0 => diffsl::execution::compiler::CompilerMode::MultiThreaded(None),
            1 => diffsl::execution::compiler::CompilerMode::SingleThreaded,
            _ => diffsl::execution::compiler::CompilerMode::MultiThreaded(Some(nthreads)),
        };
        let compiler = Compiler::from_discrete_str(text, mode)
            .map_err(|e| DiffsolError::Other(e.to_string()))?;
        let (nstates, _nparams, _nout, _ndata, _nroots, _has_mass) = compiler.get_dims();

        let compiler = if nthreads == 0 {
            let num_cpus = std::thread::available_parallelism().unwrap().get();
            let nthreads = num_cpus.min(nstates / 1000).max(1);
            Compiler::from_discrete_str(
                text,
                diffsl::execution::compiler::CompilerMode::MultiThreaded(Some(nthreads)),
            )
            .map_err(|e| DiffsolError::Other(e.to_string()))?
        } else {
            compiler
        };

        let (nstates, nparams, nout, _ndata, nroots, has_mass) = compiler.get_dims();

        let has_root = nroots > 0;
        let has_out = nout > 0;
        let data = RefCell::new(compiler.get_new_data());
        let ddata = RefCell::new(compiler.get_new_data());
        let sens_data = RefCell::new(compiler.get_new_data());
        let tmp = RefCell::new(M::V::zeros(nstates));
        let tmp2 = RefCell::new(M::V::zeros(nstates));
        let tmp_out = RefCell::new(M::V::zeros(nout));
        let tmp2_out = RefCell::new(M::V::zeros(nout));

        Ok(Self {
            compiler,
            data,
            ddata,
            sens_data,
            nparams,
            nstates,
            tmp,
            tmp2,
            tmp_out,
            tmp2_out,
            nroots,
            nout,
            has_mass,
            has_root,
            has_out,
            nthreads,
        })
    }

    pub fn recompile(&mut self, text: &str) -> Result<(), DiffsolError> {
        let mode = match self.nthreads {
            0 => diffsl::execution::compiler::CompilerMode::MultiThreaded(None),
            1 => diffsl::execution::compiler::CompilerMode::SingleThreaded,
            _ => diffsl::execution::compiler::CompilerMode::MultiThreaded(Some(self.nthreads)),
        };
        self.compiler = Compiler::from_discrete_str(text, mode)
            .map_err(|e| DiffsolError::Other(e.to_string()))?;
        let (nstates, nparams, nout, _ndata, nroots, has_mass) = self.compiler.get_dims();
        self.data = RefCell::new(self.compiler.get_new_data());
        self.ddata = RefCell::new(self.compiler.get_new_data());
        self.tmp = RefCell::new(M::V::zeros(nstates));
        self.nparams = nparams;
        self.nstates = nstates;
        self.nout = nout;
        self.nroots = nroots;
        self.has_mass = has_mass;
        Ok(())
    }
}

impl<M: Matrix<T = T>, CG: CodegenModule> Default for DiffSlContext<M, CG> {
    fn default() -> Self {
        Self::new(
            "
            u { y = 1 }
            F { -y }
            out { y }
        ",
            1,
        )
        .unwrap()
    }
}

pub struct DiffSl<M: Matrix<T = T>, CG: CodegenModule> {
    context: DiffSlContext<M, CG>,
    mass_sparsity: Option<M::Sparsity>,
    mass_coloring: Option<JacobianColoring<M>>,
    mass_transpose_sparsity: Option<M::Sparsity>,
    mass_transpose_coloring: Option<JacobianColoring<M>>,
    rhs_sparsity: Option<M::Sparsity>,
    rhs_coloring: Option<JacobianColoring<M>>,
    rhs_adjoint_sparsity: Option<M::Sparsity>,
    rhs_adjoint_coloring: Option<JacobianColoring<M>>,
}

impl<M: Matrix<T = T>, CG: CodegenModule> DiffSl<M, CG> {
    pub fn compile(code: &str) -> Result<Self, DiffsolError> {
        let context = DiffSlContext::<M, CG>::new(code, 1)?;
        Ok(Self::from_context(context))
    }
    pub fn from_context(context: DiffSlContext<M, CG>) -> Self {
        let mut ret = Self {
            context,
            mass_coloring: None,
            mass_sparsity: None,
            mass_transpose_coloring: None,
            mass_transpose_sparsity: None,
            rhs_coloring: None,
            rhs_sparsity: None,
            rhs_adjoint_coloring: None,
            rhs_adjoint_sparsity: None,
        };
        if M::is_sparse() {
            let op = ret.rhs();
            let t0 = 0.0;
            let x0 = M::V::zeros(op.nstates());
            let non_zeros = find_jacobian_non_zeros(&op, &x0, t0);
            let n = op.nstates();

            let sparsity = M::Sparsity::try_from_indices(n, n, non_zeros.clone())
                .expect("invalid sparsity pattern");
            let coloring = JacobianColoring::new(&sparsity, &non_zeros);
            ret.rhs_coloring = Some(coloring);
            ret.rhs_sparsity = Some(sparsity);

            let non_zeros = non_zeros
                .into_iter()
                .map(|(i, j)| (j, i))
                .collect::<Vec<_>>();
            let sparsity = M::Sparsity::try_from_indices(n, n, non_zeros.clone())
                .expect("invalid sparsity pattern");
            let coloring = JacobianColoring::new(&sparsity, &non_zeros);
            ret.rhs_adjoint_sparsity = Some(sparsity);
            ret.rhs_adjoint_coloring = Some(coloring);

            if let Some(op) = ret.mass() {
                let non_zeros = find_matrix_non_zeros(&op, t0);
                let sparsity = M::Sparsity::try_from_indices(n, n, non_zeros.clone())
                    .expect("invalid sparsity pattern");
                let coloring = JacobianColoring::new(&sparsity, &non_zeros);
                ret.mass_coloring = Some(coloring);
                ret.mass_sparsity = Some(sparsity);

                let non_zeros = non_zeros
                    .into_iter()
                    .map(|(i, j)| (j, i))
                    .collect::<Vec<_>>();
                let sparsity = M::Sparsity::try_from_indices(n, n, non_zeros.clone())
                    .expect("invalid sparsity pattern");
                let coloring = JacobianColoring::new(&sparsity, &non_zeros);
                ret.mass_transpose_sparsity = Some(sparsity);
                ret.mass_transpose_coloring = Some(coloring);
            }
        }
        ret
    }
}

pub struct DiffSlRoot<'a, M: Matrix<T = T>, CG: CodegenModule>(&'a DiffSl<M, CG>);
pub struct DiffSlOut<'a, M: Matrix<T = T>, CG: CodegenModule>(&'a DiffSl<M, CG>);
pub struct DiffSlRhs<'a, M: Matrix<T = T>, CG: CodegenModule>(&'a DiffSl<M, CG>);
pub struct DiffSlMass<'a, M: Matrix<T = T>, CG: CodegenModule>(&'a DiffSl<M, CG>);
pub struct DiffSlInit<'a, M: Matrix<T = T>, CG: CodegenModule>(&'a DiffSl<M, CG>);

macro_rules! impl_op_for_diffsl {
    ($name:ident) => {
        impl<M: Matrix<T = T>, CG: CodegenModule> Op for $name<'_, M, CG> {
            type M = M;
            type T = T;
            type V = M::V;

            fn nstates(&self) -> usize {
                self.0.context.nstates
            }
            #[allow(clippy::misnamed_getters)]
            fn nout(&self) -> usize {
                self.0.context.nstates
            }
            fn nparams(&self) -> usize {
                self.0.context.nparams
            }
        }
    };
}

impl_op_for_diffsl!(DiffSlRhs);
impl_op_for_diffsl!(DiffSlMass);

impl<M: Matrix<T = T>, CG: CodegenModule> Op for DiffSlInit<'_, M, CG> {
    type M = M;
    type T = T;
    type V = M::V;

    fn nstates(&self) -> usize {
        self.0.context.nstates
    }
    #[allow(clippy::misnamed_getters)]
    fn nout(&self) -> usize {
        self.0.context.nstates
    }
    fn nparams(&self) -> usize {
        self.0.context.nparams
    }
}

impl<M: Matrix<T = T>, CG: CodegenModule> Op for DiffSlRoot<'_, M, CG> {
    type M = M;
    type T = T;
    type V = M::V;

    fn nstates(&self) -> usize {
        self.0.context.nstates
    }
    #[allow(clippy::misnamed_getters)]
    fn nout(&self) -> usize {
        self.0.context.nroots
    }
    fn nparams(&self) -> usize {
        self.0.context.nparams
    }
}

impl<M: Matrix<T = T>, CG: CodegenModule> Op for DiffSlOut<'_, M, CG> {
    type M = M;
    type T = T;
    type V = M::V;

    fn nstates(&self) -> usize {
        self.0.context.nstates
    }
    fn nout(&self) -> usize {
        self.0.context.nout
    }
    fn nparams(&self) -> usize {
        self.0.context.nparams
    }
}

impl<M: Matrix<T = T>, CG: CodegenModule> ConstantOp for DiffSlInit<'_, M, CG> {
    fn call_inplace(&self, _t: Self::T, y: &mut Self::V) {
        self.0.context.compiler.set_u0(
            y.as_mut_slice(),
            self.0.context.data.borrow_mut().as_mut_slice(),
        );
    }
}

impl<M: Matrix<T = T>, CG: CodegenModule> ConstantOpSens for DiffSlInit<'_, M, CG> {
    fn sens_mul_inplace(&self, _t: Self::T, v: &Self::V, y: &mut Self::V) {
        self.0.context.compiler.set_inputs(
            v.as_slice(),
            self.0.context.sens_data.borrow_mut().as_mut_slice(),
        );
        self.0.context.compiler.set_u0_grad(
            self.0.context.tmp.borrow_mut().as_mut_slice(),
            y.as_mut_slice(),
            self.0.context.data.borrow_mut().as_mut_slice(),
            self.0.context.sens_data.borrow_mut().as_mut_slice(),
        );
    }
}

impl<M: Matrix<T = T>, CG: CodegenModule> ConstantOpSensAdjoint for DiffSlInit<'_, M, CG> {
    fn sens_transpose_mul_inplace(&self, _t: Self::T, v: &Self::V, y: &mut Self::V) {
        // copy v to tmp2
        let mut tmp2 = self.0.context.tmp2.borrow_mut();
        tmp2.copy_from(v);
        // zero out sens_data
        self.0.context.sens_data.borrow_mut().fill(0.0);
        self.0.context.compiler.set_u0_rgrad(
            self.0.context.tmp.borrow().as_slice(),
            tmp2.as_mut_slice(),
            self.0.context.data.borrow().as_slice(),
            self.0.context.sens_data.borrow_mut().as_mut_slice(),
        );
        self.0.context.compiler.get_inputs(
            y.as_mut_slice(),
            self.0.context.sens_data.borrow().as_slice(),
        );
        // negate y
        y.mul_assign(Scale(-1.0));
    }
}

impl<M: Matrix<T = T>, CG: CodegenModule> NonLinearOp for DiffSlRoot<'_, M, CG> {
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        self.0.context.compiler.calc_stop(
            t,
            x.as_slice(),
            self.0.context.data.borrow_mut().as_mut_slice(),
            y.as_mut_slice(),
        );
    }
}

impl<M: Matrix<T = T>, CG: CodegenModule> NonLinearOpJacobian for DiffSlRoot<'_, M, CG> {
    fn jac_mul_inplace(&self, _x: &Self::V, _t: Self::T, _v: &Self::V, y: &mut Self::V) {
        y.fill(0.0);
    }
}

impl<M: Matrix<T = T>, CG: CodegenModule> NonLinearOp for DiffSlOut<'_, M, CG> {
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        self.0.context.compiler.calc_out(
            t,
            x.as_slice(),
            self.0.context.data.borrow_mut().as_mut_slice(),
            y.as_mut_slice(),
        );
    }
}

impl<M: Matrix<T = T>, CG: CodegenModule> NonLinearOpJacobian for DiffSlOut<'_, M, CG> {
    fn jac_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        // init ddata with all zero except for out
        let mut ddata = self.0.context.ddata.borrow_mut();
        ddata.fill(0.0);
        self.0.context.compiler.calc_out_grad(
            t,
            x.as_slice(),
            v.as_slice(),
            self.0.context.data.borrow_mut().as_mut_slice(),
            ddata.as_mut_slice(),
            self.0.context.tmp_out.borrow().as_slice(),
            y.as_mut_slice(),
        );
    }
}

impl<M: Matrix<T = T>, CG: CodegenModule> NonLinearOpAdjoint for DiffSlOut<'_, M, CG> {
    fn jac_transpose_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        // init ddata with all zero except for out
        let mut ddata = self.0.context.ddata.borrow_mut();
        ddata.fill(0.0);
        let mut tmp2_out = self.0.context.tmp2_out.borrow_mut();
        tmp2_out.copy_from(v);
        // zero y
        y.fill(0.0);
        self.0.context.compiler.calc_out_rgrad(
            t,
            x.as_slice(),
            y.as_mut_slice(),
            self.0.context.data.borrow_mut().as_slice(),
            ddata.as_mut_slice(),
            self.0.context.tmp_out.borrow().as_slice(),
            tmp2_out.as_mut_slice(),
        );
        // negate y
        y.mul_assign(Scale(-1.0));
    }
}

impl<M: Matrix<T = T>, CG: CodegenModule> NonLinearOpSens for DiffSlOut<'_, M, CG> {
    fn sens_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        // set inputs for sens_data
        self.0.context.compiler.set_inputs(
            v.as_slice(),
            self.0.context.sens_data.borrow_mut().as_mut_slice(),
        );
        self.0.context.compiler.calc_out_sgrad(
            t,
            x.as_slice(),
            self.0.context.data.borrow_mut().as_mut_slice(),
            self.0.context.sens_data.borrow_mut().as_mut_slice(),
            self.0.context.tmp_out.borrow().as_slice(),
            y.as_mut_slice(),
        );
    }
}

impl<M: Matrix<T = T>, CG: CodegenModule> NonLinearOpSensAdjoint for DiffSlOut<'_, M, CG> {
    fn sens_transpose_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        let mut sens_data = self.0.context.sens_data.borrow_mut();
        // set outputs for sens_data (zero everything except for out)
        sens_data.fill(0.0);
        let mut tmp2_out = self.0.context.tmp2_out.borrow_mut();
        tmp2_out.copy_from(v);
        self.0.context.compiler.calc_out_srgrad(
            t,
            x.as_slice(),
            self.0.context.data.borrow_mut().as_mut_slice(),
            sens_data.as_mut_slice(),
            self.0.context.tmp_out.borrow().as_slice(),
            tmp2_out.as_mut_slice(),
        );
        // set y to the result in inputs
        self.0
            .context
            .compiler
            .get_inputs(y.as_mut_slice(), sens_data.as_slice());
        // negate y
        y.mul_assign(Scale(-1.0));
    }
}

impl<M: Matrix<T = T>, CG: CodegenModule> NonLinearOp for DiffSlRhs<'_, M, CG> {
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        self.0.context.compiler.rhs(
            t,
            x.as_slice(),
            self.0.context.data.borrow_mut().as_mut_slice(),
            y.as_mut_slice(),
        );
    }
}

impl<M: Matrix<T = T>, CG: CodegenModule> NonLinearOpJacobian for DiffSlRhs<'_, M, CG> {
    fn jac_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        let tmp = self.0.context.tmp.borrow();
        self.0.context.compiler.rhs_grad(
            t,
            x.as_slice(),
            v.as_slice(),
            self.0.context.data.borrow_mut().as_slice(),
            self.0.context.ddata.borrow_mut().as_mut_slice(),
            tmp.as_slice(),
            y.as_mut_slice(),
        );
    }

    fn jacobian_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        if let Some(coloring) = &self.0.rhs_coloring {
            coloring.jacobian_inplace(self, x, t, y);
        } else {
            self._default_jacobian_inplace(x, t, y);
        }
    }
    fn jacobian_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.0.rhs_sparsity.clone()
    }
}

impl<M: Matrix<T = T>, CG: CodegenModule> NonLinearOpAdjoint for DiffSlRhs<'_, M, CG> {
    fn jac_transpose_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        // copy v to tmp2
        let mut tmp2 = self.0.context.tmp2.borrow_mut();
        tmp2.copy_from(v);
        // zero out ddata
        self.0.context.ddata.borrow_mut().fill(0.0);
        // zero y
        y.fill(0.0);
        self.0.context.compiler.rhs_rgrad(
            t,
            x.as_slice(),
            y.as_mut_slice(),
            self.0.context.data.borrow().as_slice(),
            self.0.context.ddata.borrow_mut().as_mut_slice(),
            self.0.context.tmp.borrow().as_slice(),
            tmp2.as_mut_slice(),
        );
        // negate y
        y.mul_assign(Scale(-1.0));
    }
    fn adjoint_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::M) {
        // if we have a rhs_coloring and no rhs_adjoint_coloring, user has not called prep_adjoint
        // fail here
        if self.0.rhs_coloring.is_some() && self.0.rhs_adjoint_coloring.is_none() {
            panic!("Adjoint not prepared. Call prep_adjoint before calling adjoint_inplace");
        }
        if let Some(coloring) = &self.0.rhs_adjoint_coloring {
            coloring.jacobian_inplace(self, x, t, y);
        } else {
            self._default_adjoint_inplace(x, t, y);
        }
    }
    fn adjoint_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.0.rhs_adjoint_sparsity.clone()
    }
}

impl<M: Matrix<T = T>, CG: CodegenModule> NonLinearOpSens for DiffSlRhs<'_, M, CG> {
    fn sens_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        let tmp = self.0.context.tmp.borrow();
        self.0.context.compiler.set_inputs(
            v.as_slice(),
            self.0.context.sens_data.borrow_mut().as_mut_slice(),
        );
        self.0.context.compiler.rhs_sgrad(
            t,
            x.as_slice(),
            self.0.context.data.borrow_mut().as_slice(),
            self.0.context.sens_data.borrow_mut().as_mut_slice(),
            tmp.as_slice(),
            y.as_mut_slice(),
        );
    }
}

impl<M: Matrix<T = T>, CG: CodegenModule> NonLinearOpSensAdjoint for DiffSlRhs<'_, M, CG> {
    fn sens_transpose_mul_inplace(&self, x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        // todo: would rhs_srgrad ever use rr? I don't think so, but need to check
        let tmp = self.0.context.tmp.borrow();
        // copy v to tmp2
        let mut tmp2 = self.0.context.tmp2.borrow_mut();
        tmp2.copy_from(v);
        // zero out sens_data
        self.0.context.sens_data.borrow_mut().fill(0.0);
        self.0.context.compiler.rhs_srgrad(
            t,
            x.as_slice(),
            self.0.context.data.borrow_mut().as_mut_slice(),
            self.0.context.sens_data.borrow_mut().as_mut_slice(),
            tmp.as_slice(),
            tmp2.as_mut_slice(),
        );
        // get inputs
        self.0.context.compiler.get_inputs(
            y.as_mut_slice(),
            self.0.context.sens_data.borrow().as_slice(),
        );
        // negate y
        y.mul_assign(Scale(-1.0));
    }
}

impl<M: Matrix<T = T>, CG: CodegenModule> LinearOp for DiffSlMass<'_, M, CG> {
    fn gemv_inplace(&self, x: &Self::V, t: Self::T, beta: Self::T, y: &mut Self::V) {
        let mut tmp = self.0.context.tmp.borrow_mut();
        self.0.context.compiler.mass(
            t,
            x.as_slice(),
            self.0.context.data.borrow_mut().as_mut_slice(),
            tmp.as_mut_slice(),
        );

        // y = tmp + beta * y
        y.axpy(1.0, &tmp, beta);
    }

    fn matrix_inplace(&self, t: Self::T, y: &mut Self::M) {
        if let Some(coloring) = &self.0.mass_coloring {
            coloring.matrix_inplace(self, t, y);
        } else {
            self._default_matrix_inplace(t, y);
        }
    }
    fn sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.0.mass_sparsity.clone()
    }
}

impl<M: Matrix<T = T>, CG: CodegenModule> LinearOpTranspose for DiffSlMass<'_, M, CG> {
    fn gemv_transpose_inplace(&self, x: &Self::V, t: Self::T, beta: Self::T, y: &mut Self::V) {
        // scale y by beta
        y.mul_assign(Scale(beta));

        // copy x to tmp
        let mut tmp = self.0.context.tmp.borrow_mut();
        tmp.copy_from(x);

        // zero out ddata
        self.0.context.ddata.borrow_mut().fill(0.0);

        // y += M^T x + beta * y
        self.0.context.compiler.mass_rgrad(
            t,
            y.as_mut_slice(),
            self.0.context.data.borrow_mut().as_slice(),
            self.0.context.ddata.borrow_mut().as_mut_slice(),
            tmp.as_mut_slice(),
        );
    }

    fn transpose_inplace(&self, t: Self::T, y: &mut Self::M) {
        if let Some(coloring) = &self.0.mass_transpose_coloring {
            coloring.matrix_inplace(self, t, y);
        } else {
            self._default_matrix_inplace(t, y);
        }
    }
    fn transpose_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.0.mass_transpose_sparsity.clone()
    }
}

impl<M: Matrix<T = T>, CG: CodegenModule> Op for DiffSl<M, CG> {
    type M = M;
    type T = T;
    type V = M::V;

    fn nstates(&self) -> usize {
        self.context.nstates
    }
    fn nout(&self) -> usize {
        self.context.nout
    }
    fn nparams(&self) -> usize {
        self.context.nparams
    }
}

impl<'a, M: Matrix<T = T>, CG: CodegenModule> OdeEquationsRef<'a> for DiffSl<M, CG> {
    type Mass = DiffSlMass<'a, M, CG>;
    type Rhs = DiffSlRhs<'a, M, CG>;
    type Root = DiffSlRoot<'a, M, CG>;
    type Init = DiffSlInit<'a, M, CG>;
    type Out = DiffSlOut<'a, M, CG>;
}

impl<M: Matrix<T = T>, CG: CodegenModule> OdeEquations for DiffSl<M, CG> {
    fn rhs(&self) -> DiffSlRhs<'_, M, CG> {
        DiffSlRhs(self)
    }

    fn mass(&self) -> Option<DiffSlMass<'_, M, CG>> {
        self.context.has_mass.then_some(DiffSlMass(self))
    }

    fn root(&self) -> Option<DiffSlRoot<'_, M, CG>> {
        self.context.has_root.then_some(DiffSlRoot(self))
    }

    fn init(&self) -> DiffSlInit<'_, M, CG> {
        DiffSlInit(self)
    }

    fn out(&self) -> Option<DiffSlOut<'_, M, CG>> {
        self.context.has_out.then_some(DiffSlOut(self))
    }

    fn set_params(&mut self, p: &Self::V) {
        // set the parameters in data
        self.context
            .compiler
            .set_inputs(p.as_slice(), self.context.data.borrow_mut().as_mut_slice());

        // set_u0 will calculate all the constants in the equations based on the params
        let mut dummy = M::V::zeros(self.context.nstates);
        self.context.compiler.set_u0(
            dummy.as_mut_slice(),
            self.context.data.borrow_mut().as_mut_slice(),
        );
    }

    fn get_params(&self, p: &mut Self::V) {
        self.context
            .compiler
            .get_inputs(p.as_mut_slice(), self.context.data.borrow().as_slice());
    }
}

#[cfg(test)]
mod tests {
    use diffsl::{execution::module::CodegenModule, CraneliftModule};
    use nalgebra::DVector;

    use crate::{
        ConstantOp, LinearOp, NalgebraLU, NonLinearOp, NonLinearOpJacobian, OdeBuilder,
        OdeEquations, OdeSolverMethod, Vector,
    };

    use super::{DiffSl, DiffSlContext};

    #[test]
    fn diffsl_logistic_growth_cranelift() {
        diffsl_logistic_growth::<CraneliftModule>();
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn diffsl_logistic_growth_llvm() {
        diffsl_logistic_growth::<diffsl::LlvmModule>();
    }

    fn diffsl_logistic_growth<CG: CodegenModule>() {
        let text = "
            in = [r, k]
            r { 1 }
            k { 1 }
            u_i {
                y = 0.1,
                z = 0,
            }
            dudt_i {
                dydt = 0,
                dzdt = 0,
            }
            M_i {
                dydt,
                0,
            }
            F_i {
                (r * y) * (1 - (y / k)),
                (2 * y) - z,
            }
            out_i {
                3 * y,
                4 * z,
            }
        ";

        let k = 1.0;
        let r = 1.0;
        let context = DiffSlContext::<nalgebra::DMatrix<f64>, CG>::new(text, 1).unwrap();
        let p = DVector::from_vec(vec![r, k]);
        let mut eqn = DiffSl::from_context(context);
        eqn.set_params(&p);

        // test that the initial values look ok
        let y0 = 0.1;
        let init = eqn.init().call(0.0);
        let init_expect = DVector::from_vec(vec![y0, 0.0]);
        init.assert_eq_st(&init_expect, 1e-10);
        let rhs = eqn.rhs().call(&init, 0.0);
        let rhs_expect = DVector::from_vec(vec![r * y0 * (1.0 - y0 / k), 2.0 * y0]);
        rhs.assert_eq_st(&rhs_expect, 1e-10);
        let v = DVector::from_vec(vec![1.0, 1.0]);
        let rhs_jac = eqn.rhs().jac_mul(&init, 0.0, &v);
        let rhs_jac_expect = DVector::from_vec(vec![r * (1.0 - y0 / k) - r * y0 / k, 1.0]);
        rhs_jac.assert_eq_st(&rhs_jac_expect, 1e-10);
        let mut mass_y = DVector::from_vec(vec![0.0, 0.0]);
        let v = DVector::from_vec(vec![1.0, 1.0]);
        eqn.mass().unwrap().call_inplace(&v, 0.0, &mut mass_y);
        let mass_y_expect = DVector::from_vec(vec![1.0, 0.0]);
        mass_y.assert_eq_st(&mass_y_expect, 1e-10);

        // solver a bit and check the state and output
        let problem = OdeBuilder::<nalgebra::DMatrix<f64>>::new()
            .p([r, k])
            .build_from_eqn(eqn)
            .unwrap();
        let mut solver = problem.bdf::<NalgebraLU<f64>>().unwrap();
        let t = 1.0;
        let (ys, ts) = solver.solve(t).unwrap();
        for (i, t) in ts.iter().enumerate() {
            let y_expect = k / (1.0 + (k - y0) * (-r * t).exp() / y0);
            let z_expect = 2.0 * y_expect;
            let expected_out = DVector::from_vec(vec![3.0 * y_expect, 4.0 * z_expect]);
            ys.column(i).into_owned().assert_eq_st(&expected_out, 1e-4);
        }

        // do it again with some explicit t_evals
        let t_evals = vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 1.0];
        let mut solver = problem.bdf::<NalgebraLU<f64>>().unwrap();
        let ys = solver.solve_dense(&t_evals).unwrap();
        for (i, t) in t_evals.iter().enumerate() {
            let y_expect = k / (1.0 + (k - y0) * (-r * t).exp() / y0);
            let z_expect = 2.0 * y_expect;
            let expected_out = DVector::from_vec(vec![3.0 * y_expect, 4.0 * z_expect]);
            ys.column(i).into_owned().assert_eq_st(&expected_out, 1e-4);
        }
    }
}
