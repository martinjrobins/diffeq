use num_traits::{One, Zero};
use std::{
    cell::RefCell,
    ops::{AddAssign, SubAssign},
    rc::Rc,
};

use crate::{
    error::DiffsolError, op::nonlinear_op::NonLinearOpJacobian, AugmentedOdeEquations,
    Checkpointing, ConstantOp, ConstantOpSensAdjoint, LinearOp, LinearOpTranspose, Matrix,
    NonLinearOp, NonLinearOpAdjoint, NonLinearOpSensAdjoint, OdeEquations, OdeEquationsAdjoint,
    OdeEquationsRef, OdeSolverMethod, OdeSolverProblem, Op, Vector,
};

pub struct AdjointContext<'a, Eqn, Method>
where
    Eqn: OdeEquations,
    Method: OdeSolverMethod<'a, Eqn>,
{
    checkpointer: Checkpointing<'a, Eqn, Method>,
    x: Eqn::V,
    index: usize,
    max_index: usize,
    last_t: Option<Eqn::T>,
    col: Eqn::V,
}

impl<'a, Eqn, Method> AdjointContext<'a, Eqn, Method>
where
    Eqn: OdeEquationsAdjoint,
    Method: OdeSolverMethod<'a, Eqn>,
{
    pub fn new(checkpointer: Checkpointing<'a, Eqn, Method>, max_index: usize) -> Self {
        let x = <Eqn::V as Vector>::zeros(checkpointer.problem().eqn.rhs().nstates());
        let mut col = <Eqn::V as Vector>::zeros(max_index);
        let index = 0;
        col[0] = Eqn::T::one();
        Self {
            checkpointer,
            x,
            index,
            max_index,
            col,
            last_t: None,
        }
    }

    pub fn set_state(&mut self, t: Eqn::T) {
        if let Some(last_t) = self.last_t {
            if last_t == t {
                return;
            }
        }
        self.last_t = Some(t);
        self.checkpointer.interpolate(t, &mut self.x).unwrap();
        // for diffsl, we need to set data for the adjoint state!
        // basically just involves calling the normal rhs function with the new self.x
        // todo: this seems a bit hacky, perhaps a dedicated function on the trait for this?
        self.checkpointer.problem().eqn.rhs().call(&self.x, t);
    }

    pub fn state(&self) -> &Eqn::V {
        &self.x
    }

    pub fn col(&self) -> &Eqn::V {
        &self.col
    }

    pub fn set_index(&mut self, index: usize) {
        self.col[self.index] = Eqn::T::zero();
        self.index = index;
        self.col[self.index] = Eqn::T::one();
    }
}

pub struct AdjointMass<'a, Eqn>
where
    Eqn: OdeEquationsAdjoint,
{
    eqn: &'a Eqn,
}

impl<'a, Eqn> AdjointMass<'a, Eqn>
where
    Eqn: OdeEquationsAdjoint,
{
    pub fn new(eqn: &'a Eqn) -> Self {
        Self { eqn }
    }
}

impl<Eqn> Op for AdjointMass<'_, Eqn>
where
    Eqn: OdeEquationsAdjoint,
{
    type T = Eqn::T;
    type V = Eqn::V;
    type M = Eqn::M;

    fn nstates(&self) -> usize {
        self.eqn.rhs().nstates()
    }
    fn nout(&self) -> usize {
        self.eqn.rhs().nstates()
    }
    fn nparams(&self) -> usize {
        self.eqn.rhs().nparams()
    }
}

impl<Eqn> LinearOp for AdjointMass<'_, Eqn>
where
    Eqn: OdeEquationsAdjoint,
{
    fn gemv_inplace(&self, x: &Self::V, t: Self::T, beta: Self::T, y: &mut Self::V) {
        self.eqn
            .mass()
            .unwrap()
            .gemv_transpose_inplace(x, t, beta, y);
    }

    fn matrix_inplace(&self, t: Self::T, y: &mut Self::M) {
        self.eqn.mass().unwrap().transpose_inplace(t, y);
    }
}

pub struct AdjointInit<'a, Eqn>
where
    Eqn: OdeEquationsAdjoint,
{
    eqn: &'a Eqn,
}

impl<'a, Eqn> AdjointInit<'a, Eqn>
where
    Eqn: OdeEquationsAdjoint,
{
    pub fn new(eqn: &'a Eqn) -> Self {
        Self { eqn }
    }
}

impl<Eqn> Op for AdjointInit<'_, Eqn>
where
    Eqn: OdeEquationsAdjoint,
{
    type T = Eqn::T;
    type V = Eqn::V;
    type M = Eqn::M;

    fn nstates(&self) -> usize {
        self.eqn.rhs().nstates()
    }
    fn nout(&self) -> usize {
        self.eqn.rhs().nstates()
    }
    fn nparams(&self) -> usize {
        self.eqn.rhs().nparams()
    }
}

impl<Eqn> ConstantOp for AdjointInit<'_, Eqn>
where
    Eqn: OdeEquationsAdjoint,
{
    fn call_inplace(&self, _t: Self::T, y: &mut Self::V) {
        y.fill(Eqn::T::zero());
    }
}

/// Right-hand side of the adjoint equations is:
///
/// F(λ, x, t) = -f^T_x(x, t) λ - g^T_x(x,t)
///
/// f_x is the partial derivative of the right-hand side with respect to the state vector.
/// g_x is the partial derivative of the functional g with respect to the state vector.
///
/// We need the current state x(t), which is obtained from the checkpointed forward solve at the current time step.
pub struct AdjointRhs<'a, Eqn, Method>
where
    Eqn: OdeEquations,
    Method: OdeSolverMethod<'a, Eqn>,
{
    eqn: &'a Eqn,
    context: Rc<RefCell<AdjointContext<'a, Eqn, Method>>>,
    tmp: RefCell<Eqn::V>,
    with_out: bool,
}

impl<'a, Eqn, Method> AdjointRhs<'a, Eqn, Method>
where
    Eqn: OdeEquations,
    Method: OdeSolverMethod<'a, Eqn>,
{
    pub fn new(
        eqn: &'a Eqn,
        context: Rc<RefCell<AdjointContext<'a, Eqn, Method>>>,
        with_out: bool,
    ) -> Self {
        let tmp_n = if with_out { eqn.rhs().nstates() } else { 0 };
        let tmp = RefCell::new(<Eqn::V as Vector>::zeros(tmp_n));
        Self {
            eqn,
            context,
            tmp,
            with_out,
        }
    }
}

impl<'a, Eqn, Method> Op for AdjointRhs<'a, Eqn, Method>
where
    Eqn: OdeEquationsAdjoint,
    Method: OdeSolverMethod<'a, Eqn>,
{
    type T = Eqn::T;
    type V = Eqn::V;
    type M = Eqn::M;

    fn nstates(&self) -> usize {
        self.eqn.rhs().nstates()
    }
    fn nout(&self) -> usize {
        self.eqn.rhs().nstates()
    }
    fn nparams(&self) -> usize {
        self.eqn.rhs().nparams()
    }
}

impl<'a, Eqn, Method> NonLinearOp for AdjointRhs<'a, Eqn, Method>
where
    Eqn: OdeEquationsAdjoint,
    Method: OdeSolverMethod<'a, Eqn>,
{
    /// F(λ, x, t) = -f^T_x(x, t) λ - g^T_x(x,t)
    fn call_inplace(&self, lambda: &Self::V, t: Self::T, y: &mut Self::V) {
        self.context.borrow_mut().set_state(t);
        let context = self.context.borrow();
        let x = context.state();

        // y = -f^T_x(x, t) λ
        self.eqn.rhs().jac_transpose_mul_inplace(x, t, lambda, y);

        // y = -f^T_x(x, t) λ - g^T_x(x,t)
        if self.with_out {
            let col = context.col();
            let mut tmp = self.tmp.borrow_mut();
            self.eqn
                .out()
                .unwrap()
                .jac_transpose_mul_inplace(x, t, col, &mut tmp);
            y.add_assign(&*tmp);
        }
    }
}

impl<'a, Eqn, Method> NonLinearOpJacobian for AdjointRhs<'a, Eqn, Method>
where
    Eqn: OdeEquationsAdjoint,
    Method: OdeSolverMethod<'a, Eqn>,
{
    // J = -f^T_x(x, t)
    fn jac_mul_inplace(&self, _x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        self.context.borrow_mut().set_state(t);
        let context = self.context.borrow();
        let x = context.state();
        self.eqn.rhs().jac_transpose_mul_inplace(x, t, v, y);
    }
    fn jacobian_inplace(&self, _x: &Self::V, t: Self::T, y: &mut Self::M) {
        self.context.borrow_mut().set_state(t);
        let context = self.context.borrow();
        let x = context.state();
        self.eqn.rhs().adjoint_inplace(x, t, y);
    }
    fn jacobian_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.eqn.rhs().adjoint_sparsity()
    }
}

/// Output of the adjoint equations is:
///
/// F(λ, x, t) = -g_p^T(x, t) - f_p^T(x, t) λ
///
/// f_p is the partial derivative of the right-hand side with respect to the parameter vector
/// g_p is the partial derivative of the functional g with respect to the parameter vector
///
/// We need the current state x(t), which is obtained from the checkpointed forward solve at the current time step.
pub struct AdjointOut<'a, Eqn, Method>
where
    Eqn: OdeEquationsAdjoint,
    Method: OdeSolverMethod<'a, Eqn>,
{
    eqn: &'a Eqn,
    context: Rc<RefCell<AdjointContext<'a, Eqn, Method>>>,
    tmp: RefCell<Eqn::V>,
    with_out: bool,
}

impl<'a, Eqn, Method> AdjointOut<'a, Eqn, Method>
where
    Eqn: OdeEquationsAdjoint,
    Method: OdeSolverMethod<'a, Eqn>,
{
    pub fn new(
        eqn: &'a Eqn,
        context: Rc<RefCell<AdjointContext<'a, Eqn, Method>>>,
        with_out: bool,
    ) -> Self {
        let tmp_n = if with_out { eqn.rhs().nparams() } else { 0 };
        let tmp = RefCell::new(<Eqn::V as Vector>::zeros(tmp_n));
        Self {
            eqn,
            context,
            tmp,
            with_out,
        }
    }
}

impl<'a, Eqn, Method> Op for AdjointOut<'a, Eqn, Method>
where
    Eqn: OdeEquationsAdjoint,
    Method: OdeSolverMethod<'a, Eqn>,
{
    type T = Eqn::T;
    type V = Eqn::V;
    type M = Eqn::M;

    fn nstates(&self) -> usize {
        self.eqn.rhs().nstates()
    }
    fn nout(&self) -> usize {
        self.eqn.rhs().nparams()
    }
    fn nparams(&self) -> usize {
        self.eqn.rhs().nparams()
    }
}

impl<'a, Eqn, Method> NonLinearOp for AdjointOut<'a, Eqn, Method>
where
    Eqn: OdeEquationsAdjoint,
    Method: OdeSolverMethod<'a, Eqn>,
{
    /// F(λ, x, t) = -g_p(x, t) - λ^T f_p(x, t)
    fn call_inplace(&self, lambda: &Self::V, t: Self::T, y: &mut Self::V) {
        self.context.borrow_mut().set_state(t);
        let context = self.context.borrow();
        let x = context.state();
        self.eqn.rhs().sens_transpose_mul_inplace(x, t, lambda, y);

        if self.with_out {
            let col = context.col();
            let mut tmp = self.tmp.borrow_mut();
            self.eqn
                .out()
                .unwrap()
                .sens_transpose_mul_inplace(x, t, col, &mut tmp);
            y.add_assign(&*tmp);
        }
    }
}

impl<'a, Eqn, Method> NonLinearOpJacobian for AdjointOut<'a, Eqn, Method>
where
    Eqn: OdeEquationsAdjoint,
    Method: OdeSolverMethod<'a, Eqn>,
{
    // J = -f_p(x, t)
    fn jac_mul_inplace(&self, _x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        self.context.borrow_mut().set_state(t);
        let context = self.context.borrow();
        let x = context.state();
        self.eqn.rhs().sens_transpose_mul_inplace(x, t, v, y);
    }
    fn jacobian_inplace(&self, _x: &Self::V, t: Self::T, y: &mut Self::M) {
        self.context.borrow_mut().set_state(t);
        let context = self.context.borrow();
        let x = context.state();
        self.eqn.rhs().sens_adjoint_inplace(x, t, y);
    }
    fn jacobian_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.eqn.rhs().sens_adjoint_sparsity()
    }
}

/// Adjoint equations for ODEs
///
/// M * dλ/dt = -f^T_x(x, t) λ - g^T_x(x,t)
/// λ(T) = 0
/// g(λ, x, t) = -g_p(x, t) - λ^T f_p(x, t)
///
pub struct AdjointEquations<'a, Eqn, Method>
where
    Eqn: OdeEquationsAdjoint,
    Method: OdeSolverMethod<'a, Eqn>,
{
    eqn: &'a Eqn,
    rhs: AdjointRhs<'a, Eqn, Method>,
    out: AdjointOut<'a, Eqn, Method>,
    mass: Option<AdjointMass<'a, Eqn>>,
    context: Rc<RefCell<AdjointContext<'a, Eqn, Method>>>,
    tmp: RefCell<Eqn::V>,
    tmp2: RefCell<Eqn::V>,
    init: AdjointInit<'a, Eqn>,
    atol: Option<&'a Eqn::V>,
    rtol: Option<Eqn::T>,
    out_rtol: Option<Eqn::T>,
    out_atol: Option<&'a Eqn::V>,
}

impl<'a, Eqn, Method> Clone for AdjointEquations<'a, Eqn, Method>
where
    Eqn: OdeEquationsAdjoint,
    Method: OdeSolverMethod<'a, Eqn>,
{
    fn clone(&self) -> Self {
        let context = Rc::new(RefCell::new(AdjointContext::new(
            self.context.borrow().checkpointer.clone(),
            self.context.borrow().max_index,
        )));
        let rhs = AdjointRhs::new(self.eqn, context.clone(), self.rhs.with_out);
        let init = AdjointInit::new(self.eqn);
        let out = AdjointOut::new(self.eqn, context.clone(), self.out.with_out);
        let tmp = self.tmp.clone();
        let tmp2 = self.tmp2.clone();
        let atol = self.atol;
        let rtol = self.rtol;
        let out_atol = self.out_atol;
        let out_rtol = self.out_rtol;
        let mass = self.eqn.mass().map(|_m| AdjointMass::new(self.eqn));
        Self {
            rhs,
            init,
            mass,
            context,
            out,
            tmp,
            tmp2,
            eqn: self.eqn,
            atol,
            rtol,
            out_rtol,
            out_atol,
        }
    }
}

impl<'a, Eqn, Method> AdjointEquations<'a, Eqn, Method>
where
    Eqn: OdeEquationsAdjoint,
    Method: OdeSolverMethod<'a, Eqn>,
{
    pub(crate) fn new(
        problem: &'a OdeSolverProblem<Eqn>,
        context: Rc<RefCell<AdjointContext<'a, Eqn, Method>>>,
        with_out: bool,
    ) -> Self {
        let eqn = &problem.eqn;
        let rhs = AdjointRhs::new(eqn, context.clone(), with_out);
        let init = AdjointInit::new(eqn);
        let out = AdjointOut::new(eqn, context.clone(), with_out);
        let tmp = RefCell::new(<Eqn::V as Vector>::zeros(eqn.rhs().nparams()));
        let tmp2 = RefCell::new(<Eqn::V as Vector>::zeros(eqn.rhs().nstates()));
        let atol = problem.sens_atol.as_ref();
        let rtol = problem.sens_rtol;
        let out_atol = problem.out_atol.as_ref();
        let out_rtol = problem.out_rtol;
        let mass = eqn.mass().map(|_m| AdjointMass::new(eqn));
        Self {
            rhs,
            init,
            mass,
            context,
            out,
            tmp,
            tmp2,
            eqn,
            atol,
            rtol,
            out_rtol,
            out_atol,
        }
    }

    pub fn eqn(&self) -> &'a Eqn {
        self.eqn
    }

    pub fn correct_sg_for_init(&self, t: Eqn::T, s: &[Eqn::V], sg: &mut [Eqn::V]) {
        let mut tmp = self.tmp.borrow_mut();
        for (s_i, sg_i) in s.iter().zip(sg.iter_mut()) {
            if let Some(mass) = self.eqn.mass() {
                let mut tmp2 = self.tmp2.borrow_mut();
                mass.call_transpose_inplace(s_i, t, &mut tmp2);
                self.eqn
                    .init()
                    .sens_transpose_mul_inplace(t, &tmp2, &mut tmp);
                sg_i.sub_assign(&*tmp);
            } else {
                self.eqn.init().sens_transpose_mul_inplace(t, s_i, &mut tmp);
                sg_i.sub_assign(&*tmp);
            }
        }
    }

    pub fn interpolate_forward_state(&self, t: Eqn::T, y: &mut Eqn::V) -> Result<(), DiffsolError> {
        self.context.borrow_mut().set_state(t);
        let context = self.context.borrow();
        context.checkpointer.interpolate(t, y)
    }
}

impl<'a, Eqn, Method> std::fmt::Debug for AdjointEquations<'a, Eqn, Method>
where
    Eqn: OdeEquationsAdjoint,
    Method: OdeSolverMethod<'a, Eqn>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdjointEquations").finish()
    }
}

impl<'a, Eqn, Method> Op for AdjointEquations<'a, Eqn, Method>
where
    Eqn: OdeEquationsAdjoint,
    Method: OdeSolverMethod<'a, Eqn>,
{
    type T = Eqn::T;
    type V = Eqn::V;
    type M = Eqn::M;

    fn nstates(&self) -> usize {
        self.eqn.rhs().nstates()
    }
    fn nout(&self) -> usize {
        self.eqn.rhs().nout()
    }
    fn nparams(&self) -> usize {
        self.eqn.rhs().nparams()
    }
}

impl<'a, 'b, Eqn, Method> OdeEquationsRef<'a> for AdjointEquations<'b, Eqn, Method>
where
    Eqn: OdeEquationsAdjoint,
    Method: OdeSolverMethod<'b, Eqn>,
{
    type Rhs = &'a AdjointRhs<'b, Eqn, Method>;
    type Mass = &'a AdjointMass<'b, Eqn>;
    type Root = <Eqn as OdeEquationsRef<'a>>::Root;
    type Init = &'a AdjointInit<'b, Eqn>;
    type Out = &'a AdjointOut<'b, Eqn, Method>;
}

impl<'a, Eqn, Method> OdeEquations for AdjointEquations<'a, Eqn, Method>
where
    Eqn: OdeEquationsAdjoint,
    Method: OdeSolverMethod<'a, Eqn>,
{
    fn rhs(&self) -> &AdjointRhs<'a, Eqn, Method> {
        &self.rhs
    }
    fn mass(&self) -> Option<&AdjointMass<'a, Eqn>> {
        self.mass.as_ref()
    }
    fn root(&self) -> Option<<Eqn as OdeEquationsRef<'_>>::Root> {
        None
    }
    fn init(&self) -> &AdjointInit<'a, Eqn> {
        &self.init
    }
    fn out(&self) -> Option<&AdjointOut<'a, Eqn, Method>> {
        Some(&self.out)
    }
    fn set_params(&mut self, p: &Self::V) {
        self.eqn.set_params(p);
    }
    fn get_params(&self, p: &mut Self::V) {
        self.eqn.get_params(p);
    }
}

impl<'a, Eqn, Method> AugmentedOdeEquations<Eqn> for AdjointEquations<'a, Eqn, Method>
where
    Eqn: OdeEquationsAdjoint,
    Method: OdeSolverMethod<'a, Eqn>,
{
    fn include_in_error_control(&self) -> bool {
        self.atol.is_some() && self.rtol.is_some()
    }
    fn include_out_in_error_control(&self) -> bool {
        self.out().is_some() && self.out_atol.is_some() && self.out_rtol.is_some()
    }

    fn atol(&self) -> Option<&Eqn::V> {
        self.atol
    }
    fn out_atol(&self) -> Option<&Eqn::V> {
        self.out_atol
    }
    fn out_rtol(&self) -> Option<Eqn::T> {
        self.out_rtol
    }
    fn rtol(&self) -> Option<Eqn::T> {
        self.rtol
    }

    fn max_index(&self) -> usize {
        self.context.borrow().max_index
    }

    fn set_index(&mut self, index: usize) {
        self.context.borrow_mut().set_index(index);
    }

    fn update_rhs_out_state(&mut self, _y: &Eqn::V, _dy: &Eqn::V, _t: Eqn::T) {}

    fn update_init_state(&mut self, _t: <Eqn as Op>::T) {}

    fn integrate_main_eqn(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc};

    use crate::{
        ode_solver::{
            adjoint_equations::AdjointEquations,
            test_models::exponential_decay::exponential_decay_problem_adjoint,
        },
        AdjointContext, AugmentedOdeEquations, Checkpointing, FaerSparseLU, Matrix, MatrixCommon,
        NonLinearOp, NonLinearOpJacobian, OdeEquations, Op, SdirkState, SparseColMat, Vector,
    };
    type Mcpu = nalgebra::DMatrix<f64>;
    type Vcpu = nalgebra::DVector<f64>;
    type LS = crate::NalgebraLU<f64>;

    #[test]
    fn test_rhs_exponential() {
        // dy/dt = -ay (p = [a])
        // a = 0.1
        let (problem, _soln) = exponential_decay_problem_adjoint::<Mcpu>(true);
        let state = SdirkState {
            t: 0.0,
            y: Vcpu::from_vec(vec![1.0, 1.0]),
            dy: Vcpu::from_vec(vec![1.0, 1.0]),
            g: Vcpu::zeros(0),
            dg: Vcpu::zeros(0),
            sg: Vec::new(),
            dsg: Vec::new(),
            s: Vec::new(),
            ds: Vec::new(),
            h: 0.0,
        };
        let nout = problem.eqn.out().unwrap().nout();
        let solver = problem.esdirk34_solver::<LS>(state.clone()).unwrap();
        let checkpointer = Checkpointing::new(solver, 0, vec![state.clone(), state.clone()], None);
        let context = Rc::new(RefCell::new(AdjointContext::new(checkpointer, nout)));
        let adj_eqn = AdjointEquations::new(&problem, context.clone(), false);
        // F(λ, x, t) = -f^T_x(x, t) λ
        // f_x = |-a 0|
        //       |0 -a|
        // F(s, t)_0 =  |a 0| |1| = |a| = |0.1|
        //              |0 a| |2|   |2a| = |0.2|
        let v = Vcpu::from_vec(vec![1.0, 2.0]);
        let f = adj_eqn.rhs.call(&v, state.t);
        let f_expect = Vcpu::from_vec(vec![0.1, 0.2]);
        f.assert_eq_st(&f_expect, 1e-10);

        let mut adj_eqn = AdjointEquations::new(&problem, context.clone(), true);

        // f_x^T = |-a 0|
        //         |0 -a|
        // J = -f_x^T
        let adjoint = adj_eqn.rhs.jacobian(&state.y, state.t);
        assert_eq!(adjoint.nrows(), 2);
        assert_eq!(adjoint.ncols(), 2);
        assert_eq!(adjoint[(0, 0)], 0.1);
        assert_eq!(adjoint[(1, 1)], 0.1);

        // g_x = |1 2|
        //       |3 4|
        // S = -g^T_x(x,t)
        // so S = |-1 -3|
        //        |-2 -4|

        // f_p^T = |-x_1 -x_2 |
        //         |0   0 |
        // g_p = |0 0|
        //       |0 0|
        // g(λ, x, t) = -g_p(x, t) - λ^T f_p(x, t)
        //            = |1  1| |1| + |0| = |3|
        //              |0  0| |2|  |0|  = |0|
        adj_eqn.set_index(0);
        let out = adj_eqn.out.call(&v, state.t);
        let out_expect = Vcpu::from_vec(vec![3.0, 0.0]);
        out.assert_eq_st(&out_expect, 1e-10);

        // F(λ, x, t) = -f^T_x(x, t) λ - g^T_x(x,t)
        // f_x = |-a 0|
        //       |0 -a|
        // F(s, t)_0 =  |a 0| |1| - |1.0| = | a - 1| = |-0.9|
        //              |0 a| |2|   |2.0|   |2a - 2| = |-1.8|
        let f = adj_eqn.rhs.call(&v, state.t);
        let f_expect = Vcpu::from_vec(vec![-0.9, -1.8]);
        f.assert_eq_st(&f_expect, 1e-10);
    }

    #[test]
    fn test_rhs_exponential_sparse() {
        // dy/dt = -ay (p = [a])
        // a = 0.1
        let (problem, _soln) = exponential_decay_problem_adjoint::<SparseColMat<f64>>(true);
        let state = SdirkState {
            t: 0.0,
            y: faer::Col::from_vec(vec![1.0, 1.0]),
            dy: faer::Col::from_vec(vec![1.0, 1.0]),
            g: faer::Col::zeros(0),
            dg: faer::Col::zeros(0),
            sg: Vec::new(),
            dsg: Vec::new(),
            s: Vec::new(),
            ds: Vec::new(),
            h: 0.0,
        };
        let nout = problem.eqn.out().unwrap().nout();
        let solver = problem
            .esdirk34_solver::<FaerSparseLU<f64>>(state.clone())
            .unwrap();
        let checkpointer = Checkpointing::new(solver, 0, vec![state.clone(), state.clone()], None);
        let context = Rc::new(RefCell::new(AdjointContext::new(checkpointer, nout)));
        let mut adj_eqn = AdjointEquations::new(&problem, context, true);

        // f_x^T = |-a 0|
        //         |0 -a|
        // J = -f_x^T
        let adjoint = adj_eqn.rhs.jacobian(&state.y, state.t);
        assert_eq!(adjoint.nrows(), 2);
        assert_eq!(adjoint.ncols(), 2);
        for (i, j, v) in adjoint.triplet_iter() {
            if i == j {
                assert_eq!(*v, 0.1);
            } else {
                assert_eq!(*v, 0.0);
            }
        }

        // g_x = |1 2|
        //       |3 4|
        // S = -g^T_x(x,t)
        // so S = |-1 -3|
        //        |-2 -4|

        // F(λ, x, t) = -f^T_x(x, t) λ - g^T_x(x,t)
        // f_x = |-a 0|
        //       |0 -a|
        // F(s, t)_0 =  |a 0| |1| - |1.0| = |a - 1| = |-0.9|
        //              |0 a| |2|   |2.0|   |2a - 2| = |-1.8|
        adj_eqn.set_index(0);
        let v = faer::Col::from_vec(vec![1.0, 2.0]);
        let f = adj_eqn.rhs.call(&v, state.t);
        let f_expect = faer::Col::from_vec(vec![-0.9, -1.8]);
        f.assert_eq_st(&f_expect, 1e-10);
    }
}
