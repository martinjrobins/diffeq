use nalgebra::ComplexField;
use std::cell::Ref;
use std::ops::AddAssign;

use crate::{
    error::{DiffsolError, OdeSolverError},
    AugmentedOdeEquationsImplicit, Convergence, DefaultDenseMatrix, NoAug, StateRef, StateRefMut,
};

use num_traits::{abs, One, Pow, Zero};
use serde::Serialize;

use crate::ode_solver_error;
use crate::{
    matrix::MatrixRef, nonlinear_solver::root::RootFinder, op::bdf::BdfCallable, scalar::scale,
    AugmentedOdeEquations, BdfState, DenseMatrix, IndexType, JacobianUpdate, MatrixViewMut,
    NonLinearOp, NonLinearSolver, OdeEquationsImplicit, OdeSolverMethod, OdeSolverProblem,
    OdeSolverState, OdeSolverStopReason, Op, Scalar, Vector, VectorRef, VectorView, VectorViewMut,
};

use super::jacobian_update::SolverState;
use super::method::AugmentedOdeSolverMethod;

#[derive(Clone, Debug, Serialize, Default)]
pub struct BdfStatistics {
    pub number_of_linear_solver_setups: usize,
    pub number_of_steps: usize,
    pub number_of_error_test_failures: usize,
    pub number_of_nonlinear_solver_iterations: usize,
    pub number_of_nonlinear_solver_fails: usize,
}

impl<'a, M, Eqn, Nls, AugEqn> AugmentedOdeSolverMethod<'a, Eqn, AugEqn>
    for Bdf<'a, Eqn, Nls, M, AugEqn>
where
    Eqn: OdeEquationsImplicit,
    AugEqn: AugmentedOdeEquationsImplicit<Eqn>,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V>,
    for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
    for<'b> &'b Eqn::M: MatrixRef<Eqn::M>,
    Nls: NonLinearSolver<Eqn::M>,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T>,
{
    fn into_state_and_eqn(self) -> (Self::State, Option<AugEqn>) {
        (self.state, self.s_op.map(|op| op.eqn))
    }
    fn augmented_eqn(&self) -> Option<&AugEqn> {
        self.s_op.as_ref().map(|op| op.eqn())
    }
}

// notes quadrature.
// ndf formula rearranged to [2]:
// (1 - kappa) * gamma_k * (y_{n+1} - y^0_{n+1}) + (\sum_{m=1}^k gamma_m * y^m_n) - h * F(t_{n+1}, y_{n+1}) = 0 (1)
// where d = y_{n+1} - y^0_{n+1}
// and y^0_{n+1} = \sum_{m=0}^k y^m_n
//
// 1. use (1) to calculate d explicitly
// 2. use d to update the differences matrix
// 3. use d to calculate the predicted solution y_{n+1}

/// Implements a Backward Difference formula (BDF) implicit multistep integrator.
///
/// The basic algorithm is derived in \[1\]. This
/// particular implementation follows that implemented in the Matlab routine ode15s
/// described in \[2\] and the SciPy implementation
/// /[3/], which features the NDF formulas for improved
/// stability with associated differences in the error constants, and calculates
/// the jacobian at J(t_{n+1}, y^0_{n+1}). This implementation was based on that
/// implemented in the SciPy library \[3\], which also mainly
/// follows \[2\] but uses the more standard Jacobian update.
///
/// # References
///
/// \[1\] Byrne, G. D., & Hindmarsh, A. C. (1975). A polyalgorithm for the numerical solution of ordinary differential equations. ACM Transactions on Mathematical Software (TOMS), 1(1), 71-96.
/// \[2\] Shampine, L. F., & Reichelt, M. W. (1997). The matlab ode suite. SIAM journal on scientific computing, 18(1), 1-22.
/// \[3\] Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., ... & Van Mulbregt, P. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature methods, 17(3), 261-272.
pub struct Bdf<
    'a,
    Eqn: OdeEquationsImplicit,
    Nls: NonLinearSolver<Eqn::M>,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V> = <<Eqn as Op>::V as DefaultDenseMatrix>::M,
    AugmentedEqn: AugmentedOdeEquationsImplicit<Eqn> = NoAug<Eqn>,
> where
    Eqn::V: DefaultDenseMatrix,
{
    nonlinear_solver: Nls,
    convergence: Convergence<'a, Eqn::V>,
    ode_problem: &'a OdeSolverProblem<Eqn>,
    op: Option<BdfCallable<&'a Eqn>>,
    n_equal_steps: usize,
    y_delta: Eqn::V,
    g_delta: Eqn::V,
    y_predict: Eqn::V,
    t_predict: Eqn::T,
    s_predict: Eqn::V,
    s_op: Option<BdfCallable<AugmentedEqn>>,
    s_deltas: Vec<Eqn::V>,
    sg_deltas: Vec<Eqn::V>,
    diff_tmp: M,
    gdiff_tmp: M,
    sgdiff_tmp: M,
    u: M,
    alpha: Vec<Eqn::T>,
    gamma: Vec<Eqn::T>,
    error_const2: Vec<Eqn::T>,
    statistics: BdfStatistics,
    state: BdfState<Eqn::V, M>,
    tstop: Option<Eqn::T>,
    root_finder: Option<RootFinder<Eqn::V>>,
    is_state_modified: bool,
    jacobian_update: JacobianUpdate<Eqn::T>,
}

impl<M, Eqn, Nls, AugmentedEqn> Clone for Bdf<'_, Eqn, Nls, M, AugmentedEqn>
where
    Eqn: OdeEquationsImplicit,
    Nls: NonLinearSolver<Eqn::M>,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V>,
    AugmentedEqn: AugmentedOdeEquationsImplicit<Eqn>,
    Eqn::V: DefaultDenseMatrix,
{
    fn clone(&self) -> Self {
        let problem = self.ode_problem;
        let mut nonlinear_solver = Nls::default();
        let op = if let Some(op) = self.op.as_ref() {
            let op = op.clone_state(&self.ode_problem.eqn);
            nonlinear_solver.set_problem(&op);
            nonlinear_solver.reset_jacobian(&op, &self.state.y, self.state.t);
            Some(op)
        } else {
            None
        };
        let s_op = self.s_op.as_ref().map(|op| {
            let op = op.clone_state(op.eqn().clone());
            op
        });
        Self {
            nonlinear_solver,
            ode_problem: problem,
            convergence: self.convergence.clone(),
            op,
            s_op,
            n_equal_steps: self.n_equal_steps,
            y_delta: self.y_delta.clone(),
            g_delta: self.g_delta.clone(),
            y_predict: self.y_predict.clone(),
            t_predict: self.t_predict,
            s_predict: self.s_predict.clone(),
            s_deltas: self.s_deltas.clone(),
            sg_deltas: self.sg_deltas.clone(),
            diff_tmp: self.diff_tmp.clone(),
            gdiff_tmp: self.gdiff_tmp.clone(),
            sgdiff_tmp: self.sgdiff_tmp.clone(),
            u: self.u.clone(),
            alpha: self.alpha.clone(),
            gamma: self.gamma.clone(),
            error_const2: self.error_const2.clone(),
            statistics: self.statistics.clone(),
            state: self.state.clone(),
            tstop: self.tstop,
            root_finder: self.root_finder.clone(),
            is_state_modified: self.is_state_modified,
            jacobian_update: self.jacobian_update.clone(),
        }
    }
}

impl<'a, M, Eqn, Nls, AugmentedEqn> Bdf<'a, Eqn, Nls, M, AugmentedEqn>
where
    AugmentedEqn: AugmentedOdeEquations<Eqn> + OdeEquationsImplicit,
    Eqn: OdeEquationsImplicit,
    Eqn::V: DefaultDenseMatrix,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V>,
    for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
    for<'b> &'b Eqn::M: MatrixRef<Eqn::M>,
    Nls: NonLinearSolver<Eqn::M>,
{
    const NEWTON_MAXITER: IndexType = 4;
    const MIN_FACTOR: f64 = 0.5;
    const MAX_FACTOR: f64 = 2.1;
    const MAX_THRESHOLD: f64 = 2.0;
    const MIN_THRESHOLD: f64 = 0.9;
    const MIN_TIMESTEP: f64 = 1e-32;

    pub fn new(
        problem: &'a OdeSolverProblem<Eqn>,
        state: BdfState<Eqn::V, M>,
        nonlinear_solver: Nls,
    ) -> Result<Self, DiffsolError> {
        Self::_new(problem, state, nonlinear_solver, true)
    }

    fn _new(
        problem: &'a OdeSolverProblem<Eqn>,
        mut state: BdfState<Eqn::V, M>,
        mut nonlinear_solver: Nls,
        integrate_main_eqn: bool,
    ) -> Result<Self, DiffsolError> {
        // kappa values for difference orders, taken from Table 1 of [1]
        let kappa = [
            Eqn::T::from(0.0),
            Eqn::T::from(-0.1850),
            Eqn::T::from(-1.0) / Eqn::T::from(9.0),
            Eqn::T::from(-0.0823),
            Eqn::T::from(-0.0415),
            Eqn::T::from(0.0),
        ];
        let mut alpha = vec![Eqn::T::zero()];
        let mut gamma = vec![Eqn::T::zero()];
        let mut error_const2 = vec![Eqn::T::one()];

        let max_order: usize = BdfState::<Eqn::V, M>::MAX_ORDER;

        #[allow(clippy::needless_range_loop)]
        for i in 1..=max_order {
            let i_t = Eqn::T::from(i as f64);
            let one_over_i = Eqn::T::one() / i_t;
            let one_over_i_plus_one = Eqn::T::one() / (i_t + Eqn::T::one());
            gamma.push(gamma[i - 1] + one_over_i);
            alpha.push(Eqn::T::one() / ((Eqn::T::one() - kappa[i]) * gamma[i]));
            error_const2.push((kappa[i] * gamma[i] + one_over_i_plus_one).powi(2));
        }

        state.check_consistent_with_problem(problem)?;

        let mut convergence = Convergence::new(problem.rtol, &problem.atol);
        convergence.set_max_iter(Self::NEWTON_MAXITER);

        let op = if integrate_main_eqn {
            // setup linear solver for first step
            let bdf_callable = BdfCallable::new(&problem.eqn);
            bdf_callable.set_c(state.h, alpha[state.order]);
            nonlinear_solver.set_problem(&bdf_callable);
            nonlinear_solver.reset_jacobian(&bdf_callable, &state.y, state.t);
            Some(bdf_callable)
        } else {
            None
        };

        state.set_problem(problem)?;

        // setup root solver
        let mut root_finder = None;
        if let Some(root_fn) = problem.eqn.root() {
            root_finder = Some(RootFinder::new(root_fn.nout()));
            root_finder
                .as_ref()
                .unwrap()
                .init(&root_fn, &state.y, state.t);
        }

        // (re)allocate internal state
        let nstates = problem.eqn.rhs().nstates();
        let diff_tmp = M::zeros(nstates, BdfState::<Eqn::V, M>::MAX_ORDER + 3);
        let y_delta = <Eqn::V as Vector>::zeros(nstates);
        let y_predict = <Eqn::V as Vector>::zeros(nstates);

        let nout = if let Some(out) = problem.eqn.out() {
            out.nout()
        } else {
            0
        };
        let g_delta = <Eqn::V as Vector>::zeros(nout);
        let gdiff_tmp = M::zeros(nout, BdfState::<Eqn::V, M>::MAX_ORDER + 3);

        // init U matrix
        let u = Self::_compute_r(state.order, Eqn::T::one());
        let is_state_modified = false;

        Ok(Self {
            convergence,
            s_op: None,
            op,
            ode_problem: problem,
            nonlinear_solver,
            n_equal_steps: 0,
            diff_tmp,
            gdiff_tmp,
            sgdiff_tmp: M::zeros(0, 0),
            y_delta,
            y_predict,
            t_predict: Eqn::T::zero(),
            s_predict: Eqn::V::zeros(0),
            s_deltas: Vec::new(),
            sg_deltas: Vec::new(),
            g_delta,
            gamma,
            alpha,
            error_const2,
            u,
            statistics: BdfStatistics::default(),
            state,
            tstop: None,
            root_finder,
            is_state_modified,
            jacobian_update: JacobianUpdate::default(),
        })
    }

    pub fn new_augmented(
        state: BdfState<Eqn::V, M>,
        problem: &'a OdeSolverProblem<Eqn>,
        augmented_eqn: AugmentedEqn,
        nonlinear_solver: Nls,
    ) -> Result<Self, DiffsolError> {
        state.check_sens_consistent_with_problem(problem, &augmented_eqn)?;

        let mut ret = Self::_new(
            problem,
            state,
            nonlinear_solver,
            augmented_eqn.integrate_main_eqn(),
        )?;

        ret.state.set_augmented_problem(problem, &augmented_eqn)?;

        // allocate internal state for sensitivities
        let naug = augmented_eqn.max_index();
        let nstates = problem.eqn.rhs().nstates();

        ret.s_op = if augmented_eqn.integrate_main_eqn() {
            Some(BdfCallable::new_no_jacobian(augmented_eqn))
        } else {
            let bdf_callable = BdfCallable::new(augmented_eqn);
            bdf_callable.set_c(ret.state.h, ret.alpha[ret.state.order]);
            ret.nonlinear_solver.set_problem(&bdf_callable);
            ret.nonlinear_solver
                .reset_jacobian(&bdf_callable, &ret.state.s[0], ret.state.t);
            Some(bdf_callable)
        };

        ret.s_deltas = vec![<Eqn::V as Vector>::zeros(nstates); naug];
        ret.s_predict = <Eqn::V as Vector>::zeros(nstates);
        if let Some(out) = ret.s_op.as_ref().unwrap().eqn().out() {
            ret.sg_deltas = vec![<Eqn::V as Vector>::zeros(out.nout()); naug];
            ret.sgdiff_tmp = M::zeros(out.nout(), BdfState::<Eqn::V, M>::MAX_ORDER + 3);
        }
        Ok(ret)
    }

    pub fn get_statistics(&self) -> &BdfStatistics {
        &self.statistics
    }

    fn _compute_r(order: usize, factor: Eqn::T) -> M {
        //computes the R matrix with entries
        //given by the first equation on page 8 of [1]
        //
        //This is used to update the differences matrix when step size h is varied
        //according to factor = h_{n+1} / h_n
        //
        //Note that the U matrix also defined in the same section can be also be
        //found using factor = 1, which corresponds to R with a constant step size
        let mut r = M::zeros(order + 1, order + 1);

        // r[0, 0:order] = 1
        for j in 0..=order {
            r[(0, j)] = M::T::one();
        }
        // r[i, j] = r[i, j-1] * (j - 1 - factor * i) / j
        for i in 1..=order {
            for j in 1..=order {
                let i_t = M::T::from(i as f64);
                let j_t = M::T::from(j as f64);
                r[(i, j)] = r[(i - 1, j)] * (i_t - M::T::one() - factor * j_t) / i_t;
            }
        }
        r
    }

    fn _jacobian_updates(&mut self, c: Eqn::T, state: SolverState) {
        if self.jacobian_update.check_rhs_jacobian_update(c, &state) {
            if let Some(op) = self.op.as_mut() {
                op.set_jacobian_is_stale();
                self.nonlinear_solver
                    .reset_jacobian(op, &self.state.y, self.state.t);
            } else if let Some(s_op) = self.s_op.as_mut() {
                s_op.set_jacobian_is_stale();
                self.nonlinear_solver
                    .reset_jacobian(s_op, &self.state.s[0], self.state.t);
            }
            self.jacobian_update.update_rhs_jacobian();
            self.jacobian_update.update_jacobian(c);
        } else if self.jacobian_update.check_jacobian_update(c, &state) {
            if let Some(op) = self.op.as_mut() {
                self.nonlinear_solver
                    .reset_jacobian(op, &self.state.y, self.state.t);
            } else if let Some(s_op) = self.s_op.as_mut() {
                self.nonlinear_solver
                    .reset_jacobian(s_op, &self.state.s[0], self.state.t);
            }
            self.jacobian_update.update_jacobian(c);
        }
    }

    fn _update_step_size(&mut self, factor: Eqn::T) -> Result<Eqn::T, DiffsolError> {
        //If step size h is changed then also need to update the terms in
        //the first equation of page 9 of [1]:
        //
        //- constant c = h / (1-kappa) gamma_k term
        //- lu factorisation of (M - c * J) used in newton iteration (same equation)

        let new_h = factor * self.state.h;
        self.n_equal_steps = 0;

        // update D using equations in section 3.2 of [1]
        let order = self.state.order;
        let r = Self::_compute_r(order, factor);
        let ru = r.mat_mul(&self.u);
        {
            if self.op.is_some() {
                Self::_update_diff_for_step_size(
                    &ru,
                    &mut self.state.diff,
                    &mut self.diff_tmp,
                    order,
                );
                if self.ode_problem.integrate_out {
                    Self::_update_diff_for_step_size(
                        &ru,
                        &mut self.state.gdiff,
                        &mut self.gdiff_tmp,
                        order,
                    );
                }
            }
            for diff in self.state.sdiff.iter_mut() {
                Self::_update_diff_for_step_size(&ru, diff, &mut self.diff_tmp, order);
            }

            for diff in self.state.sgdiff.iter_mut() {
                Self::_update_diff_for_step_size(&ru, diff, &mut self.sgdiff_tmp, order);
            }
        }

        if let Some(op) = self.op.as_mut() {
            op.set_c(new_h, self.alpha[order]);
        }
        if let Some(s_op) = self.s_op.as_mut() {
            s_op.set_c(new_h, self.alpha[order]);
        }

        self.state.h = new_h;

        // if step size too small, then fail
        if self.state.h.abs() < Eqn::T::from(Self::MIN_TIMESTEP) {
            return Err(DiffsolError::from(OdeSolverError::StepSizeTooSmall {
                time: self.state.t.into(),
            }));
        }
        Ok(new_h)
    }

    fn _update_diff_for_step_size(ru: &M, diff: &mut M, diff_tmp: &mut M, order: usize) {
        // D[0:order+1] = R * U * D[0:order+1]
        {
            let d_zero_order = diff.columns(0, order + 1);
            let mut d_zero_order_tmp = diff_tmp.columns_mut(0, order + 1);
            d_zero_order_tmp.gemm_vo(Eqn::T::one(), &d_zero_order, ru, Eqn::T::zero());
            // diff_sub = diff * RU
        }
        std::mem::swap(diff, diff_tmp);
    }

    fn calculate_output_delta(&mut self) {
        // integrate output function
        let state = &mut self.state;
        let out = self.ode_problem.eqn.out().unwrap();
        out.call_inplace(&self.y_predict, self.t_predict, &mut state.dg);
        self.op.as_ref().unwrap().integrate_out(
            &state.dg,
            &state.gdiff,
            self.gamma.as_slice(),
            self.alpha.as_slice(),
            state.order,
            &mut self.g_delta,
        );
    }

    fn calculate_sens_output_delta(&mut self, i: usize) {
        let state = &mut self.state;
        let s_op = self.s_op.as_ref().unwrap();

        // integrate sensitivity output equations
        let out = s_op.eqn().out().unwrap();
        out.call_inplace(&state.s[i], self.t_predict, &mut state.dsg[i]);

        if let Some(op) = self.s_op.as_ref() {
            op.integrate_out(
                &state.dsg[i],
                &state.sgdiff[i],
                self.gamma.as_slice(),
                self.alpha.as_slice(),
                state.order,
                &mut self.sg_deltas[i],
            );
        } else if let Some(s_op) = self.s_op.as_ref() {
            s_op.integrate_out(
                &state.dsg[i],
                &state.sgdiff[i],
                self.gamma.as_slice(),
                self.alpha.as_slice(),
                state.order,
                &mut self.sg_deltas[i],
            );
        }
    }

    fn update_differences_and_integrate_out(&mut self) {
        let order = self.state.order;
        let state = &mut self.state;

        // update differences
        Self::_update_diff(order, &self.y_delta, &mut state.diff);

        // integrate output function
        if self.ode_problem.integrate_out {
            Self::_predict_using_diff(&mut state.g, &state.gdiff, order);
            state.g.axpy(Eqn::T::one(), &self.g_delta, Eqn::T::one());

            // update output difference
            Self::_update_diff(order, &self.g_delta, &mut state.gdiff);
        }

        // do the same for sensitivities
        if self.s_op.is_some() {
            for i in 0..self.s_op.as_ref().unwrap().eqn().max_index() {
                // update sensitivity differences
                Self::_update_diff(order, &self.s_deltas[i], &mut state.sdiff[i]);

                // integrate sensitivity output equations
                if self.s_op.as_ref().unwrap().eqn().out().is_some() {
                    Self::_predict_using_diff(&mut state.sg[i], &state.sgdiff[i], order);
                    state.sg[i].axpy(Eqn::T::one(), &self.sg_deltas[i], Eqn::T::one());

                    // update sensitivity output difference
                    Self::_update_diff(order, &self.sg_deltas[i], &mut state.sgdiff[i]);
                }
            }
        }
    }

    fn _update_diff(order: usize, d: &Eqn::V, diff: &mut M) {
        //update of difference equations can be done efficiently
        //by reusing d and D.
        //
        //From first equation on page 4 of [1]:
        //d = y_n - y^0_n = D^{k + 1} y_n
        //
        //Standard backwards difference gives
        //D^{j + 1} y_n = D^{j} y_n - D^{j} y_{n - 1}
        //
        //Combining these gives the following algorithm
        let d_minus_order_plus_one = d - diff.column(order + 1);
        diff.column_mut(order + 2)
            .copy_from(&d_minus_order_plus_one);
        diff.column_mut(order + 1).copy_from(d);
        for i in (0..=order).rev() {
            diff.column_axpy(Eqn::T::one(), i + 1, Eqn::T::one(), i);
        }
    }

    // predict forward to new step (eq 2 in [1])
    fn _predict_using_diff(y_predict: &mut Eqn::V, diff: &M, order: usize) {
        y_predict.fill(Eqn::T::zero());
        for i in 0..=order {
            y_predict.add_assign(diff.column(i));
        }
    }

    fn _predict_forward(&mut self) {
        let state = &self.state;
        Self::_predict_using_diff(&mut self.y_predict, &state.diff, state.order);

        // update psi and c (h, D, y0 has changed)
        if let Some(op) = self.op.as_mut() {
            op.set_psi_and_y0(
                &state.diff,
                self.gamma.as_slice(),
                self.alpha.as_slice(),
                state.order,
                &self.y_predict,
            );
        }

        // update time
        let t_new = state.t + state.h;
        self.t_predict = t_new;
    }

    fn handle_tstop(
        &mut self,
        tstop: Eqn::T,
    ) -> Result<Option<OdeSolverStopReason<Eqn::T>>, DiffsolError> {
        // check if the we are at tstop
        let state = &self.state;
        let troundoff = Eqn::T::from(100.0) * Eqn::T::EPSILON * (abs(state.t) + abs(state.h));
        if abs(state.t - tstop) <= troundoff {
            self.tstop = None;
            return Ok(Some(OdeSolverStopReason::TstopReached));
        } else if (state.h > M::T::zero() && tstop < state.t - troundoff)
            || (state.h < M::T::zero() && tstop > state.t + troundoff)
        {
            let error = OdeSolverError::StopTimeBeforeCurrentTime {
                stop_time: self.tstop.unwrap().into(),
                state_time: state.t.into(),
            };
            self.tstop = None;

            return Err(DiffsolError::from(error));
        }

        // check if the next step will be beyond tstop, if so adjust the step size
        if (state.h > M::T::zero() && state.t + state.h > tstop + troundoff)
            || (state.h < M::T::zero() && state.t + state.h < tstop - troundoff)
        {
            let factor = (tstop - state.t) / state.h;
            // update step size ignoring the possible "step size too small" error
            _ = self._update_step_size(factor);
        }
        Ok(None)
    }

    fn initialise_to_first_order(&mut self) {
        self.n_equal_steps = 0;
        self.state.initialise_diff_to_first_order();

        if self.ode_problem.integrate_out {
            self.state.initialise_gdiff_to_first_order();
        }
        if self.s_op.is_some() {
            self.state.initialise_sdiff_to_first_order();
            if self.s_op.as_ref().unwrap().eqn().out().is_some() {
                self.state.initialise_sgdiff_to_first_order();
            }
        }

        self.u = Self::_compute_r(1, Eqn::T::one());
        self.is_state_modified = false;
    }

    //interpolate solution at time values t* where t-h < t* < t
    //definition of the interpolating polynomial can be found on page 7 of [1]
    fn interpolate_from_diff(t: Eqn::T, diff: &M, t1: Eqn::T, h: Eqn::T, order: usize) -> Eqn::V {
        let mut time_factor = Eqn::T::from(1.0);
        let mut order_summation = diff.column(0).into_owned();
        for i in 0..order {
            let i_t = Eqn::T::from(i as f64);
            time_factor *= (t - (t1 - h * i_t)) / (h * (Eqn::T::one() + i_t));
            order_summation += diff.column(i + 1) * scale(time_factor);
        }
        order_summation
    }

    fn error_control(&self) -> Eqn::T {
        let state = &self.state;
        let order = state.order;
        let output_in_error_control = self.ode_problem.output_in_error_control();
        let integrate_sens = self.s_op.is_some();
        let sens_in_error_control =
            integrate_sens && self.s_op.as_ref().unwrap().eqn().include_in_error_control();
        let integrate_sens_out =
            integrate_sens && self.s_op.as_ref().unwrap().eqn().out().is_some();
        let sens_output_in_error_control = integrate_sens_out
            && self
                .s_op
                .as_ref()
                .unwrap()
                .eqn()
                .include_out_in_error_control();

        let mut error_norm = M::T::zero();
        let mut ncontrib = 0;
        if self.op.is_some() {
            let atol = &self.ode_problem.atol;
            let rtol = self.ode_problem.rtol;
            error_norm +=
                self.y_delta.squared_norm(&state.y, atol, rtol) * self.error_const2[order - 1];
            ncontrib += 1;
            if output_in_error_control {
                let rtol = self.ode_problem.out_rtol.unwrap();
                let atol = self.ode_problem.out_atol.as_ref().unwrap();
                error_norm +=
                    self.g_delta.squared_norm(&state.g, atol, rtol) * self.error_const2[order];
                ncontrib += 1;
            }
        }
        if sens_in_error_control {
            let sens_atol = self.s_op.as_ref().unwrap().eqn().atol().unwrap();
            let sens_rtol = self.s_op.as_ref().unwrap().eqn().rtol().unwrap();
            for i in 0..state.sdiff.len() {
                error_norm += self.s_deltas[i].squared_norm(&state.s[i], sens_atol, sens_rtol)
                    * self.error_const2[order];
            }
            ncontrib += state.sdiff.len();
        }
        if sens_output_in_error_control {
            let rtol = self.s_op.as_ref().unwrap().eqn().out_rtol().unwrap();
            let atol = self.s_op.as_ref().unwrap().eqn().out_atol().unwrap();
            for i in 0..state.sgdiff.len() {
                error_norm += self.sg_deltas[i].squared_norm(&state.sg[i], atol, rtol)
                    * self.error_const2[order];
            }
            ncontrib += state.sgdiff.len();
        }
        if ncontrib > 1 {
            error_norm /= Eqn::T::from(ncontrib as f64)
        }
        error_norm
    }

    fn predict_error_control(&self, order: usize) -> Eqn::T {
        let state = &self.state;
        let output_in_error_control = self.ode_problem.output_in_error_control();
        let integrate_sens = self.s_op.is_some();
        let sens_in_error_control =
            integrate_sens && self.s_op.as_ref().unwrap().eqn().include_in_error_control();
        let integrate_sens_out =
            integrate_sens && self.s_op.as_ref().unwrap().eqn().out().is_some();
        let sens_output_in_error_control = integrate_sens_out
            && self
                .s_op
                .as_ref()
                .unwrap()
                .eqn()
                .include_out_in_error_control();

        let atol = &self.ode_problem.atol;
        let rtol = self.ode_problem.rtol;
        let mut error_norm = M::T::zero();
        let mut ncontrib = 0;
        if self.op.is_some() {
            error_norm += state
                .diff
                .column(order + 1)
                .squared_norm(&state.y, atol, rtol)
                * self.error_const2[order];
            ncontrib += 1;
            if output_in_error_control {
                let rtol = self.ode_problem.out_rtol.unwrap();
                let atol = self.ode_problem.out_atol.as_ref().unwrap();
                error_norm += state
                    .gdiff
                    .column(order + 1)
                    .squared_norm(&state.g, atol, rtol)
                    * self.error_const2[order];
                ncontrib += 1;
            }
        }
        if sens_in_error_control {
            let sens_atol = self.s_op.as_ref().unwrap().eqn().atol().unwrap();
            let sens_rtol = self.s_op.as_ref().unwrap().eqn().rtol().unwrap();
            for i in 0..state.sdiff.len() {
                error_norm += state.sdiff[i].column(order + 1).squared_norm(
                    &state.s[i],
                    sens_atol,
                    sens_rtol,
                ) * self.error_const2[order];
            }
            ncontrib += state.sdiff.len();
        }
        if sens_output_in_error_control {
            let rtol = self.s_op.as_ref().unwrap().eqn().out_rtol().unwrap();
            let atol = self.s_op.as_ref().unwrap().eqn().out_atol().unwrap();
            for i in 0..state.sgdiff.len() {
                error_norm +=
                    state.sgdiff[i]
                        .column(order + 1)
                        .squared_norm(&state.sg[i], atol, rtol)
                        * self.error_const2[order];
            }
            ncontrib += state.sgdiff.len();
        }
        if ncontrib == 0 {
            error_norm
        } else {
            error_norm / Eqn::T::from(ncontrib as f64)
        }
    }

    fn sensitivity_solve(&mut self, t_new: Eqn::T) -> Result<(), DiffsolError> {
        let order = self.state.order;

        // update for new state
        if let Some(op) = self.op.as_ref() {
            let s_op = self.s_op.as_mut().unwrap();
            let dy_new = op.tmp();
            let y_new = &self.y_predict;
            s_op.eqn_mut().update_rhs_out_state(y_new, &dy_new, t_new);
        }

        // solve for sensitivities equations discretised using BDF
        let naug = self.s_op.as_mut().unwrap().eqn().max_index();
        for i in 0..naug {
            // setup
            let s_op = self.s_op.as_mut().unwrap();
            {
                let state = &self.state;
                // predict forward to new step
                Self::_predict_using_diff(&mut self.s_predict, &state.sdiff[i], order);

                // setup op
                s_op.set_psi_and_y0(
                    &state.sdiff[i],
                    self.gamma.as_slice(),
                    self.alpha.as_slice(),
                    order,
                    &self.s_predict,
                );
                s_op.eqn_mut().set_index(i);
            }

            // solve
            {
                let s_new = &mut self.state.s[i];
                s_new.copy_from(&self.s_predict);
                // todo: should be a separate convergence object?
                self.nonlinear_solver.solve_in_place(
                    &*s_op,
                    s_new,
                    t_new,
                    &self.s_predict,
                    &mut self.convergence,
                )?;
                self.statistics.number_of_nonlinear_solver_iterations += self.convergence.niter();
                let s_new = &*s_new;
                self.s_deltas[i].copy_from(s_new);
                self.s_deltas[i] -= &self.s_predict;
            }

            if s_op.eqn().out().is_some() && s_op.eqn().include_out_in_error_control() {
                self.calculate_sens_output_delta(i);
            }
        }
        Ok(())
    }
}

impl<'a, M, Eqn, Nls, AugmentedEqn> OdeSolverMethod<'a, Eqn> for Bdf<'a, Eqn, Nls, M, AugmentedEqn>
where
    Eqn: OdeEquationsImplicit,
    AugmentedEqn: AugmentedOdeEquations<Eqn> + OdeEquationsImplicit,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V>,
    Eqn::V: DefaultDenseMatrix,
    Nls: NonLinearSolver<Eqn::M>,
    for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
    for<'b> &'b Eqn::M: MatrixRef<Eqn::M>,
{
    type State = BdfState<Eqn::V, M>;

    fn order(&self) -> usize {
        self.state.order
    }

    fn jacobian(&self) -> Option<Ref<<Eqn>::M>> {
        if let Some(op) = self.op.as_ref() {
            Some(op.rhs_jac())
        } else {
            self.s_op.as_ref().map(|s_op| s_op.rhs_jac())
        }
    }

    fn set_state(&mut self, state: Self::State) {
        let old_order = self.state.order;
        self.state = state;

        if let Some(op) = self.op.as_mut() {
            op.set_c(self.state.h, self.alpha[self.state.order]);
        }

        // order might have changed
        if self.state.order != old_order {
            self.u = Self::_compute_r(self.state.order, Eqn::T::one());
        }

        // reinitialise jacobian updates as if a checkpoint was taken
        self._jacobian_updates(
            self.state.h * self.alpha[self.state.order],
            SolverState::Checkpoint,
        );
    }

    fn interpolate(&self, t: Eqn::T) -> Result<Eqn::V, DiffsolError> {
        // state must be set
        let state = &self.state;
        if self.is_state_modified {
            if t == state.t {
                return Ok(state.y.clone());
            } else {
                return Err(ode_solver_error!(InterpolationTimeOutsideCurrentStep));
            }
        }
        // check that t is before/after the current time depending on the direction
        let is_forward = state.h > Eqn::T::zero();
        if (is_forward && t > state.t) || (!is_forward && t < state.t) {
            return Err(ode_solver_error!(InterpolationTimeAfterCurrentTime));
        }
        Ok(Self::interpolate_from_diff(
            t,
            &state.diff,
            state.t,
            state.h,
            state.order,
        ))
    }

    fn interpolate_out(&self, t: Eqn::T) -> Result<Eqn::V, DiffsolError> {
        // state must be set
        let state = &self.state;
        if self.is_state_modified {
            if t == state.t {
                return Ok(state.g.clone());
            } else {
                return Err(ode_solver_error!(InterpolationTimeOutsideCurrentStep));
            }
        }
        // check that t is before/after the current time depending on the direction
        let is_forward = state.h > Eqn::T::zero();
        if (is_forward && t > state.t) || (!is_forward && t < state.t) {
            return Err(ode_solver_error!(InterpolationTimeAfterCurrentTime));
        }
        Ok(Self::interpolate_from_diff(
            t,
            &state.gdiff,
            state.t,
            state.h,
            state.order,
        ))
    }

    fn interpolate_sens(&self, t: <Eqn as Op>::T) -> Result<Vec<Eqn::V>, DiffsolError> {
        // state must be set
        let state = &self.state;
        if self.is_state_modified {
            if t == state.t {
                return Ok(state.s.clone());
            } else {
                return Err(ode_solver_error!(InterpolationTimeOutsideCurrentStep));
            }
        }
        // check that t is before/after the current time depending on the direction
        let is_forward = state.h > Eqn::T::zero();
        if (is_forward && t > state.t) || (!is_forward && t < state.t) {
            return Err(ode_solver_error!(InterpolationTimeAfterCurrentTime));
        }

        let mut s = Vec::with_capacity(state.s.len());
        for i in 0..state.s.len() {
            s.push(Self::interpolate_from_diff(
                t,
                &state.sdiff[i],
                state.t,
                state.h,
                state.order,
            ));
        }
        Ok(s)
    }

    fn problem(&self) -> &'a OdeSolverProblem<Eqn> {
        self.ode_problem
    }

    fn state(&self) -> StateRef<Eqn::V> {
        self.state.as_ref()
    }

    fn into_state(self) -> BdfState<Eqn::V, M> {
        self.state
    }

    fn state_mut(&mut self) -> StateRefMut<Eqn::V> {
        self.is_state_modified = true;
        self.state.as_mut()
    }

    fn checkpoint(&mut self) -> Self::State {
        self._jacobian_updates(
            self.state.h * self.alpha[self.state.order],
            SolverState::Checkpoint,
        );
        self.state.clone()
    }

    fn step(&mut self) -> Result<OdeSolverStopReason<Eqn::T>, DiffsolError> {
        let mut safety: Eqn::T;
        let mut error_norm: Eqn::T;
        let problem = self.ode_problem;
        let integrate_out = problem.integrate_out;
        let output_in_error_control = problem.output_in_error_control();
        let integrate_sens = self.s_op.is_some();

        let mut convergence_fail = false;

        if self.is_state_modified {
            // reinitalise root finder if needed
            if let Some(root_fn) = problem.eqn.root() {
                let state = &self.state;
                self.root_finder
                    .as_ref()
                    .unwrap()
                    .init(&root_fn, &state.y, state.t);
            }
            // reinitialise diff matrix
            self.initialise_to_first_order();

            // reinitialise tstop if needed
            if let Some(t_stop) = self.tstop {
                self.set_stop_time(t_stop)?;
            }
        }

        self._predict_forward();

        // loop until step is accepted
        loop {
            let order = self.state.order;
            self.y_delta.copy_from(&self.y_predict);

            // solve BDF equation using y0 as starting point
            let mut solve_result = Ok(());
            if let Some(op) = self.op.as_ref() {
                solve_result = self.nonlinear_solver.solve_in_place(
                    op,
                    &mut self.y_delta,
                    self.t_predict,
                    &self.y_predict,
                    &mut self.convergence,
                );
                // update statistics
                self.statistics.number_of_nonlinear_solver_iterations += self.convergence.niter();

                if solve_result.is_ok() {
                    // test error is within tolerance
                    // combine eq 3, 4 and 6 from [1] to obtain error
                    // Note that error = C_k * h^{k+1} y^{k+1}
                    // and d = D^{k+1} y_{n+1} \approx h^{k+1} y^{k+1}
                    self.y_delta -= &self.y_predict;

                    // deal with output equations
                    if integrate_out && output_in_error_control {
                        self.calculate_output_delta();
                    }
                }
            }

            // only calculate sensitivities if solve was successful
            if solve_result.is_ok()
                && integrate_sens
                && self.sensitivity_solve(self.t_predict).is_err()
            {
                solve_result = Err(ode_solver_error!(SensitivitySolveFailed));
            }

            // handle case where either nonlinear solve failed
            if solve_result.is_err() {
                self.statistics.number_of_nonlinear_solver_fails += 1;
                if convergence_fail {
                    // newton iteration did not converge, but jacobian has already been
                    // evaluated so reduce step size by 0.3 (as per [1]) and try again
                    let new_h = self._update_step_size(Eqn::T::from(0.3))?;
                    self._jacobian_updates(
                        new_h * self.alpha[order],
                        SolverState::SecondConvergenceFail,
                    );

                    // new prediction
                    self._predict_forward();

                    // update statistics
                } else {
                    // newton iteration did not converge, so update jacobian and try again
                    self._jacobian_updates(
                        self.state.h * self.alpha[order],
                        SolverState::FirstConvergenceFail,
                    );
                    convergence_fail = true;
                    // same prediction as last time
                }
                continue;
            }

            error_norm = self.error_control();

            // need to caulate safety even if step is accepted
            let maxiter = self.convergence.max_iter() as f64;
            let niter = self.convergence.niter() as f64;
            safety = Eqn::T::from(0.9 * (2.0 * maxiter + 1.0) / (2.0 * maxiter + niter));

            // do the error test
            if error_norm <= Eqn::T::from(1.0) {
                // step is accepted
                break;
            } else {
                // step is rejected
                // calculate optimal step size factor as per eq 2.46 of [2]
                // and reduce step size and try again
                let mut factor = safety * error_norm.pow(Eqn::T::from(-0.5 / (order as f64 + 1.0)));
                if factor < Eqn::T::from(Self::MIN_FACTOR) {
                    factor = Eqn::T::from(Self::MIN_FACTOR);
                }
                let new_h = self._update_step_size(factor)?;
                self._jacobian_updates(new_h * self.alpha[order], SolverState::ErrorTestFail);

                // new prediction
                self._predict_forward();

                // update statistics
                self.statistics.number_of_error_test_failures += 1;
            }
        }

        // take the accepted step
        self.update_differences_and_integrate_out();

        {
            let state = &mut self.state;
            state.y.copy_from(&self.y_predict);
            state.t = self.t_predict;
            state.dy.copy_from_view(&state.diff.column(1));
            state.dy *= scale(Eqn::T::one() / state.h);
        }

        // update statistics
        if let Some(op) = self.op.as_ref() {
            self.statistics.number_of_linear_solver_setups = op.number_of_jac_evals();
        } else if let Some(s_op) = self.s_op.as_ref() {
            self.statistics.number_of_linear_solver_setups = s_op.number_of_jac_evals();
        }
        self.statistics.number_of_steps += 1;
        self.jacobian_update.step();

        // a change in order is only done after running at order k for k + 1 steps
        // (see page 83 of [2])
        self.n_equal_steps += 1;

        if self.n_equal_steps > self.state.order {
            let factors = {
                let order = self.state.order;
                // similar to the optimal step size factor we calculated above for the current
                // order k, we need to calculate the optimal step size factors for orders
                // k-1 and k+1. To do this, we note that the error = C_k * D^{k+1} y_n
                let error_m_norm = if order > 1 {
                    self.predict_error_control(order - 1)
                } else {
                    Eqn::T::INFINITY
                };
                let error_p_norm = if order < BdfState::<Eqn::V, M>::MAX_ORDER {
                    self.predict_error_control(order + 1)
                } else {
                    Eqn::T::INFINITY
                };

                let error_norms = [error_m_norm, error_norm, error_p_norm];
                error_norms
                    .into_iter()
                    .enumerate()
                    .map(|(i, error_norm)| {
                        error_norm.pow(Eqn::T::from(-0.5 / (i as f64 + order as f64)))
                    })
                    .collect::<Vec<_>>()
            };

            // now we have the three factors for orders k-1, k and k+1, pick the maximum in
            // order to maximise the resultant step size
            let max_index = factors
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;

            // update order and update the U matrix
            let order = {
                let old_order = self.state.order;
                let new_order = match max_index {
                    0 => old_order - 1,
                    1 => old_order,
                    2 => old_order + 1,
                    _ => unreachable!(),
                };
                self.state.order = new_order;
                if max_index != 1 {
                    self.u = Self::_compute_r(new_order, Eqn::T::one());
                }
                new_order
            };

            let mut factor = safety * factors[max_index];
            if factor > Eqn::T::from(Self::MAX_FACTOR) {
                factor = Eqn::T::from(Self::MAX_FACTOR);
            }
            if factor < Eqn::T::from(Self::MIN_FACTOR) {
                factor = Eqn::T::from(Self::MIN_FACTOR);
            }
            if factor >= Eqn::T::from(Self::MAX_THRESHOLD)
                || factor < Eqn::T::from(Self::MIN_THRESHOLD)
                || max_index == 0
                || max_index == 2
            {
                let new_h = self._update_step_size(factor)?;
                self._jacobian_updates(new_h * self.alpha[order], SolverState::StepSuccess);
            }
        }

        // check for root within accepted step
        if let Some(root_fn) = self.ode_problem.eqn.root() {
            let ret = self.root_finder.as_ref().unwrap().check_root(
                &|t: <Eqn as Op>::T| self.interpolate(t),
                &root_fn,
                self.state.as_ref().y,
                self.state.as_ref().t,
            );
            if let Some(root) = ret {
                return Ok(OdeSolverStopReason::RootFound(root));
            }
        }

        if let Some(tstop) = self.tstop {
            if let Some(reason) = self.handle_tstop(tstop).unwrap() {
                return Ok(reason);
            }
        }

        // just a normal step, no roots or tstop reached
        Ok(OdeSolverStopReason::InternalTimestep)
    }

    fn set_stop_time(&mut self, tstop: <Eqn as Op>::T) -> Result<(), DiffsolError> {
        self.tstop = Some(tstop);
        if let Some(OdeSolverStopReason::TstopReached) = self.handle_tstop(tstop)? {
            let error = OdeSolverError::StopTimeAtCurrentTime;
            self.tstop = None;
            return Err(DiffsolError::from(error));
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::{
        ode_solver::{
            test_models::{
                dydt_y2::dydt_y2_problem,
                exponential_decay::{
                    exponential_decay_problem, exponential_decay_problem_adjoint,
                    exponential_decay_problem_diffsl, exponential_decay_problem_sens,
                    exponential_decay_problem_with_root, negative_exponential_decay_problem,
                },
                exponential_decay_with_algebraic::{
                    exponential_decay_with_algebraic_adjoint_problem,
                    exponential_decay_with_algebraic_problem,
                    exponential_decay_with_algebraic_problem_diffsl,
                    exponential_decay_with_algebraic_problem_sens,
                },
                foodweb::foodweb_problem,
                gaussian_decay::gaussian_decay_problem,
                heat2d::head2d_problem,
                robertson::{robertson, robertson_sens},
                robertson_ode::robertson_ode,
                robertson_ode_with_sens::robertson_ode_with_sens,
            },
            tests::{
                setup_test_adjoint, setup_test_adjoint_sum_squares, test_adjoint,
                test_adjoint_sum_squares, test_checkpointing, test_interpolate, test_ode_solver,
                test_problem, test_state_mut, test_state_mut_on_problem,
            },
        },
        FaerLU, FaerSparseLU, OdeEquations, OdeSolverMethod, Op, SparseColMat, Vector,
    };

    use num_traits::abs;

    type M = nalgebra::DMatrix<f64>;
    type LS = crate::NalgebraLU<f64>;
    #[test]
    fn bdf_state_mut() {
        test_state_mut(test_problem::<M>().bdf::<LS>().unwrap());
    }

    #[test]
    fn bdf_test_interpolate() {
        test_interpolate(test_problem::<M>().bdf::<LS>().unwrap());
    }

    #[test]
    fn bdf_test_state_mut_exponential_decay() {
        let (p, soln) = exponential_decay_problem::<M>(false);
        let s = p.bdf_solver::<LS>(p.bdf_state::<LS>().unwrap()).unwrap();
        test_state_mut_on_problem(s, soln);
    }

    #[test]
    fn bdf_test_nalgebra_negative_exponential_decay() {
        let (problem, soln) = negative_exponential_decay_problem::<M>(false);
        let mut s = problem.bdf::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[test]
    fn bdf_test_nalgebra_exponential_decay() {
        let (problem, soln) = exponential_decay_problem::<M>(false);
        let mut s = problem.bdf::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 11
        number_of_steps: 47
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 82
        number_of_nonlinear_solver_fails: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 84
        number_of_jac_muls: 2
        number_of_matrix_evals: 1
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn bdf_test_faer_sparse_exponential_decay() {
        let (problem, soln) = exponential_decay_problem::<SparseColMat<f64>>(false);
        let mut s = problem.bdf::<FaerSparseLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[test]
    fn bdf_test_checkpointing() {
        let (problem, soln) = exponential_decay_problem::<M>(false);
        let solver1 = problem.bdf::<LS>().unwrap();
        let solver2 = problem.bdf::<LS>().unwrap();
        test_checkpointing(soln, solver1, solver2);
    }

    #[test]
    fn bdf_test_faer_exponential_decay() {
        type M = faer::Mat<f64>;
        type LS = FaerLU<f64>;
        let (problem, soln) = exponential_decay_problem::<M>(false);
        let mut s = problem.bdf::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 11
        number_of_steps: 47
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 82
        number_of_nonlinear_solver_fails: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 84
        number_of_jac_muls: 2
        number_of_matrix_evals: 1
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn bdf_test_nalgebra_exponential_decay_sens() {
        let (problem, soln) = exponential_decay_problem_sens::<M>(false);
        let mut s = problem.bdf_sens::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, true);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 13
        number_of_steps: 48
        number_of_error_test_failures: 1
        number_of_nonlinear_solver_iterations: 234
        number_of_nonlinear_solver_fails: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.statistics(), @r###"
        number_of_calls: 89
        number_of_jac_muls: 151
        number_of_matrix_evals: 1
        number_of_jac_adj_muls: 0
        "###);
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn bdf_test_nalgebra_exponential_decay_diffsl_sens() {
        let (_problem, mut soln) = exponential_decay_problem_sens::<M>(false);
        let (problem, _soln) = exponential_decay_problem_diffsl::<M, diffsl::LlvmModule>(false);
        soln.atol = problem.atol.clone();
        soln.rtol = problem.rtol;
        let mut s = problem.bdf_sens::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, true);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 13
        number_of_steps: 48
        number_of_error_test_failures: 1
        number_of_nonlinear_solver_iterations: 234
        number_of_nonlinear_solver_fails: 0
        "###);
    }

    #[test]
    fn bdf_test_nalgebra_exponential_decay_adjoint() {
        let (mut problem, soln) = exponential_decay_problem_adjoint::<M>(true);
        let final_time = soln.solution_points.last().unwrap().t;
        let dgdu = setup_test_adjoint::<LS, _>(&mut problem, soln);
        let (problem, _soln) = exponential_decay_problem_adjoint::<M>(true);
        let mut s = problem.bdf::<LS>().unwrap();
        let (checkpointer, _y, _t) = s.solve_with_checkpointing(final_time, None).unwrap();
        let adjoint_solver = problem
            .bdf_solver_adjoint::<LS, _>(checkpointer, Some(dgdu.ncols()))
            .unwrap();
        test_adjoint(adjoint_solver, dgdu);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 183
        number_of_jac_muls: 6
        number_of_matrix_evals: 3
        number_of_jac_adj_muls: 392
        "###);
    }

    #[test]
    fn bdf_test_nalgebra_exponential_decay_adjoint_sum_squares() {
        let (mut problem, soln) = exponential_decay_problem_adjoint::<M>(false);
        let times = soln.solution_points.iter().map(|p| p.t).collect::<Vec<_>>();
        let (dgdp, data) = setup_test_adjoint_sum_squares::<LS, _>(&mut problem, times.as_slice());
        let (problem, _soln) = exponential_decay_problem_adjoint::<M>(false);
        let mut s = problem.bdf::<LS>().unwrap();
        let (checkpointer, soln) = s
            .solve_dense_with_checkpointing(times.as_slice(), None)
            .unwrap();
        let adjoint_solver = problem
            .bdf_solver_adjoint::<LS, _>(checkpointer, Some(dgdp.ncols()))
            .unwrap();
        test_adjoint_sum_squares(adjoint_solver, dgdp, soln, data, times.as_slice());
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 183
        number_of_jac_muls: 6
        number_of_matrix_evals: 3
        number_of_jac_adj_muls: 392
        "###);
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn bdf_test_nalgebra_exponential_decay_adjoint_diffsl() {
        let (mut problem, soln) = exponential_decay_problem_diffsl::<M, diffsl::LlvmModule>(true);
        let final_time = soln.solution_points.last().unwrap().t;
        let dgdu = setup_test_adjoint::<LS, _>(&mut problem, soln);
        let mut s = problem.bdf::<LS>().unwrap();
        let (checkpointer, _y, _t) = s.solve_with_checkpointing(final_time, None).unwrap();
        let adjoint_solver = problem
            .bdf_solver_adjoint::<LS, _>(checkpointer, Some(dgdu.ncols()))
            .unwrap();
        test_adjoint(adjoint_solver, dgdu);
    }

    #[test]
    fn bdf_test_nalgebra_exponential_decay_algebraic_adjoint() {
        let (mut problem, soln) = exponential_decay_with_algebraic_adjoint_problem::<M>(true);
        let final_time = soln.solution_points.last().unwrap().t;
        let dgdu = setup_test_adjoint::<LS, _>(&mut problem, soln);
        let (problem, _soln) = exponential_decay_with_algebraic_adjoint_problem::<M>(true);
        let mut s = problem.bdf::<LS>().unwrap();
        let (checkpointer, _y, _t) = s.solve_with_checkpointing(final_time, None).unwrap();
        let adjoint_solver = problem
            .bdf_solver_adjoint::<LS, _>(checkpointer, Some(dgdu.ncols()))
            .unwrap();
        test_adjoint(adjoint_solver, dgdu);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 279
        number_of_jac_muls: 24
        number_of_matrix_evals: 8
        number_of_jac_adj_muls: 187
        "###);
    }

    #[test]
    fn bdf_test_nalgebra_exponential_decay_algebraic_adjoint_sum_squares() {
        let (mut problem, soln) = exponential_decay_with_algebraic_adjoint_problem::<M>(false);
        let times = soln.solution_points.iter().map(|p| p.t).collect::<Vec<_>>();
        let (dgdp, data) = setup_test_adjoint_sum_squares::<LS, _>(&mut problem, times.as_slice());
        let (problem, _soln) = exponential_decay_with_algebraic_adjoint_problem::<M>(false);
        let mut s = problem.bdf::<LS>().unwrap();
        let (checkpointer, soln) = s
            .solve_dense_with_checkpointing(times.as_slice(), None)
            .unwrap();
        let adjoint_solver = problem
            .bdf_solver_adjoint::<LS, _>(checkpointer, Some(dgdp.ncols()))
            .unwrap();
        test_adjoint_sum_squares(adjoint_solver, dgdp, soln, data, times.as_slice());
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 183
        number_of_jac_muls: 6
        number_of_matrix_evals: 3
        number_of_jac_adj_muls: 392
        "###);
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn bdf_test_nalgebra_exponential_decay_with_algebraic_adjoint_diffsl() {
        let (mut problem, soln) =
            exponential_decay_with_algebraic_problem_diffsl::<M, diffsl::LlvmModule>(true);
        let final_time = soln.solution_points.last().unwrap().t;
        let dgdu = setup_test_adjoint::<LS, _>(&mut problem, soln);
        let mut s = problem.bdf::<LS>().unwrap();
        let (checkpointer, _y, _t) = s.solve_with_checkpointing(final_time, None).unwrap();
        let adjoint_solver = problem
            .bdf_solver_adjoint::<LS, _>(checkpointer, None)
            .unwrap();
        test_adjoint(adjoint_solver, dgdu);
    }

    #[test]
    fn test_bdf_nalgebra_exponential_decay_algebraic() {
        let (problem, soln) = exponential_decay_with_algebraic_problem::<M>(false);
        let mut s = problem.bdf::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 20
        number_of_steps: 41
        number_of_error_test_failures: 4
        number_of_nonlinear_solver_iterations: 79
        number_of_nonlinear_solver_fails: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 83
        number_of_jac_muls: 6
        number_of_matrix_evals: 2
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn bdf_test_faer_sparse_exponential_decay_algebraic() {
        let (problem, soln) = exponential_decay_with_algebraic_problem::<SparseColMat<f64>>(false);
        let mut s = problem.bdf::<FaerSparseLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[test]
    fn test_bdf_nalgebra_exponential_decay_algebraic_sens() {
        let (problem, soln) = exponential_decay_with_algebraic_problem_sens::<M>();
        let mut s = problem.bdf_sens::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, true);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 18
        number_of_steps: 43
        number_of_error_test_failures: 3
        number_of_nonlinear_solver_iterations: 155
        number_of_nonlinear_solver_fails: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 71
        number_of_jac_muls: 100
        number_of_matrix_evals: 3
        number_of_jac_adj_muls: 0
        "###);
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn bdf_test_nalgebra_exponential_decay_algebraic_diffsl_sens() {
        let (_problem, mut soln) = exponential_decay_with_algebraic_problem_sens::<M>();
        let (problem, _soln) =
            exponential_decay_with_algebraic_problem_diffsl::<M, diffsl::LlvmModule>(false);
        soln.atol = problem.atol.clone();
        soln.rtol = problem.rtol;
        let mut s = problem.bdf_sens::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, true);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 18
        number_of_steps: 43
        number_of_error_test_failures: 3
        number_of_nonlinear_solver_iterations: 155
        number_of_nonlinear_solver_fails: 0
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_robertson() {
        let (problem, soln) = robertson::<M>(false);
        let mut s = problem.bdf::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 77
        number_of_steps: 316
        number_of_error_test_failures: 3
        number_of_nonlinear_solver_iterations: 722
        number_of_nonlinear_solver_fails: 19
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 725
        number_of_jac_muls: 60
        number_of_matrix_evals: 20
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn bdf_test_faer_sparse_robertson() {
        let (problem, soln) = robertson::<SparseColMat<f64>>(false);
        let mut s = problem.bdf::<FaerSparseLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[cfg(feature = "suitesparse")]
    #[test]
    fn bdf_test_faer_sparse_ku_robertson() {
        let (problem, soln) = robertson::<SparseColMat<f64>>(false);
        let mut s = problem.bdf::<crate::KLU<_>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn bdf_test_nalgebra_diffsl_robertson() {
        use diffsl::LlvmModule;

        use crate::ode_solver::test_models::robertson;
        let (problem, soln) = robertson::robertson_diffsl_problem::<M, LlvmModule>();
        let mut s = problem.bdf::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[test]
    fn test_bdf_nalgebra_robertson_sens() {
        let (problem, soln) = robertson_sens::<M>();
        let mut s = problem.bdf_sens::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, true);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 160
        number_of_steps: 410
        number_of_error_test_failures: 4
        number_of_nonlinear_solver_iterations: 3107
        number_of_nonlinear_solver_fails: 81
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 996
        number_of_jac_muls: 2495
        number_of_matrix_evals: 71
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_robertson_colored() {
        let (problem, soln) = robertson::<M>(true);
        let mut s = problem.bdf::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 77
        number_of_steps: 316
        number_of_error_test_failures: 3
        number_of_nonlinear_solver_iterations: 722
        number_of_nonlinear_solver_fails: 19
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 725
        number_of_jac_muls: 63
        number_of_matrix_evals: 20
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_robertson_ode() {
        let (problem, soln) = robertson_ode::<M>(false, 3);
        let mut s = problem.bdf::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 86
        number_of_steps: 416
        number_of_error_test_failures: 1
        number_of_nonlinear_solver_iterations: 911
        number_of_nonlinear_solver_fails: 15
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 913
        number_of_jac_muls: 162
        number_of_matrix_evals: 18
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_robertson_ode_sens() {
        let (problem, soln) = robertson_ode_with_sens::<M>(false);
        let mut s = problem.bdf_sens::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, true);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 152
        number_of_steps: 512
        number_of_error_test_failures: 5
        number_of_nonlinear_solver_iterations: 3779
        number_of_nonlinear_solver_fails: 70
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 1157
        number_of_jac_muls: 2930
        number_of_matrix_evals: 54
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_dydt_y2() {
        let (problem, soln) = dydt_y2_problem::<M>(false, 10);
        let mut s = problem.bdf::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 27
        number_of_steps: 161
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 355
        number_of_nonlinear_solver_fails: 3
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 357
        number_of_jac_muls: 50
        number_of_matrix_evals: 5
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_dydt_y2_colored() {
        let (problem, soln) = dydt_y2_problem::<M>(true, 10);
        let mut s = problem.bdf::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 27
        number_of_steps: 161
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 355
        number_of_nonlinear_solver_fails: 3
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 357
        number_of_jac_muls: 15
        number_of_matrix_evals: 5
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_bdf_nalgebra_gaussian_decay() {
        let (problem, soln) = gaussian_decay_problem::<M>(false, 10);
        let mut s = problem.bdf::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 14
        number_of_steps: 66
        number_of_error_test_failures: 1
        number_of_nonlinear_solver_iterations: 130
        number_of_nonlinear_solver_fails: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 132
        number_of_jac_muls: 20
        number_of_matrix_evals: 2
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_bdf_faer_sparse_heat2d() {
        let (problem, soln) = head2d_problem::<SparseColMat<f64>, 10>();
        let mut s = problem.bdf::<FaerSparseLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 21
        number_of_steps: 167
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 330
        number_of_nonlinear_solver_fails: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 333
        number_of_jac_muls: 128
        number_of_matrix_evals: 4
        number_of_jac_adj_muls: 0
        "###);
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn test_bdf_faer_sparse_heat2d_diffsl() {
        use diffsl::LlvmModule;

        use crate::ode_solver::test_models::heat2d;
        let (problem, soln) = heat2d::heat2d_diffsl_problem::<SparseColMat<f64>, LlvmModule, 10>();
        let mut s = problem.bdf::<FaerSparseLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[test]
    fn test_bdf_faer_sparse_foodweb() {
        let (problem, soln) = foodweb_problem::<SparseColMat<f64>, 10>();
        let mut s = problem.bdf::<FaerSparseLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 45
        number_of_steps: 161
        number_of_error_test_failures: 2
        number_of_nonlinear_solver_iterations: 355
        number_of_nonlinear_solver_fails: 14
        "###);
    }

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn test_bdf_faer_sparse_foodweb_diffsl() {
        use diffsl::LlvmModule;

        use crate::ode_solver::test_models::foodweb;
        let (problem, soln) =
            foodweb::foodweb_diffsl_problem::<SparseColMat<f64>, LlvmModule, 10>();
        let mut s = problem.bdf::<FaerSparseLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[test]
    fn test_tstop_bdf() {
        let (problem, soln) = exponential_decay_problem::<M>(false);
        let mut s = problem.bdf::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, true, false);
    }

    #[test]
    fn test_root_finder_bdf() {
        let (problem, soln) = exponential_decay_problem_with_root::<M>(false);
        let mut s = problem.bdf::<LS>().unwrap();
        let y = test_ode_solver(&mut s, soln, None, false, false);
        assert!(abs(y[0] - 0.6) < 1e-6, "y[0] = {}", y[0]);
    }

    #[test]
    fn test_param_sweep_bdf() {
        let (mut problem, _soln) = exponential_decay_problem::<M>(false);
        let mut ps = Vec::new();
        for y0 in (1..10).map(f64::from) {
            ps.push(nalgebra::DVector::<f64>::from_vec(vec![0.1, y0]));
        }

        let mut old_soln: Option<nalgebra::DVector<f64>> = None;
        for p in ps {
            problem.eqn_mut().set_params(&p);
            let mut s = problem.bdf::<LS>().unwrap();
            let (ys, _ts) = s.solve(10.0).unwrap();
            // check that the new solution is different from the old one
            if let Some(old_soln) = &mut old_soln {
                let new_soln = ys.column(ys.ncols() - 1).into_owned();
                let error = new_soln - &*old_soln;
                let diff = error
                    .squared_norm(old_soln, &problem.atol, problem.rtol)
                    .sqrt();
                assert!(diff > 1.0e-6, "diff: {}", diff);
            }
            old_soln = Some(ys.column(ys.ncols() - 1).into_owned());
        }
    }

    #[cfg(feature = "diffsl")]
    #[test]
    fn test_ball_bounce_bdf() {
        use crate::ode_solver::tests::test_ball_bounce_problem;
        type M = nalgebra::DMatrix<f64>;
        type LS = crate::NalgebraLU<f64>;
        let (x, v, t) = crate::ode_solver::tests::test_ball_bounce(
            test_ball_bounce_problem::<M>().bdf::<LS>().unwrap(),
        );

        let expected_x = [
            0.003751514915514589,
            0.00750117409999241,
            0.015370589755655079,
        ];
        let expected_v = [11.202428570923361, 11.19914432101355, 11.192247396202946];
        let expected_t = [1.4281779078441663, 1.4285126937676944, 1.4292157442071036];
        for (i, ((x, v), t)) in x.iter().zip(v.iter()).zip(t.iter()).enumerate() {
            assert!((x - expected_x[i]).abs() < 1e-4);
            assert!((v - expected_v[i]).abs() < 1e-4);
            assert!((t - expected_t[i]).abs() < 1e-4);
        }
    }
}
