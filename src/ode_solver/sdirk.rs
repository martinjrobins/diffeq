use crate::error::DiffsolError;
use crate::error::OdeSolverError;
use crate::matrix::MatrixRef;
use crate::ode_solver_error;
use crate::vector::VectorRef;
use crate::LinearSolver;
use crate::NewtonNonlinearSolver;
use crate::NoAug;
use crate::OdeSolverStopReason;
use crate::RootFinder;
use crate::SdirkState;
use crate::Tableau;
use crate::{
    nonlinear_solver::NonLinearSolver, op::sdirk::SdirkCallable, scale, AugmentedOdeEquations,
    AugmentedOdeEquationsImplicit, Convergence, DefaultDenseMatrix, DenseMatrix, JacobianUpdate,
    NonLinearOp, OdeEquationsImplicit, OdeSolverMethod, OdeSolverProblem, OdeSolverState, Op,
    Scalar, StateRef, StateRefMut, Vector, VectorViewMut,
};
use num_traits::abs;
use num_traits::One;
use num_traits::Pow;
use num_traits::Zero;
use std::ops::MulAssign;

use super::bdf::BdfStatistics;
use super::jacobian_update::SolverState;
use super::method::AugmentedOdeSolverMethod;

impl<'a, M, Eqn, LS, AugEqn> AugmentedOdeSolverMethod<'a, Eqn, AugEqn>
    for Sdirk<'a, Eqn, LS, M, AugEqn>
where
    Eqn: OdeEquationsImplicit,
    AugEqn: AugmentedOdeEquationsImplicit<Eqn>,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V>,
    LS: LinearSolver<Eqn::M>,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T>,
    for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
    for<'b> &'b Eqn::M: MatrixRef<Eqn::M>,
{
    fn into_state_and_eqn(self) -> (Self::State, Option<AugEqn>) {
        (self.state, self.s_op.map(|op| op.eqn))
    }
    fn augmented_eqn(&self) -> Option<&AugEqn> {
        self.s_op.as_ref().map(|op| op.eqn())
    }
}

/// A singly diagonally implicit Runge-Kutta method. Can optionally have an explicit first stage for ESDIRK methods.
///
/// The particular method is defined by the [Tableau] used to create the solver.
/// If the `beta` matrix of the [Tableau] is present this is used for interpolation, otherwise hermite interpolation is used.
///
/// Restrictions:
/// - The upper triangular part of the `a` matrix must be zero (i.e. not fully implicit).
/// - The diagonal of the `a` matrix must be the same non-zero value for all rows (i.e. an SDIRK method), except for the first row which can be zero for ESDIRK methods.
/// - The last row of the `a` matrix must be the same as the `b` vector, and the last element of the `c` vector must be 1 (i.e. a stiffly accurate method)
pub struct Sdirk<
    'a,
    Eqn,
    LS,
    M = <<Eqn as Op>::V as DefaultDenseMatrix>::M,
    AugmentedEqn = NoAug<Eqn>,
> where
    M: DenseMatrix<T = Eqn::T, V = Eqn::V>,
    LS: LinearSolver<Eqn::M>,
    Eqn: OdeEquationsImplicit,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T>,
    AugmentedEqn: AugmentedOdeEquations<Eqn>,
{
    tableau: Tableau<M>,
    problem: &'a OdeSolverProblem<Eqn>,
    nonlinear_solver: NewtonNonlinearSolver<Eqn::M, LS>,
    convergence: Convergence<'a, Eqn::V>,
    op: Option<SdirkCallable<&'a Eqn>>,
    state: SdirkState<Eqn::V>,
    diff: M,
    sdiff: Vec<M>,
    sgdiff: Vec<M>,
    gdiff: M,
    old_g: Eqn::V,
    gamma: Eqn::T,
    is_sdirk: bool,
    s_op: Option<SdirkCallable<AugmentedEqn>>,
    old_t: Eqn::T,
    old_y: Eqn::V,
    old_y_sens: Vec<Eqn::V>,
    old_f: Eqn::V,
    old_f_sens: Vec<Eqn::V>,
    a_rows: Vec<Eqn::V>,
    statistics: BdfStatistics,
    root_finder: Option<RootFinder<Eqn::V>>,
    tstop: Option<Eqn::T>,
    is_state_mutated: bool,
    jacobian_update: JacobianUpdate<Eqn::T>,
}

impl<M, Eqn, LS, AugmentedEqn> Clone for Sdirk<'_, Eqn, LS, M, AugmentedEqn>
where
    M: DenseMatrix<T = Eqn::T, V = Eqn::V>,
    LS: LinearSolver<Eqn::M>,
    Eqn: OdeEquationsImplicit,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T>,
    AugmentedEqn: AugmentedOdeEquationsImplicit<Eqn>,
    for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
    for<'b> &'b Eqn::M: MatrixRef<Eqn::M>,
{
    fn clone(&self) -> Self {
        let problem = self.problem;
        let mut nonlinear_solver = NewtonNonlinearSolver::new(LS::default());
        let op = if let Some(op) = &self.op {
            let op = op.clone_state(&problem.eqn);
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
            tableau: self.tableau.clone(),
            problem: self.problem,
            convergence: self.convergence.clone(),
            nonlinear_solver,
            op,
            state: self.state.clone(),
            diff: self.diff.clone(),
            sdiff: self.sdiff.clone(),
            sgdiff: self.sgdiff.clone(),
            gdiff: self.gdiff.clone(),
            old_g: self.old_g.clone(),
            gamma: self.gamma,
            is_sdirk: self.is_sdirk,
            s_op,
            old_t: self.old_t,
            old_y: self.old_y.clone(),
            old_y_sens: self.old_y_sens.clone(),
            old_f: self.old_f.clone(),
            old_f_sens: self.old_f_sens.clone(),
            a_rows: self.a_rows.clone(),
            statistics: self.statistics.clone(),
            root_finder: self.root_finder.clone(),
            tstop: self.tstop,
            is_state_mutated: self.is_state_mutated,
            jacobian_update: self.jacobian_update.clone(),
        }
    }
}

impl<'a, M, Eqn, LS, AugmentedEqn> Sdirk<'a, Eqn, LS, M, AugmentedEqn>
where
    LS: LinearSolver<Eqn::M>,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V>,
    Eqn: OdeEquationsImplicit,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T>,
    AugmentedEqn: AugmentedOdeEquationsImplicit<Eqn>,
    for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
    for<'b> &'b Eqn::M: MatrixRef<Eqn::M>,
{
    const NEWTON_MAXITER: usize = 10;
    const MIN_FACTOR: f64 = 0.2;
    const MAX_FACTOR: f64 = 10.0;
    const MIN_TIMESTEP: f64 = 1e-13;

    pub fn new(
        problem: &'a OdeSolverProblem<Eqn>,
        state: SdirkState<Eqn::V>,
        tableau: Tableau<M>,
        linear_solver: LS,
    ) -> Result<Self, DiffsolError> {
        Self::_new(problem, state, tableau, linear_solver, true)
    }

    fn _new(
        problem: &'a OdeSolverProblem<Eqn>,
        mut state: SdirkState<Eqn::V>,
        tableau: Tableau<M>,
        linear_solver: LS,
        integrate_main_eqn: bool,
    ) -> Result<Self, DiffsolError> {
        // check that the upper triangular part of a is zero
        let s = tableau.s();
        for i in 0..s {
            for j in (i + 1)..s {
                assert_eq!(
                    tableau.a()[(i, j)],
                    Eqn::T::zero(),
                    "Invalid tableau, expected a(i, j) = 0 for i > j"
                );
            }
        }
        let gamma = tableau.a()[(1, 1)];
        //check that for i = 1..s-1, a(i, i) = gamma
        for i in 1..tableau.s() {
            assert_eq!(
                tableau.a()[(i, i)],
                gamma,
                "Invalid tableau, expected a(i, i) = gamma = {} for i = 1..s-1",
                gamma
            );
        }
        // if a(0, 0) = gamma, then we're a SDIRK method
        // if a(0, 0) = 0, then we're a ESDIRK method
        // otherwise, error
        let zero = Eqn::T::zero();
        if tableau.a()[(0, 0)] != zero && tableau.a()[(0, 0)] != gamma {
            panic!("Invalid tableau, expected a(0, 0) = 0 or a(0, 0) = gamma");
        }
        let is_sdirk = tableau.a()[(0, 0)] == gamma;

        let mut a_rows = Vec::with_capacity(s);
        for i in 0..s {
            let mut row = Vec::with_capacity(i);
            for j in 0..i {
                row.push(tableau.a()[(i, j)]);
            }
            a_rows.push(Eqn::V::from_vec(row));
        }

        // check last row of a is the same as b
        for i in 0..s {
            assert_eq!(
                tableau.a()[(s - 1, i)],
                tableau.b()[i],
                "Invalid tableau, expected a(s-1, i) = b(i)"
            );
        }

        // check that last c is 1
        assert_eq!(
            tableau.c()[s - 1],
            Eqn::T::one(),
            "Invalid tableau, expected c(s-1) = 1"
        );

        // check that the first c is 0 for esdirk methods
        if !is_sdirk {
            assert_eq!(
                tableau.c()[0],
                Eqn::T::zero(),
                "Invalid tableau, expected c(0) = 0 for esdirk methods"
            );
        }

        // setup linear solver for first step
        let mut jacobian_update = JacobianUpdate::default();
        jacobian_update.update_jacobian(state.h);
        jacobian_update.update_rhs_jacobian();

        let mut nonlinear_solver = NewtonNonlinearSolver::new(linear_solver);

        // set max iterations for nonlinear solver
        let mut convergence = Convergence::new(problem.rtol, &problem.atol);
        convergence.set_max_iter(Self::NEWTON_MAXITER);

        let op = if integrate_main_eqn {
            let callable = SdirkCallable::new(&problem.eqn, gamma);
            callable.set_h(state.h);
            nonlinear_solver.set_problem(&callable);
            nonlinear_solver.reset_jacobian(&callable, &state.y, state.t);
            Some(callable)
        } else {
            None
        };

        // update statistics
        let statistics = BdfStatistics::default();

        state.check_consistent_with_problem(problem)?;

        let nstates = state.y.len();
        let order = tableau.s();
        let diff = M::zeros(nstates, order);
        let gdiff_rows = if problem.integrate_out {
            problem.eqn.out().unwrap().nout()
        } else {
            0
        };
        let gdiff = M::zeros(gdiff_rows, order);

        let old_f = state.dy.clone();
        let old_t = state.t;
        let old_y = state.y.clone();
        let old_g = if problem.integrate_out {
            state.g.clone()
        } else {
            <Eqn::V as Vector>::zeros(0)
        };

        state.set_problem(problem)?;
        let root_finder = if let Some(root_fn) = problem.eqn.root() {
            let root_finder = RootFinder::new(root_fn.nout());
            root_finder.init(&root_fn, &state.y, state.t);
            Some(root_finder)
        } else {
            None
        };

        Ok(Self {
            old_y_sens: vec![],
            old_f_sens: vec![],
            old_g,
            convergence,
            diff,
            sdiff: vec![],
            sgdiff: vec![],
            tableau,
            nonlinear_solver,
            op,
            state,
            problem,
            s_op: None,
            gdiff,
            gamma,
            is_sdirk,
            old_t,
            old_y,
            a_rows,
            old_f,
            statistics,
            root_finder,
            tstop: None,
            is_state_mutated: false,
            jacobian_update,
        })
    }

    pub fn new_augmented(
        problem: &'a OdeSolverProblem<Eqn>,
        state: SdirkState<Eqn::V>,
        tableau: Tableau<M>,
        linear_solver: LS,
        augmented_eqn: AugmentedEqn,
    ) -> Result<Self, DiffsolError> {
        state.check_sens_consistent_with_problem(problem, &augmented_eqn)?;
        let mut ret = Self::_new(
            problem,
            state,
            tableau,
            linear_solver,
            augmented_eqn.integrate_main_eqn(),
        )?;
        let naug = augmented_eqn.max_index();
        let nstates = augmented_eqn.rhs().nstates();
        let order = ret.tableau.s();
        ret.sdiff = vec![M::zeros(nstates, order); naug];
        ret.old_f_sens = vec![<Eqn::V as Vector>::zeros(nstates); naug];
        ret.old_y_sens = ret.state.s.clone();
        if let Some(out) = augmented_eqn.out() {
            ret.sgdiff = vec![M::zeros(out.nout(), order); naug];
        }

        ret.s_op = if augmented_eqn.integrate_main_eqn() {
            Some(SdirkCallable::new_no_jacobian(augmented_eqn, ret.gamma))
        } else {
            let callable = SdirkCallable::new(augmented_eqn, ret.gamma);
            ret.nonlinear_solver.set_problem(&callable);
            ret.nonlinear_solver
                .reset_jacobian(&callable, &ret.state.s[0], ret.state.t);
            Some(callable)
        };
        Ok(ret)
    }

    pub fn get_statistics(&self) -> &BdfStatistics {
        &self.statistics
    }

    fn handle_tstop(
        &mut self,
        tstop: Eqn::T,
    ) -> Result<Option<OdeSolverStopReason<Eqn::T>>, DiffsolError> {
        let state = &mut self.state;

        // check if the we are at tstop
        let troundoff = Eqn::T::from(100.0) * Eqn::T::EPSILON * (abs(state.t) + abs(state.h));
        if abs(state.t - tstop) <= troundoff {
            self.tstop = None;
            return Ok(Some(OdeSolverStopReason::TstopReached));
        } else if (state.h > M::T::zero() && tstop < state.t - troundoff)
            || (state.h < M::T::zero() && tstop > state.t + troundoff)
        {
            return Err(DiffsolError::from(
                OdeSolverError::StopTimeBeforeCurrentTime {
                    stop_time: tstop.into(),
                    state_time: state.t.into(),
                },
            ));
        }

        // check if the next step will be beyond tstop, if so adjust the step size
        if (state.h > M::T::zero() && state.t + state.h > tstop + troundoff)
            || (state.h < M::T::zero() && state.t + state.h < tstop - troundoff)
        {
            let factor = (tstop - state.t) / state.h;
            state.h *= factor;
            if let Some(op) = self.op.as_mut() {
                op.set_h(state.h);
            }
            if let Some(s_op) = self.s_op.as_mut() {
                s_op.set_h(state.h);
            }
        }
        Ok(None)
    }

    fn predict_stage(i: usize, diff: &M, dy: &mut Eqn::V, tableau: &Tableau<M>) {
        if i == 0 {
            dy.fill(Eqn::T::zero());
        } else if i == 1 {
            dy.copy_from_view(&diff.column(i - 1));
        } else {
            let c =
                (tableau.c()[i] - tableau.c()[i - 2]) / (tableau.c()[i - 1] - tableau.c()[i - 2]);
            // dy = c1  + c * (c1 - c2)
            dy.copy_from_view(&diff.column(i - 1));
            dy.axpy_v(-c, &diff.column(i - 2), Eqn::T::one() + c);
        }
    }

    fn solve_for_sensitivities(&mut self, i: usize, t: Eqn::T) -> Result<(), DiffsolError> {
        let h = self.state.h;
        // update for new state
        {
            let op = self.s_op.as_mut().unwrap();
            op.eqn_mut()
                .update_rhs_out_state(&self.old_y, &self.old_f, t);
        }

        // solve for sensitivities equations discretised using sdirk equation
        for j in 0..self.sdiff.len() {
            let s0 = &self.state.s[j];
            let op = self.s_op.as_mut().unwrap();
            op.set_phi(&self.sdiff[j].columns(0, i), s0, &self.a_rows[i]);
            op.eqn_mut().set_index(j);
            let ds = &mut self.old_f_sens[j];
            Self::predict_stage(i, &self.sdiff[j], ds, &self.tableau);

            // solve
            let op = self.s_op.as_ref().unwrap();
            self.nonlinear_solver
                .solve_in_place(op, ds, t, s0, &mut self.convergence)?;

            self.old_y_sens[j].copy_from(&op.get_last_f_eval());
            self.statistics.number_of_nonlinear_solver_iterations += self.convergence.niter();

            // calculate sdg and store in sgdiff
            if let Some(out) = self.s_op.as_ref().unwrap().eqn().out() {
                let dsg = &mut self.state.dsg[j];
                out.call_inplace(&self.old_y_sens[j], t, dsg);
                self.sgdiff[j].column_mut(i).axpy(h, dsg, Eqn::T::zero());
            }
        }
        Ok(())
    }

    fn interpolate_from_diff(y0: &Eqn::V, beta_f: &Eqn::V, diff: &M) -> Eqn::V {
        // ret = old_y + sum_{i=0}^{s_star-1} beta[i] * diff[:, i]
        let mut ret = y0.clone();
        diff.gemv(Eqn::T::one(), beta_f, Eqn::T::one(), &mut ret);
        ret
    }

    fn interpolate_beta_function(theta: Eqn::T, beta: &M) -> Eqn::V {
        let poly_order = beta.ncols();
        let s_star = beta.nrows();
        let mut thetav = Vec::with_capacity(poly_order);
        thetav.push(theta);
        for i in 1..poly_order {
            thetav.push(theta * thetav[i - 1]);
        }
        // beta_poly = beta * thetav
        let thetav = Eqn::V::from_vec(thetav);
        let mut beta_f = <Eqn::V as Vector>::zeros(s_star);
        beta.gemv(Eqn::T::one(), &thetav, Eqn::T::zero(), &mut beta_f);
        beta_f
    }

    fn interpolate_hermite(theta: Eqn::T, u0: &Eqn::V, u1: &Eqn::V, diff: &M) -> Eqn::V {
        let hf0 = diff.column(0);
        let hf1 = diff.column(diff.ncols() - 1);
        u0 * scale(Eqn::T::from(1.0) - theta)
            + u1 * scale(theta)
            + ((u1 - u0) * scale(Eqn::T::from(1.0) - Eqn::T::from(2.0) * theta)
                + hf0 * scale(theta - Eqn::T::from(1.0))
                + hf1 * scale(theta))
                * scale(theta * (theta - Eqn::T::from(1.0)))
    }

    fn _jacobian_updates(&mut self, h: Eqn::T, state: SolverState) {
        if self.jacobian_update.check_rhs_jacobian_update(h, &state) {
            if let Some(op) = self.op.as_mut() {
                op.set_jacobian_is_stale();
                self.nonlinear_solver
                    .reset_jacobian(op, &self.old_f, self.state.t);
            } else if let Some(s_op) = self.s_op.as_mut() {
                s_op.set_jacobian_is_stale();
                self.nonlinear_solver
                    .reset_jacobian(s_op, &self.old_f_sens[0], self.state.t);
            }
            self.jacobian_update.update_rhs_jacobian();
            self.jacobian_update.update_jacobian(h);
        } else if self.jacobian_update.check_jacobian_update(h, &state) {
            if let Some(op) = self.op.as_ref() {
                self.nonlinear_solver
                    .reset_jacobian(op, &self.old_f, self.state.t);
            } else if let Some(s_op) = self.s_op.as_ref() {
                self.nonlinear_solver
                    .reset_jacobian(s_op, &self.old_f_sens[0], self.state.t);
            }
            self.jacobian_update.update_jacobian(h);
        }
    }

    fn _update_step_size(&mut self, factor: Eqn::T) -> Result<Eqn::T, DiffsolError> {
        let new_h = self.state.h * factor;

        // if step size too small, then fail
        if abs(new_h) < Eqn::T::from(Self::MIN_TIMESTEP) {
            return Err(DiffsolError::from(OdeSolverError::StepSizeTooSmall {
                time: self.state.t.into(),
            }));
        }

        // update h for new step size
        if let Some(op) = self.op.as_mut() {
            op.set_h(new_h);
        }
        if let Some(s_op) = self.s_op.as_mut() {
            s_op.set_h(new_h);
        }

        // update state
        self.state.h = new_h;

        Ok(new_h)
    }
}

impl<'a, M, Eqn, AugmentedEqn, LS> OdeSolverMethod<'a, Eqn> for Sdirk<'a, Eqn, LS, M, AugmentedEqn>
where
    LS: LinearSolver<Eqn::M>,
    M: DenseMatrix<T = Eqn::T, V = Eqn::V>,
    Eqn: OdeEquationsImplicit,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T>,
    AugmentedEqn: AugmentedOdeEquationsImplicit<Eqn>,
    for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
    for<'b> &'b Eqn::M: MatrixRef<Eqn::M>,
{
    type State = SdirkState<Eqn::V>;

    fn problem(&self) -> &'a OdeSolverProblem<Eqn> {
        self.problem
    }

    fn jacobian(&self) -> Option<std::cell::Ref<<Eqn>::M>> {
        if let Some(op) = self.op.as_ref() {
            Some(op.rhs_jac())
        } else {
            self.s_op.as_ref().map(|s_op| s_op.rhs_jac())
        }
    }

    fn order(&self) -> usize {
        self.tableau.order()
    }

    fn set_state(&mut self, state: Self::State) {
        self.state = state;

        // update the op with the new state
        if let Some(op) = self.op.as_mut() {
            op.set_h(self.state.h);
        }

        // reinitialise jacobian updates as if a checkpoint was taken
        self._jacobian_updates(self.state.h, SolverState::Checkpoint);
    }

    fn into_state(self) -> SdirkState<Eqn::V> {
        self.state
    }

    fn checkpoint(&mut self) -> Self::State {
        self._jacobian_updates(self.state.h, SolverState::Checkpoint);
        self.state.clone()
    }

    fn step(&mut self) -> Result<OdeSolverStopReason<Eqn::T>, DiffsolError> {
        let n = self.state.y.len();

        if self.is_state_mutated {
            // reinitalise root finder if needed
            if let Some(root_fn) = self.problem.eqn.root() {
                let state = &self.state;
                self.root_finder
                    .as_ref()
                    .unwrap()
                    .init(&root_fn, &state.y, state.t);
            }
            // reinitialise tstop if needed
            if let Some(t_stop) = self.tstop {
                self.set_stop_time(t_stop)?;
            }

            self.is_state_mutated = false;
        }

        // optionally do the first step
        let start = if self.is_sdirk { 0 } else { 1 };
        let mut updated_jacobian = false;

        // dont' reset jacobian for the first attempt at the step
        let mut error = <Eqn::V as Vector>::zeros(n);
        let out_error_control = self.problem().output_in_error_control();
        let mut out_error = if out_error_control {
            <Eqn::V as Vector>::zeros(self.problem().eqn.out().unwrap().nout())
        } else {
            <Eqn::V as Vector>::zeros(0)
        };
        let sens_error_control =
            self.s_op.is_some() && self.s_op.as_ref().unwrap().eqn().include_in_error_control();
        let mut sens_error = if sens_error_control {
            <Eqn::V as Vector>::zeros(self.s_op.as_ref().unwrap().eqn().rhs().nstates())
        } else {
            <Eqn::V as Vector>::zeros(0)
        };
        let sens_out_error_control = self.s_op.is_some()
            && self
                .s_op
                .as_ref()
                .unwrap()
                .eqn()
                .include_out_in_error_control();
        let mut sens_out_error = if sens_out_error_control {
            <Eqn::V as Vector>::zeros(self.s_op.as_ref().unwrap().eqn().out().unwrap().nout())
        } else {
            <Eqn::V as Vector>::zeros(0)
        };

        let mut factor: Eqn::T;

        // loop until step is accepted
        'step: loop {
            let t0 = self.state.t;
            let h = self.state.h;
            // if start == 1, then we need to compute the first stage
            // from the last stage of the previous step
            if start == 1 {
                {
                    let state = &self.state;
                    let mut hf = self.diff.column_mut(0);
                    hf.copy_from(&state.dy);
                    hf *= scale(h);
                }

                // sensitivities too
                if self.s_op.is_some() {
                    for (diff, dy) in self.sdiff.iter_mut().zip(self.state.ds.iter()) {
                        let mut hf = diff.column_mut(0);
                        hf.copy_from(dy);
                        hf *= scale(h);
                    }
                    for (diff, dg) in self.sgdiff.iter_mut().zip(self.state.dsg.iter()) {
                        let mut hf = diff.column_mut(0);
                        hf.copy_from(dg);
                        hf *= scale(h);
                    }
                }

                // output function
                if self.problem.integrate_out {
                    let state = &self.state;
                    let mut hf = self.gdiff.column_mut(0);
                    hf.copy_from(&state.dg);
                    hf *= scale(h);
                }
            }

            for i in start..self.tableau.s() {
                let t = t0 + self.tableau.c()[i] * h;

                // main equation
                let mut solve_result = Ok(());
                if let Some(op) = self.op.as_mut() {
                    op.set_phi(&self.diff.columns(0, i), &self.state.y, &self.a_rows[i]);
                    Self::predict_stage(i, &self.diff, &mut self.old_f, &self.tableau);
                    solve_result = self.nonlinear_solver.solve_in_place(
                        op,
                        &mut self.old_f,
                        t,
                        &self.state.y,
                        &mut self.convergence,
                    );
                    self.statistics.number_of_nonlinear_solver_iterations +=
                        self.convergence.niter();
                }

                // only calculate sensitivities if the solve succeeded
                if solve_result.is_ok() {
                    if let Some(op) = self.op.as_ref() {
                        self.old_y.copy_from(&op.get_last_f_eval());
                    }
                    // old_y now has the new y soln and old_f has the new dy soln
                    if self.s_op.is_some() {
                        solve_result = self.solve_for_sensitivities(i, t);
                    }
                }

                // handle solve failure
                if solve_result.is_err() {
                    self.statistics.number_of_nonlinear_solver_fails += 1;
                    if !updated_jacobian {
                        // newton iteration did not converge, so update jacobian and try again
                        updated_jacobian = true;
                        self._jacobian_updates(h, SolverState::FirstConvergenceFail);
                    } else {
                        // newton iteration did not converge and jacobian has been updated, so we reduce step size and try again
                        let new_h = self._update_step_size(Eqn::T::from(0.3))?;
                        self._jacobian_updates(new_h, SolverState::SecondConvergenceFail);
                    }
                    // try again....
                    continue 'step;
                };

                if self.op.is_some() {
                    // update diff with solved dy
                    self.diff.column_mut(i).copy_from(&self.old_f);

                    // calculate dg and store in gdiff
                    if self.problem.integrate_out {
                        let out = self.problem.eqn.out().unwrap();
                        out.call_inplace(&self.old_y, t, &mut self.state.dg);
                        self.gdiff
                            .column_mut(i)
                            .axpy(h, &self.state.dg, Eqn::T::zero());
                    }
                }

                if self.s_op.is_some() {
                    for (diff, old_f_sens) in self.sdiff.iter_mut().zip(self.old_f_sens.iter()) {
                        diff.column_mut(i).copy_from(old_f_sens);
                    }
                }
            }
            let mut ncontributions = 0;
            let mut error_norm = Eqn::T::zero();
            // successfully solved for all stages, now compute error
            if self.op.is_some() {
                self.diff
                    .gemv(Eqn::T::one(), self.tableau.d(), Eqn::T::zero(), &mut error);

                // compute error norm
                let atol = &self.problem().atol;
                let rtol = self.problem().rtol;
                error_norm += error.squared_norm(&self.old_y, atol, rtol);
                ncontributions += 1;

                // output errors
                if out_error_control {
                    self.gdiff.gemv(
                        Eqn::T::one(),
                        self.tableau.d(),
                        Eqn::T::zero(),
                        &mut out_error,
                    );
                    let atol = self.problem().out_atol.as_ref().unwrap();
                    let rtol = self.problem().out_rtol.unwrap();
                    let out_error_norm = out_error.squared_norm(&self.old_g, atol, rtol);
                    error_norm += out_error_norm;
                    ncontributions += 1;
                }
            }

            // sensitivity errors
            if sens_error_control {
                let atol = self.s_op.as_ref().unwrap().eqn().atol().unwrap();
                let rtol = self.s_op.as_ref().unwrap().eqn().rtol().unwrap();
                for i in 0..self.sdiff.len() {
                    self.sdiff[i].gemv(
                        Eqn::T::one(),
                        self.tableau.d(),
                        Eqn::T::zero(),
                        &mut sens_error,
                    );
                    let sens_error_norm = sens_error.squared_norm(&self.old_y_sens[i], atol, rtol);
                    error_norm += sens_error_norm;
                    ncontributions += 1;
                }
            }

            // sensitivity output errors
            if sens_out_error_control {
                let atol = self.s_op.as_ref().unwrap().eqn().out_atol().unwrap();
                let rtol = self.s_op.as_ref().unwrap().eqn().out_rtol().unwrap();
                for i in 0..self.sgdiff.len() {
                    self.sgdiff[i].gemv(
                        Eqn::T::one(),
                        self.tableau.d(),
                        Eqn::T::zero(),
                        &mut sens_out_error,
                    );
                    let sens_error_norm =
                        sens_out_error.squared_norm(&self.state.sg[i], atol, rtol);
                    error_norm += sens_error_norm;
                    ncontributions += 1;
                }
            }
            if ncontributions > 1 {
                error_norm /= Eqn::T::from(ncontributions as f64);
            }

            // adjust step size based on error
            let maxiter = self.convergence.max_iter() as f64;
            let niter = self.convergence.niter() as f64;
            let safety = Eqn::T::from(0.9 * (2.0 * maxiter + 1.0) / (2.0 * maxiter + niter));
            let order = self.tableau.order() as f64;
            factor = safety * error_norm.pow(Eqn::T::from(-0.5 / (order + 1.0)));
            if factor < Eqn::T::from(Self::MIN_FACTOR) {
                factor = Eqn::T::from(Self::MIN_FACTOR);
            }
            if factor > Eqn::T::from(Self::MAX_FACTOR) {
                factor = Eqn::T::from(Self::MAX_FACTOR);
            }

            // test error is within tolerance
            if error_norm <= Eqn::T::from(1.0) {
                break 'step;
            }
            // step is rejected, factor reduces step size, so we try again with the smaller step size
            self.statistics.number_of_error_test_failures += 1;
            let new_h = self._update_step_size(factor)?;
            self._jacobian_updates(new_h, SolverState::ErrorTestFail);
        }

        // take the step
        {
            let state = &mut self.state;
            self.old_t = state.t;
            state.t += state.h;

            // last stage is the solution and is the same as old_f
            // todo: can we get rid of old_f and just use diff?
            self.old_f.mul_assign(scale(Eqn::T::one() / state.h));
            std::mem::swap(&mut self.old_f, &mut state.dy);

            // old_y already has the new y soln
            std::mem::swap(&mut self.old_y, &mut state.y);

            for i in 0..self.sdiff.len() {
                self.old_f_sens[i].mul_assign(scale(Eqn::T::one() / state.h));
                std::mem::swap(&mut self.old_f_sens[i], &mut state.ds[i]);
                std::mem::swap(&mut self.old_y_sens[i], &mut state.s[i]);
            }

            for i in 0..self.sgdiff.len() {
                self.sgdiff[i].gemv(
                    Eqn::T::one(),
                    self.tableau.b(),
                    Eqn::T::one(),
                    &mut state.sg[i],
                );
            }

            // integrate output function
            if self.problem.integrate_out {
                self.old_g.copy_from(&state.g);
                self.gdiff
                    .gemv(Eqn::T::one(), self.tableau.b(), Eqn::T::one(), &mut state.g);
            }
        }

        // update step size for next step
        let new_h = self._update_step_size(factor)?;
        self._jacobian_updates(new_h, SolverState::StepSuccess);

        // update statistics
        if let Some(op) = self.op.as_ref() {
            self.statistics.number_of_linear_solver_setups = op.number_of_jac_evals();
        } else if let Some(s_op) = self.s_op.as_ref() {
            self.statistics.number_of_linear_solver_setups = s_op.number_of_jac_evals();
        }
        self.statistics.number_of_steps += 1;
        self.jacobian_update.step();

        // check for root within accepted step
        if let Some(root_fn) = self.problem.eqn.root() {
            let ret = self.root_finder.as_ref().unwrap().check_root(
                &|t| self.interpolate(t),
                &root_fn,
                &self.state.y,
                self.state.t,
            );
            if let Some(root) = ret {
                return Ok(OdeSolverStopReason::RootFound(root));
            }
        }

        // check if the we are at tstop
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

    fn interpolate_sens(&self, t: <Eqn as Op>::T) -> Result<Vec<<Eqn as Op>::V>, DiffsolError> {
        let state = &self.state;

        if self.is_state_mutated {
            if t == state.t {
                return Ok(state.s.clone());
            } else {
                return Err(ode_solver_error!(InterpolationTimeOutsideCurrentStep));
            }
        }

        // check that t is within the current step depending on the direction
        let is_forward = state.h > Eqn::T::zero();
        if (is_forward && (t > state.t || t < self.old_t))
            || (!is_forward && (t < state.t || t > self.old_t))
        {
            return Err(ode_solver_error!(InterpolationTimeOutsideCurrentStep));
        }
        let dt = state.t - self.old_t;
        let theta = if dt == Eqn::T::zero() {
            Eqn::T::one()
        } else {
            (t - self.old_t) / dt
        };

        if let Some(beta) = self.tableau.beta() {
            let beta_f = Self::interpolate_beta_function(theta, beta);
            let ret = self
                .old_y_sens
                .iter()
                .zip(self.sdiff.iter())
                .map(|(y, diff)| Self::interpolate_from_diff(y, &beta_f, diff))
                .collect();
            Ok(ret)
        } else {
            let ret = self
                .old_y_sens
                .iter()
                .zip(state.s.iter())
                .zip(self.sdiff.iter())
                .map(|((s0, s1), diff)| Self::interpolate_hermite(theta, s0, s1, diff))
                .collect();
            Ok(ret)
        }
    }

    fn interpolate(&self, t: <Eqn>::T) -> Result<<Eqn>::V, DiffsolError> {
        let state = &self.state;

        if self.is_state_mutated {
            if t == state.t {
                return Ok(state.y.clone());
            } else {
                return Err(ode_solver_error!(InterpolationTimeOutsideCurrentStep));
            }
        }

        // check that t is within the current step depending on the direction
        let is_forward = state.h > Eqn::T::zero();
        if (is_forward && (t > state.t || t < self.old_t))
            || (!is_forward && (t < state.t || t > self.old_t))
        {
            return Err(ode_solver_error!(InterpolationTimeOutsideCurrentStep));
        }

        let dt = state.t - self.old_t;
        let theta = if dt == Eqn::T::zero() {
            Eqn::T::one()
        } else {
            (t - self.old_t) / dt
        };

        if let Some(beta) = self.tableau.beta() {
            let beta_f = Self::interpolate_beta_function(theta, beta);
            let ret = Self::interpolate_from_diff(&self.old_y, &beta_f, &self.diff);
            Ok(ret)
        } else {
            let ret = Self::interpolate_hermite(theta, &self.old_y, &state.y, &self.diff);
            Ok(ret)
        }
    }

    fn interpolate_out(&self, t: <Eqn>::T) -> Result<<Eqn>::V, DiffsolError> {
        let state = &self.state;

        if self.is_state_mutated {
            if t == state.t {
                return Ok(state.g.clone());
            } else {
                return Err(ode_solver_error!(InterpolationTimeOutsideCurrentStep));
            }
        }

        // check that t is within the current step depending on the direction
        let is_forward = state.h > Eqn::T::zero();
        if (is_forward && (t > state.t || t < self.old_t))
            || (!is_forward && (t < state.t || t > self.old_t))
        {
            return Err(ode_solver_error!(InterpolationTimeOutsideCurrentStep));
        }

        let dt = state.t - self.old_t;
        let theta = if dt == Eqn::T::zero() {
            Eqn::T::one()
        } else {
            (t - self.old_t) / dt
        };

        if let Some(beta) = self.tableau.beta() {
            let beta_f = Self::interpolate_beta_function(theta, beta);
            let ret = Self::interpolate_from_diff(&self.old_g, &beta_f, &self.gdiff);
            Ok(ret)
        } else {
            let ret = Self::interpolate_hermite(theta, &self.old_g, &state.g, &self.gdiff);
            Ok(ret)
        }
    }

    fn state(&self) -> StateRef<Eqn::V> {
        self.state.as_ref()
    }

    fn state_mut(&mut self) -> StateRefMut<Eqn::V> {
        self.is_state_mutated = true;
        self.state.as_mut()
    }
}

#[cfg(test)]
mod test {
    use crate::{
        ode_solver::{
            test_models::{
                exponential_decay::{
                    exponential_decay_problem, exponential_decay_problem_adjoint,
                    exponential_decay_problem_sens, exponential_decay_problem_with_root,
                    negative_exponential_decay_problem,
                },
                exponential_decay_with_algebraic::exponential_decay_with_algebraic_adjoint_problem,
                heat2d::head2d_problem,
                robertson::{robertson, robertson_sens},
                robertson_ode::robertson_ode,
            },
            tests::{
                setup_test_adjoint, test_adjoint, test_checkpointing, test_interpolate,
                test_ode_solver, test_problem, test_state_mut, test_state_mut_on_problem,
            },
        },
        FaerSparseLU, NalgebraLU, OdeEquations, OdeSolverMethod, Op, SparseColMat, Vector,
    };

    use num_traits::abs;

    type M = nalgebra::DMatrix<f64>;
    type LS = NalgebraLU<f64>;

    #[test]
    fn sdirk_state_mut() {
        test_state_mut(test_problem::<M>().tr_bdf2::<LS>().unwrap());
    }
    #[test]
    fn sdirk_test_interpolate() {
        test_interpolate(test_problem::<M>().tr_bdf2::<LS>().unwrap());
    }

    #[test]
    fn sdirk_test_checkpointing() {
        let (problem, soln) = exponential_decay_problem::<M>(false);
        let s1 = problem.tr_bdf2::<LS>().unwrap();
        let s2 = problem.tr_bdf2::<LS>().unwrap();
        test_checkpointing(soln, s1, s2);
    }

    #[test]
    fn sdirk_test_state_mut_exponential_decay() {
        let (p, soln) = exponential_decay_problem::<M>(false);
        let s = p.tr_bdf2::<LS>().unwrap();
        test_state_mut_on_problem(s, soln);
    }

    #[test]
    fn sdirk_test_nalgebra_negative_exponential_decay() {
        let (problem, soln) = negative_exponential_decay_problem::<M>(false);
        let mut s = problem.esdirk34::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[test]
    fn test_tr_bdf2_nalgebra_exponential_decay() {
        let (problem, soln) = exponential_decay_problem::<M>(false);
        let mut s = problem.tr_bdf2::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 4
        number_of_steps: 29
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 116
        number_of_nonlinear_solver_fails: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 118
        number_of_jac_muls: 2
        number_of_matrix_evals: 1
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_tr_bdf2_nalgebra_exponential_decay_sens() {
        let (problem, soln) = exponential_decay_problem_sens::<M>(false);
        let mut s = problem.tr_bdf2_sens::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, true);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 8
        number_of_steps: 53
        number_of_error_test_failures: 1
        number_of_nonlinear_solver_iterations: 540
        number_of_nonlinear_solver_fails: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 218
        number_of_jac_muls: 330
        number_of_matrix_evals: 2
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_esdirk34_nalgebra_exponential_decay() {
        let (problem, soln) = exponential_decay_problem::<M>(false);
        let mut s = problem.esdirk34::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 3
        number_of_steps: 13
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 84
        number_of_nonlinear_solver_fails: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 86
        number_of_jac_muls: 2
        number_of_matrix_evals: 1
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_esdirk34_nalgebra_exponential_decay_sens() {
        let (problem, soln) = exponential_decay_problem_sens::<M>(false);
        let mut s = problem.esdirk34_sens::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, true);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 6
        number_of_steps: 20
        number_of_error_test_failures: 1
        number_of_nonlinear_solver_iterations: 332
        number_of_nonlinear_solver_fails: 0
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 128
        number_of_jac_muls: 210
        number_of_matrix_evals: 1
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn sdirk_test_esdirk34_exponential_decay_adjoint() {
        let (mut problem, soln) = exponential_decay_problem_adjoint::<M>(true);
        let final_time = soln.solution_points.last().unwrap().t;
        let dgdu = setup_test_adjoint::<LS, _>(&mut problem, soln);
        let mut s = problem.esdirk34::<LS>().unwrap();
        let (checkpointer, _y, _t) = s.solve_with_checkpointing(final_time, None).unwrap();
        let adjoint_solver = problem
            .esdirk34_solver_adjoint::<LS, _>(checkpointer)
            .unwrap();
        test_adjoint(adjoint_solver, dgdu);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 314
        number_of_jac_muls: 6
        number_of_matrix_evals: 3
        number_of_jac_adj_muls: 474
        "###);
    }

    #[test]
    fn sdirk_test_esdirk34_exponential_decay_algebraic_adjoint() {
        let (mut problem, soln) = exponential_decay_with_algebraic_adjoint_problem::<M>(true);
        let final_time = soln.solution_points.last().unwrap().t;
        let dgdu = setup_test_adjoint::<LS, _>(&mut problem, soln);
        let mut s = problem.esdirk34::<LS>().unwrap();
        let (checkpointer, _y, _t) = s.solve_with_checkpointing(final_time, None).unwrap();
        let adjoint_solver = problem
            .esdirk34_solver_adjoint::<LS, _>(checkpointer)
            .unwrap();
        test_adjoint(adjoint_solver, dgdu);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 265
        number_of_jac_muls: 12
        number_of_matrix_evals: 4
        number_of_jac_adj_muls: 191
        "###);
    }

    #[test]
    fn test_tr_bdf2_nalgebra_robertson() {
        let (problem, soln) = robertson::<M>(false);
        let mut s = problem.tr_bdf2::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 97
        number_of_steps: 232
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 1921
        number_of_nonlinear_solver_fails: 18
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 1924
        number_of_jac_muls: 36
        number_of_matrix_evals: 12
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_tr_bdf2_nalgebra_robertson_sens() {
        let (problem, soln) = robertson_sens::<M>();
        let mut s = problem.tr_bdf2_sens::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, true);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 109
        number_of_steps: 215
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 4544
        number_of_nonlinear_solver_fails: 36
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 1443
        number_of_jac_muls: 3268
        number_of_matrix_evals: 28
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_esdirk34_nalgebra_robertson() {
        let (problem, soln) = robertson::<M>(false);
        let mut s = problem.esdirk34::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 100
        number_of_steps: 141
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 1793
        number_of_nonlinear_solver_fails: 24
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 1796
        number_of_jac_muls: 54
        number_of_matrix_evals: 18
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_esdirk34_nalgebra_robertson_sens() {
        let (problem, soln) = robertson_sens::<M>();
        let mut s = problem.esdirk34_sens::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, true);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 114
        number_of_steps: 131
        number_of_error_test_failures: 0
        number_of_nonlinear_solver_iterations: 4442
        number_of_nonlinear_solver_fails: 44
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 1492
        number_of_jac_muls: 3136
        number_of_matrix_evals: 33
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_tr_bdf2_nalgebra_robertson_ode() {
        let (problem, soln) = robertson_ode::<M>(false, 1);
        let mut s = problem.tr_bdf2::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
        insta::assert_yaml_snapshot!(s.get_statistics(), @r###"
        number_of_linear_solver_setups: 113
        number_of_steps: 304
        number_of_error_test_failures: 1
        number_of_nonlinear_solver_iterations: 2601
        number_of_nonlinear_solver_fails: 15
        "###);
        insta::assert_yaml_snapshot!(problem.eqn.rhs().statistics(), @r###"
        number_of_calls: 2603
        number_of_jac_muls: 39
        number_of_matrix_evals: 13
        number_of_jac_adj_muls: 0
        "###);
    }

    #[test]
    fn test_tr_bdf2_faer_sparse_heat2d() {
        let (problem, soln) = head2d_problem::<SparseColMat<f64>, 10>();
        let mut s = problem.tr_bdf2::<FaerSparseLU<f64>>().unwrap();
        test_ode_solver(&mut s, soln, None, false, false);
    }

    #[test]
    fn test_tstop_tr_bdf2() {
        let (problem, soln) = exponential_decay_problem::<M>(false);
        let mut s = problem.tr_bdf2::<LS>().unwrap();
        test_ode_solver(&mut s, soln, None, true, false);
    }

    #[test]
    fn test_root_finder_tr_bdf2() {
        let (problem, soln) = exponential_decay_problem_with_root::<M>(false);
        let mut s = problem.tr_bdf2::<LS>().unwrap();
        let y = test_ode_solver(&mut s, soln, None, false, false);
        assert!(abs(y[0] - 0.6) < 1e-6, "y[0] = {}", y[0]);
    }

    #[test]
    fn test_param_sweep_tr_bdf2() {
        let (mut problem, _soln) = exponential_decay_problem::<M>(false);
        let mut ps = Vec::new();
        for y0 in (1..10).map(f64::from) {
            ps.push(nalgebra::DVector::<f64>::from_vec(vec![0.1, y0]));
        }

        let mut old_soln: Option<nalgebra::DVector<f64>> = None;
        for p in ps {
            problem.eqn_mut().set_params(&p);
            let mut s = problem.tr_bdf2::<LS>().unwrap();
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
    fn test_ball_bounce_tr_bdf2() {
        type M = nalgebra::DMatrix<f64>;
        type LS = crate::NalgebraLU<f64>;
        let (x, v, t) = crate::ode_solver::tests::test_ball_bounce(
            crate::ode_solver::tests::test_ball_bounce_problem::<M>()
                .tr_bdf2::<LS>()
                .unwrap(),
        );
        let expected_x = [6.375884661615263];
        let expected_v = [0.6878538646461059];
        let expected_t = [2.5];
        for (i, ((x, v), t)) in x.iter().zip(v.iter()).zip(t.iter()).enumerate() {
            assert!((x - expected_x[i]).abs() < 1e-4);
            assert!((v - expected_v[i]).abs() < 1e-4);
            assert!((t - expected_t[i]).abs() < 1e-4);
        }
    }
}
