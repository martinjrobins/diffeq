use crate::{
    error::DiffsolError, error::OdeSolverError, ode_solver_error, AugmentedOdeSolverMethod,
    DefaultDenseMatrix, DefaultSolver, DenseMatrix, Matrix, OdeEquationsSens, OdeSolverStopReason,
    Op, SensEquations, VectorViewMut,
};

pub trait SensitivitiesOdeSolverMethod<'a, Eqn>:
    AugmentedOdeSolverMethod<'a, Eqn, SensEquations<'a, Eqn>>
where
    Eqn: OdeEquationsSens + 'a,
{
    /// Using the provided state, solve the problem up to time `t_eval[t_eval.len()-1]`
    /// Returns a tuple `(y, sens)`, where `y` is a dense matrix of solution values at timepoints given by `t_eval`,
    /// and `sens` is a Vec of dense matrices, the ith element of the Vec are the the sensitivities with respect to the ith parameter.
    /// After the solver has finished, the internal state of the solver is at time `t_eval[t_eval.len()-1]`.
    #[allow(clippy::type_complexity)]
    fn solve_dense_sensitivities(
        &mut self,
        t_eval: &[Eqn::T],
    ) -> Result<
        (
            <Eqn::V as DefaultDenseMatrix>::M,
            Vec<<Eqn::V as DefaultDenseMatrix>::M>,
        ),
        DiffsolError,
    >
    where
        Eqn: OdeEquationsSens,
        Eqn::M: DefaultSolver,
        Eqn::V: DefaultDenseMatrix,
        Self: Sized,
    {
        if self.problem().integrate_out {
            return Err(ode_solver_error!(
                Other,
                "Cannot integrate out when solving for sensitivities"
            ));
        }
        let nrows = self.problem().eqn.rhs().nstates();
        let mut ret = <<Eqn::V as DefaultDenseMatrix>::M as Matrix>::zeros(nrows, t_eval.len());
        let mut ret_sens =
            vec![
                <<Eqn::V as DefaultDenseMatrix>::M as Matrix>::zeros(nrows, t_eval.len());
                self.problem().eqn.rhs().nparams()
            ];

        // check t_eval is increasing and all values are greater than or equal to the current time
        let t0 = self.state().t;
        if t_eval.windows(2).any(|w| w[0] > w[1] || w[0] < t0) {
            return Err(ode_solver_error!(InvalidTEval));
        }

        // do loop
        self.set_stop_time(t_eval[t_eval.len() - 1])?;
        let mut step_reason = OdeSolverStopReason::InternalTimestep;
        for (i, t) in t_eval.iter().take(t_eval.len() - 1).enumerate() {
            while self.state().t < *t {
                step_reason = self.step()?;
            }
            let y = self.interpolate(*t)?;
            ret.column_mut(i).copy_from(&y);
            let s = self.interpolate_sens(*t)?;
            for (j, s_j) in s.iter().enumerate() {
                ret_sens[j].column_mut(i).copy_from(s_j);
            }
        }

        // do final step
        while step_reason != OdeSolverStopReason::TstopReached {
            step_reason = self.step()?;
        }
        let y = self.state().y;
        ret.column_mut(t_eval.len() - 1).copy_from(y);
        let s = self.state().s;
        for (j, s_j) in s.iter().enumerate() {
            ret_sens[j].column_mut(t_eval.len() - 1).copy_from(s_j);
        }
        Ok((ret, ret_sens))
    }
}

impl<'a, M, Eqn> SensitivitiesOdeSolverMethod<'a, Eqn> for M
where
    M: AugmentedOdeSolverMethod<'a, Eqn, SensEquations<'a, Eqn>>,
    Eqn: OdeEquationsSens + 'a,
{
}
