use std::cell::RefCell;

use crate::{
    error::DiffsolError, other_error, OdeEquations, OdeSolverMethod, OdeSolverProblem,
    OdeSolverState, Scalar, Vector,
};
use num_traits::{abs, One};

#[derive(Clone)]
pub struct HermiteInterpolator<V>
where
    V: Vector,
{
    ys: Vec<V>,
    ydots: Vec<V>,
    ts: Vec<V::T>,
}

impl<V> Default for HermiteInterpolator<V>
where
    V: Vector,
{
    fn default() -> Self {
        HermiteInterpolator {
            ys: Vec::new(),
            ydots: Vec::new(),
            ts: Vec::new(),
        }
    }
}

impl<V> HermiteInterpolator<V>
where
    V: Vector,
{
    pub fn new(ys: Vec<V>, ydots: Vec<V>, ts: Vec<V::T>) -> Self {
        HermiteInterpolator { ys, ydots, ts }
    }
    pub fn last_t(&self) -> Option<V::T> {
        if self.ts.is_empty() {
            return None;
        }
        Some(self.ts[self.ts.len() - 1])
    }
    pub fn last_h(&self) -> Option<V::T> {
        if self.ts.len() < 2 {
            return None;
        }
        Some(self.ts[self.ts.len() - 1] - self.ts[self.ts.len() - 2])
    }
    pub fn reset<'a, Eqn, Method, State>(
        &mut self,
        solver: &mut Method,
        state0: &State,
        state1: &State,
    ) -> Result<(), DiffsolError>
    where
        Eqn: OdeEquations<V = V, T = V::T> + 'a,
        Method: OdeSolverMethod<'a, Eqn, State = State>,
        State: OdeSolverState<V>,
    {
        let state0_ref = state0.as_ref();
        let state1_ref = state1.as_ref();
        self.ys.clear();
        self.ydots.clear();
        self.ts.clear();
        self.ys.push(state0_ref.y.clone());
        self.ydots.push(state0_ref.dy.clone());
        self.ts.push(state0_ref.t);

        solver.set_state(state0.clone());
        while solver.state().t < state1_ref.t {
            solver.step()?;
            self.ys.push(solver.state().y.clone());
            self.ydots.push(solver.state().dy.clone());
            self.ts.push(solver.state().t);
        }
        Ok(())
    }

    pub fn interpolate(&self, t: V::T, y: &mut V) -> Option<()> {
        if t < self.ts[0] || t > self.ts[self.ts.len() - 1] {
            return None;
        }
        if t == self.ts[0] {
            y.copy_from(&self.ys[0]);
            return Some(());
        }
        let idx = self
            .ts
            .iter()
            .position(|&t0| t0 > t)
            .unwrap_or(self.ts.len() - 1);
        let t0 = self.ts[idx - 1];
        let t1 = self.ts[idx];
        let h = t1 - t0;
        let theta = (t - t0) / h;
        let u0 = &self.ys[idx - 1];
        let u1 = &self.ys[idx];
        let f0 = &self.ydots[idx - 1];
        let f1 = &self.ydots[idx];

        y.copy_from(u0);
        y.axpy(V::T::one(), u1, -V::T::one());
        y.axpy(
            h * (theta - V::T::from(1.0)),
            f0,
            V::T::one() - V::T::from(2.0) * theta,
        );
        y.axpy(h * theta, f1, V::T::one());
        y.axpy(
            V::T::from(1.0) - theta,
            u0,
            theta * (theta - V::T::from(1.0)),
        );
        y.axpy(theta, u1, V::T::one());
        Some(())
    }
}

pub struct Checkpointing<'a, Eqn, Method>
where
    Method: OdeSolverMethod<'a, Eqn>,
    Eqn: OdeEquations,
{
    checkpoints: Vec<Method::State>,
    segment: RefCell<HermiteInterpolator<Eqn::V>>,
    previous_segment: RefCell<Option<HermiteInterpolator<Eqn::V>>>,
    solver: RefCell<Method>,
}

impl<'a, Eqn, Method> Clone for Checkpointing<'a, Eqn, Method>
where
    Method: OdeSolverMethod<'a, Eqn>,
    Eqn: OdeEquations,
{
    fn clone(&self) -> Self {
        Checkpointing {
            checkpoints: self.checkpoints.clone(),
            segment: RefCell::new(self.segment.borrow().clone()),
            previous_segment: RefCell::new(self.previous_segment.borrow().clone()),
            solver: RefCell::new(self.solver.borrow().clone()),
        }
    }
}

impl<'a, Eqn, Method> Checkpointing<'a, Eqn, Method>
where
    Method: OdeSolverMethod<'a, Eqn>,
    Eqn: OdeEquations,
{
    pub fn new(
        mut solver: Method,
        start_idx: usize,
        checkpoints: Vec<Method::State>,
        segment: Option<HermiteInterpolator<Eqn::V>>,
    ) -> Self {
        if checkpoints.len() < 2 {
            panic!("Checkpoints must have at least 2 elements");
        }
        if start_idx >= checkpoints.len() - 1 {
            panic!("start_idx must be less than checkpoints.len() - 1");
        }
        let segment = segment.unwrap_or_else(|| {
            let mut segment = HermiteInterpolator::default();
            segment
                .reset(
                    &mut solver,
                    &checkpoints[start_idx],
                    &checkpoints[start_idx + 1],
                )
                .unwrap();
            segment
        });
        let segment = RefCell::new(segment);
        let previous_segment = RefCell::new(None);
        let solver = RefCell::new(solver);
        Checkpointing {
            checkpoints,
            segment,
            previous_segment,
            solver,
        }
    }

    pub fn last_t(&self) -> Eqn::T {
        self.segment
            .borrow()
            .last_t()
            .expect("segment should not be empty")
    }

    pub fn last_h(&self) -> Option<Eqn::T> {
        self.segment.borrow().last_h()
    }

    pub fn problem(&self) -> &'a OdeSolverProblem<Eqn> {
        self.solver.borrow().problem()
    }

    pub fn interpolate(&self, t: Eqn::T, y: &mut Eqn::V) -> Result<(), DiffsolError> {
        {
            let segment = self.segment.borrow();
            if segment.interpolate(t, y).is_some() {
                return Ok(());
            }
        }

        {
            let previous_segment = self.previous_segment.borrow();
            if let Some(previous_segment) = previous_segment.as_ref() {
                if previous_segment.interpolate(t, y).is_some() {
                    return Ok(());
                }
            }
        }

        // if t is before first segment or after last segment, return error
        let h = self.last_h().unwrap_or(Eqn::T::one());
        let troundoff = Eqn::T::from(100.0) * Eqn::T::EPSILON * (abs(t) + abs(h));
        if t < self.checkpoints[0].as_ref().t - troundoff
            || t > self.checkpoints[self.checkpoints.len() - 1].as_ref().t + troundoff
        {
            return Err(other_error!("t is outside of the checkpoints"));
        }

        // snap t to nearest checkpoint if outside of range
        let t = if t < self.checkpoints[0].as_ref().t {
            self.checkpoints[0].as_ref().t
        } else if t > self.checkpoints[self.checkpoints.len() - 1].as_ref().t {
            self.checkpoints[self.checkpoints.len() - 1].as_ref().t
        } else {
            t
        };

        // else find idx of segment
        let idx = self
            .checkpoints
            .iter()
            .skip(1)
            .position(|state| state.as_ref().t > t)
            .expect("t is not in checkpoints");
        if self.previous_segment.borrow().is_none() {
            self.previous_segment
                .replace(Some(HermiteInterpolator::default()));
        }
        let mut solver = self.solver.borrow_mut();
        let mut previous_segment = self.previous_segment.borrow_mut();
        let mut segment = self.segment.borrow_mut();
        previous_segment.as_mut().unwrap().reset(
            &mut *solver,
            &self.checkpoints[idx],
            &self.checkpoints[idx + 1],
        )?;
        std::mem::swap(&mut *segment, previous_segment.as_mut().unwrap());
        segment.interpolate(t, y).unwrap();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::{DMatrix, DVector};

    use crate::{
        ode_solver::test_models::robertson::robertson, NalgebraLU, OdeEquations, OdeSolverMethod,
        Op, Vector,
    };

    use super::{Checkpointing, HermiteInterpolator};

    #[test]
    fn test_checkpointing() {
        type M = DMatrix<f64>;
        type LS = NalgebraLU<f64>;
        let (problem, soln) = robertson::<M>(false);
        let t_final = soln.solution_points.last().unwrap().t;
        let n_steps = 30;
        let mut solver = problem.bdf::<LS>().unwrap();
        let mut checkpoints = vec![solver.checkpoint()];
        let mut i = 0;
        let mut ys = Vec::new();
        let mut ts = Vec::new();
        let mut ydots = Vec::new();
        while solver.state().t < t_final {
            ts.push(solver.state().t);
            ys.push(solver.state().y.clone());
            ydots.push(solver.state().dy.clone());
            solver.step().unwrap();
            i += 1;
            if i % n_steps == 0 && solver.state().t < t_final {
                checkpoints.push(solver.checkpoint());
                ts.clear();
                ys.clear();
                ydots.clear();
            }
        }
        checkpoints.push(solver.checkpoint());
        let segment = HermiteInterpolator::new(ys, ydots, ts);
        let checkpointer =
            Checkpointing::new(solver, checkpoints.len() - 2, checkpoints, Some(segment));
        let mut y = DVector::zeros(problem.eqn.rhs().nstates());
        for point in soln.solution_points.iter().rev() {
            checkpointer.interpolate(point.t, &mut y).unwrap();
            y.assert_eq_norm(&point.state, &problem.atol, problem.rtol, 10.0);
        }
    }
}
