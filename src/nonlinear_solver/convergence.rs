use nalgebra::ComplexField;
use num_traits::{One, Pow};
use std::rc::Rc;

use crate::{scalar::IndexType, solver::SolverProblem, NonLinearOp, Scalar, Vector};

pub struct Convergence<V: Vector> {
    rtol: V::T,
    atol: Rc<V>,
    tol: V::T,
    max_iter: IndexType,
    iter: IndexType,
    old_norm: Option<V::T>,
}

pub enum ConvergenceStatus {
    Converged,
    Diverged,
    Continue,
    MaximumIterations,
}

impl<V: Vector> Convergence<V> {
    pub fn new_from_problem<C: NonLinearOp<V = V, T = V::T>>(
        problem: &SolverProblem<C>,
        max_iter: IndexType,
    ) -> Self {
        let rtol = problem.rtol;
        let atol = problem.atol.clone();
        Self::new(rtol, atol, max_iter)
    }
    pub fn new(rtol: V::T, atol: Rc<V>, max_iter: usize) -> Self {
        let minimum_tol = V::T::from(10.0) * V::T::EPSILON / rtol;
        let maximum_tol = V::T::from(0.03);
        let mut tol = V::T::from(0.5) * rtol.pow(V::T::from(0.5));
        if tol > maximum_tol {
            tol = maximum_tol;
        }
        if tol < minimum_tol {
            tol = minimum_tol;
        }
        Self {
            rtol,
            atol,
            tol,
            max_iter,
            old_norm: None,
            iter: 0,
        }
    }
    pub fn reset(&mut self) {
        self.iter = 0;
        self.old_norm = None;
    }
    pub fn check_new_iteration(&mut self, dy: &mut V, y: &V) -> ConvergenceStatus {
        let norm = dy.squared_norm(y, &self.atol, self.rtol).sqrt();
        // if norm is zero then we are done
        if norm <= V::T::EPSILON {
            return ConvergenceStatus::Converged;
        }
        if let Some(old_norm) = self.old_norm {
            let rate = norm / old_norm;

            if rate > V::T::from(1.0) {
                return ConvergenceStatus::Diverged;
            }

            // if converged then break out of iteration successfully
            if rate / (V::T::one() - rate) * norm < self.tol {
                return ConvergenceStatus::Converged;
            }

            // if iteration is not going to converge in NEWTON_MAXITER
            // (assuming the current rate), then abort
            if rate.pow(i32::try_from(self.max_iter - self.iter).unwrap())
                / (V::T::from(1.0) - rate)
                * norm
                > self.tol
            {
                return ConvergenceStatus::Diverged;
            }
        }
        // TODO: at the moment need 2 iterations to check convergence, should be able to do it in 1?
        self.iter += 1;
        self.old_norm = Some(norm);
        if self.iter >= self.max_iter {
            ConvergenceStatus::MaximumIterations
        } else {
            ConvergenceStatus::Continue
        }
    }
}
