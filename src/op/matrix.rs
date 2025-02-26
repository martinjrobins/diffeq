use crate::{LinearOp, Matrix, MatrixSparsityRef, NonLinearOp, NonLinearOpJacobian, Op};
use num_traits::Zero;

pub struct MatrixOp<M: Matrix> {
    m: M,
}

impl<M: Matrix> MatrixOp<M> {
    pub fn new(m: M) -> Self {
        Self { m }
    }
    pub fn m_mut(&mut self) -> &mut M {
        &mut self.m
    }
    pub fn m(&self) -> &M {
        &self.m
    }
}

impl<M: Matrix> Op for MatrixOp<M> {
    type V = M::V;
    type T = M::T;
    type M = M;
    fn nstates(&self) -> usize {
        self.m.nrows()
    }
    fn nout(&self) -> usize {
        self.m.ncols()
    }
    fn nparams(&self) -> usize {
        0
    }
}

impl<M: Matrix> LinearOp for MatrixOp<M> {
    fn gemv_inplace(&self, x: &Self::V, t: Self::T, beta: Self::T, y: &mut Self::V) {
        self.m.gemv(t, x, beta, y);
    }
    fn sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.m.sparsity().map(|s| s.to_owned())
    }
}

impl<M: Matrix> NonLinearOp for MatrixOp<M> {
    fn call_inplace(&self, x: &Self::V, t: Self::T, y: &mut Self::V) {
        self.m.gemv(t, x, Self::T::zero(), y);
    }
}

impl<M: Matrix> NonLinearOpJacobian for MatrixOp<M> {
    fn jac_mul_inplace(&self, _x: &Self::V, t: Self::T, v: &Self::V, y: &mut Self::V) {
        self.m.gemv(t, v, Self::T::zero(), y);
    }

    fn jacobian(&self, _x: &Self::V, _t: Self::T) -> Self::M {
        self.m.clone()
    }

    fn jacobian_inplace(&self, _x: &Self::V, _t: Self::T, y: &mut Self::M) {
        y.copy_from(&self.m);
    }

    fn jacobian_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.m.sparsity().map(|s| s.to_owned())
    }
}
