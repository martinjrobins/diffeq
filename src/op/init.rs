use crate::{
    scale, LinearOp, Matrix, MatrixSparsityRef, NonLinearOpJacobian, OdeEquationsImplicit, Vector,
    VectorIndex,
};
use num_traits::{One, Zero};
use std::cell::RefCell;

use super::{NonLinearOp, Op};

/// Consistent initial conditions for an ODE system.
///
/// We calculate consistent initial conditions following the approach of
/// Brown, P. N., Hindmarsh, A. C., & Petzold, L. R. (1998). Consistent initial condition calculation for differential-algebraic systems. SIAM Journal on Scientific Computing, 19(5), 1495-1512.
pub struct InitOp<'a, Eqn: OdeEquationsImplicit> {
    eqn: &'a Eqn,
    jac: Eqn::M,
    pub y0: RefCell<Eqn::V>,
    pub algebraic_indices: <Eqn::V as Vector>::Index,
    neg_mass: Eqn::M,
}

impl<'a, Eqn: OdeEquationsImplicit> InitOp<'a, Eqn> {
    pub fn new(eqn: &'a Eqn, t0: Eqn::T, y0: &Eqn::V) -> Self {
        let n = eqn.rhs().nstates();
        let mass_diagonal = eqn.mass().unwrap().matrix(t0).diagonal();
        let (algebraic_indices, _) = mass_diagonal.partition_indices(|x| x == Eqn::T::zero());

        let rhs_jac = eqn.rhs().jacobian(y0, t0);
        let mass = eqn.mass().unwrap().matrix(t0);

        // equations are:
        // h(t, u, v, du) = 0
        // g(t, u, v) = 0
        // where u are differential states, v are algebraic states.
        // choose h = -M_u du + f(u, v), where M_u are the differential states of the mass matrix
        // want to solve for du, v, so jacobian is
        // J = (-M_u, df/dv)
        //     (0,    dg/dv)
        // note rhs_jac = (df/du df/dv)
        //                (dg/du dg/dv)
        // according to the algebraic indices.
        let (m_u, _, _, _) = mass.split_at_indices(&algebraic_indices);
        let m_u = m_u * scale(-Eqn::T::one());
        let (_, dfdv, _, dgdv) = rhs_jac.split_at_indices(&algebraic_indices);
        let zero_ll =
            <Eqn::M as Matrix>::zeros(algebraic_indices.len(), n - algebraic_indices.len());
        let zero_ur =
            <Eqn::M as Matrix>::zeros(n - algebraic_indices.len(), algebraic_indices.len());
        let zero_lr = <Eqn::M as Matrix>::zeros(algebraic_indices.len(), algebraic_indices.len());
        let jac = Eqn::M::combine_at_indices(&m_u, &dfdv, &zero_ll, &dgdv, &algebraic_indices);
        let neg_mass =
            Eqn::M::combine_at_indices(&m_u, &zero_ur, &zero_ll, &zero_lr, &algebraic_indices);

        let y0 = y0.clone();
        let y0 = RefCell::new(y0);
        Self {
            eqn,
            jac,
            y0,
            neg_mass,
            algebraic_indices,
        }
    }

    pub fn scatter_soln(&self, soln: &Eqn::V, y: &mut Eqn::V, dy: &mut Eqn::V) {
        let tmp = dy.clone();
        dy.copy_from(soln);
        dy.copy_from_indices(&tmp, &self.algebraic_indices);
        y.copy_from_indices(soln, &self.algebraic_indices);
    }
}

impl<Eqn: OdeEquationsImplicit> Op for InitOp<'_, Eqn> {
    type V = Eqn::V;
    type T = Eqn::T;
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

impl<Eqn: OdeEquationsImplicit> NonLinearOp for InitOp<'_, Eqn> {
    // -M_u du + f(u, v)
    // g(t, u, v)
    fn call_inplace(&self, x: &Eqn::V, t: Eqn::T, y: &mut Eqn::V) {
        // input x = (du, v)
        // self.y0 = (u, v)
        let mut y0 = self.y0.borrow_mut();
        y0.copy_from_indices(x, &self.algebraic_indices);

        // y = (f; g)
        self.eqn.rhs().call_inplace(&y0, t, y);

        // y = -M x + y
        self.neg_mass.gemv(Eqn::T::one(), x, Eqn::T::one(), y);
    }
}

impl<Eqn: OdeEquationsImplicit> NonLinearOpJacobian for InitOp<'_, Eqn> {
    // J v
    fn jac_mul_inplace(&self, _x: &Eqn::V, _t: Eqn::T, v: &Eqn::V, y: &mut Eqn::V) {
        self.jac.gemv(Eqn::T::one(), v, Eqn::T::one(), y);
    }

    // M - c * f'(y)
    fn jacobian_inplace(&self, _x: &Self::V, _t: Self::T, y: &mut Self::M) {
        y.copy_from(&self.jac);
    }

    fn jacobian_sparsity(&self) -> Option<<Self::M as Matrix>::Sparsity> {
        self.jac.sparsity().map(|x| x.to_owned())
    }
}

#[cfg(test)]
mod tests {

    use crate::ode_solver::test_models::exponential_decay_with_algebraic::exponential_decay_with_algebraic_problem;
    use crate::op::init::InitOp;
    use crate::vector::Vector;
    use crate::{NonLinearOp, NonLinearOpJacobian};

    type Mcpu = nalgebra::DMatrix<f64>;
    type Vcpu = nalgebra::DVector<f64>;

    #[test]
    fn test_initop() {
        let (problem, _soln) = exponential_decay_with_algebraic_problem::<Mcpu>(false);
        let y0 = Vcpu::from_vec(vec![1.0, 2.0, 3.0]);
        let dy0 = Vcpu::from_vec(vec![4.0, 5.0, 6.0]);
        let t = 0.0;
        let initop = InitOp::new(&problem.eqn, t, &y0);
        // check that the init function is correct
        let mut y_out = Vcpu::from_vec(vec![0.0, 0.0, 0.0]);

        // -M_u du + f(u, v)
        // g(t, u, v)
        // M = |1 0 0|
        //     |0 1 0|
        //     |0 0 0|
        //
        // y = |1| (u)
        //     |2| (u)
        //     |3| (v)
        // dy = |4| (du)
        //      |5| (du)
        //      |6| (dv)
        // i.e. f(u, v) = -0.1 u = |-0.1|
        //                         |-0.2|
        //      g(u, v) = v - u = |1|
        //      M_u = |1 0|
        //            |0 1|
        //  i.e. F(y) = |-1 * 4 + -0.1 * 1| = |-4.1|
        //              |-1 * 5 + -0.1 * 2|   |-5.2|
        //              |2 - 1|               |1|
        let du_v = Vcpu::from_vec(vec![dy0[0], dy0[1], y0[2]]);
        initop.call_inplace(&du_v, t, &mut y_out);
        let y_out_expect = Vcpu::from_vec(vec![-4.1, -5.2, 1.0]);
        y_out.assert_eq_st(&y_out_expect, 1e-10);

        // df/dv = |0|
        //         |0|
        // dg/dv = |1|
        // J = (-M_u, df/dv) = |-1 0 0|
        //                   = |0 -1 0|
        //     (0,    dg/dv) = |0 0 1|
        let jac = initop.jacobian(&du_v, t);
        assert_eq!(jac[(0, 0)], -1.0);
        assert_eq!(jac[(0, 1)], 0.0);
        assert_eq!(jac[(0, 2)], 0.0);
        assert_eq!(jac[(1, 0)], 0.0);
        assert_eq!(jac[(1, 1)], -1.0);
        assert_eq!(jac[(1, 2)], 0.0);
        assert_eq!(jac[(2, 0)], 0.0);
        assert_eq!(jac[(2, 1)], 0.0);
        assert_eq!(jac[(2, 2)], 1.0);
    }
}
