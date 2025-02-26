use crate::{
    matrix::Matrix, ode_solver::problem::OdeSolverSolution, scalar::scale, ConstantOp, OdeBuilder,
    OdeEquations, OdeEquationsAdjoint, OdeEquationsImplicit, OdeEquationsSens, OdeSolverProblem,
    Vector,
};
use nalgebra::ComplexField;
use num_traits::{One, Zero};
use std::ops::MulAssign;

// exponential decay problem
// dy/dt = -ay (p = [a, y0])
fn exponential_decay<M: Matrix>(x: &M::V, p: &M::V, _t: M::T, y: &mut M::V) {
    y.copy_from(x);
    y.mul_assign(scale(-p[0]));
}

// df/dp v = -yv (p = [a, y0])
// df/dp = | -y  0 |
//         | -y  0 |
// df/dp v = | -y  0 | |v_1| = |-yv_1|
//           | -y  0 | |v_2|   |-yv_1 |
fn exponential_decay_sens<M: Matrix>(x: &M::V, _p: &M::V, _t: M::T, v: &M::V, y: &mut M::V) {
    y.copy_from(x);
    y.mul_assign(scale(-v[0]));
}

// df/dp^T v = | -y  -y | |v_1| = |-yv_1 - yv_2|
//             |  0   0 | |v_2|   |  0  |
fn exponential_decay_sens_transpose<M: Matrix>(
    x: &M::V,
    _p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    y[0] = x[0] * v[0] + x[1] * v[1];
    y[1] = M::T::zero();
}

// J = | -a  0 |
//     | 0  -a |
// Jv = -av
fn exponential_decay_jacobian<M: Matrix>(_x: &M::V, p: &M::V, _t: M::T, v: &M::V, y: &mut M::V) {
    y.copy_from(v);
    y.mul_assign(scale(-p[0]));
}

// -J^Tv = av
fn exponential_decay_jacobian_adjoint<M: Matrix>(
    _x: &M::V,
    p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    y.copy_from(v);
    y.mul_assign(scale(p[0]));
}

fn exponential_decay_init<M: Matrix>(p: &M::V, _t: M::T) -> M::V {
    M::V::from_vec(vec![p[1], p[1]])
}

// dy0/dp = | 0 1 |
//          | 0 1 |
// dy0/dp v = | 0 1 | |v_1| = |v_2|
//            | 0 1 | |v_2|   |v_2|
fn exponential_decay_init_sens<M: Matrix>(_p: &M::V, _t: M::T, v: &M::V, y: &mut M::V) {
    y[0] = v[1];
    y[1] = v[1];
}

// dy0/dp^T v = | 0 0 | |v_1| = |0 |
//              | 1 1 | |v_2|   |v_1 + v_2|
fn exponential_decay_init_sens_adjoint<M: Matrix>(_p: &M::V, _t: M::T, v: &M::V, y: &mut M::V) {
    y[0] = M::T::zero();
    y[1] = -v[0] - v[1];
}

fn exponential_decay_root<M: Matrix>(x: &M::V, _p: &M::V, _t: M::T, y: &mut M::V) {
    y[0] = x[0] - M::T::from(0.6);
}

/// g_1 = 1 * x_1  +  2 * x_2
/// g_2 = 3 * x_1  +  4 * x_2
fn exponential_decay_out<M: Matrix>(x: &M::V, _p: &M::V, _t: M::T, y: &mut M::V) {
    y[0] = M::T::from(1.0) * x[0] + M::T::from(2.0) * x[1];
    y[1] = M::T::from(3.0) * x[0] + M::T::from(4.0) * x[1];
}

/// J = |1 2|
///    |3 4|
/// J v = |1 2| |v_1| = |v_1 + 2v_2|
///       |3 4| |v_2|   |3v_1 + 4v_2|
fn exponential_decay_out_jac_mul<M: Matrix>(
    _x: &M::V,
    _p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    y[0] = v[0] + M::T::from(2.0) * v[1];
    y[1] = M::T::from(3.0) * v[0] + M::T::from(4.0) * v[1];
}

/// J = |1 2|
///    |3 4|
/// -J^T v = |-1 -3| |v_1| = |-v_1 - 3v_2|
///         |-2 -4| |v_2|   |-2v_1 - 4v_2|
fn exponential_decay_out_adj_mul<M: Matrix>(
    _x: &M::V,
    _p: &M::V,
    _t: M::T,
    v: &M::V,
    y: &mut M::V,
) {
    y[0] = -v[0] - M::T::from(3.0) * v[1];
    y[1] = -M::T::from(2.0) * v[0] - M::T::from(4.0) * v[1];
}

/// J = |0 0|
///     |0 0|
fn exponential_decay_out_sens_adj<M: Matrix>(
    _x: &M::V,
    _p: &M::V,
    _t: M::T,
    _v: &M::V,
    y: &mut M::V,
) {
    y.fill(M::T::zero());
}

#[allow(clippy::type_complexity)]
pub fn negative_exponential_decay_problem<M: Matrix + 'static>(
    use_coloring: bool,
) -> (
    OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = M::V, T = M::T>>,
    OdeSolverSolution<M::V>,
) {
    let h = -1.0;
    let k = 0.1;
    let y0 = (-10.0 * k).exp();
    let problem = OdeBuilder::<M>::new()
        .h0(h)
        .p([k, y0])
        .use_coloring(use_coloring)
        .rhs_implicit(exponential_decay::<M>, exponential_decay_jacobian::<M>)
        .init(exponential_decay_init::<M>)
        .build()
        .unwrap();
    let p = [M::T::from(k), M::T::from(y0)];
    let mut soln = OdeSolverSolution {
        negative_time: true,
        ..Default::default()
    };
    for i in 0..10 {
        let t = M::T::from(-i as f64);
        let y0: M::V = problem.eqn.init().call(M::T::zero());
        let y = y0 * scale(M::T::exp(-p[0] * t));
        soln.push(y, t);
    }
    (problem, soln)
}

#[cfg(feature = "diffsl")]
pub fn exponential_decay_problem_diffsl<M: Matrix<T = f64>, CG: crate::CodegenModule>(
    prep_adjoint: bool,
) -> (
    OdeSolverProblem<crate::DiffSl<M, CG>>,
    OdeSolverSolution<M::V>,
) {
    let k = 0.1;
    let y0 = 1.0;
    let out = if prep_adjoint {
        "1 * x  +  2 * y, 3 * x  +  4 * y,"
    } else {
        "u_i"
    };
    let diffsl = crate::DiffSl::compile(
        format!(
            "
        in = [k, y0]
        k {{ 0.1 }}
        y0 {{ 1.0 }}
        u_i {{ x = y0, y = y0 }}
        F_i {{ -k * u_i }}
        out_i {{
            {}
        }}",
            out
        )
        .as_str(),
    )
    .unwrap();
    let problem = OdeBuilder::<M>::new()
        .p([k, y0])
        .integrate_out(prep_adjoint)
        .build_from_eqn(diffsl)
        .unwrap();
    let p = [k, y0];
    let mut soln = OdeSolverSolution {
        atol: problem.atol.clone(),
        rtol: problem.rtol,
        ..Default::default()
    };
    for i in 0..10 {
        let t = i as f64;
        let y0 = problem.eqn.init().call(0.0);
        let y = y0 * scale((-p[0] * t).exp());
        soln.push(y, t);
    }
    (problem, soln)
}

#[allow(clippy::type_complexity)]
pub fn exponential_decay_problem<M: Matrix + 'static>(
    use_coloring: bool,
) -> (
    OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = M::V, T = M::T>>,
    OdeSolverSolution<M::V>,
) {
    let h = 1.0;
    let k = 0.1;
    let y0 = 1.0;
    let problem = OdeBuilder::<M>::new()
        .h0(h)
        .p([k, y0])
        .use_coloring(use_coloring)
        .rhs_implicit(exponential_decay::<M>, exponential_decay_jacobian::<M>)
        .init(exponential_decay_init::<M>)
        .build()
        .unwrap();
    let p = [M::T::from(k), M::T::from(y0)];
    let mut soln = OdeSolverSolution::default();
    for i in 0..10 {
        let t = M::T::from(i as f64);
        let y0: M::V = problem.eqn.init().call(M::T::zero());
        let y = y0 * scale(M::T::exp(-p[0] * t));
        soln.push(y, t);
    }
    (problem, soln)
}

#[allow(clippy::type_complexity)]
pub fn exponential_decay_problem_with_root<M: Matrix + 'static>(
    use_coloring: bool,
) -> (
    OdeSolverProblem<impl OdeEquationsImplicit<M = M, V = M::V, T = M::T>>,
    OdeSolverSolution<M::V>,
) {
    let k = 0.1;
    let y0 = 1.0;
    let problem = OdeBuilder::<M>::new()
        .p([k, y0])
        .use_coloring(use_coloring)
        .rhs_implicit(exponential_decay::<M>, exponential_decay_jacobian::<M>)
        .init(exponential_decay_init::<M>)
        .root(exponential_decay_root::<M>, 1)
        .build()
        .unwrap();
    let p = [M::T::from(k), M::T::from(y0)];
    let mut soln = OdeSolverSolution::default();
    for i in 0..10 {
        let t = M::T::from(i as f64);
        let y0: M::V = problem.eqn.init().call(M::T::zero());
        let y = y0 * scale(M::T::exp(-p[0] * t));
        soln.push(y, t);
    }
    (problem, soln)
}

#[allow(clippy::type_complexity)]
pub fn exponential_decay_problem_adjoint<M: Matrix>(
    integrate_out: bool,
) -> (
    OdeSolverProblem<impl OdeEquationsAdjoint<M = M, V = M::V, T = M::T>>,
    OdeSolverSolution<M::V>,
) {
    let k = 0.1;
    let y0 = 1.0;
    let problem = OdeBuilder::<M>::new()
        .p([k, y0])
        .integrate_out(integrate_out)
        .rhs_adjoint_implicit(
            exponential_decay::<M>,
            exponential_decay_jacobian::<M>,
            exponential_decay_jacobian_adjoint::<M>,
            exponential_decay_sens_transpose::<M>,
        )
        .init_adjoint(
            exponential_decay_init::<M>,
            exponential_decay_init_sens_adjoint::<M>,
        )
        .out_adjoint_implicit(
            exponential_decay_out::<M>,
            exponential_decay_out_jac_mul::<M>,
            exponential_decay_out_adj_mul::<M>,
            exponential_decay_out_sens_adj::<M>,
            2,
        )
        .build()
        .unwrap();
    let mut soln = OdeSolverSolution {
        atol: problem.atol.clone(),
        rtol: problem.rtol,
        ..Default::default()
    };
    let t0 = M::T::from(0.0);
    let t1 = M::T::from(9.0);
    let p = [M::T::from(k), M::T::from(y0)];
    for i in 0..10 {
        let t = M::T::from(i as f64);
        let y0: M::V = problem.eqn.init().call(M::T::zero());
        let g = y0.clone() * scale((M::T::exp(-p[0] * t0) - M::T::exp(-p[0] * t)) / p[0]);
        let g = M::V::from_vec(vec![
            g[0] + M::T::from(2.0) * g[1],
            M::T::from(3.0) * g[0] + M::T::from(4.0) * g[1],
        ]);
        let dydk = y0.clone()
            * scale(
                M::T::exp(-p[0] * (t1 + t0))
                    * (M::T::exp(t0 * p[0]) * (p[0] * t1 + M::T::one())
                        - M::T::exp(t1 * p[0]) * (p[0] * t0 + M::T::one()))
                    / (p[0] * p[0]),
            );
        let dydy0 = (M::T::exp(-p[0] * t0) - M::T::exp(-p[0] * t1)) / p[0];
        let dg1dk = dydk[0] + M::T::from(2.0) * dydk[1];
        let dg2dk = M::T::from(3.0) * dydk[0] + M::T::from(4.0) * dydk[1];
        let dg1dy0 = dydy0 + M::T::from(2.0) * dydy0;
        let dg2dy0 = M::T::from(3.0) * dydy0 + M::T::from(4.0) * dydy0;
        let dg1 = M::V::from_vec(vec![dg1dk, dg1dy0]);
        let dg2 = M::V::from_vec(vec![dg2dk, dg2dy0]);
        soln.push_sens(g, t, &[dg1, dg2]);
    }
    (problem, soln)
}

#[allow(clippy::type_complexity)]
pub fn exponential_decay_problem_sens<M: Matrix + 'static>(
    use_coloring: bool,
) -> (
    OdeSolverProblem<impl OdeEquationsSens<M = M, V = M::V, T = M::T>>,
    OdeSolverSolution<M::V>,
) {
    let k = 0.1;
    let y0 = 1.0;
    let problem = OdeBuilder::<M>::new()
        .p([k, y0])
        .sens_rtol(1e-6)
        .sens_atol([1e-6, 1e-6])
        .use_coloring(use_coloring)
        .rhs_sens_implicit(
            exponential_decay::<M>,
            exponential_decay_jacobian::<M>,
            exponential_decay_sens::<M>,
        )
        .init_sens(
            exponential_decay_init::<M>,
            exponential_decay_init_sens::<M>,
        )
        .build()
        .unwrap();
    let p = [M::T::from(k), M::T::from(y0)];
    let mut soln = OdeSolverSolution::default();
    for i in 0..10 {
        let t = M::T::from(i as f64);
        let y0: M::V = problem.eqn.init().call(M::T::zero());
        let y = y0.clone() * scale(M::T::exp(-p[0] * t));
        let yp = y0 * scale(-t * M::T::exp(-p[0] * t));
        soln.push_sens(y, t, &[yp]);
    }
    (problem, soln)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "diffsl-llvm")]
    #[test]
    fn test_exponential_decay_diffsl_llvm() {
        use crate::{
            ConstantOpSens, ConstantOpSensAdjoint, NonLinearOpAdjoint, NonLinearOpJacobian,
            NonLinearOpSens, NonLinearOpSensAdjoint,
        };
        use nalgebra::{DMatrix, DVector};
        let (problem, _soln) =
            exponential_decay_problem_diffsl::<DMatrix<f64>, crate::LlvmModule>(true);
        let x = DVector::from_vec(vec![1.0, 2.0]);
        let t = 0.0;
        let v = DVector::from_vec(vec![2.0, 3.0]);
        let p = DVector::from_vec(vec![0.1, 1.0]);

        // check the adjoint jacobian
        let mut y_check = DVector::zeros(2);
        exponential_decay_jacobian_adjoint::<DMatrix<f64>>(&x, &p, t, &v, &mut y_check);
        let mut y = DVector::zeros(2);
        for _i in 0..2 {
            problem
                .eqn()
                .rhs()
                .jac_transpose_mul_inplace(&x, t, &v, &mut y);
            assert_eq!(y, y_check);
        }

        // check the sens jacobian
        let mut y_check = DVector::zeros(2);
        exponential_decay_sens::<DMatrix<f64>>(&x, &p, t, &v, &mut y_check);
        let mut y = DVector::zeros(2);
        for _i in 0..2 {
            problem.eqn().rhs().sens_mul_inplace(&x, t, &v, &mut y);
            assert_eq!(y, y_check);
        }

        // check the sens adjoint jacobian
        let mut y_check = DVector::zeros(2);
        exponential_decay_sens_transpose::<DMatrix<f64>>(&x, &p, t, &v, &mut y_check);
        let mut y = DVector::zeros(2);
        for _i in 0..2 {
            problem
                .eqn()
                .rhs()
                .sens_transpose_mul_inplace(&x, t, &v, &mut y);
            assert_eq!(y, y_check);
        }

        // check the set_u0 sens adjoint jacobian
        let mut y_check = DVector::zeros(2);
        exponential_decay_init_sens_adjoint::<DMatrix<f64>>(&p, t, &v, &mut y_check);
        let mut y = DVector::zeros(2);
        for _i in 0..2 {
            problem
                .eqn()
                .init()
                .sens_transpose_mul_inplace(t, &v, &mut y);
            assert_eq!(y, y_check);
        }

        // check the set_u0 sens jacobian
        let mut y_check = DVector::zeros(2);
        exponential_decay_init_sens::<DMatrix<f64>>(&p, t, &v, &mut y_check);
        let mut y = DVector::zeros(2);
        for _i in 0..2 {
            problem.eqn().init().sens_mul_inplace(t, &v, &mut y);
            assert_eq!(y, y_check);
        }

        // check the calc_out jacobian
        let mut y_check = DVector::zeros(2);
        exponential_decay_out_jac_mul::<DMatrix<f64>>(&x, &p, t, &v, &mut y_check);
        let mut y = DVector::zeros(2);
        for _i in 0..2 {
            problem
                .eqn()
                .out()
                .unwrap()
                .jac_mul_inplace(&x, t, &v, &mut y);
            assert_eq!(y, y_check);
        }

        // check the calc_out adjoint jacobian
        let mut y_check = DVector::zeros(2);
        exponential_decay_out_adj_mul::<DMatrix<f64>>(&x, &p, t, &v, &mut y_check);
        let mut y = DVector::zeros(2);
        for _i in 0..2 {
            problem
                .eqn()
                .out()
                .unwrap()
                .jac_transpose_mul_inplace(&x, t, &v, &mut y);
            assert_eq!(y, y_check);
        }

        // check the calc_out sens adjoint jacobian
        let mut y_check = DVector::zeros(2);
        exponential_decay_out_sens_adj::<DMatrix<f64>>(&x, &p, t, &v, &mut y_check);
        let mut y = DVector::zeros(2);
        for _i in 0..2 {
            problem
                .eqn()
                .out()
                .unwrap()
                .sens_transpose_mul_inplace(&x, t, &v, &mut y);
            assert_eq!(y, y_check);
        }
    }
}
