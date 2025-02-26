use crate::{
    op::{constant_op::ConstantOpSensAdjoint, linear_op::LinearOpTranspose, ParameterisedOp},
    ConstantOp, ConstantOpSens, LinearOp, Matrix, NonLinearOp, NonLinearOpAdjoint,
    NonLinearOpJacobian, NonLinearOpSens, NonLinearOpSensAdjoint, Op, Vector,
};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct OdeEquationsStatistics {
    pub number_of_rhs_evals: usize,
    pub number_of_jac_mul_evals: usize,
    pub number_of_mass_evals: usize,
    pub number_of_mass_matrix_evals: usize,
    pub number_of_jacobian_matrix_evals: usize,
}

impl Default for OdeEquationsStatistics {
    fn default() -> Self {
        Self::new()
    }
}

impl OdeEquationsStatistics {
    pub fn new() -> Self {
        Self {
            number_of_rhs_evals: 0,
            number_of_jac_mul_evals: 0,
            number_of_mass_evals: 0,
            number_of_mass_matrix_evals: 0,
            number_of_jacobian_matrix_evals: 0,
        }
    }
}

pub trait AugmentedOdeEquations<Eqn: OdeEquations>:
    OdeEquations<T = Eqn::T, V = Eqn::V, M = Eqn::M> + Clone
{
    fn update_rhs_out_state(&mut self, y: &Eqn::V, dy: &Eqn::V, t: Eqn::T);
    fn update_init_state(&mut self, t: Eqn::T);
    fn set_index(&mut self, index: usize);
    fn max_index(&self) -> usize;
    fn include_in_error_control(&self) -> bool;
    fn include_out_in_error_control(&self) -> bool;
    fn rtol(&self) -> Option<Eqn::T>;
    fn atol(&self) -> Option<&Eqn::V>;
    fn out_rtol(&self) -> Option<Eqn::T>;
    fn out_atol(&self) -> Option<&Eqn::V>;
    fn integrate_main_eqn(&self) -> bool;
}

pub trait AugmentedOdeEquationsImplicit<Eqn: OdeEquationsImplicit>:
    AugmentedOdeEquations<Eqn> + OdeEquationsImplicit<T = Eqn::T, V = Eqn::V, M = Eqn::M>
{
}

impl<Aug, Eqn> AugmentedOdeEquationsImplicit<Eqn> for Aug
where
    Aug: AugmentedOdeEquations<Eqn> + OdeEquationsImplicit<T = Eqn::T, V = Eqn::V, M = Eqn::M>,
    Eqn: OdeEquationsImplicit,
{
}

pub struct NoAug<Eqn: OdeEquations> {
    _phantom: std::marker::PhantomData<Eqn>,
}

impl<Eqn: OdeEquations> Clone for NoAug<Eqn> {
    fn clone(&self) -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<Eqn> Op for NoAug<Eqn>
where
    Eqn: OdeEquations,
{
    type T = Eqn::T;
    type V = Eqn::V;
    type M = Eqn::M;

    fn nout(&self) -> usize {
        panic!("This should never be called")
    }
    fn nparams(&self) -> usize {
        panic!("This should never be called")
    }
    fn nstates(&self) -> usize {
        panic!("This should never be called")
    }
    fn statistics(&self) -> crate::op::OpStatistics {
        panic!("This should never be called")
    }
}

impl<'a, Eqn: OdeEquations> OdeEquationsRef<'a> for NoAug<Eqn> {
    type Mass = <Eqn as OdeEquationsRef<'a>>::Mass;
    type Rhs = <Eqn as OdeEquationsRef<'a>>::Rhs;
    type Root = <Eqn as OdeEquationsRef<'a>>::Root;
    type Init = <Eqn as OdeEquationsRef<'a>>::Init;
    type Out = <Eqn as OdeEquationsRef<'a>>::Out;
}

impl<Eqn: OdeEquations> OdeEquations for NoAug<Eqn> {
    fn rhs(&self) -> <Self as OdeEquationsRef<'_>>::Rhs {
        panic!("This should never be called")
    }

    fn mass(&self) -> Option<<Self as OdeEquationsRef<'_>>::Mass> {
        panic!("This should never be called")
    }

    fn root(&self) -> Option<<Self as OdeEquationsRef<'_>>::Root> {
        panic!("This should never be called")
    }

    fn out(&self) -> Option<<Self as OdeEquationsRef<'_>>::Out> {
        panic!("This should never be called")
    }

    fn init(&self) -> <Self as OdeEquationsRef<'_>>::Init {
        panic!("This should never be called")
    }

    fn set_params(&mut self, _p: &Self::V) {
        panic!("This should never be called")
    }

    fn get_params(&self, _p: &mut Self::V) {
        panic!("This should never be called")
    }
}

impl<Eqn: OdeEquationsImplicit> AugmentedOdeEquations<Eqn> for NoAug<Eqn> {
    fn update_rhs_out_state(&mut self, _y: &Eqn::V, _dy: &Eqn::V, _t: Eqn::T) {
        panic!("This should never be called")
    }
    fn update_init_state(&mut self, _t: Eqn::T) {
        panic!("This should never be called")
    }
    fn set_index(&mut self, _index: usize) {
        panic!("This should never be called")
    }
    fn atol(&self) -> Option<&<Eqn as Op>::V> {
        panic!("This should never be called")
    }
    fn include_out_in_error_control(&self) -> bool {
        panic!("This should never be called")
    }
    fn out_atol(&self) -> Option<&<Eqn as Op>::V> {
        panic!("This should never be called")
    }
    fn out_rtol(&self) -> Option<<Eqn as Op>::T> {
        panic!("This should never be called")
    }
    fn rtol(&self) -> Option<<Eqn as Op>::T> {
        panic!("This should never be called")
    }
    fn max_index(&self) -> usize {
        panic!("This should never be called")
    }
    fn include_in_error_control(&self) -> bool {
        panic!("This should never be called")
    }
    fn integrate_main_eqn(&self) -> bool {
        panic!("This should never be called")
    }
}

/// this is the reference trait that defines the ODE equations of the form, this is used to define the ODE equations for a given lifetime.
/// See [OdeEquations] for the main trait that defines the ODE equations.
///
/// $$
///  M \frac{dy}{dt} = F(t, y)
///  y(t_0) = y_0(t_0)
/// $$
///
/// The ODE equations are defined by:
/// - the right-hand side function `F(t, y)`, which is given as a [NonLinearOp] using the `Rhs` associated type and [OdeEquations::rhs] function,
/// - the initial condition `y_0(t_0)`, which is given as a [ConstantOp] using the `Init` associated type and [OdeEquations::init] function.
///
/// Optionally, the ODE equations can also include:
/// - the mass matrix `M` which is given as a [LinearOp] using the `Mass` associated type and the [OdeEquations::mass] function,
/// - the root function `G(t, y)` which is given as a [NonLinearOp] using the `Root` associated type and the [OdeEquations::root] function
/// - the output function `H(t, y)` which is given as a [NonLinearOp] using the `Out` associated type and the [OdeEquations::out] function
pub trait OdeEquationsRef<'a, ImplicitBounds: Sealed = Bounds<&'a Self>>: Op {
    type Mass: LinearOp<M = Self::M, V = Self::V, T = Self::T>;
    type Rhs: NonLinearOp<M = Self::M, V = Self::V, T = Self::T>;
    type Root: NonLinearOp<M = Self::M, V = Self::V, T = Self::T>;
    type Init: ConstantOp<M = Self::M, V = Self::V, T = Self::T>;
    type Out: NonLinearOp<M = Self::M, V = Self::V, T = Self::T>;
}

impl<'a, T: OdeEquationsRef<'a>> OdeEquationsRef<'a> for &T {
    type Mass = <T as OdeEquationsRef<'a>>::Mass;
    type Rhs = <T as OdeEquationsRef<'a>>::Rhs;
    type Root = <T as OdeEquationsRef<'a>>::Root;
    type Init = <T as OdeEquationsRef<'a>>::Init;
    type Out = <T as OdeEquationsRef<'a>>::Out;
}

// seal the trait so that users must use the provided default type for ImplicitBounds
mod sealed {
    pub trait Sealed: Sized {}
    pub struct Bounds<T>(T);
    impl<T> Sealed for Bounds<T> {}
}
use sealed::{Bounds, Sealed};

/// this is the trait that defines the ODE equations of the form
///
/// $$
///  M \frac{dy}{dt} = F(t, y)
///  y(t_0) = y_0(t_0)
/// $$
///
/// The ODE equations are defined by:
/// - the right-hand side function `F(t, y)`, which is given as a [NonLinearOp] using the `Rhs` associated type and [OdeEquations::rhs] function,
/// - the initial condition `y_0(t_0)`, which is given as a [ConstantOp] using the `Init` associated type and [OdeEquations::init] function.
///
/// Optionally, the ODE equations can also include:
/// - the mass matrix `M` which is given as a [LinearOp] using the `Mass` associated type and the [OdeEquations::mass] function,
/// - the root function `G(t, y)` which is given as a [NonLinearOp] using the `Root` associated type and the [OdeEquations::root] function
/// - the output function `H(t, y)` which is given as a [NonLinearOp] using the `Out` associated type and the [OdeEquations::out] function
pub trait OdeEquations: for<'a> OdeEquationsRef<'a> {
    /// returns the right-hand side function `F(t, y)` as a [NonLinearOp]
    fn rhs(&self) -> <Self as OdeEquationsRef<'_>>::Rhs;

    /// returns the mass matrix `M` as a [LinearOp]
    fn mass(&self) -> Option<<Self as OdeEquationsRef<'_>>::Mass>;

    /// returns the root function `G(t, y)` as a [NonLinearOp]
    fn root(&self) -> Option<<Self as OdeEquationsRef<'_>>::Root> {
        None
    }

    /// returns the output function `H(t, y)` as a [NonLinearOp]
    fn out(&self) -> Option<<Self as OdeEquationsRef<'_>>::Out> {
        None
    }

    /// returns the initial condition, i.e. `y(t)`, where `t` is the initial time
    fn init(&self) -> <Self as OdeEquationsRef<'_>>::Init;

    /// sets the current parameters of the equations
    fn set_params(&mut self, p: &Self::V);

    /// gets the current parameters of the equations
    fn get_params(&self, p: &mut Self::V);
}

//impl<'a, T: OdeEquations> OdeEquationsRef<'a> for &'a mut T {
//    type Mass = <T as OdeEquationsRef<'a>>::Mass;
//    type Rhs = <T as OdeEquationsRef<'a>>::Rhs;
//    type Root = <T as OdeEquationsRef<'a>>::Root;
//    type Init = <T as OdeEquationsRef<'a>>::Init;
//    type Out = <T as OdeEquationsRef<'a>>::Out;
//}
//
//impl<T: OdeEquations> OdeEquations for &'_ mut T {
//    fn rhs(&self) -> <Self as OdeEquationsRef<'_>>::Rhs {
//        (*self).rhs()
//    }
//
//    fn mass(&self) -> Option<<Self as OdeEquationsRef<'_>>::Mass> {
//        (*self).mass()
//    }
//
//    fn root(&self) -> Option<<Self as OdeEquationsRef<'_>>::Root> {
//        (*self).root()
//    }
//
//    fn out(&self) -> Option<<Self as OdeEquationsRef<'_>>::Out> {
//        (*self).out()
//    }
//
//    fn init(&self) -> <Self as OdeEquationsRef<'_>>::Init {
//        (*self).init()
//    }
//
//    fn set_params(&mut self, p: &Self::V) {
//        (*self).set_params(p)
//    }
//}

impl<T: OdeEquations> OdeEquations for &'_ T {
    fn rhs(&self) -> <Self as OdeEquationsRef<'_>>::Rhs {
        (*self).rhs()
    }

    fn mass(&self) -> Option<<Self as OdeEquationsRef<'_>>::Mass> {
        (*self).mass()
    }

    fn root(&self) -> Option<<Self as OdeEquationsRef<'_>>::Root> {
        (*self).root()
    }

    fn out(&self) -> Option<<Self as OdeEquationsRef<'_>>::Out> {
        (*self).out()
    }

    fn init(&self) -> <Self as OdeEquationsRef<'_>>::Init {
        (*self).init()
    }

    fn set_params(&mut self, _p: &Self::V) {
        unimplemented!()
    }

    fn get_params(&self, _p: &mut Self::V) {
        unimplemented!()
    }
}

pub trait OdeEquationsImplicit:
    OdeEquations<Rhs: NonLinearOpJacobian<M = Self::M, V = Self::V, T = Self::T>>
{
}

impl<T> OdeEquationsImplicit for T where
    T: OdeEquations<Rhs: NonLinearOpJacobian<M = T::M, V = T::V, T = T::T>>
{
}

pub trait OdeEquationsSens:
    OdeEquationsImplicit<
    Rhs: NonLinearOpSens<M = Self::M, V = Self::V, T = Self::T>,
    Init: ConstantOpSens<M = Self::M, V = Self::V, T = Self::T>,
>
{
}

impl<T> OdeEquationsSens for T where
    T: OdeEquationsImplicit<
        Rhs: NonLinearOpSens<M = T::M, V = T::V, T = T::T>,
        Init: ConstantOpSens<M = T::M, V = T::V, T = T::T>,
    >
{
}

pub trait OdeEquationsAdjoint:
    OdeEquationsImplicit<
    Rhs: NonLinearOpAdjoint<M = Self::M, V = Self::V, T = Self::T>
             + NonLinearOpSensAdjoint<M = Self::M, V = Self::V, T = Self::T>,
    Init: ConstantOpSensAdjoint<M = Self::M, V = Self::V, T = Self::T>,
    Out: NonLinearOpAdjoint<M = Self::M, V = Self::V, T = Self::T>
             + NonLinearOpSensAdjoint<M = Self::M, V = Self::V, T = Self::T>,
    Mass: LinearOpTranspose<M = Self::M, V = Self::V, T = Self::T>,
>
{
}

impl<T> OdeEquationsAdjoint for T where
    T: OdeEquationsImplicit<
        Rhs: NonLinearOpAdjoint<M = T::M, V = T::V, T = T::T>
                 + NonLinearOpSensAdjoint<M = T::M, V = T::V, T = T::T>,
        Init: ConstantOpSensAdjoint<M = T::M, V = T::V, T = T::T>,
        Out: NonLinearOpAdjoint<M = T::M, V = T::V, T = T::T>
                 + NonLinearOpSensAdjoint<M = T::M, V = T::V, T = T::T>,
        Mass: LinearOpTranspose<M = T::M, V = T::V, T = T::T>,
    >
{
}

/// This struct implements the ODE equation trait [OdeEquations] for a given right-hand side op, mass op, optional root op, and initial condition function.
///
/// While the [crate::OdeBuilder] struct is the easiest way to define an ODE problem,
/// occasionally a user might want to use their own structs that define the equations instead of closures or the DiffSL languave,
/// and this can be done using [OdeSolverEquations].
///
/// The main traits that you need to implement are the [crate::Op] and [NonLinearOp] trait,
/// which define a nonlinear operator or function `F` that maps an input vector `x` to an output vector `y`, (i.e. `y = F(x)`).
/// Once you have implemented this trait, you can then pass an instance of your struct to the `rhs` argument of the [Self::new] method.
/// Once you have created an instance of [OdeSolverEquations], you can then use [crate::OdeBuilder::build_from_eqn] to create a problem.
pub struct OdeSolverEquations<M, Rhs, Init, Mass, Root, Out>
where
    M: Matrix,
{
    rhs: Rhs,
    mass: Option<Mass>,
    root: Option<Root>,
    init: Init,
    out: Option<Out>,
    p: M::V,
}

impl<M, Rhs, Init, Mass, Root, Out> OdeSolverEquations<M, Rhs, Init, Mass, Root, Out>
where
    M: Matrix,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        rhs: Rhs,
        init: Init,
        mass: Option<Mass>,
        root: Option<Root>,
        out: Option<Out>,
        p: M::V,
    ) -> Self {
        Self {
            rhs,
            mass,
            root,
            init,
            out,
            p,
        }
    }
    fn params_mut(&mut self) -> &mut M::V {
        &mut self.p
    }
    fn params(&self) -> &M::V {
        &self.p
    }
}

impl<M, Rhs, Init, Mass, Root, Out> Op for OdeSolverEquations<M, Rhs, Init, Mass, Root, Out>
where
    M: Matrix,
    Init: Op<M = M, V = M::V, T = M::T>,
    Rhs: Op<M = M, V = M::V, T = M::T>,
    Mass: Op<M = M, V = M::V, T = M::T>,
    Root: Op<M = M, V = M::V, T = M::T>,
    Out: Op<M = M, V = M::V, T = M::T>,
{
    type T = M::T;
    type V = M::V;
    type M = M;
    fn nstates(&self) -> usize {
        self.init.nstates()
    }
    fn nout(&self) -> usize {
        self.out
            .as_ref()
            .map(|out| out.nout())
            .unwrap_or(self.rhs.nout())
    }
    fn nparams(&self) -> usize {
        self.rhs.nparams()
    }
    fn statistics(&self) -> crate::op::OpStatistics {
        self.rhs.statistics()
    }
}

impl<'a, M, Rhs, Init, Mass, Root, Out> OdeEquationsRef<'a>
    for OdeSolverEquations<M, Rhs, Init, Mass, Root, Out>
where
    M: Matrix,
    Rhs: Op<M = M, V = M::V, T = M::T>,
    Init: Op<M = M, V = M::V, T = M::T>,
    Mass: Op<M = M, V = M::V, T = M::T>,
    Root: Op<M = M, V = M::V, T = M::T>,
    Out: Op<M = M, V = M::V, T = M::T>,
    ParameterisedOp<'a, Rhs>: NonLinearOp<M = M, V = M::V, T = M::T>,
    ParameterisedOp<'a, Init>: ConstantOp<M = M, V = M::V, T = M::T>,
    ParameterisedOp<'a, Mass>: LinearOp<M = M, V = M::V, T = M::T>,
    ParameterisedOp<'a, Root>: NonLinearOp<M = M, V = M::V, T = M::T>,
    ParameterisedOp<'a, Out>: NonLinearOp<M = M, V = M::V, T = M::T>,
{
    type Rhs = ParameterisedOp<'a, Rhs>;
    type Mass = ParameterisedOp<'a, Mass>;
    type Root = ParameterisedOp<'a, Root>;
    type Init = ParameterisedOp<'a, Init>;
    type Out = ParameterisedOp<'a, Out>;
}

impl<M, Rhs, Init, Mass, Root, Out> OdeEquations
    for OdeSolverEquations<M, Rhs, Init, Mass, Root, Out>
where
    M: Matrix,
    Rhs: Op<M = M, V = M::V, T = M::T>,
    Init: Op<M = M, V = M::V, T = M::T>,
    Mass: Op<M = M, V = M::V, T = M::T>,
    Root: Op<M = M, V = M::V, T = M::T>,
    Out: Op<M = M, V = M::V, T = M::T>,
    for<'a> ParameterisedOp<'a, Rhs>: NonLinearOp<M = M, V = M::V, T = M::T>,
    for<'a> ParameterisedOp<'a, Init>: ConstantOp<M = M, V = M::V, T = M::T>,
    for<'a> ParameterisedOp<'a, Mass>: LinearOp<M = M, V = M::V, T = M::T>,
    for<'a> ParameterisedOp<'a, Root>: NonLinearOp<M = M, V = M::V, T = M::T>,
    for<'a> ParameterisedOp<'a, Out>: NonLinearOp<M = M, V = M::V, T = M::T>,
{
    fn rhs(&self) -> ParameterisedOp<'_, Rhs> {
        ParameterisedOp::new(&self.rhs, self.params())
    }
    fn mass(&self) -> Option<ParameterisedOp<'_, Mass>> {
        self.mass
            .as_ref()
            .map(|mass| ParameterisedOp::new(mass, self.params()))
    }
    fn root(&self) -> Option<ParameterisedOp<'_, Root>> {
        self.root
            .as_ref()
            .map(|root| ParameterisedOp::new(root, self.params()))
    }
    fn init(&self) -> ParameterisedOp<'_, Init> {
        ParameterisedOp::new(&self.init, self.params())
    }
    fn out(&self) -> Option<ParameterisedOp<'_, Out>> {
        self.out
            .as_ref()
            .map(|out| ParameterisedOp::new(out, self.params()))
    }
    fn set_params(&mut self, p: &Self::V) {
        self.params_mut().copy_from(p);
    }
    fn get_params(&self, p: &mut Self::V) {
        p.copy_from(self.params());
    }
}

#[cfg(test)]
mod tests {
    use nalgebra::DVector;

    use crate::ode_solver::equations::OdeEquations;
    use crate::ode_solver::test_models::exponential_decay::exponential_decay_problem;
    use crate::ode_solver::test_models::exponential_decay_with_algebraic::exponential_decay_with_algebraic_problem;
    use crate::vector::Vector;
    use crate::{LinearOp, NonLinearOp, NonLinearOpJacobian};

    type Mcpu = nalgebra::DMatrix<f64>;
    type Vcpu = nalgebra::DVector<f64>;

    #[test]
    fn ode_equation_test() {
        let (problem, _soln) = exponential_decay_problem::<Mcpu>(false);
        let y = DVector::from_vec(vec![1.0, 1.0]);
        let rhs_y = problem.eqn.rhs().call(&y, 0.0);
        let expect_rhs_y = DVector::from_vec(vec![-0.1, -0.1]);
        rhs_y.assert_eq_st(&expect_rhs_y, 1e-10);
        let jac_rhs_y = problem.eqn.rhs().jac_mul(&y, 0.0, &y);
        let expect_jac_rhs_y = Vcpu::from_vec(vec![-0.1, -0.1]);
        jac_rhs_y.assert_eq_st(&expect_jac_rhs_y, 1e-10);
        assert!(problem.eqn.mass().is_none());
        let jac = problem.eqn.rhs().jacobian(&y, 0.0);
        assert_eq!(jac[(0, 0)], -0.1);
        assert_eq!(jac[(1, 1)], -0.1);
        assert_eq!(jac[(0, 1)], 0.0);
        assert_eq!(jac[(1, 0)], 0.0);
    }

    #[test]
    fn ode_with_mass_test() {
        let (problem, _soln) = exponential_decay_with_algebraic_problem::<Mcpu>(false);
        let y = DVector::from_vec(vec![1.0, 1.0, 1.0]);
        let rhs_y = problem.eqn.rhs().call(&y, 0.0);
        let expect_rhs_y = DVector::from_vec(vec![-0.1, -0.1, 0.0]);
        rhs_y.assert_eq_st(&expect_rhs_y, 1e-10);
        let jac_rhs_y = problem.eqn.rhs().jac_mul(&y, 0.0, &y);
        let expect_jac_rhs_y = Vcpu::from_vec(vec![-0.1, -0.1, 0.0]);
        jac_rhs_y.assert_eq_st(&expect_jac_rhs_y, 1e-10);
        let mass = problem.eqn.mass().unwrap().matrix(0.0);
        assert_eq!(mass[(0, 0)], 1.);
        assert_eq!(mass[(1, 1)], 1.);
        assert_eq!(mass[(2, 2)], 0.);
        assert_eq!(mass[(0, 1)], 0.);
        assert_eq!(mass[(1, 0)], 0.);
        assert_eq!(mass[(0, 2)], 0.);
        assert_eq!(mass[(2, 0)], 0.);
        assert_eq!(mass[(1, 2)], 0.);
        assert_eq!(mass[(2, 1)], 0.);
        let jac = problem.eqn.rhs().jacobian(&y, 0.0);
        assert_eq!(jac[(0, 0)], -0.1);
        assert_eq!(jac[(1, 1)], -0.1);
        assert_eq!(jac[(2, 2)], 1.0);
        assert_eq!(jac[(0, 1)], 0.0);
        assert_eq!(jac[(1, 0)], 0.0);
        assert_eq!(jac[(0, 2)], 0.0);
        assert_eq!(jac[(2, 0)], 0.0);
        assert_eq!(jac[(1, 2)], 0.0);
        assert_eq!(jac[(2, 1)], -1.0);
    }
}
