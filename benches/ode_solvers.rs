use criterion::{criterion_group, criterion_main, Criterion};
use diffsol::{
    ode_solver::test_models::{
        exponential_decay::exponential_decay_problem, foodweb::foodweb_problem,
        foodweb::FoodWebContext, heat2d::head2d_problem, robertson::robertson,
        robertson_ode::robertson_ode,
    },
    FaerLU, FaerSparseLU, NalgebraLU, SparseColMat,
};

#[cfg(feature = "suitesparse")]
use diffsol::KLU;

mod sundials_benches;

fn criterion_benchmark(c: &mut Criterion) {
    macro_rules! bench {
        ($name:ident, $solver:ident, $linear_solver:ident, $model:ident, $model_problem:ident, $matrix:ty) => {
            c.bench_function(stringify!($name), |b| {
                b.iter(|| {
                    let ls = $linear_solver::default();
                    let (problem, soln) = $model_problem::<$matrix>(false);
                    benchmarks::$solver(&problem, soln.solution_points.last().unwrap().t, ls);
                })
            });
        };
    }

    bench!(
        nalgebra_bdf_exponential_decay,
        bdf,
        NalgebraLU,
        exponential_decay,
        exponential_decay_problem,
        nalgebra::DMatrix<f64>
    );
    bench!(
        nalgebra_esdirk34_exponential_decay,
        esdirk34,
        NalgebraLU,
        exponential_decay,
        exponential_decay_problem,
        nalgebra::DMatrix<f64>
    );
    bench!(
        nalgebra_tr_bdf2_exponential_decay,
        tr_bdf2,
        NalgebraLU,
        exponential_decay,
        exponential_decay_problem,
        nalgebra::DMatrix<f64>
    );
    bench!(
        nalgebra_bdf_robertson,
        bdf,
        NalgebraLU,
        robertson,
        robertson,
        nalgebra::DMatrix<f64>
    );
    bench!(
        nalgebra_esdirk34_robertson,
        esdirk34,
        NalgebraLU,
        robertson,
        robertson,
        nalgebra::DMatrix<f64>
    );
    bench!(
        nalgebra_tr_bdf2_robertson,
        tr_bdf2,
        NalgebraLU,
        robertson,
        robertson,
        nalgebra::DMatrix<f64>
    );
    bench!(
        faer_bdf_exponential_decay,
        bdf,
        FaerLU,
        exponential_decay,
        exponential_decay_problem,
        faer::Mat<f64>
    );
    bench!(
        faer_esdirk34_exponential_decay,
        esdirk34,
        FaerLU,
        exponential_decay,
        exponential_decay_problem,
        faer::Mat<f64>
    );
    bench!(
        faer_tr_bdf2_exponential_decay,
        tr_bdf2,
        FaerLU,
        exponential_decay,
        exponential_decay_problem,
        faer::Mat<f64>
    );
    bench!(
        faer_bdf_robertson,
        bdf,
        FaerLU,
        robertson,
        robertson,
        faer::Mat<f64>
    );
    bench!(
        faer_esdirk34_robertson,
        esdirk34,
        FaerLU,
        robertson,
        robertson,
        faer::Mat<f64>
    );
    bench!(
        faer_tr_bdf2_robertson,
        tr_bdf2,
        FaerLU,
        robertson,
        robertson,
        faer::Mat<f64>
    );

    macro_rules! bench_robertson_ode {
        ($name:ident, $solver:ident, $linear_solver:ident, $model:ident, $model_problem:ident, $matrix:ty,  $($N:expr),+) => {
            $(c.bench_function(concat!(stringify!($name), "_", $N), |b| {
                b.iter(|| {
                    let ls = $linear_solver::default();
                    let (problem, soln) = $model_problem::<$matrix>(false, $N);
                    benchmarks::$solver(&problem, soln.solution_points.last().unwrap().t, ls);
                })
            });)+
        };
    }

    bench_robertson_ode!(
        faer_sparse_bdf_robertson_ode,
        bdf,
        FaerSparseLU,
        robertson_ode,
        robertson_ode,
        SparseColMat<f64>,
        25,
        100,
        400,
        900
    );

    #[cfg(feature = "suitesparse")]
    bench_robertson_ode!(
        faer_sparse_bdf_klu_robertson_ode,
        bdf,
        KLU,
        robertson_ode,
        robertson_ode,
        SparseColMat<f64>,
        25,
        100,
        400,
        900
    );

    bench_robertson_ode!(
        faer_sparse_tr_bdf2_robertson_ode,
        tr_bdf2,
        FaerSparseLU,
        robertson_ode,
        robertson_ode,
        SparseColMat<f64>,
        25,
        100,
        400,
        900
    );

    #[cfg(feature = "suitesparse")]
    bench_robertson_ode!(
        faer_sparse_tr_bdf2_klu_robertson_ode,
        tr_bdf2,
        KLU,
        robertson_ode,
        robertson_ode,
        SparseColMat<f64>,
        25,
        100,
        400,
        900
    );

    bench_robertson_ode!(
        faer_sparse_esdirk_robertson_ode,
        esdirk34,
        FaerSparseLU,
        robertson_ode,
        robertson_ode,
        SparseColMat<f64>,
        25,
        100,
        400,
        900
    );

    #[cfg(feature = "suitesparse")]
    bench_robertson_ode!(
        faer_sparse_esdirk_klu_robertson_ode,
        esdirk34,
        KLU,
        robertson_ode,
        robertson_ode,
        SparseColMat<f64>,
        25,
        100,
        400,
        900
    );

    macro_rules! bench_diffsl_robertson {
        ($name:ident, $solver:ident, $linear_solver:ident, $matrix:ty) => {
            #[cfg(feature = "diffsl-llvm")]
            c.bench_function(stringify!($name), |b| {
                use diffsol::diffsl::LlvmModule;
                use diffsol::ode_solver::test_models::robertson::*;
                let mut context = diffsol::DiffSlContext::default();
                robertson_diffsl_compile::<$matrix, LlvmModule>(&mut context);
                b.iter(|| {
                    let (problem, soln) = robertson_diffsl_problem(&mut context, false);
                    let ls = $linear_solver::default();
                    benchmarks::$solver(&problem, soln.solution_points.last().unwrap().t, ls)
                })
            });
        };
    }

    bench_diffsl_robertson!(
        nalgebra_bdf_diffsl_robertson,
        bdf,
        NalgebraLU,
        nalgebra::DMatrix<f64>
    );

    macro_rules! bench_wsize {
        ($name:ident, $solver:ident, $linear_solver:ident, $model:ident, $model_problem:ident, $matrix:ty, $($N:expr),+) => {
            $(c.bench_function(concat!(stringify!($name), "_", $N), |b| {
                b.iter(|| {
                    let (problem, soln) = $model_problem::<$matrix, $N>();
                    let ls = $linear_solver::default();
                    benchmarks::$solver(&problem, soln.solution_points.last().unwrap().t, ls)
                })
            });)+
        };
    }

    bench_wsize!(
        faer_sparse_bdf_heat2d,
        bdf,
        FaerSparseLU,
        heat2d,
        head2d_problem,
        SparseColMat<f64>,
        5,
        10,
        20,
        30
    );

    #[cfg(feature = "suitesparse")]
    bench_wsize!(
        faer_sparse_bdf_klu_heat2d,
        bdf,
        KLU,
        heat2d,
        head2d_problem,
        SparseColMat<f64>,
        5,
        10,
        20,
        30
    );

    bench_wsize!(
        faer_sparse_tr_bdf2_heat2d,
        tr_bdf2,
        FaerSparseLU,
        heat2d,
        head2d_problem,
        SparseColMat<f64>,
        5,
        10,
        20,
        30
    );

    #[cfg(feature = "suitesparse")]
    bench_wsize!(
        faer_sparse_tr_bdf2_klu_heat2d,
        tr_bdf2,
        KLU,
        heat2d,
        head2d_problem,
        SparseColMat<f64>,
        5,
        10,
        20,
        30
    );
    bench_wsize!(
        faer_sparse_esdirk_heat2d,
        esdirk34,
        FaerSparseLU,
        heat2d,
        head2d_problem,
        SparseColMat<f64>,
        5,
        10,
        20,
        30
    );

    #[cfg(feature = "suitesparse")]
    bench_wsize!(
        faer_sparse_esdirk_klu_heat2d,
        esdirk34,
        KLU,
        heat2d,
        head2d_problem,
        SparseColMat<f64>,
        5,
        10,
        20,
        30
    );

    macro_rules! bench_foodweb {
        ($name:ident, $solver:ident, $linear_solver:ident, $model:ident, $model_problem:ident, $matrix:ty, $($N:expr),+) => {
            $(c.bench_function(concat!(stringify!($name), "_", $N), |b| {
                b.iter(|| {
                    let context = FoodWebContext::default();
                    let (problem, soln) = $model_problem::<$matrix, $N>(&context);
                    let ls = $linear_solver::default();
                    benchmarks::$solver(&problem, soln.solution_points.last().unwrap().t, ls)
                })
            });)+
        };
    }

    bench_foodweb!(
        faer_sparse_bdf_foodweb,
        bdf,
        FaerSparseLU,
        foodweb,
        foodweb_problem,
        SparseColMat<f64>,
        5,
        10,
        20,
        30
    );

    #[cfg(feature = "suitesparse")]
    bench_foodweb!(
        faer_sparse_bdf_klu_foodweb,
        bdf,
        KLU,
        foodweb,
        foodweb_problem,
        SparseColMat<f64>,
        5,
        10,
        20,
        30
    );
    bench_foodweb!(
        faer_sparse_tr_bdf2_foodweb,
        tr_bdf2,
        FaerSparseLU,
        foodweb,
        foodweb_problem,
        SparseColMat<f64>,
        5,
        10,
        20,
        30
    );

    #[cfg(feature = "suitesparse")]
    bench_foodweb!(
        faer_sparse_tr_bdf2_klu_foodweb,
        tr_bdf2,
        KLU,
        foodweb,
        foodweb_problem,
        SparseColMat<f64>,
        5,
        10,
        20,
        30
    );
    bench_foodweb!(
        faer_sparse_esdirk_foodweb,
        esdirk34,
        FaerSparseLU,
        foodweb,
        foodweb_problem,
        SparseColMat<f64>,
        5,
        10,
        20,
        30
    );
    #[cfg(feature = "suitesparse")]
    bench_foodweb!(
        faer_sparse_esdirk_klu_foodweb,
        esdirk34,
        KLU,
        foodweb,
        foodweb_problem,
        SparseColMat<f64>,
        5,
        10,
        20,
        30
    );

    macro_rules! bench_diffsl_heat2d {
        ($name:ident, $solver:ident, $linear_solver:ident, $matrix:ty, $($N:expr),+) => {
            $(#[cfg(feature = "diffsl-llvm")]
            c.bench_function(concat!(stringify!($name), "_", $N), |b| {
                use diffsol::ode_solver::test_models::heat2d::*;
                use diffsol::diffsl::LlvmModule;
                let mut context = diffsol::DiffSlContext::default();
                heat2d_diffsl_compile::<$matrix, LlvmModule, $N>(&mut context);
                b.iter(|| {
                    let (problem, soln) = heat2d_diffsl_problem(&mut context);
                    let ls = $linear_solver::default();
                    benchmarks::$solver(&problem, soln.solution_points.last().unwrap().t, ls)
                })
            });)+
        };
    }
    bench_diffsl_heat2d!(
        faer_sparse_bdf_diffsl_heat2d,
        bdf,
        FaerSparseLU,
        SparseColMat<f64>,
        5,
        10,
        20,
        30
    );

    #[cfg(feature = "suitesparse")]
    bench_diffsl_heat2d!(
        faer_sparse_bdf_klu_diffsl_heat2d,
        bdf,
        KLU,
        SparseColMat<f64>,
        5,
        10,
        20,
        30
    );

    macro_rules! bench_sundials {
        ($name:ident, $solver:ident) => {
            #[cfg(feature = "sundials")]
            c.bench_function(stringify!($name), |b| {
                b.iter(|| unsafe { sundials_benches::$solver() })
            });
        };
        () => {};
    }

    bench_sundials!(sundials_heat2d_klu_5, idaHeat2d_klu_5);
    bench_sundials!(sundials_heat2d_klu_10, idaHeat2d_klu_10);
    bench_sundials!(sundials_heat2d_klu_20, idaHeat2d_klu_20);
    bench_sundials!(sundials_heat2d_klu_30, idaHeat2d_klu_30);
    bench_sundials!(sundials_foodweb_bnd_5, idaFoodWeb_bnd_5);
    bench_sundials!(sundials_foodweb_bnd_10, idaFoodWeb_bnd_10);
    bench_sundials!(sundials_foodweb_bnd_20, idaFoodWeb_bnd_20);
    bench_sundials!(sundials_foodweb_bnd_30, idaFoodWeb_bnd_30);
    bench_sundials!(sundials_roberts_dns, idaRoberts_dns);

    macro_rules! bench_sundials_ngroups {
        ($name:ident, $solver:ident, $($N:expr),+) => {
            $(#[cfg(feature = "sundials")]
            c.bench_function(concat!(stringify!($name), "_", $N), |b| {
                b.iter(|| unsafe { sundials_benches::$solver($N) })
            });)+
        };
    }

    bench_sundials_ngroups!(
        sundials_robertson_ode_klu,
        cvRoberts_block_klu,
        25,
        100,
        400,
        900
    );

    macro_rules! bench_diffsl_foodweb {
        ($name:ident, $solver:ident, $linear_solver:ident, $matrix:ty, $($N:expr),+) => {
            $(#[cfg(feature = "diffsl-llvm")]
            c.bench_function(concat!(stringify!($name), "_", $N), |b| {
                use diffsol::ode_solver::test_models::foodweb::*;
                use diffsol::diffsl::LlvmModule;
                let mut context = diffsol::DiffSlContext::default();
                foodweb_diffsl_compile::<$matrix, LlvmModule, $N>(&mut context);
                b.iter(|| {
                    let (problem, soln) = foodweb_diffsl_problem(&mut context);
                    let ls = $linear_solver::default();
                    benchmarks::$solver(&problem, soln.solution_points.last().unwrap().t, ls)
                })
            });)+

        };
    }

    bench_diffsl_foodweb!(
        faer_sparse_bdf_diffsl_foodweb,
        bdf,
        FaerSparseLU,
        SparseColMat<f64>,
        5,
        10,
        20,
        30
    );

    #[cfg(feature = "suitesparse")]
    bench_diffsl_foodweb!(
        faer_sparse_bdf_klu_diffsl_foodweb,
        bdf,
        KLU,
        SparseColMat<f64>,
        5,
        10,
        20,
        30
    );
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);

mod benchmarks {
    use diffsol::matrix::MatrixRef;
    use diffsol::vector::VectorRef;
    use diffsol::LinearSolver;
    use diffsol::{
        Bdf, DefaultDenseMatrix, DefaultSolver, Matrix, NewtonNonlinearSolver,
        OdeEquationsImplicit, OdeSolverMethod, OdeSolverProblem, OdeSolverState, Sdirk, Tableau,
    };

    // bdf
    pub fn bdf<Eqn>(
        problem: &OdeSolverProblem<Eqn>,
        t: Eqn::T,
        ls: impl LinearSolver<Eqn::M>,
    ) where
        Eqn: OdeEquationsImplicit,
        Eqn::M: Matrix + DefaultSolver,
        Eqn::V: DefaultDenseMatrix,
        for<'a> &'a Eqn::V: VectorRef<Eqn::V>,
        for<'a> &'a Eqn::M: MatrixRef<Eqn::M>,
    {
        let nls = NewtonNonlinearSolver::new(ls);
        let mut s = Bdf::<<Eqn::V as DefaultDenseMatrix>::M, _, _>::new(nls);
        let state = OdeSolverState::new(problem, &s).unwrap();
        let _y = s.solve(problem, state, t);
    }

    pub fn esdirk34<Eqn>(
        problem: &OdeSolverProblem<Eqn>,
        t: Eqn::T,
        linear_solver: impl LinearSolver<Eqn::M>,
    ) where
        Eqn: OdeEquationsImplicit,
        Eqn::M: Matrix + DefaultSolver,
        Eqn::V: DefaultDenseMatrix,
        for<'a> &'a Eqn::V: VectorRef<Eqn::V>,
        for<'a> &'a Eqn::M: MatrixRef<Eqn::M>,
    {
        let tableau = Tableau::<<Eqn::V as DefaultDenseMatrix>::M>::esdirk34();
        let mut s = Sdirk::new(tableau, linear_solver);
        let state = OdeSolverState::new(problem, &s).unwrap();
        let _y = s.solve(problem, state, t);
    }

    pub fn tr_bdf2<Eqn>(
        problem: &OdeSolverProblem<Eqn>,
        t: Eqn::T,
        linear_solver: impl LinearSolver<Eqn::M>,
    ) where
        Eqn: OdeEquationsImplicit,
        Eqn::M: Matrix + DefaultSolver,
        Eqn::V: DefaultDenseMatrix,
        for<'a> &'a Eqn::V: VectorRef<Eqn::V>,
        for<'a> &'a Eqn::M: MatrixRef<Eqn::M>,
    {
        let tableau = Tableau::<<Eqn::V as DefaultDenseMatrix>::M>::tr_bdf2();
        let mut s = Sdirk::new(tableau, linear_solver);
        let state = OdeSolverState::new(problem, &s).unwrap();
        let _y = s.solve(problem, state, t);
    }
}
