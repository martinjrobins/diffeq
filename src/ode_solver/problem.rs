use std::rc::Rc;

use crate::{
    error::{DiffsolError, OdeSolverError},
    ode_solver_error,
    vector::Vector,
    AugmentedOdeEquationsImplicit, Bdf, BdfState, DefaultDenseMatrix, DenseMatrix, LinearSolver,
    MatrixRef, NewtonNonlinearSolver, OdeEquations, OdeEquationsImplicit, OdeEquationsSens,
    OdeSolverState, Sdirk, SdirkState, SensEquations, Tableau, VectorRef,
};

pub struct OdeSolverProblem<Eqn>
where
    Eqn: OdeEquations,
{
    pub eqn: Rc<Eqn>,
    pub rtol: Eqn::T,
    pub atol: Rc<Eqn::V>,
    pub t0: Eqn::T,
    pub h0: Eqn::T,
    pub integrate_out: bool,
    pub sens_rtol: Option<Eqn::T>,
    pub sens_atol: Option<Rc<Eqn::V>>,
    pub out_rtol: Option<Eqn::T>,
    pub out_atol: Option<Rc<Eqn::V>>,
    pub param_rtol: Option<Eqn::T>,
    pub param_atol: Option<Rc<Eqn::V>>,
}

// impl clone
impl<Eqn> Clone for OdeSolverProblem<Eqn>
where
    Eqn: OdeEquations,
{
    fn clone(&self) -> Self {
        Self {
            eqn: self.eqn.clone(),
            rtol: self.rtol,
            atol: self.atol.clone(),
            t0: self.t0,
            h0: self.h0,
            integrate_out: self.integrate_out,
            out_atol: self.out_atol.clone(),
            out_rtol: self.out_rtol,
            param_atol: self.param_atol.clone(),
            param_rtol: self.param_rtol,
            sens_atol: self.sens_atol.clone(),
            sens_rtol: self.sens_rtol,
        }
    }
}

macro_rules! sdirk_solver_from_tableau {
    ($state:ident, $state_sens:ident, $method:ident, $method_sens:ident, $method_solver:ident, $method_solver_sens:ident, $tableau:ident) => {
        pub fn $state<LS: LinearSolver<Eqn::M>>(&self) -> Result<SdirkState<Eqn::V>, DiffsolError>
        where
            Eqn: OdeEquationsImplicit,
        {
            self.sdirk_state::<LS, _>(Tableau::<<Eqn::V as DefaultDenseMatrix>::M>::$tableau())
        }

        pub fn $state_sens<LS: LinearSolver<Eqn::M>>(
            &self,
        ) -> Result<SdirkState<Eqn::V>, DiffsolError>
        where
            Eqn: OdeEquationsSens,
        {
            self.sdirk_state_sens::<LS, _>(Tableau::<<Eqn::V as DefaultDenseMatrix>::M>::$tableau())
        }

        pub fn $method_solver<LS: LinearSolver<Eqn::M>>(
            &self,
            state: SdirkState<Eqn::V>,
        ) -> Result<Sdirk<'_, Eqn, LS>, DiffsolError>
        where
            Eqn: OdeEquationsImplicit,
        {
            self.sdirk_solver(
                state,
                Tableau::<<Eqn::V as DefaultDenseMatrix>::M>::$tableau(),
            )
        }

        pub fn $method_solver_sens<LS: LinearSolver<Eqn::M>>(
            &self,
            state: SdirkState<Eqn::V>,
        ) -> Result<Sdirk<'_, Eqn, LS, <Eqn::V as DefaultDenseMatrix>::M, SensEquations<Eqn>>, DiffsolError>
        where
            Eqn: OdeEquationsSens,
        {
            self.sdirk_solver_sens(
                state,
                Tableau::<<Eqn::V as DefaultDenseMatrix>::M>::$tableau(),
            )
        }

        pub fn $method<LS: LinearSolver<Eqn::M>>(&self) -> Result<Sdirk<'_, Eqn, LS>, DiffsolError>
        where
            Eqn: OdeEquationsImplicit,
        {
            let state = self.$state::<LS>()?;
            self.$method_solver::<LS>(state)
        }

        pub fn $method_sens<LS: LinearSolver<Eqn::M>>(
            &self,
        ) -> Result<Sdirk<'_, Eqn, LS, <Eqn::V as DefaultDenseMatrix>::M, SensEquations<Eqn>>, DiffsolError>
        where
            Eqn: OdeEquationsSens,
        {
            let state = self.$state_sens::<LS>()?;
            self.$method_solver_sens::<LS>(state)
        }
    };
}

impl<Eqn> OdeSolverProblem<Eqn>
where
    Eqn: OdeEquations,
{
    pub fn default_rtol() -> Eqn::T {
        Eqn::T::from(1e-6)
    }
    pub fn default_atol(nstates: usize) -> Eqn::V {
        Eqn::V::from_element(nstates, Eqn::T::from(1e-6))
    }
    pub fn output_in_error_control(&self) -> bool {
        self.integrate_out
            && self.eqn.out().is_some()
            && self.out_rtol.is_some()
            && self.out_atol.is_some()
    }
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        eqn: Rc<Eqn>,
        rtol: Eqn::T,
        atol: Rc<Eqn::V>,
        sens_rtol: Option<Eqn::T>,
        sens_atol: Option<Rc<Eqn::V>>,
        out_rtol: Option<Eqn::T>,
        out_atol: Option<Rc<Eqn::V>>,
        param_rtol: Option<Eqn::T>,
        param_atol: Option<Rc<Eqn::V>>,
        t0: Eqn::T,
        h0: Eqn::T,
        integrate_out: bool,
    ) -> Result<Self, DiffsolError> {
        Ok(Self {
            eqn,
            rtol,
            atol,
            out_atol,
            out_rtol,
            param_atol,
            param_rtol,
            sens_atol,
            sens_rtol,
            t0,
            h0,
            integrate_out,
        })
    }

    pub fn set_params(&mut self, p: Eqn::V) -> Result<(), DiffsolError> {
        let eqn =
            Rc::get_mut(&mut self.eqn).ok_or(ode_solver_error!(FailedToGetMutableReference))?;
        eqn.set_params(Rc::new(p));
        Ok(())
    }
}

impl<Eqn> OdeSolverProblem<Eqn>
where
    Eqn: OdeEquations,
    Eqn::V: DefaultDenseMatrix<T = Eqn::T>,
    for<'b> &'b Eqn::V: VectorRef<Eqn::V>,
    for<'b> &'b Eqn::M: MatrixRef<Eqn::M>,
{
    pub fn bdf_state<LS: LinearSolver<Eqn::M>>(&self) -> Result<BdfState<Eqn::V>, DiffsolError>
    where
        Eqn: OdeEquationsImplicit,
    {
        BdfState::new::<LS, Eqn>(self, 1)
    }

    pub fn bdf_state_sens<LS: LinearSolver<Eqn::M>>(&self) -> Result<BdfState<Eqn::V>, DiffsolError>
    where
        Eqn: OdeEquationsSens,
    {
        BdfState::new_with_sensitivities::<LS, Eqn>(self, 1)
    }

    pub fn bdf_solver<LS: LinearSolver<Eqn::M>>(
        &self,
        state: BdfState<Eqn::V>,
    ) -> Result<Bdf<'_, Eqn, NewtonNonlinearSolver<Eqn::M, LS>>, DiffsolError>
    where
        Eqn: OdeEquationsImplicit,
    {
        let newton_solver = NewtonNonlinearSolver::new(LS::default());
        Bdf::new(self, state, newton_solver)
    }

    pub fn bdf<LS: LinearSolver<Eqn::M>>(
        &self,
    ) -> Result<Bdf<'_, Eqn, NewtonNonlinearSolver<Eqn::M, LS>>, DiffsolError>
    where
        Eqn: OdeEquationsImplicit,
    {
        let state = self.bdf_state::<LS>()?;
        self.bdf_solver(state)
    }

    pub(crate) fn bdf_solver_aug<
        LS: LinearSolver<Eqn::M>,
        Aug: AugmentedOdeEquationsImplicit<Eqn>,
    >(
        &self,
        state: BdfState<Eqn::V>,
        aug_eqn: Aug,
    ) -> Result<
        Bdf<'_, Eqn, NewtonNonlinearSolver<Eqn::M, LS>, <Eqn::V as DefaultDenseMatrix>::M, Aug>,
        DiffsolError,
    >
    where
        Eqn: OdeEquationsImplicit,
    {
        let newton_solver = NewtonNonlinearSolver::new(LS::default());
        Bdf::new_augmented(state, self, aug_eqn, newton_solver)
    }

    pub fn bdf_solver_sens<LS: LinearSolver<Eqn::M>>(
        &self,
        state: BdfState<Eqn::V>,
    ) -> Result<
        Bdf<
            '_,
            Eqn,
            NewtonNonlinearSolver<Eqn::M, LS>,
            <Eqn::V as DefaultDenseMatrix>::M,
            SensEquations<Eqn>,
        >,
        DiffsolError,
    >
    where
        Eqn: OdeEquationsSens,
    {
        let sens_eqn = SensEquations::new(self);
        self.bdf_solver_aug(state, sens_eqn)
    }

    pub fn bdf_sens<LS: LinearSolver<Eqn::M>>(
        &self,
    ) -> Result<
        Bdf<
            '_,
            Eqn,
            NewtonNonlinearSolver<Eqn::M, LS>,
            <Eqn::V as DefaultDenseMatrix>::M,
            SensEquations<Eqn>,
        >,
        DiffsolError,
    >
    where
        Eqn: OdeEquationsSens,
    {
        let state = self.bdf_state_sens::<LS>()?;
        self.bdf_solver_sens(state)
    }

    pub fn sdirk_state<LS: LinearSolver<Eqn::M>, DM: DenseMatrix>(
        &self,
        tableau: Tableau<DM>,
    ) -> Result<SdirkState<Eqn::V>, DiffsolError>
    where
        Eqn: OdeEquationsImplicit,
    {
        SdirkState::new::<LS, _>(self, tableau.order())
    }

    pub fn sdirk_state_sens<LS: LinearSolver<Eqn::M>, DM: DenseMatrix>(
        &self,
        tableau: Tableau<DM>,
    ) -> Result<SdirkState<Eqn::V>, DiffsolError>
    where
        Eqn: OdeEquationsSens,
    {
        SdirkState::new_with_sensitivities::<LS, _>(self, tableau.order())
    }

    pub fn sdirk_solver<LS: LinearSolver<Eqn::M>, DM: DenseMatrix<V = Eqn::V, T = Eqn::T>>(
        &self,
        state: SdirkState<Eqn::V>,
        tableau: Tableau<DM>,
    ) -> Result<Sdirk<'_, Eqn, LS, DM>, DiffsolError>
    where
        Eqn: OdeEquationsImplicit,
    {
        let linear_solver = LS::default();
        Sdirk::new(self, state, tableau, linear_solver)
    }

    pub(crate) fn sdirk_solver_aug<
        LS: LinearSolver<Eqn::M>,
        DM: DenseMatrix<V = Eqn::V, T = Eqn::T>,
        Aug: AugmentedOdeEquationsImplicit<Eqn>,
    >(
        &self,
        state: SdirkState<Eqn::V>,
        tableau: Tableau<DM>,
        aug_eqn: Aug,
    ) -> Result<
        Sdirk<'_, Eqn, LS, DM, Aug>,
        DiffsolError,
    >
    where
        Eqn: OdeEquationsImplicit,
    {
        Sdirk::new_augmented(self, state, tableau, LS::default(), aug_eqn)
    }

    pub fn sdirk_solver_sens<LS: LinearSolver<Eqn::M>, DM: DenseMatrix<V = Eqn::V, T = Eqn::T>>(
        &self,
        state: SdirkState<Eqn::V>,
        tableau: Tableau<DM>,
    ) -> Result<Sdirk<'_, Eqn, LS, DM, SensEquations<Eqn>>, DiffsolError>
    where
        Eqn: OdeEquationsSens,
    {
        let sens_eqn = SensEquations::new(self);
        self.sdirk_solver_aug::<LS, DM, _>(state, tableau, sens_eqn)
    }

    sdirk_solver_from_tableau!(
        tr_bdf2_state,
        tr_bdf2_state_sens,
        tr_bdf2,
        tr_bdf2_sens,
        tr_bdf2_solver,
        tr_bdf2_solver_sens,
        tr_bdf2
    );
    sdirk_solver_from_tableau!(
        esdirk34_state,
        esdirk34_state_sens,
        esdirk34,
        esdirk34_sens,
        esdirk34_solver,
        esdirk34_solver_sens,
        esdirk34
    );
}

#[derive(Debug, Clone)]
pub struct OdeSolverSolutionPoint<V: Vector> {
    pub state: V,
    pub t: V::T,
}

pub struct OdeSolverSolution<V: Vector> {
    pub solution_points: Vec<OdeSolverSolutionPoint<V>>,
    pub sens_solution_points: Option<Vec<Vec<OdeSolverSolutionPoint<V>>>>,
    pub rtol: V::T,
    pub atol: V,
    pub negative_time: bool,
}

impl<V: Vector> OdeSolverSolution<V> {
    pub fn push(&mut self, state: V, t: V::T) {
        // find the index to insert the new point keeping the times sorted
        let index = self.get_index(t);
        // insert the new point at that index
        self.solution_points
            .insert(index, OdeSolverSolutionPoint { state, t });
    }
    fn get_index(&self, t: V::T) -> usize {
        if self.negative_time {
            self.solution_points
                .iter()
                .position(|x| x.t < t)
                .unwrap_or(self.solution_points.len())
        } else {
            self.solution_points
                .iter()
                .position(|x| x.t > t)
                .unwrap_or(self.solution_points.len())
        }
    }
    pub fn push_sens(&mut self, state: V, t: V::T, sens: &[V]) {
        // find the index to insert the new point keeping the times sorted
        let index = self.get_index(t);
        // insert the new point at that index
        self.solution_points
            .insert(index, OdeSolverSolutionPoint { state, t });
        // if the sensitivity solution is not initialized, initialize it
        if self.sens_solution_points.is_none() {
            self.sens_solution_points = Some(vec![vec![]; sens.len()]);
        }
        // insert the new sensitivity point at that index
        for (i, s) in sens.iter().enumerate() {
            self.sens_solution_points.as_mut().unwrap()[i].insert(
                index,
                OdeSolverSolutionPoint {
                    state: s.clone(),
                    t,
                },
            );
        }
    }
}

impl<V: Vector> Default for OdeSolverSolution<V> {
    fn default() -> Self {
        Self {
            solution_points: Vec::new(),
            sens_solution_points: None,
            rtol: V::T::from(1e-6),
            atol: V::from_element(1, V::T::from(1e-6)),
            negative_time: false,
        }
    }
}
