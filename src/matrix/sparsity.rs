use crate::{
    error::{DiffsolError, MatrixError},
    matrix_error,
    scalar::IndexType,
    vector::{Vector, VectorIndex},
};

use super::Matrix;

pub trait MatrixSparsity<M: Matrix>: Sized + Clone {
    fn nrows(&self) -> IndexType;
    fn ncols(&self) -> IndexType;
    fn is_sparse() -> bool;
    fn try_from_indices(
        nrows: IndexType,
        ncols: IndexType,
        indices: Vec<(IndexType, IndexType)>,
    ) -> Result<Self, DiffsolError>;
    fn indices(&self) -> Vec<(IndexType, IndexType)>;
    fn union(self, other: M::SparsityRef<'_>) -> Result<M::Sparsity, DiffsolError>;
    fn new_diagonal(n: IndexType) -> Self;
    fn as_ref(&self) -> M::SparsityRef<'_>;
    fn get_index(&self, indices: Vec<(IndexType, IndexType)>) -> <M::V as Vector>::Index;
}

pub trait MatrixSparsityRef<'a, M: Matrix> {
    fn nrows(&self) -> IndexType;
    fn ncols(&self) -> IndexType;
    fn is_sparse() -> bool;
    fn indices(&self) -> Vec<(IndexType, IndexType)>;
    fn to_owned(&self) -> M::Sparsity;
    fn get_index(&self, indices: Vec<(IndexType, IndexType)>) -> <M::V as Vector>::Index;
}

#[derive(Clone)]
pub struct Dense<M: Matrix> {
    nrows: IndexType,
    ncols: IndexType,
    _phantom: std::marker::PhantomData<M>,
}

pub struct DenseRef<'a, M: Matrix> {
    dense: &'a Dense<M>,
}

impl<M: Matrix> Dense<M> {
    pub fn new(nrows: IndexType, ncols: IndexType) -> Self {
        Dense {
            nrows,
            ncols,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, M: Matrix> DenseRef<'a, M> {
    pub fn new(dense: &'a Dense<M>) -> Self {
        DenseRef { dense }
    }
}

impl<M> MatrixSparsity<M> for Dense<M>
where
    for<'a> M: Matrix<Sparsity = Dense<M>, SparsityRef<'a> = DenseRef<'a, M>> + 'a,
{
    fn union(self, other: M::SparsityRef<'_>) -> Result<M::Sparsity, DiffsolError> {
        if self.nrows() != other.nrows() || self.ncols() != other.ncols() {
            return Err(matrix_error!(UnionIncompatibleShapes));
        }
        Ok(Self::new(self.nrows(), self.ncols()))
    }

    fn as_ref(&self) -> M::SparsityRef<'_> {
        DenseRef::new(self)
    }

    fn nrows(&self) -> IndexType {
        self.nrows
    }

    fn ncols(&self) -> IndexType {
        self.ncols
    }

    fn is_sparse() -> bool {
        false
    }

    fn try_from_indices(
        nrows: IndexType,
        ncols: IndexType,
        _indices: Vec<(IndexType, IndexType)>,
    ) -> Result<Self, DiffsolError> {
        if nrows == 0 || ncols == 0 {
            return Err(matrix_error!(MatrixShapeError));
        }
        Ok(Dense::new(nrows, ncols))
    }

    fn indices(&self) -> Vec<(IndexType, IndexType)> {
        Vec::new()
    }

    fn new_diagonal(n: IndexType) -> Self {
        Dense::new(n, n)
    }
    fn get_index(&self, rows: &[IndexType], cols: &[IndexType]) -> <M::V as Vector>::Index {
        let indices: Vec<_> = rows
            .iter()
            .zip(cols.iter())
            .map(|(i, j)| {
                if i >= &self.nrows() || j >= &self.ncols() {
                    panic!("Index out of bounds")
                }
                j * self.nrows() + i
            })
            .collect();
        <M::V as Vector>::Index::from_slice(indices.as_slice())
    }
}

impl<'a, M> MatrixSparsityRef<'a, M> for DenseRef<'a, M>
where
    M: Matrix<Sparsity = Dense<M>, SparsityRef<'a> = Self> + 'a,
{
    fn to_owned(&self) -> M::Sparsity {
        Dense::new(self.nrows(), self.ncols())
    }

    fn nrows(&self) -> IndexType {
        self.dense.nrows
    }

    fn ncols(&self) -> IndexType {
        self.dense.ncols
    }

    fn is_sparse() -> bool {
        false
    }

    fn indices(&self) -> Vec<(IndexType, IndexType)> {
        Vec::new()
    }
}
