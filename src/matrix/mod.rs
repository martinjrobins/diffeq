use std::fmt::Debug;
use std::ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

use crate::error::DiffsolError;
use crate::scalar::Scale;
use crate::{IndexType, Scalar, Vector, VectorIndex};

use num_traits::{One, Zero};
use sparsity::{MatrixSparsity, MatrixSparsityRef};

#[cfg(feature = "nalgebra")]
mod dense_nalgebra_serial;

#[cfg(feature = "faer")]
mod dense_faer_serial;

#[cfg(feature = "faer")]
pub mod sparse_faer;

pub mod default_solver;
pub mod extract_block;
mod sparse_serial;
pub mod sparsity;

#[cfg(feature = "sundials")]
pub mod sundials;

pub trait MatrixCommon: Sized + Debug {
    type V: Vector<T = Self::T>;
    type T: Scalar;

    fn nrows(&self) -> IndexType;
    fn ncols(&self) -> IndexType;
}

impl<M> MatrixCommon for &M
where
    M: MatrixCommon,
{
    type T = M::T;
    type V = M::V;

    fn nrows(&self) -> IndexType {
        M::nrows(*self)
    }
    fn ncols(&self) -> IndexType {
        M::ncols(*self)
    }
}

impl<M> MatrixCommon for &mut M
where
    M: MatrixCommon,
{
    type T = M::T;
    type V = M::V;

    fn ncols(&self) -> IndexType {
        M::ncols(*self)
    }
    fn nrows(&self) -> IndexType {
        M::nrows(*self)
    }
}

pub trait MatrixOpsByValue<Rhs = Self, Output = Self>:
    MatrixCommon + Add<Rhs, Output = Output> + Sub<Rhs, Output = Output>
{
}

impl<M, Rhs, Output> MatrixOpsByValue<Rhs, Output> for M where
    M: MatrixCommon + Add<Rhs, Output = Output> + Sub<Rhs, Output = Output>
{
}

pub trait MatrixMutOpsByValue<Rhs = Self>: MatrixCommon + AddAssign<Rhs> + SubAssign<Rhs> {}

impl<M, Rhs> MatrixMutOpsByValue<Rhs> for M where M: MatrixCommon + AddAssign<Rhs> + SubAssign<Rhs> {}

/// A trait allowing for references to implement matrix operations
pub trait MatrixRef<M: MatrixCommon>: Mul<Scale<M::T>, Output = M> {}
impl<RefT, M: MatrixCommon> MatrixRef<M> for RefT where RefT: Mul<Scale<M::T>, Output = M> {}

/// A mutable view of a dense matrix [Matrix]
pub trait MatrixViewMut<'a>:
    for<'b> MatrixMutOpsByValue<&'b Self>
    + for<'b> MatrixMutOpsByValue<&'b Self::View>
    + MulAssign<Scale<Self::T>>
{
    type Owned;
    type View;
    fn gemm_oo(&mut self, alpha: Self::T, a: &Self::Owned, b: &Self::Owned, beta: Self::T);
    fn gemm_vo(&mut self, alpha: Self::T, a: &Self::View, b: &Self::Owned, beta: Self::T);
}

/// A view of a dense matrix [Matrix]
pub trait MatrixView<'a>:
    for<'b> MatrixOpsByValue<&'b Self::Owned, Self::Owned>
    + Mul<Scale<Self::T>, Output = Self::Owned>
    + Clone
{
    type Owned;

    /// Perform a matrix-vector multiplication `y = self * x + beta * y`.
    fn gemv_v(
        &self,
        alpha: Self::T,
        x: &<Self::V as Vector>::View<'_>,
        beta: Self::T,
        y: &mut Self::V,
    );

    fn gemv_o(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V);
}

/// A base matrix trait (including sparse and dense matrices)
pub trait Matrix: MatrixCommon + Mul<Scale<Self::T>, Output = Self> + Clone + 'static {
    type Sparsity: MatrixSparsity<Self>;
    type SparsityRef<'a>: MatrixSparsityRef<'a, Self>
    where
        Self: 'a;

    /// Return sparsity information (None if the matrix is dense)
    fn sparsity(&self) -> Option<Self::SparsityRef<'_>>;

    fn is_sparse() -> bool {
        Self::zeros(1, 1).sparsity().is_some()
    }

    fn into_transpose(self) -> Self;

    /// Extract the diagonal of the matrix as an owned vector
    fn diagonal(&self) -> Self::V;

    /// Perform a matrix-vector multiplication `y = alpha * self * x + beta * y`.
    fn gemv(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V);

    /// Copy the contents of `other` into `self`
    fn copy_from(&mut self, other: &Self);

    /// Create a new matrix of shape `nrows` x `ncols` filled with zeros
    fn zeros(nrows: IndexType, ncols: IndexType) -> Self;

    /// Create a new matrix from a sparsity pattern, the non-zero elements are not initialized
    fn new_from_sparsity(
        nrows: IndexType,
        ncols: IndexType,
        sparsity: Option<Self::Sparsity>,
    ) -> Self;

    /// Create a new diagonal matrix from a [Vector] holding the diagonal elements
    fn from_diagonal(v: &Self::V) -> Self;

    fn set_column(&mut self, j: IndexType, v: &Self::V);

    fn add_column_to_vector(&self, j: IndexType, v: &mut Self::V);

    /// assign the values in `data` to the matrix at the indices in `dst_indices` from the indices in `src_indices`
    /// dst_index can be obtained using the `get_index` method on the Sparsity type
    ///      - for dense matrices, the dst_index is the data index in column-major order
    ///      - for sparse matrices, the dst_index is the index into the data array
    fn set_data_with_indices(
        &mut self,
        dst_indices: &<Self::V as Vector>::Index,
        src_indices: &<Self::V as Vector>::Index,
        data: &Self::V,
    );

    /// assign the values in `data` to the matrix at the indices in `dst_indices` from the indices in `src_indices`
    /// dst_index and src_indices can be obtained using the `get_index` method on the Sparsity type
    ///      - for dense matrices, the dst_index is the data index in column-major order
    ///      - for sparse matrices, the dst_index is the index into the data array
    fn set_data_with_indices_mat(
        &mut self,
        dst_indices: &<Self::V as Vector>::Index,
        src_indices: &<Self::V as Vector>::Index,
        data: &Self,
    );

    /// Perform the assignment self = x + beta * y where x and y are matrices and beta is a scalar
    /// Panics if the sparsity of self, x, and y do not match (i.e. sparsity of self must be the union of the sparsity of x and y)
    fn scale_add_and_assign(&mut self, x: &Self, beta: Self::T, y: &Self);

    fn triplet_iter(&self) -> impl Iterator<Item = (IndexType, IndexType, &Self::T)>;

    /// Create a new matrix from a vector of triplets (i, j, value) where i and j are the row and column indices of the value
    fn try_from_triplets(
        nrows: IndexType,
        ncols: IndexType,
        triplets: Vec<(IndexType, IndexType, Self::T)>,
    ) -> Result<Self, DiffsolError>;
}

/// A dense column-major matrix. The assumption is that the underlying matrix is stored in column-major order, so functions for taking columns views are efficient
pub trait DenseMatrix:
    Matrix
    + for<'a, 'b> MatrixOpsByValue<&'b Self::View<'a>, Self>
    + for<'a, 'b> MatrixMutOpsByValue<&'b Self::View<'a>>
    + Index<(IndexType, IndexType), Output = Self::T>
    + IndexMut<(IndexType, IndexType), Output = Self::T>
{
    /// A view of the dense matrix type
    type View<'a>: MatrixView<'a, Owned = Self, T = Self::T, V = Self::V>
    where
        Self: 'a;

    /// A mutable view of the dense matrix type
    type ViewMut<'a>: MatrixViewMut<
        'a,
        Owned = Self,
        T = Self::T,
        V = Self::V,
        View = Self::View<'a>,
    >
    where
        Self: 'a;

    /// Perform a matrix-matrix multiplication `self = alpha * a * b + beta * self`, where `alpha` and `beta` are scalars, and `a` and `b` are matrices
    fn gemm(&mut self, alpha: Self::T, a: &Self, b: &Self, beta: Self::T);

    /// Performs an axpy operation on two columns of the matrix `M[:, i] = alpha * M[:, j] + M[:, i]`
    fn column_axpy(&mut self, alpha: Self::T, j: IndexType, beta: Self::T, i: IndexType);

    /// Get a matrix view of the columns starting at `start` and ending at `start + ncols`
    fn columns(&self, start: IndexType, ncols: IndexType) -> Self::View<'_>;

    /// Get a vector view of the column `i`
    fn column(&self, i: IndexType) -> <Self::V as Vector>::View<'_>;

    /// Get a mutable matrix view of the columns starting at `start` and ending at `start + ncols`
    fn columns_mut(&mut self, start: IndexType, ncols: IndexType) -> Self::ViewMut<'_>;

    /// Get a mutable vector view of the column `i`
    fn column_mut(&mut self, i: IndexType) -> <Self::V as Vector>::ViewMut<'_>;

    /// mat_mat_mul using gemm, allocating a new matrix
    fn mat_mul(&self, b: &Self) -> Self {
        let nrows = self.nrows();
        let ncols = b.ncols();
        let mut ret = Self::zeros(nrows, ncols);
        ret.gemm(Self::T::one(), self, b, Self::T::zero());
        ret
    }
}

