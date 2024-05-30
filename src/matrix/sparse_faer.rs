use std::fmt::Debug;
use std::ops::Mul;

use super::sparsity::MatrixSparsityRef;
use super::{Matrix, MatrixCommon, MatrixSparsity};
use crate::vector::Vector;
use crate::{DefaultSolver, FaerSparseLU, IndexType, NonLinearOp, Scalar, Scale};
use anyhow::Result;
use faer::sparse::ops::{ternary_op_assign_into, union_symbolic};
use faer::sparse::{SymbolicSparseColMat, SymbolicSparseColMatRef};
use faer::Col;

pub struct SparseColMat<T: Scalar>(faer::sparse::SparseColMat<IndexType, T>);

impl<T: Scalar> Debug for SparseColMat<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl<T: Scalar> Clone for SparseColMat<T> {
    fn clone(&self) -> Self {
        let sparsity = self.0.symbolic().to_owned().unwrap();
        let values = self.0.values().to_vec();
        Self(faer::sparse::SparseColMat::new(sparsity, values))
    }
}

impl<T: Scalar> SparseColMat<T> {
    pub fn faer(&self) -> &faer::sparse::SparseColMat<IndexType, T> {
        &self.0
    }
}

impl<T: Scalar> DefaultSolver for SparseColMat<T> {
    type LS<C: NonLinearOp<M = SparseColMat<T>, V = Col<T>, T = T>> = FaerSparseLU<T, C>;
}

impl<T: Scalar> MatrixCommon for SparseColMat<T> {
    type T = T;
    type V = Col<T>;

    fn nrows(&self) -> IndexType {
        self.0.nrows()
    }
    fn ncols(&self) -> IndexType {
        self.0.ncols()
    }
}

impl<T: Scalar> MatrixSparsity<SparseColMat<T>> for SymbolicSparseColMat<IndexType> {
    fn union(
        self,
        other: SymbolicSparseColMatRef<IndexType>,
    ) -> Result<SymbolicSparseColMat<IndexType>> {
        union_symbolic(self.as_ref(), other).map_err(anyhow::Error::new)
    }

    fn as_ref(&self) -> SymbolicSparseColMatRef<IndexType> {
        self.as_ref()
    }

    fn nrows(&self) -> IndexType {
        self.nrows()
    }

    fn ncols(&self) -> IndexType {
        self.ncols()
    }

    fn is_sparse() -> bool {
        true
    }

    fn indices(&self) -> Vec<(IndexType, IndexType)> {
        let mut indices = Vec::with_capacity(self.compute_nnz());
        for col_i in 0..self.ncols() {
            for row_j in self.col_range(col_i) {
                indices.push((row_j, col_i));
            }
        }
        indices
    }

    fn new_diagonal(n: IndexType) -> Self {
        let indices = (0..n).map(|i| (i, i)).collect::<Vec<_>>();
        SymbolicSparseColMat::try_new_from_indices(n, n, indices.as_slice())
            .unwrap()
            .0
    }

    fn try_from_indices(
        nrows: IndexType,
        ncols: IndexType,
        indices: Vec<(IndexType, IndexType)>,
    ) -> Result<Self> {
        match Self::try_new_from_indices(nrows, ncols, indices.as_slice()) {
            Ok((sparsity, _)) => Ok(sparsity),
            Err(e) => Err(anyhow::anyhow!("Failed to create sparsity pattern: {}", e)),
        }
    }
}

impl<'a, T: Scalar> MatrixSparsityRef<'a, SparseColMat<T>>
    for SymbolicSparseColMatRef<'a, IndexType>
{
    fn to_owned(&self) -> SymbolicSparseColMat<IndexType> {
        self.to_owned().unwrap()
    }
    fn nrows(&self) -> IndexType {
        self.nrows()
    }

    fn ncols(&self) -> IndexType {
        self.ncols()
    }

    fn is_sparse() -> bool {
        true
    }

    fn indices(&self) -> Vec<(IndexType, IndexType)> {
        let mut indices = Vec::with_capacity(self.compute_nnz());
        for col_i in 0..self.ncols() {
            for row_j in self.col_range(col_i) {
                indices.push((row_j, col_i));
            }
        }
        indices
    }

    fn get_index(
        &self,
        rows: &[IndexType],
        cols: &[IndexType],
    ) -> <<SparseColMat<T> as MatrixCommon>::V as Vector>::Index {
        let col_ptrs = self.col_ptrs();
        let row_indices = self.row_indices();
        let mut indices = Vec::with_capacity(rows.len());
        for (&i, &j) in rows.iter().zip(cols.iter()) {
            let col_ptr = col_ptrs[j];
            let next_col_ptr = col_ptrs[j + 1];
            for (ii, &ri) in row_indices
                .iter()
                .enumerate()
                .take(next_col_ptr)
                .skip(col_ptr)
            {
                if ri == i {
                    indices.push(ii);
                    break;
                }
            }
        }
        indices
    }
}

impl<T: Scalar> Mul<Scale<T>> for SparseColMat<T> {
    type Output = SparseColMat<T>;

    fn mul(mut self, rhs: Scale<T>) -> Self::Output {
        for v in self.0.values_mut() {
            v.mul_assign(rhs.value());
        }
        self
    }
}

impl<T: Scalar> Mul<Scale<T>> for &SparseColMat<T> {
    type Output = SparseColMat<T>;

    fn mul(self, rhs: Scale<T>) -> Self::Output {
        self.clone() * rhs
    }
}

impl<T: Scalar> Matrix for SparseColMat<T> {
    type Sparsity = SymbolicSparseColMat<IndexType>;
    type SparsityRef<'a> = SymbolicSparseColMatRef<'a, IndexType>;

    fn sparsity(&self) -> Option<Self::SparsityRef<'_>> {
        Some(self.0.symbolic())
    }

    fn set_data_with_indices(
        &mut self,
        dst_indices: &<Self::V as Vector>::Index,
        src_indices: &<Self::V as Vector>::Index,
        data: &Self::V,
    ) {
        let values = self.0.values_mut();
        for (dst_i, src_i) in dst_indices.iter().zip(src_indices.iter()) {
            values[*dst_i] = data[*src_i];
        }
    }

    fn add_column_to_vector(&self, j: IndexType, v: &mut Self::V) {
        for i in self.0.col_range(j) {
            let row = self.0.row_indices()[i];
            v[row] += self.0.values()[i];
        }
    }

    fn triplet_iter(&self) -> impl Iterator<Item = (IndexType, IndexType, &Self::T)> {
        (0..self.ncols()).flat_map(move |j| {
            self.0.col_range(j).map(move |i| {
                let row = self.0.row_indices()[i];
                (row, j, &self.0.values()[i])
            })
        })
    }

    fn try_from_triplets(
        nrows: IndexType,
        ncols: IndexType,
        triplets: Vec<(IndexType, IndexType, T)>,
    ) -> Result<Self> {
        match faer::sparse::SparseColMat::try_new_from_triplets(nrows, ncols, triplets.as_slice()) {
            Ok(mat) => Ok(Self(mat)),
            Err(e) => Err(anyhow::anyhow!(
                "Failed to create matrix from triplets: {}",
                e
            )),
        }
    }
    fn gemv(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V) {
        let tmp = self.0.as_ref() * x.as_ref();
        y.axpy(alpha, &tmp, beta);
    }
    fn zeros(nrows: IndexType, ncols: IndexType) -> Self {
        Self(faer::sparse::SparseColMat::try_new_from_triplets(nrows, ncols, &[]).unwrap())
    }
    fn copy_from(&mut self, other: &Self) {
        self.0 = faer::sparse::SparseColMat::new(
            other.0.symbolic().to_owned().unwrap(),
            other.0.values().to_vec(),
        )
    }
    fn from_diagonal(v: &Col<T>) -> Self {
        let dim = v.nrows();
        let triplets = (0..dim).map(|i| (i, i, v[i])).collect::<Vec<_>>();
        Self(faer::sparse::SparseColMat::try_new_from_triplets(dim, dim, &triplets).unwrap())
    }
    fn diagonal(&self) -> Self::V {
        let mut ret = Col::zeros(self.nrows());
        for j in 0..self.ncols() {
            for i in self.0.col_range(j) {
                if self.0.row_indices()[i] == j {
                    ret[j] = self.0.values()[i];
                    break;
                }
            }
        }
        ret
    }
    fn set_column(&mut self, j: IndexType, v: &Self::V) {
        assert_eq!(v.len(), self.nrows());
        for i in self.0.col_range(j) {
            let row_i = self.0.row_indices()[i];
            self.0.values_mut()[i] = v[row_i];
        }
    }

    fn scale_add_and_assign(&mut self, x: &Self, beta: Self::T, y: &Self) {
        ternary_op_assign_into(self.0.as_mut(), x.0.as_ref(), y.0.as_ref(), |_s, x, y| {
            x + beta * y
        });
    }

    fn new_from_sparsity(
        ncols: IndexType,
        nrows: IndexType,
        sparsity: Option<Self::Sparsity>,
    ) -> Self {
        let sparsity = sparsity.expect("Sparsity pattern required for sparse matrix");
        assert_eq!(sparsity.nrows(), nrows);
        assert_eq!(sparsity.ncols(), ncols);
        let nnz = sparsity.row_indices().len();
        Self(faer::sparse::SparseColMat::new(
            sparsity,
            vec![T::zero(); nnz],
        ))
    }
}
