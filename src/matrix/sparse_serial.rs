use std::{collections::HashSet, ops::Mul};

use anyhow::Result;
use nalgebra::DVector;
use nalgebra_sparse::{pattern::SparsityPattern, CooMatrix, CscMatrix};

use crate::{scalar::Scale, IndexType, Scalar};

use super::{Matrix, MatrixCommon, MatrixSparsity};

impl<T: Scalar> MatrixCommon for CscMatrix<T> {
    type V = DVector<T>;
    type T = T;

    fn ncols(&self) -> IndexType {
        self.ncols()
    }
    fn nrows(&self) -> IndexType {
        self.nrows()
    }
}

impl<T: Scalar> Mul<Scale<T>> for CscMatrix<T> {
    type Output = CscMatrix<T>;
    fn mul(self, rhs: Scale<T>) -> Self::Output {
        self * rhs.value()
    }
}

impl MatrixSparsity for SparsityPattern {
    type Index = Vec<IndexType>;

    fn get_index(&self, rows: &[IndexType], cols: &[IndexType]) -> Self::Index {
        let mut index = Vec::with_capacity(rows.len());
        for (&i, &j) in rows.iter().zip(cols) {
            let offset = self.major_offsets()[j];
            let lane = self.lane(j);
            let lane_i = lane.iter().position(|&x| x == i).unwrap();
            index.push(offset + lane_i);
        }
        index
    }

    fn nrows(&self) -> IndexType {
        self.minor_dim()
    }

    fn ncols(&self) -> IndexType {
        self.major_dim()
    }

    fn is_sparse(&self) -> bool {
        true
    }

    fn indices(&self) -> Vec<(IndexType, IndexType)> {
        let mut indices = Vec::with_capacity(self.nnz());
        for (j, &offset) in self.major_offsets().iter().enumerate() {
            let next_offset = self
                .major_offsets()
                .get(j + 1)
                .copied()
                .unwrap_or(self.minor_indices().len());
            for i in offset..next_offset {
                indices.push((self.minor_indices()[i], j));
            }
        }
        indices
    }

    fn union(&self, other: &Self) -> Result<Self> {
        let max_nnz = self.nnz().max(other.nnz());
        let min_nnz = self.nnz().min(other.nnz());
        let mut minor_indices = Vec::with_capacity(self.nnz() + max_nnz - min_nnz);
        let mut major_offsets = Vec::with_capacity(self.major_dim());

        // loop through columns, calculate union of rows
        let mut offset = 0;
        for j in 0..self.major_dim() {
            let lane = self.lane(j);
            let other_lane = other.lane(j);
            let set: HashSet<usize> =
                HashSet::from_iter(lane.iter().chain(other_lane.iter()).cloned());
            let mut set = set.into_iter().collect::<Vec<_>>();

            major_offsets.push(offset);
            offset += set.len();

            minor_indices.append(&mut set);
        }
        SparsityPattern::try_from_offsets_and_indices(
            self.major_dim(),
            self.minor_dim(),
            major_offsets,
            minor_indices,
        )
        .map_err(anyhow::Error::new)
    }

    fn try_from_indices(
        nrows: IndexType,
        ncols: IndexType,
        indices: Vec<(IndexType, IndexType)>,
    ) -> Result<Self> {
        // use a CSC sparsity pattern (so cols are major, rows are minor)
        let major_dim = ncols;
        let minor_dim = nrows;

        // sort indices by major index
        let mut indices = indices;
        indices.sort_unstable_by_key(|&(_, j)| j);

        // split into major offsets and minor indices
        let mut curr_col = 0;
        let mut major_offsets = Vec::with_capacity(major_dim + 1);
        let mut minor_indices = Vec::with_capacity(indices.len());
        for (i, j) in indices {
            while curr_col < j {
                major_offsets.push(minor_indices.len());
                curr_col += 1;
            }
            minor_indices.push(i);
        }
        major_offsets.push(minor_indices.len());

        SparsityPattern::try_from_offsets_and_indices(
            major_dim,
            minor_dim,
            major_offsets,
            minor_indices,
        )
        .map_err(anyhow::Error::new)
    }

    fn new_diagonal(n: IndexType) -> Self {
        let mut major_offsets = Vec::with_capacity(n + 1);
        let mut minor_indices = Vec::with_capacity(n);
        for i in 0..n {
            major_offsets.push(i);
            minor_indices.push(i);
        }
        major_offsets.push(n);
        SparsityPattern::try_from_offsets_and_indices(n, n, major_offsets, minor_indices).unwrap()
    }
}

impl<T: Scalar> Matrix for CscMatrix<T> {
    type Sparsity = SparsityPattern;

    fn sparsity(&self) -> Option<&Self::Sparsity> {
        Some(self.pattern())
    }

    fn set_data_with_indices(
        &mut self,
        dst_indices: &<Self::Sparsity as MatrixSparsity>::Index,
        src_indices: &<Self::V as crate::vector::Vector>::Index,
        data: &Self::V,
    ) {
        let values = self.values_mut();
        for (&dst_i, &src_i) in dst_indices.iter().zip(src_indices.iter()) {
            values[dst_i] = data[src_i];
        }
    }

    fn try_from_triplets(
        nrows: IndexType,
        ncols: IndexType,
        triplets: Vec<(IndexType, IndexType, T)>,
    ) -> Result<Self> {
        let mut coo = CooMatrix::new(nrows, ncols);
        for (i, j, v) in triplets {
            coo.push(i, j, v);
        }
        Ok(CscMatrix::from(&coo))
    }
    fn zeros(nrows: IndexType, ncols: IndexType) -> Self {
        Self::zeros(nrows, ncols)
    }
    fn copy_from(&mut self, other: &Self) {
        self.clone_from(other);
    }
    fn gemv(&self, alpha: Self::T, x: &Self::V, beta: Self::T, y: &mut Self::V) {
        let mut tmp = self * x;
        tmp *= alpha;
        y.axpy(alpha, &tmp, beta);
    }

    fn from_diagonal(v: &DVector<T>) -> Self {
        let nrows = v.len();
        let ncols = v.len();
        let mut coo = CooMatrix::<T>::new(nrows, ncols);
        for (i, &v) in v.into_iter().enumerate() {
            coo.push(i, i, v);
        }
        CscMatrix::from(&coo)
    }
    fn diagonal(&self) -> Self::V {
        let mut ret = DVector::zeros(self.nrows());
        for (i, _j, &v) in self.diagonal_as_csc().triplet_iter() {
            ret[i] = v;
        }
        ret
    }
    fn set_column(&mut self, j: IndexType, v: &Self::V) {
        // check v is the same length as the column
        assert_eq!(v.len(), self.nrows());

        let mut col = self.col_mut(j);
        let (dst_row_indices, dst_values) = col.rows_and_values_mut();

        // copy across the non-zero values
        for (&dst_i, dst_v) in dst_row_indices.iter().zip(dst_values.iter_mut()) {
            *dst_v = v[dst_i];
        }
    }
    fn scale_add_and_assign(&mut self, x: &Self, beta: Self::T, y: &Self) {
        *self = x + y * beta;
    }
    fn new_from_sparsity(nrows: IndexType, ncols: IndexType, sparsity: Option<&Self::Sparsity>) -> Self {
        if let Some(sparsity) = sparsity {
            let values = vec![T::zero(); sparsity.nnz()];
            CscMatrix::try_from_pattern_and_values(sparsity.clone(), values).unwrap()
        } else {
            CscMatrix::zeros(nrows, ncols)
        }
    }
}
