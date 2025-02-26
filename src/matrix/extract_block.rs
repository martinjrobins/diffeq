use crate::{IndexType, Matrix, Vector, MatrixSparsity, MatrixSparsityRef, Dense};
struct ExtractBlock<M: Matrix> {
    dst_indices: <M::V as Vector>::Index,
    src_indices: <M::V as Vector>::Index,
    block: M,
}

impl<M: Matrix> ExtractBlock<M> {
    /// split a matrix into four blocks based on a predicate
    /// 
    /// M = [UL(false, false), UR(false, true)]
    ///     [LL(true, false), LR(true, true)]
    ///     
    /// if transpose is true, the blocks are each transposed but are still in the same order:
    /// return order is (UL, UR, LL, LR)
    ///
    pub fn split<F>(src: &M, f: F, transpose: bool) -> (Self, Self, Self, Self)
    where
        F: Fn(IndexType) -> bool,
    {
        let n = src.nrows();
        if n != src.ncols() {
            panic!("Matrix must be square");
        }
        let cat = (0..n).map(|i| f(i)).collect::<Vec<_>>();
        let ni = cat.iter().count();
        let nni = n - ni;

        let mut upper_indices = Vec::with_capacity(n);
        let mut lower_indices = Vec::with_capacity(n);
        let mut upper_acc = 0;
        let mut lower_acc = 0;
        for c in cat.iter() {
            lower_indices.push(lower_acc);
            upper_indices.push(upper_acc);
            if *c {
                lower_acc += 1;
            } else {
                upper_acc += 1;
            }
        }
        let mut ur_triplets = Vec::new();
        let mut ul_triplets = Vec::new();
        let mut lr_triplets = Vec::new();
        let mut ll_triplets = Vec::new();
        let mut dst_ur_indices = Vec::new();
        let mut dst_ul_indices = Vec::new();
        let mut dst_lr_indices = Vec::new();
        let mut dst_ll_indices = Vec::new();
        for (i, j, _v) in src.triplet_iter() {
            if !cat[i] && !cat[j] {
                let ii = upper_indices[i];
                let jj = upper_indices[j];
                ul_triplets.push((i, j));
                dst_ul_indices.push((ii, jj));
            } else if !cat[i] && cat[j] {
                let ii = upper_indices[i];
                let jj = lower_indices[j];
                ur_triplets.push((i, j));
                dst_ur_indices.push((ii, jj));
            } else if cat[i] && !cat[j] {
                let ii = lower_indices[i];
                let jj = upper_indices[j];
                ll_triplets.push((i, j));
                dst_ll_indices.push((ii, jj));
            } else {
                let ii = lower_indices[i];
                let jj = lower_indices[j];
                lr_triplets.push((i, j));
                dst_lr_indices.push((ii, jj));
            }
        }
        if transpose {
            dst_ul_indices.iter_mut().for_each(|(i, j)| std::mem::swap(i, j));
            dst_ur_indices.iter_mut().for_each(|(i, j)| std::mem::swap(i, j));
            dst_ll_indices.iter_mut().for_each(|(i, j)| std::mem::swap(i, j));
            dst_lr_indices.iter_mut().for_each(|(i, j)| std::mem::swap(i, j));
        }
        let ul_sparsity = M::Sparsity::try_from_indices(nni, nni, dst_ul_indices.clone()).unwrap();
        let ur_sparsity = M::Sparsity::try_from_indices(nni, ni, dst_ur_indices.clone()).unwrap();
        let ll_sparsity = M::Sparsity::try_from_indices(ni, nni, dst_ll_indices.clone()).unwrap();
        let lr_sparsity = M::Sparsity::try_from_indices(ni, ni, dst_lr_indices.clone()).unwrap();
        let dst_ul_indices = ul_sparsity.get_index(dst_ul_indices);
        let dst_ur_indices = ur_sparsity.get_index(dst_ur_indices);
        let dst_ll_indices = ll_sparsity.get_index(dst_ll_indices);
        let dst_lr_indices = lr_sparsity.get_index(dst_lr_indices);
        let (src_ul_indices, src_ur_indices, src_ll_indices, src_lr_indices) = if let Some(sparsity) = src.sparsity() {
            (sparsity.get_index(ul_triplets), sparsity.get_index(ur_triplets), sparsity.get_index(ll_triplets), sparsity.get_index(lr_triplets))
        } else {
            let sparsity = Dense::<M>::new(n, n);
            (
                sparsity.get_index(ul_triplets),
                sparsity.get_index(ur_triplets),
                sparsity.get_index(ll_triplets),
                sparsity.get_index(lr_triplets),
            )
        };

        let mut ul = M::new_from_sparsity(nni, nni, Some(ul_sparsity));
        let mut ur = M::new_from_sparsity(nni, ni, Some(ur_sparsity));
        let mut ll = M::new_from_sparsity(ni, nni, Some(ll_sparsity));
        let mut lr = M::new_from_sparsity(ni, ni, Some(lr_sparsity));
        
        ul.set_data_with_indices_mat(&dst_ul_indices, &src_ul_indices, src);
        ur.set_data_with_indices_mat(&dst_ur_indices, &src_ur_indices, src);
        ll.set_data_with_indices_mat(&dst_ll_indices, &src_ll_indices, src);
        lr.set_data_with_indices_mat(&dst_lr_indices, &src_lr_indices, src);

        (
            Self {
                dst_indices: dst_ul_indices,
                src_indices: src_ul_indices,
                block: ul,
            },
            Self {
                dst_indices: dst_ur_indices,
                src_indices: src_ur_indices,
                block: ur,
            },
            Self {
                dst_indices: dst_ll_indices,
                src_indices: src_ll_indices,
                block: ll,
            },
            Self {
                dst_indices: dst_lr_indices,
                src_indices: src_lr_indices,
                block: lr,
            },
        )
    }
    
    pub fn combine<F>(ul: &M, ur: &M, ll: &M, lr: &M, f: F) -> M
    where
        F: Fn(IndexType) -> bool,
    {
        let n = ul.nrows() + ll.nrows();
        let m = ul.ncols() + ur.ncols();
        if ul.ncols() != ll.ncols()
            || ur.ncols() != lr.ncols()
            || ul.nrows() != ur.nrows()
            || ll.nrows() != lr.nrows()
        {
            panic!("Matrices must have the same shape");
        }
        let mut triplets = Vec::new();
        let cat = (0..n).map(|i| f(i)).collect::<Vec<_>>();
        let ni = cat.iter().count();
        let mut upper_indices = Vec::with_capacity(n);
        let mut lower_indices = Vec::with_capacity(n);
        let mut upper_indices_short = Vec::with_capacity(n - ni);
        let mut lower_indices_short = Vec::with_capacity(ni);
        let mut upper_acc = 0;
        let mut lower_acc = 0;
        for (i, c) in cat.iter().enumerate() {
            lower_indices.push(lower_acc);
            upper_indices.push(upper_acc);
            if *c {
                lower_indices_short.push(i);
                lower_acc += 1;
            } else {
                upper_indices_short.push(i);
                upper_acc += 1;
            }
        }
        for (i, j, &v) in ul.triplet_iter() {
            let ii = upper_indices_short[i];
            let jj = upper_indices_short[j];
            triplets.push((ii, jj, v));
        }
        for (i, j, &v) in ur.triplet_iter() {
            let ii = upper_indices_short[i];
            let jj = lower_indices_short[j];
            triplets.push((ii, jj, v));
        }
        for (i, j, &v) in ll.triplet_iter() {
            let ii = lower_indices_short[i];
            let jj = upper_indices_short[j];
            triplets.push((ii, jj, v));
        }
        for (i, j, &v) in lr.triplet_iter() {
            let ii = lower_indices_short[i];
            let jj = lower_indices_short[j];
            triplets.push((ii, jj, v));
        }
        M::try_from_triplets(n, m, triplets).unwrap()

    } 
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SparseColMat;
    use faer::Mat;
    
    #[test]
    fn test_split_combine_faer_sparse() {
        test_split_combine::<SparseColMat<f64>>();
    }

    #[test]
    fn test_split_combine_dense() {
        test_split_combine::<Mat<f64>>();
    }
    
    fn test_split_combine<M: Matrix>() {
        let triplets = vec![
            (0, 0, 1.0),
            (1, 0, 5.0),
            (2, 0, 9.0),
            (3, 0, 13.0),
            (0, 1, 2.0),
            (1, 1, 6.0),
            (2, 1, 10.0),
            (3, 1, 14.0),
            (0, 2, 3.0),
            (1, 2, 7.0),
            (2, 2, 11.0),
            (3, 2, 15.0),
            (0, 3, 4.0),
            (1, 3, 8.0),
            (2, 3, 12.0),
            (3, 3, 16.0),
        ];
        let triplets = triplets.iter().map(|(i, j, v)| (*i, *j, M::T::from(*v))).collect::<Vec<_>>();
        let m = M::try_from_triplets(4, 4, triplets.clone()).unwrap();
        let indices = vec![0, 2];
        let (ul, ur, ll, lr) = ExtractBlock::split(&m, |&i| indices.contains(i), false);
        let ul_triplets = vec![(0, 0, 6.0), (1, 0, 14.0), (0, 1, 8.0), (1, 1, 16.0)];
        let ur_triplets = vec![(0, 0, 5.0), (1, 0, 13.0), (0, 1, 7.0), (1, 1, 15.0)];
        let ll_triplets = vec![(0, 0, 2.0), (1, 0, 10.0), (0, 1, 4.0), (1, 1, 12.0)];
        let lr_triplets = vec![(0, 0, 1.0), (1, 0, 9.0), (0, 1, 3.0), (1, 1, 11.0)];
        let ul_triplets = ul_triplets.iter().map(|(i, j, v)| (*i, *j, M::T::from(*v))).collect::<Vec<_>>();
        let ur_triplets = ur_triplets.iter().map(|(i, j, v)| (*i, *j, M::T::from(*v))).collect::<Vec<_>>();
        let ll_triplets = ll_triplets.iter().map(|(i, j, v)| (*i, *j, M::T::from(*v))).collect::<Vec<_>>();
        let lr_triplets = lr_triplets.iter().map(|(i, j, v)| (*i, *j, M::T::from(*v))).collect::<Vec<_>>();
        assert_eq!(
            ul_triplets,
            ul.block.triplet_iter()
                .map(|(i, j, v)| (i, j, *v))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            ur_triplets,
            ur.block.triplet_iter()
                .map(|(i, j, v)| (i, j, *v))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            ll_triplets,
            ll.block.triplet_iter()
                .map(|(i, j, v)| (i, j, *v))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            lr_triplets,
            lr.block.triplet_iter()
                .map(|(i, j, v)| (i, j, *v))
                .collect::<Vec<_>>()
        );

        let mat = SparseColMat::combine_at_indices(&ul, &ur, &ll, &lr, &indices);
        assert_eq!(
            triplets,
            mat.triplet_iter()
                .map(|(i, j, v)| (i, j, *v))
                .collect::<Vec<_>>()
        );
    }
}

