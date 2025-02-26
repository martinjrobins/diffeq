use crate::matrix::DenseMatrix;
use crate::scalar::Scale;
use crate::{IndexType, Scalar};
use nalgebra::ComplexField;
use num_traits::Zero;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

#[cfg(feature = "faer")]
mod faer_serial;
#[cfg(feature = "nalgebra")]
mod nalgebra_serial;
#[cfg(feature = "sundials")]
pub mod sundials;

pub trait VectorIndex: Sized + Index<IndexType, Output = IndexType> + Debug + Clone {
    fn zeros(len: IndexType) -> Self;
    fn len(&self) -> IndexType;
    fn clone_as_vec(&self) -> Vec<IndexType>;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn from_slice(slice: &[IndexType]) -> Self;
}

pub trait VectorCommon: Sized + Debug {
    type T: Scalar;
}

impl<V> VectorCommon for &V
where
    V: VectorCommon,
{
    type T = V::T;
}

impl<V> VectorCommon for &mut V
where
    V: VectorCommon,
{
    type T = V::T;
}

pub trait VectorOpsByValue<Rhs = Self, Output = Self>:
    VectorCommon + Add<Rhs, Output = Output> + Sub<Rhs, Output = Output>
{
}

impl<V, Rhs, Output> VectorOpsByValue<Rhs, Output> for V where
    V: VectorCommon + Add<Rhs, Output = Output> + Sub<Rhs, Output = Output>
{
}

pub trait VectorMutOpsByValue<Rhs = Self>: VectorCommon + AddAssign<Rhs> + SubAssign<Rhs> {}

impl<V, Rhs> VectorMutOpsByValue<Rhs> for V where V: VectorCommon + AddAssign<Rhs> + SubAssign<Rhs> {}

pub trait VectorRef<V: Vector>:
    VectorOpsByValue<V, V>
    + for<'a> VectorOpsByValue<&'a V, V>
    + for<'a> VectorOpsByValue<V::View<'a>, V>
    + for<'a, 'b> VectorOpsByValue<&'a V::View<'b>, V>
    + Mul<Scale<V::T>, Output = V>
{
}

impl<RefT, V: Vector> VectorRef<V> for RefT where
    RefT: VectorOpsByValue<V, V>
        + for<'a> VectorOpsByValue<&'a V, V>
        + for<'a> VectorOpsByValue<V::View<'a>, V>
        + for<'a, 'b> VectorOpsByValue<&'a V::View<'b>, V>
        + Mul<Scale<V::T>, Output = V>
{
}

pub trait VectorViewMut<'a>:
    VectorMutOpsByValue<Self::View>
    + VectorMutOpsByValue<Self::Owned>
    + for<'b> VectorMutOpsByValue<&'b Self::View>
    + for<'b> VectorMutOpsByValue<&'b Self::Owned>
    + MulAssign<Scale<Self::T>>
    + Index<IndexType, Output = Self::T>
    + IndexMut<IndexType, Output = Self::T>
{
    type Owned;
    type View;
    fn copy_from(&mut self, other: &Self::Owned);
    fn copy_from_view(&mut self, other: &Self::View);
    fn axpy(&mut self, alpha: Self::T, x: &Self::Owned, beta: Self::T);
}

pub trait VectorView<'a>:
    VectorOpsByValue<Self, Self::Owned>
    + VectorOpsByValue<Self::Owned, Self::Owned>
    + for<'b> VectorOpsByValue<&'b Self::Owned, Self::Owned>
    + for<'b> VectorOpsByValue<&'b Self, Self::Owned>
    + Mul<Scale<Self::T>, Output = Self::Owned>
    + Index<IndexType, Output = Self::T>
{
    type Owned;
    fn squared_norm(&self, y: &Self::Owned, atol: &Self::Owned, rtol: Self::T) -> Self::T;
    fn norm(&self) -> Self::T;
    fn into_owned(self) -> Self::Owned;
}

pub trait Vector:
    VectorOpsByValue<Self>
    + for<'b> VectorOpsByValue<&'b Self>
    // TODO: commenting out these because they cause a compiler hang (https://github.com/martinjrobins/diffsol/issues/3)
    // TODO: replace when this is fixed
    //+ for<'a> VectorOpsByValue<Self::View<'a>>
    //+ for<'a, 'b> VectorOpsByValue<&'b Self::View<'a>>
    + Mul<Scale<Self::T>, Output = Self>
    + Div<Scale<Self::T>, Output = Self>
    + VectorMutOpsByValue<Self>
    + for<'a> VectorMutOpsByValue<Self::View<'a>>
    + for<'b> VectorMutOpsByValue<&'b Self>
    + for<'a, 'b> VectorMutOpsByValue<&'b Self::View<'a>>
    + MulAssign<Scale<Self::T>>
    + Index<IndexType, Output = Self::T>
    + IndexMut<IndexType, Output = Self::T>
    + Clone
{
    type View<'a>: VectorView<'a, T = Self::T, Owned = Self>
    where
        Self: 'a;
    type ViewMut<'a>: VectorViewMut<'a, T = Self::T, Owned = Self, View = Self::View<'a>>
    where
        Self: 'a;
    type Index: VectorIndex;
    fn norm(&self) -> Self::T;
    fn squared_norm(&self, y: &Self, atol: &Self, rtol: Self::T) -> Self::T;
    fn len(&self) -> IndexType;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn map_inplace(&mut self, f: impl Fn(Self::T) -> Self::T);
    fn from_element(nstates: usize, value: Self::T) -> Self;
    fn zeros(nstates: usize) -> Self {
        Self::from_element(nstates, Self::T::zero())
    }
    fn fill(&mut self, value: Self::T) {
        self.map_inplace(|_| value);
    }
    fn as_view(&self) -> Self::View<'_>;
    fn as_view_mut(&mut self) -> Self::ViewMut<'_>;
    fn as_slice(&self) -> &[Self::T];
    fn as_mut_slice(&mut self) -> &mut [Self::T];
    fn copy_from(&mut self, other: &Self);
    fn copy_from_view(&mut self, other: &Self::View<'_>);
    fn copy_from_slice(&mut self, slice: &[Self::T]);
    fn from_vec(vec: Vec<Self::T>) -> Self;
    fn axpy(&mut self, alpha: Self::T, x: &Self, beta: Self::T);
    fn axpy_v(&mut self, alpha: Self::T, x: &Self::View<'_>, beta: Self::T);
    fn add_scalar_mut(&mut self, scalar: Self::T);
    fn component_mul_assign(&mut self, other: &Self);
    fn component_div_assign(&mut self, other: &Self);
    fn partition_indices<F: Fn(Self::T) -> bool>(&self, f: F) -> (Self::Index, Self::Index);
    fn binary_fold<B, F>(&self, other: &Self, init: B, f: F) -> B
    where
        F: Fn(B, Self::T, Self::T, IndexType) -> B;
    fn assign_at_indices(&mut self, indices: &Self::Index, value: Self::T) {
        for i in 0..indices.len() {
            self[indices[i]] = value;
        }
    }
    /// copy from `other` at the indices specified by `indices`
    /// generaly `self` and `other` have the same length
    fn copy_from_indices(&mut self, other: &Self, indices: &Self::Index) {
        for i in 0..indices.len() {
            self[indices[i]] = other[indices[i]];
        }
    }
    fn gather(&mut self, other: &Self, indices: &Self::Index) {
        for i in 0..indices.len() {
            self[i] = other[indices[i]];
        }
    }
    fn scatter(&self, indices: &Self::Index, other: &mut Self) {
        for i in 0..indices.len() {
            other[indices[i]] = self[i];
        }
    }
    fn assert_eq_st(&self, other: &Self, tol: Self::T) {
        let tol = Self::from_element(self.len(), tol);
        self.assert_eq(other, &tol);
    }
    fn assert_eq_norm(&self, other: &Self, atol: &Self, rtol: Self::T, factor: Self::T) {
        let error = self.clone() - other.clone();
        let error_norm = error.squared_norm(other, atol, rtol).sqrt();
        assert!(
            error_norm < factor,
            "error_norm: {}. self: {:?}, other: {:?}",
            error_norm,
            self,
            other
        );
    }

    fn assert_eq(&self, other: &Self, tol: &Self) {
        assert_eq!(
            self.len(),
            other.len(),
            "Vector length mismatch: {} != {}",
            self.len(),
            other.len()
        );
        for i in 0..self.len() {
            if num_traits::abs(self[i] - other[i]) > tol[i] {
                eprintln!(
                    "Vector element mismatch at index {}: {} != {}",
                    i, self[i], other[i]
                );
                if self.len() <= 3 {
                    eprintln!("left: {:?}", self);
                    eprintln!("right: {:?}", other);
                } else if i == 0 {
                    eprintln!(
                        "left: [{}, {}, {}, ...] != [{}, {}, {}, ...]",
                        self[0], self[1], self[2], other[0], other[1], other[2]
                    );
                } else if i == self.len() - 1 {
                    eprintln!(
                        "left: [..., {}, {}, {}] != [..., {}, {}, {}]",
                        self[i - 2],
                        self[i - 1],
                        self[i],
                        other[i - 2],
                        other[i - 1],
                        other[i]
                    );
                } else {
                    eprintln!(
                        "left: [..., {}, {}, {}, ...] != [..., {}, {}, {}, ...]",
                        self[i - 1],
                        self[i],
                        self[i + 1],
                        other[i - 1],
                        other[i],
                        other[i + 1]
                    );
                }
                panic!(
                    "Vector element mismatch at index {}: {} != {}",
                    i, self[i], other[i]
                );
            }
        }
    }
}

pub trait DefaultDenseMatrix: Vector {
    type M: DenseMatrix<V = Self, T = Self::T>;
}
