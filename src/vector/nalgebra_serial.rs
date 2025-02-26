use std::ops::{Div, Mul, MulAssign};

use nalgebra::{DMatrix, DVector, DVectorView, DVectorViewMut};

use crate::{IndexType, Scalar, Scale};

use super::{DefaultDenseMatrix, Vector, VectorCommon, VectorIndex, VectorView, VectorViewMut};

impl<T: Scalar> DefaultDenseMatrix for DVector<T> {
    type M = DMatrix<T>;
}

macro_rules! impl_op_for_dvector_struct {
    ($struct:ident, $trait_name:ident, $func_name:ident) => {
        impl<'a, T: Scalar> $trait_name<Scale<T>> for $struct<'a, T> {
            type Output = DVector<T>;
            fn $func_name(self, rhs: Scale<T>) -> Self::Output {
                self * rhs.value()
            }
        }
    };
}

impl_op_for_dvector_struct!(DVectorView, Mul, mul);
impl_op_for_dvector_struct!(DVectorViewMut, Mul, mul);

impl VectorIndex for DVector<IndexType> {
    fn zeros(len: IndexType) -> Self {
        DVector::from_element(len, 0)
    }
    fn len(&self) -> crate::IndexType {
        self.len()
    }
    fn from_slice(slice: &[IndexType]) -> Self {
        DVector::from_iterator(slice.len(), slice.iter().copied())
    }
    fn clone_as_vec(&self) -> Vec<IndexType> {
        self.iter().copied().collect()
    }
}

macro_rules! impl_vector_common {
    ($vector_type:ty) => {
        impl<'a, T: Scalar> VectorCommon for $vector_type {
            type T = T;
        }
    };
}

impl_vector_common!(DVector<T>);
impl_vector_common!(DVectorView<'a, T>);
impl_vector_common!(DVectorViewMut<'a, T>);

impl<'a, T: Scalar> VectorView<'a> for DVectorView<'a, T> {
    type Owned = DVector<T>;
    fn into_owned(self) -> Self::Owned {
        self.into_owned()
    }
    fn norm(&self) -> T {
        self.norm()
    }
    fn squared_norm(&self, y: &Self::Owned, atol: &Self::Owned, rtol: Self::T) -> Self::T {
        let mut acc = T::zero();
        if y.len() != self.len() || y.len() != atol.len() {
            panic!("Vector lengths do not match");
        }
        for i in 0..self.len() {
            let yi = unsafe { y.get_unchecked(i) };
            let ai = unsafe { atol.get_unchecked(i) };
            let xi = unsafe { self.get_unchecked(i) };
            acc += (*xi / (yi.abs() * rtol + *ai)).powi(2);
        }
        acc / Self::T::from(self.len() as f64)
    }
}

macro_rules! impl_mul_scale_vector {
    ($vector_type:ty) => {
        impl<T: Scalar> Mul<Scale<T>> for $vector_type {
            type Output = DVector<T>;
            fn mul(self, rhs: Scale<T>) -> Self::Output {
                self * rhs.value()
            }
        }

        impl<T: Scalar> Mul<Scale<T>> for &$vector_type {
            type Output = DVector<T>;
            fn mul(self, rhs: Scale<T>) -> Self::Output {
                self * rhs.value()
            }
        }
    };
}

impl_mul_scale_vector!(DVector<T>);
macro_rules! impl_mul_assign_scale_vector {
    ($vector_type:ty) => {
        impl<'a, T: Scalar> MulAssign<Scale<T>> for $vector_type {
            fn mul_assign(&mut self, rhs: Scale<T>) {
                *self *= rhs.value();
            }
        }
    };
}

impl_mul_assign_scale_vector!(DVector<T>);
impl_mul_assign_scale_vector!(DVectorViewMut<'a, T>);

impl<'a, T: Scalar> VectorViewMut<'a> for DVectorViewMut<'a, T> {
    type Owned = DVector<T>;
    type View = DVectorView<'a, T>;
    fn copy_from(&mut self, other: &Self::Owned) {
        self.copy_from(other);
    }
    fn copy_from_view(&mut self, other: &Self::View) {
        self.copy_from(other);
    }
    fn axpy(&mut self, alpha: Self::T, x: &Self::Owned, beta: Self::T) {
        self.axpy(alpha, x, beta);
    }
}

impl<T: Scalar> Div<Scale<T>> for DVector<T> {
    type Output = DVector<T>;
    fn div(self, rhs: Scale<T>) -> Self::Output {
        self / rhs.value()
    }
}

impl<T: Scalar> Vector for DVector<T> {
    type View<'a> = DVectorView<'a, T>;
    type ViewMut<'a> = DVectorViewMut<'a, T>;
    type Index = DVector<IndexType>;
    fn len(&self) -> IndexType {
        self.len()
    }
    fn norm(&self) -> Self::T {
        self.norm()
    }
    fn as_slice(&self) -> &[Self::T] {
        self.as_slice()
    }
    fn as_mut_slice(&mut self) -> &mut [Self::T] {
        self.as_mut_slice()
    }
    fn copy_from_slice(&mut self, slice: &[Self::T]) {
        self.copy_from_slice(slice);
    }
    fn squared_norm(&self, y: &Self, atol: &Self, rtol: Self::T) -> Self::T {
        let mut acc = T::zero();
        if y.len() != self.len() || y.len() != atol.len() {
            panic!("Vector lengths do not match");
        }
        for i in 0..self.len() {
            let yi = unsafe { y.get_unchecked(i) };
            let ai = unsafe { atol.get_unchecked(i) };
            let xi = unsafe { self.get_unchecked(i) };
            acc += (*xi / (yi.abs() * rtol + *ai)).powi(2);
        }
        acc / Self::T::from(self.len() as f64)
    }
    fn fill(&mut self, value: T) {
        self.fill(value);
    }
    fn as_view(&self) -> Self::View<'_> {
        self.as_view()
    }
    fn as_view_mut(&mut self) -> Self::ViewMut<'_> {
        self.as_view_mut()
    }
    fn copy_from(&mut self, other: &Self) {
        self.copy_from(other);
    }
    fn map_inplace(&mut self, f: impl Fn(Self::T) -> Self::T) {
        self.iter_mut().for_each(|x: &mut _| *x = f(*x));
    }
    fn copy_from_view(&mut self, other: &Self::View<'_>) {
        self.copy_from(other);
    }
    fn from_element(nstates: usize, value: T) -> Self {
        Self::from_element(nstates, value)
    }
    fn from_vec(vec: Vec<T>) -> Self {
        Self::from_vec(vec)
    }
    fn zeros(nstates: usize) -> Self {
        Self::zeros(nstates)
    }
    fn add_scalar_mut(&mut self, scalar: T) {
        self.add_scalar_mut(scalar);
    }
    fn axpy(&mut self, alpha: T, x: &Self, beta: T) {
        self.axpy(alpha, x, beta);
    }
    fn axpy_v(&mut self, alpha: Self::T, x: &Self::View<'_>, beta: Self::T) {
        self.axpy(alpha, x, beta);
    }
    fn component_div_assign(&mut self, other: &Self) {
        self.component_div_assign(other);
    }
    fn component_mul_assign(&mut self, other: &Self) {
        self.component_mul_assign(other);
    }
    fn partition_indices<F: Fn(T) -> bool>(&self, f: F) -> (Self::Index, Self::Index) {
        let mut indices_true = vec![];
        let mut indices_false = vec![];
        for (i, &x) in self.iter().enumerate() {
            if f(x) {
                indices_true.push(i as IndexType);
            } else {
                indices_false.push(i as IndexType);
            }
        }
        (
            Self::Index::from_vec(indices_true),
            Self::Index::from_vec(indices_false),
        )
    }
    fn binary_fold<B, F>(&self, other: &Self, init: B, f: F) -> B
    where
        F: Fn(B, Self::T, Self::T, IndexType) -> B,
    {
        let mut acc = init;
        for (i, (x, y)) in self.iter().zip(other.iter()).enumerate() {
            acc = f(acc, *x, *y, i);
        }
        acc
    }
}

// tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_abs() {
        let v = DVector::from_vec(vec![1.0, -2.0, 3.0]);
        let v_abs = v.abs();
        assert_eq!(v_abs, DVector::from_vec(vec![1.0, 2.0, 3.0]));
    }

    #[test]
    fn test_error_norm() {
        let v = DVector::from_vec(vec![1.0, -2.0, 3.0]);
        let y = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let atol = DVector::from_vec(vec![0.1, 0.2, 0.3]);
        let rtol = 0.1;
        let mut tmp = y.clone() * rtol;
        tmp += &atol;
        let mut r = v.clone();
        r.component_div_assign(&tmp);
        let errorn_check = r.norm_squared() / 3.0;
        assert_eq!(v.squared_norm(&y, &atol, rtol), errorn_check);
        let vview = v.as_view();
        assert_eq!(
            VectorView::squared_norm(&vview, &y, &atol, rtol),
            errorn_check
        );
    }
}
