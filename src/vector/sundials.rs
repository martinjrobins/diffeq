use std::fmt;
use std::fmt::{Debug, Display, Formatter};
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Sub, SubAssign};

use crate::sundials_sys::{
    realtype, N_VAddConst, N_VClone, N_VConst, N_VDestroy, N_VDiv, N_VGetArrayPointer,
    N_VGetLength_Serial, N_VLinearSum, N_VNew_Serial, N_VProd, N_VScale, N_VWL2Norm_Serial,
    N_Vector,
};

#[cfg(not(sundials_version_major = "5"))]
use crate::sundials_sys::{SUNContext, SUNContext_Create};

use crate::{scale, IndexType, Scale};

use super::{Vector, VectorCommon, VectorIndex, VectorView, VectorViewMut};

#[cfg(not(sundials_version_major = "5"))]
static mut SUNCONTEXT: SUNContext = std::ptr::null_mut();

#[cfg(not(sundials_version_major = "5"))]
pub fn get_suncontext() -> &'static SUNContext {
    use std::ffi::c_void;
    use std::ptr::{addr_of, addr_of_mut};
    let sun_comm_null: *mut c_void = std::ptr::null_mut::<c_void>();
    unsafe {
        if SUNCONTEXT.is_null() {
            SUNContext_Create(sun_comm_null, addr_of_mut!(SUNCONTEXT) as *mut SUNContext);
        }
        &*addr_of!(SUNCONTEXT)
    }
}

#[derive(Debug)]
pub struct SundialsVector {
    nv: N_Vector,
    // hack to allow to take a N_Vector and pass to an Op call (which needs a &V).
    // TODO: find a better way to handle this, perhaps Op calls should take a View?
    owned: bool,
}

impl SundialsVector {
    pub fn new_serial(len: usize) -> Self {
        #[cfg(not(sundials_version_major = "5"))]
        let nv = {
            let ctx = get_suncontext();
            unsafe { N_VNew_Serial(len as i64, *ctx) }
        };

        #[cfg(sundials_version_major = "5")]
        let nv = unsafe { N_VNew_Serial(len as i64) };

        SundialsVector { nv, owned: true }
    }
    /// Create a new SundialsVector with the same length and type as the given vector.
    /// data is not copied.
    pub fn new_clone(v: &SundialsVector) -> Self {
        let nv = unsafe { N_VClone(v.nv) };
        SundialsVector { nv, owned: true }
    }
    pub fn new_not_owned(v: N_Vector) -> Self {
        if v.is_null() {
            panic!("N_Vector is null");
        }
        SundialsVector {
            nv: v,
            owned: false,
        }
    }
    pub fn sundials_vector(&self) -> N_Vector {
        self.nv
    }
}

impl Drop for SundialsVector {
    fn drop(&mut self) {
        if self.owned {
            unsafe { N_VDestroy(self.nv) };
        }
    }
}

#[derive(Debug)]
pub struct SundialsVectorViewMut<'a>(&'a mut SundialsVector);

impl SundialsVectorViewMut<'_> {
    fn sundials_vector(&self) -> N_Vector {
        self.0.sundials_vector()
    }
    fn len(&self) -> IndexType {
        unsafe { N_VGetLength_Serial(self.sundials_vector()) as IndexType }
    }
}

#[derive(Debug)]
pub struct SundialsVectorView<'a>(&'a SundialsVector);

impl SundialsVectorView<'_> {
    fn sundials_vector(&self) -> N_Vector {
        self.0.sundials_vector()
    }
    fn len(&self) -> IndexType {
        unsafe { N_VGetLength_Serial(self.sundials_vector()) as IndexType }
    }
}

#[derive(Debug, Clone)]
pub struct SundialsIndexVector(Vec<IndexType>);

impl Display for SundialsIndexVector {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        for i in 0..self.0.len() {
            write!(f, "{} ", self.0[i])?;
        }
        writeln!(f)?;
        Ok(())
    }
}

impl SundialsIndexVector {
    pub fn iter(&self) -> impl Iterator<Item = &IndexType> {
        self.0.iter()
    }
}

impl Display for SundialsVector {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        for i in 0..self.len() {
            write!(f, "{} ", self[i])?;
        }
        writeln!(f)?;
        Ok(())
    }
}

impl Display for SundialsVectorViewMut<'_> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        for i in 0..self.len() {
            write!(f, "{} ", self[i])?;
        }
        writeln!(f)?;
        Ok(())
    }
}

impl Display for SundialsVectorView<'_> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        for i in 0..self.len() {
            write!(f, "{} ", self[i])?;
        }
        writeln!(f)?;
        Ok(())
    }
}

impl VectorCommon for SundialsVector {
    type T = realtype;
}

impl VectorCommon for SundialsVectorView<'_> {
    type T = realtype;
}

impl VectorCommon for SundialsVectorViewMut<'_> {
    type T = realtype;
}

macro_rules! impl_helper {
    ($trait:path, SundialsVector, $inner:tt) => {
        impl $trait for SundialsVector $inner
    };
    ($trait:path, SundialsVectorView, $inner:tt) => {
        impl<'a> $trait for SundialsVectorView<'a> $inner
    };
    ($trait:path, SundialsVectorViewMut, $inner:tt) => {
        impl<'a> $trait for SundialsVectorViewMut<'a> $inner
    };
}

// Clone
impl Clone for SundialsVector {
    fn clone(&self) -> Self {
        let mut z = SundialsVector::new_serial(self.len());
        z.copy_from(self);
        z
    }
}

// Index
macro_rules! impl_index {
    ($type:tt) => {
        impl_helper!(Index<IndexType>, $type, {
            type Output = <Self as VectorCommon>::T;
            fn index(&self, index: usize) -> &Self::Output {
                unsafe { &*(N_VGetArrayPointer(self.sundials_vector()).add(index)) }
            }
        });
    };
}

impl_index!(SundialsVector);
impl_index!(SundialsVectorView);
impl_index!(SundialsVectorViewMut);

impl Index<IndexType> for SundialsIndexVector {
    type Output = IndexType;
    fn index(&self, index: IndexType) -> &Self::Output {
        &self.0[index]
    }
}

// IndexMut
macro_rules! impl_index_mut {
    ($type:tt) => {
        impl_helper!(IndexMut<IndexType>, $type, {
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                unsafe { &mut *(N_VGetArrayPointer(self.sundials_vector()).add(index)) }
            }
        });
    };
}

impl_index_mut!(SundialsVector);
impl_index_mut!(SundialsVectorViewMut);

// div by realtype -> owned
macro_rules! impl_div {
    ($type:tt) => {
        impl_helper!(Div<Scale<realtype>>, $type, {
            type Output = SundialsVector;
            fn div(self, rhs: Scale<realtype>) -> Self::Output {
                let z = SundialsVector::new_serial(self.len());
                unsafe {
                    N_VScale(
                        1. / rhs.value(),
                        self.sundials_vector(),
                        z.sundials_vector(),
                    )
                }
                z
            }
        });
    };
}

impl_div!(SundialsVector);
impl_div!(SundialsVectorView);
impl_div!(SundialsVectorViewMut);

// mul by realtype -> owned
macro_rules! impl_mul {
    ($type:tt) => {
        impl_helper!(Mul<Scale<realtype>>, $type, {
            type Output = SundialsVector;
            fn mul(self, rhs: Scale<realtype>) -> Self::Output {
                let z = SundialsVector::new_serial(self.len());
                unsafe { N_VScale(rhs.value(), self.sundials_vector(), z.sundials_vector()) }
                z
            }
        });
    };
}

impl_mul!(SundialsVector);
impl_mul!(SundialsVectorView);
impl_mul!(SundialsVectorViewMut);

// div assign with realtype
macro_rules! impl_div_assign {
    ($type:tt) => {
        impl_helper!(DivAssign<Scale<realtype>>, $type, {
            fn div_assign(&mut self, rhs: Scale<realtype>) {
                unsafe {
                    N_VScale(
                        1. / rhs.value(),
                        self.sundials_vector(),
                        self.sundials_vector(),
                    )
                }
            }
        });
    };
}

impl_div_assign!(SundialsVectorViewMut);
impl_div_assign!(SundialsVector);
impl_div_assign!(SundialsVectorView);

// mul assign with realtype
macro_rules! impl_mul_assign {
    ($type:tt) => {
        impl_helper!(MulAssign<Scale<realtype>>, $type, {
            fn mul_assign(&mut self, rhs: Scale<realtype>) {
                unsafe { N_VScale(rhs.value(), self.sundials_vector(), self.sundials_vector()) }
            }
        });
    };
}

impl_mul_assign!(SundialsVectorViewMut);
impl_mul_assign!(SundialsVector);
impl_mul_assign!(SundialsVectorView);

// op assign with owned and view
macro_rules! impl_op_assign {
    ($type:tt, $rhs:ty, $trait:ident, $fn:ident, $op:tt) => {
        impl_helper!($trait<$rhs>, $type, {
            fn $fn(&mut self, rhs: $rhs) {
                for i in 0..self.len() {
                    self[i] $op rhs[i];
                }
            }
        });
    };
}

impl_op_assign!(SundialsVector, SundialsVector, SubAssign, sub_assign, -=);
impl_op_assign!(SundialsVector, &SundialsVector, SubAssign, sub_assign, -=);
impl_op_assign!(SundialsVector, SundialsVectorView<'_>, SubAssign, sub_assign, -=);
impl_op_assign!(SundialsVector, &SundialsVectorView<'_>, SubAssign, sub_assign, -=);
impl_op_assign!(SundialsVectorViewMut, SundialsVector, SubAssign, sub_assign, -=);
impl_op_assign!(SundialsVectorViewMut, &SundialsVector, SubAssign, sub_assign, -=);
impl_op_assign!(SundialsVectorViewMut, SundialsVectorView<'_>, SubAssign, sub_assign, -=);
impl_op_assign!(SundialsVectorViewMut, &SundialsVectorView<'_>, SubAssign, sub_assign, -=);

impl_op_assign!(SundialsVector, SundialsVector, AddAssign, add_assign, +=);
impl_op_assign!(SundialsVector, &SundialsVector, AddAssign, add_assign, +=);
impl_op_assign!(SundialsVector, SundialsVectorView<'_>, AddAssign, add_assign, +=);
impl_op_assign!(SundialsVector, &SundialsVectorView<'_>, AddAssign, add_assign, +=);
impl_op_assign!(SundialsVectorViewMut, SundialsVector, AddAssign, add_assign, +=);
impl_op_assign!(SundialsVectorViewMut, &SundialsVector, AddAssign, add_assign, +=);
impl_op_assign!(SundialsVectorViewMut, SundialsVectorView<'_>, AddAssign, add_assign, +=);
impl_op_assign!(SundialsVectorViewMut, &SundialsVectorView<'_>, AddAssign, add_assign, +=);

// owned binop by vector -> owned
macro_rules! impl_binop_owned {
    ($type:tt, $rhs:ty, $trait:ident, $fn:ident, $op:tt) => {
        impl_helper!($trait<$rhs>, $type, {
            type Output = SundialsVector;
            fn $fn(mut self, rhs: $rhs) -> Self::Output {
                self $op rhs;
                self
            }
        });
    };
}

impl_binop_owned!(SundialsVector, SundialsVector, Sub, sub, -=);
impl_binop_owned!(SundialsVector, &SundialsVector, Sub, sub, -=);
impl_binop_owned!(SundialsVector, &SundialsVectorView<'_>, Sub, sub, -=);
impl_binop_owned!(SundialsVector, SundialsVectorView<'_>, Sub, sub, -=);
impl_binop_owned!(SundialsVector, SundialsVector, Add, add, +=);
impl_binop_owned!(SundialsVector, &SundialsVector, Add, add, +=);
impl_binop_owned!(SundialsVector, &SundialsVectorView<'_>, Add, add, +=);
impl_binop_owned!(SundialsVector, SundialsVectorView<'_>, Add, add, +=);

// binop alloc owned
macro_rules! impl_binop_alloc_owned {
    ($type:tt, $rhs:ty, $trait:ident, $fn:ident, $op:tt) => {
        impl_helper!($trait<$rhs>, $type, {
            type Output = SundialsVector;
            fn $fn(self, rhs: $rhs) -> Self::Output {
                let mut z = SundialsVector::new_serial(self.len());
                z.copy_from_view(&self);
                z $op rhs;
                z
            }
        });
    };
}

impl_binop_alloc_owned!(SundialsVectorView, &SundialsVector, Sub, sub, -=);
impl_binop_alloc_owned!(SundialsVectorView, SundialsVectorView<'_>, Sub, sub, -=);
impl_binop_alloc_owned!(SundialsVectorView, &SundialsVectorView<'_>, Sub, sub, -=);
impl_binop_alloc_owned!(SundialsVectorView, &SundialsVector, Add, add, +=);
impl_binop_alloc_owned!(SundialsVectorView, SundialsVectorView<'_>, Add, add, +=);
impl_binop_alloc_owned!(SundialsVectorView, &SundialsVectorView<'_>, Add, add, +=);

// view binop by owned vector
macro_rules! impl_add_view_owned {
    ($type:tt, $rhs:ty) => {
        impl_helper!(Add<$rhs>, $type, {
            type Output = SundialsVector;
            fn add(self, mut rhs: $rhs) -> Self::Output {
                rhs += self;
                rhs
            }
        });
    };
}

impl_add_view_owned!(SundialsVectorView, SundialsVector);

macro_rules! impl_sub_view_owned {
    ($type:tt, $rhs:ty) => {
        impl_helper!(Sub<$rhs>, $type, {
            type Output = SundialsVector;
            fn sub(self, mut rhs: $rhs) -> Self::Output {
                rhs -= self;
                rhs *= scale(-1.0);
                rhs
            }
        });
    };
}

impl_sub_view_owned!(SundialsVectorView, SundialsVector);

impl<'a> VectorViewMut<'a> for SundialsVectorViewMut<'a> {
    type Owned = SundialsVector;
    type View = SundialsVectorView<'a>;
    fn copy_from(&mut self, other: &Self::Owned) {
        unsafe { N_VScale(1.0, other.sundials_vector(), self.sundials_vector()) }
    }
    fn copy_from_view(&mut self, other: &Self::View) {
        unsafe { N_VScale(1.0, other.sundials_vector(), self.sundials_vector()) }
    }
    fn axpy(&mut self, alpha: Self::T, x: &Self::Owned, beta: Self::T) {
        unsafe {
            N_VLinearSum(
                alpha,
                x.sundials_vector(),
                beta,
                self.sundials_vector(),
                self.sundials_vector(),
            )
        };
    }
}

impl<'a> VectorView<'a> for SundialsVectorView<'a> {
    type Owned = SundialsVector;
    fn into_owned(self) -> Self::Owned {
        let mut z = SundialsVector::new_serial(self.len());
        z.copy_from_view(&self);
        z
    }
    fn norm(&self) -> Self::T {
        let ones = SundialsVector::from_element(self.len(), 1.0);
        unsafe { N_VWL2Norm_Serial(self.sundials_vector(), ones.sundials_vector()) }
    }
    fn squared_norm(&self, y: &Self::Owned, atol: &Self::Owned, rtol: Self::T) -> Self::T {
        let mut acc = 0.0;
        if y.len() != self.len() || y.len() != atol.len() {
            panic!("Vector lengths do not match");
        }
        for i in 0..self.len() {
            let yi = y[i];
            let ai = atol[i];
            let xi = self[i];
            acc += (xi / (yi.abs() * rtol + ai)).powi(2);
        }
        acc / self.len() as f64
    }
}

impl VectorIndex for SundialsIndexVector {
    fn zeros(len: IndexType) -> Self {
        Self(vec![0; len])
    }
    fn len(&self) -> IndexType {
        self.0.len() as IndexType
    }
    fn from_slice(slice: &[IndexType]) -> Self {
        Self(slice.to_vec())
    }
    fn clone_as_vec(&self) -> Vec<IndexType> {
        self.0.clone()
    }
}

impl Vector for SundialsVector {
    type View<'a>
        = SundialsVectorView<'a>
    where
        Self: 'a;
    type ViewMut<'a>
        = SundialsVectorViewMut<'a>
    where
        Self: 'a;
    type Index = SundialsIndexVector;
    fn len(&self) -> IndexType {
        unsafe { N_VGetLength_Serial(self.sundials_vector()) as IndexType }
    }
    fn as_mut_slice(&mut self) -> &mut [Self::T] {
        unsafe {
            let ptr = N_VGetArrayPointer(self.sundials_vector());
            std::slice::from_raw_parts_mut(ptr, self.len())
        }
    }
    fn as_slice(&self) -> &[Self::T] {
        unsafe {
            let ptr = N_VGetArrayPointer(self.sundials_vector());
            std::slice::from_raw_parts(ptr, self.len())
        }
    }
    fn copy_from_slice(&mut self, slice: &[Self::T]) {
        if slice.len() != self.len() {
            panic!("Vector lengths do not match");
        }
        for i in 0..self.len() {
            self[i] = slice[i];
        }
    }
    fn squared_norm(&self, y: &Self, atol: &Self, rtol: Self::T) -> Self::T {
        let mut acc = 0.0;
        if y.len() != self.len() || y.len() != atol.len() {
            panic!("Vector lengths do not match");
        }
        for i in 0..self.len() {
            let yi = y[i];
            let ai = atol[i];
            let xi = self[i];
            acc += (xi / (yi.abs() * rtol + ai)).powi(2);
        }
        acc / self.len() as f64
    }

    fn norm(&self) -> Self::T {
        let ones = SundialsVector::from_element(self.len(), 1.0);
        unsafe { N_VWL2Norm_Serial(self.sundials_vector(), ones.sundials_vector()) }
    }
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn fill(&mut self, value: Self::T) {
        unsafe { N_VConst(value, self.sundials_vector()) }
    }
    fn add_scalar_mut(&mut self, scalar: Self::T) {
        unsafe { N_VAddConst(self.sundials_vector(), scalar, self.sundials_vector()) }
    }
    fn as_view(&self) -> Self::View<'_> {
        SundialsVectorView(self)
    }
    fn as_view_mut(&mut self) -> Self::ViewMut<'_> {
        SundialsVectorViewMut(self)
    }
    fn axpy(&mut self, alpha: Self::T, x: &Self, beta: Self::T) {
        unsafe {
            N_VLinearSum(
                alpha,
                x.sundials_vector(),
                beta,
                self.sundials_vector(),
                self.sundials_vector(),
            )
        };
    }
    fn axpy_v(&mut self, alpha: Self::T, x: &Self::View<'_>, beta: Self::T) {
        unsafe {
            N_VLinearSum(
                alpha,
                x.sundials_vector(),
                beta,
                self.sundials_vector(),
                self.sundials_vector(),
            )
        };
    }
    fn component_div_assign(&mut self, other: &Self) {
        unsafe {
            N_VDiv(
                self.sundials_vector(),
                other.sundials_vector(),
                self.sundials_vector(),
            )
        };
    }
    fn component_mul_assign(&mut self, other: &Self) {
        unsafe {
            N_VProd(
                self.sundials_vector(),
                other.sundials_vector(),
                self.sundials_vector(),
            )
        };
    }
    fn copy_from(&mut self, other: &Self) {
        unsafe { N_VScale(1.0, other.sundials_vector(), self.sundials_vector()) }
    }
    fn copy_from_view(&mut self, other: &Self::View<'_>) {
        unsafe { N_VScale(1.0, other.sundials_vector(), self.sundials_vector()) }
    }
    fn map_inplace(&mut self, f: impl Fn(Self::T) -> Self::T) {
        for i in 0..self.len() {
            self[i] = f(self[i]);
        }
    }
    fn partition_indices<F: Fn(Self::T) -> bool>(&self, f: F) -> (Self::Index, Self::Index) {
        let mut indices_true = vec![];
        let mut indices_false = vec![];
        for i in 0..self.len() {
            if f(self[i]) {
                indices_true.push(i);
            } else {
                indices_false.push(i);
            }
        }
        (
            SundialsIndexVector(indices_true),
            SundialsIndexVector(indices_false),
        )
    }
    fn from_element(nstates: usize, value: Self::T) -> Self {
        let v = SundialsVector::new_serial(nstates);
        unsafe { N_VConst(value, v.sundials_vector()) };
        v
    }
    fn from_vec(vec: Vec<Self::T>) -> Self {
        let mut v = SundialsVector::new_serial(vec.len());
        for (i, &x) in vec.iter().enumerate() {
            v[i] = x;
        }
        v
    }
    fn binary_fold<B, F>(&self, other: &Self, init: B, f: F) -> B
    where
        F: Fn(B, Self::T, Self::T, IndexType) -> B,
    {
        let mut acc = init;
        for i in 0..self.len() {
            acc = f(acc, self[i], other[i], i);
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_indexing() {
        let mut v = SundialsVector::new_serial(2);
        v[0] = 1.0;
        v[1] = 2.0;
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 2.0);
    }

    #[test]
    fn test_add_sub_ops() {
        let mut v = SundialsVector::new_serial(2);
        v[0] = 1.0;
        v[1] = 2.0;
        let v2 = v.clone();
        let v3 = v + &v2;
        assert_eq!(v3[0], 2.0);
        assert_eq!(v3[1], 4.0);
        let v4 = v3 - v2;
        assert_eq!(v4[0], 1.0);
        assert_eq!(v4[1], 2.0);
    }

    #[test]
    fn test_mul_div_ops() {
        let mut v = SundialsVector::new_serial(2);
        v[0] = 1.0;
        v[1] = 2.0;
        let v2 = v * scale(2.0);
        assert_eq!(v2[0], 2.0);
        assert_eq!(v2[1], 4.0);
        let v3 = v2 / scale(2.0);
        assert_eq!(v3[0], 1.0);
        assert_eq!(v3[1], 2.0);
    }

    #[test]
    fn test_axpy() {
        let mut v = SundialsVector::new_serial(2);
        v[0] = 1.0;
        v[1] = 2.0;
        let mut v2 = SundialsVector::new_serial(2);
        v2[0] = 2.0;
        v2[1] = 3.0;
        v.axpy(2.0, &v2, 1.0);
        assert_eq!(v[0], 5.0);
        assert_eq!(v[1], 8.0);
    }

    #[test]
    fn test_component_mul_div() {
        let mut v = SundialsVector::new_serial(2);
        v[0] = 1.0;
        v[1] = 2.0;
        let mut v2 = SundialsVector::new_serial(2);
        v2[0] = 2.0;
        v2[1] = 3.0;
        v.component_mul_assign(&v2);
        assert_eq!(v[0], 2.0);
        assert_eq!(v[1], 6.0);
        v.component_div_assign(&v2);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 2.0);
    }

    #[test]
    fn test_copy_from() {
        let mut v = SundialsVector::new_serial(2);
        v[0] = 1.0;
        v[1] = 2.0;
        let mut v2 = SundialsVector::new_serial(2);
        v2.copy_from(&v);
        assert_eq!(v2[0], 1.0);
        assert_eq!(v2[1], 2.0);
    }

    #[test]
    fn test_map() {
        let mut v = SundialsVector::new_serial(2);
        v[0] = 1.0;
        v[1] = 2.0;
        v.map_inplace(f64::exp);
        assert_eq!(v[0], 1.0_f64.exp());
        assert_eq!(v[1], 2.0_f64.exp());
    }

    #[test]
    fn test_filter_indices() {
        let mut v = SundialsVector::new_serial(2);
        v[0] = 1.0;
        v[1] = 2.0;
        let (indices_true, indicies_false) = v.partition_indices(|x| x > 1.0);
        assert_eq!(indices_true.len(), 1);
        assert_eq!(indices_true[0], 1);
        assert_eq!(indicies_false.len(), 1);
        assert_eq!(indicies_false[0], 0);
    }

    #[test]
    fn test_zeros() {
        let v = SundialsIndexVector::zeros(1);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0], 0);
        let v = SundialsVector::zeros(1);
        assert_eq!(v.len(), 1);
        assert_eq!(v[0], 0.);
    }

    #[test]
    fn test_from_element() {
        let v = SundialsVector::from_element(2, 1.0);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 1.0);
    }

    #[test]
    fn test_from_vec() {
        let v = SundialsVector::from_vec(vec![1.0, 2.0]);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 2.0);
    }

    #[test]
    fn test_norm() {
        let mut v = SundialsVector::new_serial(2);
        v[0] = 1.0;
        v[1] = 2.0;
        let norm = v.norm();
        assert_eq!(norm, (1.0_f64.powi(2) + 2.0_f64.powi(2)).sqrt());
    }

    #[test]
    fn test_error_norm() {
        let v = SundialsVector::from_vec(vec![1.0, -2.0, 3.0]);
        let y = SundialsVector::from_vec(vec![1.0, 2.0, 3.0]);
        let atol = SundialsVector::from_vec(vec![0.1, 0.2, 0.3]);
        let rtol = 0.1;
        let mut tmp = y.clone() * scale(rtol);
        tmp += &atol;
        let mut r = v.clone();
        r.component_div_assign(&tmp);
        let errorn_check = r.norm().powi(2) / 3.0;
        assert!((v.squared_norm(&y, &atol, rtol) - errorn_check).abs() < 1e-10);
        assert!((v.as_view().squared_norm(&y, &atol, rtol) - errorn_check).abs() < 1e-10);
    }
}
