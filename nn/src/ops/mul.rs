use crate::tensor::value::Value;
use log::debug;
use std::ops::{Mul, MulAssign};
use std::sync::Arc;

fn mul_backward(out: &Value) {
    let x = out._prev.get(0).unwrap();
    let y = out._prev.get(1).unwrap();

    let mut grad_x = (*x.grad).borrow_mut();
    let mut grad_y = (*y.grad).borrow_mut();
    let data_x = (*x.data).borrow_mut();
    let data_y = (*y.data).borrow_mut();

    *grad_x += *data_y * *(*out.grad).borrow();
    *grad_y += *data_x * *(*out.grad).borrow();

    debug!(
        "mul_backwards({}) label {} grad {}",
        out._label.borrow(),
        x._label.borrow(),
        grad_x
    );
    debug!(
        "mul_backwards({}) label {} grad {}",
        out._label.borrow(),
        y._label.borrow(),
        grad_y
    );
}
impl Value {
    fn mul(self, other: Value) -> Value {
        let mut out = Value::new(
            *(*self.data).borrow() * *(*other.data).borrow(),
            vec![Arc::new(self.clone()), Arc::new(other.clone())],
            "*".to_string(),
            "".to_string(),
        );
        out._backward = Arc::new(Box::new(mul_backward));
        out
    }
}
impl Mul<f64> for Value {
    type Output = Value;
    fn mul(self, rhs: f64) -> Self::Output {
        let other = Value::new(rhs, vec![], "".to_string(), "".to_string());
        self.mul(other)
    }
}
impl Mul<Value> for f64 {
    type Output = Value;
    fn mul(self, rhs: Value) -> Self::Output {
        let other = Value::new(self, vec![], "".to_string(), "".to_string());
        other.mul(rhs)
    }
}

impl Mul<Value> for Value {
    type Output = Value;
    fn mul(self, other: Value) -> Value {
        self.mul(other)
    }
}
// Supporting *= not working correctly.
impl MulAssign<Value> for Value {
    fn mul_assign(&mut self, other: Self) {
        let mut grad = (*self.grad).borrow_mut();
        let mut data = (*self.data).borrow_mut();
        let o_grad = *(*other.grad).borrow();
        let o_data = *(*other.data).borrow();

        *data *= o_data;
        *grad *= o_grad;
        self._backward = Arc::new(Box::new(mul_backward));
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_mul_values() {
        let x = Value::newd(3.0, "x".to_string());
        let y = Value::newd(2.0, "y".to_string());
        // let z = x.add(&y);
        let mut z = x.clone() * y.clone();
        assert_eq!(*(*z.data).borrow(), 6.0);
        z = z.clone() * Value::newd(3.0, "".to_string());
        assert_eq!(*(*z.data).borrow(), 18.0);
        println!(" z:{:#?}", z);
        let zz = z * 2.0;
        assert_eq!(*(*zz.data).borrow(), 36.0);
        zz.backward();
        println!(" z:{:#?}", zz);
        assert_eq!(*(*x.clone().grad).borrow(), 12.0);
        assert_eq!(*(*y.clone().grad).borrow(), 18.0);
    }
}
