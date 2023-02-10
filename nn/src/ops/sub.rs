use crate::tensor::value::Value;
use log::debug;
use std::ops::{Neg, Sub, SubAssign};
use std::sync::Arc;

fn sub_backward(out: &Value) {
    let x = out._prev.get(0).unwrap();
    let y = out._prev.get(1).unwrap();
    let mut grad_x = (*x.grad).borrow_mut();
    let mut grad_y = (*y.grad).borrow_mut();
    *grad_x -= *(*out.grad).borrow();
    *grad_y -= *(*out.grad).borrow();

    debug!(
        "sub_backwards({}) label {} grad {}",
        out._label.borrow(),
        x._label.borrow(),
        grad_x
    );
    debug!(
        "sub_backwards({}) label {} grad {}",
        out._label.borrow(),
        y._label.borrow(),
        grad_y
    );
}

impl Value {
    fn sub(self, other: Value) -> Value {
        let o = other * -1.0;
        self.add(o)
    }
}

impl Sub<Value> for Value {
    type Output = Value;

    fn sub(self, other: Value) -> Value {
        self.sub(other)
    }
}
impl SubAssign<Value> for Value {
    fn sub_assign(&mut self, other: Self) {
        let mut grad = (*self.grad).borrow_mut();
        let mut data = (*self.data).borrow_mut();
        let o_grad = *(*other.grad).borrow();
        let o_data = *(*other.data).borrow();

        *data -= o_data;
        *grad -= o_grad;

        self._backward = Arc::new(Box::new(sub_backward));
    }
}
impl Sub<f64> for Value {
    type Output = Value;
    fn sub(self, rhs: f64) -> Self::Output {
        let other = Value::new(rhs, vec![], "".to_string(), "".to_string());
        self.sub(other)
    }
}
impl Sub<Value> for f64 {
    type Output = Value;
    fn sub(self, rhs: Value) -> Self::Output {
        let other = Value::new(self, vec![], "".to_string(), "".to_string());
        other.sub(rhs)
    }
}
impl Neg for Value {
    type Output = Value;
    fn neg(self) -> Value {
        self * -1.0
    }
}
