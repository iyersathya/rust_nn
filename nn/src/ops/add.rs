use crate::tensor::value::Value;
use log::debug;
use std::ops::{Add, AddAssign};
use std::sync::Arc;

fn add_backward(out: &Value) {
    let x = out._prev.get(0).unwrap();
    let y = out._prev.get(1).unwrap();
    let mut grad_x = (*x.grad).borrow_mut();
    let mut grad_y = (*y.grad).borrow_mut();
    *grad_x += *(*out.grad).borrow();
    *grad_y += *(*out.grad).borrow();

    debug!(
        "add_backwards({}) label {} grad {}",
        out._label.borrow(),
        x._label.borrow(),
        grad_x
    );
    debug!(
        "add_backwards({}) label {} grad {}",
        out._label.borrow(),
        y._label.borrow(),
        grad_y
    );
}

impl Value {
    pub fn add(self, other: Value) -> Value {
        let mut out = Value::new(
            *(*self.data).borrow() + *(*other.data).borrow(),
            vec![Arc::new(self.clone()), Arc::new(other.clone())],
            "+".to_string(),
            "".to_string(),
        );
        out._backward = Arc::new(Box::new(add_backward));
        out
    }
}

impl Add<Value> for Value {
    type Output = Value;

    fn add(self, other: Value) -> Value {
        self.add(other)
    }
}
impl AddAssign<Value> for Value {
    fn add_assign(&mut self, other: Self) {
        let mut grad = (*self.grad).borrow_mut();
        let mut data = (*self.data).borrow_mut();
        let o_grad = *(*other.grad).borrow();
        let o_data = *(*other.data).borrow();

        *data += o_data;
        *grad += o_grad;
        self._backward = Arc::new(Box::new(add_backward));
    }
}
impl Add<f64> for Value {
    type Output = Value;
    fn add(self, rhs: f64) -> Self::Output {
        let other = Value::new(rhs, vec![], "".to_string(), "".to_string());
        self.add(other)
    }
}
impl Add<Value> for f64 {
    type Output = Value;
    fn add(self, rhs: Value) -> Self::Output {
        let other = Value::new(self, vec![], "".to_string(), "".to_string());
        other.add(rhs)
    }
}
