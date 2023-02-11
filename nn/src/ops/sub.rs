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

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_sub_values() {
        let x = Value::newd(3.0, "x".to_string());
        let y = Value::newd(5.0, "y".to_string());
        // let z = x.add(&y);
        let z = x.clone() - y.clone();
        z.set_label("z");
        assert_eq!(*(*z.data).borrow(), -2.0);
        let z1 = z.clone() - Value::newd(8.0, "".to_string());
        z1.set_label("z1");
        assert_eq!(*(*z1.clone().data).borrow(), -10.0);
        println!(" z1:{:#?}", z1);
        let zz1 = z1 - 1.0;
        zz1.set_label("zz1");
        assert_eq!(*(*zz1.data).borrow(), -11.0);
        let zz2 = 1.0 - zz1;
        zz2.set_label("zz2");
        assert_eq!(*(*zz2.data).borrow(), 12.0);
        zz2.backward();
        println!(" z:{:#?}", zz2);
        assert_eq!(*(*x.clone().grad).borrow(), -1.0);
        assert_eq!(*(*y.clone().grad).borrow(), 1.0);
    }
}
