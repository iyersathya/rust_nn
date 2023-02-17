use crate::tensor::value::Value;
use log::debug;
use std::ops::{Mul, MulAssign};
use std::rc::Rc;
use std::sync::Arc;
fn mul_backward(out: &Value) {
    let x = out._prev.get(0).unwrap();
    let y = out._prev.get(1).unwrap();

    x.set_grad(x.get_grad() + y.get_data() * out.get_grad());
    y.set_grad(y.get_grad() + x.get_data() * out.get_grad());

    debug!(
        "mul_backwards({}) label {} grad {}",
        out.get_label(),
        x.get_label(),
        x.get_grad()
    );
    debug!(
        "mul_backwards({}) label {} grad {}",
        out.get_label(),
        y.get_label(),
        y.get_grad()
    );
}
impl Value {
    fn mul(self, other: Value) -> Value {
        let mut out = Value::new(
            self.get_data() * other.get_data(),
            vec![Arc::new(self.clone()), Arc::new(other.clone())],
            "*".to_string(),
            "".to_string(),
        );
        out._backward = Rc::new(mul_backward);
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
        self.set_data(self.get_data() * other.get_data());
        self.set_grad(self.get_grad() * other.get_grad());

        self._backward = Rc::new(mul_backward);
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
        assert_eq!(z.get_data(), 6.0);
        z = z.clone() * Value::newd(3.0, "".to_string());
        assert_eq!(z.get_data(), 18.0);
        println!(" z:{:#?}", z);
        let zz = z * 2.0;
        assert_eq!(zz.get_data(), 36.0);
        zz.backward();
        println!(" z:{:#?}", zz);
        assert_eq!(x.get_grad(), 12.0);
        assert_eq!(y.get_grad(), 18.0);
    }
}
