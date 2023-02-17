use crate::tensor::value::Value;
use log::debug;
use std::ops::{Neg, Sub, SubAssign};
use std::rc::Rc;
use std::sync::Arc;
fn sub_backward(out: &Value) {
    let x = out._prev.get(0).unwrap();
    let y = out._prev.get(1).unwrap();

    x.set_grad(x.get_grad() - out.get_grad());
    y.set_grad(y.get_grad() - out.get_grad());

    debug!(
        "sub_backwards({}) label {} grad {}",
        out.get_label(),
        x.get_label(),
        x.get_grad()
    );
    debug!(
        "sub_backwards({}) label {} grad {}",
        out.get_label(),
        y.get_label(),
        y.get_grad()
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
        self.set_data(self.get_data() - other.get_data());
        self.set_grad(self.get_grad() - other.get_grad());

        self._backward = Rc::new(sub_backward);
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
        assert_eq!(z.get_data(), -2.0);
        let z1 = z.clone() - Value::newd(8.0, "".to_string());
        z1.set_label("z1");
        assert_eq!(z1.get_data(), -10.0);
        println!(" z1:{:#?}", z1);
        let zz1 = z1 - 1.0;
        zz1.set_label("zz1");
        assert_eq!(zz1.get_data(), -11.0);
        let zz2 = 1.0 - zz1;
        zz2.set_label("zz2");
        assert_eq!(zz2.get_data(), 12.0);
        zz2.backward();
        println!(" z:{:#?}", zz2);
        assert_eq!(x.get_grad(), -1.0);
        assert_eq!(y.get_grad(), 1.0);
    }
}
