use crate::tensor::value::Value;
use log::debug;
use std::ops::{Add, AddAssign};
use std::rc::Rc;
use std::sync::Arc;

fn add_backward(out: &Value) {
    let x = out._prev.get(0).unwrap();
    let y = out._prev.get(1).unwrap();

    x.set_grad(x.get_grad() + out.get_grad());
    y.set_grad(y.get_grad() + out.get_grad());

    debug!(
        "add_backwards({}) label {} grad {}",
        out.get_label(),
        x.get_label(),
        x.get_grad()
    );
    debug!(
        "add_backwards({}) label {} grad {}",
        out.get_label(),
        x.get_label(),
        y.get_grad()
    );
}

impl Value {
    pub fn add(self, other: Value) -> Value {
        let mut out = Value::new(
            self.get_data() + other.get_data(),
            vec![Arc::new(self.clone()), Arc::new(other.clone())],
            "+".to_string(),
            "".to_string(),
        );
        out._backward = Rc::new(add_backward);
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
        self.set_data(self.get_data() + other.get_data());
        self.set_grad(self.get_grad() + other.get_grad());
        self._backward = Rc::new(add_backward); //Arc::new(Box::new(add_backward));
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

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_add_values() {
        let x = Value::newd(3.0, "x".to_string());
        let y = Value::newd(3.0, "y".to_string());

        let mut z = x.clone() + y.clone();
        z.set_label("z");
        assert_eq!(z.get_data(), 6.0);
        z = z.clone() + Value::newd(3.0, "".to_string());
        assert_eq!(z.get_data(), 9.0);
        println!(" z:{:#?}", z);
        let zz = z + 1.0;
        assert_eq!(zz.get_data(), 10.0);
        let zz = 1.0 + zz;
        assert_eq!(zz.get_data(), 11.0);
        zz.backward();
        println!(" z:{:#?}", zz);
        assert_eq!(x.get_grad(), 1.0);
        assert_eq!(y.get_grad(), 1.0);
    }
}
