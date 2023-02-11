use crate::tensor::value::Value;
use std::ops::Div;

impl Div<Value> for Value {
    type Output = Value;

    fn div(self, other: Value) -> Value {
        self * other.powf(-1.0)
    }
}
// impl DivAssign<Value> for Value {
//     fn div_assign(&mut self, other: Self) {
//         let mut grad = (*self.grad).borrow_mut(); //&self.grad.get();
//         let mut data = (*self.data).borrow_mut(); //&self.grad.get();
//         let o_grad = *(*other.grad).borrow(); //&self.grad.get();
//         let o_data = *(*other.data).borrow(); //&self.grad.get();

//         *data /= o_data;
//         *grad /= o_grad;

//         // other * self.powf(-1.0)
//         // self.data.set(self.data.get() / other.data.get());
//         // self.grad.set(self.grad.get() / other.grad.get());
//         self._backward = Arc::new(Box::new(mul_backward));
//     }
// }
impl Div<f64> for Value {
    type Output = Value;
    fn div(self, rhs: f64) -> Self::Output {
        self * rhs.powf(-1.0)
    }
}
impl Div<Value> for f64 {
    type Output = Value;
    fn div(self, rhs: Value) -> Self::Output {
        rhs.powf(-1.0) * self
        // self.powf(-1.0) * rhs
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_div_values() {
        let x = Value::newd(100.0, "x".to_string());
        let y = Value::newd(2.0, "y".to_string());
        // let z = x.add(&y);
        let mut z = x.clone() / y.clone();
        assert_eq!(*(*z.data).borrow(), 50.0);
        z = z.clone() / Value::newd(2.0, "".to_string());
        assert_eq!(*(*z.data).borrow(), 25.0);
        println!(" z:{:#?}", z);
        let zz = z / 2.0;
        assert_eq!(*(*zz.data).borrow(), 12.5);
        let zz = 2.0 / zz;
        assert_eq!(*(*zz.data).borrow(), 0.16);
        zz.backward();
        println!(" z:{:#?}", zz);
        assert_eq!(*(*x.clone().grad).borrow(), -0.0016);
        assert_eq!(*(*y.clone().grad).borrow(), 0.08);
    }
}
