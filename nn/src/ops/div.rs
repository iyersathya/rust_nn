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
