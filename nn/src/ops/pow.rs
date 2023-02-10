use crate::tensor::value::Value;
use log::debug;
use std::sync::Arc;

fn pow_backward(out: &Value) {
    let x = out._prev.get(0).unwrap();
    let y = out._prev.get(1).unwrap();
    let mut grad_x = (*x.grad).borrow_mut();
    let data_x = *(*x.data).borrow_mut();
    let data_y = *(*y.data).borrow_mut();
    let grad_out = *(*out.grad).borrow();
    let pow_grad = data_y * data_x.powf(data_y - 1.0) * grad_out;
    // self.grad += (other * self.data**(other-1)) * out.grad
    *grad_x += pow_grad;

    debug!(
        "pow_backwards({}) label {} grad {}",
        out._label.borrow(),
        x._label.borrow(),
        grad_x
    );
}
impl Value {
    pub fn pow(self, other: Value) -> Value {
        let x = (*self.data).borrow();
        let o = (*other.data).borrow();
        let p = (*x).powf(*o);
        let mut out = Value::new(
            p,
            vec![Arc::new(self.clone()), Arc::new(other.clone())],
            "^".to_string(),
            "".to_string(),
        );
        out._backward = Arc::new(Box::new(pow_backward));
        out
    }
    pub fn powf(self, other: f64) -> Value {
        let out = Value::new(other, vec![], "^".to_string(), "powf".to_string());
        self.pow(out)
    }
}
