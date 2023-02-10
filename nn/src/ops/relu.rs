use crate::tensor::value::Value;
use log::debug;
use std::sync::Arc;

fn relu_backward(out: &Value) {
    let x = out._prev.get(0).unwrap();
    let mut grad_x = (*x.grad).borrow_mut();
    let t = *(*out.data).borrow();
    let grad_out = *(*out.grad).borrow();
    if t > 0.0 {
        *grad_x += grad_out;
        // X.grad.set(grad_x + t * out.grad.get());
    }
    debug!(
        "relu_backwards({}) label {} grad {}",
        out._label.borrow(),
        x._label.borrow(),
        grad_x
    );
}
impl Value {
    pub fn relu(self) -> Value {
        let mut x = (*self.data).borrow_mut();
        if *x < 0.0 {
            *x = 0.0;
        }
        let mut out = Value::new(
            *x,
            vec![Arc::new(self.clone())],
            "relu".to_string(),
            "".to_string(),
        );
        out._backward = Arc::new(Box::new(relu_backward));
        out
    }
}
