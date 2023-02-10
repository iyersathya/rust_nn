use crate::tensor::value::Value;
use log::debug;
use std::sync::Arc;

fn exp_backward(out: &Value) {
    let x = out._prev.get(0).unwrap();
    let mut grad_x = (*x.grad).borrow_mut();

    let data_out = *(*out.data).borrow();
    let grad_out = *(*out.grad).borrow();
    *grad_x += data_out * grad_out;

    debug!(
        "exp_backwards({}) label {} grad {}",
        out._label.borrow(),
        x._label.borrow(),
        grad_x
    );
}

impl Value {
    pub fn exp(self) -> Value {
        let x = *(*self.data).borrow();
        let exp = f64::exp(x);
        let mut out = Value::new(
            exp,
            vec![Arc::new(self)],
            "tanh".to_string(),
            "".to_string(),
        );
        out._backward = Arc::new(Box::new(exp_backward));
        out
    }
}
