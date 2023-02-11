use crate::tensor::value::Value;
use log::debug;
use std::sync::Arc;

fn tanh_backward(out: &Value) {
    let x = out._prev.get(0).unwrap();

    let mut grad_x = (*x.grad).borrow_mut();
    let t = *(*out.data).borrow();
    let grad_out = *(*out.grad).borrow();

    let grad = 1.0 - ((t * t) * grad_out);
    *grad_x += grad;

    debug!(
        "tan_backwards({}) label {} grad {}",
        out._label.borrow(),
        x._label.borrow(),
        grad_x
    );
}

impl Value {
    pub fn tanh(self) -> Value {
        let x = *(*self.data).borrow();
        let tanh = f64::tanh(x);
        let mut out = Value::new(
            tanh,
            vec![Arc::new(self)],
            "tanh".to_string(),
            "".to_string(),
        );
        out._backward = Arc::new(Box::new(tanh_backward));
        out
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_tanh() {}
}
