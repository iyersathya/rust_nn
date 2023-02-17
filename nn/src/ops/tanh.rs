use crate::tensor::value::Value;
use log::debug;
use std::rc::Rc;
use std::sync::Arc;
fn tanh_backward(out: &Value) {
    let x = out._prev.get(0).unwrap();

    let t = out.get_data();
    let grad_out = out.get_grad();

    let grad = 1.0 - ((t * t) * grad_out);

    x.set_grad(x.get_grad() + grad);

    debug!(
        "tan_backwards({}) label {} grad {}",
        out.get_label(),
        x.get_label(),
        x.get_grad()
    );
}

impl Value {
    pub fn tanh(self) -> Value {
        let x = self.get_data();
        let tanh = f64::tanh(x);
        let mut out = Value::new(
            tanh,
            vec![Arc::new(self)],
            "tanh".to_string(),
            "".to_string(),
        );
        out._backward = Rc::new(tanh_backward);
        out
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_tanh() {}
}
