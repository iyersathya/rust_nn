use crate::tensor::value::Value;
use log::debug;
use std::rc::Rc;
use std::sync::Arc;
fn relu_backward(out: &Value) {
    let x = out._prev.get(0).unwrap();
    let t = out.get_data();
    let grad_out = out.get_grad();
    if t > 0.0 {
        x.set_grad(x.get_grad() + grad_out);
    }
    debug!(
        "relu_backwards({}) label {} grad {}",
        out.get_label(),
        x.get_label(),
        x.get_grad()
    );
}
impl Value {
    pub fn relu(self) -> Value {
        if self.get_data() < 0.0 {
            self.set_data(0.0)
        }
        let mut out = Value::new(
            self.get_data(),
            vec![Arc::new(self.clone())],
            "relu".to_string(),
            "".to_string(),
        );
        out._backward = Rc::new(relu_backward);
        out
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_relu() {}
}
