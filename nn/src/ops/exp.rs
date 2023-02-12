use crate::tensor::value::Value;
use log::debug;
use std::sync::Arc;

fn exp_backward(out: &Value) {
    let x = out._prev.get(0).unwrap();

    x.set_grad(x.get_grad() + out.get_data() * out.get_grad());

    debug!(
        "exp_backwards({}) label {} grad {}",
        out.get_label(),
        x.get_label(),
        x.get_grad()
    );
}

impl Value {
    pub fn exp(self) -> Value {
        let x = self.get_data();
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

#[cfg(test)]
mod test {
    use super::*;
}
