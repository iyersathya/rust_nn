use crate::tensor::value::Value;
use log::debug;
use std::rc::Rc;
use std::sync::Arc;

fn pow_backward(out: &Value) {
    let x = out._prev.get(0).unwrap();
    let y = out._prev.get(1).unwrap();

    let data_x = x.get_data();
    let data_y = y.get_data();
    let grad_out = out.get_grad();
    let pow_grad = data_y * data_x.powf(data_y - 1.0) * grad_out;
    // self.grad += (other * self.data**(other-1)) * out.grad
    x.set_grad(x.get_grad() + pow_grad);

    debug!(
        "pow_backwards({}) label {} grad {}",
        out.get_label(),
        x.get_label(),
        x.get_grad()
    );
}
impl Value {
    pub fn pow(self, other: Value) -> Value {
        let x = self.get_data();
        let o = other.get_data();
        let p = (x).powf(o);
        let mut out = Value::new(
            p,
            vec![Arc::new(self.clone()), Arc::new(other.clone())],
            "^".to_string(),
            "".to_string(),
        );
        out._backward = Rc::new(pow_backward);
        out
    }
    pub fn powf(self, other: f64) -> Value {
        let out = Value::new(other, vec![], "^".to_string(), "powf".to_string());
        self.pow(out)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    #[test]
    fn test_pow() {}
}
