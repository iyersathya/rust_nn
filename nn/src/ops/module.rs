use crate::tensor::value::Value;

pub trait Module {
    fn zero_grad(&self) {
        for p in self.parameters() {
            p.zero_grad();
        }
    }
    fn parameters(&self) -> Vec<&Value> {
        vec![]
    }
}
