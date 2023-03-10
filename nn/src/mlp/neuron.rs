use crate::mlp::module::Module;
use crate::tensor::value::Value;
use rand::distributions::{Distribution, Uniform};
use std::{borrow::Borrow, rc::Rc};

#[derive(Debug)]
pub struct Neuron {
    w: Rc<Vec<Value>>,
    b: Value,
    nonlin: bool,
}

impl Neuron {
    pub fn new(nin: usize, nonlin: bool, l: f64) -> Neuron {
        let mut rng = rand::thread_rng();
        let die = Uniform::from(-1.0..1.0);
        let w = Rc::new(
            (0..nin)
                .map(|i| {
                    Value::new(
                        die.sample(&mut rng),
                        // 0.0001 + rng.gen_range(-1.0..1.0),
                        // (l * 0.001) + 0.1 * i as f64 * l as f64,
                        // 2.0 * i as f64 * l as f64,
                        vec![],
                        "".to_string(),
                        format!("w{}", i),
                    )
                })
                .collect(),
        );
        Neuron {
            w,
            b: Value::newd(0.0, "b".to_string()),
            nonlin,
        }
    }

    pub fn call(&self, x: Vec<Value>) -> Value {
        let act = self
            .w
            .iter()
            .zip(x.iter())
            .map(|(wi, xi)| wi.clone() * xi.clone())
            .sum::<Value>()
            + self.b.clone();
        if self.nonlin {
            act.relu()
        } else {
            act
        }
    }
}

impl Module for Neuron {
    fn parameters(&self) -> Vec<&Value> {
        let mut params = self.w.iter().collect::<Vec<&Value>>();
        params.push(&self.b);
        params
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::mlp::neuron::Neuron;

    #[test]
    fn test_neuron() {
        let neuron = Neuron::new(3, true, 0.1);
        let output = neuron.call(Value::vec(&[1.0, 2.0, 3.0]));
        output.backward();
        println!("meuron output: {:#?}", output);
    }
}
