use crate::mlp::module::Module;
use crate::mlp::neuron::Neuron;
use crate::tensor::value::Value;
use std::rc::Rc;

pub struct Layer {
    neurons: Rc<Vec<Neuron>>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize, nonlin: bool) -> Layer {
        Layer {
            neurons: Rc::new((0..nout).map(|_| Neuron::new(nin, nonlin)).collect()),
        }
    }

    pub fn call(&self, x: &Vec<Value>) -> Vec<Value> {
        self.neurons.iter().map(|n| n.call(x.to_owned())).collect()
    }
}
impl Module for Layer {
    fn parameters(&self) -> Vec<&Value> {
        self.neurons.iter().flat_map(Module::parameters).collect()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn print_value(value: &Value) {
        println!(
            "param: label:{:?}, data:{:?},grad:{:?}",
            value.get_label(),
            value.get_data(),
            value.get_grad()
        );
    }
    fn print_params(params: &Vec<&Value>) {
        for p in params.iter() {
            print_value(p);
        }
    }
    #[test]
    fn test_layer() {
        let layer = Layer::new(4, 5, true);
        let output = layer.call(&Value::vec(&[1.0, -2.0, 3.0]));
        println!("layer: {:#?}", layer.neurons);
        for o in output.iter() {
            o.backward();
            println!("layer output : {:#?}", o);
        }
        print_params(&layer.parameters());
    }

    #[test]
    fn test_layer_large() {
        let layer = Layer::new(10, 10, true);
        let output = layer.call(&Value::vec(&[1.0, -2.0, 3.0]));
        println!("layer: {:#?}", layer.neurons);
        for o in output.iter() {
            o.backward();
            println!("layer output : {:#?}", o);
        }
        print_params(&layer.parameters());
    }
}
