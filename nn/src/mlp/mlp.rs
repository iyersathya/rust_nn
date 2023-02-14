use crate::mlp::layer::Layer;
use crate::mlp::module::Module;
use crate::tensor::value::Value;
use std::rc::Rc;

pub struct MLP {
    layers: Rc<Vec<Layer>>,
}

impl MLP {
    pub fn new(nin: usize, nouts: &[usize]) -> MLP {
        let sz = [nin]
            .iter()
            .chain(nouts.iter())
            .cloned()
            .collect::<Vec<_>>();
        let layers = Rc::new(
            (0..nouts.len())
                .map(|i| Layer::new(sz[i], sz[i + 1], i != nouts.len() - 1))
                .collect(),
        );
        MLP { layers }
    }
    pub fn call(&self, x: &[f64]) -> Value {
        let mut y = Value::vec(x);
        for layer in self.layers.iter() {
            y = layer.call(&y);
        }
        y[0].clone()
    }
}
impl Module for MLP {
    fn parameters(&self) -> Vec<&Value> {
        self.layers.iter().flat_map(Module::parameters).collect()
    }
}

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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_mlp() {
        // let x = [2.0, 3.0, -1.0];
        let x = [2.0, 3.0, -1.0];

        // let mlp = MLP::new(3, &[4, 4, 1]);
        let mlp = MLP::new(2, &[4, 4, 1]);
        let output = mlp.call(&x);
        output.backward();
        println!("output: {:?}", output);
        let params = mlp.parameters();
        print_params(&params);
        let cal_params = ((2 * 4) + 4) + ((4 * 4) + 4) + (4 + 1);
        assert_eq!(params.len(), cal_params);
    }

    #[test]
    fn test_nn() {
        let x = [2.0, 3.0, -1.0];
        let mlp = MLP::new(3, &[4, 4, 1]);
        let output = mlp.call(&x);
        println!("MLP output: {:?}", output);
        let xs = [
            [2.0, 3.0, -1.0],
            [3.0, -1.0, 0.5],
            [0.5, 1.0, 1.0],
            [1.0, 1.0, -1.0],
        ];
        let ys = [1.0, -1.0, -1.0, 1.0];
        let mut loss_v = vec![];
        let mut ypred: Vec<Value> = vec![];
        for i in 0..200 {
            ypred = xs.iter().map(|x| mlp.call(x)).collect();

            // loss = sum( (yout - ygt)**2 for ygt,yout in zip(ys,ypred))
            let loss: Value = ys
                .iter()
                .zip(&ypred)
                .map(|(ygt, yout)| (yout.clone() - *ygt).powf(2.0))
                .sum();

            mlp.zero_grad();
            loss.backward();
            mlp.parameters().iter().for_each(|p| {
                let data = p.get_data() - (0.01 * p.get_grad());
                p.set_data(data);
            });
            // print_params(&mlp.parameters());
            loss_v.push(loss.get_data());
            // println!("loss : {:?}", loss.get_data());
        }

        let params = mlp.parameters();
        print_params(&params);
        let cal_params = ((3 * 4) + 4) + ((4 * 4) + 4) + (4 + 1);
        assert_eq!(params.len(), cal_params);

        ys.iter().for_each(|y| println!("Actual :{}", y));
        ypred
            .iter()
            .for_each(|yp| println!("Predicted :{}", yp.get_data()));
    }
}
