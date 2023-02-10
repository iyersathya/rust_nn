use nn::mlp::mlp::MLP;
use nn::mlp::module::Module;
use nn::tensor::value::Value;

fn main() {
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
    for i in 0..100 {
        ypred = xs.iter().map(|x| mlp.call(x)).collect();

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
        loss_v.push(loss.get_data());
    }
    ys.iter().for_each(|y| println!("Actual :{}", y));
    ypred
        .iter()
        .for_each(|yp| println!("Predicted :{}", yp.get_data()));
}
