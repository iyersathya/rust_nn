use nn::mlp::mlp::MLP;
use nn::mlp::module::Module;
use nn::tensor::value::Value;
use nn_test::moon_data::{get_X, get_Y};
use rand::seq::IteratorRandom;

fn loss(X: &[[f64; 2]; 100], y: &[f64], model: &MLP, batch_size: usize) -> (Value, f64) {
    let ri: Vec<usize> = (0..X.len())
        .into_iter()
        .choose_multiple(&mut rand::thread_rng(), batch_size);

    let xb: Vec<[f64; 2]> = X
        .iter()
        .enumerate()
        // .filter_map(|(i, x)| if ri.contains(&i) { Some(x) } else { None })
        .filter_map(|(i, x)| if ri.contains(&i) { Some(x) } else { None })
        .cloned()
        .collect();

    let mut yb = vec![];
    for (i, yy) in y.iter().enumerate() {
        if ri.contains(&i) {
            yb.push(yy);
        }
    }

    // run model
    let inputs = xb;
    let scores: Vec<Value> = inputs.iter().map(|input| model.call(input)).collect();

    let losses: Value = y
        .iter()
        .zip(scores.iter())
        .map(|(&yi, scorei)| (1.0 + scorei.clone() * (yi * -1.0)).relu())
        .sum();

    let data_loss = losses / y.len() as f64;
    let alpha = 1e-4;
    let reg_loss = alpha
        * model
            .parameters()
            .iter()
            .map(|p| p.get_grad() * p.get_data())
            .sum::<f64>();

    let total_loss = data_loss + reg_loss;

    let accuracy: Vec<f64> = y
        .iter()
        .zip(scores.iter())
        .map(|(&yi, scorei)| {
            if ((yi > 0.0) == (scorei.get_data() > 0.0)) {
                1.0
            } else {
                0.0
            }
        })
        .collect();
    (
        total_loss,
        accuracy.iter().sum::<f64>() / accuracy.len() as f64,
    )
}

fn main() {
    let model = MLP::new(2, &[32, 32, 1]);
    println!("number of parameters {}", model.parameters().len());
    let X = get_X();
    let Y = get_Y();
    let total_loss = loss(&X, &Y, &model, 12);
    print!(
        "total_loss : {:?},{}",
        total_loss.0.get_data(),
        total_loss.1
    );

    // optimization loop
    for i in 0..100 {
        // Step1: Forward
        let (total_loss, acc) = loss(&X, &Y, &model, 12);

        // Step2: backward
        model.zero_grad();
        total_loss.backward();
        // update params using SGD
        let learning_rate = 1.0 - (0.9 * i as f64) / 100.0;
        // let learning_rate = 0.0;
        model.parameters().iter().for_each(|p| {
            let data = p.get_data() - (learning_rate * p.get_grad());
            p.set_data(data);
        });
        if i % 1 == 0 {
            println!(
                "step: {} loss: {}, accuracy {}",
                i,
                total_loss.get_data(),
                acc * 100.0
            )
        }
    }
}
