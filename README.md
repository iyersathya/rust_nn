# Neural Network from Scratch in Rust
Implementation of MLP (neural net) from scratch in Rust.

## About
This library is based on [micrograd](https://github.com/karpathy/micrograd/tree/master/micrograd) Python implementation by Andrej Karpathy. A big thank you to Andrej Karpathy for [amazing video](https://youtu.be/VMj-3S1tku0) on the same topic.

The goal of this project was to implement a neural net without using any external ML library.
The library provides basic building block upon which more complex NN can be built.

Currently the library only provides basic operators like add,mul,tanh,relu check out [op directory](https://github.com/iyersathya/rust_nn/tree/main/nn/src/ops).

[MLP](https://github.com/iyersathya/rust_nn/tree/main/nn/src/mlp) directory has a basic neural net that you can use to create a MLP model architecture and train a model.

## Building
This follows standard building command for Rust.

main.rs contains a model trained on make_moons (dataset is in [moon_data.rs](https://github.com/iyersathya/rust_nn/blob/main/nn_test/src/moon_data.rs) ) data from sklearn.datasets

Command to run training 
```
cargo run (debug slow)
time cargo run --release (release fast)
```
Output of training MLP model for moon data
```
number of parameters 337
total_loss : 1.2804298139881203,0.5step: 0 loss: 1.2290946225113997, accuracy 50
step: 1 loss: 1.604931649076011, accuracy 50
step: 2 loss: 1.0216475078695255, accuracy 71.875
step: 3 loss: 0.5226021553594029, accuracy 78.125
step: 4 loss: 0.2302806116652168, accuracy 87.5
step: 5 loss: 0.38927717139283263, accuracy 81.25
step: 6 loss: 0.17364781342216315, accuracy 96.875
step: 7 loss: 0.44921714748030217, accuracy 84.375
step: 8 loss: 0.27341771736030646, accuracy 84.375
step: 9 loss: 0.32020769792411985, accuracy 81.25
step: 10 loss: 0.4075148014134693, accuracy 75
step: 11 loss: 0.15280057300720104, accuracy 93.75
step: 12 loss: 0.16237969684317136, accuracy 93.75
step: 13 loss: 0.20193180702189634, accuracy 93.75
step: 14 loss: 0.24885841450298263, accuracy 93.75
step: 15 loss: 0.25797896946361043, accuracy 87.5
step: 16 loss: 0.5208280004591873, accuracy 84.375
step: 17 loss: 0.16211630953484454, accuracy 93.75
step: 18 loss: 0.23399434422934245, accuracy 90.625
step: 19 loss: 0.166760777015555, accuracy 93.75
step: 20 loss: 0.12634106928966152, accuracy 90.625
step: 21 loss: 0.14549331877334282, accuracy 90.625
step: 22 loss: 0.0867293254218015, accuracy 93.75
step: 23 loss: 0.13519417262986905, accuracy 96.875
step: 24 loss: 0.18755396902872248, accuracy 93.75
step: 25 loss: 0.13535701340174247, accuracy 93.75
step: 26 loss: 0.056670458724066085, accuracy 96.875
step: 27 loss: 0.09981735959766636, accuracy 100
step: 28 loss: 0.08361738021428684, accuracy 100
step: 29 loss: 0.07266831358459684, accuracy 100
step: 30 loss: 0.11407100612229992, accuracy 93.75
step: 31 loss: 0.02892621278835673, accuracy 100
step: 32 loss: 0.013721860784823101, accuracy 100
step: 33 loss: 0.01715093626092444, accuracy 100
step: 34 loss: 0.07848301692476903, accuracy 100
step: 35 loss: 0.06444944419025851, accuracy 96.875
step: 36 loss: 0.011211456200114885, accuracy 100
step: 37 loss: 0.1532424068583257, accuracy 96.875
step: 38 loss: 0.31938695769449826, accuracy 84.375
step: 39 loss: 0.33060198677624913, accuracy 84.375
step: 40 loss: 0.21114771905363616, accuracy 90.625
step: 41 loss: 0.2339124560498736, accuracy 87.5
step: 42 loss: 0.04746505475294334, accuracy 96.875
step: 43 loss: 0.011828234055930826, accuracy 100
step: 44 loss: 0.023366522389568385, accuracy 100
step: 45 loss: 0.06673587000251631, accuracy 100
```

There are around 15 test for testing different aspect of the code, following rust conventions, test cases are directly available in the respective file along with code.
Command to run tests

```
cargo test
```



