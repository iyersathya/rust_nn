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
cargo run
```

There are around 15 test for testing different aspect of the code, following rust conventions, test cases are directly available in the respective file along with code.
Command to run tests

```
cargo test
```



