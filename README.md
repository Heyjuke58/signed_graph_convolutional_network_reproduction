# signed_graph_convolutional_network_reproduction
Reproduction of paper Signed Graph Convolutional Network

The result for tables 6 and 7 in our report can be found in file runs/res_2022-01-24_13-45-46.csv

## Contents

`data`: includes the datasets we used

`runs`: includes the logs of our experiment runs (Experiment 1)

`runs_dist`: includes the logs of our statistical significance analysis (Experiment 2)

`runs_lambda`: includes the logs of our hyperparameter sensitivity analysis (Experiment 3)

`src`: contains our models, the training and evaluation pipeline


## Running the Code

### Environment

We provide a conda `environment.yml` file which includes the environment which we used.

Set it up with this command:

```sh
conda env create -f environment.yml
```

### Experiments

To run Experiment 1, simply run `main.py`. For the other experiments, you can adjust the specific independent variables (`sgcn2_hyperpars_torch["lamb"]`, `REPEATS`) in `main.py`.

