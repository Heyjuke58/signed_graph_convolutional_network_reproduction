import matplotlib.pyplot as plt
import numpy as np

"""
General hyperparameters:
	test split size: 0.2
	val split size: 0.1
	embedding size: 64
	undirected graph: True
	repeats: 3
	seed: 1337

sgcn2:
	in_features: 64
	out_features: 64
	num_layers: 2
	lamb: 0, 3, 5, 7, 10, 20, 100
	num_epochs: 200
	num_batches: None
	xent_weights: [1, 1, 1]
	learning_rate: 0.01
	weight_decay: 1e-05
	learn_decay: 0.75
	ablation_version: sgcn2
	activation_fn: <built-in method tanh of type object at 0x7f5815d16ee0>
	val_interval: 5
	early_stopping_patience: 50
	loss_version: torch-geometric

"""
lambdas = np.asarray([0, 3, 5, 7, 10, 20, 100])
aucs = np.asarray([0.7333, 0.7712, 0.7784, 0.7788, 0.7843, 0.7895, 0.7932])
f1s = np.asarray([0.9374, 0.9353, 0.9336, 0.9349, 0.9343, 0.932, 0.9197])
fake_x = range(len(lambdas))
y_ticks = np.arange(0.72, 1.0, 0.02)

fig, axs = plt.subplots(1, 2, figsize=(8, 4))

axs[0].plot(fake_x, f1s)
axs[0].scatter(fake_x, f1s)
axs[0].set_xticks(fake_x, lambdas)
axs[0].set_yticks(y_ticks)
axs[0].set_xlabel("λ", fontsize=16)
axs[0].set_ylabel("F1", fontsize=16)

axs[1].plot(fake_x, aucs)
axs[1].scatter(fake_x, aucs)
axs[1].set_xticks(fake_x, lambdas)
axs[1].set_yticks(y_ticks)
axs[1].set_xlabel("λ", fontsize=16)
axs[1].set_ylabel("AUC", fontsize=16)

plt.tight_layout()
fig.savefig("lambda_plot.png")
