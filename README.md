# Comma Controls Challenge!
![Car](./imgs/car.jpg)

Machine learning models can drive cars, paint beautiful pictures and write passable rap. But they famously suck at doing low level controls. Your goal is to write a good controller. This repo contains a model that simulates the lateral movement of a car, given steering commands. The goal is to drive this "car" well for a given desired trajectory.

## Solution

The solution provided is a controller that uses bayesian optimization to minimize the cost of a PID policy over a given future path and state. This provides a robust framework that can adapt to any path and is very overfit resistant. It manages to beat the baseline by ~25%.

It is however much more computationally expensive than a simple PID controller. On a single A4000 the final report took around 9 hours to complete (vs the 25 minutes of the pid controller). Some tradeoffs have been made to reduce the computation on easy paths to enhance the computation/improvement ratio. Also, I've made sure that the optimization (on average) does not take longer to compute than the DEL_T constant (at least on my hardware).

The number of optimization steps done on each iteration and the results are directly correlated, but I have not conducted a thorough scale effect study on by how much.

### Future Work

The line of work that is more likely to lead to a significant increase in the compute/improvement ratio is to generate optimal paths for each csv and train a reinforcement learning agent with an imitation learning reward system using an off policy algorithm, like a Soft Actor Critic or a Twin Delayed DDPG. This would lead to a similar result that would require much less computation during inference.

There are also a couple heuristics placed inside the controller. These heuristics can be optimized for better results, but I would not expect a huge increase.

### Other Options Tried

My first approach was to train a reinforcement learning agent with an on policy algorithm (TRPO) to control the steer action of the car. This method managed to beat the baseline by ~3%.
While this seems like the best method to achieve an optimal and robust policy I believe the sampling of the action that TRPO uses has to be tweaked or you run the risk of getting stuck in a local minima or to have a solution with a lot of chattering.

Other options to accelerate convergence can be used, like starting with an imitation learning reward system and then change it to an optimal policy reward system when a certain threshold is reached.

### Considerations

Since the files are not IID I made a script that selects the subset of files that most closely match the underlying distribution of all files using the wasserstein distance. This is especially relevant when training a reinforcement learning agent, as it can help to stabilize the learning environment.

```
# This selects the 500 files that are most similar to the underlying distribution of all 20000 files.
python selectFiles.py --data_path data/ --n_files 500
```

## Geting Started
We'll be using a synthetic dataset based on the [comma-steering-control](https://github.com/commaai/comma-steering-control) dataset for this challenge. These are actual routes with actual car and road states.

```
# download necessary dataset (~0.6G)
bash ./download_dataset.sh

# install required packages
# recommended python==3.11
pip install -r requirements.txt

# test this works
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data/00000.csv --debug --controller pid 
```

There are some other scripts to help you get aggregate metrics: 
```
# batch Metrics of a controller on lots of routes
python tinyphysics.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --controller pid

# generate a report comparing two controllers
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 100 --test_controller pid --baseline_controller zero

```
You can also use the notebook at [`experiment.ipynb`](https://github.com/commaai/controls_challenge/blob/master/experiment.ipynb) for exploration.

## TinyPhysics
This is a "simulated car" that has been trained to mimic a very simple physics model (bicycle model) based simulator, given realistic driving noise. It is an autoregressive model similar to [ML Controls Sim](https://blog.comma.ai/096release/#ml-controls-sim) in architecture. It's inputs are the car velocity (`v_ego`), forward acceleration (`a_ego`), lateral acceleration due to road roll (`road_lataccel`), current car lateral acceleration (`current_lataccel`) and a steer input (`steer_action`) and predicts the resultant lateral acceleration of the car.


## Controllers
Your controller should implement a new [controller](https://github.com/commaai/controls_challenge/tree/master/controllers). This controller can be passed as an arg to run in-loop in the simulator to autoregressively predict the car's response.


## Evaluation
Each rollout will result in 2 costs:
- `lataccel_cost`: $\dfrac{\Sigma(actual\\_lat\\_accel - target\\_lat\\_accel)^2}{steps} * 100$

- `jerk_cost`: $\dfrac{\Sigma((actual\\_lat\\_accel\_t - actual\\_lat\\_accel\_{t-1}) / \Delta t)^2}{steps - 1} * 100$

It is important to minimize both costs. `total_cost`: $(lataccel\\_cost * 50) + jerk\\_cost$

## Submission
Run the following command, and submit `report.html` and your code to [this form](https://forms.gle/US88Hg7UR6bBuW3BA).

```
python eval.py --model_path ./models/tinyphysics.onnx --data_path ./data --num_segs 5000 --test_controller <insert your controller name> --baseline_controller pid
```

## Changelog
- With [this commit](https://github.com/commaai/controls_challenge/commit/fdafbc64868b70d6ec9c305ab5b52ec501ea4e4f) we made the simulator more robust to outlier actions and changed the cost landscape to incentivize more aggressive and interesting solutions.
- With [this commit](https://github.com/commaai/controls_challenge/commit/4282a06183c10d2f593fc891b6bc7a0859264e88) we fixed a bug that caused the simulator model to be initialized wrong.

## Work at comma
Like this sort of stuff? You might want to work at comma!
https://www.comma.ai/jobs
