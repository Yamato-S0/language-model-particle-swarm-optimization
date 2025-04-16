# Language Model Particle Swarm Optimization

This repository contains the implementation of the Language Model Particle Swarm Optimization (LMPSO) algorithm, which is a novel approach to explictly integrate large language models (LLMs) into the Particle Swarm Optimization (PSO) framework.
The paper describing this work is available at [arXiv:2504.09247](https://arxiv.org/abs/2504.09247).

## How to Use

### Install the required packages

```bash
pip install -r requirements.txt
```

### Run the code

#### Traveling Salesman Problem

```bash
cd scripts
python lmpso/tsp.py --num_of_cities <number_of_cities e.g. 5, 10, 20, 30> --seed <seed for city positions>
```

#### Heuristic Improvement for TSP

```bash
cd scripts
python lmpso/heuristic_improvement_tsp.py
```

####
