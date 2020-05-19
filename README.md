
## Deep Reinforcement Learning Global Routing
This project is based on our published paper "A Deep Reinforcement Learning Approach for Global Routing" (https://arxiv.org/pdf/1906.08809.pdf).
It uses deep Q-learning (DQN) to solve the challenging global routing problem by leveraging the "conjoint optimization" mechanism  of DQN. Superior performance compared to A* search (which is a typical baseline routing algorithm) in certain types of problems. As shown in the picture, the pipeline consists of "problem sets generator (see another open source Github repo: haiguanl/GlobalRoutingProblemGenerator for deatails)", "multipin decomposition", "A*Search Router" and "DQN Router".
<p align="center">
<img src="Fig/Pipeline.png" alt="drawing" width="800">
</p>

Two types problems are defined to analyze the results (Type I/Type II). Some basic definitions and why DQN works better than A* in Type II problems are shown below. Interestingly, most circuits design problems could be ascribeed as Type II, where the most important feature is that routing resource is scarce (some edges's capacity used up during routing). 

<p align="center">
<img src="Fig/TwoTypeProblem.png" alt="drawing" width="800">
</p>
<p align="center">
<img src="Fig/ExplainTwoType.png" alt="drawing" width="800">
</p>


#### 1. Environment Setup:
To setup your environment (which contains depencies with the correct version), use the conda environment specified in environment.yml by simply doing the following with a recent version of conda in Linux/Bash (https://conda.io/projects/conda/en/latest/user-guide/install/index.html), in the repository path:
```
conda env create
source activate GlobalRoutingProblemGenerator
```
To deactivate the virtual environment:
'''
conda deactivate
'''

#### 2. Running Experiment

#### 3. Results Evaluation
