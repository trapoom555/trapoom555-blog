---
author: "trapoom555"
title: "[Draft] Imitation Learning"
date: "2024-01-18"
math: mathjax
draft: true
---

RL suffers from several drawbacks
1. Determining a reward function that represent the true performance objectives can be challenging.
2. The reward signal may be sparse

> Imitation Learning : The reward function is described implicitly through expert demonstrations 

# Definition

Instead of defining a reward function $r_t = R(s_t, a_t)$, the set of demonstrations are provided.

# Problem Formulation

## System Dynamics / State Transition Model

- Makov Decision Process (MDP)
- We can't access to the reward function

$$p(s_t | s_{t-1}, a_{t-1})$$

## Policy

$$u_t = \pi(s_t)$$

## A Set of Expert Demonstration

Consist of state-action pairs. A set of expert demonstration are drawn from an expert policy $\pi^*$

$$\xi =  \{(s_0, u_0), (s_1, u_1), ...\}$$

## Imitation Learning Problem

A system with a state transition model $p(s_t | s_{t-1}, a_{t-1})$ with states $x \in \mathcal X$ and actions $a \in \mathcal A$, the imitation learning problem is to leverage a set of demonstrations $\Xi = \{\xi_0, ... , \xi_D\}$ from an expert policy $\pi^*$ to find the policy $\hat \pi^*$ that imitates the expert policy.

# Ways to imitate

There're 2 approaches

1. **directly** imitate expert's policy by learning expert's policy e.g. *behavior cloning, DAgger*
2. **indirectly** imitate expert's policy by learning expert's reward function (a.k.a. *inverse reinforcement learning*)

# Imitate Policy
## Behavioral Cloning

This method can be accomplished by *Supervised Learning* techniques. The task is to minimize learned policy and the expert demonstrasions

$$\hat \pi ^* = {\arg \min}_\pi \sum_{\xi \in \Xi} \sum_{s \in \xi} L(\pi(s), \pi^*(s))$$

$L$ is a different loss function which can be $p$-norms family e.g. euclidean norm or $f$-divergence e.g. KL divergence depending on the form of policy

> This approach **may not yield a good performance** because most of the time the distribution of expert demonstrations $\Xi$ is not uniformly sampled across the entire state space. There's a distributional mismatch in states seen under the expert policy and learned policy.

## DAgger

DAgger (Dataset Aggregation) : Ask the expert what action it would have taken and compare it to the rolled trajectory from learned policy for some number of timesteps. -> update policy

image


- Advantage : It reduces distribution mismatch
- Disadvantage : At each timestep, the policy needs to be retrained. It's Computationally expensive

## Issue of Policy Learning
1. don't understand the expert's intention (reasons behind the expert behavior)
2. The expert may be suboptimal (not optimal)
3. A policy that is optimal for the expert may not be optimal for the agent (different dynamics, morphology, capabilities etc.)

# Inverse Reinforcement Learning
> Try to learn expert's Reward Function

## Concept
The reward function $R$ can be assume as a inner product of learnable weights $w \in \mathbb R^n$ and a given non-linear function $\phi(s, a) : \mathcal S \times \mathcal A \rightarrow \mathbb R^n$. In other words, the reward function is a linear combination of non-linear features.

$$R(s,a) = w^T \phi(s, a)$$

Recall from the RL, the value function can be expressed as

$$V_t^\pi(s)= \mathbb E_\pi \left[ \sum_{t=0}^T \gamma^t R(s_t, \pi(s_t)) \vert s_0 = s\right]$$

where $T$ is the defined time horizon.

By plugging the parameterized reward function in, we'll get

$$V_t^\pi(s)= w^T\mathbb E_\pi \left[ \sum_{t=0}^T \gamma^t \phi(s, \pi(s_t)) \vert s_0 = s\right] = w^T \mu(\pi, s)$$

Note that $\mu(\pi, s)$ is called the feature expectation.

The definition of optimal policy is following

$$V^{\pi^*}_t(s) \ge V^\pi_t(s), \quad \forall s\in \mathcal S, \quad \forall \pi $$

$${w^*}^T \mu(\pi^*, s) \ge {w^*}^T \mu(\pi, s), \quad \forall s\in \mathcal S, \quad \forall \pi $$

This condition can be used to find the vector $w^*$ given the expert policy $\pi^*$. But there's a trivial case which is when $w=0$

## Apprenticeship Learning

Avoid that trivial case by finding a reward function vector $w$ that the expert policy *maximally outperform* other policies by doing it iteratively.

> **Algorithm** : Apprenticeship Learning <br>
> **Input** : $\mu(\pi^*), \epsilon$ (epsilon is a small value threshold to terminate) <br>
> **Output** : $\hat \pi^*$ (estimated optimal policy) <br>
> **for** $i=1$ **to** $...$ **do** <br>
> $\quad$ Compute $\mu(\pi_{i-1})$ (or approx. by Monte Carlo) <br>
> $\quad$ Compute $(w_i, t_i)$ by the following optimization problem <br>
> $ \begin{array}{rlclcl} \qquad (w_i, t_i) = \underset{w,t}{\operatorname{argmax}}& t \\
\textrm{s.t.} & w^T \mu(\pi^*) \ge w^T \mu(\pi) + t & \forall \pi\in \{\pi_0, ..., \pi_{i-1} \} \\ & \Vert w\Vert_2 \le 1  \end{array}$
> $\quad$ **if** $t_i \le \epsilon$ **then** <br>
> $\qquad \hat \pi^* \leftarrow \underset{\pi \in \{ \pi_0,...,\pi_{i-1}\}}{\operatorname{argmax\ }} w^T\mu(\pi)$  (set the estimated optimal policy to be the policy giving the max Value) <br>
> $\quad$ Find the optimal $\pi_i$ using RL with the reward function defined by $w_i$

The steps can be simplified as follows
1. Find the optimal reward function
2. Use optimal reward function to find the optimal policy
3. do 1 iteratively.

Note that the optimization problem tries to optimize the smallest performance loss across all $w$ with $\Vert w\Vert_2 \le 1$

Although it solved the aforementioned trivial case, 
this approach allows the ambiguity that there could be different policies which lead to the same feature expectation.

> My Questions : How to define a nonlinear feature mapping $\phi$

### Apprenticeship Learning V.S. Behavioral Cloning

Behavioral Cloning tries to mimic the expert policy's actions. This approach may not robust when visiting the unseen states. On the other hand, Apprenticeship Learning tries to identify the importance of expert's features which is more generalizable.

## Maximum Entropy IRL

- Avoid the ambiguity that different policies may have the same trajectories
- Use the *Maximum Entropy Principle* to find the probability density function over trajectories $\tau = \{ (s_0, \pi(s_0)), (s_1, \pi(s_1)), ...\}$ as $p(\tau)$ by using the information that the feature expectation has to be the same as the expert $\mathbb E_\pi \left[ f(\tau)\right] = E_{\pi^*} \left[ f(\tau)\right]$

$$
\begin{array}{rlclcl}
p^*(\tau) = \underset{p}{\operatorname{argmax}}& \int -p(\tau)\log(p(\tau))\  d\tau \\
\textrm{s.t.} & \int p(\tau) f(\tau)\ d\tau  = \int p_{\pi^*}(\tau) f(\tau)\ d\tau \\
& \int p(\tau)\ d\tau = 1\\
& p(\tau) \ge 0, \forall\tau
\end{array}
$$
- By solving this optimization problem analytically with the Lagrange Multiplier we will get the solution as

$$
p^*(\tau, \lambda)  = \frac{1}{Z(\lambda)}e^{\lambda^T f(\tau)},\ Z(\lambda) = \int e^{\lambda^T f(\tau)} \ d\tau
$$
- Note that $p^*$ is a probability density function that most likely similar to the expert. 
- Ideally, the expert would take the maximum reward that makes $\lambda = w^*$. But we don't know $w^*$
- We can find $w^*$ by using the Maximum likelihood giving a set of demonstrations $\Xi=\{\xi_0, \xi_1, ... \}$
$$
w^*=\underset{\lambda}{\operatorname{argmax}} \prod_{\xi_i \in \Xi} p^*(\xi_i, \lambda)
$$

- To achieve the same result with more simplicity, we can calculate the maximum log likelihood instead

$$
w^*=\underset{\lambda}{\operatorname{argmax}} \sum_{\xi_i \in \Xi} \left[ \lambda^T f(\tau) - \ln Z(\lambda)\right]
$$

- We can solve this problem by using the gradient descent algorithm to update the $\lambda$

$$
\nabla_\lambda J(\lambda)= \sum_{\xi_i \in \Xi} \left[ f(\tau) - \mathbb E_{\tau\sim p^*} \left[ f(\tau)\right] \right]
$$

Here are the steps

1. Initialize $\lambda$ and collect the set of expert demonstrations $\Xi$
2. Use RL to find the optimal policy $\pi_\lambda$ at this stage with $w=\lambda$
3. Use $\pi_\lambda$ to sample trajectories and find the $\mathbb E_{\tau\sim p^*} \left[ f(\tau)\right]$
4. Gradient update to update the value of $\lambda$
5. Do 1 until it converges

## Inverse Q-Learning
