# 01-policy-evaluation

This directory contains code and resources related to **policy evaluation** in reinforcement learning.

## Contents
- Implementations and examples for policy evaluation algorithms
- Supporting scripts and documentation

## Mathematical Principle

Policy evaluation estimates the **state-value function** \( v_\pi(s) \) for a given policy \( \pi \).

### State-Value Function
The state-value function under policy \( \pi \) is:

\[
v_\pi(s) = \mathbb{E}_\pi \left[ \sum_{t=0}^\infty \gamma^t R_{t+1} \;\middle|\; S_0 = s \right]
\]

Where:
- \( s \) = current state  
- \( \pi(a|s) \) = probability of taking action \( a \) in state \( s \) under policy \( \pi \)  
- \( R_{t+1} \) = reward after transitioning from step \( t \) to \( t+1 \)  
- \( \gamma \in [0,1] \) = discount factor for future rewards  

### Bellman Expectation Equation
This function satisfies the **Bellman expectation equation**:

\[
v_\pi(s) = \sum_{a} \pi(a|s) \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma v_\pi(s') \right]
\]

Where:
- \( p(s', r \mid s, a) \) = transition probability of moving to \( s' \) and receiving reward \( r \) from \( s \) using \( a \)  

### Iterative Policy Evaluation Algorithm
Numerically, \( v_\pi \) can be estimated iteratively:

\[
v_{k+1}(s) = \sum_{a} \pi(a|s) \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma v_k(s') \right]
\]

Repeat until convergence:

\[
\max_s \; \left| v_{k+1}(s) - v_k(s) \right| < \theta
\]
where \( \theta \) is a small positive threshold.

## References
- Sutton & Barto, *Reinforcement Learning: An Introduction*  
- Bellman, R. *Dynamic Programming* (1957)
