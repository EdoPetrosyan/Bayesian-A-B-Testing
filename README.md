# Bayesian A/B testing

This repository contains Python implementations of two popular multi-armed bandit algorithms - **Epsilon Greedy** and **Thompson Sampling**. These algorithms are commonly used in online advertising, recommendation systems, and other areas where one needs to optimize decision-making under uncertainty.

### BONUS
For better implementation:

Integrating Hoeffding's Inequality for UCB1:
For the epsilon-greedy algorithm, we could enhance our approach by incorporating Hoeffding's Inequality to derive the Upper Confidence Bound (UCB1) strategy. This would enable us to balance exploration and exploitation more effectively and potentially improve convergence rates.

Dynamic Precision Estimation:
Implement a mechanism to estimate the precision parameter dynamically during the course of the algorithm. Instead of relying on known precision values, adaptively estimate precision based on observed rewards and update the posterior distribution accordingly.

Unit Testing:
Develop comprehensive unit tests to validate the correctness and reliability of individual components and functionalities. Unit tests help ensure that changes or updates to the codebase do not introduce regressions and maintain the integrity of the project.

Configuration Files:
Utilize configuration files (e.g., YAML or JSON) to store parameters and settings for the algorithms. This makes it easier to adjust parameters without modifying the source code, promoting flexibility and experimentation.

Exploration-Exploitation Trade-off:
Visualize the exploration-exploitation trade-off by plotting the proportion of exploration versus exploitation actions taken by the algorithm at each trial. This helps in understanding how the algorithm balances between exploring new options and exploiting the best-known options.

Modular Code Structure:
Break down the code into clear modules for each type of algorithm, data management, visualization, and statistical analysis. This modular structure improves code readability, maintainability, and allows for easier collaboration among contributors.

Model Selection and Validation:
Invest some time in exploring techniques for model selection and validation, particularly for Thompson Sampling. Methods such as cross-validation or Bayesian model comparison can help us identify the most suitable model for representing reward distributions.

Parallelization and Efficiency:
Consider implementing parallelization techniques to enhance the efficiency of both algorithms, especially in scenarios with a large number of bandit arms or trials. This could significantly accelerate experimentation and improve scalability.




