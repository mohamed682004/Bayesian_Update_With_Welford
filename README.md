# Bayesian Posterior Updates with Streaming Data Chunks & Welford's Algorithm

This project demonstrates how to perform Bayesian updates for the parameters (mean and variance) of numeric features in a dataset. It processes data in chunks, simulating a streaming data scenario, and uses an empirical prior derived from the first chunk. For efficient and numerically stable variance estimation as new data arrives, the project incorporates Welford's algorithm.

The primary application shown here is on a dataset related to calorie expenditure, likely sourced from Kaggle, involving user exercise and calorie information.

## Table of Contents
- [Project Overview](#project-overview)
- [How it Works](#how-it-works)
  - [Initialization](#initialization)
  - [Processing Subsequent Chunks](#processing-subsequent-chunks)
  - [Bayesian Update Equations](#bayesian-update-equations)
- [Code Structure](#code-structure)
- [Key Features](#key-features)
- [Output](#output)
- [Potential Extensions](#potential-extensions)

## Project Overview

In many real-world scenarios, data arrives sequentially or in batches rather than being available all at once. This project simulates such a scenario by dividing a dataset into several chunks. It then iteratively updates the posterior distributions of feature means using Bayesian inference.

The core idea is to:
1.  Initialize prior beliefs about feature parameters (mean `mu` and variance `sigma_squared`) using the first chunk of data.
2.  For each subsequent chunk:
    * Calculate the likelihood of the new data.
    * Combine this likelihood with the prior belief to form an updated posterior distribution.
    * This posterior then becomes the prior for the next chunk.
3.  Use Welford's algorithm to maintain a running estimate of variance for the incoming data, which is crucial for calculating the likelihood variance.

This approach allows for an evolving understanding of the data's underlying parameters as more information becomes available.

## How it Works

The process can be broken down into the following steps:

### Initialization
1.  **Load Data**: The full dataset (e.g., `calories.csv` and `exercise.csv` merged) is loaded.
2.  **Feature Selection**: Numeric feature columns are identified for analysis.
3.  **Data Chunking**: The dataset is divided into a predefined number of chunks.
4.  **Empirical Priors**: The first chunk of data is used to establish initial (empirical) priors for the mean (`mu_prior`) and variance (`sigma_squared_prior`) of each numeric feature.
    * `mu_prior = mean of feature in chunk 1`
    * `sigma_squared_prior = variance of feature in chunk 1`
5.  **Running Statistics (Welford's Algorithm)**: Welford's algorithm is initialized using the first chunk to keep track of the count (`n`), mean, and sum of squared differences from the mean (`M2`) for variance estimation. This allows for stable, one-pass variance updates.
    * `running_stats[feature]['n'] = len(chunk1)`
    * `running_stats[feature]['mean'] = chunk1[feature].mean()`
    * `running_stats[feature]['M2'] = ((chunk1[feature] - chunk1[feature].mean())**2).sum()`

### Processing Subsequent Chunks
For each new chunk of data (starting from the second chunk):
1.  **Current Chunk Statistics**:
    * Calculate the mean of the current chunk for each feature (`x_bar_chunk`).
    * Get the number of samples in the current chunk (`n_chunk`).
2.  **Update Running Variance (Welford's Algorithm)**:
    * The overall count `running_stats['n']` is incremented by `n_chunk`.
    * The overall mean `running_stats['mean']` is updated.
    * The `M2` value `running_stats['M2']` is updated using the new data and the existing running mean.
    * The **likelihood variance** (`sigma_squared_likelihood`) is then estimated as:
        `sigma_squared_likelihood = running_stats['M2'] / (running_stats['n'] - 1)` (if `running_stats['n'] > 1`, otherwise it defaults to the prior variance to avoid division by zero or unstable estimates with very few points).
3.  **Bayesian Update**: For each feature, the posterior mean (`mu_post`) and posterior variance (`var_post`) are calculated using the current prior (`mu_prior`, `sigma_squared_prior` from the previous step/initialization) and the statistics from the current chunk (`x_bar_chunk`, `n_chunk`) along with the `sigma_squared_likelihood`.

    ### Bayesian Update Equations
    Assuming a Gaussian likelihood and Gaussian prior, the posterior is also Gaussian.
    * **Posterior Variance (`var_post`)**:
        ```
        var_post = 1 / (1/sigma_squared_prior + n_chunk/sigma_squared_likelihood)
        ```
    * **Posterior Mean (`mu_post`)**:
        ```
        mu_post = var_post * (mu_prior/sigma_squared_prior + (n_chunk * x_bar_chunk)/sigma_squared_likelihood)
        ```

4.  **Track True Mean**: The "true" sample mean is calculated by concatenating all data seen so far (from chunk 1 up to the current chunk) and computing its mean. This serves as a benchmark to compare against the Bayesian posterior mean.
5.  **Update Priors for Next Iteration**: The calculated `mu_post` and `var_post` become the `mu_prior` and `sigma_squared_prior` for the next chunk.
6.  **Output**: The posterior mean and the true mean are printed for each feature after processing each chunk.


## Code Structure

The core logic is implemented in a Jupyter Notebook (`Bayesian_Update_With_Welford.ipynb`).
The notebook typically includes:
* Loading and preprocessing data (merging, encoding).
* Dividing data into chunks.
* Implementation of the iterative Bayesian update loop.
* Welford's algorithm for running variance.
* Printing results at each step.

## Key Features
* **Iterative Bayesian Learning**: Demonstrates how beliefs about parameters can be updated sequentially.
* **Data Chunking**: Simulates processing of streaming or batch data.
* **Empirical Priors**: Uses initial data to form priors, a common practice when no strong domain-specific prior is available.
* **Welford's Algorithm**: Provides a numerically stable method for calculating running variance, avoiding issues with catastrophic cancellation that can occur with naive variance formulas.
* **Comparison with True Mean**: Shows how the posterior mean converges towards the true sample mean as more data is processed.

## Output
After processing each chunk (starting from the second chunk), the script prints the following for each numeric feature:
After Chunk {i}: {feature_name} - Posterior: {posterior_mean:.4f}, True: {true_mean_so_far:.4f}This allows observation of how the Bayesian estimate (Posterior) evolves and compares to the cumulative sample mean (True) as more data is incorporated.

## Potential Extensions
* Implement updates for other distributions (e.g., Bernoulli for binary features).
* Visualize the convergence of posterior means and the shrinking of posterior variances.
* Compare different prior choices.
* Apply the method to a real-time data stream.
* Incorporate more sophisticated models or feature engineering.
