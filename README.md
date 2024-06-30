# Bachelor Thesis
## Project Overview

This project evaluates the effectiveness of various forecasting models, with a focus on Local Linear Forests (LLFs), as introduced by Friedberg et. al (2020), in predicting the volatility of cryptocurrencies. The study compares LLFs to established models such as Generalized Autoregressive Conditional Heteroskedasticity (GARCH), GJR-GARCH, Heterogeneous Autoregressive model based on Realized Volatility (HAR-RV), and Random Forest (RF). Additionally, the study explores the models' performance across different market cycles, their utility benefits for risk-targeting investors, and their effectiveness in forecasting volatility for large-cap vs. mid-cap cryptocurrencies.

## Project Structure

The project is organized into the following directories:

- `data`: Contains all the datasets used for the study. It includes hourly and daily data of 8 cryptocurrencies (BTC, ETH, USDT, BNB, BCH, LTC, ICP, MATIC)
- `forecasting`: Includes scripts and notebooks for in-sample and out-of-sample forecasting for 8 different cryptocurrencies. It also contains the analysis of market cycles, utility benefits, and comparisons between large-cap and mid-cap cryptocurrencies. We compare the performance of Local Linear Forest against GARCH(1,1), GJR-GARCH, Random Forest and HAR-RV.
- `simulation_study`: Contains code for simulation experiments to assess the performance of LLF compared to RF, Lasso RF, BART, and XGBoost based on RMSE and QLIKE metrics.

## Forecasting Analysis

### In-Sample and Out-of-Sample Forecasting
We perform both in-sample and out-of-sample forecasting for 8 different cryptocurrencies and compare the performance of Local Linear Forest against the baseline models. To evaluate whether these improvements are significant, we use the Model Confidence Set by Hansen et. al (2011).

### Market Cycle Analysis
The study evaluates how each model performs during different phases of the market cycle, including bull runs, bear markets, and consolidation periods. This analysis helps assess the models' adaptability to varying market conditions, which is crucial for enhancing risk management strategies and facilitating informed investment decision-making.

### Utility Benefits Analysis
Using the framework proposed by Bollerslev et al. (2018), we evaluate the utility benefits of volatility models for risk-targeting investors. This involves considering mean-variance preferences to allocate wealth optimally, based on both expected returns and risk measured through volatility. The study aims to determine if LLFs can provide substantial economic benefits compared to baseline models.

### Large-Cap vs. Mid-Cap Cryptocurrencies
The project also explores whether LLFs demonstrate distinct performance characteristics when forecasting for large-cap cryptocurrencies (e.g., Bitcoin, Ethereum) compared to mid-cap altcoins (e.g., Litecoin, Polygon). This assessment aims to provide comprehensive insights into the suitability and effectiveness of LLFs across different market segments.

## Simulation Study

In the simulation study, we perform three simulations to assess the performance of LLF compared to RF, Lasso RF, BART, and XGBoost. The performance is evaluated based on the Root Mean Square Error (RMSE) and QLIKE metrics. The simulations are as follows:

- Simulation 1: Data is simulated according to the GJR-GARCH framework to simulate data that resembles the data-generating process of cryptocurrency volatility.
- Simulation 2: We employ an example from Friedman (1991). This equation allows us to evaluate an algorithm’s proficiency in capturing interactions, its capability to detect quadratic patterns and its simultaneous handling of prominent linear signals
- Simulation 3: Examines the effectiveness of handling smoothness in the regression surface and sees whether the models can capture strong local trends accurately

## References
- Friedberg, R., Tibshirani, J., Athey, S., and Wager, S. (2020). Local linear forests. Journal of Computational and Graphical Statistics, 30(2):503–517.
- Friedman, J. H. (1991). Multivariate adaptive regression splines. The Annals of Statistics, 19(1):1–67.
- Hansen, P. R., Lunde, A., and Nason, J. M. (2011). The model confidence set. Econometrica, 79(2):453–497.
- Bollerslev, T., Hood, B., Huss, J., and Pedersen, L. H. (2018). Risk Everywhere: Modeling and managing volatility. The Review of Financial Studies, 31(7):2729–2773.

## Acknowledgements
This project includes code adapted from [Local Linear Forest](https://github.com/HoustonJ2013/LocalLinearForest). We modified the original code to fit our project needs. The adaptations were made such that the local regression can utilize a different dataset than those used for splitting in the decision trees.
