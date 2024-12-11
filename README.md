# Portfolio Optimization 

[![MIT License][license-shield]][license-url]


<br />
<p align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="https://github.com/nicwjh/Portfolio-Optimization/blob/main/figures/project_logo.png" alt="Logo" width="300" height="300">
  </a>

  <h3 align="center">Portfolio Optimization</h3>

  <p align="center">
    A machine-learning-based approach to portfolio optimization.
    <br />
    <br />
  </p>
</p>



## About
Portfolio optimization refers to the quantitative method that helps investors choose the best mix of assets to achieve their investment goals. In this project, we propose a machine learning based approach to portfolio optimization. We deploy 4 supervised learning methods - Principal Components Regression (PCR), Random Forests (RF), Weighted Moving Average (WMA), and Gated Recurrent Unit (GRU) Neural Networks to predict NASDAQ-100 stock prices before optimizing our constructed portfolio using the Markowitz Mean-Variance framework to determine asset weights. In our analysis, we aim to develop a portfolio that combines cutting-edge machine learning methods with modern portfolio theory to achieve optimal return-to-risk for investors seeking controlled risk with strong returns. Through a sparsified optimization process, we present an optimization strategy that delivers superior risk-adjusted returns as compared to Standard and Poor (S\&P) 500 and NASDAQ-100 benchmark portfolios over the same time horizon. 

A portfolio rebalancing strategy for this work is still in progress. Details on current findings can be found in the [final report](https://github.com/nicwjh/Portfolio-Optimization/blob/main/Portfolio_Optimization.pdf).

## Built With

The main packages used in our project:
* pandas
* numpy
* sklearn
* matplotlib.pyplot
* os

### Methods 
* Gated Recurrent Unit (GRU) Neural Networks
* Random Forest
* Weighted Moving Average
* Principal Components Regression
* Mean-Variance Optimization

### Technologies 
* Python

## Getting Started


### Prerequisites

- [pandas](https://pandas.pydata.org/) (A data manipulation and analysis library providing data structures like DataFrames for Python.)
- [numpy](https://numpy.org/) (A library for numerical computing in Python, providing support for large, multi-dimensional arrays and matrices.)
- [scikit-learn](https://scikit-learn.org/) (A machine learning library for Python, offering tools for classification, regression, clustering, and dimensionality reduction.)




## Author
|Name     |  Handle   | 
|---------|-----------------|
|[Nicholas Wong](https://github.com/nicwjh)| @nicwjh        |

## License
Distributed under the MIT License - `LICENSE`. 

Repository  Link: [https://github.com/nicwjh/Portfolio-Optimization)

## Acknowledgements
I would like to thank Professor Soroush Saghafian for his mentorship throughout this project. The exceptional learning environment and resources provided by [Harvard University](https://github.com/harvard) has also been instrumental in shaping this work. 

[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://opensource.org/licenses/MIT
