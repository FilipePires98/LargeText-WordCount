# LargeText-WordCount
A study on Exact and Approximate Occurrences Counters

## Description

The challenge of parallel event counting in a memory efficient way is not a recent topic, but it is one still under discussion as there is great room for improvement.
Most of today’s solutions perform memory optimization by applying probabilistic counters to estimate the total number of occurrences of events.

This project focuses on 2 of the most famous approximate counters to determine an estimation of the most used words of literary works from several authors in several languages and compare them to an exact counter. 
Conclusions drawn from the study applied to the dataset are presented in the project report.

## Repository Structure

/datasets - literary works taken from [Project Gutenberg](https://www.gutenberg.org/) used as input data

/report - documentation of the conducted study

/results - outputs produced by the implemented code

/src - source code of the algorithms 

## Data Visualization

<img src="https://github.com/FilipePires98/LargeText-WordCount/blob/main/results/charts/wordCount_ACC_DU.png" width="540px">

Counter estimations of each algorithm for the top 10 words.

<img src="https://github.com/FilipePires98/LargeText-WordCount/blob/main/results/plots/counterComparison_KSM_PT.png" width="540px">

Counters deviations of each algorithm for the top 50 words.

## Instructions to Run

```console
$ cd src
$ pip3 install -r requirements.txt
$ python3 WordOccurrenceCounting.py
```

## Author

I, Filipe Pires, am the only author of the code and study conducted.
For further information, please read the [report](https://github.com/FilipePires98/LargeText-WordCount/blob/main/report/report.pdf) or contact me at filipesnetopires@ua.pt.

