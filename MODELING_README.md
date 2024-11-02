# EDO Case Study: Ad Exposure and Website Visit Analysis

This project seeks to analyze ad exposure patterns and website visit behavior,
with the aim of building a machine learning model to predict ad exposure
counts. The dataset includes records of ad exposures and website visits, along
with demographic data such as income level and dog ownership. The analysis
explores whether these features can provide enough signal for a predictive
model and examines monthly and annual patterns across demographic subgroups.

## Project Overview

The project is divided into three main parts:

1. **Data Loading and Inspection**: Load datasets related to ad exposures,
   website visits, and demographic information to prepare for analysis and
   visualization.
2. **Exploratory Data Analysis (EDA)**: Conduct visual analysis to observe
   trends in ad exposure and website visits across various subgroups (e.g.,
   income bins, dog ownership). This includes plotting distributions annually
   and monthly for each demographic group.
3. **Model Training**: Attempt to model ad exposure counts based on demographic
   features. The process involves preprocessing the data, conducting a grid
   search over multiple models, and evaluating the model’s performance on a
   test set.

## Key Findings

- **Ad Exposure and Visit Stability**: Both ad exposures and website visits
  remain consistent across months, suggesting limited seasonal variation.
- **Demographic Insights**: Income level is the primary feature that exhibits
  a correlation with ad exposure and visit counts. Other features, such as dog
  ownership and specific demographic codes, show limited predictive value.
- **Modeling Limitations**: Models trained on this dataset tend to underfit,
  largely due to the limited predictive signal available from the features.
  Income level, though somewhat predictive, offers only ordinal values,
  leading to binned predictions in continuous regression models.

## Installation

To set up the environment required to run this analysis, you can use the
provided `environment.yaml` file. From the project’s root directory, execute
the following command to create the conda environment:

```shell
conda env create --file environment.yaml
```

Once the environment is set up, activate it using:

```shell
conda activate edo-case-study
```

## Usage

This project is structured as a Jupyter notebook. To run the analysis:

1. Open the Jupyter notebook.
2. Execute each cell in sequence to load the data, visualize trends, and train
   models.
3. Explore the summary findings provided at the end of each section for
   insights.
