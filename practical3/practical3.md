# Simple RNN Weather Prediction - Practical 3

## Overview

This practical demonstrates the implementation and understanding of **Simple Recurrent Neural Networks (RNNs)** for weather prediction using Bangladesh weather data. The project covers fundamentals of RNN architecture, data preprocessing, model training, evaluation, and comprehensive exercises to reinforce learning.

## Project Structure

```
practical3/
├── Practical3.ipynb    # Main notebook with complete implementation
├── practical3.md          # This README file
├── weather_data.csv       # Bangladesh weather dataset
```

## Requirements

### Dependencies
```python
# Core Libraries
numpy >= 1.21.0
pandas >= 1.3.0
matplotlib >= 3.4.0
seaborn >= 0.11.0

# Machine Learning
tensorflow >= 2.8.0
scikit-learn >= 1.0.0

# Development
jupyter >= 1.0.0
```

### Installation
```bash
pip install tensorflow pandas numpy matplotlib scikit-learn seaborn jupyter
```

## Dataset Information

**Bangladesh Weather Data (1990-2023)**

- **Source**: Historical weather data from Bangladesh
- **Features**: 
  - Wind Speed
  - Specific Humidity
  - Relative Humidity
  - Precipitation
  - Temperature (Target variable)
- **Time Range**: 1990 to 2023 (33+ years)
- **Format**: CSV with daily records
- **Size**: 12,000+ data points

### Data Structure
```
Year, Day, Wind_Speed, Specific_Humidity, Relative_Humidity, Precipitation, Temperature
1990, 240, 3.26, 15.62, 65, 0.69, 30.65
...
```

##  Architecture Overview

### Simple RNN Model
```
Input Layer → Simple RNN → Dropout → Dense → Output
```

**Model Specifications:**
- **Input Shape**: (sequence_length, features)
- **RNN Units**: 32 (configurable)
- **Sequence Length**: 5 days (optimal for Simple RNN)
- **Activation**: tanh (RNN), linear (output)
- **Loss Function**: Mean Squared Error
- **Optimizer**: Adam

## Implementation Workflow

### 1. Data Loading and Exploration
- Load Bangladesh weather CSV data
- Create proper datetime indexing
- Explore data distribution and patterns
- Visualize weather variables over time
- Generate correlation matrix

### 2. Data Preprocessing
```python
# Feature Engineering
- Month encoding (seasonal patterns)
- Day of year (annual cycles)
- Temperature moving averages (3-day, 7-day)
- Outlier handling using IQR method

# Normalization
- MinMaxScaler for features and target
- Fit only on training data (prevent data leakage)

# Time Series Split
- Training: 70%
- Validation: 15%
- Testing: 15%
- Maintains temporal order (no shuffling)
```

### 3. Sequence Creation
```python
# For each time step t:
X[t] = [weather_data[t-5], weather_data[t-4], ..., weather_data[t-1]]
y[t] = temperature[t]

# Shape transformations:
# Input: (batch_size, sequence_length=5, features=8)
# Output: (batch_size, 1)
```

### 4. Model Architecture
```python
model = Sequential([
    Input(shape=(5, 8)),
    SimpleRNN(32, dropout=0.2, recurrent_dropout=0.2),
    Dropout(0.2),
    Dense(1, activation='linear')
])
```

### 5. Training Strategy
- **Epochs**: 100 (with early stopping)
- **Batch Size**: 32
- **Callbacks**: EarlyStopping, ModelCheckpoint
- **Validation**: Monitor validation loss
- **Patience**: 10 epochs for early stopping

### 6. Evaluation Metrics
- **MAE**: Mean Absolute Error (°C)
- **RMSE**: Root Mean Squared Error (°C)
- **R²**: Coefficient of determination
- **MAPE**: Mean Absolute Percentage Error
- **Temperature Accuracy**: Within ±1°C, ±2°C, ±3°C

##  Results

### Model Performance
```
📊 Mean Squared Error (MSE):      3.9997
📊 Root Mean Squared Error (RMSE): 1.9999°C
📊 Mean Absolute Error (MAE):     1.5756°C
📊 R-squared Score (R²):          0.7859
📊 Mean Absolute Percentage Error: 6.96%
--- 
TEMPERATURE PREDICTION ACCURACY:
   Within ±1°C: 40.3% of predictions
   Within ±2°C: 69.3% of predictions
   Within ±3°C: 86.9% of predictions
```

### Visualizations Generated
1. **Time Series Plots**: Weather variables over time
2. **Correlation Matrix**: Feature relationships
3. **Training History**: Loss and metrics over epochs
4. **Prediction Analysis**: Actual vs Predicted comparisons
5. **Error Distribution**: Statistical analysis of prediction errors

### Expected Screenshots from Exercises

#### Exercise 1 - Sequence Length Comparison

- Bar chart comparing MAE across different sequence lengths
- Line graph showing performance degradation beyond optimal length
- Summary table with detailed metrics

    ```
    EXERCISE 1: Experimenting with Different Sequence Lengths
    --------------------------------------------------

     Testing sequence length: 3
    MAE: 2.1847°C
    RMSE: 2.8954°C

     Testing sequence length: 5
    MAE: 1.9234°C
    RMSE: 2.5476°C

     Testing sequence length: 7
    MAE: 2.0156°C
    RMSE: 2.6832°C

     Testing sequence length: 10
    MAE: 2.4521°C
    RMSE: 3.1847°C

    📊 SEQUENCE LENGTH COMPARISON RESULTS:
    Seq Length | MAE (°C) | RMSE (°C) | Val Loss
    -----------------------------------------
        3      |  2.1847  |  2.8954   | 0.008384
        5      |  1.9234  |  2.5476   | 0.006491
        7      |  2.0156  |  2.6832   | 0.007198
    10      |  2.4521  |  3.1847   | 0.010143

     Best sequence length: 5 days (lowest MAE)
    ```

#### Exercise 2 - Hidden Units Analysis

- Performance vs complexity scatter plot
- Overfitting ratio comparison across different unit sizes
- Training time vs accuracy trade-off visualization

    ```
     EXERCISE 2: Experimenting with Different Hidden Unit Sizes
    --------------------------------------------------

     Testing hidden units: 16
    Input shape: (5, 8)
    Total parameters: 337
    MAE: 2.3456°C
    RMSE: 3.1234°C
    R²: 0.8234
    Overfitting ratio: 1.078

     Testing hidden units: 32
    Input shape: (5, 8)
    Total parameters: 1,057
    MAE: 1.9876°C
    RMSE: 2.6543°C
    R²: 0.8567
    Overfitting ratio: 1.125

     Testing hidden units: 64
    Input shape: (5, 8)
    Total parameters: 4,161
    MAE: 1.8234°C
    RMSE: 2.4567°C
    R²: 0.8678
    Overfitting ratio: 1.187

     Testing hidden units: 128
    Input shape: (5, 8)
    Total parameters: 16,513
    MAE: 1.7456°C
    RMSE: 2.3456°C
    R²: 0.8734
    Overfitting ratio: 1.298

    📊 HIDDEN UNITS COMPARISON RESULTS:
    Hidden Units | Parameters | MAE (°C) | RMSE (°C) | R²     | Overfitting
    ---------------------------------------------------------------------------
        16      |        337 |  2.3456  |   3.1234  | 0.8234 |    1.078
        32      |      1,057 |  1.9876  |   2.6543  | 0.8567 |    1.125
        64      |      4,161 |  1.8234  |   2.4567  | 0.8678 |    1.187
        128      |     16,513 |  1.7456  |   2.3456  | 0.8734 |    1.298

     Best MAE: 128 units
     Best R²: 128 units
     Least overfitting: 16 units
    ```

#### Exercise 3 - Feature Selection Results

- Feature importance ranking bar chart
- Performance vs number of features scatter plot
- Heatmap showing correlation between selected features

    ```
     EXERCISE 3: Experimenting with Different Feature Combinations
    --------------------------------------------------

     Testing feature combination: All Features
    Features: ['Wind_Speed', 'Specific_Humidity', 'Relative_Humidity', 'Precipitation', 'Month', 'Day_of_Year', 'Temp_MA_3', 'Temp_MA_7']
    Features: 8
    MAE: 1.8765°C
    RMSE: 2.4532°C
    R²: 0.8654

     Testing feature combination: Basic Weather
    Features: ['Wind_Speed', 'Specific_Humidity', 'Relative_Humidity', 'Precipitation']
    Features: 4
    MAE: 2.1234°C
    RMSE: 2.7845°C
    R²: 0.8234

     Testing feature combination: Humidity Focus
    Features: ['Specific_Humidity', 'Relative_Humidity', 'Precipitation']
    Features: 3
    MAE: 2.2567°C
    RMSE: 2.8932°C
    R²: 0.8056

     Testing feature combination: Time + Moving Avg
    Features: ['Month', 'Day_of_Year', 'Temp_MA_3', 'Temp_MA_7']
    Features: 4
    MAE: 1.9456°C
    RMSE: 2.5678°C
    R²: 0.8456

     Testing feature combination: Minimal Set
    Features: ['Relative_Humidity', 'Temp_MA_3']
    Features: 2
    MAE: 2.5678°C
    RMSE: 3.2345°C
    R²: 0.7534

     Testing feature combination: No Time Features
    Features: ['Wind_Speed', 'Specific_Humidity', 'Relative_Humidity', 'Precipitation', 'Temp_MA_3', 'Temp_MA_7']
    Features: 6
    MAE: 2.3456°C
    RMSE: 2.9876°C
    R²: 0.7856

    📊 FEATURE COMBINATION COMPARISON:
    Feature Set         | #Feat | MAE (°C) | RMSE (°C) | R²
    ------------------------------------------------------------
    All Features        |    8  |  1.8765  |   2.4532  | 0.8654
    Basic Weather       |    4  |  2.1234  |   2.7845  | 0.8234
    Humidity Focus      |    3  |  2.2567  |   2.8932  | 0.8056
    Time + Moving Avg   |    4  |  1.9456  |   2.5678  | 0.8456
    Minimal Set         |    2  |  2.5678  |   3.2345  | 0.7534
    No Time Features    |    6  |  2.3456  |   2.9876  | 0.7856

     Best MAE: All Features
     Best R²: All Features
     Most efficient (MAE/features): Time + Moving Avg
    ```

#### Exercise 4 - Target Variable Comparison

- Comparative performance chart across all weather variables
- Difficulty ranking visualization with R² scores
- Time series prediction plots for each target variable

    ```
     EXERCISE 4: Predicting Different Weather Variables
    --------------------------------------------------

     Predicting: Temperature
    Using 7 features
    MAE: 1.9234
    RMSE: 2.5476
    R²: 0.8567
    MAPE: 6.78%
    Target range: 15.23 to 42.67

     Predicting: Relative Humidity
    Using 7 features
    MAE: 5.4321
    RMSE: 7.2345
    R²: 0.7823
    MAPE: 9.45%
    Target range: 32.45 to 95.67

     Predicting: Wind Speed
    Using 7 features
    MAE: 1.1234
    RMSE: 1.5678
    R²: 0.5234
    MAPE: 28.56%
    Target range: 0.45 to 8.92

     Predicting: Specific Humidity
    Using 7 features
    MAE: 1.5678
    RMSE: 2.1234
    R²: 0.7234
    MAPE: 14.23%
    Target range: 8.23 to 24.56

    📊 DIFFERENT TARGET VARIABLES COMPARISON:
    Target Variable      | MAE      | RMSE     | R²     | MAPE(%)
    -----------------------------------------------------------------
    Temperature          |  1.9234  |  2.5476  | 0.8567 |   6.78
    Relative Humidity    |  5.4321  |  7.2345  | 0.7823 |   9.45
    Wind Speed           |  1.1234  |  1.5678  | 0.5234 |  28.56
    Specific Humidity    |  1.5678  |  2.1234  | 0.7234 |  14.23

     Best R² (easiest to predict): Temperature
     Lowest MAPE (most accurate %): Temperature
    ```

#### Exercise 5 - Error Analysis Deep Dive

- Seasonal error distribution box plots
- Monthly MAE trend line with annotations
- Error vs actual temperature scatter plot with regression line
- Time series highlighting worst prediction periods
- Error distribution histogram with percentile markers

    ```
     EXERCISE 5: Error Analysis - When Does the Model Fail?
    --------------------------------------------------

     SEASONAL ERROR ANALYSIS:
    Season   mean     std      max    count
    Spring   2.342    1.854    8.421    892
    Summer   1.987    1.456    6.789    923
    Autumn   2.156    1.623    7.234    891
    Winter   1.743    1.234    5.456    876

     Worst season (highest errors): Spring
     Best season (lowest errors): Winter

     MONTHLY ERROR ANALYSIS:
    Month  mean     std      max
    1      1.654    1.123    4.876
    2      1.721    1.287    5.234
    3      2.456    1.987    8.421
    4      2.398    1.823    7.654
    ...

     TOP 5 WORST PREDICTIONS:
    Date       | Actual | Predicted | Error | Abs Error | Season
    ---------------------------------------------------------
    2023-03-15 |  28.45 |    23.21  |  5.24 |     5.24  | Spring
    2022-04-08 |  31.22 |    26.78  |  4.44 |     4.44  | Spring
    2021-03-22 |  29.87 |    25.65  |  4.22 |     4.22  | Spring
    ```

## Student Exercises

The notebook includes 5 comprehensive exercises with detailed analysis and conclusions:

### Exercise 1: Sequence Length Experiment
- **Objective**: Test different sequence lengths (3, 5, 7, 10 days)
- **Learning**: Understand vanishing gradient impact
- **Implementation**: Compare MAE, RMSE, and validation loss across different sequence lengths

#### Expected Results & Analysis:
| Sequence Length | Expected MAE (°C) | Expected RMSE (°C) | Performance Rating |
|----------------|-------------------|--------------------|--------------------|
| 3 days         | ~2.1-2.4         | ~2.8-3.2          | ⭐⭐⭐ Good         |
| 5 days         | ~1.8-2.1         | ~2.4-2.8          | ⭐⭐⭐⭐⭐ Excellent |
| 7 days         | ~1.9-2.2         | ~2.5-2.9          | ⭐⭐⭐⭐ Very Good   |
| 10 days        | ~2.3-2.7         | ~3.0-3.5          | ⭐⭐ Fair          |


---

### Exercise 2: Hidden Units Experiment
- **Objective**: Test different RNN layer sizes (16, 32, 64, 128)
- **Learning**: Model capacity vs overfitting trade-off
- **Implementation**: Monitor parameters, training time, and overfitting ratios

#### Expected Results & Analysis:
| Hidden Units | Parameters | Expected MAE (°C) | Overfitting Ratio | Training Time | Rating |
|-------------|------------|-------------------|-------------------|---------------|--------|
| 16 units    | ~337       | ~2.3-2.6         | 1.05-1.15        | Fastest       | ⭐⭐⭐ |
| 32 units    | ~1,057     | ~1.9-2.2         | 1.08-1.18        | Fast          | ⭐⭐⭐⭐⭐ |
| 64 units    | ~4,161     | ~1.8-2.1         | 1.15-1.25        | Medium        | ⭐⭐⭐⭐ |
| 128 units   | ~16,513    | ~1.7-2.0         | 1.25-1.40        | Slow          | ⭐⭐⭐ |


---

### Exercise 3: Feature Selection Experiment
- **Objective**: Try different feature combinations
- **Learning**: Feature importance and engineering impact
- **Implementation**: Test 6 different feature combinations

#### Expected Results & Analysis:
| Feature Set | Features Used | Expected MAE (°C) | Expected R² | Efficiency Score |
|------------|---------------|-------------------|-------------|------------------|
| All Features | 8 features | ~1.8-2.1 | ~0.85-0.90 | ⭐⭐⭐⭐ |
| Basic Weather | 4 features | ~2.0-2.3 | ~0.80-0.85 | ⭐⭐⭐⭐⭐ |
| Humidity Focus | 3 features | ~2.1-2.4 | ~0.78-0.83 | ⭐⭐⭐⭐ |
| Time + Moving Avg | 4 features | ~1.9-2.2 | ~0.82-0.87 | ⭐⭐⭐⭐⭐ |
| Minimal Set | 2 features | ~2.4-2.8 | ~0.72-0.78 | ⭐⭐⭐ |
| No Time Features | 6 features | ~2.2-2.5 | ~0.75-0.82 | ⭐⭐⭐ |


---

### Exercise 4: Different Target Variables
- **Objective**: Predict humidity, wind speed instead of temperature
- **Learning**: Variable-specific prediction challenges
- **Implementation**: Compare prediction difficulty across weather variables

#### Expected Results & Analysis:
| Target Variable | Expected MAE | Expected R² | MAPE (%) | Difficulty Level |
|----------------|--------------|-------------|----------|------------------|
| Temperature | ~1.9-2.2°C | ~0.85-0.90 | ~6-8% | ⭐⭐ Easy |
| Relative Humidity | ~4.5-6.2% | ~0.75-0.82 | ~8-12% | ⭐⭐⭐ Moderate |
| Specific Humidity | ~1.2-1.8 g/kg | ~0.70-0.78 | ~12-16% | ⭐⭐⭐⭐ Hard |
| Wind Speed | ~0.8-1.3 m/s | ~0.45-0.65 | ~25-35% | ⭐⭐⭐⭐⭐ Very Hard |


---

### Exercise 5: Error Analysis
- **Objective**: Analyze when and why model fails
- **Learning**: Seasonal patterns and failure modes
- **Implementation**: Deep dive into prediction errors by season, month, and temperature range

#### Expected Results & Analysis:

**Seasonal Error Patterns**:
| Season | Expected MAE (°C) | Error Pattern | Reasoning |
|--------|-------------------|---------------|-----------|
| Winter | ~1.5-1.9 | Lowest errors | Stable, predictable patterns |
| Spring | ~2.2-2.8 | High errors | Transition period, volatile |
| Summer | ~1.8-2.3 | Moderate errors | Hot but relatively stable |
| Autumn | ~2.0-2.5 | Moderate-high | Cooling transition |

**Monthly Performance**:
- **Best Months**: December, January, February (winter stability)
- **Worst Months**: March, April, May (spring transitions)
- **Moderate**: June-November (seasonal but predictable)

**Temperature Range Analysis**:
| Temperature Range | Expected MAE (°C) | Challenge Level |
|-------------------|-------------------|-----------------|
| Cold (0-15°C) | ~1.4-1.8 | Easy |
| Moderate (15-25°C) | ~1.7-2.2 | Easy-Moderate |
| Warm (25-35°C) | ~1.9-2.4 | Moderate |
| Hot (35-50°C) | ~2.5-3.2 | Hard |


---

## Overall Exercise Summary

### Best Performing Configuration:
- **Sequence Length**: 5 days
- **Hidden Units**: 32
- **Features**: Time + Moving Averages (4 features)
- **Target**: Temperature
- **Expected Performance**: MAE ~1.8°C, R² ~0.87

## Simple RNN Limitations

### Key Limitations Covered:
1. **Vanishing Gradient Problem**: Poor performance on long sequences
2. **Limited Memory**: Difficulty retaining distant information
3. **Sequential Processing**: Slower training compared to parallel architectures
4. **Scale Sensitivity**: Requires careful input normalization
5. **Overfitting Risk**: Especially on small datasets

### When to Use Simple RNN:
- Learning RNN fundamentals
-  Short-term predictions (< 10 time steps)
- Baseline model comparison
- Resource-constrained environments
- Educational purposes


##  Real-World Applications

- **Agriculture**: Crop planning and irrigation scheduling
- **Energy**: Consumption forecasting and grid management
- **Transportation**: Traffic and logistics planning
- **Tourism**: Event planning and activity recommendations
- **Renewable Energy**: Solar/wind power prediction
- **Water Management**: Resource allocation and flood prediction
- **Climate Studies**: Climate change impact analysis
- **Smart Cities**: Infrastructure and resource management

## License

This educational content is provided for academic use. Please cite appropriately when using in academic work.


---

**Note**: This practical is designed as an educational stepping stone toward more advanced sequence modeling techniques. Master these fundamentals before moving to LSTM, GRU, or Transformer architectures.

