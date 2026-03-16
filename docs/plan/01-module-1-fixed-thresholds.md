## Module 1: Fixed thresholds and visual inspection

**Dataset:** NAB `machine_temperature_system_failure.csv`

**What you'll build.** Load the time series, plot it, and manually identify anomalies by eye. Then implement the simplest possible detection: a fixed upper and lower threshold. This is what ISO 10816 severity charts do — alarm if the value exceeds a fixed limit.

**Implementation.**

```python
import pandas as pd
import numpy as np

# Load and plot
df = pd.read_csv('machine_temperature_system_failure.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Fixed threshold: alarm if temperature > X
threshold_high = 90  # picked by looking at the data
df['anomaly_fixed'] = (df['value'] > threshold_high).astype(int)
```

**What to measure.** Precision (what fraction of alarms are real anomalies), recall (what fraction of real anomalies you caught), and F1. Plot the time series with your threshold line and the ground truth anomaly windows. Count false alarms in the normal region.

**What you'll learn.** Three things will become immediately obvious. First, picking the right threshold is hard — too low means constant false alarms, too high means missed detections. Second, a single fixed threshold can't handle drift, seasonality, or changing baselines. Third, point anomalies (sudden spikes) are easy to catch; contextual anomalies (unusual patterns at normal amplitude) are invisible to thresholds.

**Production connection.** This is what Emerson's AMS ships with: pre-programmed limits per equipment type. It catches severe faults reliably but generates alert fatigue on anything subtle.
