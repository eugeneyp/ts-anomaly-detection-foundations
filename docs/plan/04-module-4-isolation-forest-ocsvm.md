## Module 4: Isolation Forest and One-Class SVM — classical ML

**Dataset:** SKAB (all experiments)

**What you'll build.** Two classical ML approaches that learn the boundary of "normal" from unlabeled data.

**Isolation Forest** builds random trees and scores anomalies by path length — anomalies are isolated quickly (short paths) because they're rare and different. **One-Class SVM** maps data to kernel space and finds a tight boundary around the normal data.

**Implementation.**

```python
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler().fit(train[features])
X_train = scaler.transform(train[features])

# Isolation Forest
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.02,  # expect ~2% anomalies
    random_state=42
)
iso_forest.fit(X_train)

# Score test data
X_test = scaler.transform(test[features])
test['iso_score'] = -iso_forest.score_samples(X_test)  # higher = more anomalous

# One-Class SVM
ocsvm = OneClassSVM(kernel='rbf', nu=0.02, gamma='scale')
ocsvm.fit(X_train)
test['ocsvm_score'] = -ocsvm.decision_function(X_test)
```

**Critical experiment.** Compare Isolation Forest, OCSVM, and Mahalanobis distance across all SKAB anomaly types. You'll likely find that Mahalanobis performs competitively on many anomaly types — confirming the Zope et al. (2019) finding that simple statistical methods match ML in many industrial scenarios. Isolation Forest will likely outperform on the cavitation and rotor imbalance experiments where the anomaly manifests as a nonlinear multivariate shift that Mahalanobis (which assumes Gaussian/linear correlations) misses.

**Hyperparameter sensitivity analysis.** Sweep `contamination` from 0.005 to 0.1 and plot precision/recall curves. Sweep OCSVM's `nu` and `gamma`. Document how sensitive each method is — this is a key production consideration. Isolation Forest is robust; OCSVM is notoriously sensitive.

**What you'll learn.** Isolation Forest's key insight is that anomalies don't need to be far from normal — they just need to be *different* in a way that makes them easy to separate. This is a fundamentally different philosophy from distance-based methods (Mahalanobis, OCSVM). You'll also see that OCSVM's training time grows quadratically with data size — a practical showstopper at production scale.

**Production connection.** Isolation Forest is the most deployed classical ML anomaly detector in production PdM, largely because it runs on edge hardware (ESP32 microcontrollers) and has minimal hyperparameter sensitivity. STMicroelectronics' NanoEdge AI Studio includes Isolation Forest as a default model.
