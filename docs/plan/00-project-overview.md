# Time series anomaly detection: a foundations-first learning project

**Build domain-agnostic intuition for anomaly detection before specializing in vibration.** This project progresses from the simplest possible baseline (fixed threshold) through statistical methods, classical ML, and deep learning — all on the same datasets, so you see exactly what each layer of sophistication adds.

---

## Dataset selection: four datasets, increasing complexity

You'll use four datasets across the project. Each introduces a new challenge dimension.

### Dataset 1: Numenta Anomaly Benchmark (NAB) — univariate, labeled, diverse

**What it is.** 58 univariate time series from real-world sources: AWS CloudWatch server metrics, online ad click rates, NYC taxi demand, and machine temperature sensors. Each series has expert-labeled anomaly windows. This is the field's most widely used introductory benchmark.

**Why start here.** Univariate data strips away the complexity of multivariate correlation — you're detecting anomalies in a single stream. The diverse domains (server load, traffic, temperature) let you see how the same algorithm performs on different signal characteristics. The labeled anomaly windows enable proper precision/recall evaluation from day one.

**Download.** [https://github.com/numenta/NAB](https://github.com/numenta/NAB) — the `data/` folder contains all time series as CSVs; `labels/combined_windows.json` has the anomaly labels.

**Start with these specific files:**

- `realKnownCause/machine_temperature_system_failure.csv` — closest to industrial PdM, shows temperature drift before a machine failure
- `realAWSCloudwatch/ec2_cpu_utilization_ac20cd.csv` — server CPU with sudden spikes (point anomalies)
- `realTraffic/speed_7578.csv` — periodic traffic data with contextual anomalies (unusual patterns at normal-looking values)

### Dataset 2: SKAB (Skoltech Anomaly Benchmark) — multivariate, industrial, labeled

**What it is.** 35 experiments from a real water circulation testbed at Skoltech (Moscow) with a motor-driven pump, valves, and 8 sensor channels: two accelerometers (vibration), motor current, pressure, temperature, thermocouple, voltage, and flow rate. Each experiment contains one injected anomaly — valve closures, fluid leaks, rotor imbalance, cavitation. Every point is labeled as normal/anomalous.

**Why use this.** It's multivariate industrial sensor data from actual equipment — the closest publicly available analog to what industrial condition-monitoring sensors capture. You'll see how anomalies in one sensor propagate to others (closing a valve changes pressure, which changes flow, which changes vibration). The diversity of anomaly types (valve faults, imbalance, cavitation, leaks) maps directly to real maintenance scenarios. The fact that it includes both vibration and process variables bridges your foundations work into the vibration domain naturally.

**Download.** [https://github.com/waico/SKAB](https://github.com/waico/SKAB) or [https://www.kaggle.com/datasets/yuriykatser/skoltech-anomaly-benchmark-skab](https://www.kaggle.com/datasets/yuriykatser/skoltech-anomaly-benchmark-skab)

**Key files:**

- `anomaly-free/anomaly-free.csv` — your training baseline (normal operation)
- `valve1/` — valve closure experiments (sudden process change)
- `valve2/` — different valve closure experiments
- `other/` — rotor imbalance, fluid leaks, cavitation (gradual anomalies)

### Dataset 3: SWaT (Secure Water Treatment) — large-scale multivariate, cyber-physical

**What it is.** 946,719 timesteps from a real water treatment plant testbed at Singapore University of Technology and Design. 51 sensor channels across 6 process stages. 7 days of normal operation followed by 4 days of 36 different attack scenarios — valve manipulations, sensor spoofing, pump overrides. This is the standard benchmark for multivariate time series anomaly detection in industrial control systems.

**Why use this.** Scale (51 channels, nearly 1M rows) forces you to confront computational constraints. The attack scenarios create subtle multivariate anomalies where individual sensors may look normal but the cross-sensor relationships are violated — exactly the scenario where ML outperforms thresholds. The labeled attack segments enable rigorous evaluation.

**Download.** Request access at [https://itrust.sutd.edu.sg/itrust-labs_datasets/](https://itrust.sutd.edu.sg/itrust-labs_datasets/) (free for research, requires registration). The dataset is widely mirrored on Kaggle.

### Dataset 4: NASA SMAP and MSL — spacecraft telemetry, long sequences, gradual drift

**What it is.** Two labeled anomaly datasets from NASA's Jet Propulsion Laboratory. SMAP (Soil Moisture Active Passive) has 55 telemetry channels from a spacecraft; MSL (Mars Science Laboratory) has 27 channels from the Curiosity rover. Both contain long continuous recordings (~500K timesteps per entity) with labeled anomaly segments. Anomaly types include point anomalies, contextual anomalies, and gradual concept drift — the hardest category for static detectors.

**Why use this.** SMAP/MSL provides the right scale and complexity for deep learning methods. The sequences are long enough to train temporal models meaningfully. The gradual drift anomalies are exactly the regime where LSTM-based methods outperform static ones. The dataset was introduced in the landmark Hundman et al. (2018) KDD paper that established LSTM-AE for anomaly detection — implementing on SMAP/MSL lets you reproduce and directly compare against that benchmark.

**Download.** The dataset is available via the original NASA/KDD paper repository and Kaggle mirrors:
- GitHub: [https://github.com/khundman/telemanom](https://github.com/khundman/telemanom) (original paper repo, includes data loader)
- Kaggle: search "NASA SMAP MSL anomaly detection" for mirrored CSVs

**Key files:**
- `data/train/` — labeled normal telemetry per channel (e.g., `P-1.npy`, `S-1.npy`)
- `data/test/` — test telemetry with anomaly segments
- `labeled_anomalies.csv` — anomaly labels with channel ID, start/end timestamps, and anomaly class

---

## Project structure: seven modules, each building on the last

Each module introduces one technique, applies it to the appropriate dataset, and compares results against everything you've built before. The key discipline: **never move to the next module until you can articulate why the previous technique fails on specific examples.** That failure analysis is where the real learning happens.

### Module Overview & Status


| Module                                                                          | Status        | Notebook                                                                                                                         | Key Deliverable                                                                           |
| ------------------------------------------------------------------------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| [Module 1: Fixed thresholds](./01-module-1-fixed-thresholds.md)                 | **Completed** | `@notebooks/01-eda-fixed-thresholds.ipynb`                                                                                       | Baseline F1 on NAB; visceral understanding of false alarm tradeoff                        |
| [Module 2: Statistical baselines](./02-module-2-statistical-baselines.md)       | **Completed** | `@notebooks/02-1-statistical-z-score.ipynb`, `@notebooks/02-2-statistical-cusum.ipynb`, `@notebooks/02-3-statistical-ewma.ipynb` | z-score, EWMA, CUSUM compared; window size sensitivity analysis                           |
| [Module 3: Mahalanobis distance](./03-module-3-mahalanobis-distance.md)         | **Completed** | `@notebooks/03-skab-valve1-0.ipynb`, `@notebooks/03-skab-valve1-benchmark.ipynb`, `@notebooks/03-skab-valve2-benchmark.ipynb`, `@notebooks/03-skab-other-benchmark.ipynb`, `@notebooks/03-swat-benchmark.ipynb` | Univariate z-score vs. Mahalanobis benchmarked across SKAB valve1/valve2/other (16 experiments each) and SWaT (1.44M rows); SWaT showed decisive Mahalanobis advantage (F1 0.72 vs 0.42) due to z-score false alarm accumulation across 26 sensors |
| [Module 4: Isolation Forest](./04-module-4-isolation-forest.md)                 | **Completed** | `@notebooks/04-swat-isolation-forest.ipynb`, `@notebooks/04-skab-isolation-forest.ipynb` | IF vs. Mahalanobis on SWaT (51 features) and SKAB (valve1/valve2/other); Mahalanobis wins every benchmark; key finding: IF's `contamination` threshold lacks Mahalanobis's principled self-calibration; OCSVM omitted (O(n²) memory at 1.38M rows) |
| [Module 5: Autoencoder](./05-module-5-autoencoder.md)                           | **Completed** | `@notebooks/05-swat-autoencoder.ipynb`                                                                                         | Feedforward AE on SWaT; F1 0.7264 (p99.9 threshold), beats Mahalanobis (0.7240) and IF (0.5154); 225× normal/attack error separation; data-driven threshold calibration via validation percentile sweep |
| [Module 6: LSTM-Autoencoder](./06-module-6-lstm-autoencoder.md)                 | **Completed** | `@notebooks/06-swat-lstm-ae.ipynb`, `@notebooks/06-msl-lstm-ae.ipynb`                                                         | LSTM-AE on SWaT (F1 0.7533, p99.9, 40 epochs, stride=30) — best result in the project; +0.0269 over feedforward AE; 324× normal/attack separation. LSTM-AE on NASA MSL: per-channel training (27 channels), mean-MSE vs max-MSE score modes; micro F1 mean-MSE=0.1292 / max-MSE=0.1339 (negative result: underperforms FF-AE 0.2241 due to sparse training windows ~25–130 per channel) |
| [Module 7: USAD](./07-module-7-usad.md)                                         | **Not started** | `@notebooks/07-swat-usad.ipynb`, `@notebooks/07-msl-usad.ipynb`                                                              | USAD (two-decoder adversarial AE) on SWaT and NASA MSL; compare vs. LSTM-AE F1 0.7533 on SWaT; test whether adversarial amplification recovers MSL performance lost by LSTM-AE (micro F1 0.1292) |


---

## Implementation order and time estimates


| Module                      | Time    | Key deliverable                                                                           |
| --------------------------- | ------- | ----------------------------------------------------------------------------------------- |
| 1. Fixed thresholds         | 2–3 hrs | Baseline F1 on NAB; visceral understanding of false alarm tradeoff                        |
| 2. Statistical baselines    | 3–4 hrs | z-score, EWMA, CUSUM compared; window size sensitivity analysis                           |
| 3. Mahalanobis distance     | 3–4 hrs | First multivariate method on SKAB; comparison with univariate z-scores                    |
| 4. Isolation Forest         | 3–4 hrs | Classical ML vs. Mahalanobis comparison; hyperparameter sensitivity                       |
| 5. Autoencoder              | 4–5 hrs | Reconstruction error anomaly detection; latent space visualization; bottleneck experiment |
| 6. LSTM-Autoencoder         | 4–5 hrs | Temporal model vs. static model on slow drift; NASA SMAP/MSL benchmark                   |
| 7. USAD                     | 5–6 hrs | Two-decoder adversarial AE on SWaT and MSL; α/β sweep; comparison vs. LSTM-AE            |


**Total: ~25–30 hours over 3–4 weeks**

---

## What to read alongside each module

**Module 1–2:** Shewhart (1931) on control charts (conceptual); Page (1954) on CUSUM; Roberts (1959) on EWMA. You don't need to read the original papers — the JMP Statistical Knowledge Portal has excellent plain-language explanations at [https://www.jmp.com/en/statistics-knowledge-portal/quality-and-reliability-methods/control-charts/cusum-and-ewma-control-charts](https://www.jmp.com/en/statistics-knowledge-portal/quality-and-reliability-methods/control-charts/cusum-and-ewma-control-charts)

**Module 3:** The Hotelling (1947) T² statistic is covered in any multivariate statistics textbook. Focus on understanding why the covariance matrix matters — it's the mathematical encoding of "how do these sensors normally relate to each other?"

**Module 4:** Liu, Ting & Zhou (2008) "Isolation Forest" — [https://dl.acm.org/doi/10.1109/ICDM.2008.17](https://dl.acm.org/doi/10.1109/ICDM.2008.17). Short, elegant, and the core insight (anomalies are easier to isolate) is immediately useful intuition.

**Module 5:** No single paper — the use of autoencoders for anomaly detection emerged gradually. An & Cho (2015) "Variational Autoencoder Based Anomaly Detection Using Reconstruction Probability" is the closest to a landmark for reconstruction-based detection.

**Module 6:** Hundman et al. (2018) "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding" — KDD 2018. The NASA paper that introduced the SMAP/MSL datasets and demonstrated LSTM-based anomaly detection on real spacecraft telemetry. Practical and well-written. [https://arxiv.org/abs/1802.04431](https://arxiv.org/abs/1802.04431)

**Module 7:** Audibert, Michiardi, Guyard, Marti & Zuluaga (2020). "USAD: UnSupervised Anomaly Detection on Multivariate Time Series." *KDD 2020*. Read Section 3 (architecture) and Section 4 (experiments on SWaT and SMAP/MSL) before implementing. Reference implementation at [https://github.com/liuziyi/USAD](https://github.com/liuziyi/USAD) — use for architecture validation and hyperparameter defaults but implement from scratch.

**Overarching survey:** Chandola, Banerjee & Kumar (2009) "Anomaly Detection: A Survey" — [https://dl.acm.org/doi/10.1145/1541880.1541882](https://dl.acm.org/doi/10.1145/1541880.1541882). Read the taxonomy sections (point/contextual/collective anomalies) before starting Module 1 — it gives you the conceptual framework for understanding why different anomaly types require different detection approaches.

**Deep learning-specific survey:** Zamanzadeh Darban et al. (2024) "Deep Learning for Time Series Anomaly Detection: A Survey" — [https://dl.acm.org/doi/10.1145/3691338](https://dl.acm.org/doi/10.1145/3691338). Read after Module 6 to contextualize your autoencoder work within the broader landscape.

---

## References

### Datasets

1. **NAB (Numenta Anomaly Benchmark)** — [https://github.com/numenta/NAB](https://github.com/numenta/NAB)
2. **SKAB (Skoltech Anomaly Benchmark)** — [https://github.com/waico/SKAB](https://github.com/waico/SKAB) — also [https://www.kaggle.com/datasets/yuriykatser/skoltech-anomaly-benchmark-skab](https://www.kaggle.com/datasets/yuriykatser/skoltech-anomaly-benchmark-skab)
3. **SWaT (Secure Water Treatment)** — [https://itrust.sutd.edu.sg/itrust-labs_datasets/](https://itrust.sutd.edu.sg/itrust-labs_datasets/)
4. **NASA SMAP/MSL** — [https://github.com/khundman/telemanom](https://github.com/khundman/telemanom) (original KDD 2018 paper repo with data loader)
5. **TS-AD-Datasets (comprehensive list)** — [https://github.com/elisejiuqizhang/TS-AD-Datasets](https://github.com/elisejiuqizhang/TS-AD-Datasets)
6. **TSB-UAD Benchmark Suite** — [https://www.vldb.org/pvldb/vol15/p1697-paparrizos.pdf](https://www.vldb.org/pvldb/vol15/p1697-paparrizos.pdf)
7. **ts-anomaly-benchmark (Zamanzadeh)** — [https://github.com/zamanzadeh/ts-anomaly-benchmark](https://github.com/zamanzadeh/ts-anomaly-benchmark)

### Foundational surveys

1. Chandola, Banerjee & Kumar (2009). "Anomaly Detection: A Survey." *ACM Computing Surveys*, 41(3). — [https://dl.acm.org/doi/10.1145/1541880.1541882](https://dl.acm.org/doi/10.1145/1541880.1541882)
2. Zamanzadeh Darban, Webb, Pan, Aggarwal & Salehi (2024). "Deep Learning for Time Series Anomaly Detection: A Survey." *ACM Computing Surveys*. — [https://dl.acm.org/doi/10.1145/3691338](https://dl.acm.org/doi/10.1145/3691338)
3. Pang, Shen, Cao & van den Hengel (2021). "Deep Learning for Anomaly Detection: A Review." *ACM Computing Surveys*.
4. (2024). "Online model-based anomaly detection in multivariate time series: Taxonomy, survey, research challenges and future directions." *ScienceDirect*. — [https://www.sciencedirect.com/science/article/pii/S1574013725000632](https://www.sciencedirect.com/science/article/pii/S1574013725000632)

### Technique-defining papers

1. Liu, Ting & Zhou (2008). "Isolation Forest." *IEEE ICDM*. — [https://dl.acm.org/doi/10.1109/ICDM.2008.17](https://dl.acm.org/doi/10.1109/ICDM.2008.17)
2. Schölkopf, Platt, Shawe-Taylor, Smola & Williamson (2001). "Estimating the Support of a High-Dimensional Distribution." *Neural Computation*, 13(7). — [https://doi.org/10.1162/089976601750264965](https://doi.org/10.1162/089976601750264965)
4. Hochreiter & Schmidhuber (1997). "Long Short-Term Memory." *Neural Computation*, 9(8).
5. An & Cho (2015). "Variational Autoencoder Based Anomaly Detection Using Reconstruction Probability." *Special Lecture on IE*, 2(1).
6. Ruff et al. (2018). "Deep One-Class Classification." *ICML*.
7. Hawkins (1980). *Identification of Outliers*. Chapman & Hall.

### Statistical foundations

1. Page (1954). "Continuous Inspection Schemes." *Biometrika*. (CUSUM)
2. Roberts (1959). "Control Chart Tests Based on Geometric Moving Averages." *Technometrics*. (EWMA)
3. Hotelling (1947). "Multivariate Quality Control." (T² statistic)
4. JMP Statistical Knowledge Portal — CUSUM and EWMA Control Charts. — [https://www.jmp.com/en/statistics-knowledge-portal/quality-and-reliability-methods/control-charts/cusum-and-ewma-control-charts](https://www.jmp.com/en/statistics-knowledge-portal/quality-and-reliability-methods/control-charts/cusum-and-ewma-control-charts)

### Applied papers

1. Hundman et al. (2018). "Detecting Spacecraft Anomalies Using LSTMs and Nonparametric Dynamic Thresholding." *KDD 2018*. — [https://arxiv.org/abs/1802.04431](https://arxiv.org/abs/1802.04431)
2. Audibert, Michiardi, Guyard, Marti & Zuluaga (2020). "USAD: UnSupervised Anomaly Detection on Multivariate Time Series." *KDD 2020*. Reference implementation: [https://github.com/liuziyi/USAD](https://github.com/liuziyi/USAD)
2. Rüttinger et al. (2021). "Autoencoder-based Condition Monitoring and Anomaly Detection Method for Rotating Machines." *IEEE*. — [https://arxiv.org/pdf/2101.11539](https://arxiv.org/pdf/2101.11539)
3. Zope et al. (2019). "Anomaly Detection and Diagnosis in Manufacturing Systems." *PHM Conference*. — [https://papers.phmsociety.org/index.php/phmconf/article/view/815](https://papers.phmsociety.org/index.php/phmconf/article/view/815)
4. Katser & Kozitsin (2020). "Skoltech Anomaly Benchmark (SKAB)." *Kaggle*. — [https://www.kaggle.com/dsv/1693952](https://www.kaggle.com/dsv/1693952)

### Benchmark and evaluation papers

1. Paparrizos et al. (2022). "TSB-UAD: An End-to-End Benchmark Suite for Univariate Time-Series Anomaly Detection." *VLDB*, 15(8). — [https://www.vldb.org/pvldb/vol15/p1697-paparrizos.pdf](https://www.vldb.org/pvldb/vol15/p1697-paparrizos.pdf)
2. Wu & Keogh (2021). "Current Time Series Anomaly Detection Benchmarks are Flawed and are Creating the Illusion of Progress." *IEEE TKDE*. (Critical evaluation of benchmark datasets)
3. (2024). "TimeSeriesBench: An Industrial-Grade Benchmark for Time Series Anomaly Detection Models." — [https://arxiv.org/html/2402.10802v1](https://arxiv.org/html/2402.10802v1)

### Software and tools

1. Salesforce Merlion — Time series anomaly detection library with unified dataset loading. — [https://opensource.salesforce.com/Merlion/](https://opensource.salesforce.com/Merlion/)
2. scikit-learn — Novelty and Outlier Detection documentation. — [https://scikit-learn.org/stable/modules/outlier_detection.html](https://scikit-learn.org/stable/modules/outlier_detection.html)
3. PyOD — Python Outlier Detection library (30+ algorithms). — [https://github.com/yzhao062/pyod](https://github.com/yzhao062/pyod)
