# House Price Monitoring Pipeline - README
Hello again!

Here is the last week's code overview.

Below is an overview of the key scripts and their functionalities for creating monitoring pipeline for our use case.

---

## Scripts Overview

### 00. `create_inference_data.py`
This script generates synthetic data for model inference and introduces intentional data drift to see drift monitoring.

#### Key Features:
- **Data Loading**:
  - Loads existing train and test datasets from the catalog.
  - Uses a Random Forest model to identify the most important features influencing house prices.
- **Synthetic Data Generation**:
  - Creates two types of synthetic datasets:
    - **Normal dataset**: Follows the same distribution as the training data.
    - **Skewed dataset**: Introduces data drift by modifying features like `OverallQual` and `GrLivArea`.
  - Generates unique IDs and adds UTC timestamps to the data.
- **Data Storage and Feature Updates**:
  - Saves synthetic datasets to the tables:
    - `inference_set_normal`
    - `inference_set_skewed`
  - Updates the feature store table `house_features` with new synthetic records.
  - Triggers a pipeline update to refresh the online feature store.

---

### 01. `send_request_to_endpoint.py`
This script sends requests to the model serving endpoint using both the test and skewed datasets to simulate data drift.

#### Key Features:
- Sends requests to the endpoint in two phases:
  1. **Test Set**: Simulates normal conditions.
  2. **Skewed Data**: Mimics data drift over a period of 20-30 minutes.

---

### 02. `refresh_monitor.py`
Processes model serving payloads to monitor the quality and consistency of predictions.

#### Key Features:
- **Data Processing**:
  - Defines schemas for request and response data.
  - Parses JSON request/response payloads and extracts relevant information.
- **Enrichment**:
  - Joins with test and inference datasets to obtain actual sale prices.
  - Adds house features from the feature store.
  - Converts data types as needed for downstream processing.
- **Output**:
  - Writes enriched monitoring data to a `model_monitoring` table.
  - Triggers a refresh of the quality monitors.

---

### 03. `lakehouse_monitoring.py`
Creates and configures lakehouse monitoring using the Databricks workspace client.

---

### 04. `create_alert.py`
Sets up alerting mechanisms for the pipeline using the Databricks workspace client.

We use refresh_monitor.py script to create a separate job.