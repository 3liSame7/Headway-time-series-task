**Task: Time Series Forecasting Endpoint**

### Overview:
The task involves building a single-step time series forecasting model using traditional machine learning models with feature extraction. The project is divided between the **Data Team** and the **Data Testing Team**, each with specific deliverables and deadlines. The data is split into training and testing sets, and the project should be containerized (using Docker). Additionally, testing the inference speed of the API is required to evaluate its latency.

#### 1. Data Team Task
- **Deadline**: Saturday, 12th October
- **Deliverables**:
  - Upload the solution to a drive and share the URL with me and data testing members personally via Teams.
  - Provide a **CSV file** containing:
    - Dataset ID.
    - The corresponding number of values needed to be passed as input to make an inference (i.e., the length of the `values` list in the request body).
  - Include a **README file** that describes how to run the Django or Flask application for testing.
  - The project should be **containerized** (using Docker).
  - Ensure that the model is trained on data from the **train splits folder**. The training process should involve **feature extraction** techniques for traditional machine learning models which was explained in the lecture.
  - Structure the input request to accept a `dataset_id` and a list of time-series `values`, where each value includes a timestamp and corresponding data point.
  
    **Example Request Body**:
    ```json
    {
      "dataset_id": xxx,
      "values": [
        {"time": "YYYY-MM-DDTHH:mm:ss", "value": xxx},
        ...
      ]
    }
    ```
    **Example Response Body**:
    ```json
    {
      "prediction": xxx
    }
    ```

#### 2. Data Testing Team Task
- **Deadline**: Saturday, 19th October
- **Deliverables**:
  - Upload the testing and evaluation scripts to a drive or GitHub.
  - Share with me the results personally via Teams / GitHub invitation.
  - Test the models on data from the **test splits folder**.
  - Prepare an Excel sheet with the **Mean Squared Error (MSE)** of each data team member's model across all datasets, and the **average MSE**.
  - Additionally, measure and evaluate the **inference speed** (latency) of the API.