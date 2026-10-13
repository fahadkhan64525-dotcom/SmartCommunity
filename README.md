# 🚑 Smart Community Health Monitoring System

An AI-powered Streamlit application designed to support **early warning
and community-level monitoring of potential water-borne disease risks**.
Health workers can enter symptom counts and environmental conditions to
receive a model-generated risk category and review recent symptom
trends.

> **Important:** This project is an educational decision-support
> prototype, not a medical diagnostic tool. Predictions should be
> reviewed by qualified public-health professionals and must not replace
> official reporting, testing, or emergency procedures.

## ✨ Features

-   **Risk prediction:** Predicts a risk category from diarrhea, fever,
    vomiting, rainfall, and temperature inputs.
-   **Simple health-worker interface:** Uses Streamlit sidebar inputs
    for quick data entry.
-   **Risk status indicators:** Displays low, medium, or high risk using
    visual status messages.
-   **High-risk alert guidance:** Shows a precautionary message when the
    model predicts high risk.
-   **Historical trend visualization:** Plots recent diarrhea, fever,
    and vomiting counts from the CSV dataset.
-   **AI explanation panel:** Provides a short explanation of the
    factors considered by the application.

## 🧰 Tech Stack

-   Python
-   Streamlit
-   Pandas
-   Pickle (loading the saved model)
-   Machine Learning model (the specific algorithm depends on
    `train_model.py`)

## 📁 Project Structure

``` text
.
├── app.py
├── train_model.py
├── model.pkl
├── health_data.csv
├── requirement.txt
└── README.md
```

## ⚙️ Getting Started

### 1. Clone the repository

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

Replace the placeholders with your GitHub repository URL and folder
name.

### 2. Create and activate a virtual environment (recommended)

**Windows**

``` bash
python -m venv venv
venv\\Scripts\\activate
```

**macOS / Linux**

``` bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install -r requirement.txt
```

Make sure `requirement.txt` contains all required packages, including
`streamlit` and `pandas`, plus any packages needed to load and run the
trained model.

### 4. Run the application

``` bash
streamlit run app.py
```

Streamlit will provide a local URL to open the application in your
browser.

## 🧪 How to Use

1.  Start the app using the command above.
2.  Enter the number of diarrhea, fever, and vomiting cases in the
    sidebar.
3.  Enter rainfall in millimeters and temperature in degrees Celsius.
4.  Select **Predict Risk**.
5.  Review the predicted risk category and any displayed high-risk
    precautionary message.
6.  Review the historical symptom trend chart.

## 📊 Input Features

  Feature         Description                         Unit
  --------------- ----------------------------------- ----------------------
  `diarrhea`      Number of reported diarrhea cases   Cases
  `fever`         Number of reported fever cases      Cases
  `vomiting`      Number of reported vomiting cases   Cases
  `rainfall`      Recorded rainfall                   Millimeters (mm)
  `temperature`   Recorded temperature                Degrees Celsius (°C)

The model expects these five features in the same order used during
training. Keep the training and inference feature order consistent.

## 🧠 Risk Categories

  Model output              Application message
  ------------------------- ------------------------------------
  `0`                       🟢 Low risk --- Situation normal
  `1`                       🟡 Medium risk --- Monitor closely
  Other output (as coded)   🔴 High risk --- Alert issued

The meaning and reliability of each category depend on the model's
training labels, dataset quality, and validation results. Confirm that
the labels in `train_model.py` match the categories displayed by the
app.

## 📈 Historical Trends

The application reads `health_data.csv` and displays the latest 30 rows
for the `diarrhea`, `fever`, and `vomiting` columns.

Ensure that `health_data.csv` exists in the application's working
directory and includes these exact column names:

``` text
diarrhea, fever, vomiting
```

## 🔒 Data, Safety, and Responsible Use

-   Use appropriately authorized and privacy-protective health data.
-   Do not include personally identifiable patient information in a
    public repository.
-   Validate the model on representative data before considering any
    real-world use.
-   Review false positives and false negatives with qualified
    public-health professionals.
-   Treat alerts as prompts for human review, not confirmed outbreaks or
    diagnoses.
-   The precautionary advice shown by the prototype is general
    information and should be checked against local public-health
    guidance.

## 🛠️ Troubleshooting

**`FileNotFoundError: model.pkl` or `health_data.csv`** - Keep the files
in the same directory as `app.py`, or update the file paths in the code.

**Dependency or model-loading errors** - Install the dependencies listed
in `requirement.txt`. - Ensure the environment used to load `model.pkl`
is compatible with the environment and library versions used to train
it.

**Missing CSV columns** - Check that the dataset contains `diarrhea`,
`fever`, and `vomiting` with the exact spelling and capitalization.

**Prediction errors** - Confirm that `model.pkl` is a trained model
expecting the five input features in the order used by `app.py`.

## 🚀 Future Improvements

-   Add model evaluation metrics such as precision, recall, F1-score,
    and a confusion matrix.
-   Display prediction confidence only if it is supported and correctly
    calibrated.
-   Add date-based filters and more historical trend charts.
-   Add secure authentication and role-based access for authorized
    health workers.
-   Connect validated, reliable data sources and provide data-quality
    checks.
-   Add location-level monitoring only with appropriate privacy
    safeguards and authorization.
-   Include model versioning, validation documentation, and a clear
    human-review workflow.

## 👨‍💻 Author

**Shah Fahad**\
AI/ML Engineering Project

------------------------------------------------------------------------

If you use or extend this project, document the dataset source,
target-label definitions, model evaluation, and limitations so that
users can understand what the predictions do---and do not---mean.
