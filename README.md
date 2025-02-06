# Wind Turbine Data Processing Pipeline

## Overview
This project is a **scalable and testable data processing pipeline** for analyzing wind turbine power output. The pipeline follows the **Medallion Architecture** with Bronze, Silver, and Gold data layers.

### **Key Features**
**Data Ingestion:** Load CSV files containing turbine data.

**Data Cleaning:** Handle missing values using interpolation.

**Summary Statistics:** Compute min, max, and mean power output per turbine.

**Anomaly Detection:** Identify turbines with output outside **2 standard deviations**.

**Database Storage:** Store processed data in **SQLite**.

**Visualization:** Generate time-series anomaly detection plots.

**Testing:** Includes **unit tests** using `pytest`.

---

## 📁 **Project Structure**
```
📂 WindTurbinePipeline
│── 📂 data
│   ├── 📂 raw        # Raw CSV files (Bronze Layer)
│   ├── 📂 bronze     # Combined raw data
│   ├── 📂 silver     # Cleaned data
│   ├── 📂 gold       # Summary stats & anomalies
│── 📂 output         # SQLite database & visualizations
│── 📂 tests          # Unit tests
│── pipeline.py       # Core data processing pipeline
│── test.py           # Automated tests
│── main.py           # Pipeline execution script
│── README.md         # Project documentation
```

---

## **Setup & Installation**
### **Prerequisites**
- **Python 3.10+**
- Required libraries:
  ```sh
  pip install pandas numpy dask matplotlib sqlite3 pytest
  ```

### **Running the Pipeline**
To execute the pipeline, run:
```sh
python main.py
```

---

## **Accessing Data in SQLite**
### ** Using Python**
```python
import sqlite3
import pandas as pd
conn = sqlite3.connect("output/turbine_data.db")
df = pd.read_sql_query("SELECT * FROM cleaned_data LIMIT 10;", conn)
print(df)
conn.close()
```

---

## **Testing**
Run unit tests using:
```sh
pytest test.py
```

---

## **Scalability**
- The pipeline **automatically processes new data** added to the `Data/raw` folder.
- Designed for **seamless daily updates** without requiring modifications to the code.
