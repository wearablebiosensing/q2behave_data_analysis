# Signal Processing Pipeline Diagrams

Here is the pipeline diagram mapping out each processing step found in your signal extraction tools.

## Pipeline Diagram (`plot_signal_stages.py`)
This flowchart outlines the logic utilized in the `plot_signal_stages.py` script for calculating and filtering accelerometer and gyroscope data.

```mermaid
graph TD
    A[Load Data CSV] --> B{"Check 'DataTS' Column"}
    B -- Exists --> C["Calculate Median diff (dt)"]
    C --> C1{"Valid dt?"}
    C1 -- Yes --> C2["Calculate Fs = 1.0 / dt"]
    C1 -- No --> D["Default Fs = 20.0 Hz"]
    B -- Not Found --> D
    C2 --> E["Data Prep: Calculate Time (s) or Samples"]
    D --> E
    E --> F["Process Sub-signals (Accel X/Y/Z, Gyro X/Y/Z)"]
    F --> G["Plot Raw Data Traces"]
    F --> H{"Fs > 10.0 Hz & len >= 5?"}
    H -- Yes --> I["Detrend Signal"]
    I --> J["Butterworth Bandpass Filter\n(1 to 5 Hz, Order 5)"]
    J --> K["Plot Bandpass Trace"]
    J --> L["Moving Average Filter\n(Window = 5)"]
    L --> M["Plot Moving Average Trace"]
    
    H -- No (Skip BP) --> N["Detrend Signal Only"]
    N --> O["Moving Average Filter\n(Window = 5)"]
    
    L --> P["PSD Calculation (Welch's Method)"]
    O --> P
    P --> Q["Plot PSD (Power Density vs Frequency)"]
    
    G --> R["Compile Subplots & Show Plotly Dashboard"]
    K --> R
    M --> R
    Q --> R
```

> [!NOTE]
> The same general processing flow applies to the signal feature extractions developed in `jupyter_notebooks/plot_psd_features.ipynb`, which reads the input data and subsequently performs validation, detrending, filtering, and Welch's PSD calculation to construct the signal features.
