import pandas as pd
import numpy as np
from scipy.signal import butter, lfilter, welch
import os
import tkinter as tk
from tkinter import filedialog
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Force Plotly to open a web browser instead of printing to terminal
pio.renderers.default = "browser"

# --- Helper Signal Processing Functions ---

def moving_average(X, window_size):
    """Simple moving average filter."""
    X_new = []
    for i in range(len(X)):
        end_idx = min(i + window_size, len(X))
        X_new.append(np.mean(X[i:end_idx]))
    return np.array(X_new)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Bandpass filter."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def get_psd_for_plotly(data_array, Fs):
    """Calculates PSD for the Plotly trace just like the original stages."""
    if len(data_array) < 5 or Fs <= 0:
        return np.array([]), np.array([])
    from scipy.signal import detrend
    try:
        # Bandpass filter may fail if Fs isn't sufficient for the critical frequencies (1 to 5 Hz)
        if Fs > 10.0:
            detrended_data = detrend(data_array)
            bp_data = butter_bandpass_filter(detrended_data, 1, 5, Fs, order=5)
        else:
            # Skip bandpass if Nyquist is too low
            bp_data = detrend(data_array)
            
        ma_data = moving_average(bp_data, 5)
        nperseg = min(len(ma_data), 256)
        if nperseg > 0:
            freqs, psd = welch(ma_data, fs=Fs, nperseg=nperseg)
            return freqs, np.abs(psd)
    except Exception as e:
        print(f"PSD Error: {e}")
    return np.array([]), np.array([])


def add_signal_stages(fig, data, x_axis, Fs, trace_name, base_color, row, col):
    """Helper to plot raw, bandpass, and moving avg data into a single Plotly subplot."""
    # 1. Raw Data (Solid Base Color identical to PSD traces)
    fig.add_trace(go.Scatter(x=x_axis, y=data, mode='lines', opacity=1.0, line=dict(color=base_color, width=1.5), name=f'Raw {trace_name}'), row=row, col=col)
    
    if len(data) >= 5 and Fs > 10.0:
        from scipy.signal import detrend
        # 2. Bandpass
        try:
            detrended_data = detrend(data)
            bp_data = butter_bandpass_filter(detrended_data, 1, 5, Fs, order=5)
            # Original red color
            fig.add_trace(go.Scatter(x=x_axis, y=bp_data, mode='lines', opacity=0.8, line=dict(color='#d62728', width=2), name=f'BP {trace_name}'), row=row, col=col)
            
            # 3. Moving Average
            ma_data = moving_average(bp_data, 5)
            # Original green color
            fig.add_trace(go.Scatter(x=x_axis, y=ma_data, mode='lines', line=dict(color='#2ca02c', width=3), name=f'MA {trace_name}'), row=row, col=col)
        except Exception as e:
            print(f"Filter error for {trace_name}: {e}")

def raw_plots_plotly(df, filename, Fs):
    # Calculate time in seconds
    if 'DataTS' in df.columns:
        df['DataTS_dt'] = pd.to_datetime(df['DataTS'])
        df['Time_s'] = (df['DataTS_dt'] - df['DataTS_dt'].iloc[0]).dt.total_seconds()
        x_axis = df['Time_s']
        x_title = 'Time (s)'
    else:
        x_axis = np.arange(len(df)) / Fs
        x_title = 'Time (s)'
        
    # Get proper column names (after stripping)
    acc_x = 'Accel_X' if 'Accel_X' in df.columns else ' Accel_X'
    acc_y = 'Accel_Y' if 'Accel_Y' in df.columns else ' Accel_Y'
    acc_z = 'Accel_Z' if 'Accel_Z' in df.columns else ' Accel_Z'
    gyro_x = 'Gyro_X' if 'Gyro_X' in df.columns else ' Gyro_X'
    gyro_y = 'Gyro_Y' if 'Gyro_Y' in df.columns else ' Gyro_Y'
    gyro_z = 'Gyro_Z' if 'Gyro_Z' in df.columns else ' Gyro_Z'
    beh_code = 'BehaviorCode' if 'BehaviorCode' in df.columns else ' BehaviorCode'

    fig = make_subplots(rows=4, cols=3,
                        subplot_titles=('Accelerometer X', 'Accelerometer Y', 'Accelerometer Z',
                                        'Gyroscope X', 'Gyroscope Y', 'Gyroscope Z',
                                        'PSD Acc X', 'PSD Acc Y', 'PSD Acc Z',
                                        'PSD Gyro X', 'PSD Gyro Y', 'PSD Gyro Z'),
                        vertical_spacing=0.1) 
    
    # Original specific colors from the requested code:
    # X -> #1e81b0
    # Behavior Code -> #21130d
    # Y -> #e28743
    # Z -> #063970
    
    # ================= ROW 1: Accumulator Data =================
    if acc_x in df.columns:
        add_signal_stages(fig, df[acc_x].values, x_axis, Fs, 'Acc X', '#1e81b0', 1, 1)
    if beh_code in df.columns:
        fig.add_trace(go.Scatter(x=x_axis, y=df[beh_code], mode='lines', name='Behavior Code X', line=dict(color="#21130d")), row=1, col=1)
        
    if acc_y in df.columns:
        add_signal_stages(fig, df[acc_y].values, x_axis, Fs, 'Acc Y', '#e28743', 1, 2)
    if beh_code in df.columns:
        fig.add_trace(go.Scatter(x=x_axis, y=df[beh_code], mode='lines', name='Behavior Code Y', line=dict(color="#21130d")), row=1, col=2)

    if acc_z in df.columns:
        add_signal_stages(fig, df[acc_z].values, x_axis, Fs, 'Acc Z', '#063970', 1, 3)
    if beh_code in df.columns:
        fig.add_trace(go.Scatter(x=x_axis, y=df[beh_code], mode='lines', name='Behavior Code Z', line=dict(color="#21130d")), row=1, col=3)

    # ================= ROW 2: Gyroscope Data =================
    if gyro_x in df.columns:
        add_signal_stages(fig, df[gyro_x].values, x_axis, Fs, 'Gyro X', '#1e81b0', 2, 1)
    if beh_code in df.columns:
        fig.add_trace(go.Scatter(x=x_axis, y=df[beh_code], mode='lines', name='Behavior Code GyroX', line=dict(color="#21130d") ), row=2, col=1)

    if gyro_y in df.columns:
        add_signal_stages(fig, df[gyro_y].values, x_axis, Fs, 'Gyro Y', '#e28743', 2, 2)
    if beh_code in df.columns:
        fig.add_trace(go.Scatter(x=x_axis, y=df[beh_code], mode='lines', name='Behavior Code GyroY', line=dict(color="#21130d") ), row=2, col=2)

    if gyro_z in df.columns:
        add_signal_stages(fig, df[gyro_z].values, x_axis, Fs, 'Gyro Z', '#063970', 2, 3)
    if beh_code in df.columns:
        fig.add_trace(go.Scatter(x=x_axis, y=df[beh_code], mode='lines', name='Behavior Code GyroZ', line=dict(color="#21130d") ), row=2, col=3)
        
    # ================= ROW 3: PSD Accelerometer Data =================
    if acc_x in df.columns:
        f, p = get_psd_for_plotly(df[acc_x].values, Fs)
        fig.add_trace(go.Scatter(x=f, y=p, mode='lines', name='PSD Acc X', line=dict(color="#1e81b0")), row=3, col=1)
    if acc_y in df.columns:
        f, p = get_psd_for_plotly(df[acc_y].values, Fs)
        fig.add_trace(go.Scatter(x=f, y=p, mode='lines', name='PSD Acc Y', line=dict(color="#e28743")), row=3, col=2)
    if acc_z in df.columns:
        f, p = get_psd_for_plotly(df[acc_z].values, Fs)
        fig.add_trace(go.Scatter(x=f, y=p, mode='lines', name='PSD Acc Z', line=dict(color="#063970")), row=3, col=3)

    # ================= ROW 4: PSD Gyroscope Data =================
    if gyro_x in df.columns:
        f, p = get_psd_for_plotly(df[gyro_x].values, Fs)
        fig.add_trace(go.Scatter(x=f, y=p, mode='lines', name='PSD Gyro X', line=dict(color="#1e81b0")), row=4, col=1)
    if gyro_y in df.columns:
        f, p = get_psd_for_plotly(df[gyro_y].values, Fs)
        fig.add_trace(go.Scatter(x=f, y=p, mode='lines', name='PSD Gyro Y', line=dict(color="#e28743")), row=4, col=2)
    if gyro_z in df.columns:
        f, p = get_psd_for_plotly(df[gyro_z].values, Fs)
        fig.add_trace(go.Scatter(x=f, y=p, mode='lines', name='PSD Gyro Z', line=dict(color="#063970")), row=4, col=3)
    
    # Apply Template
    fig.update_layout(template="simple_white")

    # Update Axes for Signal subplots (Rows 1 & 2)
    for i in range(1, 4):
        fig.update_xaxes(title_text=x_title, row=1, col=i)
        fig.update_xaxes(title_text=x_title, row=2, col=i)
        # Update Axes for PSD subplots (Rows 3 & 4)
        fig.update_xaxes(title_text="Frequency (Hz)", row=3, col=i)
        fig.update_xaxes(title_text="Frequency (Hz)", row=4, col=i)
        
    fig.update_yaxes(title_text="Acc Amp", row=1, col=1)
    fig.update_yaxes(title_text="Acc Amp", row=1, col=2)
    fig.update_yaxes(title_text="Acc Amp", row=1, col=3)
    
    fig.update_yaxes(title_text="Gyro Amp", row=2, col=1)
    fig.update_yaxes(title_text="Gyro Amp", row=2, col=2)
    fig.update_yaxes(title_text="Gyro Amp", row=2, col=3)
    
    for i in range(1, 4):
        fig.update_yaxes(title_text="Power Density", type='log', exponentformat='power', row=3, col=i)
        fig.update_yaxes(title_text="Power Density", type='log', exponentformat='power', row=4, col=i)

    # Update layout with centered title including the filename
    fig.update_layout(height=1200, width=1200, 
                      margin=dict(t=150),
                      title=dict(
                          text=f"Filtered Signals & PSD Features<br><b>[{filename}]</b>",
                          x=0.5,
                          y=0.98,
                          yref='container',
                          xanchor='center',
                          yanchor='top'
                      ),
                      legend=dict(
                          orientation="h",
                          yanchor="bottom",
                          y=1.03,
                          xanchor="center",
                          x=0.5
                      ),
                      title_font=dict(size=20), font=dict(size=10))
    return fig

def get_file_path():
    """Opens a Tkinter file dialog to select the data CSV."""
    root = tk.Tk()
    root.withdraw() # Hide the main window
    
    # Force the window to be at the front
    root.call('wm', 'attributes', '.', '-topmost', True)
    
    file_path = filedialog.askopenfilename(
        title="Select Data CSV File",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    
    # Destory the root once the file is chosen
    root.destroy()
    return file_path

if __name__ == "__main__":
    
    print("Please select a data CSV file from the popup window...")
    data_path = get_file_path()
    
    if data_path and os.path.exists(data_path):
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        
        # Clean up column names by stripping leading/trailing spaces
        df.columns = df.columns.str.strip()
        
        # Calculate Sampling frequency of the watch from DataTS
        if 'DataTS' in df.columns:
            try:
                timestamps = pd.to_datetime(df['DataTS'])
                dt_seconds = timestamps.diff().dt.total_seconds()
                median_dt = dt_seconds.median()
                if pd.notna(median_dt) and median_dt > 0:
                    Fs = 1.0 / median_dt
                    print(f"Calculated sampling frequency (Fs) from DataTS: {Fs:.2f} Hz")
                else:
                    Fs = 20.0
                    print(f"Invalid dt calculated, defaulting Fs to {Fs} Hz")
            except Exception as e:
                Fs = 20.0
                print(f"Error calculating Fs from DataTS ({e}), defaulting to {Fs} Hz")
        else:
            Fs = 20.0
            print(f"'DataTS' column not found, defaulting Fs to {Fs} Hz")
        
        filename = os.path.basename(data_path)
        
        # Show Plotly plots with calculated time in seconds and PSD panels
        fig_plotly = raw_plots_plotly(df, filename, Fs)
        fig_plotly.show()
                
    else:
        print("No file selected or file does not exist. Exiting.")
