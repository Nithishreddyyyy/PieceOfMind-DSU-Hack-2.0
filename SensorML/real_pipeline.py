import serial
import time
import numpy as np
import os
from collections import deque
from scipy import signal
from scipy.stats import skew, kurtosis
import torch
import torch.nn as nn
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
try:
    import mne
    MNE_AVAILABLE = True
except ImportError:
    MNE_AVAILABLE = False
    print("MNE not available - EDF export will be skipped")

# -----------------------
# Config - EXACT MATCH with training
# -----------------------
SERIAL_PORT = 'COM11'  # Change to your Arduino port
BAUD_RATE = 115200
SAMPLING_RATE = 250       # Arduino sampling rate
TRAINING_SAMPLING_RATE = 128  # Training sampling rate
EXPECTED_TIME_STEPS = 3200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "best_model.pth"
RAW_SCALER_PATH = "raw_scaler.pkl"
NUM_CLASSES = 3
COLLECTION_SECONDS = 30  # 30 seconds collection
CONVERSION_FACTOR = 500 / 1023
DROPOUT_RATE = 0.5
print(f"Using device: {DEVICE}")

# -----------------------
# Create data directory
# -----------------------
def create_data_directory():
    """Create data directory for saving files"""
    data_dir = "data"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    return data_dir

# -----------------------
# EEG Analysis Functions
# -----------------------
def detect_spikes(data, threshold_factor=3):
    """Detect spikes in EEG data"""
    threshold = np.std(data) * threshold_factor
    spikes = np.abs(data) > threshold
    spike_count = np.sum(spikes)
    spike_rate = spike_count / (len(data) / SAMPLING_RATE)
    return spike_count, spike_rate, np.where(spikes)[0]

def calculate_band_powers(data, fs=250):
    """Calculate power in different frequency bands"""
    freqs, psd = signal.welch(data, fs=fs, nperseg=min(512, len(data)//4))
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 12),
        'Beta': (12, 30),
        'Gamma': (30, 50)
    }
    band_powers = {}
    for band_name, (low_freq, high_freq) in bands.items():
        idx = (freqs >= low_freq) & (freqs <= high_freq)
        power = np.trapz(psd[idx], freqs[idx])
        band_powers[band_name] = power
    return band_powers, freqs, psd

def create_eeg_plots(data, band_powers, freqs, psd, spike_indices, timestamp):
    """Create comprehensive EEG analysis plots"""
    fig = plt.figure(figsize=(15, 12))
    time_vec = np.linspace(0, len(data)/SAMPLING_RATE, len(data))
    
    plt.subplot(2, 3, 1)
    plt.plot(time_vec, data, 'b-', linewidth=0.5)
    if len(spike_indices) > 0:
        plt.scatter(time_vec[spike_indices], data[spike_indices],
                   color='red', s=20, alpha=0.7, label=f'Spikes ({len(spike_indices)})')
        plt.legend()
    plt.title('Raw EEG Signal (Fp1)', fontsize=12, fontweight='bold')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    plt.semilogy(freqs, psd, 'g-', linewidth=1.5)
    plt.title('Power Spectral Density', fontsize=12, fontweight='bold')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (µV²/Hz)')
    plt.xlim([0, 50])
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 3)
    band_names = list(band_powers.keys())
    powers = list(band_powers.values())
    colors = ['purple', 'blue', 'green', 'orange', 'red']
    bars = plt.bar(band_names, powers, color=colors, alpha=0.7)
    plt.title('Frequency Band Powers', fontsize=12, fontweight='bold')
    plt.ylabel('Power (µV²)')
    plt.xticks(rotation=45)
    for bar, power in zip(bars, powers):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(powers)*0.01,
                f'{power:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.subplot(2, 3, 4)
    f, t, Sxx = signal.spectrogram(data, SAMPLING_RATE, nperseg=256)
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
    plt.title('Spectrogram', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.ylim([0, 50])
    plt.colorbar(label='Power (dB)')
    
    plt.subplot(2, 3, 5)
    stats_text = f"""Signal Statistics:
Mean: {np.mean(data):.3f} µV
Std: {np.std(data):.3f} µV
Min: {np.min(data):.3f} µV
Max: {np.max(data):.3f} µV
Range: {np.max(data) - np.min(data):.3f} µV
Skewness: {skew(data):.3f}
Kurtosis: {kurtosis(data):.3f}
Theta/Alpha: {band_powers['Theta']/band_powers['Alpha']:.2f}
Beta/Alpha: {band_powers['Beta']/band_powers['Alpha']:.2f}"""
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    plt.axis('off')
    
    plt.subplot(2, 3, 6)
    powers = list(band_powers.values())
    band_names = list(band_powers.keys())
    colors = ['purple', 'blue', 'green', 'orange', 'red']
    plt.pie(powers, labels=band_names, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Relative Band Powers', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plot_filename = f"data/eeg_plot_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    return plot_filename

# -----------------------
# Feature Extraction - 47 features for analysis
# -----------------------
def extract_advanced_features(data, fs=128):
    """Extract EXACTLY 47 features for analysis"""
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    features = []

    # 1. Enhanced frequency features with 7 bands (7 features)
    freqs, psd = signal.welch(data, fs=fs, axis=-1, nperseg=min(512, data.shape[-1]//4))
    bands = [
        (0.5, 4),    # Delta
        (4, 8),      # Theta
        (8, 12),     # Alpha
        (12, 16),    # Low Beta
        (16, 24),    # High Beta
        (24, 40),    # Gamma
        (40, 60)     # High Gamma
    ]
    band_powers = []
    for low, high in bands:
        idx = (freqs >= low) & (freqs < high)
        if np.any(idx):
            band_power = np.mean(psd[..., idx], axis=-1, keepdims=True)
            band_powers.append(band_power)
        else:
            band_powers.append(np.zeros((*data.shape[:-1], 1)))
    freq_features = np.concatenate(band_powers, axis=-1)
    features.append(freq_features)

    # 2. Statistical features (10 features)
    stat_features = []
    stat_features.append(np.mean(data, axis=-1, keepdims=True))
    stat_features.append(np.std(data, axis=-1, keepdims=True))
    stat_features.append(skew(data, axis=-1)[..., np.newaxis])
    stat_features.append(kurtosis(data, axis=-1)[..., np.newaxis])
    for p in [10, 25, 75, 90]:
        stat_features.append(np.percentile(data, p, axis=-1, keepdims=True))
    zcr = np.sum(np.diff(np.sign(data), axis=-1) != 0, axis=-1, keepdims=True) / data.shape[-1]
    stat_features.append(zcr)
    rms = np.sqrt(np.mean(data**2, axis=-1, keepdims=True))
    stat_features.append(rms)
    stat_features = np.concatenate(stat_features, axis=-1)
    features.append(stat_features)

    # 3. Wavelet features (4 features)
    wavelet_features = []
    for low, high in [(1, 8), (8, 16), (16, 32), (32, 60)]:
        try:
            sos = signal.butter(4, [low, high], btype='bandpass', fs=fs, output='sos')
            filtered = signal.sosfilt(sos, data)
            wavelet_features.append(np.mean(filtered**2, axis=-1, keepdims=True))
        except:
            wavelet_features.append(np.zeros((*data.shape[:-1], 1)))
    wavelet_features = np.concatenate(wavelet_features, axis=-1)
    features.append(wavelet_features)

    # Combine all features so far: 7 + 10 + 4 = 21 features
    partial_features = np.concatenate(features, axis=-1)
    
    # 4. Add exactly 26 more features to reach 47 total
    additional_features = []
    
    # Spectral features (5 features)
    try:
        spectral_centroid = np.sum(freqs * psd.flatten()) / np.sum(psd.flatten())
        additional_features.append(np.array([spectral_centroid]).reshape(1, 1))
        spectral_spread = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd.flatten()) / np.sum(psd.flatten()))
        additional_features.append(np.array([spectral_spread]).reshape(1, 1))
        cumsum_psd = np.cumsum(psd.flatten())
        for threshold in [0.75, 0.85, 0.95]:
            rolloff_idx = np.argmax(cumsum_psd >= threshold * cumsum_psd[-1])
            additional_features.append(np.array([freqs[rolloff_idx]]).reshape(1, 1))
    except:
        for _ in range(5):
            additional_features.append(np.zeros((1, 1)))
    
    # Hjorth parameters (3 features)
    data_flat = data.flatten()
    try:
        diff_data = np.diff(data_flat)
        diff2_data = np.diff(diff_data)
        activity = np.var(data_flat)
        mobility = np.sqrt(np.var(diff_data) / activity) if activity > 0 else 0
        complexity = np.sqrt(np.var(diff2_data) / np.var(diff_data)) if np.var(diff_data) > 0 else 0
        additional_features.append(np.array([activity]).reshape(1, 1))
        additional_features.append(np.array([mobility]).reshape(1, 1))
        additional_features.append(np.array([complexity]).reshape(1, 1))
    except:
        for _ in range(3):
            additional_features.append(np.zeros((1, 1)))
    
    # Peak-to-peak (1 feature)
    additional_features.append(np.array([np.max(data_flat) - np.min(data_flat)]).reshape(1, 1))
    
    # Autocorrelation features (10 features)
    for lag in [1, 5, 10, 20, 30, 50, 100, 150, 200, 250]:
        if lag < len(data_flat):
            try:
                corr_coef = np.corrcoef(data_flat[:-lag], data_flat[lag:])[0, 1]
                if np.isnan(corr_coef):
                    corr_coef = 0.0
                additional_features.append(np.array([corr_coef]).reshape(1, 1))
            except:
                additional_features.append(np.array([0.0]).reshape(1, 1))
        else:
            additional_features.append(np.array([0.0]).reshape(1, 1))
    
    # Additional statistical measures (7 features)
    try:
        additional_features.append(np.array([np.var(data_flat)]).reshape(1, 1))
        additional_features.append(np.array([np.median(data_flat)]).reshape(1, 1))
        q75, q25 = np.percentile(data_flat, [75, 25])
        additional_features.append(np.array([q75 - q25]).reshape(1, 1))
        mad = np.mean(np.abs(data_flat - np.mean(data_flat)))
        additional_features.append(np.array([mad]).reshape(1, 1))
        energy = np.sum(data_flat ** 2)
        additional_features.append(np.array([energy]).reshape(1, 1))
        hist, _ = np.histogram(data_flat, bins=20)
        hist = hist / np.sum(hist)
        entropy = -np.sum(hist[hist > 0] * np.log2(hist[hist > 0]))
        additional_features.append(np.array([entropy]).reshape(1, 1))
        waveform_length = np.sum(np.abs(np.diff(data_flat)))
        additional_features.append(np.array([waveform_length]).reshape(1, 1))
    except:
        for _ in range(7):
            additional_features.append(np.zeros((1, 1)))
    
    # Combine additional features: 5 + 3 + 1 + 10 + 7 = 26 features
    additional_features = np.concatenate(additional_features, axis=-1)
    
    # Combine all features: 21 + 26 = 47 features
    all_features = np.concatenate([partial_features, additional_features], axis=-1)
    
    # Ensure exactly 47 features
    if all_features.shape[-1] != 47:
        print(f"Warning: Feature count mismatch. Got {all_features.shape[-1]}, expected 47")
        if all_features.shape[-1] < 47:
            padding = np.zeros((all_features.shape[0], all_features.shape[1], 47 - all_features.shape[-1]))
            all_features = np.concatenate([all_features, padding], axis=-1)
        else:
            all_features = all_features[:, :, :47]
    
    print(f"Extracted exactly {all_features.shape[-1]} features for analysis")
    return all_features

# -----------------------
# Feature Extraction - 21 features for model
# -----------------------
def extract_model_features(data, fs=128):
    """Extract EXACTLY 21 features to match pre-trained model"""
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    features = []

    # 1. Frequency features (7 features)
    freqs, psd = signal.welch(data, fs=fs, axis=-1, nperseg=min(512, data.shape[-1]//4))
    bands = [
        (0.5, 4),    # Delta
        (4, 8),      # Theta
        (8, 12),     # Alpha
        (12, 16),    # Low Beta
        (16, 24),    # High Beta
        (24, 40),    # Gamma
        (40, 60)     # High Gamma
    ]
    band_powers = []
    for low, high in bands:
        idx = (freqs >= low) & (freqs < high)
        if np.any(idx):
            band_power = np.mean(psd[..., idx], axis=-1, keepdims=True)
            band_powers.append(band_power)
        else:
            band_powers.append(np.zeros((*data.shape[:-1], 1)))
    freq_features = np.concatenate(band_powers, axis=-1)
    features.append(freq_features)

    # 2. Statistical features (10 features)
    stat_features = []
    stat_features.append(np.mean(data, axis=-1, keepdims=True))
    stat_features.append(np.std(data, axis=-1, keepdims=True))
    stat_features.append(skew(data, axis=-1)[..., np.newaxis])
    stat_features.append(kurtosis(data, axis=-1)[..., np.newaxis])
    for p in [10, 25, 75, 90]:
        stat_features.append(np.percentile(data, p, axis=-1, keepdims=True))
    zcr = np.sum(np.diff(np.sign(data), axis=-1) != 0, axis=-1, keepdims=True) / data.shape[-1]
    stat_features.append(zcr)
    rms = np.sqrt(np.mean(data**2, axis=-1, keepdims=True))
    stat_features.append(rms)
    stat_features = np.concatenate(stat_features, axis=-1)
    features.append(stat_features)

    # 3. Wavelet features (4 features)
    wavelet_features = []
    for low, high in [(1, 8), (8, 16), (16, 32), (32, 60)]:
        try:
            sos = signal.butter(4, [low, high], btype='bandpass', fs=fs, output='sos')
            filtered = signal.sosfilt(sos, data)
            wavelet_features.append(np.mean(filtered**2, axis=-1, keepdims=True))
        except:
            wavelet_features.append(np.zeros((*data.shape[:-1], 1)))
    wavelet_features = np.concatenate(wavelet_features, axis=-1)
    features.append(wavelet_features)

    # Combine: 7 + 10 + 4 = 21 features
    all_features = np.concatenate(features, axis=-1)
    
    print(f"Extracted exactly {all_features.shape[-1]} features for model input")
    return all_features

# -----------------------
# Model Architecture
# -----------------------
class EnhancedCNNLSTM(nn.Module):
    def __init__(self, num_channels=1, time_steps=3200, feature_size=21, num_classes=3):
        super(EnhancedCNNLSTM, self).__init__()
        
        self.time_steps = time_steps
        self.feature_size = feature_size
        
        # Enhanced CNN for temporal features
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(num_channels, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.Conv1d(32, 32, kernel_size=7, padding=3),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Dropout1d(0.1)
            ),
            nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Conv1d(64, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Dropout1d(0.1)
            ),
            nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(2),
                nn.Dropout1d(0.1)
            ),
            nn.Sequential(
                nn.Conv1d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.Conv1d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.AdaptiveMaxPool1d(100),
                nn.Dropout1d(0.2)
            )
        ])

        # Multi-head attention for LSTM
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=3,
            batch_first=True,
            dropout=0.2,
            bidirectional=True
        )

        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=256,  # bidirectional LSTM: 128*2
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

        # Feature processing for extracted features
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )

        # Final classification layers
        classifier_input_size = 256 + 64  # LSTM output + processed features
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE * 0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_RATE * 0.3),
            nn.Linear(64, num_classes)
        )

        self._initialize_weights()

    def forward(self, x):
        batch_size = x.size(0)
        raw_eeg = x[:, :, :self.time_steps]
        features = x[:, :, self.time_steps:]
        for conv_block in self.conv_layers:
            raw_eeg = conv_block(raw_eeg)
        raw_eeg = raw_eeg.permute(0, 2, 1)
        lstm_out, (h_n, c_n) = self.lstm(raw_eeg)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        lstm_features = torch.mean(attn_out, dim=1)
        features_flat = features.view(batch_size, -1)
        processed_features = self.feature_processor(features_flat)
        combined = torch.cat([lstm_features, processed_features], dim=1)
        output = self.classifier(combined)
        return output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_normal_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

# -----------------------
# Preprocessing
# -----------------------
def preprocess_segment(data):
    """Preprocess EEG segment - EXACT match with training pipeline"""
    data_uv = np.array(data) * CONVERSION_FACTOR
    data_uv = data_uv - np.mean(data_uv)
    data_shaped = data_uv.reshape(1, -1)
    sos = signal.butter(4, [0.5, 60], btype='bandpass', fs=SAMPLING_RATE, output='sos')
    data_filtered = signal.sosfilt(sos, data_shaped)
    data_filtered = signal.detrend(data_filtered, axis=1)
    if SAMPLING_RATE != TRAINING_SAMPLING_RATE:
        data_filtered = signal.resample(data_filtered,
                                      int(data_filtered.shape[1] * (TRAINING_SAMPLING_RATE/SAMPLING_RATE)),
                                      axis=1)
    if data_filtered.shape[1] != EXPECTED_TIME_STEPS:
        data_filtered = signal.resample(data_filtered, EXPECTED_TIME_STEPS, axis=1)
    return data_filtered

# -----------------------
# Load scalers and model
# -----------------------
def load_scalers():
    try:
        raw_scaler = joblib.load(RAW_SCALER_PATH)
        print("Raw scaler loaded successfully.")
        return raw_scaler
    except Exception as e:
        print(f"Error loading raw scaler: {e}")
        return None

def load_model():
    try:
        model = EnhancedCNNLSTM(
            num_channels=1,
            time_steps=EXPECTED_TIME_STEPS,
            feature_size=21,
            num_classes=NUM_CLASSES
        ).to(DEVICE)
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        print("Model loaded successfully (21 features).")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# -----------------------
# Input preparation
# -----------------------
def prepare_input_for_model(signal_data, raw_scaler):
    """Prepare input for model - EXACT match with training pipeline, no feature scaling"""
    X_features = extract_model_features(signal_data, fs=TRAINING_SAMPLING_RATE)
    if raw_scaler is None:
        print("Raw scaler not loaded - fitting on current sample")
        from sklearn.preprocessing import RobustScaler
        raw_scaler = RobustScaler()
        raw_scaler.fit(signal_data.reshape(1, -1))
    raw_scaled = raw_scaler.transform(signal_data.reshape(1, -1))
    raw_scaled = raw_scaled.reshape(1, 1, EXPECTED_TIME_STEPS)
    feature_scaled = X_features.reshape(1, 1, -1)  # No scaling applied
    X_combined = np.concatenate([raw_scaled, feature_scaled], axis=-1)
    X_tensor = torch.tensor(X_combined, dtype=torch.float32).to(DEVICE)
    return X_tensor

# -----------------------
# Save functions
# -----------------------
def save_data_as_edf(data, filename, sampling_rate=250):
    """Save data as EDF format with fixed physical min/max"""
    if not MNE_AVAILABLE:
        print("MNE not available - skipping EDF export")
        return
    try:
        data_array = np.array(data, dtype=np.float64).reshape(1, -1)
        ch_names = ['Fp1']
        ch_types = ['eeg']
        info = mne.create_info(ch_names=ch_names, sfreq=sampling_rate, ch_types=ch_types)
        data_min = np.min(data_array)
        data_max = np.max(data_array)
        physical_min = max(data_min, -999999.9)
        physical_max = min(data_max, 999999.9)
        raw = mne.io.RawArray(data_array, info)
        raw.set_channel_types({ch_names[0]: 'eeg'})
        raw.info['chs'][0]['range'] = physical_max - physical_min
        raw.info['chs'][0]['cal'] = 1.0
        raw.info['chs'][0]['unit'] = mne.io.constants.FIFF.FIFF_UNIT_V
        raw.info['chs'][0]['unit_mul'] = mne.io.constants.FIFF.FIFF_UNITM_UV
        raw.export(filename, fmt='edf', physical_min=physical_min, physical_max=physical_max, overwrite=True)
        print(f"Data saved as EDF: {filename}")
    except Exception as e:
        print(f"Error saving EDF file: {e}")

def save_analysis_report(raw_data, band_powers, spike_count, spike_rate, timestamp, patient_name="Patient"):
    """Save comprehensive EEG analysis report"""
    theta_alpha_ratio = band_powers['Theta'] / band_powers['Alpha'] if band_powers['Alpha'] > 0 else 0
    report_content = f"""EEG Analysis Report for {patient_name}
Analysis Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Recording Date and Time: {timestamp}
Recording Duration: {COLLECTION_SECONDS} seconds

Signal Quality Assessment:
- Total samples collected: {len(raw_data)}
- Sampling rate: {SAMPLING_RATE} Hz
- Signal range: {np.min(raw_data):.3f} to {np.max(raw_data):.3f} µV
- Mean amplitude: {np.mean(raw_data):.3f} µV
- Standard deviation: {np.std(raw_data):.3f} µV

Frequency Band Powers:
Delta (0.5-4 Hz): {band_powers['Delta']:.6f} µV²
Theta (4-8 Hz): {band_powers['Theta']:.6f} µV²
Alpha (8-12 Hz): {band_powers['Alpha']:.6f} µV²
Beta (12-30 Hz): {band_powers['Beta']:.6f} µV²
Gamma (30-50 Hz): {band_powers['Gamma']:.6f} µV²

Computed Ratios:
Theta/Alpha ratio: {theta_alpha_ratio:.2f}
Beta/Alpha ratio: {band_powers['Beta']/band_powers['Alpha']:.2f}

Artifact Detection:
Spike count: {spike_count}
Spike rate: {spike_rate:.2f} spikes/second

Statistical Analysis:
Skewness: {skew(raw_data):.3f}
Kurtosis: {kurtosis(raw_data):.3f}
Zero crossing rate: {np.sum(np.diff(np.sign(raw_data)) != 0) / len(raw_data):.4f}
RMS energy: {np.sqrt(np.mean(np.array(raw_data)**2)):.3f} µV

Clinical Notes:
- Recording from Fp1 electrode (frontal cortex)
- Data preprocessed with 0.5-60 Hz bandpass filter
- Spike detection threshold: 3 standard deviations
- Analysis performed using automated EEG processing pipeline

Generated by: Real-time EEG Analysis System
Software Version: 1.0
"""
    report_filename = f"data/eeg_analysis_report_{timestamp}.txt"
    with open(report_filename, 'w') as f:
        f.write(report_content)
    print(f"Analysis report saved: {report_filename}")
    return report_filename

# -----------------------
# Modified Classification with Probability Redistribution
# -----------------------
def modify_probabilities(original_probabilities):
    """
    Modify probabilities according to the specified requirements:
    - Map High Stress to Low Stress
    - Add 30% to Normal (Moderate Stress)
    - Add 10% to Normal 
    - Redistribute remaining to Low Stress
    """
    # Original classes: [Low Stress, Moderate Stress, High Stress] -> indices [0, 1, 2]
    low_stress_prob = original_probabilities[0]
    moderate_stress_prob = original_probabilities[1] 
    high_stress_prob = original_probabilities[2]
    
    # Map High Stress probability to Low Stress
    new_low_stress_prob = low_stress_prob + high_stress_prob
    
    # Add 40% total to Moderate Stress (30% + 10%)
    boost_factor = 0.40
    new_moderate_stress_prob = moderate_stress_prob * (1 + boost_factor)
    
    # Calculate total probability so far
    current_total = new_low_stress_prob + new_moderate_stress_prob
    
    # If total exceeds 1, normalize
    if current_total > 1.0:
        # Normalize while maintaining the relative boost to moderate stress
        new_low_stress_prob = new_low_stress_prob / current_total
        new_moderate_stress_prob = new_moderate_stress_prob / current_total
    else:
        # Add remaining probability to Low Stress
        remaining = 1.0 - current_total
        new_low_stress_prob += remaining
    
    # Create new probability array
    modified_probs = np.array([new_low_stress_prob, new_moderate_stress_prob, 0.0])
    
    # Ensure probabilities sum to 1
    modified_probs = modified_probs / np.sum(modified_probs)
    
    return modified_probs

def classify_eeg_data(preprocessed_data, model, raw_scaler):
    """Classify EEG data and return modified prediction with adjusted probabilities"""
    try:
        X_input = prepare_input_for_model(preprocessed_data, raw_scaler)
        print(f"Model input shape: {X_input.shape}")
        with torch.no_grad():
            logits = model(X_input)
            original_probabilities = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
            
            # Modify probabilities according to requirements
            modified_probabilities = modify_probabilities(original_probabilities)
            
            # Get predicted class based on modified probabilities
            predicted_class = int(np.argmax(modified_probabilities))
            
        print(f"Original probabilities: {original_probabilities}")
        print(f"Modified probabilities: {modified_probabilities}")
        
        return predicted_class, modified_probabilities
    except Exception as e:
        print(f"Error during classification: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# -----------------------
# Test function
# -----------------------
def test_feature_extraction():
    """Test feature extraction to verify it produces 21 features for model"""
    print("Testing model feature extraction...")
    dummy_data = np.random.randn(EXPECTED_TIME_STEPS) * 10
    dummy_data = dummy_data.reshape(1, -1)
    features = extract_model_features(dummy_data, fs=TRAINING_SAMPLING_RATE)
    print(f"Model feature shape: {features.shape}")
    print(f"Number of features: {features.shape[-1]}")
    if features.shape[-1] == 21:
        print("SUCCESS: Model feature extraction produces exactly 21 features!")
    else:
        print(f"ERROR: Expected 21 features, got {features.shape[-1]}")
    return features.shape[-1] == 21

def test_probability_modification():
    """Test the probability modification function"""
    print("\nTesting probability modification...")
    
    # Test case 1: High stress prediction
    test_probs_1 = np.array([0.1, 0.2, 0.7])  # High stress dominant
    modified_1 = modify_probabilities(test_probs_1)
    print(f"Test 1 - Original: {test_probs_1}, Modified: {modified_1}")
    
    # Test case 2: Moderate stress prediction
    test_probs_2 = np.array([0.2, 0.6, 0.2])  # Moderate stress dominant
    modified_2 = modify_probabilities(test_probs_2)
    print(f"Test 2 - Original: {test_probs_2}, Modified: {modified_2}")
    
    # Test case 3: Low stress prediction
    test_probs_3 = np.array([0.7, 0.2, 0.1])  # Low stress dominant
    modified_3 = modify_probabilities(test_probs_3)
    print(f"Test 3 - Original: {test_probs_3}, Modified: {modified_3}")
    
    print("Probability modification test completed!")

# -----------------------
# Main pipeline
# -----------------------
def main():
    print("=" * 60)
    print("Complete Real-time EEG Analysis Pipeline with Modified Classification")
    print("=" * 60)
    
    data_dir = create_data_directory()
    
    raw_scaler = load_scalers()
    model = load_model()
    
    if model is None:
        print("Failed to load model. Exiting...")
        return
    
    print(f"Connecting to Arduino on {SERIAL_PORT}...")
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"Connected successfully to {SERIAL_PORT}")
    except Exception as e:
        print(f"Failed to connect to Arduino: {e}")
        return
    
    print(f"Starting {COLLECTION_SECONDS}-second EEG data collection...")
    print("Please remain still and relaxed during data collection.")
    
    raw_data = []
    samples_needed = COLLECTION_SECONDS * SAMPLING_RATE
    start_time = time.time()
    
    try:
        while len(raw_data) < samples_needed:
            if ser.in_waiting > 0:
                try:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        value = float(line)
                        raw_data.append(value)
                        if len(raw_data) % 1000 == 0:
                            progress = (len(raw_data) / samples_needed) * 100
                            elapsed = time.time() - start_time
                            print(f"Progress: {progress:.1f}% ({len(raw_data)}/{samples_needed} samples) - {elapsed:.1f}s elapsed")
                except (ValueError, UnicodeDecodeError):
                    continue
            if time.time() - start_time > COLLECTION_SECONDS + 10:
                print("Collection timeout. Collected partial data.")
                break
    except KeyboardInterrupt:
        print("Data collection interrupted by user.")
    except Exception as e:
        print(f"Error during data collection: {e}")
    finally:
        ser.close()
        print("Serial connection closed.")
    
    print(f"Data collection complete. Collected {len(raw_data)} samples.")
    
    if len(raw_data) < samples_needed * 0.8:
        print("Insufficient data collected. Please try again.")
        return
    
    if len(raw_data) > samples_needed:
        raw_data = raw_data[:samples_needed]
    
    raw_data = np.array(raw_data)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    raw_filename = f"data/raw_eeg_{timestamp}.txt"
    np.savetxt(raw_filename, raw_data, fmt='%.6f')
    print(f"Raw data saved: {raw_filename}")
    
    print("Preprocessing collected data...")
    preprocessed = preprocess_segment(raw_data)
    preprocessed_flat = preprocessed.flatten()
    
    duration = EXPECTED_TIME_STEPS / TRAINING_SAMPLING_RATE
    fs_analysis = TRAINING_SAMPLING_RATE
    
    print("Performing EEG analysis...")
    spike_count, spike_rate, spike_indices = detect_spikes(preprocessed_flat, threshold_factor=3)
    spike_rate = spike_count / duration
    band_powers, freqs, psd = calculate_band_powers(preprocessed_flat, fs=fs_analysis)
    
    print("Generating analysis plots...")
    plot_filename = create_eeg_plots(preprocessed_flat, band_powers, freqs, psd, spike_indices, timestamp)
    
    edf_filename = f"data/eeg_data_{timestamp}.edf"
    save_data_as_edf(preprocessed_flat, edf_filename, fs_analysis)
    
    report_filename = save_analysis_report(preprocessed_flat, band_powers, spike_count, spike_rate, timestamp)
    
    print("Classifying EEG data with modified probability distribution...")
    predicted_class, probabilities = classify_eeg_data(preprocessed, model, raw_scaler)
    
    if predicted_class is not None:
        class_names = ["Low Stress", "Moderate Stress", "High Stress"]
        print("\n" + "=" * 60)
        print("EEG Classification Results (Modified)")
        print("=" * 60)
        print(f"Predicted Mental State: {class_names[predicted_class]} (Class ID: {predicted_class})")
        print("\nModified Class Probabilities:")
        for i, prob in enumerate(probabilities):
            print(f"{class_names[i]}: {prob * 100:.2f}%")
        confidence = np.max(probabilities) * 100
        print(f"\nConfidence Level: {confidence:.1f}%")
        if confidence > 80:
            print("Classification confidence: HIGH")
        elif confidence > 60:
            print("Classification confidence: MODERATE")
        else:
            print("Classification confidence: LOW - interpret with caution")
        
        print("=" * 60)
    else:
        print("Classification failed due to an error.")
    
    print("\nPipeline complete!")
    print(f"All files saved in: {data_dir}")
    print(f"- Raw data (TXT): {raw_filename}")
    if MNE_AVAILABLE:
        print(f"- Processed data (EDF): {edf_filename}")
    print(f"- Analysis plots: {plot_filename}")
    print(f"- Analysis report: {report_filename}")

if __name__ == "__main__":
    print("🧠 Enhanced Real-time EEG Classification Pipeline with Modified Results")
    print("=" * 60)
    
    # Test probability modification
    test_probability_modification()
    
    if test_feature_extraction():
        print("Feature extraction test passed. Starting main pipeline...\n")
        try:
            main()
            print("\n🎯 Pipeline completed successfully!")
        except Exception as e:
            print(f"❌ Pipeline failed with error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("Feature extraction test failed. Please check the implementation.")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("\n🧹 GPU memory cleared.")