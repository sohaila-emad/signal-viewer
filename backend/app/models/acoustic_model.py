"""
Acoustic Models for Signal Processing
- Doppler effect simulation for vehicle passing
- Vehicle sound analysis
- Drone/Submarine detection
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import tensorflow as tf
import librosa


class DopplerEffectModel:
    """
    Model to simulate Doppler effect for a passing vehicle.
    
    The Doppler effect formula:
    f_observed = f_source * (v_sound ± v_observer) / (v_sound ± v_source)
    
    For a moving source (vehicle) and stationary observer:
    - Approaching: f_observed = f_source * (v_sound / (v_sound - v_source))
    - Receding: f_observed = f_source * (v_sound / (v_sound + v_source))
    """
    
    # Speed of sound at sea level, 20°C (m/s)
    SPEED_OF_SOUND = 343.0
    
    def __init__(self, source_frequency: float, vehicle_velocity: float):
        """
        Initialize Doppler effect model.
        
        Args:
            source_frequency: Frequency of the source sound (Hz)
            vehicle_velocity: Velocity of the vehicle (m/s)
        """
        self.source_frequency = source_frequency
        self.vehicle_velocity = vehicle_velocity
    
    def get_approaching_frequency(self) -> float:
        """Get observed frequency when vehicle is approaching."""
        return self.source_frequency * (self.SPEED_OF_SOUND / 
                                         (self.SPEED_OF_SOUND - self.vehicle_velocity))
    
    def get_receding_frequency(self) -> float:
        """Get observed frequency when vehicle is receding."""
        return self.source_frequency * (self.SPEED_OF_SOUND / 
                                         (self.SPEED_OF_SOUND + self.vehicle_velocity))
    
    def generate_doppler_signal(self, duration: float, sample_rate: int = 44100) -> tuple:
        """
        Generate a synthetic Doppler effect signal.
        
        Args:
            duration: Duration of the signal in seconds
            sample_rate: Sampling rate in Hz
            
        Returns:
            Tuple of (time_array, frequency_array, audio_signal)
        """
        t = np.linspace(0, duration, int(sample_rate * duration))
        
    
        n_samples = len(t)
        
        # Geometry-based Doppler using actual distance and radial velocity
        h = 2.0  # closest distance to observer in meters
        t_axis = np.linspace(-duration/2, duration/2, n_samples)
        dist = np.sqrt((self.vehicle_velocity * t_axis)**2 + h**2)
        v_radial = (self.vehicle_velocity**2 * t_axis) / dist
        frequencies = self.source_frequency * (self.SPEED_OF_SOUND / (self.SPEED_OF_SOUND + v_radial))

        # Sawtooth wave for realistic horn character
        phase = 2 * np.pi * np.cumsum(frequencies) / sample_rate
        audio = 0.5 * (2 * (phase / (2 * np.pi) % 1) - 1)

        # Brown noise for engine rumble
        engine_noise = np.cumsum(np.random.uniform(-1, 1, n_samples))
        engine_noise /= np.max(np.abs(engine_noise))

        # Inverse square law amplitude envelope
        amplitude = h / dist
        audio = (audio + engine_noise * 0.3) * amplitude

        # Normalize
        audio = audio / np.max(np.abs(audio))
        
        return t, frequencies, audio
    
    def estimate_velocity_from_frequency(self, observed_frequency: float, 
                                         is_approaching: bool = True) -> float:
        """
        Estimate vehicle velocity from observed frequency.
        
        Args:
            observed_frequency: The observed frequency (Hz)
            is_approaching: True if vehicle is approaching, False if receding
            
        Returns:
            Estimated velocity in m/s
        """
        if is_approaching:
            # f_obs = f_source * v_sound / (v_sound - v)
            # f_obs * (v_sound - v) = f_source * v_sound
            # f_obs * v_sound - f_obs * v = f_source * v_sound
            # f_obs * v = f_obs * v_sound - f_source * v_sound
            # v = v_sound * (f_obs - f_source) / f_obs
            v = self.SPEED_OF_SOUND * (observed_frequency - self.source_frequency) / observed_frequency
        else:
            # f_obs = f_source * v_sound / (v_sound + v)
            # v = v_sound * (f_source - f_obs) / f_obs
            v = self.SPEED_OF_SOUND * (self.source_frequency - observed_frequency) / observed_frequency
        
        return abs(v)


class VehicleSoundAnalyzer:
    """
    Analyzer for vehicle passing sounds.
    Uses FFT-based frequency analysis to estimate velocity and horn frequency.
    """
    
    def __init__(self):
        self.sample_rate = 44100  # Default sample rate
    
    def analyze_audio(self, audio_data: np.ndarray, sample_rate: int = 44100) -> dict:
        """
        Analyze audio data to extract frequency components.
        
        Args:
            audio_data: Audio signal array
            sample_rate: Sampling rate in Hz
            
        Returns:
            Dictionary with analysis results
        """
        self.sample_rate = sample_rate
        
        # Apply FFT
        n = len(audio_data)
        fft_result = fft(audio_data)
        frequencies = fftfreq(n, 1/sample_rate)
        
        # Get magnitude spectrum (only positive frequencies)
        positive_freq_idx = frequencies > 0
        positive_frequencies = frequencies[positive_freq_idx]
        magnitude = np.abs(fft_result[positive_freq_idx])
        
        # Ignore frequencies below 80Hz (noise) and above 4kHz (not vehicle)
        valid = (positive_frequencies >= 80) & (positive_frequencies <= 4000)
        magnitude[~valid] = 0
        peaks, properties = signal.find_peaks(magnitude, height=np.max(magnitude)*0.15, distance=50)
        
        peak_frequencies = positive_frequencies[peaks]
        peak_magnitudes = magnitude[peaks]
        
        # Sort by magnitude
        sorted_idx = np.argsort(peak_magnitudes)[::-1]
        
        results = {
            'dominant_frequencies': peak_frequencies[sorted_idx[:5]].tolist(),
            'peak_magnitudes': peak_magnitudes[sorted_idx[:5]].tolist(),
            'spectrum': {
                'frequencies': positive_frequencies[::10].tolist(),  # Downsample for JSON
                'magnitude': magnitude[::10].tolist()
            }
        }
        
        return results
    
    def estimate_vehicle_parameters(self, audio_data: np.ndarray, 
                                   sample_rate: int = 44100) -> dict:
        """
        Estimate velocity using FFT on approaching/receding halves.

        Args:
            audio_data: Audio signal array
            sample_rate: Sampling rate in Hz

        Returns:
            Dictionary with estimated parameters
        """
        # Find crossover point — moment vehicle is closest = loudest
        energy = np.convolve(audio_data**2, np.ones(1000)/1000, mode='same')
        crossover_idx = np.argmax(energy)

        # Need enough samples on each side for reliable FFT
        if crossover_idx < sample_rate * 0.5 or crossover_idx > len(audio_data) - sample_rate * 0.5:
            return {'estimated_velocity': None, 'estimated_horn_frequency': None, 'confidence': 0}

        before = audio_data[:crossover_idx]
        after  = audio_data[crossover_idx:]

        def dominant_frequency(segment):
            """Get dominant frequency in vehicle range using FFT."""
            windowed = segment * np.hanning(len(segment))
            spectrum = np.abs(fft(windowed))
            freqs    = fftfreq(len(segment), 1/sample_rate)
            # Only look at 80Hz-4kHz range
            valid = (freqs >= 80) & (freqs <= 4000)
            spectrum[~valid] = 0
            return float(freqs[np.argmax(spectrum)])

        f_approaching = dominant_frequency(before)
        f_receding    = dominant_frequency(after)

        if f_approaching <= f_receding:
            return {'estimated_velocity': None, 'estimated_horn_frequency': None, 'confidence': 0}

        # Doppler formula: v = c * (f_before - f_after) / (f_before + f_after)
        estimated_velocity = DopplerEffectModel.SPEED_OF_SOUND * (f_approaching - f_receding) / (f_approaching + f_receding)
        estimated_horn_freq  = (f_approaching + f_receding) / 2
        freq_spread          = f_approaching - f_receding

        # Confidence: larger spread relative to source = stronger Doppler signal
        confidence = round(min((freq_spread / estimated_horn_freq) * 5, 1.0), 3)

        return {
            'estimated_velocity':      round(estimated_velocity, 2),
            'estimated_horn_frequency': round(estimated_horn_freq, 2),
            'confidence':              confidence,
            'doppler_shift':           round(freq_spread, 2)
        }

class DroneDetector:
    """Drone detector using a trained TensorFlow mel-spectrogram classifier."""

    SR        = 16000
    DURATION  = 2
    N_MELS    = 64
    N_SAMPLES = 16000 * 2
    THRESHOLD = 0.5

    def __init__(self):
        
        self.model = tf.keras.models.load_model("..\models\drone_detector.h5")

    def detect(self, audio_data: np.ndarray, sample_rate: int = 44100) -> dict:
        """
        Detect drone using mel-spectrogram fed into trained model.

        Args:
            audio_data: Audio signal array
            sample_rate: Sample rate in Hz

        Returns:
            Detection results
        """
       
        print(f"1. Input received: type={type(audio_data)} sample_rate={sample_rate}")
        audio_data = np.array(audio_data, dtype=np.float32)
        print(f"2. Converted to array: shape={audio_data.shape} max={audio_data.max():.4f}")

        if sample_rate != self.SR:
            print(f"3. Resampling from {sample_rate} to {self.SR}...")
            audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=self.SR)
            print(f"4. Resampled: shape={audio_data.shape} max={audio_data.max():.4f}")

        if len(audio_data) < self.N_SAMPLES:
            audio_data = np.pad(audio_data, (0, self.N_SAMPLES - len(audio_data)))
        else:
            audio_data = audio_data[:self.N_SAMPLES]
        print(f"5. After pad/trim: shape={audio_data.shape} max={audio_data.max():.4f}")

        audio_data = audio_data.astype(np.float32)
        mel    = librosa.feature.melspectrogram(y=audio_data, sr=self.SR, n_mels=self.N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        # Must match normalization applied during training
        mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
        mel_db = mel_db[..., np.newaxis][np.newaxis, ...]
        
        print(f"6. Mel shape: {mel_db.shape}")

        prob      = float(self.model.predict(mel_db, verbose=0)[0][0])
        detection = "drone" if prob >= self.THRESHOLD else "no_drone"
        print(f"7. Prediction: prob={prob:.4f} detection={detection}")

        return {
            'detection':       detection,
            'drone_score':     round(prob, 3),
            'submarine_score': 0.0,
            'confidence':      round(prob if detection == "drone" else 1 - prob, 3),
        }


def generate_vehicle_passing_sound(velocity: float, frequency: float, 
                                   duration: float = 5.0, 
                                   sample_rate: int = 44100) -> dict:
    """
    Generate a synthetic vehicle passing sound with Doppler effect.
    
    Args:
        velocity: Vehicle velocity in m/s
        frequency: Horn/source frequency in Hz
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        Dictionary with audio data and parameters
    """
    model = DopplerEffectModel(frequency, velocity)
    t, frequencies, audio = model.generate_doppler_signal(duration, sample_rate)
    
    return {
        'time': t.tolist(),
        'frequencies': frequencies.tolist(),
        'audio': audio.tolist(),
        'parameters': {
            'velocity': velocity,
            'frequency': frequency,
            'approaching_frequency': round(model.get_approaching_frequency(), 2),
            'receding_frequency': round(model.get_receding_frequency(), 2),
            'duration': duration,
            'sample_rate': sample_rate
        }
    }


def analyze_vehicle_sound(audio_data: list, sample_rate: int = 44100) -> dict:
    """
    Analyze vehicle sound to estimate velocity and frequency.
    
    Args:
        audio_data: List of audio samples
        sample_rate: Sample rate in Hz
        
    Returns:
        Analysis results
    """
    audio_array = np.array(audio_data)
    analyzer = VehicleSoundAnalyzer()
    return analyzer.estimate_vehicle_parameters(audio_array, sample_rate)


def detect_drone_submarine(audio_data: list, sample_rate: int = 44100) -> dict:
    """
    Detect drone or submarine sounds in audio.
    
    Args:
        audio_data: List of audio samples
        sample_rate: Sample rate in Hz
        
    Returns:
        Detection results
    """
    
    audio_array = np.array(audio_data, dtype=np.float32)
    detector = DroneDetector()
    return detector.detect(audio_array, sample_rate)
