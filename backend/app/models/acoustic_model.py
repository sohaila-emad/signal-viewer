"""
Acoustic Models for Signal Processing
- Doppler effect simulation for vehicle passing
- Vehicle sound analysis
- Drone/Submarine detection
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq


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
        
        # Calculate frequency over time as vehicle approaches then passes
        # Vehicle starts at distance d approaching, passes at t=duration/2
        n_samples = len(t)
        mid_point = n_samples // 2
        
        frequencies = np.zeros(n_samples)
        
        for i in range(n_samples):
            if i < mid_point:
                # Approaching - frequency increases
                progress = i / mid_point
                # Start from lower frequency, approach max
                f_start = self.get_receding_frequency() * 0.8
                frequencies[i] = f_start + progress * (self.source_frequency - f_start)
            else:
                # Receding - frequency decreases
                progress = (i - mid_point) / (n_samples - mid_point)
                frequencies[i] = self.source_frequency * (self.SPEED_OF_SOUND / 
                                                          (self.SPEED_OF_SOUND + 
                                                           self.vehicle_velocity * progress))
        
        # Generate audio signal using the varying frequencies
        # Use sine wave with frequency modulation
        audio = np.zeros(n_samples)
        phase = 0
        for i in range(n_samples):
            phase += 2 * np.pi * frequencies[i] / sample_rate
            audio[i] = np.sin(phase)
        
        # Apply envelope to simulate vehicle approaching and receding
        envelope = np.zeros(n_samples)
        for i in range(n_samples):
            if i < mid_point:
                envelope[i] = (i / mid_point) * 0.8  # Ramp up
            else:
                envelope[i] = ((n_samples - i) / (n_samples - mid_point)) * 0.8
        
        audio = audio * envelope
        
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
        
        # Find peaks in the spectrum
        peaks, properties = signal.find_peaks(magnitude, height=np.max(magnitude)*0.1)
        
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
        Estimate vehicle velocity and horn frequency from audio.
        
        Uses spectral analysis to find the fundamental frequency and
        estimate the Doppler shift to calculate velocity.
        
        Args:
            audio_data: Audio signal array
            sample_rate: Sampling rate in Hz
            
        Returns:
            Dictionary with estimated parameters
        """
        analysis = self.analyze_audio(audio_data, sample_rate)
        
        if not analysis['dominant_frequencies']:
            return {
                'estimated_velocity': None,
                'estimated_horn_frequency': None,
                'confidence': 0
            }
        
        # Assume the dominant frequency is the horn frequency
        # For a more accurate estimate, we'd need to track frequency change over time
        estimated_horn_freq = analysis['dominant_frequencies'][0]
        
        # Simple velocity estimation based on frequency spread
        # This is a simplified model - real implementation would track frequency over time
        freq_spread = max(analysis['dominant_frequencies']) - min(analysis['dominant_frequencies'])
        
        # Estimate velocity based on expected Doppler shift
        # This is a rough estimate
        doppler_model = DopplerEffectModel(estimated_horn_freq, 30)  # Assume 30 m/s max
        max_doppler_shift = doppler_model.get_approaching_frequency() - estimated_horn_freq
        
        estimated_velocity = (freq_spread / max_doppler_shift) * 30 if max_doppler_shift > 0 else 0
        
        return {
            'estimated_velocity': round(estimated_velocity, 2),
            'estimated_horn_frequency': round(estimated_horn_freq, 2),
            'confidence': 0.7 if freq_spread > 10 else 0.3,
            'analysis': analysis
        }


class DroneDetector:
    """
    Detector for drone and submarine sounds using spectral analysis.
    Uses characteristic frequency patterns to identify drone sounds.
    """
    
    # Typical drone frequency ranges (Hz)
    DRONE_FREQUENCY_RANGES = {
        'propeller': (50, 500),
        'motor': (500, 2000),
        'electronics': (2000, 10000)
    }
    
    # Typical submarine frequency ranges (Hz)
    SUBMARINE_FREQUENCY_RANGES = {
        'propeller': (10, 100),
        'machinery': (100, 500),
        'sonar_ping': (1000, 5000)
    }
    
    def __init__(self):
        self.sample_rate = 44100
    
    def compute_spectral_features(self, audio_data: np.ndarray, 
                                  sample_rate: int = 44100) -> dict:
        """
        Compute spectral features for classification.
        
        Args:
            audio_data: Audio signal array
            sample_rate: Sampling rate in Hz
            
        Returns:
            Dictionary with spectral features
        """
        self.sample_rate = sample_rate
        
        # Compute spectrogram
        f, t, Sxx = signal.spectrogram(audio_data, fs=sample_rate)
        
        # Compute spectral features
        spectral_centroid = np.sum(f[:, np.newaxis] * Sxx, axis=0) / (np.sum(Sxx, axis=0) + 1e-10)
        spectral_bandwidth = np.sqrt(np.sum(((f[:, np.newaxis] - spectral_centroid) ** 2) * Sxx, axis=0) / 
                                      (np.sum(Sxx, axis=0) + 1e-10))
        
        # Compute power in different frequency bands
        band_powers = {}
        for band_name, (low, high) in self.DRONE_FREQUENCY_RANGES.items():
            band_idx = (f >= low) & (f <= high)
            band_powers[f'drone_{band_name}'] = np.mean(Sxx[band_idx, :]) if band_idx.any() else 0
        
        for band_name, (low, high) in self.SUBMARINE_FREQUENCY_RANGES.items():
            band_idx = (f >= low) & (f <= high)
            band_powers[f'submarine_{band_name}'] = np.mean(Sxx[band_idx, :]) if band_idx.any() else 0
        
        return {
            'spectral_centroid': float(np.mean(spectral_centroid)),
            'spectral_bandwidth': float(np.mean(spectral_bandwidth)),
            'band_powers': band_powers,
            'total_power': float(np.mean(Sxx))
        }
    
    def detect(self, audio_data: np.ndarray, sample_rate: int = 44100) -> dict:
        """
        Detect if the audio contains drone or submarine sounds.
        
        Args:
            audio_data: Audio signal array
            sample_rate: Sampling rate in Hz
            
        Returns:
            Detection results
        """
        features = self.compute_spectral_features(audio_data, sample_rate)
        
        # Simple rule-based detection
        # In a real system, this would use machine learning
        band_powers = features['band_powers']
        
        drone_score = (band_powers.get('drone_propeller', 0) * 2 + 
                     band_powers.get('drone_motor', 0) * 1.5 +
                     band_powers.get('drone_electronics', 0))
        
        submarine_score = (band_powers.get('submarine_propeller', 0) * 2 +
                         band_powers.get('submarine_machinery', 0) * 1.5 +
                         band_powers.get('submarine_sonar_ping', 0))
        
        total_power = features['total_power'] + 1e-10
        
        # Normalize scores
        drone_confidence = min(drone_score / total_power / 10, 1.0)
        submarine_confidence = min(submarine_score / total_power / 10, 1.0)
        
        # Determine detection
        detection_type = None
        confidence = 0
        
        if drone_confidence > 0.3:
            detection_type = 'drone'
            confidence = drone_confidence
        elif submarine_confidence > 0.3:
            detection_type = 'submarine'
            confidence = submarine_confidence
        else:
            detection_type = 'unknown'
            confidence = max(drone_confidence, submarine_confidence)
        
        return {
            'detection': detection_type,
            'confidence': round(confidence, 3),
            'drone_score': round(drone_confidence, 3),
            'submarine_score': round(submarine_confidence, 3),
            'features': features
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
    audio_array = np.array(audio_data)
    detector = DroneDetector()
    return detector.detect(audio_array, sample_rate)
