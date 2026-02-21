"""
Acoustic Service for Signal Processing
Provides service layer for acoustic signal processing
"""
import io
import base64
from scipy.io import wavfile
import numpy as np
from scipy import signal


from ..models.acoustic_model import (
    generate_vehicle_passing_sound,
    analyze_vehicle_sound,
    detect_drone_submarine,
    DopplerEffectModel,
    VehicleSoundAnalyzer,
    DroneDetector
)


class AcousticService:
    """Service for acoustic signal processing operations."""
    
    def __init__(self):
        self.doppler_model = None
        self.vehicle_analyzer = VehicleSoundAnalyzer()
        self.drone_detector = DroneDetector()
    


    def generate_doppler_sound_base64(self, velocity, frequency, duration=5.0, sample_rate=44100):
        """
        Generate Doppler sound and return base64 WAV for browser playback.
        """
        # Step 1: Get raw audio from your existing function
        raw = generate_vehicle_passing_sound(velocity, frequency, duration, sample_rate)
        audio_array = np.array(raw["audio"], dtype=np.float32)  # make sure it's float32

        # Step 2: Scale to int16 for WAV
        audio_int16 = np.int16(audio_array / np.max(np.abs(audio_array)) * 32767)

        # Step 3: Write WAV to memory
        buffer = io.BytesIO()
        wavfile.write(buffer, sample_rate, audio_int16)
        buffer.seek(0)

        # Step 4: Encode as base64
        audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        # Step 5: Return JSON
        return {
            "audio_base64": audio_base64,
            "sample_rate": sample_rate,
            "duration": duration
        }

    def generate_doppler_sound(self, velocity: float, frequency: float, 
                               duration: float = 5.0, 
                               sample_rate: int = 44100) -> dict:
        """
        Generate a synthetic vehicle passing sound with Doppler effect.
        
        Args:
            velocity: Vehicle velocity in m/s (typical: 10-50 m/s)
            frequency: Horn/source frequency in Hz (typical: 200-1000 Hz)
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            
        Returns:
            Dictionary with audio data and parameters
        """
        # Validate inputs
        if velocity <= 0 or velocity >= 343:  # Can't exceed speed of sound
            raise ValueError("Velocity must be between 0 and 343 m/s (speed of sound)")
        
        if frequency <= 0:
            raise ValueError("Frequency must be positive")
        
        if duration <= 0 or duration > 30:
            raise ValueError("Duration must be between 0 and 30 seconds")
        
        return generate_vehicle_passing_sound(velocity, frequency, duration, sample_rate)
    
    def analyze_vehicle_passing(self, audio_data: list, 
                                sample_rate: int = 44100) -> dict:
        """
        Analyze vehicle passing sound to estimate velocity and frequency.
        
        Args:
            audio_data: List of audio samples (normalized to -1 to 1)
            sample_rate: Sample rate in Hz
            
        Returns:
            Dictionary with estimated parameters
        """
        if not audio_data:
            raise ValueError("Audio data cannot be empty")
        
        # Convert to numpy array if needed
        if isinstance(audio_data, list):
            audio_array = np.array(audio_data, dtype=np.float64)
        else:
            audio_array = audio_data
        
        # Ensure audio is normalized
        max_val = np.max(np.abs(audio_array))
        if max_val > 1.0:
            audio_array = audio_array / max_val

        
        # Pre-filter to isolate vehicle frequency range (80Hz - 4kHz)
        sos = signal.butter(4, [80, 4000], btype='band', fs=sample_rate, output='sos')
        audio_array = signal.sosfilt(sos, audio_array)
                
        return analyze_vehicle_sound(audio_array.tolist(), sample_rate)
    
    def detect_unmanned_vehicle(self, audio_data: list, 
                               sample_rate: int = 44100,
                               vehicle_type: str = 'auto') -> dict:
        """
        Detect drone or submarine sounds in audio.
        
        Args:
            audio_data: List of audio samples
            sample_rate: Sample rate in Hz
            vehicle_type: 'auto', 'drone', or 'submarine'
            
        Returns:
            Detection results
        """
        if not audio_data:
            raise ValueError("Audio data cannot be empty")
        
        # Convert to numpy array
        if isinstance(audio_data, list):
            audio_array = np.array(audio_data, dtype=np.float64)
        else:
            audio_array = audio_data
        
        # Normalize
        max_val = np.max(np.abs(audio_array))
        if max_val > 1.0:
            audio_array = audio_array / max_val
        
        result = detect_drone_submarine(audio_array.tolist(), sample_rate)
        
        # Filter by requested vehicle type if not auto
        if vehicle_type != 'auto':
            if vehicle_type == 'drone':
                result['detection'] = 'drone' if result['drone_score'] > result['submarine_score'] else 'unknown'
            elif vehicle_type == 'submarine':
                result['detection'] = 'submarine' if result['submarine_score'] > result['drone_score'] else 'unknown'
        
        return result
    
    def get_doppler_parameters(self, velocity: float, frequency: float) -> dict:
        """
        Get Doppler effect parameters without generating audio.
        
        Args:
            velocity: Vehicle velocity in m/s
            frequency: Source frequency in Hz
            
        Returns:
            Dictionary with Doppler parameters
        """
        model = DopplerEffectModel(frequency, velocity)
        
        return {
            'source_frequency': frequency,
            'vehicle_velocity': velocity,
            'approaching_frequency': round(model.get_approaching_frequency(), 2),
            'receding_frequency': round(model.get_receding_frequency(), 2),
            'doppler_shift_approaching': round(model.get_approaching_frequency() - frequency, 2),
            'doppler_shift_receding': round(frequency - model.get_receding_frequency(), 2),
            'speed_of_sound': model.SPEED_OF_SOUND
        }
    
    def compute_spectrogram(self, audio_data: list, 
                           sample_rate: int = 44100,
                           nperseg: int = 256) -> dict:
        """
        Compute spectrogram of audio data.
        
        Args:
            audio_data: List of audio samples
            sample_rate: Sample rate in Hz
            nperseg: Length of each segment for FFT
            
        Returns:
            Spectrogram data
        """
       
        audio_array = np.array(audio_data, dtype=np.float64)
        
        # Compute spectrogram
        f, t, Sxx = signal.spectrogram(audio_array, fs=sample_rate, nperseg=nperseg)
        
        # Convert to dB scale
        Sxx_db = 10 * np.log10(Sxx + 1e-10)
        
        return {
            'frequencies': f.tolist(),
            'time': t.tolist(),
            'spectrogram': Sxx_db.tolist(),
            'parameters': {
                'sample_rate': sample_rate,
                'nperseg': nperseg,
                'nfft': nperseg
            }
        }


# Singleton instance
acoustic_service = AcousticService()


def get_acoustic_service() -> AcousticService:
    """Get the acoustic service instance."""
    return acoustic_service
