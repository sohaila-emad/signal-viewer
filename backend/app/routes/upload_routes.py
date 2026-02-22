from flask import Blueprint, request, jsonify
import os
import numpy as np
import pandas as pd
import tempfile
import json
import struct
from werkzeug.utils import secure_filename

# Try to import wfdb for reading WFDB format files
try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False

upload_bp = Blueprint('upload', __name__)

# Simplified extensions
ALLOWED_EXTENSIONS = {
    'medical': {'.edf', '.csv', '.mat', '.hea', '.dat'},
    'acoustic': {'.wav', '.mp3'},
    'stock': {'.csv', '.xlsx'},
    'microbiome': {'.biom', '.fasta', '.tsv'}
}

def safe_convert_to_list(data):
    """Safely convert numpy array to list"""
    if isinstance(data, np.ndarray):
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        return data.tolist()
    return data

def process_medical_file(filepath, filename):
    """Process medical files: .edf, .csv, .mat, .hea (WFDB), .dat (WFDB)"""
    ext = os.path.splitext(filename)[1].lower()
    
    try:
        # CSV files
        if ext == '.csv':
            df = pd.read_csv(filepath)
            
            # Find date column
            date_col = None
            for i, col in enumerate(df.columns):
                if 'date' in str(col).lower() or 'time' in str(col).lower():
                    date_col = i
                    break
            
            if date_col is not None:
                # Stock-like data with dates
                date_labels = df.iloc[:, date_col].astype(str).tolist()
                time_data = np.arange(len(df))
                
                # Get numeric columns as channels
                channel_data = []
                channel_names = []
                for i, col in enumerate(df.columns):
                    if i != date_col and pd.api.types.is_numeric_dtype(df[col]):
                        channel_data.append(df[col].values)
                        channel_names.append(str(col))
                
                return {
                    'data': safe_convert_to_list(np.array(channel_data)),
                    'time': safe_convert_to_list(time_data),
                    'date_labels': date_labels,
                    'channels': len(channel_data),
                    'channel_names': channel_names,
                    'filename': filename,
                    'type': 'medical',
                    'fs': 1
                }
            else:
                # Regular medical data
                # Assume first column is time
                time_data = df.iloc[:, 0].values
                channel_data = []
                channel_names = []
                
                for i in range(1, df.shape[1]):
                    channel_data.append(df.iloc[:, i].values)
                    channel_names.append(str(df.columns[i]))
                
                # Calculate sampling rate
                fs = 250
                if len(time_data) > 1:
                    time_diff = np.diff(time_data)
                    time_diff = time_diff[~np.isnan(time_diff)]
                    if len(time_diff) > 0 and np.mean(time_diff) > 0:
                        fs = int(1 / np.mean(time_diff))
                
                return {
                    'data': safe_convert_to_list(np.array(channel_data)),
                    'time': safe_convert_to_list(time_data),
                    'channels': len(channel_data),
                    'channel_names': channel_names,
                    'filename': filename,
                    'type': 'medical',
                    'fs': fs
                }
        
        # WFDB files (.hea or .dat)
        elif ext in ['.hea', '.dat']:
            if not WFDB_AVAILABLE:
                return {'error': 'wfdb library not installed. Install with: pip install wfdb'}
            
            try:
                # For WFDB files, we need both .hea and .dat files
                # First, try to find the companion file in the same directory
                base_path = filepath.replace('.hea', '').replace('.dat', '')
                dir_path = os.path.dirname(filepath)
                base_name = os.path.basename(base_path)
                
                # Check if both files exist in the temp directory
                hea_file = os.path.join(dir_path, base_name + '.hea')
                dat_file = os.path.join(dir_path, base_name + '.dat')
                
                # Try to read WFDB file from temp directory if both exist
                files_missing = []
                if not os.path.exists(hea_file):
                    files_missing.append('.hea')
                if not os.path.exists(dat_file):
                    files_missing.append('.dat')
                
                if files_missing:
                    # For browser uploads, user must upload both files together
                    missing_str = ' and '.join(files_missing)
                    return {
                        'error': f'WFDB files require both .hea and .dat files. Missing: {missing_str}. '
                                 f'Please upload both files (header and data files) together.'
                    }
                
                # Read WFDB record
                record = wfdb.rdrecord(base_path)
                
                # Extract signal data
                signal_data = record.p_signal.T.tolist()  # Transpose to channel-first format
                fs = record.fs
                time = np.arange(record.p_signal.shape[0]) / fs
                channel_names = record.sig_name
                
                return {
                    'data': signal_data,
                    'time': time.tolist(),
                    'channels': len(signal_data),
                    'channel_names': channel_names,
                    'filename': filename,
                    'type': 'medical',
                    'fs': fs
                }
            except Exception as e:
                return {'error': f'Error reading WFDB file: {str(e)}'}
        
        # MATLAB files
        elif ext == '.mat':
            import scipy.io as sio
            mat_data = sio.loadmat(filepath)
            
            # Find the largest array that's not metadata
            for key in mat_data:
                if not key.startswith('__') and isinstance(mat_data[key], np.ndarray):
                    data = mat_data[key]
                    if data.size > 1000:  # Likely the signal
                        if data.ndim == 1:
                            signal_data = [data.tolist()]
                            time = np.arange(len(data))
                        else:
                            signal_data = data.T.tolist()
                            time = np.arange(data.shape[0])
                        
                        return {
                            'data': signal_data,
                            'time': time.tolist(),
                            'channels': len(signal_data),
                            'channel_names': [f'Channel {i+1}' for i in range(len(signal_data))],
                            'filename': filename,
                            'type': 'medical',
                            'fs': 250
                        }
        
        # EDF files
        elif ext == '.edf':
            import pyedflib
            f = pyedflib.EdfReader(filepath)
            n_channels = f.signals_in_file
            signal_data = []
            for i in range(min(n_channels, 16)):  # Limit to 16 channels
                signal = f.readSignal(i)
                signal_data.append(signal.tolist())
            fs = f.getSampleFrequency(0)
            time = np.arange(len(signal_data[0])) / fs
            channel_names = f.getSignalLabels()[:16]
            
            f.close()
            
            return {
                'data': signal_data,
                'time': time.tolist(),
                'channels': len(signal_data),
                'channel_names': channel_names,
                'filename': filename,
                'type': 'medical',
                'fs': fs
            }
    
    except Exception as e:
        print(f"Error processing medical file: {str(e)}")
        return {'error': str(e)}
    
    return {'error': 'Unsupported file format'}

def process_acoustic_file(filepath, filename):
    """Process audio files: .wav, .mp3"""
    ext = os.path.splitext(filename)[1].lower()
    
    try:
        # WAV files
        if ext == '.wav':
            import wave
            with wave.open(filepath, 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                channels = wav_file.getnchannels()
                
                # Read audio data
                audio_data = wav_file.readframes(frames)
                
                # Convert to numpy array
                if wav_file.getsampwidth() == 2:  # 16-bit
                    fmt = f"{frames * channels}h"
                    audio_array = np.array(struct.unpack(fmt, audio_data)).astype(np.float32)
                else:  # 8-bit
                    audio_array = np.frombuffer(audio_data, dtype=np.uint8).astype(np.float32)
                
                audio_array = audio_array.reshape(-1, channels).T
                time = np.arange(audio_array.shape[1]) / rate
                
                # Limit to first 30 seconds if too long
                max_samples = min(audio_array.shape[1], rate * 30)
                audio_array = audio_array[:, :max_samples]
                time = time[:max_samples]
                
                return {
                    'data': safe_convert_to_list(audio_array),
                    'time': safe_convert_to_list(time),
                    'channels': channels,
                    'channel_names': [f'Channel {i+1}' for i in range(channels)],
                    'filename': filename,
                    'type': 'acoustic',
                    'fs': rate
                }
        
        # MP3 files
        elif ext == '.mp3':
            from pydub import AudioSegment
            import io
            
            # Load MP3
            audio = AudioSegment.from_mp3(filepath)
            
            # Convert to numpy array
            samples = np.array(audio.get_array_of_samples())
            rate = audio.frame_rate
            channels = audio.channels
            
            if channels > 1:
                samples = samples.reshape(-1, channels).T
            else:
                samples = samples.reshape(1, -1)
            
            # Normalize to float between -1 and 1
            samples = samples.astype(np.float32) / 32768.0
            
            # Limit to first 30 seconds
            max_samples = min(samples.shape[1], rate * 30)
            samples = samples[:, :max_samples]
            time = np.arange(samples.shape[1]) / rate
            
            return {
                'data': safe_convert_to_list(samples),
                'time': safe_convert_to_list(time),
                'channels': channels,
                'channel_names': [f'Channel {i+1}' for i in range(channels)],
                'filename': filename,
                'type': 'acoustic',
                'fs': rate
            }
    
    except Exception as e:
        print(f"Error processing acoustic file: {str(e)}")
        return {'error': str(e)}
    
    return {'error': 'Unsupported file format'}

def process_stock_file(filepath, filename):
    """Process stock files: .csv, .xlsx"""
    ext = os.path.splitext(filename)[1].lower()
    
    try:
        # Read file
        if ext == '.csv':
            df = pd.read_csv(filepath)
        elif ext == '.xlsx':
            df = pd.read_excel(filepath)
        else:
            return {'error': 'Unsupported format'}
        
        # Find date column
        date_col = None
        for i, col in enumerate(df.columns):
            if 'date' in str(col).lower() or 'time' in str(col).lower():
                date_col = i
                break
        
        if date_col is not None:
            date_labels = df.iloc[:, date_col].astype(str).tolist()
            time_data = np.arange(len(df))
        else:
            date_labels = [str(i) for i in range(len(df))]
            time_data = np.arange(len(df))
        
        # Get numeric columns as channels
        channel_data = []
        channel_names = []
        
        for i, col in enumerate(df.columns):
            if i != date_col and pd.api.types.is_numeric_dtype(df[col]):
                channel_data.append(df[col].values)
                channel_names.append(str(col))
        
        if not channel_data:
            return {'error': 'No numeric columns found'}
        
        return {
            'data': safe_convert_to_list(np.array(channel_data)),
            'time': safe_convert_to_list(time_data),
            'date_labels': date_labels,
            'channels': len(channel_data),
            'channel_names': channel_names,
            'filename': filename,
            'type': 'stock',
            'fs': 1,
            'time_unit': 'days'
        }
    
    except Exception as e:
        print(f"Error processing stock file: {str(e)}")
        return {'error': str(e)}

def process_microbiome_file(filepath, filename):
    """Process microbiome files: .biom, .fasta, .tsv"""
    ext = os.path.splitext(filename)[1].lower()
    
    try:
        # TSV files
        if ext == '.tsv':
            df = pd.read_csv(filepath, sep='\t')
            
            # Assume first column is sample IDs
            sample_ids = df.iloc[:, 0].astype(str).tolist()
            
            # Get numeric columns as features
            feature_data = []
            feature_names = []
            
            for i, col in enumerate(df.columns):
                if i > 0 and pd.api.types.is_numeric_dtype(df[col]):
                    feature_data.append(df[col].values)
                    feature_names.append(str(col))
            
            return {
                'data': safe_convert_to_list(np.array(feature_data)),
                'sample_ids': sample_ids,
                'feature_names': feature_names,
                'channels': len(feature_data),
                'samples': len(sample_ids),
                'filename': filename,
                'type': 'microbiome'
            }
        
        # BIOM files
        elif ext == '.biom':
            import biom
            table = biom.load_table(filepath)
            data = table.matrix_data.toarray()
            sample_ids = table.ids('sample').tolist()
            obs_ids = table.ids('observation').tolist()
            
            return {
                'data': data.tolist(),
                'sample_ids': sample_ids,
                'observation_ids': obs_ids,
                'channels': data.shape[0],
                'samples': data.shape[1],
                'filename': filename,
                'type': 'microbiome'
            }
        
        # FASTA files
        elif ext == '.fasta':
            from Bio import SeqIO
            sequences = []
            ids = []
            
            for record in SeqIO.parse(filepath, "fasta"):
                sequences.append(str(record.seq))
                ids.append(record.id)
                if len(sequences) >= 100:  # Limit to 100 sequences
                    break
            
            # Create simple features: length and GC content
            feature_data = []
            for seq in sequences:
                length = len(seq)
                gc_count = seq.count('G') + seq.count('C')
                gc_content = gc_count / length if length > 0 else 0
                feature_data.append([length, gc_content])
            
            return {
                'data': safe_convert_to_list(np.array(feature_data).T),
                'sequence_ids': ids,
                'feature_names': ['Length', 'GC_Content'],
                'channels': 2,
                'samples': len(sequences),
                'filename': filename,
                'type': 'microbiome'
            }
    
    except Exception as e:
        print(f"Error processing microbiome file: {str(e)}")
        return {'error': str(e)}
    
    return {'error': 'Unsupported file format'}

@upload_bp.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload (single or multiple files for WFDB)"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        signal_type = request.form.get('type', 'medical')
        
        # Handle both single and multiple file uploads
        files = request.files.getlist('file')
        if not files or files[0].filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Create temp directory for all files
        temp_dir = tempfile.mkdtemp()
        saved_files = []
        
        # Save all files to the same temp directory
        for file in files:
            if file.filename == '':
                continue
            
            # Check extension
            ext = os.path.splitext(file.filename)[1].lower()
            if ext not in ALLOWED_EXTENSIONS.get(signal_type, set()):
                return jsonify({'error': f'Invalid file type. Expected: {ALLOWED_EXTENSIONS[signal_type]}'}), 400
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(temp_dir, filename)
            file.save(filepath)
            saved_files.append((filepath, filename))
        
        if not saved_files:
            return jsonify({'error': 'No valid files provided'}), 400
        
        # Process based on type (use first file as the primary file)
        processors = {
            'medical': process_medical_file,
            'acoustic': process_acoustic_file,
            'stock': process_stock_file,
            'microbiome': process_microbiome_file
        }
        
        processor = processors.get(signal_type)
        if not processor:
            return jsonify({'error': 'Invalid signal type'}), 400
        
        # Process the primary file (first file or .hea file for WFDB)
        primary_file = saved_files[0]
        
        # For WFDB, prefer processing the .hea file
        if signal_type == 'medical':
            hea_file = next((f for f in saved_files if f[1].endswith('.hea')), None)
            if hea_file:
                primary_file = hea_file
        
        result = processor(primary_file[0], primary_file[1])
        
        # Clean up
        try:
            for filepath, _ in saved_files:
                os.remove(filepath)
            os.rmdir(temp_dir)
        except:
            pass
        
        if result and 'error' not in result:
            return jsonify(result)
        else:
            return jsonify({'error': result.get('error', 'Processing failed')}), 400
            
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@upload_bp.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Upload routes are working!"})
