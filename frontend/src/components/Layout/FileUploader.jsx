import React, { useState, forwardRef, useImperativeHandle, useRef, useEffect } from 'react';
import axios from 'axios';

const FileUploader = forwardRef(({ onDataLoaded, signalType }, ref) => {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState(null);
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [showPreview, setShowPreview] = useState(false);
  const fileInputRef = useRef(null);

  // Force set multiple attribute on mount
  useEffect(() => {
    if (fileInputRef.current) {
      fileInputRef.current.setAttribute('multiple', 'multiple');
      fileInputRef.current.multiple = true;
      console.log('✓ Multiple attribute set via ref');
    }
  }, []);

  const sizeLimits = {
    medical: 500,
    acoustic: 200,
    stock: 100,
    microbiome: 500
  };

  const handleFileSelect = (e) => {
    console.log('handleFileSelect called');
    const files = e.target.files;
    console.log('Files from input:', files);
    console.log('Number of files:', files ? files.length : 0);
    
    if (!files || files.length === 0) {
      console.log('No files selected');
      return;
    }

    const fileArray = Array.from(files);
    console.log('✓ Selected', fileArray.length, 'files:', fileArray.map(f => `${f.name} (${f.size} bytes)`));
    setSelectedFiles(fileArray);
    setShowPreview(true);
    setError(null);
  };

  const validateFiles = () => {
    if (selectedFiles.length === 0) {
      setError('No files selected');
      return false;
    }

    // Check for WFDB files
    if (selectedFiles.length > 1) {
      const names = selectedFiles.map(f => f.name);
      const hasHea = names.some(f => f.endsWith('.hea'));
      const hasDat = names.some(f => f.endsWith('.dat'));

      if (!hasHea || !hasDat) {
        setError(`Both .hea and .dat required. You have: ${names.join(', ')}`);
        return false;
      }
    }

    // Check sizes
    const maxSize = sizeLimits[signalType] * 1024 * 1024;
    for (let f of selectedFiles) {
      if (f.size > maxSize) {
        setError(`"${f.name}" exceeds limit (${(f.size / 1024 / 1024).toFixed(2)}MB > ${sizeLimits[signalType]}MB)`);
        return false;
      }
    }

    return true;
  };

  const doUpload = async () => {
    if (!validateFiles()) return;

    setError(null);
    setUploading(true);

    const formData = new FormData();
    selectedFiles.forEach(f => formData.append('file', f));
    formData.append('type', signalType);

    try {
      console.log('Uploading:', selectedFiles.map(f => f.name));
      const response = await axios.post('http://localhost:5000/api/upload', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });

      if (response.data.error) {
        setError(response.data.error);
      } else {
        console.log('✓ Upload success! Files:', selectedFiles.map(f => f.name).join(', '));
        onDataLoaded(response.data);
        setSelectedFiles([]);
        setShowPreview(false);
        // Reset file input
        if (fileInputRef.current) {
          fileInputRef.current.value = '';
          console.log('✓ File input cleared, ready for next selection');
        }
      }
    } catch (err) {
      console.error('Upload failed:', err);
      setError(err.response?.data?.error || err.message || 'Upload failed');
    } finally {
      setUploading(false);
    }
  };

  useImperativeHandle(ref, () => ({
    uploadFile: (file) => {
      setSelectedFiles(Array.isArray(file) ? file : [file]);
      setShowPreview(true);
    }
  }));

  return (
    <div className="file-uploader" style={{
      border: '2px dashed #ccc',
      borderRadius: '8px',
      padding: '20px',
      textAlign: 'center',
      marginBottom: '20px',
      backgroundColor: uploading ? '#f5f5f5' : 'white'
    }}>
      <h3 style={{ marginBottom: '15px' }}>
        📤 Upload {signalType.charAt(0).toUpperCase() + signalType.slice(1)}
        {signalType === 'medical' && <div style={{ fontSize: '0.75em', color: '#666', marginTop: '4px' }}>(Ctrl+Click both .hea + .dat for WFDB)</div>}
      </h3>

      <input
        ref={fileInputRef}
        type="file"
        onChange={handleFileSelect}
        accept="*"
        style={{ display: 'none' }}
      />

      <button
        onClick={() => {
          console.log('📁 Select File(s) button clicked');
          if (fileInputRef.current) {
            fileInputRef.current.value = ''; // Reset before opening
            fileInputRef.current.click();
            console.log('✓ File dialog opened (multiple selection enabled)');
          }
        }}
        disabled={uploading}
        style={{
          padding: '12px 24px',
          backgroundColor: '#4CAF50',
          color: 'white',
          border: 'none',
          borderRadius: '4px',
          cursor: uploading ? 'not-allowed' : 'pointer',
          opacity: uploading ? 0.5 : 1,
          fontSize: '1rem',
          fontWeight: 'bold',
          marginBottom: '15px'
        }}
      >
        📁 {uploading ? 'Uploading...' : 'Select File(s)'}
      </button>

      {/* FILES PREVIEW */}
      {showPreview && selectedFiles.length > 0 && (
        <div style={{
          marginBottom: '15px',
          padding: '12px',
          backgroundColor: '#e8f5e9',
          borderRadius: '4px',
          border: '2px solid #4CAF50',
          textAlign: 'left'
        }}>
          <div style={{ marginBottom: '10px', fontWeight: 'bold', color: '#2e7d32', fontSize: '0.95em' }}>
            ✓ Selected {selectedFiles.length} file{selectedFiles.length > 1 ? 's' : ''}:
          </div>
          {selectedFiles.map((file, idx) => (
            <div key={idx} style={{ color: '#1b5e20', fontSize: '0.9em', marginBottom: '5px', paddingLeft: '10px' }}>
              • {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
            </div>
          ))}
          
          <div style={{ marginTop: '12px', display: 'flex', gap: '8px' }}>
            <button
              onClick={doUpload}
              disabled={uploading}
              style={{
                flex: 1,
                padding: '10px',
                backgroundColor: '#2e7d32',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontWeight: 'bold',
                fontSize: '0.9em'
              }}
            >
              {uploading ? '⏳ Uploading...' : '⬆️ Upload'}
            </button>
            <button
              onClick={() => {
                console.log('Clear clicked - resetting selection');
                setSelectedFiles([]);
                setShowPreview(false);
                if (fileInputRef.current) {
                  fileInputRef.current.value = '';
                  console.log('✓ Selection cleared');
                }
              }}
              disabled={uploading}
              style={{
                flex: 1,
                padding: '10px',
                backgroundColor: '#c62828',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontWeight: 'bold',
                fontSize: '0.9em'
              }}
            >
              ✕ Clear
            </button>
          </div>
        </div>
      )}

      {/* ERROR MESSAGE */}
      {error && (
        <div style={{
          color: '#d32f2f',
          backgroundColor: '#ffebee',
          padding: '10px',
          borderRadius: '4px',
          marginBottom: '15px',
          fontSize: '0.9em',
          border: '1px solid #ef5350'
        }}>
          ❌ {error}
        </div>
      )}

      {/* INFO */}
      <div style={{ marginTop: '15px', fontSize: '0.8em', color: '#666', lineHeight: '1.5' }}>
        <strong>Formats:</strong> CSV, EDF, MAT {signalType === 'medical' && '(+ WFDB)'} | Max: {sizeLimits[signalType]}MB
      </div>
    </div>
  );
});

export default FileUploader;
