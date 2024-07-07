import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const App = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [data, setData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (!selectedFile) {
      setPreview(null);
      return;
    }
    const objectUrl = URL.createObjectURL(selectedFile);
    setPreview(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [selectedFile]);

  const onSelectFile = (event) => {
    if (!event.target.files || event.target.files.length === 0) {
      setSelectedFile(null);
      setData(null);
      return;
    }
    setSelectedFile(event.target.files[0]);
    setData(null);
  };

  const uploadFile = async () => {
    if (!selectedFile) return;
    setIsLoading(true);
    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await axios.post('http://localhost:8000/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setData(response.data);
    } catch (error) {
      console.error('Error uploading file:', error);
      setData(null);
    } finally {
      setIsLoading(false);
    }
  };

  const clearData = () => {
    setSelectedFile(null);
    setPreview(null);
    setData(null);
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>Image Upload and Prediction</h1>
        <div className="upload-container">
          <input type="file" onChange={onSelectFile} />
          {selectedFile && <button onClick={uploadFile}>Upload</button>}
          {preview && <img src={preview} alt="Preview" className="preview-image" />}
          {isLoading && <p>Loading...</p>}
          {data && (
            <div className="result">
              <p>Class: {data.class}</p>
              <p>Confidence: {(data.confidence * 100).toFixed(2)}%</p>
              <button onClick={clearData}>Clear</button>
            </div>
          )}
        </div>
      </header>
    </div>
  );
};

export default App;
