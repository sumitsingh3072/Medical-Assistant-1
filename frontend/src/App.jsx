import React, { useState } from 'react';
import { Routes, Route } from 'react-router-dom';
import LandingPage from './LandingPage';
import UploadPage from './UploadPage';
import ResultPage from './ResultPage';
import Header from './components/Header';
import Footer from './components/Footer';
import Contact from './Contact';
import ResultChatPage from './ResultChatPage';
import DoctorSearchPage from './DoctorSearchPage';

function App() {
  // Global state for selected image type and processed data
  const [selectedImageType, setSelectedImageType] = useState(null);
  const [processedData, setProcessedData] = useState(null);

  return (
    <div className="flex flex-col min-h-screen bg-gray-50 dark:bg-gray-900">
      <Header />
      <main className="flex-grow">
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route 
            path="/upload" 
            element={
              <UploadPage 
                selectedImageType={selectedImageType} 
                setSelectedImageType={setSelectedImageType} 
                setProcessedData={setProcessedData} 
              />
            } 
          />
          <Route 
            path="/results" 
            element={
              <ResultPage 
                processedData={processedData} 
                selectedImageType={selectedImageType} 
              />
            } 
          />
          <Route path="/resultchat" element={<ResultChatPage />} />
          <Route path="/search-doctor" element={<DoctorSearchPage />} />
          <Route path='/contact' element={<Contact/>} />
        </Routes>
      </main>
      <Footer />
    </div>
  );
}

export default App;
