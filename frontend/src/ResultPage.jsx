import React, { useEffect, useState } from 'react';
import ReportCard from './ReportCard';

// Mock data to simulate API response
const mockReportData = {
  symptoms: ['Persistent cough', 'Shortness of breath', 'Chest pain'],
  report: 'The analysis indicates signs of mild respiratory distress. Patterns are consistent with early bronchial inflammation.',
  confidence: 88
};

const ResultsPage = () => {
  const [reportData, setReportData] = useState(null);

  useEffect(() => {
    // Simulate API call
    const fetchData = async () => {
      // Uncomment and replace with your actual API call:
      // try {
      //   const response = await fetch('/api/report');
      //   const data = await response.json();
      //   setReportData(data);
      // } catch (error) {
      //   console.error('Error fetching report data:', error);
      // }

      // Using mock data for now
      setTimeout(() => {
        setReportData(mockReportData);
      }, 500); // simulate network delay
    };

    fetchData();
  }, []);

  return (
    <div className="p-6">
      {reportData ? (
        <ReportCard reportData={reportData} />
      ) : (
        <div className="text-center text-slate-500">Loading report...</div>
      )}
    </div>
  );
};

export default ResultsPage;
