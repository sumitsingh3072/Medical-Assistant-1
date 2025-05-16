import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import ReportCard from './ReportCard';
import { Button } from './components/ui/button'; // adjust the path if needed

const BASE_API_URL = 'http://127.0.0.1:8000';

const ResultsPage = () => {
  const [rawResults, setRawResults] = useState(null);
  const [reportData, setReportData] = useState(null);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch(`${BASE_API_URL}/get_latest_results/`);
        if (!res.ok) throw new Error('Network response was not ok');
        const data = await res.json();
        setRawResults(data);

        const entries = Object.entries(data);
        if (entries.length === 0) {
          setReportData({
            symptoms: [],
            report: 'No conditions detected.',
            confidence: 0,
          });
          return;
        }

        const sorted = entries.sort((a, b) => b[1] - a[1]);
        const topK = sorted.slice(0, 3);
        const symptoms = topK.map(([cond]) => cond);
        const [bestCond, bestScore] = sorted[0];
        const report = `Most likely condition is ${bestCond}.`;
        const confidence = Math.round(bestScore * 100);

        setReportData({ symptoms, report, confidence });
      } catch (err) {
        console.error(err);
        setError('Failed to load report. Please try again.');
      }
    };

    fetchData();
  }, []);

  if (error) {
    return (
      <div className="p-6 text-center text-2xl text-red-600 opacity-60 min-h-screen">
        {error}😢
      </div>
    );
  }

  if (!reportData) {
    return (
      <div className="p-6 text-center text-slate-500 min-h-screen">
        Loading report...
      </div>
    );
  }

  return (
    <div className="p-6 min-h-screen space-y-6">
      <ReportCard reportData={reportData} />
      <div className="flex justify-center">
        <Button
          className="mt-4"
          onClick={() => navigate('/resultchat')}
        >
          Chat About This Report
        </Button>
      </div>
    </div>
  );
};

export default ResultsPage;
