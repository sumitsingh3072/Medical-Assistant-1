import React, { useEffect, useState } from 'react';
import ReportCard from './ReportCard';

const ResultsPage = () => {
  const [rawResults, setRawResults] = useState(null);
  const [reportData, setReportData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch('http://127.0.0.1:8000/get_latest_results/');
        if (!res.ok) throw new Error('Network response was not ok');
        const data = await res.json();
        setRawResults(data);

        // Transform rawResults into reportData shape
        const entries = Object.entries(data);
        if (entries.length === 0) {
          setReportData({
            symptoms: [],
            report: 'No conditions detected.',
            confidence: 0
          });
          return;
        }

        // Sort by descending score
        const sorted = entries.sort((a, b) => b[1] - a[1]);
        const topK = sorted.slice(0, 3);

        // symptoms = list of the top-3 conditions
        const symptoms = topK.map(([cond]) => cond);

        // report text summarizing the top condition
        const [bestCond, bestScore] = sorted[0];
        const report = `Most likely condition is ${bestCond}.`;

        // confidence as a whole number percentage
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
    return <div className="p-6 text-center text-2xl text-red-600 opacity-40 min-h-screen">{error}😢</div>;
  }
  if (!reportData) {
    return <div className="p-6 text-center text-slate-500 min-h-screen">Loading report...</div>;
  }

  return (
    <div className="p-6 min-h-screen">
      <ReportCard reportData={reportData} />
    </div>
  );
};

export default ResultsPage;
