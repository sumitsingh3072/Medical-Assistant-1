import React, { useEffect, useState } from 'react';
import ReportCard from './ReportCard';

const BASE_API_URL = 'http://127.0.0.1:8000';

const ResultsPage = () => {
  const [reportData, setReportData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await fetch(`${BASE_API_URL}/get_latest_results/`);
        if (!res.ok) throw new Error('Network response was not ok');
        const data = await res.json();

        const entries = Object.entries(data || {});
        if (entries.length === 0) {
          setReportData({
            symptoms: [],
            diagnosis: 'No conditions detected.',
            confidence: 0,
            recommendations: ['Maintain regular checkups'],
            suggested_tests: [],
            specialty: 'General Physician',
            timestamp: new Date().toISOString(),
          });
        } else {
          const sorted = entries.sort((a, b) => b[1] - a[1]);
          const topK = sorted.slice(0, 3);
          const symptoms = topK.map(([cond]) => cond);
          const [bestCond, bestScore] = sorted[0];

          // Simple specialty matching logic (can be refined)
          const specialtyMap = {
            Diabetes: 'Endocrinologist',
            Pneumonia: 'Pulmonologist',
            Depression: 'Psychiatrist',
            'Heart Disease': 'Cardiologist',
          };

          const specialty = specialtyMap[bestCond] || 'General Physician';

          setReportData({
            symptoms,
            diagnosis: bestCond,
            confidence: Math.round(bestScore * 100),
            recommendations: [
              `Consult a ${specialty}`,
              'Follow a healthy lifestyle',
              'Get relevant tests done'
            ],
            suggested_tests: ['Blood Test', 'Imaging', 'Consultation'],
            specialty,
            timestamp: new Date().toISOString(),
          });
        }
      } catch (err) {
        console.error(err);
        setError('Failed to load report. Please try again.');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="p-6 text-center text-slate-500 min-h-screen">
        Loading report...
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 text-center text-2xl text-red-600 opacity-60 min-h-screen">
        {error} 😢
      </div>
    );
  }

  return (
    <div className="p-6 min-h-screen space-y-6 max-w-4xl mx-auto">

      <div id="report-content">
        <ReportCard report={reportData} />
      </div>
    </div>
  );
};

export default ResultsPage;
