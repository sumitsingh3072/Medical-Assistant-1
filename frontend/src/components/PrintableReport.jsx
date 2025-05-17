import React from "react";

const PrintableReport = React.forwardRef(({ reportData }, ref) => {
  const { symptoms = [], report = '', confidence = 0 } = reportData || {};

  return (
    <div ref={ref} className="p-6 font-sans text-black w-[210mm] bg-white">
      <div className="mb-6 border-b pb-4">
        <h1 className="text-2xl font-bold text-blue-700">MediVision AI</h1>
        <p className="text-sm text-gray-600">Diagnostic Report</p>
        <p className="text-sm text-gray-500">Generated on {new Date().toLocaleString()}</p>
      </div>

      <section className="mb-6">
        <h2 className="text-lg font-semibold mb-1">Summary</h2>
        <p className="text-sm leading-relaxed">{report}</p>
      </section>

      <section className="mb-6">
        <h2 className="text-lg font-semibold mb-1">Confidence Score</h2>
        <p className="text-sm">{confidence}%</p>
        <p className="text-xs text-gray-600">
          {confidence >= 85 ? "High confidence" :
           confidence >= 70 ? "Moderate confidence" :
           "Low confidence - follow up recommended"}
        </p>
      </section>

      <section className="mb-6">
        <h2 className="text-lg font-semibold mb-1">Detected Symptoms</h2>
        <ul className="list-disc list-inside text-sm">
          {symptoms.length ? symptoms.map((s, i) => <li key={i}>{s}</li>) : <li>No symptoms detected</li>}
        </ul>
      </section>

      <section>
        <h2 className="text-lg font-semibold mb-1">Recommendations</h2>
        <ul className="list-disc list-inside text-sm">
          <li>Schedule follow-up in 3 months</li>
          <li>Monitor any symptom changes</li>
          <li>Consult specialist if symptoms persist</li>
        </ul>
      </section>

      <footer className="mt-10 text-center text-xs text-gray-400 border-t pt-4">
        Powered by MediVision AI • www.medivision.ai
      </footer>
    </div>
  );
});

export default PrintableReport;
