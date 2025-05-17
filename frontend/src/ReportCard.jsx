import React, { useRef, useState } from 'react';
import { pdf } from '@react-pdf/renderer';
import { saveAs } from 'file-saver';
import ReportPDF from './components/ReportPDF';
import { useNavigate } from 'react-router-dom';

import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter
} from './components/ui/card';
import { Badge } from './components/ui/badge';
import { Tabs, TabsList, TabsTrigger, TabsContent } from './components/ui/tabs';
import { Button } from './components/ui/button';
import { Download, Printer, Share2, Check, AlertTriangle, Stethoscope, TestTube2, MessageSquare } from 'lucide-react';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from './components/ui/dialog';
import { InteractiveHoverButton } from './components/magicui/interactive-hover-button';

const ReportCard = ({ report }) => {
  const {
    symptoms = [],
    diagnosis = 'Unknown',
    confidence = 0,
    recommendations = [],
    suggested_tests = [],
    specialty = 'General Physician',
    report: detailedReport = '',
    timestamp = new Date().toISOString()
  } = report || {};

  const navigate = useNavigate();
  const cardRef = useRef(null);
  const [openDialog, setOpenDialog] = useState(false);

  const getConfidenceColor = (score) => {
    if (score >= 85) return 'text-green-600';
    if (score >= 70) return 'text-amber-600';
    return 'text-red-600';
  };

  const getConfidenceBadge = (score) => {
    if (score >= 85) return 'bg-green-100 text-green-800 border-green-200';
    if (score >= 70) return 'bg-amber-100 text-amber-800 border-amber-200';
    return 'bg-red-100 text-red-800 border-red-200';
  };

  const handleDownload = async () => {
    const fullReport = { ...report, report: detailedReport };
    const blob = await pdf(<ReportPDF reportData={fullReport} />).toBlob();
    saveAs(blob, 'diagnostic-report.pdf');
  };

  const handlePrint = () => window.print();

  const handleShare = () => {
    if (navigator.share) {
      navigator.share({
        title: 'Diagnostic Report',
        text: `Diagnosis: ${diagnosis} (Confidence: ${confidence}%)`,
        url: window.location.href
      });
    } else {
      alert('Sharing not supported in this browser.');
    }
  };

  return (
    <Card className="shadow-md w-full min-h-screen print:min-h-0" ref={cardRef}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-xl font-semibold">Diagnostic Report</CardTitle>
          <Badge variant="outline" className={getConfidenceBadge(confidence)}>
            {confidence}% Confidence
          </Badge>
        </div>
        <CardDescription>
          Analysis generated on {new Date(timestamp).toLocaleDateString()} at{' '}
          {new Date(timestamp).toLocaleTimeString()}
        </CardDescription>
      </CardHeader>

      <Tabs defaultValue="summary" className="w-full">
        <div className="px-6">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="summary">Summary</TabsTrigger>
            <TabsTrigger value="findings">Findings</TabsTrigger>
            <TabsTrigger value="recommendations">Recommendations</TabsTrigger>
            <TabsTrigger value="tests">Suggested Tests</TabsTrigger>
            <TabsTrigger value="detailed">Detailed</TabsTrigger>
          </TabsList>
        </div>

        <TabsContent value="summary" className="pt-3">
          <CardContent>
            <div className="mb-4">
              <h4 className="font-medium text-sm text-slate-700 mb-2">Diagnosis</h4>
              <p className="text-base font-medium text-blue-700">{diagnosis}</p>
            </div>

            <div>
              <h4 className="font-medium text-sm text-slate-600 mb-2">AI Confidence Assessment</h4>
              <div className="flex items-center gap-2 mb-1">
                <div className={`font-medium ${getConfidenceColor(confidence)}`}>{confidence}%</div>
                {confidence >= 85
                  ? <Check className="h-4 w-4 text-green-600" />
                  : <AlertTriangle className="h-4 w-4 text-amber-600" />}
              </div>
              <p className="text-xs text-slate-500">
                {confidence >= 85
                  ? 'High confidence in diagnosis.'
                  : confidence >= 70
                    ? 'Moderate confidence. Consider second opinion.'
                    : 'Low confidence. Urged to consult a specialist.'}
              </p>
            </div>

            <div className="mt-4">
              <h4 className="font-medium text-sm text-slate-600 mb-1">Suggested Specialist</h4>
              <div className="flex items-center text-sm text-slate-500 gap-2">
                <Stethoscope className="w-4 h-4 text-emerald-500" />
                {specialty}
              </div>
            </div>
          </CardContent>
        </TabsContent>

        <TabsContent value="findings" className="pt-3">
          <CardContent>
            <h4 className="font-medium text-sm text-slate-700 mb-2">Detected Conditions</h4>
            <ul className="space-y-2">
              {symptoms.length ? symptoms.map((symptom, index) => (
                <li key={index} className="flex items-start gap-2">
                  <div className="mt-0.5 rounded-full bg-blue-100 p-1">
                    <div className="h-1.5 w-1.5 rounded-full bg-blue-600" />
                  </div>
                  <span className="text-sm text-slate-600">{symptom}</span>
                </li>
              )) : (
                <p className="text-sm text-slate-500 italic">No specific conditions detected.</p>
              )}
            </ul>
          </CardContent>
        </TabsContent>

        <TabsContent value="recommendations" className="pt-3">
          <CardContent>
            <h4 className="font-medium text-sm text-slate-600 mb-2">Clinical Recommendations</h4>
            <ul className="space-y-2">
              {recommendations.length ? recommendations.map((rec, i) => (
                <li key={i} className="flex items-start gap-2">
                  <div className="mt-0.5 rounded-full bg-green-100 p-1">
                    <Check className="h-3 w-3 text-green-600" />
                  </div>
                  <span className="text-sm text-slate-500">{rec}</span>
                </li>
              )) : (
                <p className="text-sm text-slate-500 italic">No recommendations provided.</p>
              )}
            </ul>
          </CardContent>
        </TabsContent>

        <TabsContent value="tests" className="pt-3">
          <CardContent>
            <h4 className="font-medium text-sm text-slate-600 mb-2">Suggested Diagnostic Tests</h4>
            <ul className="space-y-2">
              {suggested_tests.length ? suggested_tests.map((test, i) => (
                <li key={i} className="flex items-start gap-2">
                  <TestTube2 className="w-4 h-4 text-indigo-600 mt-0.5" />
                  <span className="text-sm text-slate-500">{test}</span>
                </li>
              )) : (
                <p className="text-sm text-slate-500 italic">No tests suggested.</p>
              )}
            </ul>
          </CardContent>
        </TabsContent>

        <TabsContent value="detailed" className="pt-3">
          <CardContent>
            <h4 className="font-medium text-sm text-slate-400 mb-3">Full Diagnostic Explanation</h4>
            {detailedReport ? (
              <div className="whitespace-pre-line text-sm bg-transparent text-slate-400 leading-relaxed border rounded-md p-4">
                {detailedReport}
              </div>
            ) : (
              <p className="text-sm italic text-slate-500">No detailed explanation available from the AI model.</p>
            )}
          </CardContent>
        </TabsContent>
      </Tabs>

      <CardFooter className="pt-2 pb-4 px-6 flex flex-wrap gap-3 print:hidden">
        <Button onClick={handleDownload} className="bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800">
          <Download className="mr-2 h-4 w-4" /> Download
        </Button>
        <Button variant="outline" onClick={handlePrint}>
          <Printer className="mr-2 h-4 w-4" /> Print
        </Button>
        <Button variant="outline" onClick={handleShare}>
          <Share2 className="mr-2 h-4 w-4" /> Share
        </Button>

        {/* More Info Dialog */}
        <Dialog open={openDialog} onOpenChange={setOpenDialog}>
          <DialogTrigger asChild>
            <Button variant="outline">More Info</Button>
          </DialogTrigger>
          <DialogContent className="max-w-lg">
            <DialogHeader>
              <DialogTitle>Diagnostic Details</DialogTitle>
            </DialogHeader>
            <CardContent className="space-y-3 text-sm text-slate-400">
              <div>
                <strong>Diagnosis:</strong> {diagnosis}
              </div>
              <div>
                <strong>Confidence:</strong> {confidence}%
              </div>
              <div>
                <strong>Suggested Specialist:</strong> {specialty}
              </div>
              <div>
                <strong>Symptoms Detected:</strong>
                {symptoms.length ? (
                  <ul className="list-disc ml-5 mt-1">
                    {symptoms.map((s, i) => <li key={i}>{s}</li>)}
                  </ul>
                ) : (
                  <p className="italic text-slate-500">No symptoms detected.</p>
                )}
              </div>
              <div>
                <strong>Recommendations:</strong>
                {recommendations.length ? (
                  <ul className="list-disc ml-5 mt-1">
                    {recommendations.map((r, i) => <li key={i}>{r}</li>)}
                  </ul>
                ) : (
                  <p className="italic text-slate-500">No recommendations provided.</p>
                )}
              </div>
              <div>
                <strong>Suggested Tests:</strong>
                {suggested_tests.length ? (
                  <ul className="list-disc ml-5 mt-1">
                    {suggested_tests.map((t, i) => <li key={i}>{t}</li>)}
                  </ul>
                ) : (
                  <p className="italic text-slate-500">No suggested tests.</p>
                )}
              </div>
            </CardContent>
          </DialogContent>
        </Dialog>

        {/* Chat with Report Button */}
        <Button
          variant="secondary"
          onClick={() => navigate('/resultchat')}
          className="ml-auto"
        >
          <MessageSquare className="mr-2 h-4 w-4" /> Chat with Report
        </Button>

        {/* Search Doctor Button */}
        <Button
          variant="secondary"
          onClick={() => navigate('/search-doctor')}
        >
          <Stethoscope className="mr-2 h-4 w-4" /> Search Doctor
        </Button>
      </CardFooter>
    </Card>
  );
};

export default ReportCard;
