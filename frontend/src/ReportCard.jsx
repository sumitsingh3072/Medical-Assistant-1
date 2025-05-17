import React, { useRef, useState } from 'react'
import { pdf } from '@react-pdf/renderer';
import { saveAs } from 'file-saver';
import ReportPDF from "./components/ReportPDF"

import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
  CardFooter
} from './components/ui/card'
import { Badge } from './components/ui/badge'
import { Tabs, TabsList, TabsTrigger, TabsContent } from "./components/ui/tabs"
import { Button } from './components/ui/button'
import { Download, Printer, Share2, Check, AlertTriangle, FileText } from 'lucide-react'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from './components/ui/dialog'

const ReportCard = ({ reportData }) => {
  const { symptoms = [], report = '', confidence = 0 } = reportData || {}
  const cardRef = useRef(null)
  const [openDialog, setOpenDialog] = useState(false)

  const getConfidenceColor = (score) => {
    if (score >= 85) return 'text-green-600'
    if (score >= 70) return 'text-amber-600'
    return 'text-red-600'
  }

  const getConfidenceBadge = (score) => {
    if (score >= 85) return 'bg-green-100 text-green-800 border-green-200'
    if (score >= 70) return 'bg-amber-100 text-amber-800 border-amber-200'
    return 'bg-red-100 text-red-800 border-red-200'
  }

  const handleDownload = async () => {
    const blob = await pdf(<ReportPDF reportData={reportData} />).toBlob();
    saveAs(blob, 'diagnostic-report.pdf');
  };

  const handlePrint = () => window.print()

  const handleShare = () => {
    if (navigator.share) {
      navigator.share({
        title: 'Diagnostic Report',
        text: `Diagnostic Report with ${confidence}% confidence`,
        url: window.location.href
      })
    } else {
      alert('Sharing not supported in this browser.')
    }
  }

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
          Analysis completed {new Date().toLocaleDateString()} at {new Date().toLocaleTimeString()}
        </CardDescription>
      </CardHeader>

      <Tabs defaultValue="summary" className="w-full">
        <div className="px-6">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="summary">Summary</TabsTrigger>
            <TabsTrigger value="findings">Findings</TabsTrigger>
            <TabsTrigger value="recommendations">Recommendations</TabsTrigger>
          </TabsList>
        </div>

        <TabsContent value="summary" className="pt-3">
          <CardContent>
            <div className="mb-4">
              <h4 className="font-medium text-sm text-slate-700 mb-2">Analysis Summary</h4>
              <p className="text-sm text-slate-600 leading-relaxed">{report}</p>
            </div>
            <div>
              <h4 className="font-medium text-sm text-slate-700 mb-2">AI Confidence Assessment</h4>
              <div className="flex items-center gap-2 mb-1">
                <div className={`font-medium ${getConfidenceColor(confidence)}`}>{confidence}%</div>
                {confidence >= 85
                  ? <Check className="h-4 w-4 text-green-600" />
                  : <AlertTriangle className="h-4 w-4 text-amber-600" />}
              </div>
              <p className="text-xs text-slate-500">
                {confidence >= 85
                  ? 'High confidence in analysis results.'
                  : confidence >= 70
                    ? 'Moderate confidence, consider secondary review.'
                    : 'Low confidence, requires specialist review.'}
              </p>
            </div>
          </CardContent>
        </TabsContent>

        <TabsContent value="findings" className="pt-3">
          <CardContent>
            <h4 className="font-medium text-sm text-slate-700 mb-2">Detected Findings</h4>
            <ul className="space-y-2">
              {symptoms.map((symptom, index) => (
                <li key={index} className="flex items-start gap-2">
                  <div className="mt-0.5 rounded-full bg-blue-100 p-1">
                    <div className="h-1.5 w-1.5 rounded-full bg-blue-600"></div>
                  </div>
                  <span className="text-sm text-slate-600">{symptom}</span>
                </li>
              ))}
            </ul>
          </CardContent>
        </TabsContent>

        <TabsContent value="recommendations" className="pt-3">
          <CardContent>
            <h4 className="font-medium text-sm text-slate-700 mb-2">Clinical Recommendations</h4>
            <p className="text-sm text-slate-600 mb-4 leading-relaxed">
              Based on the analysis results, the following clinical actions are recommended:
            </p>
            <ul className="space-y-2">
              {[
                'Schedule follow-up screening in 3 months',
                'Monitor for changes in symptoms',
                'Consider consultation with specialist if symptoms persist'
              ].map((rec, i) => (
                <li key={i} className="flex items-start gap-2">
                  <div className="mt-0.5 rounded-full bg-green-100 p-1">
                    <Check className="h-3 w-3 text-green-600" />
                  </div>
                  <span className="text-sm text-slate-600">{rec}</span>
                </li>
              ))}
            </ul>
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
        <Dialog open={openDialog} onOpenChange={setOpenDialog}>
          <DialogTrigger asChild>
            <Button variant="ghost" className="ml-auto">
              <FileText className="mr-2 h-4 w-4" /> View Full Analysis
            </Button>
          </DialogTrigger>
          <DialogContent className="max-w-xl">
            <DialogHeader>
              <DialogTitle>Full Diagnostic Report</DialogTitle>
            </DialogHeader>
            <div className="space-y-3">
              <div>
                <p className="text-sm font-semibold">Summary:</p>
                <p className="text-sm text-slate-600">{report}</p>
              </div>
              <div>
                <p className="text-sm font-semibold">Symptoms:</p>
                <ul className="list-disc list-inside text-sm text-slate-600">
                  {symptoms.map((s, i) => <li key={i}>{s}</li>)}
                </ul>
              </div>
              <div>
                <p className="text-sm font-semibold">Confidence:</p>
                <p className="text-sm text-slate-600">{confidence}%</p>
              </div>
              <div>
                <p className="text-sm font-semibold">Date:</p>
                <p className="text-sm text-slate-600">{new Date().toLocaleString()}</p>
              </div>
            </div>
          </DialogContent>
        </Dialog>
      </CardFooter>
    </Card>
  )
}

export default ReportCard
