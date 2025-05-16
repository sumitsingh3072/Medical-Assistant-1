import React, { useState } from 'react';
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from './ui/card';
import { Slider } from './ui/slider';
import { Button } from './ui/button';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from './ui/tooltip';
import { Badge } from './ui/badge';
import { 
  ZoomIn, 
  ZoomOut, 
  RotateCcw, 
  Maximize2, 
  SlidersHorizontal,
  Eye,
  EyeOff,
  Grid,
  Image
} from 'lucide-react';

const SegmentedImageViewer = ({ imageUrl, imageType }) => {
  const [zoom, setZoom] = useState(100);
  const [showOverlay, setShowOverlay] = useState(true);
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);
  
  const handleZoomIn = () => {
    setZoom(prev => Math.min(prev + 10, 200));
  };
  
  const handleZoomOut = () => {
    setZoom(prev => Math.max(prev - 10, 50));
  };
  
  const handleReset = () => {
    setZoom(100);
    setBrightness(100);
    setContrast(100);
  };

  const imageTypeLabel = imageType ? imageType.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase()) : 'Medical Image';
  
  return (
    <Card className="shadow-md w-full">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-xl font-semibold">Analyzed Image</CardTitle>
          <Badge variant="outline" className="bg-blue-100 text-blue-800 border-blue-200">
            {imageTypeLabel}
          </Badge>
        </div>
      </CardHeader>
      
      <CardContent className="flex flex-col items-center">
        <div className="relative h-64 md:h-80 w-full max-w-lg mx-auto mb-4 overflow-hidden rounded-lg border border-slate-200 bg-slate-100">
          {imageUrl ? (
            <div className="relative h-full w-full">
              <img 
                src={imageUrl} 
                alt="Segmented Medical Image" 
                className="absolute top-0 left-0 w-full h-full object-contain transition-transform duration-300"
                style={{ 
                  transform: `scale(${zoom / 100})`,
                  filter: `brightness(${brightness}%) contrast(${contrast}%)`
                }}
              />
              
              {showOverlay && (
                <div className="absolute top-0 left-0 w-full h-full pointer-events-none">
                  {/* Example segmentation overlay - in a real app this would be an SVG or canvas layer */}
                  <svg width="100%" height="100%" viewBox="0 0 100 100" className="text-blue-500 opacity-40">
                    <circle cx="50" cy="50" r="20" fill="none" stroke="currentColor" strokeWidth="1" strokeDasharray="2 2" />
                    <rect x="30" y="30" width="40" height="40" fill="none" stroke="currentColor" strokeWidth="1" />
                    <path d="M20,80 Q50,20 80,80" fill="none" stroke="currentColor" strokeWidth="1" />
                  </svg>
                </div>
              )}
            </div>
          ) : (
            <div className="flex items-center justify-center h-full text-slate-400">
              <div className="flex flex-col items-center">
                <Image className="h-12 w-12 mb-2" />
                <p className="text-sm">Image preview not available</p>
              </div>
            </div>
          )}
        </div>
        
        <div className="w-full max-w-lg flex flex-col gap-4">
          <div className="flex items-center gap-4">
            <TooltipProvider>
              <div className="flex items-center gap-1">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="outline" size="icon" onClick={handleZoomOut} disabled={zoom <= 50}>
                      <ZoomOut className="h-4 w-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Zoom Out</TooltipContent>
                </Tooltip>
                
                <Slider
                  value={[zoom]}
                  min={50}
                  max={200}
                  step={5}
                  onValueChange={(value) => setZoom(value[0])}
                  className="w-24"
                />
                
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button variant="outline" size="icon" onClick={handleZoomIn} disabled={zoom >= 200}>
                      <ZoomIn className="h-4 w-4" />
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent>Zoom In</TooltipContent>
                </Tooltip>
              </div>
              
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="outline" size="icon" onClick={() => setShowOverlay(prev => !prev)}>
                    {showOverlay ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </Button>
                </TooltipTrigger>
                <TooltipContent>{showOverlay ? "Hide Overlay" : "Show Overlay"}</TooltipContent>
              </Tooltip>
              
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="outline" size="icon" onClick={handleReset}>
                    <RotateCcw className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Reset View</TooltipContent>
              </Tooltip>
              
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="outline" size="icon">
                    <Maximize2 className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Full Screen</TooltipContent>
              </Tooltip>
              
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="outline" size="icon">
                    <Grid className="h-4 w-4" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>Grid View</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
          
          <div className="flex items-center gap-2">
            <SlidersHorizontal className="h-4 w-4 text-slate-500" />
            <div className="grid grid-cols-2 gap-4 flex-1">
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <label className="text-xs text-slate-500">Brightness</label>
                  <span className="text-xs font-medium">{brightness}%</span>
                </div>
                <Slider
                  value={[brightness]}
                  min={50}
                  max={150}
                  step={5}
                  onValueChange={(value) => setBrightness(value[0])}
                />
              </div>
              
              <div className="space-y-1">
                <div className="flex items-center justify-between">
                  <label className="text-xs text-slate-500">Contrast</label>
                  <span className="text-xs font-medium">{contrast}%</span>
                </div>
                <Slider
                  value={[contrast]} 
                  min={50}
                  max={150}
                  step={5}
                  onValueChange={(value) => setContrast(value[0])}
                />
              </div>
            </div>
          </div>
        </div>
      </CardContent>
      
      <CardFooter className="pt-0 pb-4 text-xs text-slate-500 justify-center">
        Image processed using advanced AI segmentation algorithms
      </CardFooter>
    </Card>
  );
};

export default SegmentedImageViewer;