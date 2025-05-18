import React from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Brain, Heart, Stethoscope, Eye, Dna, Zap, Bone, Thermometer, BarChart, ScanHeart, AudioWaveform } from 'lucide-react';

const imageTypes = [
  // {
  //   id: 'mri_2d',
  //   name: 'MRI (2D)',
  //   description: '2D MRI scans for quick anatomical overview',
  //   icon: Brain,
  //   color: 'bg-purple-400',
  // },
  {
    id: 'mri_3d',
    name: 'MRI',
    description: '3D MRI scans for volumetric soft tissue analysis',
    icon: Brain,
    color: 'bg-purple-700',
  },
  {
    id: 'xray',
    name: 'X-Ray',
    description: 'Bone fractures, chest screenings, and more',
    icon: Bone,
    color: 'bg-orange-500',
  },
  {
    id: 'ct_2d',
    name: 'CT Scan (2D)',
    description: '2D CT slices for cross-sectional anatomy',
    icon: ScanHeart,
    color: 'bg-indigo-400',
  },
  {
    id: 'ct_3d',
    name: 'CT Scan (tumor)',
    description: '3D reconstructed CT scans for volumetric imaging',
    icon: ScanHeart,
    color: 'bg-indigo-700',
  },
  {
    id: 'ultrasound',
    name: 'Ultrasound',
    description: 'Soft tissue and pregnancy imaging in real-time',
    icon: AudioWaveform,
    color: 'bg-green-500',
  },
];



const ImageTypeSelector = ({ selectedImageType, setSelectedImageType }) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {imageTypes.map((type) => {
        const IconComponent = type.icon;
        const isSelected = selectedImageType === type.id;

        return (
          <Card
            key={type.id}
            onClick={() => setSelectedImageType(type.id)}
            className={`cursor-pointer border-2 transition-all duration-300 hover:shadow-lg ${
              isSelected ? 'border-blue-500 ' : 'border-transparent hover:border-slate-200'
            }`}
          >
            <CardHeader className="pb-2">
              <div className="flex justify-between items-start">
                <div className={`p-3 rounded-lg ${type.color} text-white mb-3`}>
                  <IconComponent className="h-6 w-6" />
                </div>
                {isSelected && (
                  <Badge variant="outline" className="bg-blue-100 text-blue-800 border-blue-200">
                    Selected
                  </Badge>
                )}
              </div>
              <CardTitle className="text-lg">{type.name}</CardTitle>
            </CardHeader>
            <CardContent>
              <CardDescription className="text-sm text-slate-600">
                {type.description}
              </CardDescription>
            </CardContent>
            <CardFooter className="pt-0">
              <p className="text-xs text-slate-500">
                {isSelected ? 'Currently selected' : 'Click to select'}
              </p>
            </CardFooter>
          </Card>
        );
      })}
    </div>
  );
};

export default ImageTypeSelector;
