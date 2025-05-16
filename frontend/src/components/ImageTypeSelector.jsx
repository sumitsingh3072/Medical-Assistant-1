import React from 'react';
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';
import { Brain, Heart, Stethoscope, Eye, Dna, Zap, Bone, Thermometer, BarChart } from 'lucide-react';

const imageTypes = [
  {
    id: 'brain_mri',
    name: 'Brain MRI',
    description: 'Neurological conditions, tumors, hemorrhage detection',
    icon: Brain,
    color: 'bg-purple-500',
  },
  {
    id: 'chest_xray',
    name: 'Chest X-Ray',
    description: 'Pneumonia, lung cancer, tuberculosis screening',
    icon: Stethoscope,
    color: 'bg-blue-500',
  },
  {
    id: 'cardiac_mri',
    name: 'Cardiac MRI',
    description: 'Heart structure, function, and tissue characterization',
    icon: Heart,
    color: 'bg-red-500',
  },
  {
    id: 'retinal_scan',
    name: 'Retinal Scan',
    description: 'Diabetic retinopathy, macular degeneration',
    icon: Eye,
    color: 'bg-emerald-500',
  },
  {
    id: 'bone_xray',
    name: 'Bone X-Ray',
    description: 'Fractures, osteoporosis, bone density',
    icon: Bone,
    color: 'bg-amber-500',
  },
  {
    id: 'histopathology',
    name: 'Histopathology',
    description: 'Tissue analysis for cancer and disease',
    icon: Dna,
    color: 'bg-pink-500',
  },
  {
    id: 'ecg',
    name: 'ECG Analysis',
    description: 'Heart rhythm abnormalities, arrhythmias',
    icon: Zap,
    color: 'bg-rose-500',
  },
  {
    id: 'thermography',
    name: 'Thermography',
    description: 'Heat patterns for inflammation detection',
    icon: Thermometer,
    color: 'bg-orange-500',
  },
  {
    id: 'endoscopy',
    name: 'Endoscopy',
    description: 'GI tract analysis for polyps and lesions',
    icon: BarChart,
    color: 'bg-teal-500',
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
