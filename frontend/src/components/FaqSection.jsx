'use client';

import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "./ui/accordion"

const faqs = [
  {
    question: 'How does the diagnosis system work?',
    answer:
      'Our system analyzes medical images using AI models trained on large datasets. It then returns the most probable conditions with a confidence score.',
  },
  {
    question: 'Is my data secure?',
    answer:
      'Yes. Your uploaded images and data are stored securely and are not shared with third parties.',
  },
  {
    question: 'Can I talk to a doctor through this platform?',
    answer:
      'Currently, this platform only provides AI-generated insights. However, we recommend consulting a healthcare professional for a final opinion.',
  },
  {
    question: 'What image types are supported?',
    answer:
      'We support common formats like JPG, PNG, and DICOM. Make sure your image is clear and well-lit for best results.',
  },
  {
    question: 'Is this a free service?',
    answer:
      'Yes, all core functionalities are free to use during the beta phase.',
  },
];

const FaqSection = () => {
  return (
    <Accordion type="multiple" className="w-full max-w-3xl mx-auto space-y-2">
      {faqs.map((faq, idx) => (
        <AccordionItem key={idx} value={`faq-${idx}`}>
          <AccordionTrigger className="text-left text-lg font-medium">
            {faq.question}
          </AccordionTrigger>
          <AccordionContent className="text-muted-foreground">
            {faq.answer}
          </AccordionContent>
        </AccordionItem>
      ))}
    </Accordion>
  );
};

export default FaqSection;
