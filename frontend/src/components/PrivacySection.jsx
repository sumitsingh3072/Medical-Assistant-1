import { ShieldCheck } from "lucide-react";
import { Card } from "./ui/card"

export default function PrivacySection() {
  return (
    <Card className="relative overflow-hidden rounded-2xl bg-gradient-to-br from-blue-800 to-green-900 text-white shadow-xl">
      <div className="grid grid-cols-1 md:grid-cols-2 items-center p-8 gap-6 bg-[url('/grid.svg')] bg-cover bg-opacity-10">
        {/* Left Section */}
        <div className="flex flex-col items-start gap-4">
          <ShieldCheck className="w-10 h-10 text-white" />
          <h2 className="text-3xl font-semibold leading-tight">
            Your Privacy is<br />Our Priority
          </h2>
        </div>

        {/* Right Section */}
        <div className="flex flex-col items-center md:items-end text-sm text-white/90 space-y-4">
          {/* <div className="flex items-center gap-6">
            <img
              src="/hipaa.png"
              alt="HIPAA Compliant"
              width="64"
              height="64"
              className="object-contain"
            />
            <img
              src="/gdpr.png"
              alt="GDPR Compliant"
              width="64"
              height="64"
              className="object-contain"
            />
          </div> */}
          <p className="max-w-md text-center md:text-right leading-relaxed">
            MediVision AI never requires personally identifiable information to assist you.
            Your health data remains secure, protected by industry-leading encryption,
            and fully compliant with HIPAA and GDPR standards.
          </p>
        </div>
      </div>
    </Card>
  );
}
