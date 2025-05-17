import React, { useState } from 'react';
import { Input } from './components/ui/input';
import { Button } from './components/ui/button';
import { Card, CardHeader, CardContent, CardTitle } from './components/ui/card';
import axios from 'axios';

const BASE_API_URL = 'http://127.0.0.1:8000';

const DoctorSearchPage = () => {
  const [location, setLocation] = useState('');
  const [doctors, setDoctors] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleSearch = async () => {
    if (!location) return;
    setLoading(true);
    try {
      const res = await axios.get(`${BASE_API_URL}/search-doctor/`, {
        params: { location }
      });
      setDoctors(res.data.results || []);
    } catch (err) {
      console.error('Error fetching doctors:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-3xl mx-auto space-y-4 ">
      <Card>
        <CardHeader>
          <CardTitle>Find a Nearby Doctor</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="flex gap-2">
            <Input
              placeholder="Enter your location (e.g. Delhi)"
              value={location}
              onChange={(e) => setLocation(e.target.value)}
            />
            <Button onClick={handleSearch} disabled={loading}>
              {loading ? 'Searching...' : 'Search'}
            </Button>
          </div>
        </CardContent>
      </Card>

      {doctors.length > 0 && (
        <div className="space-y-3">
          {doctors.map((doc, idx) => (
            <Card key={idx} className="p-4">
              <h4 className="font-semibold text-lg">{doc.name}</h4>
              <p className="text-sm text-muted-foreground">{doc.specialty} - {doc.location}</p>
              <a
                href={`tel:${doc.phone}`}
                className="mt-2 inline-block text-blue-600 hover:underline"
              >
                Call: {doc.phone}
              </a>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
};

export default DoctorSearchPage;
