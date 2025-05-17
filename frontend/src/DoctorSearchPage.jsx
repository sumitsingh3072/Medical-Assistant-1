import React, { useEffect, useState } from "react";
import axios from "axios";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { Input } from "./components/ui/input";
import { Button } from "./components/ui/button";

const DoctorSearchPage = ({ defaultSpecialty = "" }) => {
  const [location, setLocation] = useState("");
  const [specialty, setSpecialty] = useState(defaultSpecialty);
  const [doctors, setDoctors] = useState([]);
  const [coords, setCoords] = useState(null);
  const [loading, setLoading] = useState(false);
  const [geoError, setGeoError] = useState(false);

  // Try get current location on mount
  useEffect(() => {
    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        const { latitude, longitude } = pos.coords;
        setCoords({ lat: latitude, lng: longitude });

        try {
          const res = await axios.get("https://nominatim.openstreetmap.org/reverse", {
            params: {
              lat: latitude,
              lon: longitude,
              format: "json",
            },
          });

          const loc =
            res.data.address?.suburb ||
            res.data.address?.city ||
            res.data.address?.village ||
            "";
          setLocation(loc);
        } catch (err) {
          console.error("Error in reverse geocoding:", err);
        }
      },
      () => {
        console.warn("Geolocation permission denied or failed");
        setGeoError(true);
      }
    );
  }, []);

  // Geocode a location string to lat,lng
  const geocodeLocation = async (locStr) => {
    try {
      const res = await axios.get("https://nominatim.openstreetmap.org/search", {
        params: { q: locStr + ", India", format: "json", limit: 1 },
      });
      if (res.data && res.data.length > 0) {
        return { lat: parseFloat(res.data[0].lat), lng: parseFloat(res.data[0].lon) };
      }
    } catch (e) {
      console.error("Geocoding error:", e);
    }
    return null;
  };

  const handleSearch = async () => {
    if (!location) {
      alert("Please enter a location.");
      return;
    }
    setLoading(true);
    setDoctors([]);
    try {
      // Geocode input location to center the map
      const centerCoords = await geocodeLocation(location);
      if (centerCoords) {
        setCoords(centerCoords);
      } else {
        alert("Could not find location on map.");
        setCoords(null);
      }

      // Fetch doctors from backend
      const res = await axios.get("http://127.0.0.1:8000/api/search-doctors", {
        params: { location, specialty },
      });
      setDoctors(res.data);
    } catch (err) {
      console.error("Error fetching doctors:", err);
      alert("Failed to fetch doctors. Please try again.");
      setCoords(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 space-y-4">
      <div className="flex flex-wrap gap-2 items-center">
        <Input
          placeholder="Enter location"
          value={location}
          onChange={(e) => setLocation(e.target.value)}
          className="w-full sm:w-1/3"
          disabled={loading}
        />
        <Input
          placeholder="Specialty (e.g. Cardiologist)"
          value={specialty}
          onChange={(e) => setSpecialty(e.target.value)}
          className="w-full sm:w-1/3"
          disabled={loading}
        />
        <Button onClick={handleSearch} disabled={loading} className="w-full sm:w-auto">
          {loading ? "Loading..." : "Search"}
        </Button>
      </div>

      {geoError && (
        <p className="text-red-600">
          Could not get your location automatically. Please enter manually.
        </p>
      )}

      {/* Map only if coords set */}
      {coords ? (
        <MapContainer
          key={`${coords.lat}-${coords.lng}`} // remount on coords change
          center={[coords.lat, coords.lng]}
          zoom={13}
          className="h-[400px] w-full rounded-xl"
        >
          <TileLayer
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            attribution='&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a>'
          />
          {doctors.map((doc, idx) => {
            const lat = parseFloat(doc.lat);
            const lng = parseFloat(doc.lng);
            if (isNaN(lat) || isNaN(lng)) return null;
            return (
              <Marker key={idx} position={[lat, lng]}>
                <Popup>
                  <strong>{doc.name}</strong>
                  <br />
                  {doc.specialty}
                  <br />
                  {doc.phone}
                </Popup>
              </Marker>
            );
          })}
        </MapContainer>
      ) : (
        <p className="text-gray-500">Map will show here after searching or location detection.</p>
      )}

      {/* List of doctors */}
      <div className="grid gap-4 pt-4">
        {!loading && doctors.length === 0 && <p>No doctors found.</p>}
        {doctors.map((doc, idx) => (
          <div key={idx} className="border p-4 rounded-xl shadow-sm">
            <h2 className="text-lg font-bold">{doc.name}</h2>
            <p>{doc.specialty}</p>
            <p>{doc.location}</p>
            <p>{doc.phone}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default DoctorSearchPage;
