import React, { useEffect, useState } from "react";
import axios from "axios";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import { Input } from "./components/ui/input";
import { Button } from "./components/ui/button";
import {
  MapPin,
  UserSearch,
  Phone,
  Mail,
  Loader2,
  Map as MapIcon,
} from "lucide-react";

const DoctorSearchPage = ({ defaultSpecialty = "" }) => {
  const [location, setLocation] = useState("");
  const [specialty, setSpecialty] = useState(defaultSpecialty);
  const [doctors, setDoctors] = useState([]);
  const [coords, setCoords] = useState(null);
  const [loading, setLoading] = useState(false);
  const [geoError, setGeoError] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const doctorsPerPage = 6;

  const paginatedDoctors = doctors.slice(
    (currentPage - 1) * doctorsPerPage,
    currentPage * doctorsPerPage
  );

  // Try get current location on mount
  useEffect(() => {
    navigator.geolocation.getCurrentPosition(
      async (pos) => {
        const { latitude, longitude } = pos.coords;
        setCoords({ lat: latitude, lng: longitude });

        try {
          const res = await axios.get(
            "https://nominatim.openstreetmap.org/reverse",
            {
              params: {
                lat: latitude,
                lon: longitude,
                format: "json",
              },
            }
          );

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
        return {
          lat: parseFloat(res.data[0].lat),
          lng: parseFloat(res.data[0].lon),
        };
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
    <div className="p-6 space-y-6 bg-white dark:bg-black rounded-lg shadow-lg">
      <h1 className="text-3xl font-semibold text-gray-900 dark:text-gray-100 mb-4 flex items-center gap-2">
        <UserSearch size={28} /> Find Doctors Near You
      </h1>

      <div className="flex flex-wrap gap-4 items-center">
        <div className="relative w-full sm:w-1/3">
          <MapPin className="absolute top-1/2 left-3 -translate-y-1/2 text-gray-400" size={20} />
          <Input
            placeholder="Enter location"
            value={location}
            onChange={(e) => setLocation(e.target.value)}
            className="pl-10"
            disabled={loading}
          />
        </div>

        <div className="relative w-full sm:w-1/3">
          <UserSearch className="absolute top-1/2 left-3 -translate-y-1/2 text-gray-400" size={20} />
          <Input
            placeholder="Specialty (e.g. Cardiologist)"
            value={specialty}
            onChange={(e) => setSpecialty(e.target.value)}
            className="pl-10"
            disabled={loading}
          />
        </div>

        <Button
          onClick={handleSearch}
          disabled={loading}
          className="flex items-center gap-2 px-6 py-2 font-semibold"
        >
          {loading && <Loader2 className="animate-spin" size={18} />}
          Search
        </Button>
      </div>

      {geoError && (
        <p className="text-red-600 font-medium">
          Could not get your location automatically. Please enter manually.
        </p>
      )}

      {/* Map */}
      {coords ? (
        <MapContainer
          key={`${coords.lat}-${coords.lng}`} // remount on coords change
          center={[coords.lat, coords.lng]}
          zoom={13}
          className="h-[400px] w-full rounded-xl shadow-md"
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
        <p className="text-gray-500 italic flex items-center gap-2">
          <MapIcon size={18} /> Map will show here after searching or location detection.
        </p>
      )}

      {/* List of doctors */}
      <div className="grid gap-6 pt-6">
        {!loading && doctors.length === 0 && (
          <p className="text-center text-gray-500 italic">No doctors found.</p>
        )}
        {paginatedDoctors.map((doc, idx) => (
          <div
            key={idx}
            className="border border-gray-300 dark:border-gray-700 p-5 rounded-xl shadow-sm hover:shadow-lg transition-shadow duration-300"
          >
            <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-1 flex items-center gap-2">
              <UserSearch size={20} /> {doc.name}
            </h2>
            <p className="text-gray-600 dark:text-gray-300 mb-1">
              <strong>Specialty: </strong> {doc.specialty}
            </p>
            <p className="text-gray-600 dark:text-gray-300 mb-1 flex items-center gap-2">
              <MapPin size={16} /> {doc.location}
            </p>
            <p className="text-gray-600 dark:text-gray-300 mb-3 flex items-center gap-2">
              <Phone size={16} /> {doc.phone}
            </p>
            <Button
              variant="outline"
              className="flex items-center gap-2 px-4 py-2"
              onClick={() => window.open(`mailto:${doc.email || ""}`, "_blank")}
            >
              <Mail size={16} /> Contact
            </Button>
          </div>
        ))}
        {doctors.length > doctorsPerPage && (
  <div className="flex justify-center gap-2 pt-4">
    <Button disabled={currentPage === 1} onClick={() => setCurrentPage(p => p - 1)}>Prev</Button>
    <Button disabled={currentPage * doctorsPerPage >= doctors.length} onClick={() => setCurrentPage(p => p + 1)}>Next</Button>
  </div>
)}
      </div>
    </div>
  );
};

export default DoctorSearchPage;
