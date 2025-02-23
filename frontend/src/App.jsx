import './App.css'
import 'leaflet/dist/leaflet.css'

import { useEffect, useRef, useState } from 'react'

import L from 'leaflet'

const countryCoordinates = {
  "Andorra": { latitude: 42.546245, longitude: 1.601554 },
  "United Arab Emirates": { latitude: 23.424076, longitude: 53.847818 },
  "Afghanistan": { latitude: 33.93911, longitude: 67.709953 },
  "Antigua and Barbuda": { latitude: 17.060816, longitude: -61.796428 },
  "Anguilla": { latitude: 18.220554, longitude: -63.068615 },
  "Albania": { latitude: 41.153332, longitude: 20.168331 },
  "Armenia": { latitude: 40.069099, longitude: 45.038189 },
  "Netherlands Antilles": { latitude: 12.226079, longitude: -69.060087 },
  "Angola": { latitude: -11.202692, longitude: 17.873887 },
  "Antarctica": { latitude: -75.250973, longitude: -0.071389 },
  "Argentina": { latitude: -38.416097, longitude: -63.616672 },
  "American Samoa": { latitude: -14.270972, longitude: -170.132217 },
  "Austria": { latitude: 47.516231, longitude: 14.550072 },
  "Australia": { latitude: -25.274398, longitude: 133.775136 },
  "Aruba": { latitude: 12.52111, longitude: -69.968338 },
  "Azerbaijan": { latitude: 40.143105, longitude: 47.576927 },
  "Bosnia and Herzegovina": { latitude: 43.915886, longitude: 17.679076 },
  "Barbados": { latitude: 13.193887, longitude: -59.543198 },
  "Bangladesh": { latitude: 23.684994, longitude: 90.356331 },
  "Belgium": { latitude: 50.503887, longitude: 4.469936 },
  "Burkina Faso": { latitude: 12.238333, longitude: -1.561593 },
  "Bulgaria": { latitude: 42.733883, longitude: 25.48583 },
  "Bahrain": { latitude: 25.930414, longitude: 50.637772 },
  "Burundi": { latitude: -3.373056, longitude: 29.918886 },
  "Benin": { latitude: 9.30769, longitude: 2.315834 },
  "Bermuda": { latitude: 32.321384, longitude: -64.75737 },
  "Brunei": { latitude: 4.535277, longitude: 114.727669 },
  "Bolivia": { latitude: -16.290154, longitude: -63.588653 },
  "Brazil": { latitude: -14.235004, longitude: -51.92528 },
  "Bahamas": { latitude: 25.03428, longitude: -77.39628 },
  "Bhutan": { latitude: 27.514162, longitude: 90.433601 },
  "Bouvet Island": { latitude: -54.423199, longitude: 3.413194 },
  "Botswana": { latitude: -22.328474, longitude: 24.684866 },
  "Belarus": { latitude: 53.709807, longitude: 27.953389 },
  "Belize": { latitude: 17.189877, longitude: -88.49765 },
  "Canada": { latitude: 56.130366, longitude: -106.346771 },
  "Cocos [Keeling] Islands": { latitude: -12.164165, longitude: 96.870956 },
  "Congo [DRC]": { latitude: -4.038333, longitude: 21.758664 },
  "Central African Republic": { latitude: 6.611111, longitude: 20.939444 },
  "Congo [Republic]": { latitude: -0.228021, longitude: 15.827659 },
  "Switzerland": { latitude: 46.818188, longitude: 8.227512 },
  "Côte d'Ivoire": { latitude: 7.539989, longitude: -5.54708 },
  "Cook Islands": { latitude: -21.236736, longitude: -159.777671 },
  "Chile": { latitude: -35.675147, longitude: -71.542969 },
  "Cameroon": { latitude: 7.369722, longitude: 12.354722 },
  "China": { latitude: 35.86166, longitude: 104.195397 },
  "Colombia": { latitude: 4.570868, longitude: -74.297333 },
  "Costa Rica": { latitude: 9.748917, longitude: -83.753428 },
  "Cuba": { latitude: 21.521757, longitude: -77.781167 },
  "Cape Verde": { latitude: 16.002082, longitude: -24.013197 },
  "Christmas Island": { latitude: -10.447525, longitude: 105.690449 },
  "Cyprus": { latitude: 35.126413, longitude: 33.429859 },
  "Czech Republic": { latitude: 49.817492, longitude: 15.472962 },
  "Germany": { latitude: 51.165691, longitude: 10.451526 },
  "Djibouti": { latitude: 11.825138, longitude: 42.590275 },
  "Denmark": { latitude: 56.26392, longitude: 9.501785 },
  "Dominica": { latitude: 15.414999, longitude: -61.370976 },
  "Dominican Republic": { latitude: 18.735693, longitude: -70.162651 },
  "Algeria": { latitude: 28.033886, longitude: 1.659626 },
  "Ecuador": { latitude: -1.831239, longitude: -78.183406 },
  "Estonia": { latitude: 58.595272, longitude: 25.013607 },
  "Egypt": { latitude: 26.820553, longitude: 30.802498 },
  "Western Sahara": { latitude: 24.215527, longitude: -12.885834 },
  "Eritrea": { latitude: 15.179384, longitude: 39.782334 },
  "Spain": { latitude: 40.463667, longitude: -3.74922 },
  "Ethiopia": { latitude: 9.145, longitude: 40.489673 },
  "Finland": { latitude: 61.92411, longitude: 25.748151 },
  "Fiji": { latitude: -16.578193, longitude: 179.414413 },
  "Falkland Islands [Islas Malvinas]": { latitude: -51.796253, longitude: -59.523613 },
  "Micronesia": { latitude: 7.425554, longitude: 150.550812 },
  "Faroe Islands": { latitude: 61.892635, longitude: -6.911806 },
  "France": { latitude: 46.227638, longitude: 2.213749 },
  "Gabon": { latitude: -0.803689, longitude: 11.609444 },
  "United Kingdom": { latitude: 55.378051, longitude: -3.435973 },
  "Grenada": { latitude: 12.262776, longitude: -61.604171 },
  "Georgia": { latitude: 42.315407, longitude: 43.356892 },
  "French Guiana": { latitude: 3.933889, longitude: -53.125782 },
  "Guernsey": { latitude: 49.465691, longitude: -2.585278 },
  "Ghana": { latitude: 7.946527, longitude: -1.023194 },
  "Gibraltar": { latitude: 36.137741, longitude: -5.345374 },
  "Greenland": { latitude: 71.706936, longitude: -42.604303 },
  "Gambia": { latitude: 13.443182, longitude: -15.310139 },
  "Guinea": { latitude: 9.945587, longitude: -9.696645 },
  "Guadeloupe": { latitude: 16.995971, longitude: -62.067641 },
  "Equatorial Guinea": { latitude: 1.650801, longitude: 10.267895 },
  "Greece": { latitude: 39.074208, longitude: 21.824312 },
  "South Georgia and the South Sandwich Islands": { latitude: -54.429579, longitude: -36.587909 },
  "Guatemala": { latitude: 15.783471, longitude: -90.230759 },
  "Guam": { latitude: 13.444304, longitude: 144.793731 },
  "Guinea-Bissau": { latitude: 11.803749, longitude: -15.180413 },
  "Guyana": { latitude: 4.860416, longitude: -58.93018 },
  "Gaza Strip": { latitude: 31.354676, longitude: 34.308825 },
  "Hong Kong": { latitude: 22.396428, longitude: 114.109497 },
  "Heard Island and McDonald Islands": { latitude: -53.08181, longitude: 73.504158 },
  "Honduras": { latitude: 15.199999, longitude: -86.241905 },
  "Croatia": { latitude: 45.1, longitude: 15.2 },
  "Haiti": { latitude: 18.971187, longitude: -72.285215 },
  "Hungary": { latitude: 47.162494, longitude: 19.503304 },
  "Indonesia": { latitude: -0.789275, longitude: 113.921327 },
  "Ireland": { latitude: 53.41291, longitude: -8.24389 },
  "Israel": { latitude: 31.046051, longitude: 34.851612 },
  "Isle of Man": { latitude: 54.236107, longitude: -4.548056 },
  "India": { latitude: 20.593684, longitude: 78.96288 },
  "British Indian Ocean Territory": { latitude: -6.343194, longitude: 71.876519 },
  "Iraq": { latitude: 33.223191, longitude: 43.679291 },
  "Iran": { latitude: 32.427908, longitude: 53.688046 },
  "Iceland": { latitude: 64.963051, longitude: -19.020835 },
  "Italy": { latitude: 41.87194, longitude: 12.56738 },
  "Jersey": { latitude: 49.214439, longitude: -2.13125 },
  "Jamaica": { latitude: 18.109581, longitude: -77.297508 },
  "Jordan": { latitude: 30.585164, longitude: 36.238414 },
  "Japan": { latitude: 36.204824, longitude: 138.252924 },
  "Kenya": { latitude: -0.023559, longitude: 37.906193 },
  "Kyrgyzstan": { latitude: 41.20438, longitude: 74.766098 },
  "Cambodia": { latitude: 12.565679, longitude: 104.990963 },
  "Kiribati": { latitude: -3.370417, longitude: -168.734039 },
  "Comoros": { latitude: -11.875001, longitude: 43.872219 },
  "Saint Kitts and Nevis": { latitude: 17.357822, longitude: -62.782998 },
  "North Korea": { latitude: 40.339852, longitude: 127.510093 },
  "South Korea": { latitude: 35.907757, longitude: 127.766922 },
  "Kuwait": { latitude: 29.31166, longitude: 47.481766 },
  "Cayman Islands": { latitude: 19.513469, longitude: -80.566956 },
  "Kazakhstan": { latitude: 48.019573, longitude: 66.923684 },
  "Laos": { latitude: 19.85627, longitude: 102.495496 },
  "Lebanon": { latitude: 33.854721, longitude: 35.862285 },
  "Saint Lucia": { latitude: 13.909444, longitude: -60.978893 },
  "Liechtenstein": { latitude: 47.166, longitude: 9.555373 },
  "Sri Lanka": { latitude: 7.873054, longitude: 80.771797 },
  "Liberia": { latitude: 6.428055, longitude: -9.429499 },
  "Lesotho": { latitude: -29.609988, longitude: 28.233608 },
  "Lithuania": { latitude: 55.169438, longitude: 23.881275 },
  "Luxembourg": { latitude: 49.815273, longitude: 6.129583 },
  "Latvia": { latitude: 56.879635, longitude: 24.603189 },
  "Libya": { latitude: 26.3351, longitude: 17.228331 },
  "Morocco": { latitude: 31.791702, longitude: -7.09262 },
  "Monaco": { latitude: 43.750298, longitude: 7.412841 },
  "Moldova": { latitude: 47.411631, longitude: 28.369885 },
  "Montenegro": { latitude: 42.708678, longitude: 19.37439 },
  "Madagascar": { latitude: -18.766947, longitude: 46.869107 },
  "Marshall Islands": { latitude: 7.131474, longitude: 171.184478 },
  "Macedonia [FYROM]": { latitude: 41.608635, longitude: 21.745275 },
  "Mali": { latitude: 17.570692, longitude: -3.996166 },
  "Myanmar [Burma]": { latitude: 21.913965, longitude: 95.956223 },
  "Mongolia": { latitude: 46.862496, longitude: 103.846656 },
  "Macau": { latitude: 22.198745, longitude: 113.543873 },
  "Northern Mariana Islands": { latitude: 17.33083, longitude: 145.38469 },
  "Martinique": { latitude: 14.641528, longitude: -61.024174 },
  "Mauritania": { latitude: 21.00789, longitude: -10.940835 },
  "Montserrat": { latitude: 16.742498, longitude: -62.187366 },
  "Malta": { latitude: 35.937496, longitude: 14.375416 },
  "Mauritius": { latitude: -20.348404, longitude: 57.552152 },
  "Maldives": { latitude: 3.202778, longitude: 73.22068 },
  "Malawi": { latitude: -13.254308, longitude: 34.301525 },
  "Mexico": { latitude: 23.634501, longitude: -102.552784 },
  "Malaysia": { latitude: 4.210484, longitude: 101.975766 },
  "Mozambique": { latitude: -18.665695, longitude: 35.529562 },
  "Namibia": { latitude: -22.95764, longitude: 18.49041 },
  "New Caledonia": { latitude: -20.904305, longitude: 165.618042 },
  "Niger": { latitude: 17.607789, longitude: 8.081666 },
  "Norfolk Island": { latitude: -29.040835, longitude: 167.954712 },
  "Nigeria": { latitude: 9.081999, longitude: 8.675277 },
  "Nicaragua": { latitude: 12.865416, longitude: -85.207229 },
  "Netherlands": { latitude: 52.132633, longitude: 5.291266 },
  "Norway": { latitude: 60.472024, longitude: 8.468946 },
  "Nepal": { latitude: 28.394857, longitude: 84.124008 },
  "Nauru": { latitude: -0.522778, longitude: 166.931503 },
  "Niue": { latitude: -19.054445, longitude: -169.867233 },
  "New Zealand": { latitude: -40.900557, longitude: 174.885971 },
  "Oman": { latitude: 21.512583, longitude: 55.923255 },
  "Panama": { latitude: 8.537981, longitude: -80.782127 },
  "Peru": { latitude: -9.189967, longitude: -75.015152 },
  "French Polynesia": { latitude: -17.679742, longitude: -149.406843 },
  "Papua New Guinea": { latitude: -6.314993, longitude: 143.95555 },
  "Philippines": { latitude: 12.879721, longitude: 121.774017 },
  "Pakistan": { latitude: 30.375321, longitude: 69.345116 },
  "Poland": { latitude: 51.919438, longitude: 19.145136 },
  "Saint Pierre and Miquelon": { latitude: 46.941936, longitude: -56.27111 },
  "Pitcairn Islands": { latitude: -24.703615, longitude: -127.439308 },
  "Puerto Rico": { latitude: 18.220833, longitude: -66.590149 },
  "Palestinian Territories": { latitude: 31.952162, longitude: 35.233154 },
  "Portugal": { latitude: 39.399872, longitude: -8.224454 },
  "Palau": { latitude: 7.51498, longitude: 134.58252 },
  "Paraguay": { latitude: -23.442503, longitude: -58.443832 },
  "Qatar": { latitude: 25.354826, longitude: 51.183884 },
  "Réunion": { latitude: -21.115141, longitude: 55.536384 },
  "Romania": { latitude: 45.943161, longitude: 24.96676 },
  "Serbia": { latitude: 44.016521, longitude: 21.005859 },
  "Russia": { latitude: 61.52401, longitude: 105.318756 },
  "Rwanda": { latitude: -1.940278, longitude: 29.873888 },
  "Saudi Arabia": { latitude: 23.885942, longitude: 45.079162 },
  "Solomon Islands": { latitude: -9.64571, longitude: 160.156194 },
  "Seychelles": { latitude: -4.679574, longitude: 55.491977 },
  "Sudan": { latitude: 12.862807, longitude: 30.217636 },
  "Sweden": { latitude: 60.128161, longitude: 18.643501 },
  "Singapore": { latitude: 1.352083, longitude: 103.819836 },
  "Saint Helena": { latitude: -24.143474, longitude: -10.030696 },
  "Slovenia": { latitude: 46.151241, longitude: 14.995463 },
  "Svalbard and Jan Mayen": { latitude: 77.553604, longitude: 23.670272 },
  "Slovakia": { latitude: 48.669026, longitude: 19.699024 },
  "Sierra Leone": { latitude: 8.460555, longitude: -11.779889 },
  "San Marino": { latitude: 43.94236, longitude: 12.457777 },
  "Senegal": { latitude: 14.497401, longitude: -14.452362 },
  "Somalia": { latitude: 5.152149, longitude: 46.199616 },
  "Suriname": { latitude: 3.919305, longitude: -56.027783 },
  "São Tomé and Príncipe": { latitude: 0.18636, longitude: 6.613081 },
  "El Salvador": { latitude: 13.794185, longitude: -88.89653 },
  "Syria": { latitude: 34.802075, longitude: 38.996815 },
  "Swaziland": { latitude: -26.522503, longitude: 31.465866 },
  "Turks and Caicos Islands": { latitude: 21.694025, longitude: -71.797928 },
  "Chad": { latitude: 15.454166, longitude: 18.732207 },
  "French Southern Territories": { latitude: -49.280366, longitude: 69.348557 },
  "Togo": { latitude: 8.619543, longitude: 0.824782 },
  "Thailand": { latitude: 15.870032, longitude: 100.992541 },
  "Tajikistan": { latitude: 38.861034, longitude: 71.276093 },
  "Tokelau": { latitude: -8.967363, longitude: -171.855881 },
  "Timor-Leste": { latitude: -8.874217, longitude: 125.727539 },
  "Turkmenistan": { latitude: 38.969719, longitude: 59.556278 },
  "Tunisia": { latitude: 33.886917, longitude: 9.537499 },
  "Tonga": { latitude: -21.178986, longitude: -175.198242 },
  "Turkey": { latitude: 38.963745, longitude: 35.243322 },
  "Trinidad and Tobago": { latitude: 10.691803, longitude: -61.222503 },
  "Tuvalu": { latitude: -7.109535, longitude: 177.64933 },
  "Taiwan": { latitude: 23.69781, longitude: 120.960515 },
  "Tanzania": { latitude: -6.369028, longitude: 34.888822 },
  "Ukraine": { latitude: 48.379433, longitude: 31.16558 },
  "Uganda": { latitude: 1.373333, longitude: 32.290275 },
  "United States": { latitude: 37.09024, longitude: -95.712891 },
  "Uruguay": { latitude: -32.522779, longitude: -55.765835 },
  "Uzbekistan": { latitude: 41.377491, longitude: 64.585262 },
  "Vatican City": { latitude: 41.902916, longitude: 12.453389 },
  "Saint Vincent and the Grenadines": { latitude: 12.984305, longitude: -61.287228 },
  "Venezuela": { latitude: 6.42375, longitude: -66.58973 },
  "British Virgin Islands": { latitude: 18.420695, longitude: -64.639968 },
  "U.S. Virgin Islands": { latitude: 18.335765, longitude: -64.896335 },
  "Vietnam": { latitude: 14.058324, longitude: 108.277199 },
  "Vanuatu": { latitude: -15.376706, longitude: 166.959158 },
  "Wallis and Futuna": { latitude: -13.768752, longitude: -177.156097 },
  "Samoa": { latitude: -13.759029, longitude: -172.104629 },
  "Kosovo": { latitude: 42.602636, longitude: 20.902977 },
  "Yemen": { latitude: 15.552727, longitude: 48.516388 },
  "Mayotte": { latitude: -12.8275, longitude: 45.166244 },
  "South Africa": { latitude: -30.559482, longitude: 22.937506 },
  "Zambia": { latitude: -13.133897, longitude: 27.849332 },
  "Zimbabwe": { latitude: -19.015438, longitude: 29.154857 }
};

function SidebarUploader({ setImage, setIsLoading, setTopCountry, setTopPercentage, setTop3Countries, setTop3Percentages }) {
  const fileInputRef = useRef(null);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    console.log('Selected file:', file);
    
    if (file && (file.type === 'image/jpeg' || file.type === 'image/png')) {
        const formData = new FormData();
        formData.append('image', file);

        setIsLoading(true);
        fetch('http://localhost:8000/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('API Response:', data);
            // Update state with the response data
            setTopCountry(data.top_country);
            setTopPercentage(data.top_percentage);
            setTop3Countries(data.top_3_countries);
            setTop3Percentages(data.top_3_percentages);
        })
        .catch(error => {
            console.error('Error details:', error);
            alert(`Error predicting country: ${error.message}`);
        })
        .finally(() => {
            setIsLoading(false);
        });
    } else {
        alert("Please upload a valid image file of JPG, JPEG, or PNG type");
    }
  };

  return (
    <div className="sidebar-uploader">
      <input 
        type="file" 
        accept="image/*" 
        onChange={handleImageUpload}
        ref={fileInputRef}
        style={{ display: 'none' }}
      />
      <button onClick={() => fileInputRef.current.click()}>
        Upload Image
      </button>
    </div>
  );
}

function Map({ givenCountry }) {
  const mapRef = useRef(null);

  useEffect(() => {
    if (!mapRef.current) {
      mapRef.current = L.map("map").setView([25, 0], 2);
      L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
        subdomains: 'abcd',
        maxZoom: 20,
        minZoom: 2,
      }).addTo(mapRef.current);
    }

    if (givenCountry) {
      const { latitude, longitude } = getFileValues(givenCountry);
      mapRef.current.setView([latitude, longitude], 5);

      // Clear any existing markers
      mapRef.current.eachLayer((layer) => {
        if (layer instanceof L.Marker) {
          mapRef.current.removeLayer(layer);
        }
      });

      // Add new marker
      const customIcon = L.icon({
        iconUrl: "https://cdn-icons-png.flaticon.com/512/684/684908.png",
        iconSize: [38, 38]
      });
      L.marker([latitude, longitude], { icon: customIcon }).addTo(mapRef.current);
    }

    return () => {
      if (mapRef.current) {
        mapRef.current.remove();
        mapRef.current = null;
      }
    };
  }, [givenCountry]);

  return <div id="map" style={{ width: "1000px", height: "600px" }}></div>;
}

function getFileValues(givenCountry) {
  let latitude = 0;
  let longitude = 0;
  
  if (givenCountry) {
    const coordinates = countryCoordinates[givenCountry];
    if (coordinates) {
      latitude = coordinates.latitude;
      longitude = coordinates.longitude;
    } else {
      console.warn(`Coordinates not found for country: ${givenCountry}`);
    }
  }

  return { latitude, longitude };
}

function App() {
  const [image, setImage] = useState(null);
  const [topCountry, setTopCountry] = useState(""); // Ensure this is defined
  const [topPercentage, setTopPercentage] = useState(0); // Ensure this is defined
  const [top3Countries, setTop3Countries] = useState([]); // Ensure this is defined
  const [top3Percentages, setTop3Percentages] = useState([]); // Ensure this is defined
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    if (image) {
      setIsLoading(true);
      fetch('http://localhost:8000/predict', {
        method: 'POST',
        body: image
      })
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
      })
      .then(data => {
        console.log('API Response:', data);
        // Check if the expected properties exist
        if (data.top_country && data.top_percentage && data.top_3_countries && data.top_3_percentages) {
          setTopCountry(data.top_country);
          setTopPercentage(data.top_percentage);
          setTop3Countries(data.top_3_countries);
          setTop3Percentages(data.top_3_percentages);
        } else {
          console.error('Unexpected API response structure:', data);
          alert('Unexpected response from the server. Please try again.');
        }
      })
      .catch(error => {
        console.error('Error details:', error);
        alert(`Error predicting country: ${error.message}`);
      })
      .finally(() => {
        setIsLoading(false);
      });
    }
  }, [image]);

  return (
    <>
      <div className='website-name'>
        <h1>GEOGUESSR.AI</h1>
        <p className="subtitle">Upload any location image and let AI predict the country</p>
      </div>
      
      <div className='map-content'>
        {isLoading ? (
          <div>Predicting country...</div>
        ) : (
          <Map givenCountry={topCountry} />
        )}
        <div className='upload-content'>
          <SidebarUploader 
            setImage={setImage} 
            setIsLoading={setIsLoading} 
            setTopCountry={setTopCountry} 
            setTopPercentage={setTopPercentage} 
            setTop3Countries={setTop3Countries} 
            setTop3Percentages={setTop3Percentages} 
          />
          {topCountry && (
            <div>
              <h3>Top Country: {topCountry} ({topPercentage.toFixed(2)}%)</h3>
              <h4>Other Options:</h4>
              <ul>
                {top3Countries.map((country, index) => (
                  <li key={index}>
                    {country}: {top3Percentages[index].toFixed(2)}%
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </>
  );
}

export default App;