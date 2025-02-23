import torch
import clip
from PIL import Image
from pathlib import Path
import json
import pytesseract
import easyocr
import re

class GeoGuessrPredictor:
    def __init__(self, model_name="ViT-B/32"):

        # Load the model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        
        # Load guide data
        guide_path = Path("geoguessr_data/geoguessr_guide.json")
        with open(guide_path, 'r', encoding='utf-8') as f:
            self.guide_data = json.load(f)
            
        # Extract features and examples from guide
        self.guide_features = self._extract_guide_features()
        
        # Combine with existing categories
        self.categories = [
            "architecture", "landscape", "vegetation",
            "infrastructure", "road_signs", "driving_side",
            "utility_poles", "bollards", "license_plates"
        ] + self.guide_features
        
        # All countries/territories in GeoGuessr
        self.countries = [
            "Albania", "Andorra", "Argentina", "Australia", "Austria", "Bangladesh", 
            "Belgium", "Bermuda", "Bhutan", "Bolivia", "Botswana", "Brazil", "Bulgaria", 
            "Cambodia", "Canada", "Chile", "China", "Colombia", "Croatia", "Czechia", 
            "Denmark", "Dominican Republic", "Ecuador", "Egypt", "Estonia", "Eswatini", 
            "Faroe Islands", "Finland", "France", "Germany", "Ghana", "Greece", "Greenland", 
            "Guatemala", "Hong Kong", "Hungary", "Iceland", "India", "Indonesia", "Ireland", 
            "Israel", "Italy", "Japan", "Jordan", "Kenya", "Kyrgyzstan", "Laos", "Latvia", 
            "Lesotho", "Lithuania", "Luxembourg", "Malaysia", "Mexico", "Mongolia", 
            "Montenegro", "Nepal", "Netherlands", "New Zealand", "Nigeria", "North Macedonia", 
            "North Mariana Islands", "Norway", "Pakistan", "Palestine", "Panama", "Peru", 
            "Philippines", "Poland", "Portugal", "Puerto Rico", "Réunion", "Romania", 
            "Russia", "Rwanda", "Senegal", "Serbia", "Singapore", "Slovakia", "Slovenia", 
            "South Africa", "South Korea", "Spain", "Sri Lanka", "Sweden", "Switzerland", 
            "Taiwan", "Thailand", "Turkey", "Uganda", "Ukraine", "United Arab Emirates", 
            "United Kingdom", "United States", "Uruguay", "Vietnam"
        ]

    def _extract_guide_features(self):
        """Extract relevant features from the guide data"""
        features = []
        for item in self.guide_data:
            if item['type'] == 'h4':
                features.append(item['text'].lower())
            if item['type'] == 'image': 
                if item['caption']:
                    # Process caption to extract key features
                    # This would need a more sophisticated NLP though
                    pass
        return list(set(features))  # Remove duplicates

    def predict(self, image_path):
        """Combined method to analyze image and predict location"""
        image = Image.open(image_path)
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        # Add OCR analysis
        try:
            text = pytesseract.image_to_string(image)
            # Look for country-specific indicators
            domain_indicators = { # All the domains in the world.
                '.fr': 'France',
                '.de': 'Germany',
                '.uk': 'United Kingdom',
                '.es': 'Spain',
                '.it': 'Italy',
                '.pl': 'Poland',
                '.nl': 'Netherlands',
                '.be': 'Belgium',
                '.at': 'Austria',
                '.ch': 'Switzerland',
                '.us': 'United States',
                '.ca': 'Canada',
                '.au': 'Australia',
                '.nz': 'New Zealand',
                '.sg': 'Singapore',
                '.hk': 'Hong Kong',
                '.jp': 'Japan',
                '.kr': 'South Korea',
                '.tw': 'Taiwan',
                '.th': 'Thailand',
                '.tr': 'Turkey',
                '.ua': 'Ukraine',
                '.ru': 'Russia',
                '.br': 'Brazil',
                '.mx': 'Mexico',
                '.ar': 'Argentina',
                '.cl': 'Chile',
                '.co': 'Colombia',
                '.pe': 'Peru',
                '.za': 'South Africa',
                '.eg': 'Egypt',
                '.ma': 'Morocco',
                '.ng': 'Nigeria',
                '.ke': 'Kenya',
                '.ug': 'Uganda',
                '.in': 'India',
                '.cn': 'China',
                '.id': 'Indonesia',
                '.my': 'Malaysia',
                '.ph': 'Philippines',
                '.vn': 'Vietnam',
                '.lk': 'Sri Lanka',
                '.ae': 'United Arab Emirates',
                '.il': 'Israel',
                '.sa': 'Saudi Arabia',
                '.pk': 'Pakistan',
                '.dk': 'Denmark',
                '.fi': 'Finland',
                '.no': 'Norway',
                '.se': 'Sweden',
                '.pt': 'Portugal',
                '.gr': 'Greece',
                '.ie': 'Ireland',
                '.cz': 'Czech Republic',
                '.hu': 'Hungary',
                '.ro': 'Romania',
                '.bg': 'Bulgaria',
                '.hr': 'Croatia',
                '.rs': 'Serbia',
                '.sk': 'Slovakia',
                '.si': 'Slovenia',
                '.ee': 'Estonia',
                '.lv': 'Latvia',
                '.lt': 'Lithuania',
                '.uy': 'Uruguay'
            }
            
            # Adjust country scores based on found TLDs
            country_scores = {}
            for tld, country in domain_indicators.items():
                if tld in text.lower():
                    # Boost confidence for this country to atleast 90%
                    country_scores[country] = max(country_scores.get(country, 0), 0.9)
        except:
            pass  # OCR is optional, continue if it fails
        
        # Analyze features
        feature_prompts = [f"This image shows {category} typical of" for category in self.categories]
        
        # Create country prompts
        country_prompts = []
        for country in self.countries:
            prompts = [
                f"A street view photo from {country}",
                f"This looks like {country}",
                f"The architecture and landscape of {country}",
                f"The infrastructure and roads of {country}"
            ]
            country_prompts.extend(prompts)
        
        all_prompts = feature_prompts + country_prompts
        text_tokens = clip.tokenize(all_prompts).to(self.device)
        
        with torch.no_grad():
            # Get image and text features
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_tokens)
            
            # Normalize features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
        # Process feature predictions
        feature_predictions = {}
        for i, category in enumerate(self.categories):
            feature_predictions[category] = similarity[0][i].item()
            
        # Process country predictions
        country_scores = {}
        offset = len(self.categories)
        for i, country in enumerate(self.countries):
            scores = similarity[0][offset + i*4:offset + (i+1)*4]
            country_scores[country] = scores.mean().item()
        
        # Get top 3 predictions
        sorted_countries = sorted(country_scores.items(), key=lambda x: x[1], reverse=True)
        predicted_regions = [country for country, _ in sorted_countries[:3]]
        confidence_scores = [score for _, score in sorted_countries[:3]]
        
        return {
            'features': feature_predictions,
            'predicted_regions': predicted_regions,
            'confidence_scores': confidence_scores
        }

def load_predictor():
    return GeoGuessrPredictor()

# def extract_domains_from_image(image_path): # TODO: Work on domain extraction.
#     # Initialize EasyOCR reader
#     reader = easyocr.Reader(['en'])
    
#     # Read text from image
#     results = reader.readtext(image_path)
    
#     # Extract all text from results
#     text = ' '.join([result[1] for result in results])
    
#     # Domain pattern matching
#     # This pattern matches common domain formats
#     domain_pattern = r'(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}'
#     domains = re.findall(domain_pattern, text)
    
#     return domains

# # Example usage
# image_path = 'test4.png'
# found_domains = extract_domains_from_image(image_path)
# print("Found domains:", found_domains)