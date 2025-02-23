import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin
import os
from pathlib import Path
import hashlib

def download_image(url, save_dir):
    """Download image and return local path"""
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            # Create unique filename using hash of URL
            file_hash = hashlib.md5(url.encode()).hexdigest()
            file_ext = url.split('.')[-1].split('?')[0]  # Get extension before any URL parameters
            filename = f"{file_hash}.{file_ext}"
            filepath = save_dir / filename
            
            # Save image
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return str(filepath)
    except Exception as e:
        print(f"Error downloading {url}: {e}")
    return None

def scrape_page(url, images_dir):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    content = []
    
    for element in soup.find_all(['h2', 'h3', 'h4', 'p', 'img']):
        if element.name == 'img':
            img_url = urljoin(url, element.get('src', ''))
            
            # Get all possible text associated with the image
            img_data = {
                'type': 'image',
                'original_url': img_url,
                'alt': element.get('alt', ''),
                'title': element.get('title', ''),
                'caption': ''
            }
            
            # Try to find caption
            caption = element.find_next('figcaption')
            if not caption:
                caption = element.find_parent('figure').find('figcaption') if element.find_parent('figure') else None
            if not caption:
                next_p = element.find_next('p')
                if next_p and len(next_p.text.strip()) < 200:
                    caption = next_p
            
            if caption:
                img_data['caption'] = caption.text.strip()
            
            # Download image
            local_path = download_image(img_url, images_dir)
            if local_path:
                img_data['local_path'] = local_path
                content.append(img_data)
            
        else:
            # Regular text content
            content.append({
                'type': element.name,
                'text': element.text.strip()
            })
    
    return content

def main():
    # Create directories
    data_dir = Path('geoguessr_data')
    images_dir = data_dir / 'images'
    data_dir.mkdir(exist_ok=True)
    images_dir.mkdir(exist_ok=True)
    
    # Base URL and pages to scrape
    base_url = "https://www.plonkit.net/beginners-guide-"
    pages = [2, 3, 4]
    
    # Scrape all pages
    all_content = []
    for page in pages:
        url = f"{base_url}{page}"
        print(f"Scraping page {page}...")
        try:
            page_content = scrape_page(url, images_dir)
            if page_content:  # Check if we got any content
                all_content.extend(page_content)
                print(f"Found {len(page_content)} items on page {page}")
            else:
                print(f"No content found on page {page}")
        except Exception as e:
            print(f"Error scraping page {page}: {e}")
    
    # Save to JSON file with error handling
    json_path = data_dir / 'geoguessr_guide.json'
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_content, f, ensure_ascii=False, indent=2)
            f.flush()  # Force write to disk
            os.fsync(f.fileno())  # Ensure it's written to disk
        
        # Verify the file was written
        if json_path.exists():
            print(f"\nScraping complete!")
            print(f"Data saved to: {json_path}")
            print(f"Images saved to: {images_dir}")
            print(f"Total items scraped: {len(all_content)}")
        else:
            print("Error: JSON file was not created")
            
    except Exception as e:
        print(f"Error saving JSON file: {e}")

if __name__ == "__main__":
    main()