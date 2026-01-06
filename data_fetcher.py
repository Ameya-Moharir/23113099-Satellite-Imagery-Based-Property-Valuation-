"""
Satellite Image Fetcher
Downloads satellite imagery from free APIs using property coordinates
"""

import pandas as pd
import requests
import time
import math
from pathlib import Path
from PIL import Image
from io import BytesIO
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SatelliteImageFetcher:
    """
    Fetches satellite imagery using multiple free APIs
    """
    
    def __init__(self, output_dir, zoom=18, image_size=640):
        """
        Args:
            output_dir: Directory to save images
            zoom: Zoom level (12-19, higher = more detail)
            image_size: Image dimensions in pixels
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.zoom = zoom
        self.image_size = image_size
        
    def deg2num(self, lat_deg, lon_deg, zoom):
        """Convert lat/lon to tile numbers"""
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        xtile = int((lon_deg + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return (xtile, ytile)
    
    def fetch_esri_satellite(self, lat, lon, property_id):
        """
        Fetch from ESRI ArcGIS World Imagery (FREE, NO API KEY)
        Best free option for satellite imagery
        """
        try:
            xtile, ytile = self.deg2num(lat, lon, self.zoom)
            
            # ESRI World Imagery - Free satellite tiles
            url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{self.zoom}/{ytile}/{xtile}"
            
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                img = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)
                
                image_path = self.output_dir / f"{property_id}.jpg"
                img.save(image_path, 'JPEG', quality=95)
                
                return str(image_path)
            else:
                logger.warning(f"Failed for property {property_id}: Status {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error for property {property_id}: {str(e)}")
            return None
    
    def fetch_mapbox_satellite(self, lat, lon, property_id, api_key):
        """
        Fetch from Mapbox Static Images API
        Requires free API key: https://account.mapbox.com/
        Free tier: 50,000 requests/month
        """
        if not api_key:
            logger.error("Mapbox API key required. Sign up at https://account.mapbox.com/")
            return None
            
        try:
            url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{lon},{lat},{self.zoom}/{self.image_size}x{self.image_size}@2x"
            
            params = {'access_token': api_key}
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                image_path = self.output_dir / f"{property_id}.jpg"
                img.save(image_path, 'JPEG', quality=95)
                return str(image_path)
            else:
                logger.warning(f"Failed for property {property_id}: Status {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error for property {property_id}: {str(e)}")
            return None
    
    def fetch_google_satellite(self, lat, lon, property_id, api_key):
        """
        Fetch from Google Maps Static API
        Requires API key: https://developers.google.com/maps/documentation/maps-static
        Free tier: $200 credit/month (~28,000 loads)
        """
        if not api_key:
            logger.error("Google Maps API key required")
            return None
            
        try:
            url = "https://maps.googleapis.com/maps/api/staticmap"
            
            params = {
                'center': f"{lat},{lon}",
                'zoom': self.zoom,
                'size': f"{self.image_size}x{self.image_size}",
                'maptype': 'satellite',
                'key': api_key
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content))
                image_path = self.output_dir / f"{property_id}.jpg"
                img.save(image_path, 'JPEG', quality=95)
                return str(image_path)
            else:
                logger.warning(f"Failed for property {property_id}: Status {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error for property {property_id}: {str(e)}")
            return None
    
    def fetch_dataset(self, csv_path, method='esri', api_key=None, delay=0.5, 
                 sample_size=None, id_col='id', lat_col='lat', lon_col='long'):
        """
        Download images for entire dataset with parallel processing
        """
        import concurrent.futures
        
        df = pd.read_csv(csv_path)
        
        if sample_size:
            df = df.head(sample_size)
        
        logger.info(f"Downloading {len(df)} satellite images using {method}...")
        
        # Check existing images
        to_download = []
        image_paths = [None] * len(df)
        
        for idx, row in df.iterrows():
            property_id = row[id_col]
            existing_path = self.output_dir / f"{property_id}.jpg"
            if existing_path.exists():
                image_paths[idx] = str(existing_path)
            else:
                to_download.append((idx, row))
        
        already_downloaded = len(df) - len(to_download)
        logger.info(f"Already have {already_downloaded} images, downloading {len(to_download)} new ones")
        
        if len(to_download) == 0:
            df['image_path'] = image_paths
            return df
        
        # Download function for parallel processing
        def download_single(item):
            idx, row = item
            property_id = row[id_col]
            lat = row[lat_col]
            lon = row[lon_col]
            
            if method == 'esri':
                path = self.fetch_esri_satellite(lat, lon, property_id)
            elif method == 'mapbox':
                path = self.fetch_mapbox_satellite(lat, lon, property_id, api_key)
            elif method == 'google':
                path = self.fetch_google_satellite(lat, lon, property_id, api_key)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            time.sleep(delay)  # Still be nice to API
            return idx, path
        
        # Parallel download with progress bar
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(download_single, item) for item in to_download]
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(to_download), desc="Downloading"):
                idx, path = future.result()
                image_paths[idx] = path
        
        df['image_path'] = image_paths
        
        # Count successful downloads
        success_count = df['image_path'].notna().sum()
        logger.info(f"Successfully downloaded {success_count}/{len(df)} images")
        
        return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Download satellite images')
    parser.add_argument('--csv', required=True, help='Path to CSV file')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--method', default='esri', choices=['esri', 'mapbox', 'google'])
    parser.add_argument('--api-key', default=None, help='API key (for mapbox/google)')
    parser.add_argument('--zoom', type=int, default=18, help='Zoom level')
    parser.add_argument('--size', type=int, default=640, help='Image size')
    parser.add_argument('--sample', type=int, default=None, help='Sample size for testing')
    
    args = parser.parse_args()
    
    fetcher = SatelliteImageFetcher(args.output, zoom=args.zoom, image_size=args.size)
    result_df = fetcher.fetch_dataset(
        args.csv, 
        method=args.method,
        api_key=args.api_key,
        sample_size=args.sample
    )
    
    output_csv = Path(args.output).parent / f"{Path(args.csv).stem}_with_images.csv"
    result_df.to_csv(output_csv, index=False)
    logger.info(f"Saved results to {output_csv}")
