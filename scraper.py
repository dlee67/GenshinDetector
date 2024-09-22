import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm

def create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f'Directory "{folder}" created.')
    else:
        print(f'Directory "{folder}" already exists.')

def get_image_urls(soup, base_url):
    image_urls = []
    for img in soup.find_all('img'):
        img_url = img.get('src')
        if img_url:
            full_url = urljoin(base_url, img_url)
            image_urls.append(full_url)
    return image_urls

def download_image(img_url, folder, image_number):
    img_extension = img_url.split('.')[-1]
    img_name = f"{image_number}.{img_extension}"
    img_name = img_name.split(".jpg")[0] + ".jpg"
    img_path = os.path.join(folder, img_name)
    
    img_response = requests.get(img_url)
    
    if img_response.status_code == 200:
        with open(img_path, 'wb') as f:
            f.write(img_response.content)
        print(f'Downloaded {img_name}')
    else:
        print(f'Failed to download {img_url}')

def scrape_images(url, folder):
    response = requests.get(url)
    if response.status_code != 200:
        print(f'Failed to load page: {url}')
        return
    
    soup = BeautifulSoup(response.text, 'html.parser')
    image_urls = get_image_urls(soup, url)
    
    if not image_urls:
        print(f'No images found on the page: {url}')
        return
    
    for i, img_url in enumerate(tqdm(image_urls, desc="Downloading images"), start=1):
        download_image(img_url, folder, i)

if __name__ == "__main__":
    base_url = 'https://safebooru.org/index.php?page=post&s=list&tags=solo+raiden_shogun+'
    folder = 'Raiden'    
    create_folder(folder)
    scrape_images(base_url, folder)
