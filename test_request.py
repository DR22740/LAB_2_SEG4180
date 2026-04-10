import requests
from PIL import Image

url = 'http://127.0.0.1:5001/predict'
headers = {'x-api-key': 'my_super_secret_password_123'}

print("Creating dummy.jpg...")
img = Image.new('RGB', (224, 224), color='red')
img.save('dummy.jpg')

print("Sending image to server...")
with open('dummy.jpg', 'rb') as f:
    files = {'image': ('dummy.jpg', f, 'image/jpeg')}
    response = requests.post(url, headers=headers, files=files)

print("\n--- SERVER RESPONSE ---")
if response.status_code == 200:
    print(response.json())
else:
    print(f"Error {response.status_code}: {response.text}")