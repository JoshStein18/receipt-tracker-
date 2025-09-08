from app import app
import io

print('Testing advanced receipt tracker...')
with app.test_client() as client:
    response = client.get('/api/health')
    print(f'Health check: {response.status_code}')
    print(f'Response: {response.get_json()}')
    
    # Test upload
    data = {'file': (io.BytesIO(b'fake image data'), 'test_receipt.jpg')}
    response = client.post('/api/upload', data=data, content_type='multipart/form-data')
    print(f'Upload test: {response.status_code}')
    result = response.get_json()
    print(f'Success: {result["success"]}')
    print(f'Items: {len(result["data"]["items"])}')
    for item in result['data']['items']:
        print(f'  {item["description"]} - ${item["total_price"]:.2f} - {item["category"]}')
