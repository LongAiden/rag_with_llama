import requests
import tempfile
import os

def test_api():
    """Test the file validation API endpoints."""
    base_url = "http://localhost:8000"

    print("Testing File Validation API...")

    # Test 1: Get supported types
    print("\n1. Testing /supported-types/ endpoint:")
    try:
        response = requests.get(f"{base_url}/supported-types/")
        if response.status_code == 200:
            print("✓ Supported types:", response.json())
        else:
            print("✗ Failed to get supported types")
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test 2: Create a test PDF file and validate it
    print("\n2. Testing /validate-file/ endpoint with a fake PDF:")
    try:
        # Create a fake PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(b"%PDF-1.4\nHello World PDF content")
            temp_file.flush()

            with open(temp_file.name, 'rb') as f:
                files = {'file': ('test.pdf', f, 'application/pdf')}
                response = requests.post(f"{base_url}/validate-file/", files=files)

                if response.status_code == 200:
                    result = response.json()
                    print("✓ Validation result:", result)
                else:
                    print(f"✗ Validation failed: {response.text}")

            # Clean up
            os.unlink(temp_file.name)
    except Exception as e:
        print(f"✗ Error: {e}")

    # Test 3: Create a test TXT file and validate it
    print("\n3. Testing /validate-file/ endpoint with a text file:")
    try:
        # Create a test TXT file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp_file:
            temp_file.write(b"This is a test text file content.")
            temp_file.flush()

            with open(temp_file.name, 'rb') as f:
                files = {'file': ('test.txt', f, 'text/plain')}
                response = requests.post(f"{base_url}/validate-file/", files=files)

                if response.status_code == 200:
                    result = response.json()
                    print("✓ Validation result:", result)
                else:
                    print(f"✗ Validation failed: {response.text}")

            # Clean up
            os.unlink(temp_file.name)
    except Exception as e:
        print(f"✗ Error: {e}")

if __name__ == "__main__":
    print("Make sure the API server is running with: python file_validation_api.py")
    print("Then run this test script.")
    test_api()