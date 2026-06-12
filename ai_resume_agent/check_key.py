import sys
import os

def test_key(api_key):
    print("Testing Gemini API Key with new google-genai SDK...")
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        
        print("\nListing models available for this API Key:")
        models = list(client.models.list())
        for m in models:
            print(f" - {m.name}")
            
        print("\nTesting simple query with gemini-1.5-flash...")
        response = client.models.generate_content(
            model='gemini-1.5-flash',
            contents='Hello, respond with OK',
        )
        print(f"Success! Response: {response.text}")
        print("[OK] Your API Key is valid and working!")
        
    except Exception as e:
        print(f"\n[ERROR] The API call failed: {str(e)}")
        print("\nCommon reasons for this error:")
        print("1. The API key was copied incorrectly.")
        print("2. The key has expired or has been deleted.")
        print("3. The Google Cloud project associated with your API key doesn't have the 'Generative Language API' enabled.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if env_key:
            test_key(env_key)
        else:
            print("Usage: python check_key.py YOUR_API_KEY")
            print("Or set GEMINI_API_KEY environment variable.")
    else:
        test_key(sys.argv[1].strip())
