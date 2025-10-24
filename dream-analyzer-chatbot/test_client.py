"""
Simple test script to verify the chatbot is working.
Run this after starting the server.
"""
import asyncio
import httpx
import json

BASE_URL = "http://localhost:8001"

async def test_health():
    """Test health endpoint"""
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/health")
        print(f"✓ Health check: {response.json()}")

async def test_chat():
    """Test chat message endpoint"""
    async with httpx.AsyncClient() as client:
        data = {
            "message": "Hello! Can you help me understand my dreams?",
            "conversation_history": []
        }
        
        response = await client.post(
            f"{BASE_URL}/api/chat/message",
            json=data,
            timeout=30.0
        )
        
        result = response.json()
        print(f"\n✓ Chat Response:")
        print(f"  {result['message'][:100]}...")

async def test_streaming():
    """Test streaming endpoint"""
    async with httpx.AsyncClient() as client:
        data = {
            "message": "What are common dream symbols?",
            "conversation_history": []
        }
        
        print(f"\n✓ Streaming Response:")
        print("  ", end="")
        
        async with client.stream(
            'POST',
            f"{BASE_URL}/api/chat/message/stream",
            json=data,
            timeout=30.0
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith('data: '):
                    data = json.loads(line[6:])
                    if not data.get('done'):
                        print(data.get('content', ''), end='', flush=True)
        
        print("\n")

async def main():
    """Run all tests"""
    print("Testing Dream Analyzer Chatbot API\n")
    print("=" * 50)
    
    try:
        await test_health()
        await test_chat()
        await test_streaming()
        
        print("\n" + "=" * 50)
        print("✓ All tests passed!")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure the server is running:")
        print("  uvicorn app.main:app --reload --port 8001")

if __name__ == "__main__":
    asyncio.run(main())
