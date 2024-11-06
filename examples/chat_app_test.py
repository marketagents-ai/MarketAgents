import httpx
import asyncio
from typing import List, Dict, Any
import json
from datetime import datetime
import backoff

class ChatAPIError(Exception):
    pass

async def test_chat_api():
    base_url = "http://127.0.0.1:8000/api/v1"
    timeout = httpx.Timeout(60.0, connect=5.0)
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            # Create first chat
            print(f"\n=== Creating new chat thread at {datetime.now()} ===")
            response = await client.post(f"{base_url}/chats/")
            if response.status_code != 201:
                raise ChatAPIError(f"Failed to create chat: {response.text}")
            
            chat_data = response.json()
            chat_id = chat_data["id"]
            print(f"Created chat with ID: {chat_id} and UUID: {chat_data['uuid']}")

            # Send multiple messages
            messages = [
                "What is recursion in programming?",
                "Can you explain it with a simple example?",
                "What are the potential pitfalls of recursion?"
            ]
            
            @backoff.on_exception(
                backoff.expo,
                (httpx.ReadTimeout, httpx.TimeoutException, ChatAPIError),
                max_tries=3,
                max_time=300
            )
            async def send_message_with_retry(chat_id: int, content: str):
                response = await client.post(
                    f"{base_url}/chats/{chat_id}/messages/",
                    json={"content": content}
                )
                if response.status_code != 200:
                    raise ChatAPIError(f"Failed to send message: {response.text}")
                return response
            
            print("\n=== Sending multiple messages to same thread ===")
            for i, message in enumerate(messages, 1):
                print(f"\nSending message {i}/{len(messages)}: {message}")
                try:
                    response = await send_message_with_retry(chat_id, message)
                    result = response.json()
                    print(f"Response received - History length: {len(result.get('history', []))}")
                    if result.get('new_message'):
                        print(f"Latest message: {result['new_message']}")
                    await asyncio.sleep(1)  # Small delay between messages
                except Exception as e:
                    print(f"Error processing message: {str(e)}")
                    continue

            # # Create second chat
            # print("\n=== Creating second chat thread ===")
            # response = await client.post(f"{base_url}/chats/")
            # if response.status_code != 201:
            #     raise ChatAPIError(f"Failed to create second chat: {response.text}")
            
            # second_chat_data = response.json()
            # second_chat_id = second_chat_data["id"]
            # print(f"Created second chat with ID: {second_chat_id}")

            # # Send message to second chat
            # print("\n=== Sending message to second thread ===")
            # try:
            #     response = await send_message_with_retry(
            #         second_chat_id,
            #         "Tell me a programmer joke about Python"
            #     )
            #     result = response.json()
            #     print(f"Response received for second chat")
            #     if result.get('new_message'):
            #         print(f"Response content: {result['new_message']}")
            # except Exception as e:
            #     print(f"Error sending message to second chat: {str(e)}")

            # # List all chats
            # print("\n=== Listing all chats ===")
            # try:
            #     response = await client.get(f"{base_url}/chats/")
            #     if response.status_code != 200:
            #         raise ChatAPIError(f"Failed to list chats: {response.text}")
                
            #     chats = response.json()
            #     print(f"Found {len(chats)} chats")
            #     for chat in chats:
            #         print(f"Chat ID: {chat['id']}, Messages: {len(chat.get('history', []))}")
            # except Exception as e:
            #     print(f"Error listing chats: {str(e)}")

            # # Cleanup
            # print("\n=== Cleaning up ===")
            # for id_to_delete in [chat_id, second_chat_id]:
            #     try:
            #         response = await client.delete(f"{base_url}/chats/{id_to_delete}")
            #         if response.status_code == 204:
            #             print(f"Successfully deleted chat {id_to_delete}")
            #         else:
            #             print(f"Warning: Failed to delete chat {id_to_delete}: {response.text}")
            #     except Exception as e:
            #         print(f"Error deleting chat {id_to_delete}: {str(e)}")

        except Exception as e:
            print(f"\nError during test: {str(e)}")
            raise
        finally:
            print("\n=== Test completed ===")

if __name__ == "__main__":
    asyncio.run(test_chat_api())