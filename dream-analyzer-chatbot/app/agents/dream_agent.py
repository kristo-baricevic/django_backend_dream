from typing import List, Dict, Any
import json
from app.models.schemas import ChatMessage
from app.services.llm_service import llm_service
from app.services.drf_client import api_client
from openai import AsyncOpenAI
from app.config import settings
import httpx

client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

class DreamAgent:
    """
    Agentic orchestrator for dream analysis conversations.
    Uses function calling to interact with Django and Dream Analyzer APIs.
    """
    
    def __init__(self):
        self.system_prompt = """You are a helpful dream analysis assistant with access to a dream journal system.

IMPORTANT CONVERSATION FLOW FOR NEW DREAMS:
When a user starts describing a dream (e.g., "I had a dream about flying"), DO NOT immediately save it. Instead:
1. Acknowledge what they shared
2. Ask gentle follow-up questions to help them recall more details:
   - "What else do you remember about the dream?"
   - "How did you feel during the dream?"
   - "Were there any specific colors, places, or people you noticed?"
   - "What happened next?"
   - "Where were you in the dream?"
3. Build the full dream narrative through conversation
4. ONLY call save_dream() when:
   - The user explicitly says they're done (e.g., "that's all I remember", "that's it")
   - You've gathered a substantial dream description (at least 2-3 exchanges)
   - The user asks you to save it or analyze it

Other capabilities:
- View their past dreams
- Get AI-powered analysis of their dreams
- Search through their dream history
- Get cumulative insights across multiple dreams
- Answer custom questions about their dreams

Be warm, curious, and patient. Help users remember their dreams by asking thoughtful questions."""

        # Define tools for OpenAI function calling
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "save_dream",
                    "description": "Save a complete dream to the user's journal. ONLY use this after having a conversation to gather the full dream details, or when the user explicitly asks to save their dream. Do NOT call this immediately when someone first mentions a dream.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The complete dream content collected through conversation"
                            }
                        },
                        "required": ["content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_recent_dreams",
                    "description": "Get the user's recent dreams from their journal",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Number of dreams to retrieve (default 5)",
                                "default": 5
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_dream",
                    "description": "Get AI-powered analysis of a dream",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "The dream content to analyze"
                            }
                        },
                        "required": ["content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_dreams",
                    "description": "Search through the user's dreams by keyword or theme",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query (keyword, theme, or phrase)"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_cumulative_analysis",
                    "description": "Get cumulative insights across multiple dreams",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Number of recent dreams to analyze (default 10)",
                                "default": 10
                            }
                        }
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "ask_custom_question",
                    "description": "Ask a custom question about the user's dreams",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The question to ask about the dreams"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Number of recent dreams to consider (default 10)",
                                "default": 10
                            }
                        },
                        "required": ["question"]
                    }
                }
            }
        ]
    
    async def process_message(
        self, 
        user_message: str,
        conversation_history: List[ChatMessage]
    ) -> str:
        """
        Process user message using function calling to determine actions.
        """
        # Build messages for OpenAI
        messages = [{"role": "system", "content": self.system_prompt}]
        
        for msg in conversation_history:
            messages.append({"role": msg.role, "content": msg.content})
        
        messages.append({"role": "user", "content": user_message})
        
        # Call OpenAI with function calling
        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=messages,
            tools=self.tools,
            tool_choice="auto"
        )
        
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        
        # If no tool calls, return the message directly
        if not tool_calls:
            return response_message.content
        
        # Execute tool calls
        messages.append(response_message)
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            # Execute the function
            function_response = await self._execute_tool(function_name, function_args)
            
            # Add function response to messages
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": json.dumps(function_response)
            })
        
        # Get final response from OpenAI
        final_response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=messages
        )
        
        return final_response.choices[0].message.content
    
    async def _execute_tool(self, function_name: str, args: Dict[str, Any]) -> Dict:
        """Execute a tool and return the result"""
        try:
           
            # inside _execute_tool(...)
            if function_name == "save_dream":
                content = args["content"]

                # 1) Save entry (returns UUID)
                saved = await api_client.create_journal_entry(content)
                entry_id = saved["id"]

                # 2) Trigger Django's analyzer via the /update/ endpoint
                base = settings.DJANGO_API_URL.rstrip("/")
                url = f"{base}/entries/{entry_id}/update/"
                async with httpx.AsyncClient(timeout=30) as hc:
                    r = await hc.patch(url, json={}) 
                    r.raise_for_status()
                    updated = r.json()

                return {
                    "success": True,
                    "entry_id": entry_id,
                    "message": "Dream saved and analyzed",
                    "entry": updated,
                    "analysis_saved": bool(updated.get("analysis")),
                }

            elif function_name == "analyze_dream":
                result = await api_client.analyze_dream(args["content"])
                return {
                    "success": True,
                    "analysis": {
                        "mood": result.get("mood"),
                        "subject": result.get("subject"),
                        "summary": result.get("summary"),
                        "interpretation": result.get("interpretation"),
                        "symbols": result.get("symbols", [])
                    }
                }
            
            elif function_name == "search_dreams":
                result = await api_client.get_journal_entries(search=args["query"])
                dreams = result.get("results", [])
                return {
                    "success": True,
                    "found": len(dreams),
                    "dreams": [
                        {
                            "id": d["id"],
                            "content": d["content"][:200] + "..." if len(d["content"]) > 200 else d["content"],
                            "subject": d.get("analysis", {}).get("subject"),
                            "mood": d.get("analysis", {}).get("mood")
                        }
                        for d in dreams
                    ]
                }
            
            elif function_name == "get_cumulative_analysis":
                limit = args.get("limit", 10)
                # Get recent entries
                entries_response = await api_client.get_journal_entries(limit=limit)
                entries = entries_response.get("results", [])
                
                # Start cumulative analysis workflow
                workflow_response = await api_client.get_cumulative_analysis_workflow(entries)
                workflow_id = workflow_response.get("workflow_id")
                
                # Poll for completion (simplified - in production use webhooks)
                import asyncio
                for _ in range(30):  # Wait up to 30 seconds
                    await asyncio.sleep(1)
                    status = await api_client.get_workflow_status(workflow_id)
                    if status.get("status") == "completed":
                        return {
                            "success": True,
                            "analysis": status.get("final_result")
                        }
                
                return {
                    "success": False,
                    "message": "Analysis timed out. Please check back later.",
                    "workflow_id": workflow_id
                }
            
            elif function_name == "ask_custom_question":
                limit = args.get("limit", 10)
                question = args["question"]
                
                # Get recent entries
                entries_response = await api_client.get_journal_entries(limit=limit)
                entries = entries_response.get("results", [])
                
                # Start custom question workflow
                workflow_response = await api_client.ask_custom_question_workflow(
                    question, entries
                )
                workflow_id = workflow_response.get("workflow_id")
                
                # Poll for completion
                import asyncio
                for _ in range(30):
                    await asyncio.sleep(1)
                    status = await api_client.get_workflow_status(workflow_id)
                    if status.get("status") == "completed":
                        return {
                            "success": True,
                            "answer": status.get("final_result")
                        }
                
                return {
                    "success": False,
                    "message": "Question processing timed out. Please check back later.",
                    "workflow_id": workflow_id
                }
            
            else:
                return {"success": False, "error": f"Unknown function: {function_name}"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def stream_response(
        self,
        user_message: str,
        conversation_history: List[ChatMessage]
    ):
        """Stream response for real-time updates"""
        # For function calling, we can't stream effectively
        # So we'll process and then stream the final response
        result = await self.process_message(user_message, conversation_history)
        
        # Stream the result word by word for better UX
        words = result.split()
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            import asyncio
            await asyncio.sleep(0.05)  # Small delay for streaming effect

dream_agent = DreamAgent()