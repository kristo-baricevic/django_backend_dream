import os
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
import openai

# Emotion types and colors (equivalent to your emotions parameter)
class EmotionType(str, Enum):
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    ANXIETY = "anxiety"
    CONTENTMENT = "contentment"
    EXCITEMENT = "excitement"
    MELANCHOLY = "melancholy"

# Emotion color mapping
EMOTION_COLORS = {
    EmotionType.JOY: "#FFD700",
    EmotionType.SADNESS: "#4169E1",
    EmotionType.ANGER: "#DC143C",
    EmotionType.FEAR: "#800080",
    EmotionType.SURPRISE: "#FF69B4",
    EmotionType.DISGUST: "#228B22",
    EmotionType.ANXIETY: "#FF4500",
    EmotionType.CONTENTMENT: "#32CD32",
    EmotionType.EXCITEMENT: "#FF1493",
    EmotionType.MELANCHOLY: "#708090"
}

def get_emotion_color(emotion: EmotionType) -> str:
    return EMOTION_COLORS.get(emotion, "#808080")

# Personality types
PERSONALITIES = {
    "empathetic": "You are an empathetic and compassionate analyst who focuses on emotional understanding and healing.",
    "analytical": "You are a logical and systematic analyst who breaks down patterns and provides structured insights.",
    "mystical": "You are a mystical and spiritual interpreter who sees deeper meanings and cosmic connections.",
    "practical": "You are a practical and solution-oriented analyst who focuses on actionable insights and real-world applications."
}

def get_personality(personality_type: str) -> str:
    return PERSONALITIES.get(personality_type, PERSONALITIES["empathetic"])

# Pydantic model for structured output (equivalent to Zod schema)
class JournalAnalysis(BaseModel):
    mood: EmotionType = Field(description="the mood of the person who wrote the journal entry")
    summary: str = Field(description="quick summary of the entire entry")
    negative: bool = Field(description="is the journal entry negative? (i.e. does it contain negative emotions?)")
    subject: str = Field(description="a whimsical title for the dream")
    color: str = Field(description="a hexadecimal color code that represents the mood of the entry")
    interpretation: str = Field(description="your final analysis of the dream in about 5 or 6 sentences. Make this a dramatic interpretation. When you are done, suggest a song to listen to and a snack to eat.")
    sentiment_score: int = Field(description="sentiment of the text and rated on a scale from -10 to 10, where -10 is extremely negative, 0 is neutral, and 10 is extremely positive")

class JournalEntry(BaseModel):
    id: str
    created_at: datetime
    content: str

class DreamJournalAnalyzer:
    def __init__(self, openai_api_key: str):
        """Initialize the analyzer with OpenAI API key."""
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.llm = ChatOpenAI(temperature=0.8, model_name='gpt-3.5-turbo')
        self.embeddings = OpenAIEmbeddings()
        
    async def qa_analysis(self, question: str, entries: List[JournalEntry]) -> str:
        """
        Function 1: Generate cumulative analysis using QA chain over journal entries.
        Equivalent to the qa() function in your JS code.
        """
        try:
            # Convert entries to LangChain Documents
            docs = [
                Document(
                    page_content=entry.content,
                    metadata={"source": entry.id, "date": entry.created_at.isoformat()}
                )
                for entry in entries
            ]
            
            if not docs:
                raise ValueError("No journal entries provided")
            
            # Create vector store from documents
            vectorstore = FAISS.from_documents(docs, self.embeddings)
            
            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 4})
            )
            
            # Get answer
            result = qa_chain.run(question)
            return result
            
        except Exception as error:
            print(f'Error in QA process: {error}')
            raise Exception('Failed to process QA request')

    async def ai_generate(self, question: str) -> str:
        """
        Function 2: Generate sample dream content.
        Equivalent to the aiGenerate() function in your JS code.
        """
        try:
            model = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
            result = model.invoke(question)
            return result.content
        except Exception as error:
            print(f'Error in AI generation: {error}')
            raise Exception('Failed to generate AI content')

    async def analyze_entry(self, content: str, personality_type: str = "empathetic") -> JournalAnalysis:
        """
        Function 3: Analyze journal entry with structured output.
        Equivalent to the analyze() function in your JS code.
        """
        try:
            # Get personality description
            personality = get_personality(personality_type)
            
            # Set up the parser
            parser = PydanticOutputParser(pydantic_object=JournalAnalysis)
            
            # Create the prompt template
            prompt = PromptTemplate(
                template="""
                {personality}
                Analyze the following journal entry holistically. Follow the instructions and format your response to match the format instructions.
                
                {format_instructions}

                Journal Entry:
                {content}
                """,
                input_variables=["content", "personality"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )
            
            # Format the prompt
            formatted_prompt = prompt.format(content=content, personality=personality)
            
            # Get analysis from LLM
            model = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
            result = model.invoke(formatted_prompt)
            result = result.content
            
            # Parse the result
            parsed_result = parser.parse(result)
            
            # Set the color based on mood
            parsed_result.color = get_emotion_color(parsed_result.mood)
            
            return parsed_result
            
        except Exception as error:
            print(f'Failed to parse analysis result: {error}')
            raise Exception('Failed to analyze dream journal entry')

    async def batch_analyze_entries(self, entries: List[JournalEntry], personality_type: str = "empathetic") -> List[JournalAnalysis]:
        """
        Analyze multiple journal entries in batch.
        """
        results = []
        for entry in entries:
            try:
                analysis = await self.analyze_entry(entry.content, personality_type)
                results.append(analysis)
            except Exception as error:
                print(f'Failed to analyze entry {entry.id}: {error}')
                # Continue with other entries
                continue
        return results

# Example usage and helper functions
class DreamJournalService:
    def __init__(self, openai_api_key: str):
        self.analyzer = DreamJournalAnalyzer(openai_api_key)
    
    async def get_cumulative_analysis(self, entries: List[JournalEntry]) -> str:
        """Get overall analysis across all entries."""
        question = "Provide a comprehensive analysis of these journal entries, identifying patterns, themes, and emotional trends over time."
        return await self.analyzer.qa_analysis(question, entries)
    
    async def generate_sample_dream(self, theme: str = "flying") -> str:
        """Generate a sample dream for inspiration."""
        prompt = f"Write a vivid and imaginative dream about {theme}. Make it mysterious and emotionally rich, about 100-150 words."
        return await self.analyzer.ai_generate(prompt)
    
    async def analyze_single_entry(self, content: str, personality: str = "empathetic") -> JournalAnalysis:
        """Analyze a single journal entry."""
        return await self.analyzer.analyze_entry(content, personality)

# # Example usage
# async def main():
#     # Initialize the service
#     service = DreamJournalService("your-openai-api-key")
    
#     # Sample journal entries
#     entries = [
#         JournalEntry(
#             id="1",
#             created_at=datetime.now(),
#             content="I had the most vivid dream about flying over a beautiful landscape. I felt so free and peaceful."
#         ),
#         JournalEntry(
#             id="2", 
#             created_at=datetime.now(),
#             content="Dreamed I was lost in a dark forest, feeling anxious and scared. The trees seemed to be closing in on me."
#         )
#     ]
    
#     # Example 1: Get cumulative analysis
#     cumulative_analysis = await service.get_cumulative_analysis(entries)
#     print("Cumulative Analysis:", cumulative_analysis)
    
#     # Example 2: Generate sample dream
#     sample_dream = await service.generate_sample_dream("underwater adventure")
#     print("Sample Dream:", sample_dream)
    
#     # Example 3: Analyze single entry
#     analysis = await service.analyze_single_entry(
#         "I dreamed I was dancing with stars in an endless ballroom made of clouds.",
#         "mystical"
#     )
#     print("Analysis:", analysis)

# if __name__ == "__main__":
#     import asyncio
#     asyncio.run(main())