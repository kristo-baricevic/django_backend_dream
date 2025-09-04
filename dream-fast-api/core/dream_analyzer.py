import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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

class DreamKnowledgeBase:
    def __init__(self, files_directory: str, vector_directory: str, embeddings):
        self.files_directory = files_directory
        self.vector_directory = vector_directory
        self.vectorstore = None
        self.embeddings = embeddings
        
    async def initialize(self):
        """Load existing vectors or create new ones."""
        if os.path.exists(os.path.join(self.vector_directory, "index.faiss")):
            print("Loading existing knowledge base...")
            try:
                self.vectorstore = FAISS.load_local(
                    self.vector_directory, 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except Exception as e:
                print(f"Failed to load existing knowledge base: {e}")
                await self.build_knowledge_base()
        else:
            print("Building knowledge base from files...")
            await self.build_knowledge_base()
    
    async def build_knowledge_base(self):
        """Process PDFs and text files to create FAISS index."""
        if not os.path.exists(self.files_directory):
            print(f"Knowledge directory {self.files_directory} not found")
            return
            
        documents = []
        
        try:
            # Load PDFs
            pdf_loader = DirectoryLoader(
                self.files_directory,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader
            )
            pdf_docs = pdf_loader.load()
            documents.extend(pdf_docs)
            
            # Load text files
            txt_loader = DirectoryLoader(
                self.files_directory,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf8'}
            )
            txt_docs = txt_loader.load()
            documents.extend(txt_docs)
            
            if not documents:
                print("No PDFs or text files found in knowledge base directory")
                return
                
            print(f"Found {len(pdf_docs)} PDFs and {len(txt_docs)} text files")
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            chunks = text_splitter.split_documents(documents)
            
            # Create FAISS index
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
            
            # Save to disk
            os.makedirs(self.vector_directory, exist_ok=True)
            self.vectorstore.save_local(self.vector_directory)
            print(f"Knowledge base created with {len(chunks)} chunks from {len(documents)} files")
            
        except Exception as e:
            print(f"Error building knowledge base: {e}")
    
    async def search_relevant_knowledge(self, query: str, k: int = 3) -> List[Document]:
        """Search for relevant passages."""
        if not self.vectorstore:
            print("⚠️  Knowledge base not initialized - no search performed")
            return []
        try:
            print(f"🔍 Searching knowledge base with query: '{query[:100]}...'")
            print(f"📚 Retrieving top {k} most relevant documents")
            
            results = self.vectorstore.similarity_search(query, k=k)
            
            if results:
                print(f"✅ Found {len(results)} relevant documents:")
                for i, doc in enumerate(results, 1):
                    source = doc.metadata.get('source', 'Unknown')
                    page = doc.metadata.get('page', 'N/A')
                    print(f"   {i}. Source: {source} (Page: {page})")
                    print(f"      FULL CONTENT RETRIEVED:")
                    print(f"      ======================================")
                    print(f"      {doc.page_content}")
                    print(f"      ======================================")
            else:
                print("❌ No relevant documents found in knowledge base")
            
            return results
        except Exception as e:
            print(f"❌ Error searching knowledge base: {e}")
            return []

class DreamJournalAnalyzer:
    def __init__(self, openai_api_key: str):
        """Initialize the analyzer with OpenAI API key."""
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.llm = ChatOpenAI(temperature=0.8, model_name='gpt-3.5-turbo')
        self.embeddings = OpenAIEmbeddings()
        
        # Initialize knowledge base
        self.knowledge_base = DreamKnowledgeBase(
            files_directory="knowledge_base/files",
            vector_directory="knowledge_base/vectors",
            embeddings=self.embeddings
        )
        
    async def initialize_knowledge_base(self):
        """Call this during service startup."""
        await self.knowledge_base.initialize()

    async def extract_dream_elements(self, entries: List[JournalEntry]) -> str:
        """Extract key themes, symbols, and elements from dream entries."""
        all_content = " ".join([entry.content.lower() for entry in entries])

        # Load extracted dream symbols
        try:
            with open('core/extracted_dream_symbols.json', 'r') as f:
                symbol_data = json.load(f)
            dream_symbols = symbol_data.get('all_symbols_list', [])
            print(f"Loaded {len(dream_symbols)} extracted symbols from knowledge base")
        except FileNotFoundError:
            print("Extracted symbols file not found, using default symbols")
            dream_symbols = [
                "flying", "falling", "water", "ocean", "river", "rain", "swimming",
                "animals", "dog", "cat", "snake", "bird", "horse", "spider",
                "death", "dying", "birth", "baby", "pregnancy",
                "house", "home", "room", "door", "window", "stairs",
                "car", "driving", "train", "airplane", "travel",
                "chasing", "running", "hiding", "escaping", "trapped",
                "fire", "burning", "smoke", "darkness", "light"
            ]
        except Exception as e:
            print(f"Error loading extracted symbols: {e}")
            dream_symbols = []

        found_elements = [symbol for symbol in dream_symbols if symbol in all_content]

        # Add emotional keywords
        emotions = ["fear", "anxiety", "joy", "happiness", "sadness", "anger", "love", "hate", "worry", "peace"]
        found_elements.extend([emotion for emotion in emotions if emotion in all_content])

        result = " ".join(found_elements) if found_elements else "dreams symbols interpretation meaning"
        print(f"Extracted dream elements: {result}")
        return result

    def deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """Remove duplicate documents based on content similarity."""
        if not documents:
            return []
        
        unique_docs = []
        seen_content = set()
        
        for doc in documents:
            # Use first 200 characters as a fingerprint for deduplication
            fingerprint = doc.page_content[:200].strip()
            if fingerprint not in seen_content:
                seen_content.add(fingerprint)
                unique_docs.append(doc)
        
        return unique_docs

    async def enhanced_knowledge_search(self, entries: List[JournalEntry]) -> List[Document]:
        """Multi-stage search pipeline for better knowledge retrieval."""
        print(f"\n=== ENHANCED KNOWLEDGE SEARCH PIPELINE ===")
        all_knowledge_docs = []
        
        # Stage 1: Extract themes and search for specific symbols
        print(f"Stage 1: Extracting and searching dream themes...")
        dream_themes = await self.extract_dream_elements(entries)
        print(f"Extracted themes: {dream_themes}")
        theme_docs = await self.knowledge_base.search_relevant_knowledge(dream_themes, k=2)
        all_knowledge_docs.extend(theme_docs)
        print(f"Found {len(theme_docs)} theme-based documents")
        
        # Stage 2: Search with raw dream content  
        print(f"Stage 2: Searching with combined dream content...")
        combined_content = " ".join([entry.content for entry in entries])
        content_search = combined_content[:500]  # Limit to avoid too long queries
        content_docs = await self.knowledge_base.search_relevant_knowledge(content_search, k=2)
        all_knowledge_docs.extend(content_docs)
        print(f"Found {len(content_docs)} content-based documents")
        
        # Stage 3: Targeted searches for different aspects
        print(f"Stage 3: Targeted searches for emotions and symbols...")
        
        emotion_search = "fear anxiety joy sadness anger dream emotions psychological feelings mood"
        emotion_docs = await self.knowledge_base.search_relevant_knowledge(emotion_search, k=1)
        all_knowledge_docs.extend(emotion_docs)
        print(f"Found {len(emotion_docs)} emotion-focused documents")
        
        symbol_search = "flying water animals death birth transformation symbols meaning interpretation significance"  
        symbol_docs = await self.knowledge_base.search_relevant_knowledge(symbol_search, k=1)
        all_knowledge_docs.extend(symbol_docs)
        print(f"Found {len(symbol_docs)} symbol-focused documents")
        
        # Remove duplicates and return
        unique_docs = self.deduplicate_documents(all_knowledge_docs)
        print(f"After deduplication: {len(unique_docs)} unique documents")
        print(f"=== END ENHANCED SEARCH PIPELINE ===\n")
        
        return unique_docs
        
    async def qa_analysis(self, question: str, entries: List[JournalEntry]) -> str:
        """
        Function 1: Generate cumulative analysis using QA chain over journal entries.
        Equivalent to the qa() function in your JS code.
        """
        try:
            print(f"\n=== Q&A ANALYSIS WITH KNOWLEDGE BASE ===")
            print(f"Question: {question}")
            print(f"Analyzing {len(entries)} journal entries")
            
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
            
            print(f"Created vector store from {len(docs)} journal entries")
            
            # Search knowledge base for context
            print(f"Searching knowledge base for additional context...")
            # knowledge_docs = await self.knowledge_base.search_relevant_knowledge(question, k=2)
            knowledge_docs = await self.enhanced_knowledge_search(entries)

            knowledge_context = ""
            if knowledge_docs:
                print(f"Adding {len(knowledge_docs)} knowledge references to Q&A context")
                knowledge_context = "\n\nReference material:\n"
                for doc in knowledge_docs:
                    snippet = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    snippet = snippet.replace("{", "{{").replace("}", "}}")
                    knowledge_context += f"- {snippet}\n"
            else:
                print("No relevant knowledge found - proceeding with journal entries only")
            
            # Create QA chain with enhanced prompt
            qa_prompt = f"""
            You are a dream interpretation expert. Analyze the journal entries using SPECIFICALLY the dream interpretation theory provided below. You MUST reference and apply these concepts directly in your analysis.

            REQUIRED SOURCE MATERIAL TO USE:
            {knowledge_context}

            Apply the above dream interpretation principles to analyze patterns and themes in these journal entries: {{context}}

            Question: {{question}}

            Answer by directly referencing and applying the dream interpretation theory provided above:"""
            
            print(f"Final Q&A prompt includes:")
            print(f"  - Journal entries context: YES")
            print(f"  - Knowledge base context: {'YES' if knowledge_context else 'NO'}")
            print(f"  - Total context length: {len(qa_prompt)} characters")
            print(f"  - Total context: {knowledge_context}")

            
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
                chain_type_kwargs={"prompt": PromptTemplate.from_template(qa_prompt)}
            )
            
            print(f"Executing Q&A chain...")
            # Get answer
            
            draft = qa_chain.run(question)
            print(f"Stage 1 draft length: {len(draft)} characters")

            # --- Stage 2: Refinement ---
            refine_prompt = f"""
            Here is your first draft interpretation of the dream entries:

            --- BEGIN DRAFT ---
            {draft}
            --- END DRAFT ---

            Refine this analysis by:
            - Do not refer to "the dreamer." Address the reader as if they are your patient. Use "you" and other pronouns.
            - Instead of addressing dreams sequentially, focus on finding patterns linking the dreams together.
            - Making it clearer, structured, and concise
            - Highlighting key symbols and emotional themes
            - Quote the knowledge context if applicable
            - Grounding insights in the dream interpretation theory provided, and reference your citations

            THEORY:
            {knowledge_context}

            Journal Entries: {{context}}

            Return the refined interpretation:
            """

            refine_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
                chain_type_kwargs={"prompt": PromptTemplate.from_template(refine_prompt)}
            )
            refined = refine_chain.run("Refine the draft interpretation.")

            print(f"Stage 2 refinement length: {len(refined)} characters")

            print(f"=== END Q&A ANALYSIS ===\n")
            
            return refined
            
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

    async def custom_question_analysis(self, custom_question: str, entries: List[JournalEntry]) -> str:
        """Handle custom user questions with knowledge base integration."""
        try:
            print(f"\n=== CUSTOM QUESTION ANALYSIS ===")
            print(f"Custom Question: {custom_question}")
            
            # Convert entries to documents
            docs = [
                Document(
                    page_content=entry.content,
                    metadata={"source": entry.id, "date": entry.created_at.isoformat()}
                )
                for entry in entries
            ]
            
            # Create vector store
            vectorstore = FAISS.from_documents(docs, self.embeddings)
            
            # Get knowledge base context
            knowledge_docs = await self.enhanced_knowledge_search(entries)
            knowledge_context = ""
            if knowledge_docs:
                knowledge_context = "\n\nRelevant dream interpretation theory:\n"
                for doc in knowledge_docs:
                    snippet = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                    snippet = snippet.replace("{", "{{").replace("}", "}}")
                    knowledge_context += f"- {snippet}\n"
            
            # Create prompt with user's question
            prompt = f"""
            Answer the following question about the dream journal entries using the provided dream interpretation theory.
            
            {knowledge_context}
            
            Journal Entries: {{context}}
            Question: {custom_question}
            Answer:"""
            
            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff", 
                retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
                chain_type_kwargs={"prompt": PromptTemplate.from_template(prompt)}
            )
            
            result = qa_chain.run(custom_question)
            return result
            
        except Exception as error:
            print(f'Error in custom question analysis: {error}')
            raise Exception('Failed to process custom question')

    async def analyze_entry(self, content: str, personality_type: str = "empathetic") -> JournalAnalysis:
        """
        Function 3: Analyze journal entry with structured output.
        Equivalent to the analyze() function in your JS code.
        """
        try:
            print(f"\n=== KNOWLEDGE BASE SEARCH ===")
            print(f"Searching knowledge base for: '{content[:100]}...'")
            
            # Search for relevant dream interpretation knowledge
            # knowledge_docs = await self.knowledge_base.search_relevant_knowledge(content, k=3)
            fake_entry = JournalEntry(id="temp", created_at=datetime.now(), content=content)
            knowledge_docs = await self.enhanced_knowledge_search([fake_entry])

            print(f"Found {len(knowledge_docs)} relevant knowledge documents")
            
            knowledge_context = ""
            
            if knowledge_docs:
                print(f"Knowledge documents retrieved:")
                knowledge_context = "\n\nRelevant dream interpretation references:\n"
                for i, doc in enumerate(knowledge_docs, 1):
                    # Limit context length
                    snippet = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                    knowledge_context += f"{i}. {snippet}\n"
                    
                    # Log what was found
                    source_file = doc.metadata.get('source', 'Unknown file')
                    print(f"  {i}. From: {source_file}")
                    print(f"     Content preview: {snippet[:150]}...")
            else:
                print("No relevant knowledge found - proceeding with basic analysis")
            
            personality = get_personality(personality_type)
            
            prompt = f"""
            {personality}
            
            Analyze the following dream journal entry. If relevant references are provided below, incorporate insights from established dream interpretation theory into your analysis.
            
            {knowledge_context}
            
            Consider the FULL RANGE of emotions present. Choose the PRIMARY emotion from these options: joy, sadness, anger, fear, surprise, disgust, anxiety, contentment, excitement, melancholy
            
            Do NOT default to excitement - carefully consider which emotion best represents the overall feeling of the dream.

            Examples of mood analysis:
            - Flying dreams often indicate "joy" or "contentment"  
            - Being chased indicates "fear" or "anxiety"
            - Losing something indicates "sadness" or "melancholy"

            Return ONLY a valid JSON response with these exact fields:
            
            {{
                "mood": "choose one: joy, sadness, anger, fear, surprise, disgust, anxiety, contentment, excitement, melancholy",
                "summary": "brief summary of the dream",
                "negative": true or false,
                "subject": "creative title for the dream", 
                "color": "hex color code representing the mood",
                "interpretation": "5-6 sentence analysis incorporating dream theory if available, with song and snack suggestions",
                "sentiment_score": integer from -10 to 10
            }}
            
            Dream Journal Entry: {content}
            
            Return only the JSON object, no other text:
            """
            
            print(f"\n=== FINAL PROMPT TO LLM ===")
            print(f"Prompt length: {len(prompt)} characters")
            print(f"Knowledge context length: {len(knowledge_context)} characters")
            if knowledge_context:
                print(f"Knowledge integration: YES - {len(knowledge_docs)} references included")
            else:
                print(f"Knowledge integration: NO - proceeding without references")
            print(f"Full prompt preview (first 500 chars):")
            print(f"{prompt[:500]}...")
            print(f"=== END PROMPT PREVIEW ===\n")
            
            model = ChatOpenAI(temperature=0.3, model_name='gpt-3.5-turbo')
            result = model.invoke(prompt)
            result_content = result.content
            
            print(f"Raw LLM output: {result_content}")
            
            # Parse JSON directly
            json_data = json.loads(result_content)
            
            # Create JournalAnalysis object manually
            parsed_result = JournalAnalysis(
                mood=EmotionType(json_data['mood']),
                summary=json_data['summary'],
                negative=json_data['negative'],
                subject=json_data['subject'],
                color=json_data['color'],
                interpretation=json_data['interpretation'],
                sentiment_score=json_data['sentiment_score']
            )
            
            print(f"Parsed mood: {parsed_result.mood}")
            
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

    async def refine_analysis(self, draft: str, entries: List[JournalEntry]) -> str:
        """
        Stage 2 refinement: take the first draft analysis and improve it
        using the knowledge base + dream entries for grounding.
        """
        try:
            print(f"\n=== REFINEMENT STAGE ===")
            print(f"Draft length: {len(draft)} characters")
            
            # Get knowledge context again (so model doesn't drift)
            knowledge_docs = await self.enhanced_knowledge_search(entries)
            knowledge_context = ""
            if knowledge_docs:
                knowledge_context = "\n\nRelevant dream interpretation theory:\n"
                for doc in knowledge_docs:
                    snippet = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
                    snippet = snippet.replace("{", "{{").replace("}", "}}")
                    knowledge_context += f"- {snippet}\n"

            prompt = f"""
            You wrote the following first draft analysis of the dream journal entries:

            --- BEGIN DRAFT ---
            {draft}
            --- END DRAFT ---

            Refine and improve this analysis by:
            - Making it clearer, better structured, and more concise
            - Highlighting key symbols and emotional themes
            - Grounding interpretations in the following dream interpretation theory if relevant:
            {knowledge_context}

            Journal Entries for reference:
            {{context}}

            Return the refined interpretation:
            """

            # Create RetrievalQA to keep entries accessible
            docs = [
                Document(
                    page_content=entry.content,
                    metadata={"source": entry.id, "date": entry.created_at.isoformat()}
                )
                for entry in entries
            ]
            vectorstore = FAISS.from_documents(docs, self.embeddings)

            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
                chain_type_kwargs={"prompt": PromptTemplate.from_template(prompt)}
            )

            result = qa_chain.run("Refine the draft interpretation.")
            print(f"Refinement complete. Length: {len(result)} characters")
            return result

        except Exception as e:
            print(f"Error in refinement stage: {e}")
            raise


# Example usage and helper functions
class DreamJournalService:
    def __init__(self, openai_api_key: str):
        self.analyzer = DreamJournalAnalyzer(openai_api_key)
        
    async def initialize(self):
        """Initialize the knowledge base on startup."""
        await self.analyzer.initialize_knowledge_base()
    
    async def get_cumulative_analysis(self, entries: List[JournalEntry]) -> str:
        """Get overall analysis across all entries."""
        question = "Provide a comprehensive analysis of these journal entries, identifying patterns, themes, and emotional trends over time. Reference dream interpretation theory where relevant."
        return await self.analyzer.qa_analysis(question, entries)
    
    async def generate_sample_dream(self, theme: str = "flying") -> str:
        """Generate a sample dream for inspiration."""
        prompt = f"Write a vivid and imaginative dream about {theme}. Make it mysterious and emotionally rich, about 100-150 words."
        return await self.analyzer.ai_generate(prompt)
    
    async def analyze_single_entry(self, content: str, personality: str = "empathetic") -> JournalAnalysis:
        """Analyze a single journal entry."""
        return await self.analyzer.analyze_entry(content, personality)

    async def ask_custom_question(self, question: str, entries: List[JournalEntry]) -> str:
        """Ask a custom question about the dreams."""
        return await self.analyzer.custom_question_analysis(question, entries)

    