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
from asgiref.sync import sync_to_async
from myapp.models import CumulativeAnalysis, CustomQuestion

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

class AstrologyKnowledgeBase:
    """Direct file lookup for astrology references - no vector search needed"""
    def __init__(self, base_directory: str = "knowledge_base/astrology"):
        self.base_directory = base_directory
        
    def get_sign_context(self, sign: str, sign_type: str) -> str:
        """
        Retrieve astrology context for a specific sign.
        sign_type: 'sun', 'moon', or 'rising'
        """
        if not sign:
            return ""
            
        file_path = os.path.join(
            self.base_directory, 
            f"{sign_type}_signs", 
            f"{sign.lower()}.txt"
        )
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"✅ Loaded {sign_type} sign: {sign}")
                return content
        except FileNotFoundError:
            print(f"⚠️  Astrology file not found: {file_path}")
            return ""
        except Exception as e:
            print(f"❌ Error reading astrology file: {e}")
            return ""
    
    def get_full_chart_context(self, sun: str = None, moon: str = None, rising: str = None) -> str:
        """Get combined context for user's full astrological chart"""
        contexts = []
        
        if sun:
            sun_context = self.get_sign_context(sun, "sun")
            if sun_context:
                contexts.append(f"SUN SIGN ({sun.upper()}):\n{sun_context}")
        
        if moon:
            moon_context = self.get_sign_context(moon, "moon")
            if moon_context:
                contexts.append(f"MOON SIGN ({moon.upper()}):\n{moon_context}")
        
        if rising:
            rising_context = self.get_sign_context(rising, "rising")
            if rising_context:
                contexts.append(f"RISING SIGN ({rising.upper()}):\n{rising_context}")
        
        if not contexts:
            return ""
        
        combined = "\n\n".join(contexts)
        print(f"📊 Assembled astrology context: {len(combined)} characters")
        return combined

class PersonalityKnowledgeBase:
    """Direct file lookup for MBTI personality types"""
    def __init__(self, base_directory: str = "knowledge_base/personality"):
        self.base_directory = base_directory
    
    def get_personality_context(self, personality_type: str) -> str:
        """Retrieve MBTI personality context"""
        if not personality_type:
            return ""
        
        file_path = os.path.join(
            self.base_directory,
            f"{personality_type.upper()}.txt"
        )
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                print(f"✅ Loaded personality type: {personality_type}")
                return content
        except FileNotFoundError:
            print(f"⚠️  Personality file not found: {file_path}")
            return ""
        except Exception as e:
            print(f"❌ Error reading personality file: {e}")
            return ""

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

        self.astrology_kb = AstrologyKnowledgeBase()
        self.personality_kb = PersonalityKnowledgeBase()

        
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

    def assemble_full_context( self, dream_theory_docs: List[Document], settings: Dict[str, Any] = None) -> str:
        """
        Assemble full context with proper weighting:
        - Dream theory: 70%
        - Astrology: 15%
        - Personality: 15%
        """
        print(f"\n=== ASSEMBLING FULL CONTEXT ===")
        
        # 70% - Dream theory context
        dream_context = ""
        if dream_theory_docs:
            dream_context = "\n\n=== DREAM INTERPRETATION THEORY (Primary Reference) ===\n"
            for i, doc in enumerate(dream_theory_docs, 1):
                snippet = doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content
                snippet = snippet.replace("{", "{{").replace("}", "}}")
                dream_context += f"\n[Reference {i}]\n{snippet}\n"
            print(f"✅ Dream theory context: {len(dream_context)} chars")
        
        # 15% - Astrology context
        astrology_context = ""
        if settings and settings.get('astrology'):
            astro = settings['astrology']
            astro_text = self.astrology_kb.get_full_chart_context(
                sun=astro.get('sun'),
                moon=astro.get('moon'),
                rising=astro.get('rising')
            )
            if astro_text:
                astrology_context = f"\n\n=== ASTROLOGICAL PROFILE (Secondary Context) ===\n{astro_text}\n"
                print(f"✅ Astrology context: {len(astrology_context)} chars")
        
        # 15% - Personality context
        personality_context = ""
        if settings and settings.get('personality'):
            personality_text = self.personality_kb.get_personality_context(
                settings['personality']
            )
            if personality_text:
                personality_context = f"\n\n=== PERSONALITY PROFILE (Secondary Context) ===\n{personality_text}\n"
                print(f"✅ Personality context: {len(personality_context)} chars")
        
        full_context = dream_context + astrology_context + personality_context
        
        print(f"📊 Total context assembled: {len(full_context)} characters")
        print(f"   - Dream theory: ~{len(dream_context)} chars (~70%)")
        print(f"   - Astrology: ~{len(astrology_context)} chars (~15%)")
        print(f"   - Personality: ~{len(personality_context)} chars (~15%)")
        print(f"=== END CONTEXT ASSEMBLY ===\n")
        
        return full_context

    async def comprehensive_knowledge_retrieval(self, entries: List[JournalEntry], k: int = 20) -> List[Document]:
        """
        Single comprehensive retrieval pass to get ALL relevant knowledge.
        This is more efficient than multiple small searches.
        """
        print(f"\n=== COMPREHENSIVE KNOWLEDGE RETRIEVAL ===")
        
        # Build a rich, semantic query from dream content
        query_components = []
        
        # 1. Extract actual dream elements
        dream_elements = await self.extract_dream_elements(entries)
        query_components.append(dream_elements)
        
        # 2. Include full dream content (don't truncate - embeddings handle this)
        combined_content = " ".join([entry.content for entry in entries[-3:]])  # Last 3 entries
        query_components.append(combined_content)
        
        # 3. Add theoretical framing to guide retrieval
        query_components.append("dream interpretation psychological meaning symbolism theory analysis")
        
        # Build final query
        full_query = " ".join(query_components)
        
        print(f"📝 Query length: {len(full_query)} chars")
        print(f"🔍 Retrieving top {k} documents from knowledge base")
        
        # Single comprehensive search - let embeddings find what's relevant
        docs = await self.knowledge_base.search_relevant_knowledge(full_query, k=k)
        
        # Optional: Use MMR for diversity to avoid redundant similar passages
        if len(docs) > 10:
            docs = await self.apply_mmr_reranking(docs, full_query, lambda_mult=0.7)
        
        print(f"✅ Retrieved {len(docs)} relevant knowledge passages")
        print(f"=== END KNOWLEDGE RETRIEVAL ===\n")
        
        return docs


    async def apply_mmr_reranking(self, docs: List[Document], query: str, lambda_mult: float = 0.5) -> List[Document]:
        """
        Maximal Marginal Relevance re-ranking to get diverse but relevant results.
        Prevents retrieving 20 nearly-identical passages about the same topic.
        """
        if not docs or len(docs) <= 5:
            return docs
        
        # Use vectorstore's MMR search if available
        if hasattr(self.knowledge_base.vectorstore, 'max_marginal_relevance_search'):
            return self.knowledge_base.vectorstore.max_marginal_relevance_search(
                query, 
                k=len(docs), 
                lambda_mult=lambda_mult
            )
        
        # Otherwise return as-is
        return docs


    def assemble_knowledge_context(self, knowledge_docs: List[Document], max_tokens: int = 8000) -> str:
        """
        Assemble knowledge base context efficiently.
        Don't truncate individual docs - let them be full passages.
        """
        print(f"\n=== ASSEMBLING KNOWLEDGE CONTEXT ===")
        
        context_parts = ["=== DREAM INTERPRETATION KNOWLEDGE BASE ===\n"]
        current_length = 0
        docs_included = 0
        
        for i, doc in enumerate(knowledge_docs, 1):
            source = doc.metadata.get('source', 'Unknown')
            
            # Don't truncate - include full passage
            passage = doc.page_content
            passage_tokens = len(passage) // 4  # Rough estimate
            
            # Stop if we'd exceed token budget
            if current_length + passage_tokens > max_tokens:
                print(f"⚠️  Reached token limit, stopping at {docs_included} documents")
                break
            
            # Escape any formatting characters
            passage = passage.replace("{", "{{").replace("}", "}}")
            
            context_parts.append(f"\n[Source {i}: {source}]\n{passage}\n")
            current_length += passage_tokens
            docs_included += 1
        
        context = "\n".join(context_parts)
        
        print(f"✅ Assembled {docs_included}/{len(knowledge_docs)} documents")
        print(f"📊 Approx {current_length} tokens")
        print(f"=== END CONTEXT ASSEMBLY ===\n")
        
        return context


    def assemble_user_context(self, settings: Dict[str, Any] = None) -> str:
        """
        Separate method for user-specific context (astrology, personality, medical).
        Keep this lean - it's supplementary to dream theory.
        """
        context_parts = []
        
        if settings and settings.get('astrology'):
            astro = settings['astrology']
            astro_text = self.astrology_kb.get_full_chart_context(
                sun=astro.get('sun'),
                moon=astro.get('moon'),
                rising=astro.get('rising')
            )
            if astro_text:
                context_parts.append(f"=== ASTROLOGICAL PROFILE ===\n{astro_text}")
        
        if settings and settings.get('personality'):
            personality_text = self.personality_kb.get_personality_context(
                settings['personality']
            )
            if personality_text:
                context_parts.append(f"=== PERSONALITY PROFILE ===\n{personality_text}")
        
        if settings:
            user_bg = []
            if settings.get('occupation'):
                user_bg.append(f"Occupation: {settings['occupation']}")
            if settings.get('medicalHistory'):
                med = settings['medicalHistory']
                if med.get('psychological'):
                    user_bg.append(f"Psychological history: {', '.join(med['psychological'])}")
                if med.get('physical'):
                    user_bg.append(f"Physical health: {', '.join(med['physical'])}")
            
            if user_bg:
                context_parts.append(f"=== USER BACKGROUND ===\n" + "\n".join(user_bg))
        
        return "\n\n".join(context_parts)


    def validate_analysis_quality(self, analysis: str, knowledge_docs: List[Document]) -> Dict[str, Any]:
        """
        Validate that the analysis actually uses the knowledge base.
        Returns metrics about knowledge base usage.
        """
        print(f"\n=== VALIDATING ANALYSIS QUALITY ===")
        
        # Extract key theoretical concepts from knowledge base
        kb_concepts = set()
        analysis_lower = analysis.lower()
        
        # Look for specific theoretical frameworks present in your knowledge base
        theoretical_terms = [
            "subjective dreams", "physical dreams", "spiritual dreams",
            "astral experience", "subjective intelligence", 
            "animal mind", "spiritual mind", "soul",
            "subjective plane", "warning", "prophecy",
            "allegory", "symbols", "telepathic",
            "mental transmissions", "psychic currents"
        ]
        
        concepts_used = [term for term in theoretical_terms if term in analysis_lower]
        
        # Check for forbidden terms (LLM using its own training instead of KB)
        forbidden_terms = [
            "jungian", "jung", "freud", "freudian", 
            "collective unconscious", "shadow self", "anima", "animus",
            "ego", "id", "superego"  # Unless these appear in your KB
        ]
        forbidden_found = [term for term in forbidden_terms if term in analysis_lower]
        
        # Check for unwanted citation language
        citation_language = [
            "quote 1", "quote 2", "quote 3", "quote 4",
            "the knowledge base", "according to the theory",
            "the text states"
        ]
        citations_visible = [phrase for phrase in citation_language if phrase in analysis_lower]
        
        metrics = {
            "theoretical_concepts_used": concepts_used,
            "forbidden_terms_found": forbidden_found,
            "visible_citations": citations_visible,
            "quality_score": len(concepts_used) - len(forbidden_found) - len(citations_visible)
        }
        
        print(f"📊 Analysis Validation Results:")
        print(f"   ✓ Theoretical concepts used: {len(concepts_used)}")
        if concepts_used:
            print(f"      {concepts_used}")
        
        if forbidden_found:
            print(f"   ⚠️  Forbidden terms (using LLM training): {forbidden_found}")
        
        if citations_visible:
            print(f"   ⚠️  Visible citation artifacts: {citations_visible}")
        
        print(f"   📈 Quality score: {metrics['quality_score']}")
        
        if len(concepts_used) < 2:
            print(f"   ⚠️  WARNING: Analysis may not be using knowledge base theories!")
        if forbidden_found:
            print(f"   ⚠️  WARNING: Analysis using LLM training instead of knowledge base!")
        if citations_visible:
            print(f"   ⚠️  WARNING: Citation scaffolding visible in output!")
        
        print(f"=== END VALIDATION ===\n")
        
        return metrics


    async def qa_analysis(self, entries: List[JournalEntry], personality: str = None, settings: Dict[str, Any] = None) -> str:    
        """
        Two-stage analysis: Extract relevant quotes first, then synthesize.
        This forces the LLM to engage with the knowledge base content.
        """
        try:
            print(f"\n=== DREAM ANALYSIS WITH ENHANCED RAG ===")
            
            # Create vectorstore from journal entries
            docs = [
                Document(
                    page_content=entry.content,
                    metadata={"source": entry.id, "date": entry.created_at.isoformat()}
                )
                for entry in entries
            ]
            vectorstore = FAISS.from_documents(docs, self.embeddings)
            
            # COMPREHENSIVE knowledge retrieval
            knowledge_docs = await self.comprehensive_knowledge_retrieval(entries, k=20)
            
            # STAGE 1: Extract relevant theoretical frameworks
            print(f"\n=== STAGE 1: EXTRACTING RELEVANT FRAMEWORKS ===")
            quotes_context = self.assemble_knowledge_context(knowledge_docs, max_tokens=6000)
            
            extraction_prompt = f"""You are analyzing dreams using specific dream interpretation theory. Your task is to identify the THEORETICAL FRAMEWORKS and CONCEPTS from the knowledge base that apply to these dreams.

            KNOWLEDGE BASE:
            {quotes_context}

            DREAM CONTENT:
            {{context}}

            Extract 5-7 THEORETICAL CONCEPTS or FRAMEWORKS that are most relevant:
            - Identify the theoretical classification system (e.g., types of dreams)
            - Extract symbolic meanings for specific elements present in the dreams
            - Note any interpretive frameworks or principles
            - Highlight relevant warnings or prophecies the theory discusses

            For each concept, provide:
            1. The theoretical concept/framework
            2. The specific passage or explanation from the knowledge base
            3. How it applies to these specific dreams

            Example format:
            CONCEPT: Three types of dreams (subjective, physical, spiritual)
            PASSAGE: "There are three pure types of dreams, namely, subjective, physical and spiritual. They relate to the past, present and future..."
            APPLICATION: The time-travel dream showing past and future suggests a spiritual-type dream with prophetic elements.

            Extract the relevant theoretical frameworks:"""

            extract_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": min(6, len(docs))}),
                chain_type_kwargs={"prompt": PromptTemplate.from_template(extraction_prompt)}
            )
            
            relevant_quotes = extract_chain.run("Extract the most relevant theoretical frameworks for these dreams.")
            print(f"✅ Extracted relevant theoretical frameworks:")
            print(relevant_quotes)
            print(f"=== END FRAMEWORK EXTRACTION ===\n")
            
            # STAGE 2: Synthesize analysis using extracted quotes
            print(f"\n=== STAGE 2: SYNTHESIZING ANALYSIS ===")
            
            user_context = self.assemble_user_context(settings)
            
            synthesis_prompt_parts = [
                "You are an expert dream analyst with deep knowledge of dream interpretation theory.",
                "\n\nRELEVANT THEORETICAL FRAMEWORKS FOR THESE DREAMS:",
                relevant_quotes,
            ]
            
            if user_context:
                synthesis_prompt_parts.append(f"\n\nDreamer's profile:\n{user_context}")
            
            if personality:
                synthesis_prompt_parts.append(f"\n\nAnalysis style: {personality}")
            
            synthesis_prompt_parts.append("""
            \n\nDream entries: {context}

            YOUR TASK:
            Write a cohesive dream analysis that naturally integrates the theoretical frameworks listed above.

            STRUCTURE:
            - Paragraph 1: Central pattern across all dreams
            - Paragraphs 2-3: Deep analysis applying the theoretical frameworks (weave the concepts naturally into your interpretation)
            - Paragraph 4: Connection to dreamer's profile and current life situation

            CRITICAL REQUIREMENTS:
            - Apply the CONCEPTS and FRAMEWORKS from above naturally - write as if you've internalized this theory
            - DO NOT use phrases like "Quote 1", "the framework states", "according to the knowledge base"
            - Ground ALL symbolic interpretations in these specific theoretical concepts
            - Write in flowing, authoritative paragraphs addressing the dreamer as "you"
            - NO numbered lists, NO explicit citations

            STYLE:
            Your analysis should read as if written by an expert who naturally thinks in terms of these specific dream theories.
            The theoretical framework should be invisible scaffolding that shapes your insights - not visible citations.

            Analysis:""")
            
            final_prompt = "".join(synthesis_prompt_parts)
            
            synthesis_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": min(6, len(docs))}),
                chain_type_kwargs={"prompt": PromptTemplate.from_template(final_prompt)}
            )
            
            result = synthesis_chain.run("Provide the dream analysis using the quoted theory.")
            
            # Validate that analysis actually uses knowledge base
            validation_metrics = self.validate_analysis_quality(result, knowledge_docs)
            
            print(f"✅ Analysis complete")
            print(f"=== END DREAM ANALYSIS ===\n")
            user_id = settings.get('user_id') if settings else None
            doctor_personality = settings.get('doctorPersonality', '') if settings else ''

            # user = entries[0].user if entries and entries[0].user else None
            cumulative = await sync_to_async(CumulativeAnalysis.objects.create)(
                user_id=user_id,
                analysis=result,
                doctor_personality=doctor_personality
            )
            print(f"💾 Saved cumulative analysis: {cumulative.id}")

            return result
            
        except Exception as error:
            print(f'❌ Error in QA analysis: {error}')
            raise

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

    async def custom_question_analysis(
        self, 
        custom_question: str, 
        entries: List[JournalEntry], 
        personality: str = None, 
        settings: Dict[str, Any] = None
    ) -> str:
        """Handle custom questions with RAG architecture"""
        try:
            print(f"\n=== CUSTOM QUESTION ANALYSIS ===")
            print(custom_question)

            
            docs = [
                Document(
                    page_content=entry.content,
                    metadata={"source": entry.id, "date": entry.created_at.isoformat()}
                )
                for entry in entries
            ]
            
            vectorstore = FAISS.from_documents(docs, self.embeddings)
            
            # Get full context
            dream_theory_docs = await self.enhanced_knowledge_search(entries)
            full_context = self.assemble_full_context(dream_theory_docs, settings)

            personality_instruction = ""
            if personality:
                personality_instruction = f"\n\nResponse Style:\n{personality}\n"

            prompt = f"""
            Answer the following question about the dream journal entries.
            
            Use this instruction to guide your own personality and background: {personality_instruction}
            

            This is the context of dream theory that you can use to inform your response:
            {full_context}
            
            These are the journal entries you can source your data from to inform your answer to the question:
            Journal Entries: {{context}}

            This is the question you must answer:
            Question: {custom_question}
            Answer:"""
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff", 
                retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
                chain_type_kwargs={"prompt": PromptTemplate.from_template(prompt)}
            )
            
            result = qa_chain.run(custom_question)
            user_id = settings.get('user_id') if settings else None
            doctor_personality = settings.get('doctorPersonality', '') if settings else ''

            # user = entries[0].user if entries and entries[0].user else None
            saved_question = await sync_to_async(CustomQuestion.objects.create)(
                user_id=user_id,
                question=custom_question,
                answer=result,
                doctor_personality=doctor_personality,

            )
            print(f"💾 Saved cumulative analysis: {saved_question.id}")

            return result
            
        except Exception as error:
            print(f'Error in custom question: {error}')
            raise

    async def analyze_entry(self, content: str, personality_type: str = "empathetic", settings: Dict[str, Any] = None) -> JournalAnalysis:
        """Analyze single entry with RAG architecture"""
        try:
            fake_entry = JournalEntry(id="temp", created_at=datetime.now(), content=content)
            dream_theory_docs = await self.enhanced_knowledge_search([fake_entry])
            
            full_context = self.assemble_full_context(dream_theory_docs, settings)
            
            personality = get_personality(personality_type)
            
            prompt = f"""
            {personality}
            
            Analyze this dream journal entry using the reference material below.
            
            {full_context}
            
            Choose the PRIMARY emotion from: joy, sadness, anger, fear, surprise, disgust, anxiety, contentment, excitement, melancholy
            
            Return ONLY valid JSON:
            {{
                "mood": "one of the emotions above",
                "summary": "brief summary",
                "negative": true or false,
                "subject": "creative title",
                "color": "hex color",
                "interpretation": "5-6 sentence analysis with song and snack suggestions",
                "sentiment_score": -10 to 10
            }}
            
            Dream: {content}
            
            JSON only:
            """
            
            model = ChatOpenAI(temperature=0.3, model_name='gpt-3.5-turbo')
            result = model.invoke(prompt)
            
            json_data = json.loads(result.content)
            
            parsed_result = JournalAnalysis(
                mood=EmotionType(json_data['mood']),
                summary=json_data['summary'],
                negative=json_data['negative'],
                subject=json_data['subject'],
                color=get_emotion_color(EmotionType(json_data['mood'])),
                interpretation=json_data['interpretation'],
                sentiment_score=json_data['sentiment_score']
            )
            
            return parsed_result
            
        except Exception as error:
            print(f'Failed to analyze: {error}')
            raise

    # async def analyze_entry(self, content: str, personality_type: str = "empathetic") -> JournalAnalysis:
    #     """
    #     Function 3: Analyze journal entry with structured output.
    #     Equivalent to the analyze() function in your JS code.
    #     """
    #     try:
    #         print(f"\n=== KNOWLEDGE BASE SEARCH ===")
    #         print(f"Searching knowledge base for: '{content[:100]}...'")
            
    #         # Search for relevant dream interpretation knowledge
    #         # knowledge_docs = await self.knowledge_base.search_relevant_knowledge(content, k=3)
    #         fake_entry = JournalEntry(id="temp", created_at=datetime.now(), content=content)
    #         knowledge_docs = await self.enhanced_knowledge_search([fake_entry])

    #         print(f"Found {len(knowledge_docs)} relevant knowledge documents")
            
    #         knowledge_context = ""
            
    #         if knowledge_docs:
    #             print(f"Knowledge documents retrieved:")
    #             knowledge_context = "\n\nRelevant dream interpretation references:\n"
    #             for i, doc in enumerate(knowledge_docs, 1):
    #                 # Limit context length
    #                 snippet = doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content
    #                 knowledge_context += f"{i}. {snippet}\n"
                    
    #                 # Log what was found
    #                 source_file = doc.metadata.get('source', 'Unknown file')
    #                 print(f"  {i}. From: {source_file}")
    #                 print(f"     Content preview: {snippet[:150]}...")
    #         else:
    #             print("No relevant knowledge found - proceeding with basic analysis")
            
    #         personality = get_personality(personality_type)
            
    #         prompt = f"""
    #         {personality}
            
    #         Analyze the following dream journal entry. If relevant references are provided below, incorporate insights from established dream interpretation theory into your analysis.
            
    #         {knowledge_context}
            
    #         Consider the FULL RANGE of emotions present. Choose the PRIMARY emotion from these options: joy, sadness, anger, fear, surprise, disgust, anxiety, contentment, excitement, melancholy
            
    #         Do NOT default to excitement - carefully consider which emotion best represents the overall feeling of the dream.

    #         Examples of mood analysis:
    #         - Flying dreams often indicate "joy" or "contentment"  
    #         - Being chased indicates "fear" or "anxiety"
    #         - Losing something indicates "sadness" or "melancholy"

    #         Return ONLY a valid JSON response with these exact fields:
            
    #         {{
    #             "mood": "choose one: joy, sadness, anger, fear, surprise, disgust, anxiety, contentment, excitement, melancholy",
    #             "summary": "brief summary of the dream",
    #             "negative": true or false,
    #             "subject": "creative title for the dream", 
    #             "color": "hex color code representing the mood",
    #             "interpretation": "5-6 sentence analysis incorporating dream theory if available, with song and snack suggestions",
    #             "sentiment_score": integer from -10 to 10
    #         }}
            
    #         Dream Journal Entry: {content}
            
    #         Return only the JSON object, no other text:
    #         """
            
    #         print(f"\n=== FINAL PROMPT TO LLM ===")
    #         print(f"Prompt length: {len(prompt)} characters")
    #         print(f"Knowledge context length: {len(knowledge_context)} characters")
    #         if knowledge_context:
    #             print(f"Knowledge integration: YES - {len(knowledge_docs)} references included")
    #         else:
    #             print(f"Knowledge integration: NO - proceeding without references")
    #         print(f"Full prompt preview (first 500 chars):")
    #         print(f"{prompt[:500]}...")
    #         print(f"=== END PROMPT PREVIEW ===\n")
            
    #         model = ChatOpenAI(temperature=0.3, model_name='gpt-3.5-turbo')
    #         result = model.invoke(prompt)
    #         result_content = result.content
            
    #         print(f"Raw LLM output: {result_content}")
            
    #         # Parse JSON directly
    #         json_data = json.loads(result_content)
            
    #         # Create JournalAnalysis object manually
    #         parsed_result = JournalAnalysis(
    #             mood=EmotionType(json_data['mood']),
    #             summary=json_data['summary'],
    #             negative=json_data['negative'],
    #             subject=json_data['subject'],
    #             color=json_data['color'],
    #             interpretation=json_data['interpretation'],
    #             sentiment_score=json_data['sentiment_score']
    #         )
            
    #         print(f"Parsed mood: {parsed_result.mood}")
            
    #         # Set the color based on mood
    #         parsed_result.color = get_emotion_color(parsed_result.mood)
            
    #         return parsed_result
            
    #     except Exception as error:
    #         print(f'Failed to parse analysis result: {error}')
    #         raise Exception('Failed to analyze dream journal entry')

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
    
    async def get_cumulative_analysis(self, entries: List[JournalEntry], personality: str = None, settings: Dict[str, Any] = None ) -> str:
        """Get overall analysis across all entries."""
        question = "Provide a comprehensive analysis of these journal entries, identifying patterns, themes, and emotional trends over time. Reference dream interpretation theory where relevant."
        return await self.analyzer.qa_analysis(entries, personality, settings)
    
    async def generate_sample_dream(self, theme: str = "flying") -> str:
        """Generate a sample dream for inspiration."""
        prompt = f"Write a vivid and imaginative dream about {theme}. Make it mysterious and emotionally rich, about 100-150 words."
        return await self.analyzer.ai_generate(prompt)
    
    async def analyze_single_entry(self, content: str, personality: str = "empathetic") -> JournalAnalysis:
        """Analyze a single journal entry."""
        return await self.analyzer.analyze_entry(content, personality)

    async def ask_custom_question(self, question: str, entries: List[JournalEntry], personality: str = None, settings: Dict[str, Any] = None) -> str:
        """Ask a custom question about the dreams."""
        return await self.analyzer.custom_question_analysis(question, entries, personality, settings)

    