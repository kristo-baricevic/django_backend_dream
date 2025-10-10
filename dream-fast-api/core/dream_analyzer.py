import os
import json
from typing import List, Dict, Any, Optional, Tuple
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
from myapp.models import CumulativeAnalysis, CustomQuestion, DoctorProfile
from core.workflow_tracker import WorkflowTracker
import re
from dataclasses import dataclass
import spacy
from collections import Counter
from typing import List, Dict, Set
import re

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
    doctor_personality: str = Field(description="the doctor personality used for this analysis")  # ADD THIS
    weights: Dict[str, float] = Field(description="the final weights used for this analysis")  # ADD THIS

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
        """Load existing vectors or create new ones if files changed."""
        index_path = os.path.join(self.vector_directory, "index.faiss")
        
        if os.path.exists(index_path):
            if self._should_rebuild():  # ← ADD THIS CHECK
                print("📦 Files changed - rebuilding vectorstore...")
                await self.build_knowledge_base()
            else:
                print("✅ Loading existing vectorstore...")
                try:
                    self.vectorstore = FAISS.load_local(
                        self.vector_directory, 
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
                except Exception as e:
                    print(f"Failed to load, rebuilding: {e}")
                    await self.build_knowledge_base()
        else:
            print("🔨 Building new vectorstore...")
            await self.build_knowledge_base()
    
    def _should_rebuild(self) -> bool:  # ← ADD THIS METHOD
        """Check if any files are newer than the existing index."""
        index_path = os.path.join(self.vector_directory, "index.faiss")
        
        if not os.path.exists(index_path):
            return True
        
        index_time = os.path.getmtime(index_path)
        
        for root, dirs, files in os.walk(self.files_directory):
            for file in files:
                if file.endswith(('.pdf', '.txt')):
                    file_path = os.path.join(root, file)
                    if os.path.getmtime(file_path) > index_time:
                        print(f"🆕 Detected newer file: {file}")
                        return True
        
        return False
    
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
        """Search with minimal logging."""
        if not self.vectorstore:
            return []
        
        results = self.vectorstore.similarity_search(query, k=k)
        
        # Log only summary, not full content!
        if results:
            print(f"✅ Found {len(results)} documents from {len(set(d.metadata.get('source') for d in results))} sources")
        
        return results

class DreamJournalAnalyzer:
    def __init__(self, openai_api_key: str):
        """Initialize the analyzer with OpenAI API key."""
        os.environ["OPENAI_API_KEY"] = openai_api_key
        self.llm = ChatOpenAI(temperature=0.8, model_name='gpt-3.5-turbo')
        self.embeddings = OpenAIEmbeddings()
        self.symbol_extractor = DreamSymbolExtractor()

        # Initialize knowledge base
        self.knowledge_base = DreamKnowledgeBase(
            files_directory="knowledge_base",
            vector_directory="knowledge_base/vectors",
            embeddings=self.embeddings
        )

        self.astrology_kb = AstrologyKnowledgeBase()
        self.personality_kb = PersonalityKnowledgeBase()

    @dataclass
    class DoctorProfile:
        name: str
        description: str
        raw_text: str
        weights: Dict[str, float]

    DEFAULT_PROFILE = DoctorProfile(
        name="Academic",
        description="Fallback academic profile.",
        raw_text="",
        weights={"theory": 0.7, "astrology": 0.15, "personality": 0.15, "medicalHistory": 0.0},
    )
    
    
    async def initialize_knowledge_base(self):
        """Call this during service startup."""
        await self.knowledge_base.initialize()


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
        dream_themes = await self.symbol_extractor.extract_dream_elements(entries)
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

    def _take_fraction(self, text: str, frac: float) -> str:
        if frac <= 0: return ""
        frac = min(max(frac, 0.0), 1.0)
        n = max(0, int(len(text) * frac))
        return text[:n]

    def _build_medical_context(self, settings: Dict[str, Any]) -> str:
        if not settings or not settings.get("medicalHistory"):
            return ""
        med = settings["medicalHistory"]
        parts = []
        if med.get("psychological"):
            parts.append("Psychological history: " + ", ".join(med["psychological"]))
        if med.get("physical"):
            parts.append("Physical health: " + ", ".join(med["physical"]))
        return ("=== MEDICAL HISTORY ===\n" + "\n".join(parts)) if parts else ""

    def _blend(self, a: float, b: float, t: float) -> float:
        return a + t * (b - a)

    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        s = sum(max(0.0, v) for v in weights.values())
        if s <= 0:
            # defensively default to theory
            return {"theory": 1.0, "astrology": 0.0, "personality": 0.0, "medicalHistory": 0.0}
        return {k: max(0.0, v) / s for k, v in weights.items()}

    def _compute_final_weights(self,
                            user_inf: Dict[str, float],
                            doctor_w: Dict[str, float],
                            doctor_influence: float) -> Dict[str, float]:
        # Neutral baseline (your previous documented split, no medical):
        base = {"theory": 0.70, "astrology": 0.15, "personality": 0.15, "medicalHistory": 0.0}

        blended = {k: self._blend(base[k], doctor_w.get(k, 0.0), doctor_influence) for k in base.keys()}

        # Apply user sliders as multiplicative factors
        raw = {
            "theory": blended["theory"] * max(0.0, 1.0),  # theory has no direct user slider; keep 1.0 or add if you want
            "astrology": blended["astrology"] * max(0.0, user_inf.get("astrology", 0.15)),
            "personality": blended["personality"] * max(0.0, user_inf.get("personality", 0.15)),
            "medicalHistory": blended["medicalHistory"] * max(0.0, user_inf.get("medicalHistory", 0.10)),
        }

        return self._normalize_weights(raw)


    async def assemble_full_context(self, dream_theory_docs: List[Document], settings: Dict[str, Any] = None, weights: Optional[Dict[str, float]] = None) -> str:
        print("\n=== ASSEMBLING FULL CONTEXT (RAG Doctor + Weights) ===")

        settings = settings or {}
        doctor_name = settings.get("doctorPersonality", "Academic")
        user_inf = settings.get("influence", {})
        doctor_influence = float(user_inf.get("doctor", 0.7))  # 0..1

        # 1) RAG: fetch doctor profile (rich text + weights)
        profile = await self._get_doctor_profile(doctor_name)
        final_w = self._compute_final_weights(user_inf, profile.weights, doctor_influence)

        print(f"🩺 Doctor: {profile.name} | weights(raw)={profile.weights} | user={user_inf} | doctorInfluence={doctor_influence}")
        print(f"🎚 Final normalized weights => {final_w}")

        # 2) DREAM THEORY (weighted by final_w['theory'])
        dream_context = ""
        if dream_theory_docs:
            dream_context = "\n\n=== DREAM INTERPRETATION THEORY (Primary) ===\n"
            for i, doc in enumerate(dream_theory_docs, 1):
                snippet = doc.page_content.replace("{", "{{").replace("}", "}}")
                snippet = self._take_fraction(snippet, final_w["theory"])
                if not snippet: break
                dream_context += f"\n[Reference {i}]\n{snippet}\n"

        # 3) ASTROLOGY (weighted)
        astrology_context = ""
        if settings.get("astrology"):
            astro = settings["astrology"]
            astro_text = self.astrology_kb.get_full_chart_context(
                sun=astro.get("sun"), moon=astro.get("moon"), rising=astro.get("rising")
            )
            if astro_text:
                astro_text = self._take_fraction(astro_text, final_w["astrology"])
                if astro_text:
                    astrology_context = f"\n\n=== ASTROLOGICAL PROFILE === (w={final_w['astrology']:.2f})\n{astro_text}\n"

        # 4) PERSONALITY (MBTI) (weighted)
        personality_context = ""
        if settings.get("personality"):
            ptxt = self.personality_kb.get_personality_context(settings["personality"])
            if ptxt:
                ptxt = self._take_fraction(ptxt, final_w["personality"])
                if ptxt:
                    personality_context = f"\n\n=== PERSONALITY PROFILE === (w={final_w['personality']:.2f})\n{ptxt}\n"

        # 5) MEDICAL HISTORY (weighted)
        medical_context = self._build_medical_context(settings)
        if medical_context:
            medical_context = self._take_fraction(medical_context, final_w["medicalHistory"])

        # 6) DOCTOR VOICE (full, un-truncated guidance at the end)
        doctor_voice = f"\n\n=== DOCTOR VOICE ===\n{profile.raw_text}\n"

        full = dream_context + astrology_context + personality_context + (("\n\n" + medical_context) if medical_context else "") + doctor_voice

        print(f"📊 Assembled context lengths — "
            f"dream:{len(dream_context)} astro:{len(astrology_context)} "
            f"pers:{len(personality_context)} med:{len(medical_context)} "
            f"voice:{len(doctor_voice)} total:{len(full)}")
        print("=== END CONTEXT ASSEMBLY ===\n")
        return full

    async def comprehensive_knowledge_retrieval(
        self, 
        entries: List[JournalEntry], 
        k: int = 20,
        total_k: int = 20
    ) -> List[Document]:
        """
        Multi-query retrieval using structured dream elements.
        """
        print(f"\n=== DIVERSE KNOWLEDGE RETRIEVAL ===")
        
        dream_elements = await self.symbol_extractor.extract_dream_elements(entries)
        
        # Build diverse, focused queries from structured data
        queries = self._build_queries_from_elements(dream_elements)
        
        all_docs = []
        seen_content = set()
        source_counts = {}
        
        for query in queries:
            print(f"  Querying: {query[:60]}...")
            
            docs = await self.knowledge_base.search_relevant_knowledge(query, k=k)
            
            for doc in docs:
                # Deduplicate by content hash
                content_hash = hash(doc.page_content[:200])
                source = doc.metadata.get('source', 'unknown')
                
                if content_hash in seen_content:
                    continue
                if source_counts.get(source, 0) >= 3:
                    continue
                
                seen_content.add(content_hash)
                source_counts[source] = source_counts.get(source, 0) + 1
                all_docs.append(doc)
                
                if len(all_docs) >= total_k:
                    break
            
            if len(all_docs) >= total_k:
                break
        
        print(f"✅ Retrieved {len(all_docs)} passages from {len(source_counts)} sources")
        print(f"   Source distribution: {source_counts}")
        print(f"Total docs in vectorstore: {len(self.knowledge_base.vectorstore.docstore._dict)}")
        print(f"Sources: {set(doc.metadata.get('source') for doc in self.knowledge_base.vectorstore.docstore._dict.values())}")

        print(f"=== END RETRIEVAL ===\n")
        
        return all_docs


    def _build_queries_from_elements(self, dream_elements: dict) -> List[str]:
        """
        Build targeted queries from structured dream elements.
        """
        queries = []
        
        # 1. Symbol-based queries (primary_symbols)
        symbols = dream_elements.get('primary_symbols', [])[:3]  # Top 3
        for symbol in symbols:
            if symbol and len(symbol) > 2:  # Skip very short/generic words
                queries.append(f"{symbol} symbolism dreams Jungian interpretation")
                queries.append(f"what does {symbol} represent in dream analysis")
        
        # 2. Key phrase queries (more specific than single symbols)
        phrases = dream_elements.get('key_phrases', [])[:3]
        for phrase in phrases:
            if phrase and len(phrase) > 5:
                queries.append(f"{phrase} dream meaning psychological significance")
        
        # 3. Emotion-based queries
        emotions = dream_elements.get('emotions', [])
        unique_emotions = list(set(e['emotion'] for e in emotions))[:2]  # Top 2 unique
        for emotion in unique_emotions:
            queries.append(f"{emotion} in dreams emotional processing theory")
        
        # 4. Action-based queries
        actions = dream_elements.get('actions', [])[:2]
        for action in actions:
            if action and len(action) > 3:
                queries.append(f"{action} action dreams behavioral symbolism")
        
        # 5. Entity/location queries
        entities = dream_elements.get('entities', [])
        locations = [e[0] for e in entities if e[1] == 'LOC'][:2]
        for loc in locations:
            queries.append(f"{loc} setting dreams environmental symbolism")
        
        # 6. Thematic/general queries (as fallback)
        queries.extend([
            "recurring dream patterns archetypal meaning",
            "dream emotions unconscious mind processing",
            "symbolic dream interpretation depth psychology"
        ])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique_queries.append(q)
        
        return unique_queries[:8]  # Return top 8 diverse queries

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
            
    async def analyze_entry(
        self,
        content: str,
        settings: Dict[str, Any] = None
    ) -> JournalAnalysis:
        """Analyze single entry with doctor personality weighting and RAG architecture"""
        from myapp.models import DoctorProfile
        from asgiref.sync import sync_to_async

        try:
            fake_entry = JournalEntry(id="temp", created_at=datetime.now(), content=content)
            dream_theory_docs = await self.comprehensive_knowledge_retrieval([fake_entry], k=20)  # ← CHANGE THIS


            # === Load doctor profile & compute final weights ===
            doctor_name = settings.get("doctorPersonality", "Academic") if settings else "Academic"
            print(f"doctor name === {doctor_name}")
            print(f"settings === {settings}")

            # Try database first, then fallback to vectorstore profile
            # doctor_profile = await sync_to_async(DoctorProfile.objects.filter(name__iexact=doctor_name).first)()
            # if doctor_profile:
            #     profile = doctor_profile
            # else:
            profile = await self._get_doctor_profile(doctor_name)
            profile_weights = profile.weights

            print(f"doctor profile === {profile}")
            user_inf = settings.get("influence", {}) if settings else {}
            doctor_influence = settings.get("doctor_influence", 0.5) if settings else 0.5
            print(f"user_inf === === {user_inf}")
            print(f"doctor_influence === === {doctor_influence}")
            print(f"profile_weights === === {profile_weights}")

            final_weights = self._compute_final_weights(
                user_inf=user_inf,
                doctor_w=profile_weights,
                doctor_influence=doctor_influence,
            )

            full_context = await self.assemble_full_context(
                dream_theory_docs=dream_theory_docs,
                settings=settings,
                weights=final_weights
            )

            print(f"profile_weights === === {final_weights}")


            # Doctor’s tone and description
            doctor_intro = f"You are {profile.name}, a dream analyst.\n{profile.description}\n\n"

            prompt = f"""
            {doctor_intro}
            
            Analyze this dream journal entry using the weighted reference material below.

            WEIGHTED CONTEXT (based on doctor & user influence):
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
                sentiment_score=json_data['sentiment_score'],
                doctor_personality = doctor_name,
                weights = final_weights
            )

            return parsed_result

        except Exception as error:
            print(f'❌ Failed to analyze entry: {error}')
            raise

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
    
    async def qa_analysis_with_workflow(
        self,
        entries: List[Dict],
        settings: Dict = None,
        existing_workflow_id: str = None
    ) -> Tuple[str, str]:
        """Cumulative analysis with workflow tracking"""
        from myapp.models import DoctorProfile, CumulativeAnalysis, WorkflowExecution
        from asgiref.sync import sync_to_async

        user_id = settings.get('user_id') if settings else None

        # Workflow setup
        if existing_workflow_id:
            execution = await sync_to_async(WorkflowExecution.objects.get)(id=existing_workflow_id)
            tracker = WorkflowTracker.__new__(WorkflowTracker)
            tracker.workflow_type = execution.workflow_type
            tracker.routine_name = execution.routine_name
            tracker.user_id = execution.user_id
            tracker.execution = execution
            tracker.current_step_number = 0
            workflow_id = existing_workflow_id
        else:
            tracker = WorkflowTracker(
                workflow_type="cumulative_analysis",
                routine_name="Dream Analysis",
                user_id=settings.get('user_id') if settings else None
            )
            workflow_id = await tracker.start_workflow()

        # === LOAD DOCTOR PROFILE & COMPUTE FINAL WEIGHTS ===
        profile = None
        try:
            # Only import if you actually created this model. If not, keep the except.
            from myapp.models import DoctorProfile as DoctorProfileModel  # JSONField 'weights' expected
            doctor_name = (settings or {}).get("doctorPersonality", "Academic")
            doctor_profile = await sync_to_async(
                DoctorProfileModel.objects.filter(name__iexact=doctor_name).first
            )()
            if doctor_profile:
                # Ensure it has the shape we need
                profile = type("TmpProfile", (), {})()
                profile.name = doctor_profile.name or "Doctor"
                profile.description = doctor_profile.description or ""
                profile.raw_text = getattr(doctor_profile, "raw_text", "") or ""
                profile.weights = doctor_profile.weights or {
                    "theory": 0.7, "astrology": 0.15, "personality": 0.15, "medicalHistory": 0.0
                }
        except Exception:
            pass

        if profile is None:
            profile = self.DEFAULT_PROFILE

        user_inf = (settings or {}).get("influence", {})
        doctor_influence = float((settings or {}).get("doctor_influence", 0.5))

        final_weights = self._compute_final_weights(
            user_inf=user_inf,
            doctor_w=profile.weights,
            doctor_influence=doctor_influence,
        )

        print(f"🔮Final blended weights for {doctor_name}: {final_weights}")

        try:
            # STEP 1: Create vectorstore from entries
            step1 = await tracker.start_step(
                name="Build Dream Vector Database",
                step_type="vectorstore_creation",
                input_data={"entry_count": len(entries)}
            )

            docs = [
                Document(
                    page_content=entry.content,
                    metadata={"source": entry.id, "date": entry.created_at.isoformat()}
                )
                for entry in entries
            ]
            vectorstore = FAISS.from_documents(docs, self.embeddings)

            await step1.complete(
                output={"documents_processed": len(docs)},
                confidence=1.0,
                reasoning="Successfully created vector database from dream entries"
            )

            # STEP 2: Knowledge base retrieval
            step2 = await tracker.start_step(
                name="Search Dream Theory Knowledge Base",
                step_type="knowledge_retrieval"
            )

            knowledge_docs = await self.comprehensive_knowledge_retrieval(entries, k=20)

            citations = [
                {
                    'source': 'dream_science_papers',
                    'content': doc.page_content[:200],
                    'confidence': 0.85,
                    'reference': doc.metadata.get('source', 'Unknown')
                }
                for doc in knowledge_docs[:5]
            ]

            await step2.complete(
                output={"documents_found": len(knowledge_docs)},
                confidence=0.9 if knowledge_docs else 0.3,
                reasoning=f"Retrieved {len(knowledge_docs)} relevant dream interpretation passages",
                citations=citations
            )

            # STEP 3: Extract theoretical frameworks
            step3 = await tracker.start_step(
                name="Extract Theoretical Frameworks",
                step_type="framework_extraction"
            )

            quotes_context = self.assemble_knowledge_context(knowledge_docs, max_tokens=6000)

            extraction_prompt = f"""You are analyzing dreams using specific dream interpretation theory...
            {quotes_context}
            {{context}}
            Extract the most relevant theoretical frameworks for these dreams:"""

            extract_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": min(6, len(docs))}),
                chain_type_kwargs={"prompt": PromptTemplate.from_template(extraction_prompt)}
            )

            relevant_quotes = extract_chain.run("Extract the most relevant theoretical frameworks for these dreams.")

            await step3.complete(
                output={"frameworks_extracted": len(relevant_quotes.split('\n'))},
                confidence=0.85,
                reasoning="Extracted key theoretical frameworks from knowledge base",
                model="gpt-3.5-turbo",
                tokens=len(relevant_quotes) // 4
            )

            # STEP 4: Load user context
            step4 = await tracker.start_step(
                name="Load User Profile Context",
                step_type="user_context"
            )

            user_context = self.assemble_user_context(settings)

            astro_citations = []
            if settings and settings.get('astrology'):
                astro = settings['astrology']
                if astro.get('sun'):
                    astro_citations.append({
                        'source': 'natal_chart',
                        'content': f"Sun in {astro['sun']}",
                        'confidence': 1.0,
                        'reference': 'User Natal Chart'
                    })

            await step4.complete(
                output={"context_loaded": bool(user_context)},
                confidence=1.0,
                reasoning="Loaded astrology and personality context for dreamer",
                citations=astro_citations
            )

            # STEP 5: Synthesize final analysis (using weighted context)
            step5 = await tracker.start_step(
                name="Synthesize Final Interpretation",
                step_type="synthesis"
            )

            full_context = await self.assemble_full_context(
                dream_theory_docs=knowledge_docs,
                settings=settings,
                weights=final_weights,
            )

            synthesis_prompt = f"""You are {profile.name}, a dream analyst.
            {profile.description}

            WEIGHTED CONTEXT (based on doctor & user influence):
            {full_context}

            RELEVANT THEORETICAL FRAMEWORKS:
            {relevant_quotes}

            {user_context if user_context else ''}

            Dream entries: {{context}}

            Analysis:"""

            synthesis_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": min(6, len(docs))}),
                chain_type_kwargs={"prompt": PromptTemplate.from_template(synthesis_prompt)}
            )

            result = synthesis_chain.run("Provide the dream analysis using the weighted theoretical context.")

            await step5.complete(
                output={"analysis_length": len(result)},
                confidence=0.87,
                reasoning="Synthesized comprehensive interpretation from all sources",
                model="gpt-3.5-turbo",
                tokens=len(result) // 4
            )

            # STEP 6: Validate analysis
            step6 = await tracker.start_step(
                name="Validate Analysis Quality",
                step_type="validation"
            )

            validation_metrics = self.validate_analysis_quality(result, knowledge_docs)

            await step6.complete(
                output=validation_metrics,
                confidence=validation_metrics['quality_score'] / 10,
                reasoning=f"Quality score: {validation_metrics['quality_score']}"
            )

            # Complete workflow
            total_citations = len(citations) + len(astro_citations)
            await tracker.complete_workflow(
                result=result,
                confidence=0.87,
                total_citations=total_citations
            )

            # Save cumulative analysis
            doctor_personality = settings.get('doctorPersonality', '') if settings else ''
            cumulative = await sync_to_async(CumulativeAnalysis.objects.create, thread_sensitive=True)(
                user_id=user_id,
                analysis=result,
                doctor_personality=doctor_personality,
                weights=final_weights,
                workflow_execution_id=workflow_id
            )

            return result, workflow_id

        except Exception as error:
            print(f'❌ Error in QA analysis: {error}')
            await tracker.fail_workflow(str(error))
            raise

    async def custom_question_with_workflow(
        self,
        question: str,
        entries: List[JournalEntry],
        settings: Dict[str, Any] = None,
        existing_workflow_id: str = None
    ) -> tuple[str, str]:
        """Enhanced custom question analysis with doctor weighting and workflow tracking."""
        from myapp.models import DoctorProfile, CustomQuestion, WorkflowExecution
        from asgiref.sync import sync_to_async

        user_id = settings.get('user_id') if settings else None
        tracker = WorkflowTracker(
            workflow_type='custom_question',
            routine_name=f'Custom Q&A: {question[:50]}...',
            user_id=user_id
        )

        if existing_workflow_id:
            execution = await sync_to_async(WorkflowExecution.objects.get)(id=existing_workflow_id)
            tracker.execution = execution
            workflow_id = existing_workflow_id
        else:
            workflow_id = await tracker.start_workflow()

        # === LOAD DOCTOR PROFILE & COMPUTE FINAL WEIGHTS ===
        doctor_name = settings.get("doctorPersonality", "Academic") if settings else "Academic"

        # Try database first, then fallback to vectorstore profile
        doctor_profile = await sync_to_async(DoctorProfile.objects.filter(name__iexact=doctor_name).first)()
        if doctor_profile:
            profile = doctor_profile
        else:
            profile = await self._get_doctor_profile(doctor_name)

        user_inf = settings.get("influence", {}) if settings else {}
        doctor_influence = float(settings.get("doctor_influence", 0.5)) if settings else 0.5

        final_weights = self._compute_final_weights(
            user_inf=user_inf,
            doctor_w=profile.weights,
            doctor_influence=doctor_influence,
        )
        print(f"🔮 Final user_inf weights: {user_inf}")

        print(f"🔮 Final blended weights for {doctor_name}: {final_weights}")

        try:
            # STEP 1: Prepare dream entries
            step1 = await tracker.start_step(
                name="Prepare Dream Entries",
                step_type="data_preparation",
                input_data={"question": question, "entry_count": len(entries)}
            )

            docs = [
                Document(
                    page_content=entry.content,
                    metadata={"source": entry.id, "date": entry.created_at.isoformat()}
                )
                for entry in entries
            ]
            vectorstore = FAISS.from_documents(docs, self.embeddings)

            await step1.complete(
                output={"entries_processed": len(docs)},
                confidence=1.0,
                reasoning="Prepared dream entries for analysis"
            )

            # STEP 2: Search knowledge base
            step2 = await tracker.start_step(
                name="Search Relevant Dream Theory",
                step_type="knowledge_search"
            )

            dream_theory_docs = await self.comprehensive_knowledge_retrieval(entries)

            await step2.complete(
                output={"theory_docs_found": len(dream_theory_docs)},
                confidence=0.8,
                reasoning="Found relevant dream interpretation theory"
            )

            # STEP 3: Assemble weighted context
            step3 = await tracker.start_step(
                name="Assemble Full Context (Weighted)",
                step_type="context_assembly"
            )

            full_context = await self.assemble_full_context(
                dream_theory_docs=dream_theory_docs,
                settings=settings,
                weights=final_weights
            )

            await step3.complete(
                output={"context_size": len(full_context)},
                confidence=1.0,
                reasoning="Assembled weighted context using doctor and user influence"
            )

            # STEP 4: Generate answer
            step4 = await tracker.start_step(
                name="Generate Answer",
                step_type="answer_generation"
            )

            personality_instruction = ""
            if settings and settings.get("doctorPersonality"):
                personality_instruction = f"\n\nResponse Style:\n{settings['doctorPersonality']}\n"


            prompt = f"""You are {profile.name}, a dream analyst.
            {profile.description}

            WEIGHTED CONTEXT (based on doctor & user influence):
            {full_context}

            {personality_instruction}

            Journal Entries: {{context}}
            Question: {question}
            Answer:"""

            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
                chain_type_kwargs={"prompt": PromptTemplate.from_template(prompt)}
            )

            result = qa_chain.run(question)

            await step4.complete(
                output={"answer_length": len(result)},
                confidence=0.85,
                reasoning="Generated answer using weighted dream context and entries",
                model="gpt-3.5-turbo"
            )

            # Complete workflow
            await tracker.complete_workflow(result=result, confidence=0.85)

            # Save CustomQuestion result
            doctor_personality = settings.get('doctorPersonality', '') if settings else ''
            saved_question = await sync_to_async(CustomQuestion.objects.create)(
                user_id=user_id,
                question=question,
                answer=result,
                doctor_personality=doctor_personality,
                weights=final_weights,
                workflow_execution_id=workflow_id
            )

            return result, workflow_id

        except Exception as error:
            print(f'❌ Error in custom_question_with_workflow: {error}')
            await tracker.fail_workflow(str(error))
            raise
   
    def _parse_weights_from_text(self, text: str) -> Dict[str, float]:
        """
        Robust parser: handles a 'Weights:' block with lines like 'theory: 0.7'.
        Falls back to 0 if missing. Does not require YAML.
        """
        weights = {
            "theory": 0.0,
            "astrology": 0.0,
            "personality": 0.0,
            "medicalHistory": 0.0,
        }
        
        # Find the Weights: section
        if "Weights:" not in text:
            print(f"block_match === None (no 'Weights:' found)")
            print(f"w ==www== {weights}")
            return weights
        
        # Split text into lines and find weights section
        lines = text.split('\n')
        in_weights_section = False
        
        for line in lines:
            if 'Weights:' in line:
                in_weights_section = True
                continue
                
            if in_weights_section:
                # Stop when we hit a non-indented line (end of weights block)
                if line and not line[0].isspace():
                    break
                    
                # Try to parse each weight key
                for key in weights.keys():
                    # Match pattern: "key: 0.6" (with any amount of whitespace)
                    match = re.search(rf'{key}\s*:\s*([0-9]*\.?[0-9]+)', line, re.IGNORECASE)
                    if match:
                        weights[key] = float(match.group(1))
                        break
        
        print(f"block_match === {in_weights_section}")
        print(f"w ==www== {weights}")
        return weights

        return w

    async def _get_doctor_profile(self, name: str) -> DoctorProfile:
        """Load doctor profile directly from file."""
        file_path = f"knowledge_base/doctor_profiles/{name.lower()}.txt"
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                
            print(f"✅ Loaded doctor profile from: {file_path}")
            
            # Parse weights
            weights = self._parse_weights_from_text(text)
            
            # Parse name
            nm = re.search(r"(?im)^Name\s*:\s*(.+)$", text, re.MULTILINE)
            profile_name = nm.group(1).strip() if nm else name
            
            # Parse description (everything between "Background:" and "Weights:")
            desc_match = re.search(r"Background:(.*?)(?=Weights:)", text, re.DOTALL | re.IGNORECASE)
            description = desc_match.group(1).strip() if desc_match else "Dream analyst profile."
            
            return self.DoctorProfile(
                name=profile_name,
                description=description,
                raw_text=text,
                weights=weights
            )
            
        except FileNotFoundError:
            print(f"⚠️ Profile file not found: {file_path}")
            print(f"⚠️ Falling back to DEFAULT_PROFILE")
            return self.DEFAULT_PROFILE
        except Exception as e:
            print(f"❌ Error loading profile: {e}")
            return self.DEFAULT_PROFILE

    async def _build_weighted_context(self, dream: str, weights: Dict[str, float], doctor_profile: DoctorProfile) -> str:
        """Build context that actually uses the weights to prioritize sources."""
        
        sections = []
        
        # Add doctor voice FIRST
        if doctor_profile.description:
            sections.append(f"=== YOUR VOICE & APPROACH ===\n{doctor_profile.description}\n")
        
        # Fetch and weight each category
        categories = [
            ("theory", "Dream Theory & Symbolism", "dream symbolism theory interpretation"),
            ("astrology", "Astrological Context", "astrology zodiac signs planets"),
            ("personality", "Personality Psychology", "personality traits psychology behavior"),
            ("medicalHistory", "Medical Context", "medical psychological mental health")
        ]
        
        for key, label, search_term in categories:
            weight = weights.get(key, 0.0)
            if weight > 0.05:  # Only include if weight is meaningful
                k = max(1, int(weight * 5))
                
                # Use your EXISTING search method
                query = f"{search_term} {dream[:100]}"
                docs = await self.knowledge_base.search_relevant_knowledge(query, k=k)
                
                if docs:
                    clean_content = self._extract_relevant_snippets(docs, dream, max_chars=500)
                    sections.append(f"=== {label.upper()} (weight: {weight:.0%}) ===\n{clean_content}\n")
        
        return "\n".join(sections)
        
    def _extract_relevant_snippets(self, docs, dream: str, max_chars: int = 500) -> str:
        """Extract only the most relevant parts, not entire docs."""
        snippets = []
        
        for doc in docs:
            text = doc.page_content.strip()
            # Remove obvious noise
            if any(x in text.lower() for x in ['recipe', 'cook', 'ingredients', 'oven']):
                continue
            
            # Take first coherent paragraph or sentence that seems relevant
            sentences = text.split('.')
            for sentence in sentences[:3]:  # First 3 sentences only
                if len(sentence.strip()) > 20:
                    snippets.append(sentence.strip() + '.')
                    break
            
            if sum(len(s) for s in snippets) > max_chars:
                break
        
        return ' '.join(snippets)

class DreamSymbolExtractor:
    def __init__(self):
        # Load once at initialization
        self.nlp = spacy.load("en_core_web_sm")
        self.symbol_categories = self._load_symbol_taxonomy()
        self.emotion_lexicon = self._load_emotion_lexicon()
        
    def _load_symbol_taxonomy(self) -> Dict[str, Set[str]]:
        """Load symbols organized by category with synonyms."""
        try:
            with open('core/extracted_dream_symbols.json', 'r') as f:
                data = json.load(f)
            
            # Convert flat list to categorized dict
            all_symbols = data.get('all_symbols_list', [])
            
            # Quick categorization (or use the fallback below)
            return {
                'all': set(all_symbols)  # Put everything in one category for now
            }
        except:
            return {
                'motion': {'flying', 'falling', 'running', 'chasing', 'escaping', 'floating'},
                'water': {'ocean', 'river', 'lake', 'rain', 'flood', 'swimming', 'drowning'},
                'animals': {'dog', 'cat', 'snake', 'bird', 'horse', 'spider', 'lion'},
                'structures': {'house', 'building', 'room', 'door', 'window', 'stairs', 'bridge'},
                'vehicles': {'car', 'train', 'airplane', 'boat', 'bicycle'},
                'transitions': {'death', 'birth', 'wedding', 'graduation', 'journey'},
            }
    
    def _load_emotion_lexicon(self) -> Dict[str, List[str]]:
        """Map emotions to their synonyms and related terms."""
        return {
            'fear': ['afraid', 'scared', 'terrified', 'frightened', 'anxious', 'nervous', 'worried', 'panic'],
            'joy': ['happy', 'joyful', 'excited', 'elated', 'cheerful', 'delighted', 'thrilled'],
            'sadness': ['sad', 'depressed', 'melancholy', 'sorrowful', 'grief', 'mourning'],
            'anger': ['angry', 'furious', 'rage', 'irritated', 'frustrated', 'hostile'],
            'confusion': ['confused', 'disoriented', 'lost', 'uncertain', 'bewildered'],
        }
    
    async def extract_dream_elements(self, entries: List[JournalEntry]) -> Dict[str, any]:
        """
        Extract symbols, themes, and emotions using NLP.
        Returns structured data instead of just a string.
        """
        combined_text = " ".join([entry.content for entry in entries])
        doc = self.nlp(combined_text)
        
        # 1. Extract noun phrases (captures "starting line", "adventure race")
        noun_phrases = self._extract_meaningful_phrases(doc)
        
        # 2. Extract named entities (people, places, organizations)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # 3. Match against symbol taxonomy with word boundaries
        matched_symbols = self._match_symbols_with_context(doc)
        
        # 4. Extract emotions with semantic matching
        emotions = self._extract_emotions(doc)
        
        # 5. Extract actions/verbs (what's happening in the dream)
        actions = self._extract_key_actions(doc)
        
        # 6. Weight by frequency and importance
        weighted_elements = self._weight_elements(
            noun_phrases, matched_symbols, emotions, actions
        )
        
        result = {
            'primary_symbols': weighted_elements['symbols'][:5],
            'key_phrases': weighted_elements['phrases'][:5],
            'emotions': emotions,
            'actions': actions[:5],
            'entities': entities,
            'raw_text_sample': combined_text[:200]  # For context
        }
        
        print(f"Extracted: {len(result['primary_symbols'])} symbols, "
              f"{len(result['key_phrases'])} phrases, "
              f"{len(result['emotions'])} emotions")
        
        return result
    
    def _extract_meaningful_phrases(self, doc) -> List[str]:
        """Extract multi-word noun phrases, not just single words."""
        phrases = []
        for chunk in doc.noun_chunks:
            # Filter out very generic or short phrases
            if len(chunk.text.split()) >= 2 and chunk.root.pos_ == 'NOUN':
                phrases.append(chunk.text.lower())
        return phrases
    
    def _match_symbols_with_context(self, doc) -> Dict[str, Dict]:
        """Match symbols using word boundaries and track context."""
        matched = {}
        print(f"🔍 Symbol categories loaded: {list(self.symbol_categories.keys())}")
        print(f"🔍 Total symbols to match: {sum(len(v) for v in self.symbol_categories.values())}")
        print(f"🔍 First 10 tokens in dream: {[token.text for token in doc[:10]]}")
        print(f"🔍 First 10 lemmas: {[token.lemma_ for token in doc[:10]]}")
    

        for category, symbols in self.symbol_categories.items():
            for token in doc:
                lemma = token.lemma_.lower()
                text = token.text.lower()
                
                # Check if token matches any symbol (using lemma for better matching)
                if lemma in symbols or text in symbols:
                    # Get context (surrounding words)
                    context_start = max(0, token.i - 3)
                    context_end = min(len(doc), token.i + 4)
                    context = doc[context_start:context_end].text
                    
                    symbol_key = lemma if lemma in symbols else text
                    
                    if symbol_key not in matched:
                        matched[symbol_key] = {
                            'category': category,
                            'frequency': 0,
                            'contexts': []
                        }
                    
                    matched[symbol_key]['frequency'] += 1
                    matched[symbol_key]['contexts'].append(context)
        
        return matched
    
    def _extract_emotions(self, doc) -> List[Dict[str, any]]:
        """Extract emotions with intensity and context."""
        found_emotions = []
        
        for base_emotion, variants in self.emotion_lexicon.items():
            for token in doc:
                lemma = token.lemma_.lower()
                text = token.text.lower()
                
                if text in variants or lemma in variants:
                    # Check for intensifiers (very, extremely, slightly)
                    intensity = self._get_emotion_intensity(token)
                    
                    found_emotions.append({
                        'emotion': base_emotion,
                        'word': text,
                        'intensity': intensity,
                        'context': doc[max(0, token.i-2):min(len(doc), token.i+3)].text
                    })
        
        return found_emotions
    
    def _get_emotion_intensity(self, token) -> str:
        """Determine intensity based on modifiers."""
        intensifiers = {'very', 'extremely', 'incredibly', 'absolutely'}
        diminishers = {'slightly', 'somewhat', 'a bit', 'kind of'}
        
        # Check previous token
        if token.i > 0:
            prev = token.doc[token.i - 1].text.lower()
            if prev in intensifiers:
                return 'high'
            elif prev in diminishers:
                return 'low'
        
        return 'medium'
    
    def _extract_key_actions(self, doc) -> List[str]:
        """Extract main verbs/actions happening in the dream."""
        actions = []
        for token in doc:
            if token.pos_ == 'VERB' and token.dep_ in ('ROOT', 'conj'):
                # Get the action with its object if present
                action_phrase = token.lemma_
                # Add direct object if exists
                for child in token.children:
                    if child.dep_ == 'dobj':
                        action_phrase += f" {child.text}"
                actions.append(action_phrase.lower())
        return list(set(actions))  # Remove duplicates
    
    def _weight_elements(self, phrases, symbols, emotions, actions) -> Dict:
        """Weight elements by frequency AND symbolic importance."""
        
        symbol_weights = {}
        for symbol, data in symbols.items():
            # data['frequency'] = how often it appears in THIS dream
            # We should also consider how "dream-specific" this word is
            
            # Get the symbol's overall frequency from the loaded taxonomy
            base_importance = self._get_symbol_importance(symbol)
            
            # Weight = frequency in dream × symbolic importance
            symbol_weights[symbol] = data['frequency'] * base_importance
        
        sorted_symbols = sorted(symbol_weights.items(), key=lambda x: x[1], reverse=True)
        
        phrase_weights = Counter(phrases)
        sorted_phrases = sorted(phrase_weights.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'symbols': [s[0] for s in sorted_symbols],
            'phrases': [p[0] for p in sorted_phrases]
        }

    def _get_symbol_importance(self, symbol: str) -> float:
        """Score how 'dream-specific' a word is based on JSON frequency."""
        # Higher JSON frequency = more important dream symbol
        with open('core/extracted_dream_symbols.json', 'r') as f:
            data = json.load(f)
        
        freq = data.get('symbol_frequencies', {}).get(symbol, 1)
        
        # Normalize: symbols with 100+ occurrences get higher weight
        return min(freq / 50, 5.0)  # Cap at 5x multiplier

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
    
    async def analyze_single_entry(self, content: str, settings: Dict[str, Any] = None) -> JournalAnalysis:
        """Analyze a single journal entry."""
        return await self.analyzer.analyze_entry(content, settings)

    async def ask_custom_question(self, question: str, entries: List[JournalEntry], personality: str = None, settings: Dict[str, Any] = None) -> str:
        """Ask a custom question about the dreams."""
        return await self.analyzer.custom_question_analysis(question, entries, personality, settings)

