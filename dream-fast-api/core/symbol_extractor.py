import os
import json
import re
from collections import Counter, defaultdict
from typing import List, Dict, Set
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader, TextLoader

class DreamSymbolExtractor:
    def __init__(self, knowledge_base_path: str):
        self.knowledge_base_path = knowledge_base_path
        self.stopwords = set(stopwords.words('english'))
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            nltk.download('averaged_perceptron_tagger')

    def load_documents(self) -> List[str]:
        """Load all documents from the knowledge base."""
        documents = []
        
        if not os.path.exists(self.knowledge_base_path):
            print(f"Knowledge base path {self.knowledge_base_path} not found")
            return documents
            
        try:
            # Load PDFs
            pdf_loader = DirectoryLoader(
                self.knowledge_base_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader
            )
            pdf_docs = pdf_loader.load()
            documents.extend([doc.page_content for doc in pdf_docs])
            print(f"Loaded {len(pdf_docs)} PDF documents")
            
            # Load text files
            txt_loader = DirectoryLoader(
                self.knowledge_base_path,
                glob="**/*.txt",
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf8'}
            )
            txt_docs = txt_loader.load()
            documents.extend([doc.page_content for doc in txt_docs])
            print(f"Loaded {len(txt_docs)} text documents")
            
        except Exception as e:
            print(f"Error loading documents: {e}")
            
        return documents

    def extract_potential_symbols(self, text: str) -> List[str]:
        """Extract potential dream symbols using various techniques."""
        symbols = []
        
        # Convert to lowercase for processing
        text_lower = text.lower()
        
        # Remove extra whitespace and normalize
        text_clean = re.sub(r'\s+', ' ', text_lower).strip()
        
        # Method 1: Look for common dream symbol patterns
        # Phrases like "dreaming of X", "dreams about Y", "to dream of Z"
        dream_patterns = [
            r'dream(?:ing|s)?\s+(?:of|about|that)\s+([a-zA-Z\s]{2,20}?)(?:\s+(?:means|indicates|suggests|represents|symbolizes))',
            r'(?:to\s+)?dream\s+(?:of|about)\s+([a-zA-Z\s]{2,20}?)(?:\s|\.|\,)',
            r'([a-zA-Z\s]{2,20}?)\s+(?:in\s+(?:a\s+)?dream|dreams)\s+(?:often\s+)?(?:means|indicates|suggests|represents|symbolizes)',
            r'(?:the\s+)?symbol(?:ism)?\s+of\s+([a-zA-Z\s]{2,20}?)(?:\s+(?:in|represents))',
            r'([a-zA-Z\s]{2,20}?)\s+(?:is\s+a\s+)?(?:dream\s+)?symbol',
        ]
        
        for pattern in dream_patterns:
            matches = re.findall(pattern, text_clean)
            for match in matches:
                clean_match = re.sub(r'\b(?:a|an|the|and|or|but|in|on|at|to|for|of|with|by)\b', '', match).strip()
                clean_match = re.sub(r'\s+', ' ', clean_match).strip()
                if len(clean_match) > 1 and clean_match not in self.stopwords:
                    symbols.append(clean_match)
        
        # Method 2: Extract nouns that appear frequently in dream contexts
        sentences_with_dream = []
        sentences = text.split('.')
        for sentence in sentences:
            if any(word in sentence.lower() for word in ['dream', 'dreaming', 'dreams', 'symbol', 'symbolism']):
                sentences_with_dream.append(sentence)
        
        dream_context_text = '. '.join(sentences_with_dream)
        
        # Tokenize and POS tag
        tokens = word_tokenize(dream_context_text.lower())
        pos_tags = pos_tag(tokens)
        
        # Extract nouns that could be symbols
        for word, tag in pos_tags:
            if (tag.startswith('NN') and 
                len(word) > 2 and 
                word.isalpha() and 
                word not in self.stopwords and
                word not in ['dream', 'dreams', 'dreaming', 'symbol', 'symbolism', 'interpretation']):
                symbols.append(word)
        
        return symbols

    def categorize_symbols(self, symbols: List[str]) -> Dict[str, List[str]]:
        """Categorize symbols into different types."""
        categories = {
            'animals': [],
            'nature': [],
            'people': [],
            'objects': [],
            'actions': [],
            'emotions': [],
            'places': [],
            'body_parts': [],
            'colors': [],
            'transportation': [],
            'supernatural': []
        }
        
        # Define category keywords (could be expanded)
        category_keywords = {
            'animals': ['dog', 'cat', 'snake', 'bird', 'fish', 'horse', 'lion', 'tiger', 'bear', 'wolf', 
                       'elephant', 'mouse', 'rabbit', 'spider', 'bee', 'butterfly', 'eagle', 'owl',
                       'cow', 'pig', 'sheep', 'goat', 'deer', 'fox', 'whale', 'dolphin', 'shark'],
            'nature': ['water', 'ocean', 'sea', 'river', 'lake', 'rain', 'storm', 'wind', 'fire', 
                      'tree', 'forest', 'mountain', 'hill', 'valley', 'desert', 'sky', 'sun', 'moon',
                      'star', 'cloud', 'lightning', 'snow', 'ice', 'flower', 'grass', 'rock', 'stone'],
            'people': ['mother', 'father', 'child', 'baby', 'friend', 'stranger', 'teacher', 'doctor',
                      'police', 'soldier', 'priest', 'family', 'parent', 'sibling', 'husband', 'wife'],
            'objects': ['house', 'door', 'window', 'key', 'mirror', 'book', 'money', 'gold', 'jewelry',
                       'clothes', 'food', 'knife', 'gun', 'telephone', 'computer', 'television'],
            'actions': ['flying', 'falling', 'running', 'chasing', 'hiding', 'swimming', 'driving',
                       'climbing', 'dancing', 'singing', 'crying', 'laughing', 'fighting', 'dying'],
            'emotions': ['fear', 'anxiety', 'joy', 'happiness', 'sadness', 'anger', 'love', 'hate',
                        'worry', 'peace', 'excitement', 'confusion', 'guilt', 'shame'],
            'places': ['school', 'church', 'hospital', 'office', 'store', 'restaurant', 'hotel',
                      'airport', 'prison', 'cemetery', 'park', 'beach', 'city', 'village'],
            'body_parts': ['teeth', 'hair', 'eyes', 'hands', 'feet', 'head', 'heart', 'blood',
                          'face', 'mouth', 'nose', 'ears', 'arms', 'legs'],
            'colors': ['red', 'blue', 'green', 'yellow', 'black', 'white', 'purple', 'orange',
                      'pink', 'brown', 'gray', 'silver', 'golden'],
            'transportation': ['car', 'train', 'airplane', 'boat', 'ship', 'bicycle', 'motorcycle',
                             'bus', 'truck', 'helicopter'],
            'supernatural': ['ghost', 'spirit', 'angel', 'demon', 'devil', 'god', 'magic', 'witch',
                           'monster', 'alien', 'vampire', 'zombie']
        }
        
        # Categorize symbols
        for symbol in symbols:
            categorized = False
            for category, keywords in category_keywords.items():
                if any(keyword in symbol.lower() or symbol.lower() in keyword for keyword in keywords):
                    if symbol not in categories[category]:
                        categories[category].append(symbol)
                    categorized = True
                    break
            
            # If not categorized, put in objects as default
            if not categorized:
                categories['objects'].append(symbol)
        
        return categories

    def analyze_and_extract(self) -> Dict[str, any]:
        """Main method to analyze texts and extract symbols."""
        print("Loading documents from knowledge base...")
        documents = self.load_documents()
        
        if not documents:
            print("No documents found!")
            return {}
        
        print(f"Analyzing {len(documents)} documents...")
        all_symbols = []
        
        # Extract symbols from each document
        for i, doc in enumerate(documents):
            print(f"Processing document {i+1}/{len(documents)}")
            symbols = self.extract_potential_symbols(doc)
            all_symbols.extend(symbols)
        
        print(f"Found {len(all_symbols)} potential symbols")
        
        # Count frequency and filter
        symbol_counts = Counter(all_symbols)
        
        # Filter symbols that appear at least 2 times and are reasonable length
        filtered_symbols = [
            symbol for symbol, count in symbol_counts.items() 
            if count >= 2 and 2 <= len(symbol.split()) <= 3 and len(symbol) <= 30
        ]
        
        print(f"After filtering: {len(filtered_symbols)} symbols")
        
        # Categorize symbols
        categorized = self.categorize_symbols(filtered_symbols)
        
        # Create final results
        results = {
            'total_symbols_found': len(all_symbols),
            'filtered_symbols_count': len(filtered_symbols),
            'top_symbols': dict(symbol_counts.most_common(50)),
            'categorized_symbols': {k: v for k, v in categorized.items() if v},
            'all_symbols_list': sorted(set(filtered_symbols)),
            'symbol_frequencies': dict(symbol_counts)
        }
        
        return results

    def generate_python_code(self, results: Dict) -> str:
        """Generate Python code with the extracted symbols."""
        symbols_list = results['all_symbols_list']
        
        code = '''# Auto-generated dream symbols from knowledge base analysis
        DREAM_SYMBOLS = [
        '''
        
        for symbol in sorted(symbols_list):
            code += f'    "{symbol}",\n'
        
        code += ''']

        # Categorized symbols
        CATEGORIZED_DREAM_SYMBOLS = {
        '''
        
        for category, symbols in results['categorized_symbols'].items():
            if symbols:
                code += f'    "{category}": [\n'
                for symbol in sorted(symbols):
                    code += f'        "{symbol}",\n'
                code += '    ],\n'
        
        code += '}'
        
        return code

def main():
    # Initialize extractor
    extractor = DreamSymbolExtractor("../knowledge_base/files")
    
    # Analyze and extract symbols
    print("Starting dream symbol extraction...")
    results = extractor.analyze_and_extract()
    
    if not results:
        print("No results generated. Check your knowledge base path.")
        return
    
    # Print summary
    print("\n=== EXTRACTION SUMMARY ===")
    print(f"Total symbols found: {results['total_symbols_found']}")
    print(f"Filtered symbols: {results['filtered_symbols_count']}")
    print(f"Categories with symbols: {len([k for k, v in results['categorized_symbols'].items() if v])}")
    
    # Print top symbols
    print("\n=== TOP 20 MOST FREQUENT SYMBOLS ===")
    for symbol, count in list(results['top_symbols'].items())[:20]:
        print(f"{symbol}: {count}")
    
    # Print categories
    print("\n=== SYMBOLS BY CATEGORY ===")
    for category, symbols in results['categorized_symbols'].items():
        if symbols:
            print(f"{category.upper()}: {len(symbols)} symbols")
            print(f"  Examples: {', '.join(symbols[:5])}")
    
    # Generate Python code
    python_code = extractor.generate_python_code(results)
    
    # Save results
    with open('extracted_dream_symbols.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    with open('dream_symbols.py', 'w') as f:
        f.write(python_code)
    
    print("\n=== FILES GENERATED ===")
    print("1. extracted_dream_symbols.json - Full analysis results")
    print("2. dream_symbols.py - Python code with symbol lists")
    
    print("\nTo use in your dream analyzer, copy the symbols from dream_symbols.py")
    print("into the extract_dream_elements method.")

if __name__ == "__main__":
    main()