import pandas as pd
import numpy as np
import re
import string
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

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
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class ProfessionalBankChatbot:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(
            min_df=1,
            stop_words='english',
            ngram_range=(1, 2),
            max_features=5000
        )
        self.label_encoder = LabelEncoder()
        self.model = SVC(kernel='linear', probability=True, C=1.0)
        
    def preprocess_text(self, text):
        """Enhanced text preprocessing"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        tokens = word_tokenize(text)
        tokens = [self.lemmatizer.lemmatize(token) 
                 for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        return ' '.join(tokens)
    
    def format_documents_professionally(self, text):
        """Format document lists professionally with bullet points"""
        # Split by common separators for documents
        separators = [r'[;-]\s*', r'\.\s*(?=[A-Z])', r'(?:and|or)\s+(?=[A-Z])']
        
        # Clean up the text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Try to identify document sections
        doc_patterns = [
            r'(Latest\s+Partnership\s+Deed.*?)(?=\s*[A-Z][a-z]+|$)',
            r'(Certificate\s+of\s+Registration.*?)(?=\s*[A-Z][a-z]+|$)',
            r'(Valid\s+Business\s+License.*?)(?=\s*[A-Z][a-z]+|$)',
            r'(PAN\s+Card.*?)(?=\s*[A-Z][a-z]+|$)',
            r'(Self-signed\s+cheque.*?)(?=\s*[A-Z][a-z]+|$)',
            r'(Landline\s+telephone\s+bill.*?)(?=\s*[A-Z][a-z]+|$)',
            r'(Property\s+Ownership\s+Deed.*?)(?=\s*[A-Z][a-z]+|$)',
            r'(Latest\s+property\s+tax.*?)(?=\s*[A-Z][a-z]+|$)',
            r'(TAN\s+Allotment\s+Letter.*?)(?=\s*[A-Z][a-z]+|$)',
            r'(Existing\s+bank\s+account.*?)(?=\s*[A-Z][a-z]+|$)'
        ]
        
        formatted_lines = []
        
        # Split by semicolons and dashes for document lists
        parts = re.split(r'[;-]\s*', text)
        
        for part in parts:
            part = part.strip()
            if part and len(part) > 5:
                # Clean up the part
                part = re.sub(r'\s+', ' ', part)
                # Capitalize first letter
                part = part[0].upper() + part[1:] if part else part
                formatted_lines.append(f"• {part}")
        
        if len(formatted_lines) > 1:
            return '\n'.join(formatted_lines)
        
        return text
    
    def format_answer_professionally(self, answer):
        """Format answers in a professional, structured manner"""
        # Handle document requirements specifically
        if 'documents required' in answer.lower() or 'following documents' in answer.lower():
            return self.format_documents_professionally(answer)
        
        # Handle general formatting
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        if len(sentences) > 2:
            formatted = []
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    formatted.append(sentence)
            return '\n\n'.join(formatted)
        
        return answer
    
    def load_and_train(self):
        """Load data and train improved model"""
        data = pd.read_csv('BankFAQs.csv')
        questions = data['Question'].apply(self.preprocess_text)
        X = self.vectorizer.fit_transform(questions)
        y = self.label_encoder.fit_transform(data['Class'])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.model.fit(X_train, y_train)
        self.data = data
        self.X = X
        
        accuracy = self.model.score(X_test, y_test)
        print(f"Model accuracy: {accuracy:.2f}")
        return self
    
    def get_relevant_answers(self, user_query, top_k=1):
        """Get most relevant answers with professional formatting"""
        processed_query = self.preprocess_text(user_query)
        query_vector = self.vectorizer.transform([processed_query])
        
        predicted_class = self.model.predict(query_vector)[0]
        class_name = self.label_encoder.inverse_transform([predicted_class])[0]
        
        class_mask = self.data['Class'] == class_name
        class_questions = self.data[class_mask]
        
        similarities = cosine_similarity(
            query_vector, 
            self.vectorizer.transform(class_questions['Question'].apply(self.preprocess_text))
        )[0]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            actual_idx = class_questions.index[idx]
            formatted_answer = self.format_answer_professionally(
                self.data.loc[actual_idx, 'Answer']
            )
            results.append({
                'question': self.data.loc[actual_idx, 'Question'],
                'answer': formatted_answer,
                'similarity': similarities[idx],
                'confidence': self.model.predict_proba(query_vector)[0][predicted_class]
            })
        
        return results
    
    def chat(self):
        """Interactive chat interface with professional formatting"""
        print("🏦 Professional Bank FAQ Assistant")
        print("=" * 50)
        print("Ask me anything about banking services!")
        print("Type 'quit' to exit\n")
        
        while True:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n🏦 Thank you for using our banking assistant!")
                break
            
            if not user_input:
                continue
            
            try:
                results = self.get_relevant_answers(user_input, top_k=1)
                
                if results:
                    result = results[0]
                    print(f"\n🏦 **Answer:**")
                    print("─" * 50)
                    print(result['answer'])
                    print("─" * 50)
                    
                    # Ask for feedback
                    helpful = input("\nWas this helpful? (yes/no): ").lower()
                    if helpful == 'no':
                        print("🏦 Let me try to find better answers...")
                        
                else:
                    print("🏦 I couldn't find relevant answers. Please try rephrasing your question.")
                    
            except Exception as e:
                print(f"🏦 Sorry, I encountered an error: {str(e)}")

if __name__ == "__main__":
    chatbot = ProfessionalBankChatbot()
    chatbot.load_and_train()
    chatbot.chat()
