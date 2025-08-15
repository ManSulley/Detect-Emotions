import numpy as np
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Just update the __init__ method in SimpleBERTEmbeddings class (around line 8):

class SimpleBERTEmbeddings:
    def __init__(self, model_name='bert-base-uncased'):
        # Use BERT-base-uncased - the optimal choice for emotion detection
        # Why this model?
        # Perfect for Reddit/social media text (GoEmotions dataset)
        # Proven performance: 75-85% accuracy on emotion tasks
        # Fast training: 12 layers, 768 dimensions
        # Memory efficient: Works on most GPUs/systems
        # Research-proven: Used in 1000+ papers as baseline
        #
        # Alternative models:
        # bert-large-uncased: 2x slower, only ~3% better accuracy
        # roberta-base: Often better but much slower training
        # distilbert-base-uncased: 40% faster but ~5% lower accuracy
        self.model_name = model_name  # Now configurable
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_model(self):
        """Load BERT model"""
        try:
            with st.spinner("🤖 Loading BERT model..."):
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model.eval()
                
                # Test the model
                test_input = "This is a test."
                test_encoding = self.tokenizer(
                    test_input,
                    return_tensors='pt',
                    max_length=128,
                    truncation=True,
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    _ = self.model(**test_encoding)
                
                st.success("✅ BERT model loaded successfully!")
                return True
                
        except Exception as e:
            st.error(f"❌ Error loading BERT model: {str(e)}")
            return False
    
    def generate_embeddings(self, texts, batch_size=16):
        """🎯 FIXED: Generate BERT embeddings for texts (handles all batch sizes)"""
        if not isinstance(texts, list):
            texts = list(texts)
        
        # Clean texts
        texts = [str(text).strip() for text in texts if text and str(text).strip()]
        
        if not texts:
            st.error("❌ No valid texts provided")
            return None
        
        if self.model is None or self.tokenizer is None:
            st.error("❌ BERT model not loaded. Please load model first.")
            return None
        
        embeddings = []
        
        # 🎯 FIXED: Always process in batches, show progress for large datasets
        show_progress = len(texts) > 100
        
        if show_progress:
            st.write(f"🔄 Generating embeddings for {len(texts):,} texts...")
            progress_bar = st.progress(0)
        
        try:
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Update progress for large batches
                if show_progress:
                    progress = (i + batch_size) / len(texts)
                    progress_bar.progress(min(progress, 1.0))
                
                # Tokenize with optimized settings for emotion detection
                encoded = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,           # Full context for emotions
                    return_tensors='pt'
                ).to(self.device)
                
                # Generate embeddings
                with torch.no_grad():
                    outputs = self.model(**encoded)
                    # Use [CLS] token embeddings (best for sentence-level emotion classification)
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    embeddings.extend(batch_embeddings)
            
            if show_progress:
                progress_bar.progress(1.0)
            
            # Convert to numpy array
            embeddings_array = np.array(embeddings)
            
            st.success(f"✅ Generated embeddings: {embeddings_array.shape} (samples × features)")
            return embeddings_array
            
        except Exception as e:
            st.error(f"❌ Error generating embeddings: {str(e)}")
            return None
    
    def get_single_embedding(self, text):
        """Get embedding for single text with better error handling"""
        if not text or not text.strip():
            st.warning("⚠️ Empty text provided for embedding")
            return None
        
        if self.model is None or self.tokenizer is None:
            st.error("❌ BERT model not loaded. Please load model first.")
            return None
        
        try:
            # Tokenize with same settings as batch processing
            encoded = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                outputs = self.model(**encoded)
                # Use [CLS] token embedding
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            return embedding[0]  # Return single embedding
            
        except Exception as e:
            st.error(f"❌ Error generating single embedding: {str(e)}")
            return None
    
    def get_model_info(self):
        """Get information about the loaded model"""
        if self.model is None:
            return "No model loaded"
        
        # Get model size info
        total_params = sum(p.numel() for p in self.model.parameters())
        
        model_info = {
            'name': self.model_name,
            'parameters': f"{total_params:,}",
            'device': str(self.device),
            'embedding_dim': self.model.config.hidden_size if hasattr(self.model, 'config') else 'Unknown'
        }
        
        return model_info
    
    def clear_model(self):
        """Clear model from memory"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        st.info("🧹 Model cleared from memory")