import numpy as np
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import gc
import time

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
        """Load BERT model with enhanced error handling"""
        try:
            with st.spinner("Loading BERT model..."):
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
                
                st.success("BERT model loaded successfully!")
                return True
                
        except Exception as e:
            st.error(f"Error loading BERT model: {str(e)}")
            st.warning("💡 **Fallback suggestion**: Try using 'distilbert-base-uncased' if BERT-base fails")
            return False
    
    def generate_embeddings(self, texts, batch_size=8):  # FIXED: Reduced default batch size for CPU
        """FIXED: Generate BERT embeddings with chunked processing for large datasets"""
        if not isinstance(texts, list):
            texts = list(texts)
        
        # Clean texts
        texts = [str(text).strip() for text in texts if text and str(text).strip()]
        
        if not texts:
            st.error("No valid texts provided")
            return None
        
        if self.model is None or self.tokenizer is None:
            st.error("BERT model not loaded. Please load model first.")
            return None
        
        # FIXED: Adaptive batch size based on dataset size and available memory
        if len(texts) > 50000:
            batch_size = min(4, batch_size)  # Very small batches for huge datasets
            st.warning(f"🔧 Large dataset detected ({len(texts):,} samples). Using smaller batch size: {batch_size}")
        elif len(texts) > 10000:
            batch_size = min(8, batch_size)  # Small batches for large datasets
        
        embeddings = []
        
        # 🎯 FIXED: Always show progress for datasets > 500 samples
        show_progress = len(texts) > 500
        
        if show_progress:
            st.write(f"Generating embeddings for {len(texts):,} texts in batches of {batch_size}...")
            progress_bar = st.progress(0)
            status_text = st.empty()
            start_time = time.time()
        
        try:
            # FIXED: Process in smaller chunks with memory monitoring
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            for batch_idx in range(0, len(texts), batch_size):
                batch_texts = texts[batch_idx:batch_idx+batch_size]
                current_batch = (batch_idx // batch_size) + 1
                
                # Update progress for large batches
                if show_progress:
                    progress = min(batch_idx / len(texts), 1.0)
                    progress_bar.progress(progress)
                    
                    elapsed_time = time.time() - start_time
                    if elapsed_time > 0:
                        rate = batch_idx / elapsed_time
                        eta = (len(texts) - batch_idx) / rate if rate > 0 else 0
                        status_text.text(f"Batch {current_batch}/{total_batches} | ETA: {eta:.0f}s | Rate: {rate:.1f} texts/s")
                
                # FIXED: Tokenize with memory-optimized settings
                try:
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
                    
                    # FIXED: Clear intermediate tensors to prevent memory buildup
                    del encoded, outputs, batch_embeddings
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        st.error(f"❌ Out of memory at batch {current_batch}. Try reducing batch size or dataset size.")
                        return None
                    else:
                        raise e
                
                # FIXED: Force garbage collection every 50 batches for large datasets
                if current_batch % 50 == 0:
                    gc.collect()
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
            
            if show_progress:
                progress_bar.progress(1.0)
                elapsed_time = time.time() - start_time
                status_text.text(f"✅ Completed in {elapsed_time:.1f}s")
            
            # Convert to numpy array
            embeddings_array = np.array(embeddings)
            
            st.success(f"Generated embeddings: {embeddings_array.shape} (samples × features)")
            return embeddings_array
            
        except Exception as e:
            st.error(f"Error generating embeddings: {str(e)}")
            return None
    
    def get_single_embedding(self, text):
        """Get embedding for single text with better error handling"""
        if not text or not text.strip():
            st.warning("Empty text provided for embedding")
            return None
        
        if self.model is None or self.tokenizer is None:
            st.error("BERT model not loaded. Please load model first.")
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
            
            # Clean up tensors
            del encoded, outputs
            
            return embedding[0]  # Return single embedding
            
        except Exception as e:
            st.error(f"Error generating single embedding: {str(e)}")
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
        """FIXED: Enhanced model clearing with proper memory management"""
        try:
            if self.model is not None:
                # Move to CPU first if on GPU, then delete
                if self.device.type == 'cuda':
                    self.model.cpu()
                del self.model
                self.model = None
            
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            
            # Comprehensive memory cleanup
            gc.collect()  # Force garbage collection
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            st.info("🧹 Model and memory cleared successfully")
            
        except Exception as e:
            st.warning(f"Memory cleanup warning: {e}")
    
    def estimate_memory_usage(self, num_texts):
        """FIXED: Estimate memory usage for given number of texts"""
        # Rough estimates based on BERT-base
        embedding_size = 768  # BERT-base hidden size
        bytes_per_float = 4
        
        # Memory for embeddings
        embeddings_mb = (num_texts * embedding_size * bytes_per_float) / (1024 * 1024)
        
        # Model memory (roughly 110M parameters)
        model_mb = 440  # Approximate for BERT-base
        
        total_mb = embeddings_mb + model_mb
        
        return {
            'embeddings_mb': embeddings_mb,
            'model_mb': model_mb,
            'total_mb': total_mb,
            'warning': total_mb > 8000,  # Warn if > 8GB
            'recommendation': "Consider processing in smaller batches" if total_mb > 4000 else "Should fit in memory"
        }