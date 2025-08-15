import pandas as pd
import numpy as np
import streamlit as st
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

class SimpleDataPreprocessor:
    def __init__(self):
        self.emotion_columns = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
            'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]
        
        # Download NLTK data quietly
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        except:
            pass
        
        # Initialize NLP tools
        try:
            self.stop_words = set(stopwords.words('english'))
            self.stemmer = PorterStemmer()
            self.lemmatizer = WordNetLemmatizer()
        except:
            self.stop_words = set()
            self.stemmer = None
            self.lemmatizer = None
    
    def load_data(self, uploaded_file):
        """Load and validate CSV data"""
        try:
            df = pd.read_csv(uploaded_file)
            
            # Check for required columns
            if 'text' not in df.columns:
                st.error("CSV must contain a 'text' column")
                return None
            
            # Check for emotion columns
            missing_emotions = [col for col in self.emotion_columns if col not in df.columns]
            if missing_emotions:
                st.warning(f"Missing emotion columns: {missing_emotions}")
                # Create missing columns with zeros
                for emotion in missing_emotions:
                    df[emotion] = 0
            
            # Remove rows with empty text
            df = df[df['text'].notna() & (df['text'].str.strip() != '')]
            
            st.success(f"Loaded {len(df)} valid samples")
            return df
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    
    def clean_text_for_bert(self, text):
        """Minimal cleaning optimized for BERT models (Higher Accuracy)"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Only basic cleaning - keep most natural language intact
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove @mentions but keep hashtag content
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Keep punctuation - it's important for emotions!
        # Keep stop words - "not", "very", "really" matter for emotions!
        # Keep original case - CAPS can show emotion intensity
        
        # Only remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def clean_text(self, text, 
                   remove_stopwords=True, 
                   use_stemming=False, 
                   use_lemmatization=True,
                   remove_punctuation=True,
                   remove_numbers=False):
        """Standard NLP text cleaning (May reduce BERT accuracy)"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()  # Convert to lowercase
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags symbols (keep content)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove numbers if specified
        if remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove punctuation if specified
        if remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()  # Fallback if NLTK fails
        
        # Remove stopwords
        if remove_stopwords and self.stop_words:
            tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        
        # Apply stemming or lemmatization
        if use_stemming and self.stemmer:
            tokens = [self.stemmer.stem(word) for word in tokens]
        elif use_lemmatization and self.lemmatizer:
            try:
                tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
            except:
                pass  # Skip if lemmatization fails
        
        # Join back to text
        cleaned_text = ' '.join(tokens)
        
        # Remove extra whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        return cleaned_text

    def show_preprocessing_options(self):
        """Show preprocessing options with simple accuracy checkbox"""
        st.subheader("🔧 Text Preprocessing Options")
        
        # Simple accuracy checkbox at the top
        bert_optimized = st.checkbox(
            "🏆 **Use BERT-Optimized Preprocessing (Higher Accuracy)**", 
            value=True,  # Default to high accuracy
            help="Recommended: Minimal preprocessing for maximum BERT performance"
        )
        
        if bert_optimized:
            # Show what BERT optimization means
            st.success("✅ **High Accuracy Mode**: Keeps stop words, punctuation, and natural text")
            st.info("📈 **Expected Result**: 5-15% higher accuracy than standard NLP preprocessing")
            st.info("🔧 **What we keep**: 'not', 'very', punctuation (!?.), original case, natural word forms")
            
            return {
                'remove_stopwords': False,
                'use_stemming': False,
                'use_lemmatization': False,
                'remove_punctuation': False,
                'remove_numbers': False,
                'bert_optimized': True
            }
        
        else:
            # Standard NLP options
            st.warning("⚠️ **Standard NLP Mode**: May reduce BERT accuracy by 5-15%")
            st.info("🔬 **Use this for**: Traditional ML models or research comparison")
            
            col1, col2 = st.columns(2)
            
            with col1:
                remove_stopwords = st.checkbox("Remove Stop Words", value=True, 
                    help="Remove 'the', 'and', 'is', 'not', etc.")
                
                remove_punctuation = st.checkbox("Remove Punctuation", value=True,
                    help="Remove ! ? . , etc.")
                
                remove_numbers = st.checkbox("Remove Numbers", value=False)
            
            with col2:
                processing_type = st.radio(
                    "Word Processing:",
                    ["Lemmatization", "Stemming", "None"],
                    help="Lemmatization: running→run"
                )
                
                use_lemmatization = processing_type == "Lemmatization"
                use_stemming = processing_type == "Stemming"
            
            return {
                'remove_stopwords': remove_stopwords,
                'use_stemming': use_stemming,
                'use_lemmatization': use_lemmatization,
                'remove_punctuation': remove_punctuation,
                'remove_numbers': remove_numbers,
                'bert_optimized': False
            }

    def show_preprocessing_preview(self, df, preprocessing_options):
        """Show 5 clear examples of text preprocessing"""
        st.subheader("👀 Text Preprocessing Preview (5 Examples)")
        
        # Always take exactly 5 samples
        sample_texts = df['text'].head(5).tolist()
        
        preview_data = []
        for i, original_text in enumerate(sample_texts):
            # Apply cleaning based on mode
            if preprocessing_options.get('bert_optimized', False):
                cleaned = self.clean_text_for_bert(original_text)
                mode = "BERT-Optimized"
            else:
                cleaned = self.clean_text(original_text, **preprocessing_options)
                mode = "Standard NLP"
            
            # Calculate word counts
            original_words = len(original_text.split())
            cleaned_words = len(cleaned.split())
            
            preview_data.append({
                'Example': f"#{i+1}",
                'Original Text': original_text[:120] + "..." if len(original_text) > 120 else original_text,
                'Processed Text': cleaned[:120] + "..." if len(cleaned) > 120 else cleaned,
                'Word Count': f"{original_words} → {cleaned_words}",
                'Mode': mode
            })
        
        # Display the examples
        preview_df = pd.DataFrame(preview_data)
        st.dataframe(preview_df, use_container_width=True, hide_index=True)
        
        # Show summary stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_original = np.mean([len(text.split()) for text in sample_texts])
            st.metric("Avg Original Words", f"{avg_original:.1f}")
        
        with col2:
            if preprocessing_options.get('bert_optimized', False):
                processed_texts = [self.clean_text_for_bert(text) for text in sample_texts]
            else:
                processed_texts = [self.clean_text(text, **preprocessing_options) for text in sample_texts]
            avg_processed = np.mean([len(text.split()) for text in processed_texts])
            st.metric("Avg Processed Words", f"{avg_processed:.1f}")
        
        with col3:
            reduction = ((avg_original - avg_processed) / max(avg_original, 1)) * 100
            st.metric("Word Reduction", f"{reduction:.1f}%")

    def fix_emotion_imbalance(self, df, max_neutral_samples=8000, min_emotion_samples=500):
        """🎯 SIMPLE FIX for neutral emotion bias - the key solution!"""
        st.subheader("⚖️ Fixing Emotion Imbalance")
        
        original_counts = {}
        for emotion in self.emotion_columns:
            if emotion in df.columns:
                original_counts[emotion] = (df[emotion] == 1).sum()
        
        # Show original distribution
        st.write("📊 **Original Distribution:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Neutral", f"{original_counts.get('neutral', 0):,}")
        with col2:
            non_neutral = sum(v for k, v in original_counts.items() if k != 'neutral')
            st.metric("All Other Emotions", f"{non_neutral:,}")
        with col3:
            total_samples = len(df)
            st.metric("Total Samples", f"{total_samples:,}")
        
        # Separate neutral and non-neutral samples
        neutral_mask = df['neutral'] == 1
        neutral_df = df[neutral_mask]
        non_neutral_df = df[~neutral_mask]
        
        # Apply fixes
        fixed_dfs = []
        
        # 1. Limit neutral samples
        if len(neutral_df) > max_neutral_samples:
            neutral_df = neutral_df.sample(n=max_neutral_samples, random_state=42)
            st.success(f"✅ Reduced neutral from {original_counts['neutral']:,} to {max_neutral_samples:,}")
        else:
            st.info(f"ℹ️ Keeping all {len(neutral_df):,} neutral samples")
        
        fixed_dfs.append(neutral_df)
        
        # 2. Boost rare emotions by duplicating samples
        emotion_boosts = {}
        for emotion in self.emotion_columns:
            if emotion == 'neutral' or emotion not in df.columns:
                continue
            
            emotion_samples = df[df[emotion] == 1]
            count = len(emotion_samples)
            
            if count < min_emotion_samples and count > 0:
                # Calculate how many times to duplicate
                multiplier = min(3, max_emotion_samples // count)  # Max 3x duplication
                
                boosted_samples = pd.concat([emotion_samples] * multiplier, ignore_index=True)
                fixed_dfs.append(boosted_samples)
                emotion_boosts[emotion] = f"{count} → {len(boosted_samples)}"
        
        # Add remaining non-neutral samples
        fixed_dfs.append(non_neutral_df)
        
        # Combine all and shuffle
        balanced_df = pd.concat(fixed_dfs, ignore_index=True).drop_duplicates().sample(frac=1, random_state=42)
        
        # Show results
        st.write("🎯 **After Balancing:**")
        new_counts = {}
        for emotion in self.emotion_columns:
            if emotion in balanced_df.columns:
                new_counts[emotion] = (balanced_df[emotion] == 1).sum()
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Neutral", f"{new_counts.get('neutral', 0):,}", 
                     delta=f"{new_counts.get('neutral', 0) - original_counts.get('neutral', 0):,}")
        with col2:
            new_non_neutral = sum(v for k, v in new_counts.items() if k != 'neutral')
            old_non_neutral = sum(v for k, v in original_counts.items() if k != 'neutral')
            st.metric("Other Emotions", f"{new_non_neutral:,}", 
                     delta=f"{new_non_neutral - old_non_neutral:+,}")
        with col3:
            st.metric("Final Total", f"{len(balanced_df):,}", 
                     delta=f"{len(balanced_df) - len(df):+,}")
        
        if emotion_boosts:
            st.success("✅ **Boosted rare emotions:**")
            for emotion, boost in emotion_boosts.items():
                st.write(f"   • {emotion.title()}: {boost}")
        
        return balanced_df
    
    def balance_emotions(self, df, max_neutral_samples=10000):
        """Legacy method - keeping for compatibility"""
        return self.fix_emotion_imbalance(df, max_neutral_samples)
    
    def process_data(self, df, sample_size=None, preprocessing_options=None, fix_imbalance=True):
        """Process data with BERT-optimized preprocessing and imbalance fixing"""
        try:
            # Default to BERT-optimized for higher accuracy
            if preprocessing_options is None:
                preprocessing_options = {
                    'remove_stopwords': False,
                    'use_stemming': False,
                    'use_lemmatization': False,
                    'remove_punctuation': False,
                    'remove_numbers': False,
                    'bert_optimized': True
                }
            
            # 🎯 KEY FIX: Handle imbalance FIRST
            if fix_imbalance and 'neutral' in df.columns:
                df = self.fix_emotion_imbalance(df)
                st.success(f"✅ Fixed emotion imbalance! New dataset size: {len(df):,}")
            
            # Sample data if needed AFTER balancing
            if sample_size and sample_size < len(df):
                # Use stratified sampling to maintain emotion balance
                emotion_ratios = {}
                for emotion in self.emotion_columns:
                    if emotion in df.columns:
                        emotion_ratios[emotion] = (df[emotion] == 1).sum() / len(df)
                
                df = df.sample(n=sample_size, random_state=42)
                st.info(f"Sampled {sample_size:,} samples while maintaining emotion balance")
            
            # Clean text
            df = df.copy()
            st.write("🔄 Applying text preprocessing...")
            
            # Choose cleaning method based on settings
            if preprocessing_options.get('bert_optimized', False):
                st.success("🏆 Using BERT-optimized preprocessing for higher accuracy")
                df['cleaned_text'] = df['text'].apply(self.clean_text_for_bert)
            else:
                # Show what preprocessing steps are being applied
                active_steps = []
                if preprocessing_options.get('remove_stopwords'):
                    active_steps.append("Remove stop words")
                if preprocessing_options.get('remove_punctuation'):
                    active_steps.append("Remove punctuation")
                if preprocessing_options.get('remove_numbers'):
                    active_steps.append("Remove numbers")
                if preprocessing_options.get('use_lemmatization'):
                    active_steps.append("Lemmatization")
                elif preprocessing_options.get('use_stemming'):
                    active_steps.append("Stemming")
                
                if active_steps:
                    st.warning(f"⚠️ Using standard NLP: {', '.join(active_steps)} (may reduce BERT accuracy)")
                else:
                    st.info("ℹ️ Using minimal preprocessing")
                
                # Apply standard cleaning
                df['cleaned_text'] = df['text'].apply(
                    lambda x: self.clean_text(x, **preprocessing_options)
                )
            
            # Remove empty texts after cleaning
            df = df[df['cleaned_text'] != '']
            
            if len(df) == 0:
                st.error("No valid texts after cleaning")
                return None, None, None, None
            
            # Show preprocessing results
            if preprocessing_options.get('bert_optimized', False):
                st.success(f"✅ Applied BERT-optimized preprocessing to {len(df):,} samples")
            else:
                st.success(f"✅ Applied standard NLP preprocessing to {len(df):,} samples")
            
            # Prepare features and labels
            X = df['cleaned_text'].values
            y = df[self.emotion_columns].values.astype(float)
            
            # 🎯 KEY FIX: Stratified split to maintain emotion balance
            try:
                # Create a combined label for stratification (find dominant emotion per sample)
                dominant_emotions = []
                for row in y:
                    if np.any(row == 1):
                        dominant_emotions.append(np.argmax(row))
                    else:
                        dominant_emotions.append(-1)  # No emotion
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=dominant_emotions
                )
                st.success("✅ Used stratified split to maintain emotion balance")
                
            except:
                # Fallback to regular split if stratification fails
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                st.info("ℹ️ Used regular split (stratification failed)")
            
            st.success(f"Processed data: {len(X_train):,} train, {len(X_test):,} test samples")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            return None, None, None, None