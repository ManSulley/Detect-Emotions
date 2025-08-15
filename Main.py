import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
from pathlib import Path
import pickle
import os
import joblib  # ADDED: For individual component caching

# Import simplified modules
from Bert_config import SimpleBERTEmbeddings
from data_preprocessing import SimpleDataPreprocessor
from Train_models import SimpleEmotionClassifiers
from Model_evaluation import SimpleModelEvaluator

# 🗄️ UPDATED CACHING FUNCTIONS
def save_processed_data(X_train, X_test, y_train, y_test, dataset_name="demo"):
    """Save processed data to cache"""
    cache_dir = "demo_cache/"
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_data = {
        'X_train': X_train,
        'X_test': X_test, 
        'y_train': y_train,
        'y_test': y_test,
        'dataset_name': dataset_name,
        'timestamp': time.time()
    }
    
    with open(f"{cache_dir}/processed_data_{dataset_name}.pkl", "wb") as f:
        pickle.dump(cache_data, f)

def save_trained_models(embeddings_train, embeddings_test, classifiers, bert_embedder, dataset_name="demo"):
    """🔧 UPDATED: Save embeddings + individual model components"""
    cache_dir = "demo_cache/"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Save embeddings (as before)
    cache_data = {
        'embeddings_train': embeddings_train,
        'embeddings_test': embeddings_test,
        'bert_model_name': bert_embedder.model_name,
        'dataset_name': dataset_name,
        'timestamp': time.time()
    }
    
    with open(f"{cache_dir}/trained_models_{dataset_name}.pkl", "wb") as f:
        pickle.dump(cache_data, f)
    
    # 🔧 NEW: Save individual model components using joblib
    success, components = classifiers.save_model_components(dataset_name)
    
    return success, components

def load_cached_processed_data(dataset_name="demo"):
    """Load cached processed data"""
    try:
        cache_file = f"demo_cache/processed_data_{dataset_name}.pkl"
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        st.warning(f"Could not load cached data: {e}")
    return None

def load_cached_trained_models(dataset_name="demo"):
    """Load cached trained models"""
    try:
        cache_file = f"demo_cache/trained_models_{dataset_name}.pkl"
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                return pickle.load(f)
    except Exception as e:
        st.warning(f"Could not load cached models: {e}")
    return None

def load_cached_model_components(classifiers, dataset_name="demo"):
    """🔧 NEW: Load individual model components"""
    try:
        success, components = classifiers.load_model_components(dataset_name)
        return success, components
    except Exception as e:
        st.warning(f"Could not load cached model components: {e}")
        return False, []

def get_cache_info():
    """Show what's cached"""
    cache_dir = "demo_cache/"
    if not os.path.exists(cache_dir):
        return []
    
    cached_files = []
    for file in os.listdir(cache_dir):
        if file.endswith('.pkl') or file.endswith('.joblib'):
            file_path = os.path.join(cache_dir, file)
            size_mb = os.path.getsize(file_path) / (1024*1024)
            cached_files.append({
                'file': file,
                'size_mb': f"{size_mb:.1f} MB",
                'modified': time.ctime(os.path.getmtime(file_path))
            })
    return cached_files

# Essential emotion labels (27 emotions)
EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# Team info
TEAM_NAMES = [
    "Zoe Akua Ohene-Ampofo, 22252412",
    "Yvette S. Nerquaye-Tetteh, 22253082", 
    "Theophilus Arthur, 11410587",
    "Suleman Abdul-Razark, 22256374",
    "Steve Afrifa-Yamoah, 22252462",
]

# Page config
st.set_page_config(
    page_title="GoEmotions: Emotion Detection and Prediction",
    layout="wide",
    page_icon="🎭"
)

# CSS styling (same as before)
ACCENT_START = "#22d3ee"
ACCENT_MID = "#a78bfa" 
ACCENT_END = "#f472b6"

st.markdown(
    f"""
    <style>
        .stApp {{
            background: radial-gradient(1200px 800px at 10% 10%, #0d1321 0%, #0a0f1c 30%, #070b14 55%, #05080f 100%) !important;
        }}
        .glass {{
            background: rgba(255,255,255,0.04);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            box-shadow: 0 8px 24px rgba(0,0,0,0.35);
        }}
        .main-header {{
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: 0.5px;
            background: linear-gradient(90deg, {ACCENT_START}, {ACCENT_MID}, {ACCENT_END});
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            margin: 0.2rem 0 0.2rem 0;
            text-align: center;
        }}
        .chip {{
            display: inline-block;
            padding: 6px 10px;
            border-radius: 999px;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.08);
            margin: 4px 6px 0 0;
            font-size: 0.9rem;
            color: #e5e7eb;
        }}
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{ color: #e5e7eb !important; }}
        [data-testid="stMetric"] {{
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 0.9rem 0.9rem;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.04), 0 6px 18px rgba(0,0,0,0.35);
        }}
        [data-testid="stMetric"] [data-testid="stMetricDelta"] {{ color: {ACCENT_START} !important; }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
        .stTabs [data-baseweb="tab"] {{
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 10px 16px;
            color: #d1d5db;
        }}
        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, {ACCENT_START}33, {ACCENT_MID}33);
            color: #ffffff !important;
            border-color: {ACCENT_MID}66;
            box-shadow: 0 0 0 1px {ACCENT_START}33, 0 8px 22px rgba(0,0,0,0.35);
        }}
        .stButton > button {{
            background: linear-gradient(135deg, {ACCENT_START} 0%, {ACCENT_MID} 50%, {ACCENT_END} 100%);
            border: none;
            color: white;
            font-weight: 700;
            padding: 0.6rem 1.1rem;
            border-radius: 12px;
            transition: transform 0.06s ease-in-out, box-shadow 0.2s ease;
            box-shadow: 0 8px 22px {ACCENT_START}29, 0 2px 8px rgba(0,0,0,0.35);
        }}
        .stButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 12px 28px {ACCENT_MID}38, 0 4px 12px rgba(0,0,0,0.4);
        }}
        .stButton > button:active {{ transform: translateY(0px) scale(0.99); }}
        .stTextInput > div > div input, textarea {{
            background: rgba(255,255,255,0.06) !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            color: #e5e7eb !important;
            border-radius: 10px !important;
        }}
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, rgba(17,24,39,0.95), rgba(10,15,28,0.95));
            border-right: 1px solid rgba(255,255,255,0.06);
        }}
        .stDataFrame div[role="table"] {{
            background: rgba(255,255,255,0.02) !important;
            border-radius: 12px !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
        }}
        .footer {{ text-align: center; color: #9ca3af; padding: 18px; }}
        .step-container {{
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 8px 24px rgba(0,0,0,0.35);
        }}
        .progress-step {{
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 10px;
            padding: 0.8rem;
            margin: 0.3rem;
            text-align: center;
            font-weight: 600;
        }}
        .progress-step.active {{
            background: linear-gradient(135deg, {ACCENT_START}33, {ACCENT_MID}33);
            border-color: {ACCENT_MID}66;
            color: #ffffff !important;
        }}
        .progress-step.completed {{
            background: linear-gradient(135deg, #10b981, #059669);
            border-color: #10b981;
            color: #ffffff !important;
        }}
        .emotion-card {{
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.12);
            border-radius: 12px;
            padding: 1rem;
            margin: 0.5rem 0;
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize components
@st.cache_resource
def get_components():
    preprocessor = SimpleDataPreprocessor()
    bert_embedder = SimpleBERTEmbeddings()
    classifiers = SimpleEmotionClassifiers()
    evaluator = SimpleModelEvaluator()
    return preprocessor, bert_embedder, classifiers, evaluator

preprocessor, bert_embedder, classifiers, evaluator = get_components()

# Session state
if 'step' not in st.session_state:
    st.session_state.step = 1

# Header with original sophisticated styling
header_container = st.container()
with header_container:
    st.markdown('<div class="main-header">🎭 GoEmotions: Emotion Detection and Prediction</div>', unsafe_allow_html=True)
    
    # Team names as chips (original style)
    chips = " ".join([f"<span class='chip'>{name}</span>" for name in TEAM_NAMES])
    st.markdown(f"<div style='text-align: center; margin-bottom: 1rem;'>{chips}</div>", unsafe_allow_html=True)
    
    st.markdown(
        "<div style='color:#9ca3af; margin-top:6px; text-align: center;'>"
        "<strong>🚀 SYSTEM:</strong> Individual component caching + optimized predictions &nbsp;|&nbsp; "
        "<strong>📊 Performance:</strong> Precision, Recall, F-Measure, ROC-AUC &nbsp;|&nbsp; "
        "<strong>🎯 Results:</strong> Top 3 emotion predictions"
        "</div>",
        unsafe_allow_html=True,
    )

# Progress indicator
st.markdown("<div style='margin-top:1rem; margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

progress_steps = ["Upload Data", "Process Data", "Train Model", "Evaluate Model", "Predict Emotion"]
current_step = st.session_state.get('step', 1)

# Create progress indicator
progress_html = "<div style='display: flex; justify-content: center; gap: 0.5rem; margin: 1rem 0;'>"
for i, step_name in enumerate(progress_steps, 1):
    if i < current_step:
        progress_html += f"<div class='progress-step completed'>{step_name}</div>"
    elif i == current_step:
        progress_html += f"<div class='progress-step active'>{step_name}</div>"
    else:
        progress_html += f"<div class='progress-step'>{step_name}</div>"

progress_html += "</div>"
st.markdown(progress_html, unsafe_allow_html=True)

st.divider()

# Main interface based on current step
if current_step == 1:
    # Step 1: Data Upload (REMOVED cache options - now in sidebar only)
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.header("Step 1: Upload Your Data")
    
    uploaded_file = st.file_uploader(
        "Upload GoEmotions CSV file",
        type=['csv'],
        help="Upload your CSV file with 'text' column and emotion labels"
    )
    
    if uploaded_file:
        df = preprocessor.load_data(uploaded_file)
        if df is not None:
            st.session_state.df = df
            
            # Show basic info
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Samples", f"{len(df):,}")
            with col2:
                st.metric("Emotions", len(EMOTION_LABELS))
            with col3:
                avg_length = df['text'].str.len().mean()
                st.metric("Avg Text Length", f"{avg_length:.0f}")
            
            # Show preview
            st.subheader("Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Emotion chart (same as before)
            st.subheader("Emotion Distribution in Dataset")
            
            emotion_counts = {}
            for emotion in EMOTION_LABELS:
                if emotion in df.columns:
                    count = (df[emotion] == 1).sum()
                    emotion_counts[emotion] = count
            
            if emotion_counts:
                emotion_df = pd.DataFrame([
                    {'Emotion': emotion.title(), 'Count': count} 
                    for emotion, count in emotion_counts.items()
                ]).sort_values('Count', ascending=False)
                
                neutral_count = emotion_counts.get('neutral', 0)
                total_count = sum(emotion_counts.values())
                neutral_percentage = (neutral_count / total_count) * 100 if total_count > 0 else 0
                
                if neutral_percentage > 40:
                    st.error(f"⚠️ **CRITICAL BIAS DETECTED!** Neutral emotion represents {neutral_percentage:.1f}% of your data.")
                    st.info("💡 **Don't worry!** Our system will automatically fix this in Step 2.")
                elif neutral_percentage > 25:
                    st.warning(f"⚠️ **Moderate bias detected.** Neutral emotion is {neutral_percentage:.1f}% of data.")
                else:
                    st.success(f"✅ **Good balance!** Neutral emotion is {neutral_percentage:.1f}% of data.")
                
                fig = px.bar(
                    emotion_df, 
                    x='Emotion', 
                    y='Count',
                    title="Frequency of Emotions in Dataset",
                    labels={'Count': 'Number of Samples', 'Emotion': 'Emotion Type'}
                )
                
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    xaxis_tickangle=-45,
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            if st.button("✅ Continue to Processing", type="primary"):
                st.session_state.step = 2
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

elif current_step == 2:
    # Step 2: Process Data (REMOVED main cache options)
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.header("Step 2: Process Data")
    
    if 'df' not in st.session_state:
        st.warning("Please upload data first")
        if st.button("Back to Upload"):
            st.session_state.step = 1
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        df = st.session_state.df
        st.write(f"📊 Working with {len(df):,} samples")
        
        # Emotion imbalance controls
        st.subheader("Fix Emotion Imbalance (Recommended)")
        
        col1, col2 = st.columns(2)
        with col1:
            fix_imbalance = st.checkbox(
                "**Fix Neutral Emotion Bias**", 
                value=True,
                help="Highly recommended: Reduces neutral samples and balances emotions"
            )
            
            if fix_imbalance:
                max_neutral = st.slider(
                    "Max Neutral Samples", 
                    min_value=300, 
                    max_value=15000, 
                    value=8000,
                    help="Limit neutral samples to reduce bias"
                )
        
        with col2:
            sample_size = st.slider(
                "Final Dataset Size", 
                min_value=1000, 
                max_value=min(50000, len(df)), 
                value=min(15000, len(df)),
                help="After balancing, limit total size for faster training"
            )
        
        # Show current imbalance
        if 'neutral' in df.columns:
            neutral_count = (df['neutral'] == 1).sum()
            total_count = len(df)
            neutral_pct = (neutral_count / total_count) * 100
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current Neutral", f"{neutral_count:,}", delta=f"{neutral_pct:.1f}%")
            with col2:
                other_count = total_count - neutral_count
                other_pct = 100 - neutral_pct
                st.metric("Other Emotions", f"{other_count:,}", delta=f"{other_pct:.1f}%")
            with col3:
                if neutral_pct > 40:
                    st.metric("Bias Level", "🔴 High", help="Strong neutral bias detected")
                elif neutral_pct > 25:
                    st.metric("Bias Level", "🟡 Moderate", help="Some neutral bias")
                else:
                    st.metric("Bias Level", "🟢 Good", help="Balanced dataset")
        
        # Process button
        process_button = st.button("Process Data", type="primary", key="process_btn")
        
        if process_button:
            with st.spinner("Processing data with imbalance fixing..."):
                try:
                    X_train, X_test, y_train, y_test = preprocessor.process_data(
                        df, 
                        sample_size=sample_size,
                        fix_imbalance=fix_imbalance
                    )
                    
                    if X_train is not None and X_test is not None:
                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
                        st.session_state.data_processed = True
                        
                        st.success("✅ Data processed successfully with bias correction!")
                        st.balloons()
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Training Samples", len(X_train))
                        with col2:
                            st.metric("Test Samples", len(X_test))
                        with col3:
                            st.metric("Ready for Training", "✅")
                        
                        st.info("🚀 **Auto-advancing to Step 3 in 2 seconds...**")
                        time.sleep(2)
                        st.session_state.step = 3
                        st.rerun()
                        
                    else:
                        st.error("❌ Data processing failed")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Continue button if data exists
        if all(key in st.session_state for key in ['X_train', 'X_test', 'y_train', 'y_test']):
            st.divider()
            st.success("✅ Data is processed and ready for training!")
            
            if st.button("**CONTINUE TO TRAINING**", type="primary", key="continue_big"):
                st.session_state.step = 3
                st.success("Moving to Step 3...")
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)

elif current_step == 3:
    # Step 3: Train Models (REMOVED main cache options - now in sidebar)
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.header("Step 3: Train Models")
    
    if not all(key in st.session_state for key in ['X_train', 'X_test', 'y_train', 'y_test']):
        st.warning("Please process data first")
        if st.button("Back to Processing"):
            st.session_state.step = 2
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.write(f"Ready to train with {len(st.session_state.X_train):,} training samples")
        
        # Model Selection (same as before)
        st.subheader("Model Configuration")
        
        model_options = {
            'bert-base-uncased': {
                'name': 'BERT Base Uncased (Recommended)',
                'description': '**Best for emotion detection** - Optimized for social media text',
            },
            'bert-large-uncased': {
                'name': 'BERT Large Uncased',
                'description': '**Slower but slightly more accurate** - 2x computational cost',
            },
        }
        
        selected_model = st.selectbox(
            "Choose BERT Model:",
            options=list(model_options.keys()),
            index=0,
            format_func=lambda x: model_options[x]['name']
        )
        
        # Update embedder
        if selected_model != bert_embedder.model_name:
            bert_embedder.model_name = selected_model
            bert_embedder.model = None
            bert_embedder.tokenizer = None
        
        st.divider()
        
        # Check if models are already trained
        if st.session_state.get('models_trained', False):
            st.success("Models have been trained successfully!")
            
            if st.button("**CONTINUE TO EVALUATION**", type="primary", key="proceed_to_evaluation"):
                st.session_state.step = 4
                st.rerun()
        
        else:
            # FIXED: Show memory usage estimate before training
            if 'X_train' in st.session_state:
                num_samples = len(st.session_state.X_train)
                memory_est = bert_embedder.estimate_memory_usage(num_samples)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Training Samples", f"{num_samples:,}")
                    st.metric("Estimated Memory", f"{memory_est['total_mb']:.0f} MB")
                
                with col2:
                    if memory_est['warning']:
                        st.error("⚠️ High memory usage expected")
                        st.write("Consider using smaller dataset")
                    else:
                        st.success("✅ Memory usage acceptable")
                    st.write(f"💡 {memory_est['recommendation']}")
            
            train_button = st.button("🚀 Start Training", type="primary", key="start_training_btn")
            
            if train_button:
                progress_bar = st.progress(0)
                status = st.empty()
                
                try:
                    # Check if we have cached embeddings
                    embeddings_already_cached = hasattr(st.session_state, 'X_train_embeddings') and hasattr(st.session_state, 'X_test_embeddings')
                    
                    if embeddings_already_cached:
                        st.info("⚡ Using cached embeddings - training classifiers only...")
                        X_train_embeddings = st.session_state.X_train_embeddings
                        X_test_embeddings = st.session_state.X_test_embeddings
                        progress_bar.progress(60)
                    else:
                        # Generate BERT embeddings
                        status.text(f"Loading {selected_model} model...")
                        success = bert_embedder.load_model()
                        if not success:
                            st.error("Failed to load BERT model")
                            st.stop()
                        
                        progress_bar.progress(20)
                        
                        status.text("Generating training embeddings...")
                        X_train_embeddings = bert_embedder.generate_embeddings(st.session_state.X_train)
                        if X_train_embeddings is None:
                            st.error("Failed to generate training embeddings")
                            st.stop()
                        
                        progress_bar.progress(40)
                        
                        status.text("Generating test embeddings...")
                        X_test_embeddings = bert_embedder.generate_embeddings(st.session_state.X_test)
                        if X_test_embeddings is None:
                            st.error("Failed to generate test embeddings")
                            st.stop()
                        
                        progress_bar.progress(60)
                    
                    # Train models
                    status.text("Training Naive Bayes with GaussianNB + PCA...")
                    nb_success = classifiers.train_naive_bayes(X_train_embeddings, st.session_state.y_train)
                    
                    progress_bar.progress(80)
                    
                    status.text("Training Random Forest with imbalance handling...")
                    rf_success = classifiers.train_random_forest(X_train_embeddings, st.session_state.y_train)
                    
                    progress_bar.progress(100)
                    
                    # Store results
                    st.session_state.X_train_embeddings = X_train_embeddings
                    st.session_state.X_test_embeddings = X_test_embeddings
                    st.session_state.classifiers = classifiers
                    st.session_state.bert_embedder = bert_embedder
                    st.session_state.models_trained = True
                    st.session_state.selected_model = selected_model
                    
                    status.text("✅ Training completed!")
                    st.success("🎉 Models trained successfully!")
                    st.balloons()
                    
                    # Show training summary
                    st.subheader("Training Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("BERT Model", selected_model)
                        st.metric("Embedding Dimension", "768" if 'base' in selected_model else "1024")
                    
                    with col2:
                        st.metric("Training Samples", len(X_train_embeddings))
                        st.metric("Test Samples", len(X_test_embeddings))
                    
                    with col3:
                        models_trained = []
                        if nb_success:
                            models_trained.append("GaussianNB + PCA")
                        if rf_success:
                            models_trained.append("Random Forest")
                        st.metric("Models Trained", len(models_trained))
                        st.metric("Status", "Ready")
                    
                    st.info("📊 **Auto-advancing to Step 4 in 2 seconds...**")
                    time.sleep(2)
                    st.session_state.step = 4
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
                    st.exception(e)
        
        st.markdown('</div>', unsafe_allow_html=True)

elif current_step == 4:
    # Step 4: Evaluate Models (same as before)
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.header("Step 4: Evaluate Models")
    
    if all(key in st.session_state for key in ['X_test_embeddings', 'y_test', 'classifiers']):
        
        if st.button("Evaluate Models", type="primary"):
            with st.spinner("Evaluating models with all required metrics..."):
                results = evaluator.evaluate_models(
                    st.session_state.classifiers,
                    st.session_state.X_test_embeddings,
                    st.session_state.y_test
                )
                
                if results:
                    st.session_state.results = results
                    
                    summary_data = evaluator.display_performance_summary(results)
                    
                    st.subheader("Model Recommendations")
                    recommendations = evaluator.get_model_recommendations(results)
                    for recommendation in recommendations:
                        st.write(recommendation)
                    
                    evaluator.explain_metrics()
                    
                    if summary_data and len(summary_data) > 1:
                        best_model_data = max(summary_data, key=lambda x: float(x['Accuracy'].rstrip('%')))
                        best_model_name = best_model_data['Model'].lower().replace(' ', '_')
                        st.session_state.best_model_for_prediction = best_model_name
                        
                        st.success(f"🏆 **{best_model_data['Model']}** will be used for predictions")
                    else:
                        st.session_state.best_model_for_prediction = 'random_forest'
                    
                    if st.button("Continue to Predictions", type="primary"):
                        st.session_state.step = 5
                        st.rerun()
    else:
        st.warning("Please train models first")
        if st.button("Back to Training"):
            st.session_state.step = 3
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

elif current_step == 5:
    # Step 5: Make Predictions (🎯 UPDATED: Show top 3 emotions nicely)
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.header("Step 5: Make Predictions")
    
    if all(key in st.session_state for key in ['classifiers', 'bert_embedder']):
        
        best_model = st.session_state.get('best_model_for_prediction', 'random_forest')
        
        if 'results' in st.session_state:
            best_model_display = best_model.replace('_', ' ').title()
            
            if best_model in st.session_state.results:
                metrics = st.session_state.results[best_model]
                hamming_accuracy = metrics.get('hamming_accuracy', 0) * 100
                roc_auc = metrics.get('roc_auc', 0) * 100
                
                st.info(f"🏆 Using best model: **{best_model_display}** "
                       f"(Accuracy: {hamming_accuracy:.1f}%, ROC-AUC: {roc_auc:.1f}%)")
            else:
                st.info(f"🤖 Using model: **{best_model_display}**")
        
        # Prediction tabs
        tab1, tab2 = st.tabs(["📝 Single Text", "📄 Batch Upload"])
        
        with tab1:
            st.subheader("📝 Single Text Prediction")
            
            # Emotion examples
            with st.expander("💡 Try these example texts"):
                example_texts = [
                    "I'm so excited about this new opportunity! Can't wait to start!",
                    "This situation is really frustrating and making me angry.",
                    "I feel so sad and disappointed about what happened.",
                    "I'm worried and nervous about the upcoming presentation.",
                    "That joke was absolutely hilarious! I can't stop laughing.",
                    "I'm grateful for all the support you've given me.",
                    "This is just a regular update about the project status."
                ]
                
                for i, example in enumerate(example_texts):
                    if st.button(f"Use Example {i+1}: \"{example[:50]}...\"", key=f"example_{i}"):
                        st.session_state.example_text = example
            
            text_input = st.text_area(
                "Enter text to analyze:",
                value=st.session_state.get('example_text', ''),
                placeholder="Type something like: 'I'm so excited about this!' or 'This makes me angry.'"
            )
            
            if st.button("🎯 Predict Emotions", type="primary"):
                if text_input.strip():
                    with st.spinner("Analyzing emotions..."):
                        results = classifiers.predict_single_text(
                            text_input, 
                            st.session_state.bert_embedder,
                            model_type=best_model,
                            threshold=0.3
                        )
                        
                        if results and 'top_3_emotions' in results:
                            st.subheader("🎭 Top 3 Predicted Emotions")
                            
                            # 🎯 UPDATED: Show top 3 emotions in nice cards
                            for i, (emotion, prob) in enumerate(zip(results['top_3_emotions'], results['top_3_probabilities'])):
                                rank = i + 1
                                confidence_pct = prob * 100
                                
                                # Medal icons
                                if rank == 1:
                                    icon = "🥇"
                                    color = "#FFD700"
                                elif rank == 2:
                                    icon = "🥈" 
                                    color = "#C0C0C0"
                                else:
                                    icon = "🥉"
                                    color = "#CD7F32"
                                
                                # Create emotion card
                                st.markdown(f"""
                                <div class="emotion-card" style="border-left: 4px solid {color};">
                                    <div style="display: flex; justify-content: space-between; align-items: center;">
                                        <div>
                                            <h4>{icon} #{rank} {emotion.title()}</h4>
                                            <p style="margin: 0; color: #9ca3af;">Confidence: {confidence_pct:.1f}%</p>
                                        </div>
                                        <div style="font-size: 2rem; opacity: 0.7;">{confidence_pct:.0f}%</div>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Progress bar
                                st.progress(prob, text=f"{emotion.title()}: {confidence_pct:.1f}%")
                            
                            # Show insight based on top emotion
                            top_emotion = results['top_3_emotions'][0]
                            top_confidence = results['top_3_probabilities'][0] * 100
                            
                            if top_confidence >= 60:
                                insight = f"Strong **{top_emotion}** emotion detected! The model is quite confident."
                            elif top_confidence >= 40:
                                insight = f"Moderate **{top_emotion}** emotion detected with reasonable confidence."
                            else:
                                insight = f"Weak **{top_emotion}** signal. The text may be emotionally neutral or contain mixed emotions."
                            
                            st.info(f"🧠 **Analysis**: {insight}")
                            
                            # Show emotion distribution chart
                            if len(results['top_3_emotions']) >= 3:
                                emotion_df = pd.DataFrame({
                                    'Emotion': [e.title() for e in results['top_3_emotions']],
                                    'Confidence': [p*100 for p in results['top_3_probabilities']]
                                })
                                
                                fig = px.bar(
                                    emotion_df,
                                    x='Emotion',
                                    y='Confidence',
                                    title="Top 3 Emotion Predictions",
                                    color='Confidence',
                                    color_continuous_scale='viridis'
                                )
                                
                                fig.update_layout(
                                    template="plotly_dark",
                                    paper_bgcolor="rgba(0,0,0,0)",
                                    plot_bgcolor="rgba(0,0,0,0)",
                                    height=400
                                )
                                
                                st.plotly_chart(fig, use_container_width=True)
                        
                        else:
                            st.error("Failed to analyze emotions. Please try again.")
                else:
                    st.warning("Please enter some text")
        
        with tab2:
            # Batch upload (same as before)
            st.subheader("📄 Batch File Prediction")
            
            uploaded_file = st.file_uploader(
                "Upload CSV file with 'text' column",
                type=['csv'],
                key="batch_upload"
            )
            
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    if 'text' not in df.columns:
                        st.error("File must contain a 'text' column")
                    else:
                        st.success(f"Loaded {len(df):,} texts for prediction")
                        st.dataframe(df.head(), use_container_width=True)
                        
                        if st.button("Process Batch", type="primary"):
                            with st.spinner(f"Processing {len(df):,} texts..."):
                                results_df = classifiers.predict_batch(
                                    df, 
                                    st.session_state.bert_embedder,
                                    model_type=best_model,
                                    threshold=0.25
                                )
                                
                                if results_df is not None:
                                    st.success(f"✅ Processed {len(results_df):,} texts!")
                                    
                                    # Show results
                                    st.subheader("First 50 Results")
                                    display_df = results_df.head(50).copy()
                                    
                                    if 'text' in display_df.columns:
                                        display_df['text'] = display_df['text'].apply(lambda x: x[:80] + "..." if len(str(x)) > 80 else x)
                                    
                                    if 'confidence' in display_df.columns:
                                        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x*100:.1f}%")
                                    
                                    display_columns = ['text', 'top_emotion', 'confidence', 'top_3_emotions']
                                    available_columns = [col for col in display_columns if col in display_df.columns]
                                    
                                    if available_columns:
                                        st.dataframe(display_df[available_columns], use_container_width=True)
                                    else:
                                        st.dataframe(display_df, use_container_width=True)
                                    
                                    # Download section
                                    st.subheader("Download Results")
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        csv_data = results_df.to_csv(index=False)
                                        st.download_button(
                                            "📥 Download Complete Results",
                                            data=csv_data,
                                            file_name=f"emotion_predictions_{len(results_df)}.csv",
                                            mime="text/csv",
                                            type="primary"
                                        )
                                    
                                    with col2:
                                        summary = {
                                            'Total_Processed': len(results_df),
                                            'Most_Common_Emotion': results_df['top_emotion'].mode().iloc[0] if 'top_emotion' in results_df.columns else 'N/A',
                                            'Unique_Emotions': results_df['top_emotion'].nunique() if 'top_emotion' in results_df.columns else 0,
                                            'Average_Confidence': f"{results_df['confidence'].mean()*100:.1f}%" if 'confidence' in results_df.columns else 'N/A'
                                        }
                                        
                                        summary_csv = pd.DataFrame([summary]).to_csv(index=False)
                                        st.download_button(
                                            "📊 Download Summary", 
                                            data=summary_csv,
                                            file_name="emotion_summary.csv",
                                            mime="text/csv"
                                        )
                                else:
                                    st.error("❌ Batch prediction failed")
                
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
    else:
        st.warning("Please complete evaluation first")
        if st.button("Back to Evaluation"):
            st.session_state.step = 4
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# 🔧 UPDATED SIDEBAR: Cache options moved here + component caching
with st.sidebar:
    st.header("📊 System Status")
    
    status_items = [
        ("Data Loaded", 'df' in st.session_state),
        ("Data Processed", 'data_processed' in st.session_state and st.session_state.data_processed),
        ("Models Trained", 'models_trained' in st.session_state and st.session_state.models_trained),
        ("Models Evaluated", 'results' in st.session_state)
    ]
    
    for label, status in status_items:
        if status:
            st.success(f"✅ {label}")
        else:
            st.error(f"❌ {label}")
    
    # Progress indicator
    completed_steps = sum(1 for _, status in status_items if status)
    total_steps = len(status_items)
    progress = completed_steps / total_steps
    
    st.subheader("Overall Progress")
    st.progress(progress)
    st.write(f"**{completed_steps}/{total_steps} steps completed**")
    
    st.divider()
    
    # 🔧 UPDATED: Cache Management moved to sidebar with component caching
    st.header("💾 Cache Management")
    
    cache_info = get_cache_info()
    if cache_info:
        st.success(f"📁 {len(cache_info)} cached files")
        
        with st.expander("📁 View Cache Details"):
            for info in cache_info:
                file_type = "📦 Model Component" if info['file'].endswith('.joblib') else "📊 Data"
                st.write(f"{file_type} **{info['file']}**")
                st.caption(f"Size: {info['size_mb']} | Modified: {info['modified']}")
        
        # Cache options
        col1, col2 = st.columns(2)
        
        with col1:
            # Load cached data
            cached_data = load_cached_processed_data("demo")
            if cached_data:
                if st.button("⚡ Load Data", help="Load cached processed data"):
                    st.session_state.X_train = cached_data['X_train']
                    st.session_state.X_test = cached_data['X_test']
                    st.session_state.y_train = cached_data['y_train']
                    st.session_state.y_test = cached_data['y_test']
                    st.session_state.data_processed = True
                    st.success("⚡ Data loaded!")
                    st.rerun()
            
            # Load cached models
            cached_models = load_cached_trained_models("demo")
            if cached_models:
                if st.button("⚡ Load Embeddings", help="Load cached BERT embeddings"):
                    st.session_state.X_train_embeddings = cached_models['embeddings_train']
                    st.session_state.X_test_embeddings = cached_models['embeddings_test']
                    st.session_state.selected_model = cached_models['bert_model_name']
                    st.success("⚡ Embeddings loaded!")
        
        with col2:
            # Load cached model components
            if st.button("⚡ Load Models", help="Load cached trained models"):
                success, components = load_cached_model_components(classifiers, "demo")
                if success and components:
                    st.success(f"⚡ Loaded: {', '.join(components)}")
                    st.session_state.models_trained = True
                else:
                    st.warning("No model components found")
            
            # Save current models  
            if hasattr(st.session_state, 'classifiers') and st.session_state.get('models_trained', False):
                if st.button("💾 Save Models", help="Save current trained models"):
                    success, components = save_trained_models(
                        st.session_state.get('X_train_embeddings'),
                        st.session_state.get('X_test_embeddings'),
                        st.session_state.classifiers,
                        st.session_state.bert_embedder,
                        "demo"
                    )
                    if success:
                        st.success(f"💾 Saved: {', '.join(components)}")
        
        # FIXED: Enhanced cache management with safety checks
        if st.button("🧹 Clear All Cache", help="Delete all cached files"):
            try:
                import shutil
                if os.path.exists("demo_cache/"):
                    # Clear BERT model from memory first
                    if hasattr(st.session_state, 'bert_embedder'):
                        st.session_state.bert_embedder.clear_model()
                    
                    shutil.rmtree("demo_cache/")
                    st.success("🗑️ Cache cleared!")
                    
                    # Clear session state related to cache
                    cache_keys = ['X_train_embeddings', 'X_test_embeddings', 'models_trained']
                    for key in cache_keys:
                        if key in st.session_state:
                            del st.session_state[key]
                    
                    st.rerun()
                else:
                    st.info("No cache to clear")
            except Exception as e:
                st.error(f"Error clearing cache: {e}")
    else:
        st.info("📁 No cache files found")
        
        # Show save options for current data
        if hasattr(st.session_state, 'X_train'):
            if st.button("💾 Save Current Data"):
                save_processed_data(
                    st.session_state.X_train,
                    st.session_state.X_test,
                    st.session_state.y_train,
                    st.session_state.y_test,
                    "demo"
                )
                st.success("💾 Data saved!")
    
    st.divider()
    
    # Model performance (same as before)
    if hasattr(st.session_state, 'results') and st.session_state.results:
        st.header("🏆 Model Performance")
        results = st.session_state.results
        
        for model_name, metrics in results.items():
            hamming_accuracy = metrics.get('hamming_accuracy', 0) * 100
            roc_auc = metrics.get('roc_auc', 0) * 100
            
            if hamming_accuracy >= 75:
                status_icon = "🟢"
                status_text = "Excellent"
            elif hamming_accuracy >= 65:
                status_icon = "🟡"
                status_text = "Good"
            else:
                status_icon = "🟠"
                status_text = "Fair"
            
            st.write(f"**{model_name.replace('_', ' ').title()}**")
            st.write(f"{status_icon} {hamming_accuracy:.1f}% {status_text}")
            st.write(f"   ROC-AUC: {roc_auc:.1f}%")
    
    st.divider()
    
    # Manual navigation
    st.header("🔧 Manual Navigation")
    current_step_num = st.session_state.get('step', 1)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Previous") and current_step_num > 1:
            st.session_state.step = current_step_num - 1
            st.rerun()
    
    with col2:
        if st.button("Next →") and current_step_num < 5:
            st.session_state.step = current_step_num + 1
            st.rerun()
    
    st.write(f"Step: {current_step_num}/5")
    
    if st.button("🔄 Start Over"):
        for key in list(st.session_state.keys()):
            if key != 'step':
                del st.session_state[key]
        st.session_state.step = 1
        st.rerun()

# Footer
st.divider()
st.markdown(
    "<div class='footer'>"
    "<p><strong>🎯 GoEmotions: Enhanced Emotion Detection System</strong></p>"
    "<p>Individual Component Caching | Top 3 Emotion Predictions | All Required Metrics</p>"
    "</div>",
    unsafe_allow_html=True
)