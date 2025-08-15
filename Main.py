import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import time
from pathlib import Path

# Import simplified modules
from Bert_config import SimpleBERTEmbeddings
from data_preprocessing import SimpleDataPreprocessor
from Train_models import SimpleEmotionClassifiers
from Model_evaluation import SimpleModelEvaluator

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
    page_title="GoEmotions: Simple Emotion Detection",
    layout="wide",
    page_icon="🎭"
)

# Original sophisticated CSS theme
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
    st.markdown('<div class="main-header">🎭 GoEmotions: Simple Emotion Detection</div>', unsafe_allow_html=True)
    
    # Team names as chips (original style)
    chips = " ".join([f"<span class='chip'>{name}</span>" for name in TEAM_NAMES])
    st.markdown(f"<div style='text-align: center; margin-bottom: 1rem;'>{chips}</div>", unsafe_allow_html=True)
    
    st.markdown(
        "<div style='color:#9ca3af; margin-top:6px; text-align: center;'>"
        "<strong>🎯 Fixed System:</strong> Emotion detection with neutral bias correction &nbsp;|&nbsp; "
        "<strong>Dataset:</strong> GoEmotions (27 labels) &nbsp;|&nbsp; "
        "<strong>🚀 Improved:</strong> Better emotion balance and accuracy"
        "</div>",
        unsafe_allow_html=True,
    )

# Progress indicator with original glass styling
st.markdown("<div style='margin-top:1rem; margin-bottom: 1rem;'></div>", unsafe_allow_html=True)

progress_steps = ["📁 Upload Data", "🔄 Process", "🤖 Train Models", "📊 Evaluate", "🎯 Predict"]
current_step = st.session_state.get('step', 1)

# Create progress indicator with glass effect
progress_html = "<div style='display: flex; justify-content: center; gap: 0.5rem; margin: 1rem 0;'>"
for i, step_name in enumerate(progress_steps, 1):
    if i < current_step:
        # Completed step
        progress_html += f"<div class='progress-step completed'>{step_name}</div>"
    elif i == current_step:
        # Current step
        progress_html += f"<div class='progress-step active'>{step_name}</div>"
    else:
        # Future step
        progress_html += f"<div class='progress-step'>{step_name}</div>"

progress_html += "</div>"
st.markdown(progress_html, unsafe_allow_html=True)

st.divider()

# Main interface based on current step
if current_step == 1:
    # Step 1: Data Upload with glass styling
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.header("📁 Step 1: Upload Your Data")
    
    uploaded_file = st.file_uploader(
        "Upload GoEmotions CSV file",
        type=['csv'],
        help="Upload your CSV file with 'text' column and emotion labels"
    )
    
    if uploaded_file:
        df = preprocessor.load_data(uploaded_file)
        if df is not None:
            st.session_state.df = df
            
            # Show basic info in glass containers
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Samples", f"{len(df):,}")
            with col2:
                st.metric("Emotions", len(EMOTION_LABELS))
            with col3:
                avg_length = df['text'].str.len().mean()
                st.metric("Avg Text Length", f"{avg_length:.0f}")
            
            # Show preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Emotion Frequency Chart with Bias Warning
            st.subheader("📊 Emotion Distribution in Dataset")
            
            # Calculate emotion frequencies
            emotion_counts = {}
            for emotion in EMOTION_LABELS:
                if emotion in df.columns:
                    count = (df[emotion] == 1).sum()
                    emotion_counts[emotion] = count
            
            if emotion_counts:
                # Create DataFrame for plotting
                emotion_df = pd.DataFrame([
                    {'Emotion': emotion.title(), 'Count': count} 
                    for emotion, count in emotion_counts.items()
                ]).sort_values('Count', ascending=False)
                
                # 🎯 CHECK FOR BIAS and show warning
                neutral_count = emotion_counts.get('neutral', 0)
                total_count = sum(emotion_counts.values())
                neutral_percentage = (neutral_count / total_count) * 100 if total_count > 0 else 0
                
                if neutral_percentage > 40:
                    st.error(f"⚠️ **CRITICAL BIAS DETECTED!** Neutral emotion represents {neutral_percentage:.1f}% of your data. This will severely impact model performance!")
                    st.info("💡 **Don't worry!** Our system will automatically fix this in Step 2.")
                elif neutral_percentage > 25:
                    st.warning(f"⚠️ **Moderate bias detected.** Neutral emotion is {neutral_percentage:.1f}% of data. We'll balance this in Step 2.")
                else:
                    st.success(f"✅ **Good balance!** Neutral emotion is {neutral_percentage:.1f}% of data.")
                
                # Create bar chart
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
                
                # Show top emotions with bias indicators
                top_5_emotions = emotion_df.head(5)
                st.subheader("🏆 Top 5 Most Common Emotions")
                
                for i, row in top_5_emotions.iterrows():
                    percentage = (row['Count'] / len(df)) * 100
                    
                    # Add bias indicator
                    if row['Emotion'].lower() == 'neutral' and percentage > 40:
                        icon = "🔴"
                        status = "High Bias"
                    elif row['Emotion'].lower() == 'neutral' and percentage > 25:
                        icon = "🟡"
                        status = "Moderate Bias"
                    else:
                        icon = "✅"
                        status = "Good"
                    
                    st.metric(
                        f"{icon} #{i+1} {row['Emotion']}", 
                        f"{row['Count']:,} samples",
                        delta=f"{percentage:.1f}% - {status}"
                    )
            
            if st.button("✅ Continue to Processing", type="primary"):
                st.session_state.step = 2
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

elif current_step == 2:
    # Step 2: Process Data with FIXED IMBALANCE HANDLING
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.header("🔄 Step 2: Process Data")
    
    if 'df' not in st.session_state:
        st.warning("Please upload data first")
        if st.button("🔙 Back to Upload"):
            st.session_state.step = 1
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        df = st.session_state.df
        st.write(f"📊 Working with {len(df):,} samples")
        
        # 🎯 KEY ADDITION: Imbalance Fixing Controls
        st.subheader("⚖️ Fix Emotion Imbalance (Recommended)")
        
        col1, col2 = st.columns(2)
        with col1:
            fix_imbalance = st.checkbox(
                "🎯 **Fix Neutral Emotion Bias**", 
                value=True,
                help="Highly recommended: Reduces neutral samples and balances emotions"
            )
            
            if fix_imbalance:
                max_neutral = st.slider(
                    "Max Neutral Samples", 
                    min_value=3000, 
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
        process_button = st.button("🚀 Process Data", type="primary", key="process_btn")
        
        # Process data when button is clicked
        if process_button:
            with st.spinner("Processing data with imbalance fixing..."):
                try:
                    # 🎯 Pass imbalance fixing parameters
                    X_train, X_test, y_train, y_test = preprocessor.process_data(
                        df, 
                        sample_size=sample_size,
                        fix_imbalance=fix_imbalance
                    )
                    
                    if X_train is not None and X_test is not None:
                        # Store in session state
                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
                        st.session_state.data_processed = True
                        
                        st.success("✅ Data processed successfully with bias correction!")
                        st.balloons()
                        
                        # Show results
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Training Samples", len(X_train))
                        with col2:
                            st.metric("Test Samples", len(X_test))
                        with col3:
                            st.metric("Ready for Training", "✅")
                        
                        # AUTO-ADVANCE TO NEXT STEP
                        st.info("🚀 **Auto-advancing to Step 3 in 2 seconds...**")
                        time.sleep(2)
                        st.session_state.step = 3
                        st.rerun()
                        
                    else:
                        st.error("❌ Data processing failed")
                        
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # ALWAYS show continue button if data exists
        st.divider()
        st.subheader("📋 Current Status")
        
        if all(key in st.session_state for key in ['X_train', 'X_test', 'y_train', 'y_test']):
            st.success("✅ Data is processed and ready for training!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Training Samples", len(st.session_state.X_train))
            with col2:
                st.metric("Test Samples", len(st.session_state.X_test))
            with col3:
                st.metric("Status", "Ready ✅")
            
            # BIG PROMINENT BUTTON
            st.markdown("### 🎯 Ready to Train Models!")
            
            # Multiple ways to continue
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🤖 **CONTINUE TO TRAINING**", type="primary", key="continue_big"):
                    st.session_state.step = 3
                    st.success("Moving to Step 3...")
                    st.rerun()
            
            with col2:
                if st.button("🚀 Skip to Training Now", key="skip_to_training"):
                    st.session_state.step = 3
                    st.rerun()
        
        else:
            st.info("⏳ Process your data first using the button above")
        
        st.markdown('</div>', unsafe_allow_html=True)

elif current_step == 3:
    # Step 3: Train Models - SAME AS BEFORE (already good)
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.header("🤖 Step 3: Train Models")
    
    # Check if we have the required data
    if not all(key in st.session_state for key in ['X_train', 'X_test', 'y_train', 'y_test']):
        st.warning("Please process data first")
        if st.button("🔙 Back to Processing"):
            st.session_state.step = 2
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.write(f"📊 Ready to train with {len(st.session_state.X_train):,} training samples")
        
        # Model Selection Section
        st.subheader("🎛️ Model Configuration")
        
        # Model selection dropdown
        model_options = {
            'bert-base-uncased': {
                'name': 'BERT Base Uncased (Recommended)',
                'description': '✅ **Best for emotion detection** - Optimized for social media text',
                'pros': '• Perfect for Reddit/social media text\n• Proven 75-85% accuracy on emotion tasks\n• Fast training (12 layers, 768 dimensions)\n• Memory efficient\n• Research-proven baseline',
                'cons': '• None significant for this task'
            },
            'bert-large-uncased': {
                'name': 'BERT Large Uncased',
                'description': '⚠️ **Slower but slightly more accurate** - 2x computational cost',
                'pros': '• ~3% better accuracy potential\n• More parameters (24 layers, 1024 dimensions)',
                'cons': '• 2x slower training\n• Requires more memory\n• Marginal improvement for emotion tasks'
            },
            'distilbert-base-uncased': {
                'name': 'DistilBERT Base Uncased',
                'description': '⚡ **Fastest option** - Good for quick prototyping',
                'pros': '• 40% faster training\n• Lower memory requirements\n• Good for testing',
                'cons': '• ~5% lower accuracy\n• Less robust for complex emotions'
            },
            'roberta-base': {
                'name': 'RoBERTa Base',
                'description': '🔬 **Research alternative** - Often better but slower',
                'pros': '• Sometimes better performance\n• Robust to different text types',
                'cons': '• Much slower training\n• More complex tokenization\n• Overkill for this task'
            }
        }
        
        selected_model = st.selectbox(
            "Choose BERT Model:",
            options=list(model_options.keys()),
            index=0,  # Default to bert-base-uncased
            format_func=lambda x: model_options[x]['name']
        )
        
        # Show model details
        model_info = model_options[selected_model]
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Selected:** {model_info['description']}")
            st.markdown(f"**Advantages:**\n{model_info['pros']}")
        
        with col2:
            st.markdown(f"**Considerations:**\n{model_info['cons']}")
            
            if selected_model == 'bert-base-uncased':
                st.success("🏆 **Recommended Choice!** Perfect balance of speed and accuracy for emotion detection.")
            elif selected_model == 'bert-large-uncased':
                st.warning("⚠️ Will take 2x longer to train. Only use if you need maximum accuracy.")
            elif selected_model == 'distilbert-base-uncased':
                st.info("⚡ Good for quick testing, but accuracy may be lower.")
            else:
                st.info("🔬 Advanced option. May not provide better results for emotion detection.")
        
        # Update the embedder with selected model
        if selected_model != bert_embedder.model_name:
            bert_embedder.model_name = selected_model
            # Clear any previously loaded model
            bert_embedder.model = None
            bert_embedder.tokenizer = None
        
        st.divider()
        
        # Check if models are already trained
        if st.session_state.get('models_trained', False):
            # Models already trained - show results and continue button
            st.success("✅ Models have been trained successfully!")
            
            # Show training results
            if all(key in st.session_state for key in ['X_train_embeddings', 'X_test_embeddings', 'classifiers']):
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("BERT Model", selected_model)
                    st.metric("Embedding Dimension", "768" if 'base' in selected_model else "1024")
                with col2:
                    st.metric("Training Samples", len(st.session_state.X_train_embeddings))
                    st.metric("Test Samples", len(st.session_state.X_test_embeddings))
                with col3:
                    st.metric("Models Trained", "2")
                    st.metric("Status", "✅ Ready")
            
            # Big prominent button to continue
            st.markdown("### 📊 Ready for Model Evaluation")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 **CONTINUE TO EVALUATION**", type="primary", key="proceed_to_evaluation"):
                    st.session_state.step = 4
                    st.rerun()
            
            with col2:
                if st.button("🚀 Skip to Evaluation Now", key="skip_to_evaluation"):
                    st.session_state.step = 4
                    st.rerun()
            
            # Option to retrain
            if st.button("🔄 Retrain Models", key="retrain_models"):
                st.session_state.models_trained = False
                # Clear old training data
                for key in ['X_train_embeddings', 'X_test_embeddings', 'classifiers', 'bert_embedder']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        
        else:
            # Models not yet trained - show training interface
            st.write(f"Ready to train **{model_info['name']}** and classification models:")
            
            train_button = st.button("🚀 Start Training", type="primary", key="start_training_btn")
            
            # Start training when button is clicked
            if train_button:
                progress_bar = st.progress(0)
                status = st.empty()
                
                try:
                    # Step 3.1: Generate BERT embeddings
                    status.text(f"Loading {selected_model} model...")
                    success = bert_embedder.load_model()
                    if not success:
                        st.error("❌ Failed to load BERT model")
                        st.stop()
                    
                    progress_bar.progress(20)
                    
                    status.text("Generating training embeddings...")
                    X_train_embeddings = bert_embedder.generate_embeddings(st.session_state.X_train)
                    
                    if X_train_embeddings is None:
                        st.error("❌ Failed to generate training embeddings")
                        st.stop()
                    
                    progress_bar.progress(40)
                    
                    status.text("Generating test embeddings...")
                    X_test_embeddings = bert_embedder.generate_embeddings(st.session_state.X_test)
                    
                    if X_test_embeddings is None:
                        st.error("❌ Failed to generate test embeddings")
                        st.stop()
                    
                    progress_bar.progress(60)
                    
                    # Step 3.2: Train models
                    status.text("Training Naive Bayes with class balancing...")
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
                    st.session_state.models_trained = True  # Set completion flag
                    st.session_state.selected_model = selected_model  # Store selected model
                    
                    status.text("✅ Training completed!")
                    st.success("🎉 Models trained successfully with bias correction!")
                    st.balloons()
                    
                    # Show training summary
                    st.subheader("📋 Training Summary")
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
                            models_trained.append("Naive Bayes")
                        if rf_success:
                            models_trained.append("Random Forest")
                        st.metric("Models Trained", len(models_trained))
                        st.metric("Status", "✅ Ready")
                    
                    # AUTO-ADVANCE TO NEXT STEP
                    st.info("📊 **Auto-advancing to Step 4 in 2 seconds...**")
                    time.sleep(2)
                    st.session_state.step = 4
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")
                    st.exception(e)
        
        # ALWAYS show continue button if models exist
        st.divider()
        st.subheader("📋 Current Status")
        
        if all(key in st.session_state for key in ['X_train_embeddings', 'X_test_embeddings', 'classifiers']):
            st.success("✅ Models are trained and ready for evaluation!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Embeddings Generated", "✅")
            with col2:
                st.metric("Models Trained", "✅")
            with col3:
                st.metric("Status", "Ready for Evaluation")
            
            # BIG PROMINENT BUTTON
            st.markdown("### 🎯 Ready to Evaluate Performance!")
            
            # Multiple ways to continue
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📊 **CONTINUE TO EVALUATION**", type="primary", key="continue_eval_big"):
                    st.session_state.step = 4
                    st.success("Moving to Step 4...")
                    st.rerun()
            
            with col2:
                if st.button("🚀 Skip to Evaluation", key="skip_eval_now"):
                    st.session_state.step = 4
                    st.rerun()
        
        else:
            st.info("⏳ Train your models first using the button above")
        
        st.markdown('</div>', unsafe_allow_html=True)

elif current_step == 4:
    # Step 4: Evaluate Models with FIXED EVALUATION
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.header("📊 Step 4: Evaluate Models")
    
    if all(key in st.session_state for key in ['X_test_embeddings', 'y_test', 'classifiers']):
        
        if st.button("📊 Evaluate Models", type="primary"):
            with st.spinner("Evaluating models with bias-aware metrics..."):
                results = evaluator.evaluate_models(
                    st.session_state.classifiers,
                    st.session_state.X_test_embeddings,
                    st.session_state.y_test
                )
                
                if results:
                    st.session_state.results = results
                    
                    # 🎯 NEW: Show the improved performance summary
                    summary_data = evaluator.display_performance_summary(results)
                    
                    # Show recommendations
                    st.subheader("💡 Recommendations")
                    recommendations = evaluator.get_model_recommendations(results)
                    for recommendation in recommendations:
                        st.write(recommendation)
                    
                    # Show metrics explanation
                    evaluator.explain_metrics()
                    
                    # Show which model is best for predictions
                    if summary_data and len(summary_data) > 1:
                        best_model_data = max(summary_data, key=lambda x: float(x['Composite Score'].rstrip('%')))
                        best_model_name = best_model_data['Model'].lower().replace(' ', '_')
                        st.session_state.best_model_for_prediction = best_model_name
                        
                        st.success(f"🏆 **{best_model_data['Model']}** will be used for predictions")
                    else:
                        # Default to random forest if only one model or comparison fails
                        st.session_state.best_model_for_prediction = 'random_forest'
                    
                    if st.button("✅ Continue to Predictions", type="primary"):
                        st.session_state.step = 5
                        st.rerun()
    else:
        st.warning("Please train models first")
        if st.button("🔙 Back to Training"):
            st.session_state.step = 3
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

elif current_step == 5:
    # Step 5: Make Predictions with IMPROVED PREDICTIONS
    st.markdown('<div class="step-container">', unsafe_allow_html=True)
    st.header("🎯 Step 5: Make Predictions")
    
    if all(key in st.session_state for key in ['classifiers', 'bert_embedder']):
        
        # Determine best model to use
        best_model = st.session_state.get('best_model_for_prediction', 'random_forest')
        
        # Show best model info
        if 'results' in st.session_state:
            best_model_display = best_model.replace('_', ' ').title()
            
            if best_model in st.session_state.results:
                metrics = st.session_state.results[best_model]
                composite_score = evaluator.calculate_performance_percentage(metrics)
                balance_score = metrics.get('balance_score', 0) * 100
                
                st.info(f"🏆 Using best model: **{best_model_display}** "
                       f"(Composite Score: {composite_score:.1f}%, Balance Score: {balance_score:.1f}%)")
            else:
                st.info(f"🤖 Using model: **{best_model_display}**")
        else:
            st.info(f"🤖 Using model: **{best_model.replace('_', ' ').title()}**")
        
        # Prediction tabs with original styling
        tab1, tab2 = st.tabs(["📝 Single Text", "📁 Batch Upload"])
        
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
                    with st.spinner("Analyzing emotions with bias-corrected model..."):
                        results = classifiers.predict_single_text(
                            text_input, 
                            st.session_state.bert_embedder,
                            model_type=best_model
                        )
                        
                        if results:
                            st.subheader("🎭 Predicted Emotions")
                            
                            # Sort emotions by confidence
                            sorted_emotions = sorted(results.items(), key=lambda x: x[1], reverse=True)
                            
                            # Show top emotions with confidence bars
                            for emotion, confidence in sorted_emotions[:5]:  # Top 5
                                confidence_pct = confidence * 100
                                
                                # Color coding based on confidence
                                if confidence_pct >= 50:
                                    color = "🟢"
                                elif confidence_pct >= 30:
                                    color = "🟡"
                                else:
                                    color = "🔶"
                                
                                # Create a simple progress bar using streamlit
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.write(f"{color} **{emotion.title()}**")
                                    st.progress(confidence)
                                with col2:
                                    st.metric("", f"{confidence_pct:.1f}%")
                            
                            # Show emotional insight
                            top_emotion = sorted_emotions[0][0]
                            top_confidence = sorted_emotions[0][1] * 100
                            
                            if top_confidence >= 60:
                                insight = f"Strong {top_emotion} emotion detected!"
                            elif top_confidence >= 40:
                                insight = f"Moderate {top_emotion} emotion detected."
                            else:
                                insight = f"Weak {top_emotion} signal. Text may be emotionally neutral or mixed."
                            
                            st.info(f"🧠 **Insight**: {insight}")
                else:
                    st.warning("Please enter some text")
        
        with tab2:
            st.subheader("📁 Batch File Prediction")
            
            uploaded_file = st.file_uploader(
                "Upload CSV file with 'text' column",
                type=['csv'],
                key="batch_upload"
            )
            
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    if 'text' not in df.columns:
                        st.error("❌ File must contain a 'text' column")
                    else:
                        st.success(f"✅ Loaded {len(df):,} texts for prediction")
                        st.dataframe(df.head(), use_container_width=True)
                        
                        if st.button("🚀 Process Batch", type="primary"):
                            with st.spinner(f"Processing {len(df):,} texts with bias-corrected model..."):
                                # Process batch
                                results_df = classifiers.predict_batch(
                                    df, 
                                    st.session_state.bert_embedder,
                                    model_type=best_model
                                )
                                
                                if results_df is not None:
                                    st.success(f"✅ Processed {len(results_df):,} texts!")
                                    
                                    # Show first 50 results
                                    st.subheader("📊 First 50 Results")
                                    display_df = results_df.head(50).copy()
                                    
                                    # Format for display
                                    if 'text' in display_df.columns:
                                        display_df['text'] = display_df['text'].apply(lambda x: x[:80] + "..." if len(str(x)) > 80 else x)
                                    
                                    if 'confidence' in display_df.columns:
                                        display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x*100:.1f}%")
                                    
                                    # Select columns to display
                                    display_columns = ['text', 'top_emotion', 'confidence', 'top_3_emotions']
                                    available_columns = [col for col in display_columns if col in display_df.columns]
                                    
                                    if available_columns:
                                        st.dataframe(display_df[available_columns], use_container_width=True)
                                    else:
                                        st.dataframe(display_df, use_container_width=True)
                                    
                                    # Download section
                                    st.subheader("💾 Download Results")
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
                                        # Create summary
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
                
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
    else:
        st.warning("Please complete evaluation first")
        if st.button("🔙 Back to Evaluation"):
            st.session_state.step = 4
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar with original sophisticated styling
with st.sidebar:
    st.header("📊 System Status")
    
    status_items = [
        ("📁 Data Loaded", 'df' in st.session_state),
        ("🔄 Data Processed", 'data_processed' in st.session_state and st.session_state.data_processed),
        ("🤖 Models Trained", 'models_trained' in st.session_state and st.session_state.models_trained),
        ("📊 Models Evaluated", 'results' in st.session_state)
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
    
    # Next step guidance
    if completed_steps < total_steps:
        next_steps = [
            "📁 Upload data in Step 1",
            "🔄 Process data in Step 2", 
            "🤖 Train models in Step 3",
            "📊 Evaluate in Step 4"
        ]
        
        st.subheader("Next Step")
        st.info(next_steps[completed_steps])
    else:
        st.success("🎉 All steps completed! Ready for predictions.")
    
    st.divider()
    
    # Show bias correction status
    if 'df' in st.session_state and 'neutral' in st.session_state.df.columns:
        st.header("⚖️ Bias Status")
        
        # Original bias
        original_neutral = (st.session_state.df['neutral'] == 1).sum()
        original_total = len(st.session_state.df)
        original_pct = (original_neutral / original_total) * 100
        
        if original_pct > 40:
            st.error(f"🔴 Original: {original_pct:.1f}% neutral")
        elif original_pct > 25:
            st.warning(f"🟡 Original: {original_pct:.1f}% neutral")
        else:
            st.success(f"🟢 Original: {original_pct:.1f}% neutral")
        
        if st.session_state.get('data_processed', False):
            st.success("✅ Bias correction applied!")
    
    st.divider()
    
    # Show selected model if available
    if 'selected_model' in st.session_state:
        st.header("🤖 Current Model")
        model_name = st.session_state.selected_model
        st.success(f"✅ {model_name}")
        
        if model_name == 'bert-base-uncased':
            st.caption("🏆 Recommended choice!")
        elif model_name == 'bert-large-uncased':
            st.caption("🔬 High accuracy mode")
        elif model_name == 'distilbert-base-uncased':
            st.caption("⚡ Fast mode")
        else:
            st.caption("🔬 Advanced mode")
    
    st.divider()
    
    # Manual step navigation (for debugging/troubleshooting)
    st.header("🔧 Manual Navigation")
    st.caption("Use this if you get stuck between steps")
    
    current_step_num = st.session_state.get('step', 1)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Previous Step") and current_step_num > 1:
            st.session_state.step = current_step_num - 1
            st.rerun()
    
    with col2:
        if st.button("Next Step →") and current_step_num < 5:
            # Check if we can advance
            can_advance = False
            if current_step_num == 1 and 'df' in st.session_state:
                can_advance = True
            elif current_step_num == 2 and st.session_state.get('data_processed', False):
                can_advance = True
            elif current_step_num == 3 and st.session_state.get('models_trained', False):
                can_advance = True
            elif current_step_num == 4 and 'results' in st.session_state:
                can_advance = True
            else:
                can_advance = True  # Allow manual override
            
            if can_advance:
                st.session_state.step = current_step_num + 1
                st.rerun()
            else:
                st.warning("⚠️ Complete current step first")
    
    st.write(f"Current Step: {current_step_num}/5")
    
    st.divider()
    
    # Model performance with improved metrics
    if hasattr(st.session_state, 'results') and st.session_state.results:
        st.header("🏆 Model Performance")
        results = st.session_state.results
        
        for model_name, metrics in results.items():
            composite_score = evaluator.calculate_performance_percentage(metrics)
            balance_score = metrics.get('balance_score', 0) * 100
            emotion_f1 = metrics.get('non_neutral_f1', 0) * 100
            
            if composite_score >= 80 and balance_score >= 70:
                status_icon = "🟢"
                status_text = "Excellent"
            elif composite_score >= 70 and balance_score >= 60:
                status_icon = "🟡"
                status_text = "Good"
            elif emotion_f1 >= 50:
                status_icon = "🟠"
                status_text = "Fair"
            else:
                status_icon = "🔴"
                status_text = "Needs Work"
            
            st.write(f"**{model_name.replace('_', ' ').title()}**")
            st.write(f"{status_icon} {composite_score:.1f}% {status_text}")
            st.write(f"   Balance: {balance_score:.1f}%")
    
    st.divider()
    
    # Quick stats
    if 'df' in st.session_state:
        st.header("📈 Quick Stats")
        df = st.session_state.df
        st.metric("Dataset Size", f"{len(df):,}")
        st.metric("Emotions", "27")
        if hasattr(st.session_state, 'X_train_embeddings'):
            embedding_dim = "768" if 'base' in st.session_state.get('selected_model', 'bert-base-uncased') else "1024"
            st.metric("Embedding Dim", embedding_dim)
    
    st.divider()
    
    # Reset button with original styling
    if st.button("🔄 Start Over"):
        for key in list(st.session_state.keys()):
            if key != 'step':
                del st.session_state[key]
        st.session_state.step = 1
        st.rerun()

# Footer with original sophisticated styling
st.divider()
st.markdown(
    "<div class='footer'>"
    "<p><strong>🎯 GoEmotions: Bias-Corrected Emotion Detection System</strong></p>"
    "<p>Built with Streamlit + BERT + Scikit-learn | "
    "Fixed Neutral Bias | Improved Emotion Balance | Production-Ready Predictions</p>"
    "</div>",
    unsafe_allow_html=True
)