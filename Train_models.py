import numpy as np
import pandas as pd
import streamlit as st
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import time

class SimpleEmotionClassifiers:
    def __init__(self):
        self.nb_classifier = None
        self.rf_classifier = None
        
        self.emotion_labels = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
            'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]
    
    def compute_emotion_weights(self, y):
        """🎯 Compute class weights for each emotion to handle imbalance"""
        weights_per_emotion = {}
        
        for i, emotion in enumerate(self.emotion_labels):
            emotion_column = y[:, i]
            unique_classes = np.unique(emotion_column)
            
            if len(unique_classes) > 1:
                # Compute weights for this emotion
                class_weights = compute_class_weight(
                    'balanced', 
                    classes=unique_classes, 
                    y=emotion_column
                )
                weights_per_emotion[emotion] = dict(zip(unique_classes, class_weights))
            else:
                # Only one class present
                weights_per_emotion[emotion] = {unique_classes[0]: 1.0}
        
        return weights_per_emotion
    
    def train_naive_bayes(self, X, y):
        """Train Naive Bayes with balanced class weights"""
        try:
            st.write("🧠 Training Naive Bayes with class balancing...")
            
            # Compute class weights
            emotion_weights = self.compute_emotion_weights(y)
            
            # Create individual classifiers with class weights
            classifiers = []
            for i, emotion in enumerate(self.emotion_labels):
                emotion_column = y[:, i]
                
                # Get weights for this emotion
                weights = emotion_weights.get(emotion, {0: 1.0, 1: 1.0})
                
                # Create classifier with computed weights
                if len(np.unique(emotion_column)) > 1:
                    # Create sample weights
                    sample_weights = np.array([weights.get(label, 1.0) for label in emotion_column])
                    
                    # Use smoothed alpha for better performance
                    nb = MultinomialNB(alpha=0.1)
                    
                    # Create pipeline with scaling
                    pipeline = Pipeline([
                        ('scaler', MinMaxScaler()),
                        ('classifier', nb)
                    ])
                    
                    # Fit with sample weights
                    pipeline.fit(X, emotion_column, classifier__sample_weight=sample_weights)
                    classifiers.append(pipeline)
                else:
                    # Handle case where emotion has only one class
                    from sklearn.dummy import DummyClassifier
                    dummy = DummyClassifier(strategy='constant', constant=emotion_column[0])
                    dummy.fit(X, emotion_column)
                    classifiers.append(dummy)
            
            # Create custom multi-output wrapper
            class BalancedMultiOutputNB:
                def __init__(self, classifiers, emotion_labels):
                    self.classifiers = classifiers
                    self.emotion_labels = emotion_labels
                
                def predict(self, X):
                    predictions = []
                    for clf in self.classifiers:
                        pred = clf.predict(X)
                        predictions.append(pred)
                    return np.array(predictions).T
                
                def predict_proba(self, X):
                    probabilities = []
                    for clf in self.classifiers:
                        if hasattr(clf, 'predict_proba'):
                            proba = clf.predict_proba(X)
                            if proba.shape[1] == 2:
                                probabilities.append(proba[:, 1])  # Positive class
                            else:
                                probabilities.append(proba[:, 0])
                        else:
                            # Dummy classifier
                            pred = clf.predict(X)
                            probabilities.append(pred.astype(float))
                    return np.array(probabilities).T
            
            self.nb_classifier = BalancedMultiOutputNB(classifiers, self.emotion_labels)
            
            st.success("✅ Naive Bayes trained with class balancing")
            return True
            
        except Exception as e:
            st.error(f"Error training Naive Bayes: {str(e)}")
            return False
    
    def train_random_forest(self, X, y):
        """🎯 Train Random Forest with optimized settings for emotion imbalance"""
        try:
            st.write("🌲 Training Random Forest with imbalance handling...")
            
            # Create Random Forest with balanced class weights and optimized parameters
            rf = RandomForestClassifier(
                n_estimators=150,          # More trees for better performance
                max_depth=15,              # Deeper trees for emotion complexity
                min_samples_split=5,       # Prevent overfitting
                min_samples_leaf=2,        # Prevent overfitting
                class_weight='balanced',   # 🎯 KEY: Handle class imbalance
                random_state=42,
                n_jobs=-1,
                bootstrap=True,
                max_features='sqrt',       # Good default for classification
                criterion='gini'           # Better for imbalanced data
            )
            
            # Use MultiOutputClassifier with balanced settings
            classifier = MultiOutputClassifier(
                rf, 
                n_jobs=-1
            )
            
            # Fit the model
            classifier.fit(X, y)
            self.rf_classifier = classifier
            
            # Show feature importance for top emotions if possible
            try:
                feature_importances = []
                for i, estimator in enumerate(classifier.estimators_):
                    if hasattr(estimator, 'feature_importances_'):
                        importance = np.mean(estimator.feature_importances_)
                        feature_importances.append((self.emotion_labels[i], importance))
                
                # Show top 3 most important emotions
                feature_importances.sort(key=lambda x: x[1], reverse=True)
                top_emotions = feature_importances[:3]
                
                st.info("🎯 Top emotions by model importance:")
                for emotion, importance in top_emotions:
                    st.write(f"   • {emotion.title()}: {importance:.3f}")
                    
            except:
                pass  # Skip if feature importance extraction fails
            
            st.success("✅ Random Forest trained with class balancing")
            return True
            
        except Exception as e:
            st.error(f"Error training Random Forest: {str(e)}")
            return False
    
    def predict_single_text(self, text, bert_embedder, model_type='random_forest'):
        """🎯 Predict emotions with confidence threshold filtering"""
        try:
            # Generate embedding
            embedding = bert_embedder.get_single_embedding(text)
            if embedding is None:
                return None
            
            # Reshape for prediction
            embedding = embedding.reshape(1, -1)
            
            # Make prediction
            if model_type == 'naive_bayes' and self.nb_classifier:
                probabilities = self.nb_classifier.predict_proba(embedding)[0]
            elif model_type == 'random_forest' and self.rf_classifier:
                probabilities = self.rf_classifier.predict_proba(embedding)
                # Handle MultiOutputClassifier format
                if isinstance(probabilities, list):
                    proba_values = []
                    for emotion_proba in probabilities:
                        if emotion_proba.shape[1] == 2:  # Binary classifier
                            proba_values.append(emotion_proba[0, 1])  # Positive class
                        else:
                            proba_values.append(emotion_proba[0, 0])  # Single value
                    probabilities = np.array(proba_values)
                else:
                    probabilities = probabilities[0]
            else:
                st.error("Model not available")
                return None
            
            # 🎯 KEY FIX: Apply confidence threshold to reduce neutral bias
            CONFIDENCE_THRESHOLD = 0.3  # Only show emotions with >30% confidence
            
            # Create emotion-probability dictionary
            results = {}
            for emotion, prob in zip(self.emotion_labels, probabilities):
                # Apply threshold, but always include at least the top emotion
                if prob >= CONFIDENCE_THRESHOLD or emotion == self.emotion_labels[np.argmax(probabilities)]:
                    results[emotion] = float(prob)
            
            # If no emotions meet threshold, show top 3
            if len(results) == 0:
                top_3_idx = np.argsort(probabilities)[-3:][::-1]
                for idx in top_3_idx:
                    results[self.emotion_labels[idx]] = float(probabilities[idx])
            
            return results
            
        except Exception as e:
            st.error(f"Error in single text prediction: {str(e)}")
            return None
    
    def predict_batch(self, df, bert_embedder, model_type='random_forest'):
        """🎯 Predict emotions for batch with improved confidence handling"""
        try:
            if 'text' not in df.columns:
                st.error("DataFrame must contain 'text' column")
                return None
            
            texts = df['text'].tolist()
            
            # Remove empty texts
            texts = [str(text).strip() for text in texts if text and str(text).strip()]
            
            if not texts:
                st.error("No valid texts found after cleaning")
                return None
            
            results = []
            
            # Progress tracking
            progress_bar = st.progress(0)
            
            # Process in smaller batches
            batch_size = 50
            total_batches = (len(texts) + batch_size - 1) // batch_size
            
            for batch_idx in range(0, len(texts), batch_size):
                batch_texts = texts[batch_idx:batch_idx + batch_size]
                
                # Generate embeddings for batch
                embeddings = bert_embedder.generate_embeddings(batch_texts)
                if embeddings is None:
                    st.error(f"Failed to generate embeddings for batch {batch_idx//batch_size + 1}")
                    continue
                
                # Make predictions
                try:
                    if model_type == 'naive_bayes' and self.nb_classifier:
                        batch_probabilities = self.nb_classifier.predict_proba(embeddings)
                    elif model_type == 'random_forest' and self.rf_classifier:
                        batch_probabilities = self.rf_classifier.predict_proba(embeddings)
                    else:
                        st.error(f"Model {model_type} not available")
                        return None
                    
                except Exception as e:
                    st.error(f"Prediction error in batch {batch_idx//batch_size + 1}: {str(e)}")
                    continue
                
                # Process predictions
                try:
                    if isinstance(batch_probabilities, list):
                        # MultiOutputClassifier returns list of arrays
                        proba_matrix = np.array([
                            [proba[:, 1] if proba.shape[1] > 1 else proba[:, 0] for proba in batch_probabilities]
                        ]).T
                    else:
                        proba_matrix = batch_probabilities
                    
                except Exception as e:
                    st.error(f"Error processing probabilities: {str(e)}")
                    continue
                
                # 🎯 KEY FIX: Apply confidence threshold and emotion filtering
                CONFIDENCE_THRESHOLD = 0.25  # Lower threshold for batch processing
                
                # Process each text in batch
                for i, (text, probabilities) in enumerate(zip(batch_texts, proba_matrix)):
                    try:
                        # Find emotions above threshold
                        above_threshold = probabilities >= CONFIDENCE_THRESHOLD
                        
                        if np.any(above_threshold):
                            # Use emotions above threshold
                            filtered_probs = probabilities.copy()
                            filtered_probs[~above_threshold] = 0
                            top_idx = np.argmax(filtered_probs)
                        else:
                            # If no emotions above threshold, use highest
                            top_idx = np.argmax(probabilities)
                        
                        top_emotion = self.emotion_labels[top_idx]
                        top_confidence = float(probabilities[top_idx])
                        
                        # Get top 3 emotions above threshold or top 3 overall
                        if np.sum(above_threshold) >= 3:
                            candidate_indices = np.where(above_threshold)[0]
                            candidate_probs = probabilities[candidate_indices]
                            top_3_candidate_idx = np.argsort(candidate_probs)[-3:][::-1]
                            top_3_idx = candidate_indices[top_3_candidate_idx]
                        else:
                            top_3_idx = np.argsort(probabilities)[-3:][::-1]
                        
                        top_3_emotions = [self.emotion_labels[idx] for idx in top_3_idx]
                        top_3_scores = [float(probabilities[idx]) for idx in top_3_idx]
                        
                        # 🎯 Additional filtering: Don't show neutral if it's weak and other emotions exist
                        if (top_emotion == 'neutral' and 
                            top_confidence < 0.6 and 
                            len([s for s in top_3_scores if s > 0.2]) > 1):
                            # Find best non-neutral emotion
                            non_neutral_mask = np.array([e != 'neutral' for e in self.emotion_labels])
                            non_neutral_probs = probabilities.copy()
                            non_neutral_probs[~non_neutral_mask] = 0
                            if np.max(non_neutral_probs) > 0.15:  # If there's a decent non-neutral emotion
                                top_idx = np.argmax(non_neutral_probs)
                                top_emotion = self.emotion_labels[top_idx]
                                top_confidence = float(probabilities[top_idx])
                        
                        results.append({
                            'original_index': batch_idx + i,
                            'text': text,
                            'top_emotion': top_emotion,
                            'confidence': top_confidence,
                            'top_3_emotions': ', '.join(top_3_emotions),
                            'top_3_scores': ', '.join([f"{score:.3f}" for score in top_3_scores])
                        })
                        
                    except Exception as e:
                        st.error(f"Error processing text {i}: {str(e)}")
                        continue
                
                # Update progress
                current_progress = min(1.0, (batch_idx + batch_size) / len(texts))
                progress_bar.progress(current_progress)
            
            progress_bar.progress(1.0)
            
            if not results:
                st.error("No results generated")
                return None
            
            # Create results DataFrame
            results_df = pd.DataFrame(results)
            
            # 🎯 Show quick emotion distribution summary
            emotion_dist = results_df['top_emotion'].value_counts()
            st.subheader("📊 Predicted Emotion Distribution")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if 'neutral' in emotion_dist:
                    neutral_pct = (emotion_dist['neutral'] / len(results_df)) * 100
                    st.metric("Neutral", f"{neutral_pct:.1f}%")
                else:
                    st.metric("Neutral", "0%")
            
            with col2:
                non_neutral_count = len(results_df[results_df['top_emotion'] != 'neutral'])
                non_neutral_pct = (non_neutral_count / len(results_df)) * 100
                st.metric("Other Emotions", f"{non_neutral_pct:.1f}%")
            
            with col3:
                avg_confidence = results_df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            
            # Show top emotions
            st.write("🏆 **Top 5 Predicted Emotions:**")
            top_5_emotions = emotion_dist.head(5)
            for emotion, count in top_5_emotions.items():
                pct = (count / len(results_df)) * 100
                st.write(f"   • {emotion.title()}: {count:,} samples ({pct:.1f}%)")
            
            return results_df
            
        except Exception as e:
            st.error(f"Error in batch prediction: {str(e)}")
            st.exception(e)
            return None