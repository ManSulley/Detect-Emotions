import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

class SimpleModelEvaluator:
    def __init__(self):
        self.emotion_labels = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
            'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]
    
    def evaluate_models(self, classifiers, X_test, y_test):
        """🎯 Evaluate models with metrics designed for imbalanced emotion data"""
        results = {}
        
        try:
            # Evaluate Naive Bayes
            if classifiers.nb_classifier:
                nb_metrics = self._evaluate_single_model(
                    classifiers.nb_classifier, X_test, y_test, "Naive Bayes"
                )
                if nb_metrics:
                    results['naive_bayes'] = nb_metrics
            
            # Evaluate Random Forest
            if classifiers.rf_classifier:
                rf_metrics = self._evaluate_single_model(
                    classifiers.rf_classifier, X_test, y_test, "Random Forest"
                )
                if rf_metrics:
                    results['random_forest'] = rf_metrics
            
            # 🎯 Show detailed comparison
            if results:
                self._show_detailed_comparison(results, y_test)
            
            return results
            
        except Exception as e:
            st.error(f"Error evaluating models: {str(e)}")
            return None
    
    def _evaluate_single_model(self, model, X_test, y_test, model_name):
        """🎯 Evaluate model with focus on imbalanced emotion metrics"""
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Handle different probability formats
            if isinstance(y_pred_proba, list):
                proba_matrix = np.array([
                    [proba[1] if proba.shape[0] > 1 else 0.5 for proba in sample_probas]
                    for sample_probas in zip(*y_pred_proba)
                ])
            elif len(y_pred_proba.shape) == 2:
                proba_matrix = y_pred_proba
            else:
                proba_matrix = np.zeros_like(y_pred, dtype=float)
            
            # 🎯 KEY METRICS for imbalanced emotion data
            metrics = {}
            
            # 1. Overall metrics
            metrics['accuracy'] = float(accuracy_score(y_test, y_pred))
            metrics['macro_f1'] = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
            metrics['weighted_f1'] = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
            metrics['macro_precision'] = float(precision_score(y_test, y_pred, average='macro', zero_division=0))
            metrics['macro_recall'] = float(recall_score(y_test, y_pred, average='macro', zero_division=0))
            
            # 2. 🎯 IMPORTANT: Per-emotion performance (key for imbalanced data)
            emotion_performance = {}
            for i, emotion in enumerate(self.emotion_labels):
                if i < y_test.shape[1]:
                    y_true_emotion = y_test[:, i]
                    y_pred_emotion = y_pred[:, i]
                    
                    # Only calculate if emotion exists in test set
                    if len(np.unique(y_true_emotion)) > 1:
                        emotion_f1 = f1_score(y_true_emotion, y_pred_emotion, zero_division=0)
                        emotion_precision = precision_score(y_true_emotion, y_pred_emotion, zero_division=0)
                        emotion_recall = recall_score(y_true_emotion, y_pred_emotion, zero_division=0)
                        
                        emotion_performance[emotion] = {
                            'f1': float(emotion_f1),
                            'precision': float(emotion_precision),
                            'recall': float(emotion_recall),
                            'support': int(np.sum(y_true_emotion))
                        }
            
            metrics['emotion_performance'] = emotion_performance
            
            # 3. 🎯 Neutral vs Non-neutral performance (crucial metric)
            neutral_idx = self.emotion_labels.index('neutral') if 'neutral' in self.emotion_labels else -1
            if neutral_idx < y_test.shape[1] and neutral_idx >= 0:
                neutral_true = y_test[:, neutral_idx]
                neutral_pred = y_pred[:, neutral_idx]
                
                # Non-neutral = any emotion except neutral
                non_neutral_true = np.any(y_test[:, [i for i in range(y_test.shape[1]) if i != neutral_idx]] == 1, axis=1)
                non_neutral_pred = np.any(y_pred[:, [i for i in range(y_pred.shape[1]) if i != neutral_idx]] == 1, axis=1)
                
                metrics['neutral_accuracy'] = float(accuracy_score(neutral_true, neutral_pred))
                metrics['non_neutral_accuracy'] = float(accuracy_score(non_neutral_true, non_neutral_pred))
                
                metrics['neutral_f1'] = float(f1_score(neutral_true, neutral_pred, zero_division=0))
                metrics['non_neutral_f1'] = float(f1_score(non_neutral_true, non_neutral_pred, zero_division=0))
                
                # Balance score: how well does model balance neutral vs emotions
                neutral_ratio_true = np.mean(neutral_true)
                neutral_ratio_pred = np.mean(neutral_pred)
                balance_score = 1 - abs(neutral_ratio_true - neutral_ratio_pred)
                metrics['balance_score'] = float(balance_score)
            
            # 4. Exact match accuracy
            exact_matches = np.all(y_test == y_pred, axis=1)
            metrics['exact_match'] = float(np.mean(exact_matches))
            
            # 5. 🎯 Confidence metrics
            if proba_matrix.size > 0:
                avg_confidence = np.mean(np.max(proba_matrix, axis=1))
                metrics['avg_confidence'] = float(avg_confidence)
                
                # Confidence distribution
                high_confidence = np.mean(np.max(proba_matrix, axis=1) > 0.7)
                metrics['high_confidence_ratio'] = float(high_confidence)
            
            st.success(f"✅ {model_name} evaluated with {len(emotion_performance)} emotions")
            
            return metrics
            
        except Exception as e:
            st.error(f"Error evaluating {model_name}: {str(e)}")
            return None
    
    def _show_detailed_comparison(self, results, y_test):
        """🎯 Show detailed model comparison focused on emotion balance"""
        st.subheader("🎯 Detailed Model Comparison")
        
        # 1. Overall Performance Summary
        st.write("**📊 Overall Performance:**")
        comparison_data = []
        
        for model_name, metrics in results.items():
            model_display_name = model_name.replace('_', ' ').title()
            
            # Calculate composite score that penalizes neutral bias
            accuracy = metrics.get('accuracy', 0) * 100
            macro_f1 = metrics.get('macro_f1', 0) * 100
            balance_score = metrics.get('balance_score', 0) * 100
            non_neutral_f1 = metrics.get('non_neutral_f1', 0) * 100
            
            # Composite score emphasizing emotion detection over neutral
            composite_score = (accuracy * 0.3 + macro_f1 * 0.3 + balance_score * 0.2 + non_neutral_f1 * 0.2)
            
            comparison_data.append({
                'Model': model_display_name,
                'Overall Accuracy': f"{accuracy:.1f}%",
                'Macro F1-Score': f"{macro_f1:.1f}%",
                'Balance Score': f"{balance_score:.1f}%",
                'Non-Neutral F1': f"{non_neutral_f1:.1f}%",
                'Composite Score': f"{composite_score:.1f}%"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # 2. 🎯 Neutral vs Emotion Performance
        st.write("**⚖️ Neutral vs Emotion Balance:**")
        
        balance_data = []
        for model_name, metrics in results.items():
            model_display_name = model_name.replace('_', ' ').title()
            
            neutral_f1 = metrics.get('neutral_f1', 0) * 100
            non_neutral_f1 = metrics.get('non_neutral_f1', 0) * 100
            balance_score = metrics.get('balance_score', 0) * 100
            
            # Determine balance status
            if balance_score >= 80:
                status = "🟢 Well Balanced"
            elif balance_score >= 60:
                status = "🟡 Moderately Balanced"
            else:
                status = "🔴 Imbalanced"
            
            balance_data.append({
                'Model': model_display_name,
                'Neutral F1': f"{neutral_f1:.1f}%",
                'Emotion F1': f"{non_neutral_f1:.1f}%",
                'Balance Score': f"{balance_score:.1f}%",
                'Status': status
            })
        
        balance_df = pd.DataFrame(balance_data)
        st.dataframe(balance_df, use_container_width=True, hide_index=True)
        
        # 3. Top Performing Emotions
        st.write("**🏆 Top Performing Emotions per Model:**")
        
        for model_name, metrics in results.items():
            model_display_name = model_name.replace('_', ' ').title()
            emotion_perf = metrics.get('emotion_performance', {})
            
            if emotion_perf:
                # Sort emotions by F1 score
                sorted_emotions = sorted(
                    emotion_perf.items(), 
                    key=lambda x: x[1]['f1'], 
                    reverse=True
                )
                
                st.write(f"**{model_display_name}:**")
                
                # Show top 5 emotions
                top_emotions = sorted_emotions[:5]
                cols = st.columns(len(top_emotions))
                
                for i, (emotion, perf) in enumerate(top_emotions):
                    with cols[i]:
                        f1_score = perf['f1'] * 100
                        support = perf['support']
                        
                        if f1_score >= 70:
                            icon = "🟢"
                        elif f1_score >= 50:
                            icon = "🟡"
                        else:
                            icon = "🔴"
                        
                        st.metric(
                            f"{icon} {emotion.title()}", 
                            f"{f1_score:.1f}%",
                            delta=f"{support} samples"
                        )
    
    def calculate_performance_percentage(self, metrics):
        """🎯 Calculate performance emphasizing emotion detection over neutral bias"""
        if not metrics:
            return 0.0
        
        # Weighted scoring that penalizes neutral bias
        weights = {
            'macro_f1': 0.35,           # Emphasis on balanced emotion detection
            'non_neutral_f1': 0.25,     # Emphasis on detecting actual emotions
            'balance_score': 0.20,      # Penalty for neutral bias
            'accuracy': 0.20            # Overall accuracy (but lower weight)
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                weighted_score += metrics[metric] * weight
                total_weight += weight
        
        if total_weight > 0:
            return (weighted_score / total_weight) * 100
        else:
            return 0.0
    
    def display_performance_summary(self, results):
        """🎯 Display performance summary with focus on emotion balance"""
        st.subheader("📈 Performance Summary")
        
        if not results:
            st.error("No results to display")
            return
        
        summary_data = []
        
        for model_name, metrics in results.items():
            performance_pct = self.calculate_performance_percentage(metrics)
            
            # 🎯 More nuanced status based on emotion balance
            balance_score = metrics.get('balance_score', 0) * 100
            non_neutral_f1 = metrics.get('non_neutral_f1', 0) * 100
            
            if performance_pct >= 80 and balance_score >= 70:
                status = "🟢 Excellent & Balanced"
            elif performance_pct >= 70 and balance_score >= 60:
                status = "🟡 Good Performance"
            elif performance_pct >= 60 or non_neutral_f1 >= 50:
                status = "🟠 Fair (Check Balance)"
            else:
                status = "🔴 Needs Improvement"
            
            summary_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Composite Score': f"{performance_pct:.1f}%",
                'Emotion F1': f"{non_neutral_f1:.1f}%",
                'Balance Score': f"{balance_score:.1f}%",
                'Overall Accuracy': f"{metrics.get('accuracy', 0) * 100:.1f}%",
                'Status': status
            })
        
        # Display as table
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Show best model with explanation
        if len(summary_data) > 1:
            best_model = max(summary_data, key=lambda x: float(x['Composite Score'].rstrip('%')))
            st.success(f"🏆 **Best Model:** {best_model['Model']} ({best_model['Composite Score']})")
            
            # Explain why this model is best
            st.info(f"**Why this model is best:**\n"
                   f"• Composite Score: {best_model['Composite Score']} (balanced performance)\n"
                   f"• Emotion Detection: {best_model['Emotion F1']} (non-neutral emotions)\n"
                   f"• Balance Score: {best_model['Balance Score']} (avoids neutral bias)")
        
        return summary_data
    
    def explain_metrics(self):
        """🎯 Explain metrics with focus on emotion imbalance"""
        with st.expander("📚 Understanding the Metrics (Emotion-Focused)"):
            st.write("""
            **🎯 Composite Score**: Overall performance emphasizing emotion detection over neutral bias
            
            **⚖️ Balance Score**: How well the model balances neutral vs actual emotions (higher = less biased)
            
            **🎭 Emotion F1**: Performance on detecting actual emotions (not neutral)
            
            **📊 Macro F1-Score**: Average performance across all emotions (treats each emotion equally)
            
            **🎪 Overall Accuracy**: Traditional accuracy (but less important for imbalanced data)
            
            **🎨 Status Guide**:
            - 🟢 **Excellent & Balanced** (80%+ composite, 70%+ balance): Production ready
            - 🟡 **Good Performance** (70%+ composite, 60%+ balance): Good for most uses
            - 🟠 **Fair** (60%+ composite OR 50%+ emotion F1): Usable but check for neutral bias
            - 🔴 **Needs Improvement**: Consider more balanced training data
            
            **💡 Key Insight**: For emotion detection, a model with 75% accuracy but good emotion balance 
            is better than 85% accuracy with heavy neutral bias!
            """)
    
    def get_model_recommendations(self, results):
        """🎯 Provide recommendations focused on emotion detection quality"""
        recommendations = []
        
        for model_name, metrics in results.items():
            performance = self.calculate_performance_percentage(metrics)
            balance_score = metrics.get('balance_score', 0) * 100
            emotion_f1 = metrics.get('non_neutral_f1', 0) * 100
            
            model_title = model_name.replace('_', ' ').title()
            
            # Main recommendation
            if performance >= 80 and balance_score >= 70:
                recommendations.append(f"🟢 **{model_title}**: Excellent! Ready for production use.")
            elif performance >= 70 and balance_score >= 60:
                recommendations.append(f"🟡 **{model_title}**: Good performance, suitable for most applications.")
            elif emotion_f1 >= 50:
                recommendations.append(f"🟠 **{model_title}**: Fair emotion detection but may have neutral bias.")
            else:
                recommendations.append(f"🔴 **{model_title}**: Needs improvement in emotion detection.")
            
            # Specific guidance
            if balance_score < 50:
                recommendations.append(f"   ⚠️ **{model_title}**: Strong neutral bias detected. Consider rebalancing training data.")
            
            if emotion_f1 < 40:
                recommendations.append(f"   ⚠️ **{model_title}**: Poor emotion detection. May predict neutral too often.")
            
            if performance > 90:
                recommendations.append(f"   🎉 **{model_title}**: Outstanding performance! This model excels at emotion detection.")
        
        return recommendations