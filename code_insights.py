"""
CODE INSIGHTS: HOW SWIMMING ANALYSIS WORKS
===========================================
This section shows the most important code snippets that make
the swimming stroke recognition system work.
"""

import streamlit as st


class CodeInsightsSection:
    """
    Educational section showing critical code snippets from the swimming app.
    Each snippet explains key concepts in swimming analytics.
    """

    def __init__(self):
        self.snippets = self._create_snippets()

    def _create_snippets(self):
        """Create all code snippets with explanations."""
        return {
            'feature_extraction': [
                {
                    'title': "📊 Calculating Stroke Frequency",
                    'code': '''def calculate_stroke_frequency(acceleration_data):
    # Find peaks in acceleration = each arm stroke
    peaks, _ = find_peaks(acceleration_data,
                         height=2.0,
                         distance=20)

    if len(peaks) >= 2:
        # Time between first and last peak
        total_time = peaks[-1] - peaks[0]
        # Strokes per second
        frequency = len(peaks) / total_time
    else:
        frequency = 0

    return frequency  # e.g., 1.8 Hz''',
                    'explanation': {
                        'what': "Detects stroke rate by finding acceleration peaks",
                        'swimming': "How many arm pulls per second (elite: 1.8-2.0 Hz)",
                        'importance': "Most important feature for distinguishing strokes"
                    },
                    'output_example': "1.8 Hz (108 strokes per minute)"
                },
                {
                    'title': "🔄 Detecting Body Roll Asymmetry",
                    'code': '''def calculate_roll_asymmetry(gyro_data):
    # Separate left and right rotation
    left_roll = gyro_data[gyro_data < 0]   # Negative = left rotation
    right_roll = gyro_data[gyro_data > 0]  # Positive = right rotation

    if len(left_roll) > 0 and len(right_roll) > 0:
        # Compare average rotation power
        left_power = np.mean(np.abs(left_roll))
        right_power = np.mean(np.abs(right_roll))

        # Asymmetry score (0 = perfect balance)
        asymmetry = abs(left_power - right_power) / max(left_power, right_power)
    else:
        asymmetry = 0

    return asymmetry''',
                    'explanation': {
                        'what': "Compares left vs right body rotation",
                        'swimming': "Detects uneven rotation (common swimming flaw)",
                        'importance': "Improving symmetry increases efficiency 15-20%"
                    },
                    'output_example': "0.25 (25% imbalance - needs work)"
                }
            ],
            'knn_algorithm': [
                {
                    'title': "🎯 KNN Feature Selection",
                    'code': '''# KNN uses ONLY motion features (8 total)
feature_names = [
    "acc_mean",    # Overall movement intensity
    "acc_rms",     # Root mean square acceleration
    "acc_std",     # Movement variability
    "acc_p2p",     # Stroke power
    "gyro_mean",   # Rotation intensity
    "gyro_rms",    # RMS rotation
    "gyro_std",    # Rotation variability
    "gyro_p2p"     # Rotation range
]

# Select only these 8 features for KNN
X_knn = features_df[feature_names].copy()''',
                    'explanation': {
                        'what': "KNN works best with simple motion metrics",
                        'swimming': "Compares 'how much' you move, not 'how' you move",
                        'importance': "Too many features confuse KNN's distance calculations"
                    },
                    'output_example': "Uses 8 motion features only"
                },
                {
                    'title': "⚖️ KNN Data Scaling (Critical!)",
                    'code': '''# KNN REQUIRES feature scaling!
from sklearn.preprocessing import StandardScaler

# Initialize scaler
scaler = StandardScaler()

# Scale training data
X_train_scaled = scaler.fit_transform(X_train)

# Scale test data using SAME transformation
X_test_scaled = scaler.transform(X_test)

# Why? stroke_frequency (1.5) vs acc_mean (12.5)
# have different units without scaling!''',
                    'explanation': {
                        'what': "Normalizes all features to same scale",
                        'swimming': "Makes stroke rate comparable to acceleration",
                        'importance': "Without scaling, KNN accuracy drops 20-30%"
                    },
                    'output_example': "Before scaling: 65% accuracy, After: 85% accuracy"
                }
            ],
            'random_forest': [
                {
                    'title': "🌳 RF Uses All Features",
                    'code': '''# Random Forest uses ALL 16 features
feature_names = [
    # Motion Features (8)
    "acc_mean", "acc_rms", "acc_std", "acc_p2p",
    "gyro_mean", "gyro_rms", "gyro_std", "gyro_p2p",

    # Shape Features (8) - KNN can't use these!
    "stroke_frequency",      # Stroke rate
    "roll_asymmetry",        # Rotation balance
    "pitch_peak_count",      # Stroke smoothness
    "rhythm_consistency",    # Timing consistency
    "pitch_kurtosis",        # Pitch distribution
    "pitch_skewness",        # Pitch asymmetry
    "gyro_kurtosis",         # Rotation distribution
    "gyro_skewness"          # Rotation asymmetry
]

# RF can handle all features without issues
X_rf = features_df[feature_names].copy()''',
                    'explanation': {
                        'what': "RF works with complete swimming profile",
                        'swimming': "Analyzes both movement AND technique patterns",
                        'importance': "Captures complex swimming biomechanics"
                    },
                    'output_example': "Uses all 16 motion + shape features"
                },
                {
                    'title': "🏆 RF Feature Importance",
                    'code': '''# After training, RF shows what matters most
feature_importance = rf_model.feature_importances_

# Results show which features are most useful
importance_dict = {
    "stroke_frequency": 0.28,    # Most important!
    "roll_asymmetry": 0.19,      # Rotation balance
    "acc_p2p": 0.15,            # Stroke power
    "rhythm_consistency": 0.12,  # Timing
    "gyro_mean": 0.08,          # Rotation intensity
    # ... other features
}

# Coach sees: "Focus on stroke frequency first!"''',
                    'explanation': {
                        'what': "Shows which features influence decisions most",
                        'swimming': "Tells coaches what to focus on in training",
                        'importance': "Turns black-box model into coaching tool"
                    },
                    'output_example': "stroke_frequency: 28% importance (train this first!)"
                }
            ],
            'comparison': [
                {
                    'title': "🤔 KNN vs RF: The Key Differences",
                    'code': '''# SIDE-BY-SIDE COMPARISON
# ==================== KNN ===================  =============== RF ================
# Features:     8 motion only                   # 16 motion + shape
# Scaling:     REQUIRED (StandardScaler)        # Not needed
# Training:    Fast (just stores data)          # Slow (builds 100+ trees)
# Prediction:  Slow (compares to all data)      # Fast (tree traversal)
# Output:      "Similar to swimmer X"           # "Improve feature Y"
# Best for:    Matching similar swimmers        # Technique analysis''',
                    'explanation': {
                        'what': "Different algorithms for different goals",
                        'swimming': "KNN: Find training partners, RF: Improve technique",
                        'importance': "Choose the right tool for your coaching needs"
                    },
                    'output_example': "KNN: 82% accuracy, RF: 91% accuracy"
                }
            ]
        }

    def display_snippet(self, category, snippet_idx):
        """Display a single code snippet with explanation."""
        snippet = self.snippets[category][snippet_idx]

        st.markdown(f"### {snippet['title']}")

        # Code display
        with st.expander("📝 View Code", expanded=True):
            st.code(snippet['code'], language='python')

        # Explanation
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**What this does:**\n{snippet['explanation']['what']}")
        with col2:
            st.success(f"**Swimming meaning:**\n{snippet['explanation']['swimming']}")
        with col3:
            st.warning(f"**Why it matters:**\n{snippet['explanation']['importance']}")

        # Output example
        if snippet['output_example']:
            st.markdown(f"**Example output:** `{snippet['output_example']}`")

        st.divider()

    def display_category(self, category_name, category_label):
        """Display all snippets in a category."""
        st.markdown(f"## {category_label}")

        for i in range(len(self.snippets[category_name])):
            self.display_snippet(category_name, i)

    def display_all(self):
        """Display the complete code insights section."""
        st.title("🔍 Code Insights: How Swimming Analysis Works")
        st.markdown("""
        This section shows the **most important code** that makes the swimming stroke
        recognition system work. Each snippet explains a key concept in swimming analytics.
        """)

        # Feature Extraction
        self.display_category(
            'feature_extraction',
            "📊 Feature Extraction: From Sensors to Swimming Metrics"
        )

        # KNN Algorithm
        self.display_category(
            'knn_algorithm',
            "🎯 K-Nearest Neighbors: Finding Similar Swimmers"
        )

        # Random Forest
        self.display_category(
            'random_forest',
            "🌳 Random Forest: Learning Swimming Patterns"
        )

        # Comparison
        self.display_category(
            'comparison',
            "🤔 Algorithm Comparison: When to Use Each"
        )

        # Summary
        st.markdown("---")
        st.markdown("""
        ### 🎓 Key Takeaways:

        1. **Feature Engineering is Everything**: Good features beat complex algorithms
        2. **Different Algorithms, Different Purposes**:
           - KNN: "Who swims like me?"
           - RF: "How can I improve?"
        3. **Data Quality Matters**: Clean, well-processed data = better results
        
        Keep exploring and happy swimming! 🏊‍♂️
        """)


def add_code_insights_section():
    """Add the code insights section to the app."""
    insights = CodeInsightsSection()
    insights.display_all()
