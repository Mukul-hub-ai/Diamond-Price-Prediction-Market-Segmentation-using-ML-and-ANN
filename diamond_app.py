import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Diamond Price & Segment Predictor",
    page_icon="💎",
    layout="wide"
)

# App title
st.title("💎 Diamond Analytics Dashboard")
st.markdown("Predict diamond prices and market segments using machine learning models")

# Load models
@st.cache_resource
def load_models():
    models = {}
    
    try:
        # Load regression model
        with open('best_regression_model.pkl', 'rb') as f:
            models['regression'] = pickle.load(f)
        st.success("✅ Regression model loaded successfully!")
    except:
        st.error("❌ Could not load regression model")
        models['regression'] = None
    
    try:
        # Load clustering model
        with open('clustering_model.pkl', 'rb') as f:
            models['clustering'] = pickle.load(f)
        st.success("✅ Clustering model loaded successfully!")
    except:
        st.error("❌ Could not load clustering model")
        models['clustering'] = None
    
    return models

# ANN Model class (same as training)
class SimpleANN(nn.Module):
    def __init__(self, input_size):
        super(SimpleANN, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 16)
        self.output = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.output(x)
        return x

# Mapping dictionaries (same as training)
CUT_MAPPING = {'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4}
COLOR_MAPPING = {'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6}
CLARITY_MAPPING = {'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4, 'VVS2': 5, 'VVS1': 6, 'IF': 7}

# Cluster names
CLUSTER_NAMES = {
    0: "Mid-Range Balanced Diamonds",
    1: "Premium Large Diamonds",
    2: "Standard Quality Diamonds",
    3: "Luxury Large Diamonds",
    4: "Small Budget Diamonds"
}

# Cluster descriptions
CLUSTER_DESCRIPTIONS = {
    0: "Medium-sized diamonds with good value for money. Balanced features make them popular for everyday jewelry.",
    1: "Large diamonds with premium quality. Best sellers for engagement rings and special occasions.",
    2: "Standard quality diamonds suitable for regular market. Reliable choice for gifts and accessories.",
    3: "Luxury large diamonds with exceptional quality. Targeted at high-end market and investors.",
    4: "Small diamonds with high price per carat. Ideal for budget-conscious buyers and accent stones."
}

# Calculate derived features function
def calculate_derived_features(carat, x, y, z, cut_value):
    """
    Calculate all derived features needed for models
    """
    # Basic calculations
    volume = x * y * z
    dimension_ratio = x / y if y > 0 else 1
    density = carat / volume if volume > 0 else 0
    table_depth_ratio = x / z if z > 0 else 1
    size_index = (x + y + z) / 3
    
    # Premium cut flag (Ideal and Premium cuts are considered premium)
    premium_cut_flag = 1 if cut_value >= 3 else 0  # Premium=3, Ideal=4
    
    return {
        'volume': volume,
        'dimension_ratio': dimension_ratio,
        'density': density,
        'table_depth_ratio': table_depth_ratio,
        'size_index': size_index,
        'premium_cut_flag': premium_cut_flag
    }

# Create input dataframe for regression
def create_regression_input(carat, x, y, z, cut_encoded, color_encoded, clarity_encoded, derived_features):
    """
    Create input dataframe in exact same format as training data
    """
    input_data = {
        'carat': carat,
        'x': x,
        'y': y,
        'z': z,
        'volume': derived_features['volume'],
        'price_per_carat': 0,  # Will be calculated after price prediction
        'dimension_ratio': derived_features['dimension_ratio'],
        'density': derived_features['density'],
        'table_depth_ratio': derived_features['table_depth_ratio'],
        'size_index': derived_features['size_index'],
        'premium_cut_flag': derived_features['premium_cut_flag'],
        'cut_encoded': cut_encoded,
        'color_encoded': color_encoded,
        'clarity_encoded': clarity_encoded
    }
    
    return pd.DataFrame([input_data])

# Create input dataframe for clustering
def create_clustering_input(carat, x, y, z, cut_encoded, color_encoded, clarity_encoded, 
                           derived_features, predicted_price):
    """
    Create input dataframe for clustering model
    """
    # Calculate price per carat
    price_per_carat = predicted_price / carat if carat > 0 else 0
    
    input_data = {
        'carat': carat,
        'price': np.log(predicted_price),  # Log price as in training
        'x': x,
        'y': y,
        'z': z,
        'volume': derived_features['volume'],
        'price_per_carat': price_per_carat,
        'dimension_ratio': derived_features['dimension_ratio'],
        'density': derived_features['density'],
        'table_depth_ratio': derived_features['table_depth_ratio'],
        'size_index': derived_features['size_index'],
        'premium_cut_flag': derived_features['premium_cut_flag'],
        'cut_encoded': cut_encoded,
        'color_encoded': color_encoded,
        'clarity_encoded': clarity_encoded
    }
    
    return pd.DataFrame([input_data])

# Main app
def main():
    # Sidebar
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Choose Module",
        ["🏠 Home", "💰 Price Prediction", "🏷️ Market Segmentation"]
    )
    
    # Load models
    models = load_models()
    
    if app_mode == "🏠 Home":
        st.header("Welcome to Diamond Analytics Dashboard")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("💰 **Price Prediction Module**")
            st.markdown("""
            - Predict diamond price using trained ML model
            - Input diamond characteristics
            - Get price in both USD and INR
            - Uses best performing model from training
            """)
            
        with col2:
            st.success("🏷️ **Market Segmentation Module**")
            st.markdown("""
            - Predict market segment/cluster
            - Uses KMeans clustering model
            - Get business-friendly segment name
            - Understand target market for your diamond
            """)
        
        st.markdown("---")
        st.subheader("📊 Market Segments Overview")
        
        # Display cluster information
        clusters_df = pd.DataFrame({
            'Segment': list(CLUSTER_NAMES.values()),
            'Description': list(CLUSTER_DESCRIPTIONS.values()),
            'Avg Carat': [0.49, 0.92, 0.72, 1.10, 0.30],
            'Target Market': [
                'Middle-class, Everyday jewelry',
                'Engagement rings, Special occasions',
                'Regular gifts, Accessories',
                'Luxury buyers, Investors',
                'Budget buyers, Accent stones'
            ]
        })
        
        st.dataframe(clusters_df, use_container_width=True)
        
    elif app_mode == "💰 Price Prediction":
        st.header("💰 Diamond Price Prediction")
        
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.subheader("📏 Physical Dimensions")
                carat = st.number_input("Carat Weight", 
                                       min_value=0.1, 
                                       max_value=10.0, 
                                       value=0.7, 
                                       step=0.01,
                                       help="Weight of the diamond in carats")
                
                x = st.number_input("Length (x in mm)", 
                                   min_value=0.1, 
                                   max_value=20.0, 
                                   value=5.7, 
                                   step=0.1,
                                   help="Length dimension in millimeters")
                
                y = st.number_input("Width (y in mm)", 
                                   min_value=0.1, 
                                   max_value=20.0, 
                                   value=5.7, 
                                   step=0.1,
                                   help="Width dimension in millimeters")
                
                z = st.number_input("Depth (z in mm)", 
                                   min_value=0.1, 
                                   max_value=20.0, 
                                   value=3.5, 
                                   step=0.1,
                                   help="Depth dimension in millimeters")
            
            with col2:
                st.subheader("💎 Quality Grades")
                cut = st.selectbox("Cut Quality", 
                                  options=list(CUT_MAPPING.keys()),
                                  index=4,  # Default to Ideal
                                  help="Quality of the diamond's cut")
                
                color = st.selectbox("Color Grade", 
                                    options=list(COLOR_MAPPING.keys()),
                                    index=6,  # Default to D (best)
                                    help="Diamond color grade (D is best, J is worst)")
                
                clarity = st.selectbox("Clarity Grade", 
                                      options=list(CLARITY_MAPPING.keys()),
                                      index=7,  # Default to IF (best)
                                      help="Diamond clarity grade")
            
            with col3:
                st.subheader("ℹ️ Information")
                st.markdown("""
                **Cut Quality:**
                - Fair: 0, Good: 1, Very Good: 2
                - Premium: 3, Ideal: 4
                
                **Color Grade:**
                - J (worst) to D (best)
                
                **Clarity Grade:**
                - I1 (worst) to IF (best)
                """)
            
            submit_button = st.form_submit_button("🚀 Predict Price", type="primary")
            
            if submit_button and models['regression']:
                try:
                    # Get encoded values
                    cut_encoded = CUT_MAPPING[cut]
                    color_encoded = COLOR_MAPPING[color]
                    clarity_encoded = CLARITY_MAPPING[clarity]
                    
                    # Calculate derived features
                    derived = calculate_derived_features(carat, x, y, z, cut_encoded)
                    
                    # Create input dataframe
                    input_df = create_regression_input(carat, x, y, z, cut_encoded, 
                                                      color_encoded, clarity_encoded, derived)
                    
                    # Make prediction
                    predicted_log_price = models['regression'].predict(input_df)[0]
                    
                    # Convert from log to actual price
                    predicted_price_usd = np.exp(predicted_log_price)
                    
                    # Calculate price per carat
                    price_per_carat = predicted_price_usd / carat if carat > 0 else 0
                    
                    # Update input_df with actual price for clustering
                    input_df['price_per_carat'] = price_per_carat
                    
                    # Store in session state
                    st.session_state['predicted_price_usd'] = predicted_price_usd
                    st.session_state['input_features'] = {
                        'carat': carat,
                        'x': x,
                        'y': y,
                        'z': z,
                        'cut_encoded': cut_encoded,
                        'color_encoded': color_encoded,
                        'clarity_encoded': clarity_encoded,
                        'derived': derived,
                        'price_per_carat': price_per_carat
                    }
                    
                    # Display results
                    st.success("✅ Prediction Complete!")
                    
                    # Convert to INR
                    predicted_price_inr = predicted_price_usd * 83
                    
                    # Display metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Predicted Price (USD)", f"${predicted_price_usd:,.2f}")
                    
                    with col2:
                        st.metric("Predicted Price (INR)", f"₹{predicted_price_inr:,.2f}")
                    
                    with col3:
                        st.metric("Price per Carat (USD)", f"${price_per_carat:,.2f}")
                    
                    with col4:
                        st.metric("Carat Weight", f"{carat} ct")
                    
                    # Show input summary
                    with st.expander("📋 View Input Summary"):
                        summary_df = pd.DataFrame({
                            'Feature': ['Carat', 'Length (x)', 'Width (y)', 'Depth (z)', 
                                       'Cut', 'Color', 'Clarity', 'Volume', 'Size Index'],
                            'Value': [f"{carat} ct", f"{x} mm", f"{y} mm", f"{z} mm",
                                     cut, color, clarity, 
                                     f"{derived['volume']:.2f} mm³", 
                                     f"{derived['size_index']:.2f}"]
                        })
                        st.table(summary_df)
                    
                except Exception as e:
                    st.error(f"❌ Prediction error: {str(e)}")
                    st.info("Please check your inputs and try again.")
            
            elif submit_button and not models['regression']:
                st.error("Regression model not available. Please check if model file exists.")
    
    elif app_mode == "🏷️ Market Segmentation":
        st.header("🏷️ Market Segment Prediction")
        
        if 'predicted_price_usd' not in st.session_state:
            st.warning("⚠️ Please make a price prediction first using the 'Price Prediction' module.")
            st.info("Go to the Price Prediction module, enter diamond details, and click 'Predict Price'.")
        elif not models['clustering']:
            st.error("Clustering model not available. Please check if model file exists.")
        else:
            st.success("✅ Using previously predicted diamond for segmentation")
            
            # Get stored data
            predicted_price = st.session_state['predicted_price_usd']
            features = st.session_state['input_features']
            
            # Display current diamond info
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Predicted Price", f"${predicted_price:,.2f}")
                st.metric("Carat Weight", f"{features['carat']} ct")
            
            with col2:
                st.metric("Price in INR", f"₹{predicted_price * 83:,.2f}")
                st.metric("Price per Carat", f"${features['price_per_carat']:,.2f}")
            
            if st.button("🔍 Predict Market Segment", type="primary"):
                try:
                    # Create clustering input
                    clustering_input = create_clustering_input(
                        carat=features['carat'],
                        x=features['x'],
                        y=features['y'],
                        z=features['z'],
                        cut_encoded=features['cut_encoded'],
                        color_encoded=features['color_encoded'],
                        clarity_encoded=features['clarity_encoded'],
                        derived_features=features['derived'],
                        predicted_price=predicted_price
                    )
                    
                    # Predict cluster
                    cluster_label = models['clustering'].predict(clustering_input)[0]
                    cluster_name = CLUSTER_NAMES.get(cluster_label, f"Cluster {cluster_label}")
                    cluster_desc = CLUSTER_DESCRIPTIONS.get(cluster_label, "")
                    
                    # Display results
                    st.success("✅ Segment Prediction Complete!")
                    
                    # Create nice display
                    st.markdown(f"""
                    <div style='background-color: #f0f2f6; padding: 20px; border-radius: 10px;'>
                        <h3>🎯 Predicted Market Segment</h3>
                        <h2 style='color: #ff4b4b;'>{cluster_name}</h2>
                        <p><strong>Segment Number:</strong> {cluster_label}</p>
                        <p><strong>Description:</strong> {cluster_desc}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show cluster characteristics
                    st.subheader("📈 Segment Characteristics")
                    
                    # Cluster statistics (from your groupby output)
                    cluster_stats = {
                        0: {'carat': 0.49, 'price_log': 7.65, 'color': 3.73, 'clarity': 3.02},
                        1: {'carat': 0.92, 'price_log': 9.13, 'color': 2.60, 'clarity': 2.61},
                        2: {'carat': 0.72, 'price_log': 8.56, 'color': 3.23, 'clarity': 2.48},
                        3: {'carat': 1.10, 'price_log': 9.35, 'color': 1.85, 'clarity': 1.69},
                        4: {'carat': 0.30, 'price_log': 6.64, 'color': 3.73, 'clarity': 3.85}
                    }
                    
                    if cluster_label in cluster_stats:
                        avg_stats = cluster_stats[cluster_label]
                        
                        comparison_data = {
                            'Metric': ['Carat Weight', 'Log Price', 'Color Grade', 'Clarity Grade'],
                            'Your Diamond': [
                                f"{features['carat']:.2f}",
                                f"{np.log(predicted_price):.2f}",
                                features['color_encoded'],
                                features['clarity_encoded']
                            ],
                            'Segment Average': [
                                f"{avg_stats['carat']:.2f}",
                                f"{avg_stats['price_log']:.2f}",
                                f"{avg_stats['color']:.1f}",
                                f"{avg_stats['clarity']:.1f}"
                            ]
                        }
                        
                        comparison_df = pd.DataFrame(comparison_data)
                        st.dataframe(comparison_df, use_container_width=True)
                        
                        # Business recommendations
                        st.subheader("💡 Business Recommendations")
                        
                        recommendations = {
                            0: "Market as value-for-money diamonds. Suitable for middle-class buyers and anniversary gifts.",
                            1: "Promote as engagement ring diamonds. Target social media advertising for special occasions.",
                            2: "Bundle with jewelry sets. Offer festival discounts and promotions.",
                            3: "Highlight as investment pieces. Provide certification and luxury packaging.",
                            4: "Market as accent stones. Create budget-friendly jewelry collections."
                        }
                        
                        st.info(recommendations.get(cluster_label, "No specific recommendations available."))
                    
                except Exception as e:
                    st.error(f"❌ Segmentation error: {str(e)}")
        
        # Show all clusters info
        st.markdown("---")
        st.subheader("📊 All Market Segments")
        
        for cluster_num, cluster_name in CLUSTER_NAMES.items():
            with st.expander(f"{cluster_name} (Cluster {cluster_num})"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Description:** {CLUSTER_DESCRIPTIONS[cluster_num]}")
                with col2:
                    st.write(f"**Target Audience:** {[
                        'Middle-class buyers',
                        'Engagement ring shoppers',
                        'Regular jewelry buyers',
                        'Luxury investors',
                        'Budget-conscious buyers'
                    ][cluster_num]}")

    # Footer
    st.markdown("---")
    st.caption("💎 Diamond Analytics Dashboard | Price Prediction & Market Segmentation")

if __name__ == "__main__":
    # Initialize session state
    if 'predicted_price_usd' not in st.session_state:
        st.session_state['predicted_price_usd'] = None
    if 'input_features' not in st.session_state:
        st.session_state['input_features'] = None
    
    main()