import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("Mall_Customers.csv")

def main():
    st.title("🛍️ Mall Customer Segmentation Analysis")
    st.write("Interactive K-Means Clustering for Customer Groups")

    # Load and preprocess data
    df = load_data()
    df_clean = df.drop("CustomerID", axis=1)
    df_clean["Gender"] = df_clean["Gender"].map({"Male": 0, "Female": 1})

    # Sidebar controls
    st.sidebar.header("Clustering Parameters")
    features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    x_feat = st.sidebar.selectbox("X-axis Feature", features, index=1)
    y_feat = st.sidebar.selectbox("Y-axis Feature", features, index=2)
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 8, 5)

    # Prepare data for clustering
    X = df_clean[[x_feat, y_feat]]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_clean["Cluster"] = kmeans.fit_predict(X_scaled)

    # Main display
    st.header("Cluster Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df_clean,
        x=x_feat,
        y=y_feat,
        hue="Cluster",
        palette="viridis",
        s=100,
        ax=ax
    )
    plt.title(f"Customer Clusters ({n_clusters} groups)")
    st.pyplot(fig)

    # Cluster analysis
    st.header("Cluster Characteristics")
    cluster_summary = df_clean.groupby("Cluster").agg({
        "Age": "mean",
        "Annual Income (k$)": "mean",
        "Spending Score (1-100)": "mean",
        "Gender": lambda x: (x.mean() * 100).round(1)  # % Female
    }).rename(columns={"Gender": "Female %"})

    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Mean Values per Cluster")
        st.dataframe(cluster_summary.style.background_gradient(cmap="Blues"))
    
    with col2:
        st.write("### Cluster Distribution")
        cluster_dist = df_clean["Cluster"].value_counts().sort_index()
        st.bar_chart(cluster_dist)
        
    # Raw data explorer
    st.header("Data Explorer")
    st.write("Filter clusters to inspect individual customers:")
    selected_clusters = st.multiselect(
        "Select Clusters to View",
        options=df_clean["Cluster"].unique(),
        default=df_clean["Cluster"].unique()[0]
    )
    st.dataframe(df_clean[df_clean["Cluster"].isin(selected_clusters)])

if __name__ == "__main__":
    main()