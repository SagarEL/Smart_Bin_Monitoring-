import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import folium
from folium import plugins
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import math
from itertools import permutations

# ThingSpeak Configuration
THINGSPEAK_CHANNEL_ID = "2972661"  # Replace with your ThingSpeak channel ID
THINGSPEAK_READ_API_KEY = "W1XZ25JKU231F3BK"  # Replace with your ThingSpeak read API key
THINGSPEAK_URL = f"https://api.thingspeak.com/channels/{THINGSPEAK_CHANNEL_ID}/feeds.json"
BIN_FULL_THRESHOLD = 85  # Percentage at which a bin is considered full

# Set page configuration
st.set_page_config(
    page_title="Smart Bin Monitoring System",
    page_icon="🗑️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App title and description
st.title("🗑️ Smart Bin Monitoring System")
st.markdown("Real-time monitoring of smart waste bins")

# Sidebar configuration
with st.sidebar:
    st.header("Settings")
    refresh_interval = st.slider("Data refresh interval (seconds)", 5, 60, 15)
    full_threshold = st.slider("Bin full threshold (%)", 50, 100, BIN_FULL_THRESHOLD)
    st.divider()
    st.info("This application fetches real-time data from ThingSpeak and displays bin fill levels and locations.")
    
    # Add last updated timestamp
    last_update_placeholder = st.empty()

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["Dashboard", "Map View", "Historical Data"])

@st.cache_data(ttl=refresh_interval)
def fetch_thingspeak_data(results=100):
    """Fetch data from ThingSpeak API"""
    params = {
        "api_key": THINGSPEAK_READ_API_KEY,
        "results": results
    }
    
    try:
        response = requests.get(THINGSPEAK_URL, params=params)
        if response.status_code == 200:
            data = response.json()
            return data["feeds"], data["channel"]
        else:
            st.error(f"Error fetching data: {response.status_code}")
            return None, None
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None

def process_thingspeak_data(feeds, channel):
    """Process ThingSpeak data into a DataFrame"""
    if not feeds:
        return None
    
    df = pd.DataFrame(feeds)
    
    # Rename columns based on ThingSpeak field names
    field_names = {
        "field1": "bin_id",
        "field2": "fill_level", 
        "field3": "latitude",
        "field4": "longitude"
    }
    
    for field, name in field_names.items():
        if field in df.columns:
            df.rename(columns={field: name}, inplace=True)
    
    # Convert data types
    if "fill_level" in df.columns:
        df["fill_level"] = pd.to_numeric(df["fill_level"], errors="coerce")
    
    if "latitude" in df.columns and "longitude" in df.columns:
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    
    # Convert timestamp
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"])
    
    return df

def create_latest_data_view(df):
    """Create view for latest data from each bin"""
    if df is None or df.empty:
        st.warning("No data available")
        return
    
    # Get the latest reading for each bin
    latest_data = df.sort_values("created_at").groupby("bin_id").last().reset_index()
    
    # Count unique bins
    num_bins = len(latest_data)
    st.info(f"Monitoring {num_bins} unique bins")
    
    # Create a grid layout based on the number of bins
    cols_per_row = min(3, num_bins)  # Max 3 bins per row
    num_rows = (num_bins + cols_per_row - 1) // cols_per_row
    
    # Display bins in a grid
    for row in range(num_rows):
        cols = st.columns(cols_per_row)
        for col in range(cols_per_row):
            bin_idx = row * cols_per_row + col
            if bin_idx < num_bins:
                bin_data = latest_data.iloc[bin_idx]
                bin_id = bin_data["bin_id"]
                fill_level = bin_data["fill_level"]
                
                # Determine color based on fill level
                if fill_level >= full_threshold:
                    color = "red"
                    status = "FULL"
                    delta_type = "inverse"  # Red/negative (danger)
                elif fill_level >= 70:
                    color = "orange"
                    status = "HIGH"
                    delta_type = "inverse"  # Red/negative (warning)
                elif fill_level >= 40:
                    color = "blue"
                    status = "MEDIUM"
                    delta_type = "normal"   # Normal
                else:
                    color = "green"
                    status = "LOW"
                    delta_type = "normal"   # Green/positive (good)
                
                with cols[col]:
                    st.subheader(f"Bin {bin_id}")
                    
                    # Add metrics
                    st.metric(
                        label=f"Fill Level",
                        value=f"{fill_level:.1f}%",
                        delta=status,
                        delta_color=delta_type
                    )
                    
                    location_str = f"📍 {bin_data['latitude']:.6f}, {bin_data['longitude']:.6f}"
                    st.caption(location_str)
                    
                    # Create a gauge chart for fill level
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=fill_level,
                        title={"text": f"Bin {bin_id}"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": color},
                            "steps": [
                                {"range": [0, 40], "color": "lightgreen"},
                                {"range": [40, 70], "color": "lightyellow"},
                                {"range": [70, full_threshold], "color": "orange"},
                                {"range": [full_threshold, 100], "color": "lightcoral"}
                            ],
                            "threshold": {
                                "line": {"color": "red", "width": 4},
                                "thickness": 0.75,
                                "value": full_threshold
                            }
                        }
                    ))
                    
                    fig.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show last updated time
                    st.caption(f"Last updated: {bin_data['created_at'].strftime('%H:%M:%S')}")
                    
                    # Add quick action button for each bin
                    if fill_level >= full_threshold:
                        st.button(f"Send truck to Bin {bin_id}", key=f"quick_send_{bin_id}", type="primary")
                    elif fill_level >= 70:
                        st.button(f"Monitor Bin {bin_id}", key=f"quick_monitor_{bin_id}")
    
    # Add a summary table of all bins
    with st.expander("View All Bins Summary Table", expanded=False):
        st.dataframe(
            latest_data[["bin_id", "fill_level", "latitude", "longitude", "created_at"]],
            hide_index=True,
            column_config={
                "bin_id": "Bin ID",
                "fill_level": st.column_config.NumberColumn("Fill Level", format="%.1f%%"),
                "latitude": st.column_config.NumberColumn("Latitude", format="%.6f"),
                "longitude": st.column_config.NumberColumn("Longitude", format="%.6f"),
                "created_at": "Last Updated"
            }
        )

def create_line_chart(df):
    """Create line chart of fill levels over time"""
    if df is None or df.empty:
        return None
    
    # Count unique bins
    unique_bins = df["bin_id"].unique()
    
    if len(unique_bins) > 1:
        # Create tabs for individual bin charts and all bins
        chart_tabs = st.tabs(["All Bins"] + [f"Bin {bin_id}" for bin_id in unique_bins])
        
        # All bins tab
        with chart_tabs[0]:
            fig = px.line(
                df, 
                x="created_at", 
                y="fill_level", 
                color="bin_id",
                title="Fill Level History - All Bins",
                labels={"fill_level": "Fill Level (%)", "created_at": "Time", "bin_id": "Bin ID"}
            )
            
            # Add threshold line
            fig.add_shape(
                type="line",
                x0=df["created_at"].min(),
                x1=df["created_at"].max(),
                y0=full_threshold,
                y1=full_threshold,
                line=dict(color="red", width=2, dash="dash"),
            )
            
            fig.update_layout(height=400, legend_title_text="Bin ID")
            st.plotly_chart(fig, use_container_width=True)
        
        # Individual bin tabs
        for i, bin_id in enumerate(unique_bins):
            with chart_tabs[i+1]:
                bin_df = df[df["bin_id"] == bin_id]
                
                fig = px.line(
                    bin_df, 
                    x="created_at", 
                    y="fill_level",
                    title=f"Fill Level History - Bin {bin_id}",
                    labels={"fill_level": "Fill Level (%)", "created_at": "Time"}
                )
                
                # Add threshold line
                fig.add_shape(
                    type="line",
                    x0=bin_df["created_at"].min(),
                    x1=bin_df["created_at"].max(),
                    y0=full_threshold,
                    y1=full_threshold,
                    line=dict(color="red", width=2, dash="dash"),
                )
                
                # Customize the line for each bin
                fig.update_traces(line=dict(width=3))
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Add statistics for this bin
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_fill = bin_df["fill_level"].mean()
                    st.metric("Average Fill Level", f"{avg_fill:.1f}%")
                with col2:
                    max_fill = bin_df["fill_level"].max()
                    st.metric("Maximum Fill Level", f"{max_fill:.1f}%")
                with col3:
                    current_fill = bin_df.iloc[-1]["fill_level"] if not bin_df.empty else 0
                    st.metric("Current Fill Level", f"{current_fill:.1f}%")
    else:
        # Just one bin, show a simple chart
        fig = px.line(
            df, 
            x="created_at", 
            y="fill_level",
            title="Fill Level History",
            labels={"fill_level": "Fill Level (%)", "created_at": "Time"}
        )
        
        # Add threshold line
        fig.add_shape(
            type="line",
            x0=df["created_at"].min(),
            x1=df["created_at"].max(),
            y0=full_threshold,
            y1=full_threshold,
            line=dict(color="red", width=2, dash="dash"),
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    return fig

def create_map_view(df):
    """Create map view of bin locations"""
    if df is None or df.empty or "latitude" not in df.columns:
        st.warning("No location data available")
        return None
    
    # Get the latest reading for each bin
    latest_data = df.sort_values("created_at").groupby("bin_id").last().reset_index()
    
    # Remove rows with invalid coordinates
    map_data = latest_data.dropna(subset=["latitude", "longitude"])
    
    if map_data.empty:
        st.warning("No valid location data available")
        return None
    
    # Find center of the map
    center_lat = map_data["latitude"].mean()
    center_lon = map_data["longitude"].mean()
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=15)
    
    # Create marker clusters
    marker_cluster = plugins.MarkerCluster().add_to(m)
    
    # Add waypoints for bins over threshold
    waypoints = []
    
    # Create a dictionary to group bins by status for legend
    bin_status_counts = {
        "FULL": 0,
        "HIGH": 0,
        "MEDIUM": 0,
        "LOW": 0
    }
    
    # Add markers for each bin
    for _, bin_data in map_data.iterrows():
        bin_id = bin_data["bin_id"]
        fill_level = bin_data["fill_level"]
        lat = bin_data["latitude"]
        lon = bin_data["longitude"]
        
        # Determine color and status based on fill level
        if fill_level >= full_threshold:
            color = "red"
            icon = "trash"
            status = "FULL"
            # Add to waypoints if full
            waypoints.append((lat, lon))
        elif fill_level >= 70:
            color = "orange"
            icon = "trash"
            status = "HIGH"
        elif fill_level >= 40:
            color = "blue"
            icon = "trash"
            status = "MEDIUM"
        else:
            color = "green"
            icon = "trash"
            status = "LOW"
        
        # Update status count
        bin_status_counts[status] += 1
        
        # Create popup content
        popup_content = f"""
        <div style="width:200px">
            <h4>Bin {bin_id}</h4>
            <b>Fill Level:</b> {fill_level:.1f}%<br>
            <b>Status:</b> {status}<br>
            <b>Coordinates:</b> {lat:.6f}, {lon:.6f}<br>
            <b>Last Updated:</b> {bin_data['created_at'].strftime('%Y-%m-%d %H:%M:%S')}
        </div>
        """
        
        # Add marker to cluster
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_content, max_width=300),
            icon=folium.Icon(color=color, icon=icon, prefix='fa'),
            tooltip=f"Bin {bin_id}: {fill_level:.1f}%"
        ).add_to(marker_cluster)
    
    # Add a legend for bin status
    legend_html = f'''
    <div style="position: fixed; 
        bottom: 50px; right: 50px; width: 150px; 
        border:2px solid grey; z-index:9999; background-color:white;
        padding: 10px;
        font-size: 14px;
        opacity: 0.8;
    ">
        <p style="margin-top: 0;"><b>Bin Status</b></p>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="background-color: red; width: 20px; height: 20px; margin-right: 5px;"></div>
            <span>FULL ({bin_status_counts["FULL"]})</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="background-color: orange; width: 20px; height: 20px; margin-right: 5px;"></div>
            <span>HIGH ({bin_status_counts["HIGH"]})</span>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 5px;">
            <div style="background-color: blue; width: 20px; height: 20px; margin-right: 5px;"></div>
            <span>MEDIUM ({bin_status_counts["MEDIUM"]})</span>
        </div>
        <div style="display: flex; align-items: center;">
            <div style="background-color: green; width: 20px; height: 20px; margin-right: 5px;"></div>
            <span>LOW ({bin_status_counts["LOW"]})</span>
        </div>
    </div>
    '''
    
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add waypoint route if there are full bins
    if waypoints:
        # Add a start point (could be a service center or current location)
        start_point = waypoints[0] if waypoints else (center_lat, center_lon)  # Using first full bin as start for demo
        
        # Create a route through all waypoints
        waypoints_plugin = plugins.AntPath(
            locations=[start_point] + waypoints,
            color='red',
            weight=5,
            opacity=0.7,
            dash_array=[10, 20]
        )
        waypoints_plugin.add_to(m)
        
        # Add start point marker
        folium.Marker(
            location=start_point,
            icon=folium.Icon(color="green", icon="play", prefix='fa'),
            popup="Start Point"
        ).add_to(m)
    
    return m

def optimize_route(start_point, waypoints):
    """Find the optimal route using the best available algorithm based on the number of points"""
    if not waypoints:
        return [start_point]  # Just return the start point if no waypoints
    
    # If we have a small number of points, we can use brute force (optimal solution)
    if len(waypoints) <= 8:
        return optimize_route_brute_force(start_point, waypoints)
    else:
        # For larger sets, use a more efficient but less optimal algorithm
        return optimize_route_nearest_neighbor(start_point, waypoints)

def optimize_route_nearest_neighbor(start_point, waypoints):
    """Find a route using the nearest neighbor algorithm (greedy approach)
    
    This algorithm starts at the depot and repeatedly visits the nearest unvisited bin
    until all bins have been visited. It's fast but may not find the optimal route.
    """
    current_point = start_point
    route = [current_point]
    remaining_points = waypoints.copy()
    
    # Remove start point if it's in the waypoints
    if start_point in remaining_points:
        remaining_points.remove(start_point)
    
    # Iteratively find the nearest unvisited point
    while remaining_points:
        # Find the nearest point
        nearest_point = min(remaining_points, 
                           key=lambda point: calculate_distance(current_point, point))
        
        # Add to route and remove from remaining points
        route.append(nearest_point)
        remaining_points.remove(nearest_point)
        
        # Update current point
        current_point = nearest_point
    
    return route

def optimize_route_brute_force(start_point, waypoints):
    """Find the optimal route using brute force (all permutations)
    
    This algorithm evaluates all possible routes and selects the one with the minimum
    total distance. It's guaranteed to find the optimal route but is computationally
    expensive for more than ~10 points.
    """
    if not waypoints:
        return [start_point]
    
    # Remove start point if it's in the waypoints
    waypoints_without_start = waypoints.copy()
    if start_point in waypoints_without_start:
        waypoints_without_start.remove(start_point)
    
    # Try all permutations
    best_route = None
    best_distance = float('inf')
    
    for perm in permutations(waypoints_without_start):
        # Create full route including start point
        route = [start_point] + list(perm)
        
        # Calculate total distance
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += calculate_distance(route[i], route[i + 1])
        
        # Check if this is the best route
        if total_distance < best_distance:
            best_distance = total_distance
            best_route = route
    
    return best_route

def calculate_distance(point1, point2):
    """Calculate Haversine distance between two points in kilometers
    
    The Haversine formula determines the great-circle distance between two points
    on a sphere given their longitudes and latitudes.
    """
    # Earth radius in kilometers
    R = 6371.0
    
    lat1, lon1 = point1
    lat2, lon2 = point2
    
    # Convert to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance = R * c
    return distance

def create_route_map(df, depot_location=None, custom_threshold=None):
    """Create map with optimized route for bin collection"""
    if df is None or df.empty or "latitude" not in df.columns:
        st.warning("No location data available")
        return None
    
    # Use the provided threshold or default to global setting
    threshold = custom_threshold if custom_threshold is not None else full_threshold
    
    # Get the latest reading for each bin
    latest_data = df.sort_values("created_at").groupby("bin_id").last().reset_index()
    
    # Remove rows with invalid coordinates
    map_data = latest_data.dropna(subset=["latitude", "longitude"])
    
    if map_data.empty:
        st.warning("No valid location data available")
        return None
    
    # Find center of the map
    center_lat = map_data["latitude"].mean()
    center_lon = map_data["longitude"].mean()
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=15)
    
    # Add waypoints for bins over threshold
    waypoints = []
    bin_ids = []  # Keep track of bin IDs for waypoints
    waypoint_fill_levels = []  # Keep track of fill levels for waypoints
    
    # Add markers for each bin
    for _, bin_data in map_data.iterrows():
        bin_id = bin_data["bin_id"]
        fill_level = bin_data["fill_level"]
        lat = bin_data["latitude"]
        lon = bin_data["longitude"]
        
        # Determine color based on fill level
        if fill_level >= threshold:
            color = "red"
            icon = "trash"
            # Add to waypoints if full
            waypoints.append((lat, lon))
            bin_ids.append(bin_id)
            waypoint_fill_levels.append(fill_level)
        elif fill_level >= 70:
            color = "orange"
            icon = "trash"
        elif fill_level >= 40:
            color = "blue"
            icon = "trash"
        else:
            color = "green"
            icon = "trash"
        
        # Create popup content
        popup_content = f"""
        <div style="width:200px">
            <h4>Bin {bin_id}</h4>
            <b>Fill Level:</b> {fill_level:.1f}%<br>
            <b>Status:</b> {'FULL' if fill_level >= threshold else 'OK'}<br>
            <b>Coordinates:</b> {lat:.6f}, {lon:.6f}<br>
            <b>Last Updated:</b> {bin_data['created_at'].strftime('%Y-%m-%d %H:%M:%S')}
        </div>
        """
        
        # Add marker to the map (not in cluster for better visibility of the route)
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_content, max_width=300),
            icon=folium.Icon(color=color, icon=icon, prefix='fa'),
        ).add_to(m)
    
    # Use depot location if provided, otherwise use center of map
    if depot_location:
        start_point = depot_location
    else:
        start_point = (center_lat, center_lon)
    
    # Add depot marker with special styling
    folium.Marker(
        location=start_point,
        icon=folium.Icon(color="black", icon="home", prefix='fa'),
        popup="Depot / Starting Point"
    ).add_to(m)
    
    # Add route if there are bins above threshold
    if waypoints:
        # Optimize the route
        optimized_route = optimize_route(start_point, waypoints)
        
        # Create a full route including return to depot (for round trip)
        full_route = optimized_route + [start_point]
        
        # Draw route lines segment by segment with arrows
        for i in range(len(full_route) - 1):
            # Create segment with arrow
            route_segment = [full_route[i], full_route[i+1]]
            
            # Determine if this is a return segment
            is_return = i == len(full_route) - 2
            
            # Create line with different style for return trip
            folium.PolyLine(
                route_segment,
                color='blue' if not is_return else 'green',
                weight=4,
                opacity=0.8,
                dash_array=None if not is_return else [10, 6],
                tooltip=f"Return to Depot" if is_return else f"Segment {i+1}"
            ).add_to(m)
            
            # Add arrow marker mid-segment to show direction
            mid_lat = (route_segment[0][0] + route_segment[1][0]) / 2
            mid_lon = (route_segment[0][1] + route_segment[1][1]) / 2
            
            # Calculate bearing for arrow rotation
            y = math.sin(route_segment[1][1] - route_segment[0][1]) * math.cos(route_segment[1][0])
            x = math.cos(route_segment[0][0]) * math.sin(route_segment[1][0]) - math.sin(route_segment[0][0]) * math.cos(route_segment[1][0]) * math.cos(route_segment[1][1] - route_segment[0][1])
            bearing = math.degrees(math.atan2(y, x))
            
            # Only add arrows for longer segments (reduces clutter)
            segment_distance = calculate_distance(route_segment[0], route_segment[1])
            if segment_distance > 0.2:  # Only add arrow if segment is longer than 200m
                folium.RegularPolygonMarker(
                    location=[mid_lat, mid_lon],
                    number_of_sides=3,
                    radius=6,
                    rotation=bearing,
                    color='blue' if not is_return else 'green',
                    fill=True,
                    fill_color='blue' if not is_return else 'green',
                    fill_opacity=0.8
                ).add_to(m)
        
        # Add numbered markers for the route sequence (excluding the depot at the start)
        for i, point in enumerate(optimized_route[1:], 1):  # Skip depot (0)
            # Find the bin_id for this waypoint
            wp_index = waypoints.index(point) if point in waypoints else -1
            
            if wp_index >= 0:
                bin_id = bin_ids[wp_index]
                fill_level = waypoint_fill_levels[wp_index]
                
                # Create a visually distinct marker for each stop
                folium.Marker(
                    location=point,
                    icon=folium.DivIcon(
                        icon_size=(40, 40),
                        icon_anchor=(20, 20),
                        html=f'''
                        <div style="
                            background-color: white; 
                            border-radius: 50%; 
                            width: 40px; 
                            height: 40px; 
                            display: flex; 
                            justify-content: center; 
                            align-items: center; 
                            font-weight: bold;
                            font-size: 14px;
                            border: 3px solid blue;
                            color: blue;
                        ">{i}</div>
                        '''
                    ),
                    popup=f"""
                    <div style="width:200px">
                        <h4>Stop #{i}</h4>
                        <b>Bin ID:</b> {bin_id}<br>
                        <b>Fill Level:</b> {fill_level:.1f}%<br>
                        <b>Coordinates:</b> {point[0]:.6f}, {point[1]:.6f}<br>
                    </div>
                    """
                ).add_to(m)
    
        # Add distance information
        total_distance = 0
        for i in range(len(optimized_route) - 1):
            total_distance += calculate_distance(optimized_route[i], optimized_route[i+1])
        
        # Add return distance
        return_distance = calculate_distance(optimized_route[-1], optimized_route[0])
        
        # Add a legend for the route
        legend_html = f'''
        <div style="position: fixed; 
            bottom: 50px; left: 50px; width: 220px; height: 140px; 
            border:2px solid grey; z-index:9999; background-color:white;
            padding: 10px;
            font-size: 14px;
            opacity: 0.8;
        ">
            <p style="margin-top: 0; margin-bottom: 5px;"><b>Route Information</b></p>
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <div style="background-color: blue; width: 20px; height: 4px; margin-right: 10px;"></div>
                <span>Collection Route ({total_distance:.2f} km)</span>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <div style="background-color: green; width: 20px; height: 4px; margin-right: 10px; border-style: dashed;"></div>
                <span>Return to Depot ({return_distance:.2f} km)</span>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 5px;">
                <div style="
                    background-color: white; 
                    border-radius: 50%; 
                    width: 20px; 
                    height: 20px; 
                    margin-right: 10px;
                    border: 2px solid blue;
                    display: flex; 
                    justify-content: center; 
                    align-items: center; 
                    font-size: 10px;
                    color: blue;
                ">1</div>
                <span>Stop Number</span>
            </div>
            <div style="margin-top: 5px;">
                <b>Total:</b> {total_distance + return_distance:.2f} km
            </div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
    
    return m

def main():
    try:
        # Fetch data from ThingSpeak
        feeds, channel = fetch_thingspeak_data()
        df = process_thingspeak_data(feeds, channel)
        
        # Update last updated time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        last_update_placeholder.text(f"Last updated: {current_time}")
        
        # Handle empty data case
        if df is None or df.empty:
            st.warning("⚠️ No data available from ThingSpeak channel. Please check your connection and channel settings.")
            return
        
        # Count unique bins for dashboard metrics
        unique_bins = df["bin_id"].unique()
        num_bins = len(unique_bins)
        
        # Add summary metrics at the top
        metric_cols = st.columns(4)
        
        with metric_cols[0]:
            st.metric("Total Bins", num_bins)
        
        with metric_cols[1]:
            # Get latest fill levels for each bin
            latest_data = df.sort_values("created_at").groupby("bin_id").last().reset_index()
            full_bins = latest_data[latest_data["fill_level"] >= full_threshold]
            st.metric("Bins Requiring Collection", len(full_bins), delta=f"{len(full_bins)}/{num_bins}")
        
        with metric_cols[2]:
            avg_fill = latest_data["fill_level"].mean()
            st.metric("Average Fill Level", f"{avg_fill:.1f}%")
        
        with metric_cols[3]:
            max_fill = latest_data["fill_level"].max()
            st.metric("Highest Fill Level", f"{max_fill:.1f}%")
            
        # Continue with tab views
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
      # Dashboard tab
    with tab1:
        st.subheader("Current Fill Levels")
        if df is not None and not df.empty:
            create_latest_data_view(df)
            
            st.subheader("Fill Level History")
            line_chart = create_line_chart(df)
            # if line_chart:
            #     st.plotly_chart(line_chart, use_container_width=True)
        else:
            st.warning("No data available from ThingSpeak")
      # Map view tab
    with tab2:
        st.subheader("Bin Locations")
        
        # Add a container for the "Send Trucks" button and route options
        route_options_container = st.container()
        
        if df is not None and not df.empty:
            # Get latest data for each bin
            latest_data = df.sort_values("created_at").groupby("bin_id").last().reset_index()
            
            # Find full bins
            full_bins = latest_data[latest_data["fill_level"] >= full_threshold].copy()
            
            # Show standard map view first
            map_view = create_map_view(df)
            map_container = st.container()
            
            if map_view:
                with map_container:
                    folium_static(map_view, width=1000)
                
                # Display full bins table
                if not full_bins.empty:
                    st.subheader("Bins Requiring Collection")
                    st.dataframe(
                        full_bins[["bin_id", "fill_level", "latitude", "longitude", "created_at"]],
                        hide_index=True,
                        column_config={
                            "bin_id": "Bin ID",
                            "fill_level": st.column_config.NumberColumn("Fill Level", format="%.1f%%"),
                            "latitude": st.column_config.NumberColumn("Latitude", format="%.6f"),
                            "longitude": st.column_config.NumberColumn("Longitude", format="%.6f"),
                            "created_at": "Last Updated"
                        }
                    )
                    
                    # Add "Send Trucks" feature in the options container
                    with route_options_container:
                        st.divider()
                        st.subheader("🚚 Route Optimization")
                        
                        # Define columns for route options
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Option to set custom threshold
                            custom_threshold = st.slider(
                                "Include bins with fill level above (%)", 
                                min_value=50, 
                                max_value=100, 
                                value=full_threshold
                            )
                            
                            # Get depot coordinates (default to center of map)
                            center_lat = latest_data["latitude"].mean()
                            center_lon = latest_data["longitude"].mean()
                            
                            st.text("Depot/Starting Location:")
                            depot_lat = st.number_input(
                                "Latitude", 
                                value=center_lat,
                                format="%.6f",
                                step=0.000001
                            )
                            depot_lon = st.number_input(
                                "Longitude", 
                                value=center_lon,
                                format="%.6f",
                                step=0.000001
                            )
                            
                        with col2:
                            # Count bins above threshold
                            bins_above_threshold = len(latest_data[latest_data["fill_level"] >= custom_threshold])
                            
                            st.metric(
                                "Bins Above Threshold", 
                                bins_above_threshold,
                                delta=f"{bins_above_threshold - len(full_bins)} from default threshold",
                                delta_color="off"
                            )
                            
                            st.text("")  # Spacer
                            st.text("")  # Spacer
                            
                            # Add the Send Trucks button
                            send_trucks = st.button(
                                "🚚 Send Trucks",
                                type="primary",
                                use_container_width=True
                            )
                            
                            if bins_above_threshold == 0:
                                st.warning("No bins above the selected threshold.")
                                send_trucks = False
                        
                        if send_trucks:
                            # Show optimized route
                            with st.spinner("Calculating optimal route..."):
                                # Get the optimized route map
                                route_map = create_route_map(
                                    df,
                                    depot_location=(depot_lat, depot_lon),
                                    custom_threshold=custom_threshold
                                )
                                
                                if route_map:
                                    # Replace the normal map with the route map
                                    with map_container:
                                        st.subheader("📍 Optimized Collection Route")
                                        folium_static(route_map, width=1000)
                                    
                                    # Calculate route details
                                    route_bins = latest_data[latest_data["fill_level"] >= custom_threshold].copy()
                                    
                                    if not route_bins.empty:
                                        # Extract coordinates for route planning
                                        waypoints = [(row["latitude"], row["longitude"]) for _, row in route_bins.iterrows()]
                                        depot = (depot_lat, depot_lon)
                                        
                                        # Get optimized route
                                        optimized_route = optimize_route(depot, waypoints)
                                        
                                        # Calculate total distance
                                        total_distance = 0
                                        for i in range(len(optimized_route) - 1):
                                            total_distance += calculate_distance(optimized_route[i], optimized_route[i + 1])
                                        
                                        # Calculate return to depot distance
                                        return_distance = calculate_distance(optimized_route[-1], optimized_route[0])
                                        total_round_trip = total_distance + return_distance
                                        
                                        # Show route statistics
                                        st.subheader("📊 Route Statistics")
                                        
                                        # Display route stats in columns
                                        stats_cols = st.columns(3)
                                        with stats_cols[0]:
                                            st.metric("Number of Stops", len(waypoints))
                                        with stats_cols[1]:
                                            st.metric("Route Distance", f"{total_distance:.2f} km")
                                        with stats_cols[2]:
                                            st.metric("Round Trip Distance", f"{total_round_trip:.2f} km")
                                        
                                        # Create route sequence table
                                        st.subheader("📋 Collection Sequence")
                                        
                                        # Generate route sequence data
                                        route_data = []
                                        
                                        # Add depot as start
                                        route_data.append({
                                            "Stop": 0,
                                            "Location": "Depot",
                                            "Bin ID": "N/A",
                                            "Fill Level": "N/A",
                                            "Coordinates": f"{depot[0]:.6f}, {depot[1]:.6f}"
                                        })
                                        
                                        # Add each stop in sequence
                                        for i, point in enumerate(optimized_route[1:], 1):
                                            # Find matching bin in route_bins
                                            bin_match = route_bins[
                                                (abs(route_bins["latitude"] - point[0]) < 0.0001) & 
                                                (abs(route_bins["longitude"] - point[1]) < 0.0001)
                                            ]
                                            
                                            if not bin_match.empty:
                                                row = bin_match.iloc[0]
                                                route_data.append({
                                                    "Stop": i,
                                                    "Location": f"Bin {row['bin_id']}",
                                                    "Bin ID": row["bin_id"],
                                                    "Fill Level": f"{row['fill_level']:.1f}%",
                                                    "Coordinates": f"{point[0]:.6f}, {point[1]:.6f}"
                                                })
                                        
                                        # Add depot as end (for round trip)
                                        route_data.append({
                                            "Stop": len(optimized_route),
                                            "Location": "Return to Depot",
                                            "Bin ID": "N/A",
                                            "Fill Level": "N/A",
                                            "Coordinates": f"{depot[0]:.6f}, {depot[1]:.6f}"
                                        })
                                        
                                        # Display route sequence table
                                        route_df = pd.DataFrame(route_data)
                                        st.dataframe(
                                            route_df,
                                            hide_index=True,
                                            column_config={
                                                "Stop": "Stop #",
                                                "Location": "Location",
                                                "Bin ID": "Bin ID",
                                                "Fill Level": "Fill Level",
                                                "Coordinates": "Coordinates"
                                            }
                                        )
                                else:
                                    st.error("Could not generate route map. Please try again.")
                else:
                    with route_options_container:
                        st.info("No bins are currently above the full threshold. No route optimization needed.")
            else:
                st.warning("Cannot create map view. Check that you have valid coordinate data.")
        else:
            st.warning("No location data available")
      # Historical data tab
    with tab3:
        st.subheader("Historical Data Analysis")
        
        if df is not None and not df.empty:
            # Date range selector
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start date",
                    value=datetime.now().date() - timedelta(days=7)
                )
            with col2:
                end_date = st.date_input(
                    "End date",
                    value=datetime.now().date()
                )
            
            # Filter data by date range
            mask = (df["created_at"].dt.date >= start_date) & (df["created_at"].dt.date <= end_date)
            filtered_df = df[mask]
            
            if not filtered_df.empty:
                # Group by day and bin_id
                daily_data = filtered_df.copy()
                daily_data["date"] = daily_data["created_at"].dt.date
                daily_avg = daily_data.groupby(["date", "bin_id"])["fill_level"].mean().reset_index()
                
                # Create heatmap
                pivot_data = daily_avg.pivot(index="date", columns="bin_id", values="fill_level")
                
                fig = px.imshow(
                    pivot_data,
                    labels=dict(x="Bin ID", y="Date", color="Fill Level (%)"),
                    x=pivot_data.columns,
                    y=pivot_data.index,
                    color_continuous_scale="RdYlGn_r",
                    title="Daily Average Fill Levels"
                )
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Bin collection frequency
                st.subheader("Bin Collection Analysis")
                
                # Detect when bins were emptied (significant drop in fill level)
                collections = []
                
                for bin_id in filtered_df["bin_id"].unique():
                    bin_data = filtered_df[filtered_df["bin_id"] == bin_id].sort_values("created_at")
                    
                    if len(bin_data) < 2:
                        continue
                    
                    for i in range(1, len(bin_data)):
                        current = bin_data.iloc[i]["fill_level"]
                        previous = bin_data.iloc[i-1]["fill_level"]
                        
                        # If fill level dropped by at least 30%, consider it a collection
                        if previous - current >= 30 and previous >= 50:
                            collections.append({
                                "bin_id": bin_id,
                                "collection_time": bin_data.iloc[i]["created_at"],
                                "previous_level": previous,
                                "new_level": current
                            })
                
                if collections:
                    collections_df = pd.DataFrame(collections)
                    
                    # Count collections by bin
                    collection_counts = collections_df.groupby("bin_id").size().reset_index(name="count")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Collection frequency chart
                        fig = px.bar(
                            collection_counts,
                            x="bin_id",
                            y="count",
                            title="Number of Collections per Bin",
                            labels={"bin_id": "Bin ID", "count": "Number of Collections"}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Show collection events
                        st.dataframe(
                            collections_df.sort_values("collection_time", ascending=False),
                            hide_index=True,
                            column_config={
                                "bin_id": "Bin ID",
                                "collection_time": "Collection Time",
                                "previous_level": st.column_config.NumberColumn("Previous Level", format="%.1f%%"),
                                "new_level": st.column_config.NumberColumn("New Level", format="%.1f%%")
                            }
                        )
                else:
                    st.info("No bin collections detected in the selected date range.")
            else:
                st.warning("No data available for the selected date range")
                
            # Raw data view with download option
            st.subheader("Raw Data")
            st.dataframe(
                filtered_df,
                hide_index=True,
                column_config={
                    "bin_id": "Bin ID",
                    "fill_level": st.column_config.NumberColumn("Fill Level", format="%.1f%%"),
                    "latitude": "Latitude",
                    "longitude": "Longitude",
                    "created_at": "Timestamp"
                }
            )
            
            # Download button for CSV
            csv = filtered_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download as CSV",
                csv,
                f"smart_bin_data_{start_date}_to_{end_date}.csv",
                "text/csv",
                key='download-csv'
            )
        else:
            st.warning("No data available")

if __name__ == "__main__":
    main()
    
    # Auto-refresh the app
    time.sleep(refresh_interval)
    st.rerun()
