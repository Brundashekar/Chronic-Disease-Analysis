import streamlit as st
import subprocess
import pandas as pd
import plotly.express as px
import os
from datetime import datetime
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from googleapiclient.discovery import build
import json
from googleapiclient.http import MediaFileUpload  # Note: Not MediaFileUploader
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials

def upload_file(service, file_path, file_name):
    file_metadata = {'name': file_name}
    media = MediaFileUpload(file_path,
                          mimetype='application/octet-stream',
                          resumable=True)
    
    file = service.files().create(body=file_metadata,
                                media_body=media,
                                fields='id').execute()
    return file.get('id')
# Configure page
st.set_page_config(
    page_title="C-V2X Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Device configurations
DEVICE_CONFIG = {
    'OBU42': {'type': 'OBU', 'location': 'X500', 'ip': '192.168.1.42'},
    'OBU62': {'type': 'OBU', 'location': 'rover', 'ip': '192.168.1.62'},
    'RSU22': {'type': 'RSU', 'location': 'west_side', 'ip': '192.168.1.22'},
    'RSU31': {'type': 'RSU', 'location': 'north_airfield', 'ip': '192.168.1.31'}
}

def initialize_google_drive():
    """Initialize Google Drive API connection."""
    try:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=['https://www.googleapis.com/auth/drive.file']
        )
        return build('drive', 'v3', credentials=credentials)
    except Exception as e:
        st.error(f"Failed to initialize Google Drive: {e}")
        return None

def create_daily_folder(service):
    """Create daily folder in Google Drive."""
    folder_name = datetime.now().strftime("%Y%m%d")
    
    try:
        # Create main folder
        folder_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [st.secrets["google_drive_folder_id"]]
        }
        
        folder = service.files().create(body=folder_metadata, fields='id').execute()
        
        # Create device subfolders
        for device in DEVICE_CONFIG.keys():
            subfolder_metadata = {
                'name': device,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [folder['id']]
            }
            service.files().create(body=subfolder_metadata).execute()
            
        return folder['id']
    except Exception as e:
        st.error(f"Failed to create folder structure: {e}")
        return None

def process_data(df):
    """Process DataFrame with elevation bins and gap analysis."""
    try:
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Create elevation bins
        elevation_bins = [0, 25, 75, 125, 175, 225, 275, 325]
        labels = ['0-25ft', '25-75ft', '75-125ft', '125-175ft',
                 '175-225ft', '225-275ft', '275-325ft']
        
        # Add elevation bins
        df['elevation_bin'] = pd.cut(df['elevation'],
                                   bins=elevation_bins,
                                   labels=labels,
                                   include_lowest=True)
        
        # Mark gaps
        df['is_gap'] = df['time_diff'] > 3
        
        return df
    except Exception as e:
        st.error(f"Error in process_data: {e}")
        return None

def extract_pcap_data(pcap_file):
    """Extract data from PCAP file using tshark."""
    try:
        tshark_path = "C:/Program Files/Wireshark/tshark.exe"
        tshark_cmd = [
            tshark_path,
            "-r", pcap_file,
            "-T", "fields",
            "-e", "frame.time_epoch",
            "-e", "frame.time_delta",
            "-e", "j2735.lat",
            "-e", "j2735.long",
            "-e", "j2735.elev",
            "-E", "separator=,",
            "-E", "occurrence=f",
            "-Y", "j2735"
        ]
        
        result = subprocess.run(tshark_cmd, capture_output=True, text=True, shell=True)
        
        if result.stderr:
            st.error(f"TShark error: {result.stderr}")
            return None
        
        if not result.stdout:
            st.warning("No data from TShark")
            return None
        
        data = []
        previous_time = None
        
        for line in result.stdout.strip().split('\n'):
            if line:
                try:
                    fields = line.split(',')
                    if len(fields) == 5:
                        time, delta, lat, lon, elev = fields
                        epoch_time = float(time)
                        
                        if previous_time:
                            time_diff = epoch_time - previous_time
                        else:
                            time_diff = 0
                        previous_time = epoch_time
                        
                        data.append({
                            'timestamp': epoch_time,
                            'time_delta': float(delta) * 1000 if delta else 0,
                            'time_diff': time_diff,
                            'latitude': float(lat) / 10000000,
                            'longitude': float(lon) / 10000000,
                            'elevation': ((float(elev) / 10) - 156) * 3.28084
                        })
                except (ValueError, IndexError) as e:
                    st.warning(f"Error processing line: {line} - {e}")
                    continue
        
        if not data:
            st.warning("No valid data extracted")
            return None
            
        df = pd.DataFrame(data)
        return process_data(df)
        
    except Exception as e:
        st.error(f"Error in extract_pcap_data: {e}")
        return None

def upload_to_drive(service, file_path, folder_id, device_id):
    """Upload processed file to Google Drive."""
    try:
        file_metadata = {
            'name': os.path.basename(file_path),
            'parents': [folder_id]
        }
        
        media = MediaFileUploader(
            file_path,
            mimetype='text/csv',
            resumable=True
        )
        
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        
        return file.get('id')
    except Exception as e:
        st.error(f"Failed to upload file: {e}")
        return None

def create_dashboard():
    st.title("C-V2X Analysis Dashboard")
    
    # Initialize Google Drive
    drive_service = initialize_google_drive()
    if not drive_service:
        st.error("Failed to connect to Google Drive")
        return
    
    # Sidebar controls
    st.sidebar.header("Analysis Controls")
    selected_device = st.sidebar.selectbox(
        "Select Device",
        options=list(DEVICE_CONFIG.keys()),
        format_func=lambda x: f"{x} ({DEVICE_CONFIG[x]['location']})"
    )
    
    elevation_min = st.sidebar.number_input("Min Elevation (ft)", value=0)
    elevation_max = st.sidebar.number_input("Max Elevation (ft)", value=325)
    gap_threshold = st.sidebar.number_input("Gap Threshold (seconds)", value=3.0)
    
    # Display device information
    st.sidebar.markdown("---")
    st.sidebar.subheader("Device Information")
    device_info = DEVICE_CONFIG[selected_device]
    st.sidebar.write(f"Type: {device_info['type']}")
    st.sidebar.write(f"Location: {device_info['location']}")
    st.sidebar.write(f"IP Address: {device_info['ip']}")
    
    # File upload section
    uploaded_files = st.file_uploader(
        "Upload PCAP files",
        type=['pcap'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Create daily folder
        folder_id = create_daily_folder(drive_service)
        if not folder_id:
            st.error("Failed to create folder structure in Google Drive")
            return
            
        results = []
        for file in uploaded_files:
            with st.spinner(f"Processing {file.name}..."):
                temp_path = f"temp_{file.name}"
                with open(temp_path, "wb") as f:
                    f.write(file.getvalue())
                
                df = extract_pcap_data(temp_path)
                if df is not None:
                    results.append({
                        'filename': file.name,
                        'data': df,
                        'temp_path': temp_path
                    })
                    
                    # Save processed CSV
                    csv_path = f"{temp_path}_processed.csv"
                    df.to_csv(csv_path, index=False)
                    
                    # Upload to Google Drive
                    upload_to_drive(
                        drive_service,
                        csv_path,
                        folder_id,
                        selected_device
                    )
                    
                    # Cleanup
                    os.remove(csv_path)
                os.remove(temp_path)
        
        if results:
            for result in results:
                st.header(f"Analysis for {result['filename']}")
                df = result['data']
                
                # Create tabs for different analyses
                tab1, tab2, tab3, tab4 = st.tabs([
                    "Summary",
                    "Elevation",
                    "Time Analysis",
                    "Raw Data"
                ])
                
                with tab1:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Data Summary")
                        st.write(f"Total records: {len(df)}")
                        st.write("Time Statistics (ms):")
                        st.write(df['time_delta'].describe())
                    
                    with col2:
                        st.subheader("Gap Analysis")
                        gaps = df[df['is_gap']]
                        st.write(f"Number of gaps: {len(gaps)}")
                        if not gaps.empty:
                            st.write("Largest gaps:")
                            st.dataframe(
                                gaps.nlargest(5, 'time_diff')[
                                    ['timestamp', 'time_diff', 'elevation_bin']
                                ]
                            )
                
                with tab2:
                    st.subheader("Elevation Distribution")
                    fig = px.histogram(
                        df,
                        x='elevation_bin',
                        title='Messages by Elevation Range',
                        labels={'elevation_bin': 'Elevation Range'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    st.subheader("Time Analysis")
                    df_sorted = df.sort_values('elevation_bin')
                    bin_data = df_sorted.groupby('elevation_bin')[
                        'time_delta'
                    ].mean().reset_index()
                    
                    fig = px.scatter(
                        bin_data,
                        x='elevation_bin',
                        y='time_delta',
                        title='Average Time Delta vs Elevation Range',
                        labels={
                            'elevation_bin': 'Elevation Range',
                            'time_delta': 'Average Time Delta (ms)'
                        }
                    )
                    fig.update_yaxes(type="log")
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab4:
                    st.subheader("Raw Data")
                    st.dataframe(df)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label=f"Download {result['filename']} data",
                        data=csv,
                        file_name=f"{result['filename']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime='text/csv'
                    )
    else:
        st.info("Please upload PCAP files to begin analysis")

if __name__ == "__main__":
    create_dashboard()import streamlit as st
import subprocess
import pandas as pd
import plotly.express as px
import os
from datetime import datetime
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from googleapiclient.discovery import build
import json
from googleapiclient.http import MediaFileUpload  # Note: Not MediaFileUploader
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials

def upload_file(service, file_path, file_name):
    file_metadata = {'name': file_name}
    media = MediaFileUpload(file_path,
                          mimetype='application/octet-stream',
                          resumable=True)
    
    file = service.files().create(body=file_metadata,
                                media_body=media,
                                fields='id').execute()
    return file.get('id')
# Configure page
st.set_page_config(
    page_title="C-V2X Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Device configurations
DEVICE_CONFIG = {
    'OBU42': {'type': 'OBU', 'location': 'X500', 'ip': '192.168.1.42'},
    'OBU62': {'type': 'OBU', 'location': 'rover', 'ip': '192.168.1.62'},
    'RSU22': {'type': 'RSU', 'location': 'west_side', 'ip': '192.168.1.22'},
    'RSU31': {'type': 'RSU', 'location': 'north_airfield', 'ip': '192.168.1.31'}
}

def initialize_google_drive():
    """Initialize Google Drive API connection."""
    try:
        credentials = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"],
            scopes=['https://www.googleapis.com/auth/drive.file']
        )
        return build('drive', 'v3', credentials=credentials)
    except Exception as e:
        st.error(f"Failed to initialize Google Drive: {e}")
        return None

def create_daily_folder(service):
    """Create daily folder in Google Drive."""
    folder_name = datetime.now().strftime("%Y%m%d")
    
    try:
        # Create main folder
        folder_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [st.secrets["google_drive_folder_id"]]
        }
        
        folder = service.files().create(body=folder_metadata, fields='id').execute()
        
        # Create device subfolders
        for device in DEVICE_CONFIG.keys():
            subfolder_metadata = {
                'name': device,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [folder['id']]
            }
            service.files().create(body=subfolder_metadata).execute()
            
        return folder['id']
    except Exception as e:
        st.error(f"Failed to create folder structure: {e}")
        return None

def process_data(df):
    """Process DataFrame with elevation bins and gap analysis."""
    try:
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Create elevation bins
        elevation_bins = [0, 25, 75, 125, 175, 225, 275, 325]
        labels = ['0-25ft', '25-75ft', '75-125ft', '125-175ft',
                 '175-225ft', '225-275ft', '275-325ft']
        
        # Add elevation bins
        df['elevation_bin'] = pd.cut(df['elevation'],
                                   bins=elevation_bins,
                                   labels=labels,
                                   include_lowest=True)
        
        # Mark gaps
        df['is_gap'] = df['time_diff'] > 3
        
        return df
    except Exception as e:
        st.error(f"Error in process_data: {e}")
        return None

def extract_pcap_data(pcap_file):
    """Extract data from PCAP file using tshark."""
    try:
        tshark_path = "C:/Program Files/Wireshark/tshark.exe"
        tshark_cmd = [
            tshark_path,
            "-r", pcap_file,
            "-T", "fields",
            "-e", "frame.time_epoch",
            "-e", "frame.time_delta",
            "-e", "j2735.lat",
            "-e", "j2735.long",
            "-e", "j2735.elev",
            "-E", "separator=,",
            "-E", "occurrence=f",
            "-Y", "j2735"
        ]
        
        result = subprocess.run(tshark_cmd, capture_output=True, text=True, shell=True)
        
        if result.stderr:
            st.error(f"TShark error: {result.stderr}")
            return None
        
        if not result.stdout:
            st.warning("No data from TShark")
            return None
        
        data = []
        previous_time = None
        
        for line in result.stdout.strip().split('\n'):
            if line:
                try:
                    fields = line.split(',')
                    if len(fields) == 5:
                        time, delta, lat, lon, elev = fields
                        epoch_time = float(time)
                        
                        if previous_time:
                            time_diff = epoch_time - previous_time
                        else:
                            time_diff = 0
                        previous_time = epoch_time
                        
                        data.append({
                            'timestamp': epoch_time,
                            'time_delta': float(delta) * 1000 if delta else 0,
                            'time_diff': time_diff,
                            'latitude': float(lat) / 10000000,
                            'longitude': float(lon) / 10000000,
                            'elevation': ((float(elev) / 10) - 156) * 3.28084
                        })
                except (ValueError, IndexError) as e:
                    st.warning(f"Error processing line: {line} - {e}")
                    continue
        
        if not data:
            st.warning("No valid data extracted")
            return None
            
        df = pd.DataFrame(data)
        return process_data(df)
        
    except Exception as e:
        st.error(f"Error in extract_pcap_data: {e}")
        return None

def upload_to_drive(service, file_path, folder_id, device_id):
    """Upload processed file to Google Drive."""
    try:
        file_metadata = {
            'name': os.path.basename(file_path),
            'parents': [folder_id]
        }
        
        media = MediaFileUploader(
            file_path,
            mimetype='text/csv',
            resumable=True
        )
        
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        
        return file.get('id')
    except Exception as e:
        st.error(f"Failed to upload file: {e}")
        return None

def create_dashboard():
    st.title("C-V2X Analysis Dashboard")
    
    # Initialize Google Drive
    drive_service = initialize_google_drive()
    if not drive_service:
        st.error("Failed to connect to Google Drive")
        return
    
    # Sidebar controls
    st.sidebar.header("Analysis Controls")
    selected_device = st.sidebar.selectbox(
        "Select Device",
        options=list(DEVICE_CONFIG.keys()),
        format_func=lambda x: f"{x} ({DEVICE_CONFIG[x]['location']})"
    )
    
    elevation_min = st.sidebar.number_input("Min Elevation (ft)", value=0)
    elevation_max = st.sidebar.number_input("Max Elevation (ft)", value=325)
    gap_threshold = st.sidebar.number_input("Gap Threshold (seconds)", value=3.0)
    
    # Display device information
    st.sidebar.markdown("---")
    st.sidebar.subheader("Device Information")
    device_info = DEVICE_CONFIG[selected_device]
    st.sidebar.write(f"Type: {device_info['type']}")
    st.sidebar.write(f"Location: {device_info['location']}")
    st.sidebar.write(f"IP Address: {device_info['ip']}")
    
    # File upload section
    uploaded_files = st.file_uploader(
        "Upload PCAP files",
        type=['pcap'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Create daily folder
        folder_id = create_daily_folder(drive_service)
        if not folder_id:
            st.error("Failed to create folder structure in Google Drive")
            return
            
        results = []
        for file in uploaded_files:
            with st.spinner(f"Processing {file.name}..."):
                temp_path = f"temp_{file.name}"
                with open(temp_path, "wb") as f:
                    f.write(file.getvalue())
                
                df = extract_pcap_data(temp_path)
                if df is not None:
                    results.append({
                        'filename': file.name,
                        'data': df,
                        'temp_path': temp_path
                    })
                    
                    # Save processed CSV
                    csv_path = f"{temp_path}_processed.csv"
                    df.to_csv(csv_path, index=False)
                    
                    # Upload to Google Drive
                    upload_to_drive(
                        drive_service,
                        csv_path,
                        folder_id,
                        selected_device
                    )
                    
                    # Cleanup
                    os.remove(csv_path)
                os.remove(temp_path)
        
        if results:
            for result in results:
                st.header(f"Analysis for {result['filename']}")
                df = result['data']
                
                # Create tabs for different analyses
                tab1, tab2, tab3, tab4 = st.tabs([
                    "Summary",
                    "Elevation",
                    "Time Analysis",
                    "Raw Data"
                ])
                
                with tab1:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Data Summary")
                        st.write(f"Total records: {len(df)}")
                        st.write("Time Statistics (ms):")
                        st.write(df['time_delta'].describe())
                    
                    with col2:
                        st.subheader("Gap Analysis")
                        gaps = df[df['is_gap']]
                        st.write(f"Number of gaps: {len(gaps)}")
                        if not gaps.empty:
                            st.write("Largest gaps:")
                            st.dataframe(
                                gaps.nlargest(5, 'time_diff')[
                                    ['timestamp', 'time_diff', 'elevation_bin']
                                ]
                            )
                
                with tab2:
                    st.subheader("Elevation Distribution")
                    fig = px.histogram(
                        df,
                        x='elevation_bin',
                        title='Messages by Elevation Range',
                        labels={'elevation_bin': 'Elevation Range'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab3:
                    st.subheader("Time Analysis")
                    df_sorted = df.sort_values('elevation_bin')
                    bin_data = df_sorted.groupby('elevation_bin')[
                        'time_delta'
                    ].mean().reset_index()
                    
                    fig = px.scatter(
                        bin_data,
                        x='elevation_bin',
                        y='time_delta',
                        title='Average Time Delta vs Elevation Range',
                        labels={
                            'elevation_bin': 'Elevation Range',
                            'time_delta': 'Average Time Delta (ms)'
                        }
                    )
                    fig.update_yaxes(type="log")
                    st.plotly_chart(fig, use_container_width=True)
                
                with tab4:
                    st.subheader("Raw Data")
                    st.dataframe(df)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label=f"Download {result['filename']} data",
                        data=csv,
                        file_name=f"{result['filename']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime='text/csv'
                    )
    else:
        st.info("Please upload PCAP files to begin analysis")

if __name__ == "__main__":
    create_dashboard()