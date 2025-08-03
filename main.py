import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
from datetime import datetime
import re
import numpy as np

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="OMSETKU AI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a professional look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 700;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 2rem;
        text-align: center;
    }
    
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        background-color: #f8f9fa;
    }
    
    .stButton button {
        background-color: #1E88E5;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        border: none;
    }
    
    .stButton button:hover {
        background-color: #1565C0;
    }
    
    .student-list {
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    .student-item {
        padding: 0.5rem 0;
        border-bottom: 1px solid #eee;
    }
</style>
""", unsafe_allow_html=True)

# Improved function to standardize column names
def standardize_columns(df):
    """
    Standardize column names to handle various possible header formats
    """
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Dictionary of possible column name variations and their standardized names
    column_mappings = {
        # School information
        'School Name': ['school name', 'school', 'nama sekolah', 'sekolah', 'institution', 'institution name'],
        'Address': ['address', 'alamat', 'school address', 'alamat sekolah', 'location', 'lokasi'],
        'Phone': ['phone', 'telepon', 'school phone', 'telepon sekolah', 'contact', 'kontak'],
        'Province': ['province', 'provinsi', 'state', 'wilayah'],
        
        # Student information
        'Student Name': ['student name', 'student', 'nama siswa', 'siswa', 'name', 'nama'],
        'Student Phone': ['student phone', 'telepon siswa', 'phone number', 'nomor telepon'],
        'Student NIB': ['student nib', 'nib', 'nomor induk', 'student id', 'id siswa'],
        'Student Link': ['student link', 'link siswa', 'profile', 'profil'],
        'Class Name': ['class name', 'class', 'nama kelas', 'kelas'],
        
        # Product information
        'Products': ['products', 'produk', 'item', 'items', 'barang'],
        'Product Details': ['product details', 'detail produk', 'details', 'detail', 'description', 'deskripsi'],
        'Product Categories': ['product categories', 'kategori produk', 'categories', 'kategori'],
        
        # Revenue information
        'Total Revenue': ['total revenue', 'revenue', 'pendapatan', 'omset', 'total pendapatan', 'total omset', 'income', 'sales'],
        
        # AQRF information
        'AQRF Level': ['aqrf level', 'level aqrf', 'level', 'tingkat', 'grade'],
        
        # Date information
        'Date': ['date', 'tanggal', 'transaction date', 'tanggal transaksi', 'order date', 'tanggal order']
    }
    
    # Standardize column names
    renamed_columns = {}
    for col in df_copy.columns:
        col_lower = col.lower().strip()
        for standard_name, variations in column_mappings.items():
            if col_lower in variations or col_lower == standard_name.lower():
                renamed_columns[col] = standard_name
                break
    
    # Rename columns if matches found
    if renamed_columns:
        df_copy = df_copy.rename(columns=renamed_columns)
    
    return df_copy

# Function to fix revenue scaling issues
def fix_revenue_scaling(df, revenue_column='Total Revenue'):
    """
    Fix revenue scaling issues by detecting and correcting values that are too large
    """
    if revenue_column not in df.columns:
        return df
    
    df_copy = df.copy()
    
    # Check if values are numeric
    if not pd.api.types.is_numeric_dtype(df_copy[revenue_column]):
        # Try to convert to numeric
        df_copy[revenue_column] = pd.to_numeric(df_copy[revenue_column], errors='coerce')
    
    # Check if we need to scale down the values
    # We'll use a heuristic: if median value is over 1 million, it's likely inflated
    non_zero_values = df_copy[df_copy[revenue_column] > 0][revenue_column]
    
    if len(non_zero_values) > 0:
        median_value = non_zero_values.median()
        
        # Check if values seem too large
        if median_value > 1000000:
            # Determine the appropriate scaling factor
            if median_value > 1000000000:  # If in billions
                scale_factor = 1000000
                st.warning("Revenue values appear to be in billions. Scaling down by 1,000,000.")
            elif median_value > 1000000:  # If in millions
                scale_factor = 1000
                st.warning("Revenue values appear to be in millions. Scaling down by 1,000.")
            
            # Apply scaling
            df_copy[revenue_column] = df_copy[revenue_column] / scale_factor
    
    return df_copy

# Improved function to preprocess data
def preprocess_data(df):
    """
    Comprehensive data preprocessing to handle various data formats and issues
    """
    # Standardize column names
    df = standardize_columns(df)
    
    # Create a copy to avoid modifying the original
    df_copy = df.copy()
    
    # Handle revenue data
    if 'Total Revenue' in df_copy.columns:
        # Convert revenue to numeric, handling various formats
        try:
            # If it's already numeric, this will work directly
            if not pd.api.types.is_numeric_dtype(df_copy['Total Revenue']):
                # Clean the data by removing currency symbols and thousand separators
                df_copy['Total Revenue'] = df_copy['Total Revenue'].astype(str)
                df_copy['Total Revenue'] = df_copy['Total Revenue'].str.replace(r'[Rp.,\s]', '', regex=True)
                df_copy['Total Revenue'] = df_copy['Total Revenue'].str.replace(',', '.', regex=False)
                
                # Replace empty strings with NaN
                df_copy['Total Revenue'] = df_copy['Total Revenue'].replace('', np.nan)
                
                # Convert to numeric
                df_copy['Total Revenue'] = pd.to_numeric(df_copy['Total Revenue'], errors='coerce')
            
            # Fix revenue scaling issues
            df_copy = fix_revenue_scaling(df_copy, 'Total Revenue')
            
        except Exception as e:
            st.warning(f"Warning: Error processing revenue data: {e}")
    
    # Handle AQRF Level standardization
    if 'AQRF Level' in df_copy.columns:
        try:
            # Extract numeric level from various formats
            def extract_level(value):
                if pd.isna(value):
                    return value
                
                value_str = str(value).lower().strip()
                
                # Try to extract numeric part
                level_match = re.search(r'(\d+)', value_str)
                if level_match:
                    return level_match.group(1)
                
                # If no numeric part, return original value
                return value
            
            # Apply the extraction function
            df_copy['AQRF Level'] = df_copy['AQRF Level'].apply(extract_level)
            
            # For consistency in display, format as "Level X"
            df_copy['AQRF Level Display'] = df_copy['AQRF Level'].apply(
                lambda x: f"Level {x}" if pd.notna(x) else np.nan
            )
        except Exception as e:
            st.warning(f"Warning: Error standardizing AQRF Level: {e}")
    
    # Handle date columns
    date_columns = [col for col in df_copy.columns if any(
        term in col.lower() for term in ['date', 'tanggal', 'waktu', 'time']
    )]
    
    for col in date_columns:
        try:
            df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')
        except Exception as e:
            st.warning(f"Warning: Error converting {col} to datetime: {e}")
    
    # Ensure student names are properly formatted
    if 'Student Name' in df_copy.columns:
        # Remove extra whitespace
        df_copy['Student Name'] = df_copy['Student Name'].astype(str).str.strip()
        # Replace empty strings with NaN
        df_copy['Student Name'] = df_copy['Student Name'].replace('', np.nan)
    
    # Ensure school names are properly formatted
    if 'School Name' in df_copy.columns:
        # Remove extra whitespace
        df_copy['School Name'] = df_copy['School Name'].astype(str).str.strip()
        # Replace empty strings with NaN
        df_copy['School Name'] = df_copy['School Name'].replace('', np.nan)
    
    return df_copy

# Helper function to extract date range from query
def extract_date_range(query):
    """Extract date range from query text"""
    # Common date formats in Indonesian
    months = {
        'januari': 1, 'jan': 1,
        'februari': 2, 'feb': 2,
        'maret': 3, 'mar': 3,
        'april': 4, 'apr': 4,
        'mei': 5, 'may': 5,
        'juni': 6, 'jun': 6,
        'juli': 7, 'jul': 7,
        'agustus': 8, 'aug': 8, 'agt': 8,
        'september': 9, 'sep': 9,
        'oktober': 10, 'oct': 10, 'okt': 10,
        'november': 11, 'nov': 11,
        'desember': 12, 'dec': 12, 'des': 12
    }
    
    # Try to extract month and year patterns
    month_year_pattern = r'(?:bulan\s+)?([a-zA-Z]+)(?:\s+|-)?(\d{4})'
    range_pattern = r'([a-zA-Z]+)(?:\s+|-)?(\d{4})\s*(?:-|sampai|hingga|s\.d\.|sd)\s*([a-zA-Z]+)(?:\s+|-)?(\d{4})'
    
    # Check for date range first
    range_match = re.search(range_pattern, query.lower())
    if range_match:
        start_month, start_year, end_month, end_year = range_match.groups()
        
        if start_month in months and end_month in months:
            start_date = f"{start_year}-{months[start_month]:02d}-01"
            # Set end date to last day of month
            if int(end_year) == 2 and months[end_month] == 2:
                end_day = 29 if int(end_year) % 4 == 0 else 28
            elif months[end_month] in [4, 6, 9, 11]:
                end_day = 30
            else:
                end_day = 31
            end_date = f"{end_year}-{months[end_month]:02d}-{end_day}"
            return start_date, end_date
    
    # Check for single month-year
    matches = re.findall(month_year_pattern, query.lower())
    if matches:
        date_ranges = []
        for month_name, year in matches:
            if month_name in months:
                month_num = months[month_name]
                # Set start date to first day of month
                start_date = f"{year}-{month_num:02d}-01"
                # Set end date to last day of month
                if month_num == 2:
                    end_day = 29 if int(year) % 4 == 0 else 28
                elif month_num in [4, 6, 9, 11]:
                    end_day = 30
                else:
                    end_day = 31
                end_date = f"{year}-{month_num:02d}-{end_day}"
                date_ranges.append((start_date, end_date))
        
        if date_ranges:
            # Return the earliest start date and latest end date
            earliest_start = min([d[0] for d in date_ranges])
            latest_end = max([d[1] for d in date_ranges])
            return earliest_start, latest_end
    
    return None, None

# Helper function to extract AQRF level from query
def extract_aqrf_level(query):
    """Extract AQRF level from query text"""
    level_pattern = r'(?:level|tingkat)\s*(\d+)'
    match = re.search(level_pattern, query.lower())
    if match:
        return match.group(1)
    
    # Check for direct level mention
    for i in range(1, 9):  # AQRF levels typically 1-8
        if f"level {i}" in query.lower() or f"level{i}" in query.lower():
            return str(i)
    
    return None

# Helper function to extract location from query
def extract_location(query):
    """Extract location information from query text"""
    # Common provinces and cities in Indonesia
    locations = {
        "jawa barat": ["jabar", "west java", "bandung", "bogor", "depok", "bekasi", "cimahi", "tasikmalaya", "cirebon", "sukabumi", "indramayu"],
        "jawa timur": ["jatim", "east java", "surabaya", "malang", "sidoarjo", "gresik", "mojokerto", "pasuruan", "probolinggo", "kediri", "madiun"],
        "jawa tengah": ["jateng", "central java", "semarang", "solo", "surakarta", "magelang", "pekalongan", "tegal", "salatiga"],
        "dki jakarta": ["jakarta", "jakarta pusat", "jakarta barat", "jakarta timur", "jakarta selatan", "jakarta utara"],
        "banten": ["tangerang", "serang", "cilegon", "tangerang selatan"],
        "yogyakarta": ["diy", "jogja", "yogya", "sleman", "bantul", "kulon progo"],
        "bali": ["denpasar", "badung", "gianyar", "tabanan"],
        "sumatera utara": ["sumut", "medan", "binjai", "tebing tinggi"],
        "sumatera barat": ["sumbar", "padang", "bukittinggi", "payakumbuh"],
        "sumatera selatan": ["sumsel", "palembang", "prabumulih"],
        "lampung": ["bandar lampung", "metro"],
        "kalimantan timur": ["kaltim", "samarinda", "balikpapan"],
        "kalimantan selatan": ["kalsel", "banjarmasin", "banjarbaru"],
        "sulawesi selatan": ["sulsel", "makassar", "parepare"],
        "nusa tenggara barat": ["ntb", "mataram", "bima"],
        "nusa tenggara timur": ["ntt", "kupang"]
    }
    
    query_lower = query.lower()
    
    # Check for province or city mentions
    for province, keywords in locations.items():
        if province in query_lower:
            return province
        for city in keywords:
            if city in query_lower:
                return province
    
    return None

# Helper function to extract school type from query
def extract_school_type(query):
    """Extract school type from query text"""
    query_lower = query.lower()
    
    school_types = {
        "SMK": ["smk", "sekolah menengah kejuruan"],
        "SMA": ["sma", "sekolah menengah atas"],
        "SMP": ["smp", "sekolah menengah pertama"],
        "SD": ["sd", "sekolah dasar"],
        "MI": ["mi", "madrasah ibtidaiyah"],
        "MTs": ["mts", "madrasah tsanawiyah"],
        "MA": ["ma", "madrasah aliyah"]
    }
    
    for school_type, keywords in school_types.items():
        for keyword in keywords:
            if keyword in query_lower:
                return school_type
    
    return None

# Function to filter students based on criteria
def get_filtered_students(data, aqrf_level=None, location=None, school_type=None, date_range=None):
    """Get filtered list of students based on various criteria"""
    
    data_copy = data.copy()
    
    # Filter by AQRF Level
    if aqrf_level and 'AQRF Level' in data_copy.columns:
        # Try different formats of AQRF level
        level_mask = (
            (data_copy['AQRF Level'] == aqrf_level) | 
            (data_copy['AQRF Level'] == f"Level {aqrf_level}") | 
            (data_copy['AQRF Level'] == f"level {aqrf_level}") |
            (data_copy['AQRF Level'].astype(str).str.contains(f"^{aqrf_level}$", regex=True, na=False))
        )
        data_copy = data_copy[level_mask]
    
    # Filter by location
    if location:
        location_filtered = False
        
        # Try different location columns
        for col in ['Province', 'Address', 'School Address', 'Location']:
            if col in data_copy.columns:
                try:
                    mask = data_copy[col].astype(str).str.contains(location, case=False, na=False)
                    if mask.any():
                        data_copy = data_copy[mask]
                        location_filtered = True
                        break
                except:
                    pass
        
        # If no location column matched, try school name
        if not location_filtered and 'School Name' in data_copy.columns:
            try:
                mask = data_copy['School Name'].astype(str).str.contains(location, case=False, na=False)
                if mask.any():
                    data_copy = data_copy[mask]
            except:
                pass
    
    # Filter by school type
    if school_type and 'School Name' in data_copy.columns:
        try:
            mask = data_copy['School Name'].astype(str).str.contains(school_type, case=False, na=False)
            if mask.any():
                data_copy = data_copy[mask]
        except:
            pass
    
    # Filter by date range
    if date_range and date_range[0] and date_range[1]:
        start_date, end_date = date_range
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Find date columns
        date_columns = [col for col in data_copy.columns if 
                        any(term in col.lower() for term in ['date', 'tanggal', 'waktu', 'time'])]
        
        if date_columns:
            for date_col in date_columns:
                try:
                    # Convert to datetime if not already
                    if not pd.api.types.is_datetime64_any_dtype(data_copy[date_col]):
                        data_copy[date_col] = pd.to_datetime(data_copy[date_col], errors='coerce')
                    
                    # Apply date filter
                    mask = (data_copy[date_col] >= start_date) & (data_copy[date_col] <= end_date)
                    if mask.any():
                        data_copy = data_copy[mask]
                        break
                except:
                    continue
    
    return data_copy

# Function to get school revenue ranking
def get_school_revenue_ranking(data, location=None, school_type=None, ascending=False):
    """Get ranking of schools by total revenue"""
    
    data_copy = data.copy()
    
    # Ensure we have the necessary columns
    if 'School Name' not in data_copy.columns or 'Total Revenue' not in data_copy.columns:
        return pd.DataFrame(), "Missing required columns (School Name or Total Revenue)"
    
    # Remove rows with missing school names or revenue
    valid_data = data_copy.dropna(subset=['School Name', 'Total Revenue'])
    
    if valid_data.empty:
        return pd.DataFrame(), "No valid revenue data available"
    
    # Apply filters before grouping
    filtered_data = valid_data.copy()
    
    # Filter by location
    if location:
        location_filtered = False
        
        # Try different location columns
        for col in ['Province', 'Address', 'School Address', 'Location']:
            if col in filtered_data.columns:
                try:
                    mask = filtered_data[col].astype(str).str.contains(location, case=False, na=False)
                    if mask.any():
                        filtered_data = filtered_data[mask]
                        location_filtered = True
                        break
                except:
                    pass
        
        # If no location column matched, try school name
        if not location_filtered:
            try:
                mask = filtered_data['School Name'].astype(str).str.contains(location, case=False, na=False)
                if mask.any():
                    filtered_data = filtered_data[mask]
                else:
                    # If still no match, return empty result
                    return pd.DataFrame(), f"No schools found in location: {location}"
            except:
                pass
    
    # Filter by school type
    if school_type:
        try:
            mask = filtered_data['School Name'].astype(str).str.contains(school_type, case=False, na=False)
            if mask.any():
                filtered_data = filtered_data[mask]
            else:
                # If no match, return empty result
                return pd.DataFrame(), f"No {school_type} schools found"
        except:
            pass
    
    # Group by school and calculate total revenue
    school_revenue = filtered_data.groupby('School Name')['Total Revenue'].sum().reset_index()
    
    # Sort by revenue
    school_revenue = school_revenue.sort_values(by='Total Revenue', ascending=ascending)
    
    if school_revenue.empty:
        return pd.DataFrame(), "No schools match the criteria"
    
    return school_revenue, None

# Function to preprocess data for AI analysis based on question
def preprocess_for_analysis(data, question):
    """Preprocess data based on the question to provide relevant context to AI"""
    
    # Extract key information from question
    aqrf_level = extract_aqrf_level(question)
    location = extract_location(question)
    school_type = extract_school_type(question)
    date_range = extract_date_range(question)
    
    question_lower = question.lower()
    
    # If question about school revenue ranking
    if any(term in question_lower for term in ['urutan', 'ranking', 'peringkat']):
        if 'sekolah' in question_lower:
            ascending = any(term in question_lower for term in ['terendah', 'terkecil', 'paling sedikit'])
            school_ranking, error_msg = get_school_revenue_ranking(
                data, location=location, school_type=school_type, ascending=ascending
            )
            
            if error_msg:
                # Return a small sample of the original data with the error message
                return data.head(10), error_msg
            
            return school_ranking, None
    
    # If question about students with specific criteria
    if any(term in question_lower for term in ['siswa', 'murid', 'pelajar']):
        filtered_students = get_filtered_students(
            data, aqrf_level=aqrf_level, location=location, 
            school_type=school_type, date_range=date_range
        )
        
        if filtered_students.empty:
            return data.head(10), "No students match the specified criteria"
        
        return filtered_students, None
    
    # If question about total revenue
    if any(term in question_lower for term in ['total pendapatan', 'total omset', 'pendapatan keseluruhan']):
        # Filter data based on criteria
        filtered_data = get_filtered_students(
            data, aqrf_level=aqrf_level, location=location, 
            school_type=school_type, date_range=date_range
        )
        
        if 'Total Revenue' in filtered_data.columns:
            # Create a summary DataFrame
            summary = pd.DataFrame({
                'Total Revenue': [filtered_data['Total Revenue'].sum()],
                'Number of Students': [filtered_data['Student Name'].nunique()],
                'Number of Schools': [filtered_data['School Name'].nunique()]
            })
            
            return summary, None
    
    # If question about highest/lowest revenue
    if any(term in question_lower for term in ['tertinggi', 'terbesar', 'terbanyak', 'terendah', 'terkecil']):
        if 'Total Revenue' in data.columns:
            ascending = any(term in question_lower for term in ['terendah', 'terkecil'])
            
            # Filter data based on criteria
            filtered_data = get_filtered_students(
                data, aqrf_level=aqrf_level, location=location, 
                school_type=school_type, date_range=date_range
            )
            
            if filtered_data.empty:
                return data.head(10), "No data matches the specified criteria"
            
            # Sort by revenue
            sorted_data = filtered_data.sort_values(by='Total Revenue', ascending=ascending)
            
            # Return top/bottom results
            return sorted_data.head(20), None
    
    # If question about AQRF levels
    if 'aqrf' in question_lower or 'level' in question_lower:
        if 'AQRF Level' in data.columns:
            # If asking about specific level
            if aqrf_level:
                filtered_data = get_filtered_students(data, aqrf_level=aqrf_level)
                if not filtered_data.empty:
                    return filtered_data, None
            
            # If general question about AQRF levels
            aqrf_summary = data.groupby('AQRF Level').agg({
                'Student Name': 'nunique',
                'School Name': 'nunique',
                'Total Revenue': 'sum'
            }).reset_index()
            
            aqrf_summary.columns = ['AQRF Level', 'Number of Students', 'Number of Schools', 'Total Revenue']
            
            return aqrf_summary, None
    
    # Default: return a reasonable sample of the data
    return data.head(50), None

# Function to get direct answers for common questions
def get_direct_answer(data, question):
    """Generate direct answers for common question patterns"""
    
    question_lower = question.lower()
    
    # Extract key information from question
    aqrf_level = extract_aqrf_level(question)
    location = extract_location(question)
    school_type = extract_school_type(question)
    date_range = extract_date_range(question)
    
    try:
        # Question about school revenue ranking
        if any(term in question_lower for term in ['urutan', 'ranking', 'peringkat']):
            if 'sekolah' in question_lower:
                ascending = any(term in question_lower for term in ['terendah', 'terkecil', 'paling sedikit'])
                school_ranking, error_msg = get_school_revenue_ranking(
                    data, location=location, school_type=school_type, ascending=ascending
                )
                
                if error_msg:
                    return f"Maaf, {error_msg.lower()}."
                
                if not school_ranking.empty:
                    # Check if we should include revenue in the output
                    include_revenue = "tanpa omset" not in question_lower and "tanpa pendapatan" not in question_lower
                    
                    # Format the response
                    result = f"Urutan sekolah"
                    if school_type:
                        result += f" {school_type}"
                    if location:
                        result += f" di {location}"
                    result += f" berdasarkan pendapatan {'terendah' if ascending else 'tertinggi'}:\n\n"
                    
                    # Add the list of schools with their revenue
                    for i, (_, row) in enumerate(school_ranking.head(20).iterrows(), 1):
                        if include_revenue:
                            # Format revenue with thousand separators and 2 decimal places
                            formatted_revenue = f"{row['Total Revenue']:,.2f}"
                            result += f"{i}. {row['School Name']} dengan total pendapatan {formatted_revenue}\n"
                        else:
                            result += f"{i}. {row['School Name']}\n"
                    
                    if len(school_ranking) > 20:
                        return result
        
        # Question about students with specific AQRF level
        if any(term in question_lower for term in ['siapa', 'nama', 'daftar', 'list']):
            if any(term in question_lower for term in ['siswa', 'murid', 'pelajar']):
                filtered_students = get_filtered_students(
                    data, aqrf_level=aqrf_level, location=location, 
                    school_type=school_type, date_range=date_range
                )
                
                if not filtered_students.empty:
                    # Format the response
                    result = f"Berikut daftar siswa"
                    if school_type:
                        result += f" {school_type}"
                    if location:
                        result += f" di {location}"
                    if aqrf_level:
                        result += f" dengan level AQRF {aqrf_level}"
                    if date_range and date_range[0] and date_range[1]:
                        start_date = pd.to_datetime(date_range[0]).strftime('%d %B %Y')
                        end_date = pd.to_datetime(date_range[1]).strftime('%d %B %Y')
                        result += f" pada periode {start_date} - {end_date}"
                    result += ":\n\n"
                    
                    # Add the list of students
                    for i, (_, row) in enumerate(filtered_students.head(30).iterrows(), 1):
                        student_info = f"{i}. {row['Student Name']}"
                        
                        if 'School Name' in row and pd.notna(row['School Name']):
                            student_info += f" dari {row['School Name']}"
                        
                        if 'AQRF Level' in row and pd.notna(row['AQRF Level']):
                            student_info += f" (Level AQRF: {row['AQRF Level']})"
                        
                        if 'Total Revenue' in row and pd.notna(row['Total Revenue']):
                            # Check if we should include revenue in the output
                            if "tanpa omset" not in question_lower and "tanpa pendapatan" not in question_lower:
                                student_info += f" - Pendapatan: {row['Total Revenue']:,.2f}"
                        
                        result += student_info + "\n"
                    
                    if len(filtered_students) > 30:
                        result += f"\n... dan {len(filtered_students) - 30} siswa lainnya."
                    
                    return result
                else:
                    return f"Tidak ditemukan siswa yang memenuhi kriteria tersebut."
        
        # Question about number of students with specific AQRF level
        if any(term in question_lower for term in ['berapa', 'jumlah', 'total']):
            if any(term in question_lower for term in ['siswa', 'murid', 'pelajar']):
                filtered_students = get_filtered_students(
                    data, aqrf_level=aqrf_level, location=location, 
                    school_type=school_type, date_range=date_range
                )
                
                student_count = filtered_students['Student Name'].nunique() if not filtered_students.empty else 0
                
                # Format the response
                result = f"Terdapat {student_count} siswa"
                if school_type:
                    result += f" {school_type}"
                if location:
                    result += f" di {location}"
                if aqrf_level:
                    result += f" dengan level AQRF {aqrf_level}"
                if date_range and date_range[0] and date_range[1]:
                    start_date = pd.to_datetime(date_range[0]).strftime('%d %B %Y')
                    end_date = pd.to_datetime(date_range[1]).strftime('%d %B %Y')
                    result += f" pada periode {start_date} - {end_date}"
                result += "."
                
                return result
        
        # Question about total revenue
        if any(term in question_lower for term in ['total pendapatan', 'total omset', 'pendapatan keseluruhan']):
            # Filter data based on criteria
            filtered_data = get_filtered_students(
                data, aqrf_level=aqrf_level, location=location, 
                school_type=school_type, date_range=date_range
            )
            
            if 'Total Revenue' in filtered_data.columns:
                total_revenue = filtered_data['Total Revenue'].sum()
                
                # Format the response
                result = f"Total pendapatan"
                if school_type:
                    result += f" untuk {school_type}"
                if location:
                    result += f" di {location}"
                if aqrf_level:
                    result += f" dengan level AQRF {aqrf_level}"
                if date_range and date_range[0] and date_range[1]:
                    start_date = pd.to_datetime(date_range[0]).strftime('%d %B %Y')
                    end_date = pd.to_datetime(date_range[1]).strftime('%d %B %Y')
                    result += f" pada periode {start_date} - {end_date}"
                result += f" adalah {total_revenue:,.2f}."
                
                return result
        
        # Question about highest/lowest revenue students
        if any(term in question_lower for term in ['tertinggi', 'terbesar', 'terbanyak', 'terendah', 'terkecil']):
            if 'siswa' in question_lower and 'Total Revenue' in data.columns:
                ascending = any(term in question_lower for term in ['terendah', 'terkecil'])
                
                # Filter data based on criteria
                filtered_data = get_filtered_students(
                    data, aqrf_level=aqrf_level, location=location, 
                    school_type=school_type, date_range=date_range
                )
                
                if filtered_data.empty:
                    return "Tidak ditemukan siswa yang memenuhi kriteria tersebut."
                
                # Sort by revenue
                sorted_data = filtered_data.sort_values(by='Total Revenue', ascending=ascending)
                
                # Check if we should include revenue in the output
                include_revenue = "tanpa omset" not in question_lower and "tanpa pendapatan" not in question_lower
                
                # Format the response
                result = f"Siswa dengan pendapatan {'terendah' if ascending else 'tertinggi'}"
                if school_type:
                    result += f" di {school_type}"
                if location:
                    result += f" di {location}"
                if aqrf_level:
                    result += f" dengan level AQRF {aqrf_level}"
                if date_range and date_range[0] and date_range[1]:
                    start_date = pd.to_datetime(date_range[0]).strftime('%d %B %Y')
                    end_date = pd.to_datetime(date_range[1]).strftime('%d %B %Y')
                    result += f" pada periode {start_date} - {end_date}"
                result += ":\n\n"
                
                # Add the list of students
                for i, (_, row) in enumerate(sorted_data.head(10).iterrows(), 1):
                    student_info = f"{i}. {row['Student Name']}"
                    
                    if 'School Name' in row and pd.notna(row['School Name']):
                        student_info += f" dari {row['School Name']}"
                    
                    if 'AQRF Level' in row and pd.notna(row['AQRF Level']):
                        student_info += f" (Level AQRF: {row['AQRF Level']})"
                    
                    if include_revenue and 'Total Revenue' in row and pd.notna(row['Total Revenue']):
                        student_info += f" - Pendapatan: {row['Total Revenue']:,.2f}"
                    
                    result += student_info + "\n"
                
                return result
        
        # Question about number of schools
        if any(term in question_lower for term in ['berapa sekolah', 'jumlah sekolah']):
            # Filter data based on criteria
            filtered_data = get_filtered_students(
                data, aqrf_level=aqrf_level, location=location, 
                school_type=school_type, date_range=date_range
            )
            
            school_count = filtered_data['School Name'].nunique() if not filtered_data.empty else 0
            
            # Format the response
            result = f"Terdapat {school_count} sekolah"
            if school_type:
                result += f" {school_type}"
            if location:
                result += f" di {location}"
            if aqrf_level:
                result += f" dengan siswa level AQRF {aqrf_level}"
            if date_range and date_range[0] and date_range[1]:
                start_date = pd.to_datetime(date_range[0]).strftime('%d %B %Y')
                end_date = pd.to_datetime(date_range[1]).strftime('%d %B %Y')
                result += f" pada periode {start_date} - {end_date}"
            result += "."
            
            return result
        
        # Question about AQRF level distribution
        if 'distribusi' in question_lower and ('aqrf' in question_lower or 'level' in question_lower):
            if 'AQRF Level' in data.columns:
                # Create AQRF level distribution
                aqrf_counts = data.groupby('AQRF Level')['Student Name'].nunique().reset_index()
                aqrf_counts.columns = ['AQRF Level', 'Jumlah Siswa']
                aqrf_counts = aqrf_counts.sort_values(by='AQRF Level')
                
                result = "Distribusi siswa berdasarkan level AQRF:\n\n"
                for _, row in aqrf_counts.iterrows():
                    result += f"Level {row['AQRF Level']}: {row['Jumlah Siswa']} siswa\n"
                
                return result
    
    except Exception as e:
        st.error(f"Error in get_direct_answer: {e}")
        return None
    
    # No direct answer available
    return None

# Header
st.markdown('<div class="main-header">OMSETKU AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Asisten Data Cerdas Anda</div>', unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'agent' not in st.session_state:
    st.session_state.agent = None

# Sidebar
with st.sidebar:
    st.markdown("### Unggah Data")
    
    uploaded_file = st.file_uploader("Pilih file Excel", type=['xlsx', 'xls', 'csv'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                raw_data = pd.read_csv(uploaded_file)
            else:
                raw_data = pd.read_excel(uploaded_file)
            
            # Apply comprehensive preprocessing
            data = preprocess_data(raw_data)
            
            # Store both raw and processed data
            st.session_state.raw_data = raw_data
            st.session_state.data = data
            
            st.success("Data berhasil dimuat!")
            
            # Display data info
            st.markdown("### Ringkasan Data")
            st.write(f"Jumlah Baris: {data.shape[0]}")
            st.write(f"Jumlah Kolom: {data.shape[1]}")
            
            # Sample data
            with st.expander("Lihat Contoh Data"):
                st.dataframe(data.head(3))
            
            # Debugging info
            with st.expander("Debug Data"):
                st.write("Kolom dalam data:", data.columns.tolist())
                
                if 'AQRF Level' in data.columns:
                    st.write("Nilai unik AQRF Level:", data['AQRF Level'].unique())
                
                if 'Student Name' in data.columns:
                    st.write("Jumlah nama siswa unik:", data['Student Name'].nunique())
                
                if 'Total Revenue' in data.columns:
                    st.write("Statistik Total Revenue:", data['Total Revenue'].describe())
                    
                    # Show top schools by revenue
                    top_schools = data.groupby('School Name')['Total Revenue'].sum().sort_values(ascending=False).head(5)
                    st.write("Top 5 sekolah berdasarkan pendapatan:")
                    st.write(top_schools)
                
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memuat data: {e}")
    
    st.markdown("---")
    st.markdown("### Tentang Aplikasi")
    st.markdown("""
    OMSETKU AI menyediakan wawasan cerdas dari data platform Omsetku Anda.
    
    Anda dapat bertanya tentang:
    - Informasi sekolah
    - Detail siswa
    - Produk dan kategori
    - Analisis pendapatan
    - Level AQRF
    """)

# Main content
if st.session_state.data is not None:
    # Initialize AI agent
    if st.session_state.agent is None:
        with st.spinner("Menginisialisasi agen AI..."):
            try:
                # Create LLM
                llm = ChatOpenAI(
                    model="gpt-4o",
                    temperature=0.2,
                    api_key=os.getenv("OPENAI_API_KEY")
                )
                
                # Create agent
                data_analyst = Agent(
                    role="Analis Data Omsetku",
                    goal="Menganalisis data Omsetku dan memberikan wawasan yang akurat",
                    backstory="Anda adalah seorang analis data ahli yang mengkhususkan diri pada data pendidikan dan bisnis. Anda unggul dalam menemukan pola dan mengekstrak wawasan yang bermakna dari data Omsetku.",
                    verbose=True,
                    llm=llm
                )
                
                # Create crew with single agent to simplify
                st.session_state.agent = data_analyst
            except Exception as e:
                st.error(f"Error saat menginisialisasi agen AI: {e}")
    
    # Chat interface
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("💬 Tanya OMSETKU AI")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f"**Anda:** {message['content']}")
        else:
            st.markdown(f"**OMSETKU AI:** {message['content']}")
    
    # Input for new question
    user_question = st.text_input("Ajukan pertanyaan tentang data Omsetku Anda:", key="user_question")
    
    if st.button("Dapatkan Jawaban"):
        if user_question:
            # Add user question to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            
            try:
                # Try to get direct answer for common questions
                direct_answer = get_direct_answer(st.session_state.data, user_question)
                
                if direct_answer:
                    # Add direct answer to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": direct_answer})
                    st.rerun()
                else:
                    # Preprocess data for AI analysis
                    filtered_data, error_msg = preprocess_for_analysis(st.session_state.data, user_question)
                    
                    # Extract key information from question
                    aqrf_level = extract_aqrf_level(user_question)
                    location = extract_location(user_question)
                    school_type = extract_school_type(user_question)
                    date_range = extract_date_range(user_question)
                    
                    # Create task description with context
                    task_description = f"""
                    Jawab pertanyaan berikut tentang data Omsetku: {user_question}
                    
                    Data yang tersedia memiliki {len(st.session_state.data)} baris dengan kolom: {', '.join(st.session_state.data.columns)}
                    
                    Berikut adalah sampel data yang relevan dengan pertanyaan:
                    {filtered_data.to_markdown()}
                    
                    Informasi penting:
                    - Data ini adalah data per-siswa, bukan per-sekolah
                    - Satu sekolah bisa memiliki banyak siswa
                    - Kolom "Total Revenue" menunjukkan pendapatan yang dihasilkan oleh siswa
                    - Jumlah total baris: {len(st.session_state.data)}
                    """
                    
                    # Add extracted information to task description
                    if aqrf_level:
                        task_description += f"\n- Pertanyaan terkait dengan AQRF Level {aqrf_level}"
                    if location:
                        task_description += f"\n- Pertanyaan terkait dengan lokasi: {location}"
                    if school_type:
                        task_description += f"\n- Pertanyaan terkait dengan tipe sekolah: {school_type}"
                    if date_range and date_range[0] and date_range[1]:
                        start_date = pd.to_datetime(date_range[0]).strftime('%d %B %Y')
                        end_date = pd.to_datetime(date_range[1]).strftime('%d %B %Y')
                        task_description += f"\n- Pertanyaan terkait dengan periode: {start_date} - {end_date}"
                    
                    if error_msg:
                        task_description += f"\n- Catatan: {error_msg}"
                    
                    task_description += """
                    
                    Berikan jawaban yang ringkas namun informatif dalam Bahasa Indonesia yang baik.
                    Fokus pada menjawab pertanyaan dengan tepat tanpa informasi berlebihan.
                    
                    Jika pertanyaan tentang daftar siswa atau sekolah, berikan daftar lengkap dengan format yang rapi.
                    Jika tidak ada data yang sesuai dengan kriteria, jelaskan dengan jelas bahwa data tidak ditemukan.
                    
                    Untuk nilai pendapatan, selalu tampilkan dengan format angka yang mudah dibaca (dengan pemisah ribuan dan 2 angka desimal).
                    """
                    
                    task = Task(
                        description=task_description,
                        expected_output="Jawaban ringkas dan informatif untuk pertanyaan pengguna",
                        agent=st.session_state.agent
                    )
                    
                    # Create crew for this specific task
                    crew = Crew(
                        agents=[st.session_state.agent],
                        tasks=[task],
                        verbose=True
                    )
                    
                    # Execute the task
                    with st.spinner("OMSETKU AI sedang menganalisis data Anda..."):
                        result = crew.kickoff()
                    
                    # Add AI response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": result})
                    
                    # Force a rerun to display the updated chat history
                    st.rerun()
            except Exception as e:
                # Handle error with user-friendly message
                error_message = f"Maaf, terjadi kesalahan saat memproses pertanyaan Anda: {str(e)}"
                st.session_state.chat_history.append({"role": "assistant", "content": error_message})
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
else:
    # Display welcome message and instructions when no data is loaded
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("""
    ## Selamat Datang di OMSETKU AI! 👋
    
    Asisten cerdas ini membantu Anda menganalisis dan memahami data platform Omsetku Anda.
    
    ### Cara Memulai:
    1. Unggah file data Excel Anda menggunakan panel samping
    2. Ajukan pertanyaan tentang sekolah, siswa, produk, dan pendapatan
    3. Dapatkan wawasan bertenaga AI
    
    ### Contoh Pertanyaan:
    - "Berapa total pendapatan dari semua siswa?"
    - "Siswa mana yang memiliki pendapatan tertinggi?"
    - "Urutan sekolah di Jawa Barat berdasarkan pendapatan tertinggi?"
    - "Siapa saja siswa SMK di Jawa Barat dengan level AQRF 5?"
    - "Berapa banyak siswa yang berada di Level AQRF 8?"
    - "Siapa saja siswa dengan level AQRF 8?"
    - "Bandingkan pendapatan antara level AQRF yang berbeda"
    """)
    st.markdown('</div>', unsafe_allow_html=True)
