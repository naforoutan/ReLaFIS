"""
Biomedical and clinical datasets.

This module is the natural home for datasets tied to health, biology, or
clinical research — including anything fetched live from the UCI ML Repository
via ucimlrepo.  Keeping them separate from the generic tabular modules makes
it easy to add genomics, EHR, or omics datasets later without cluttering the
general-purpose files.

Live fetch vs. local file
-------------------------
- *Live fetch* classes (e.g. Digits_UCI_Repo) pull data from the UCI API at
  runtime.  They require an internet connection and the `ucimlrepo` package.
- *Local file* classes (e.g. Diabetes) read from the project ./data/ tree.

To add a new dataset here:
  - Use ucimlrepo.fetch_ucirepo(id=<id>) for live fetch.
  - Set self.df = data.data.features and self.target = data.data.targets.
  - Call super().__init__() as usual.
"""

import pandas as pd
from .base import Test
from scipy.io import arff
import numpy as np

class Diabetes(Test):
    """Diabetes dataset from a local CSV file (different from Pima / UCI versions).

    Expects ./data/diabetes/diabetes.csv with a header row; the last column
    is treated as the target by the parent class default.
    """

    def __init__(self, *args, **kwargs) -> None:
        self.df = pd.read_csv("./data/diabetes/diabetes.csv")
        # Target column name is inferred by the Test base class from the last
        # column when self.target is not set explicitly — set it here if the
        # CSV layout differs from the default.
        super().__init__(*args, **kwargs)


class Digits_UCI_Repo(Test):
    """Optical Recognition of Handwritten Digits fetched live from UCI (id=81).

    Requires: pip install ucimlrepo
    """

    def __init__(self, *args, **kwargs) -> None:
        from ucimlrepo import fetch_ucirepo

        data = fetch_ucirepo(id=81)
        self.df = data.data.features
        self.target = data.data.targets
        super().__init__(*args, **kwargs)



class Colon(Test):
    """Colon Cancer dataset from local ARFF file (sparse format).
    
    62 samples, 2000 features, binary classification.
    """
    
    def __init__(self, *args, **kwargs) -> None:
        df, _, y = parse_sparse_arff("./data/colon-cancer.arff")
        
        # Features are all columns except the last one (target)
        self.df = df.iloc[:, :-1]
        self.target = df.iloc[:, -1]
        super().__init__(*args, **kwargs)


class Leukemia(Test):
    """Leukemia dataset from local file.
    
    72 samples, 7129 features, binary classification (ALL vs AML).
    Golub et al., 1999.
    
    Format: CSV with target as last column (ALL or AML).
    The file may have inconsistent column counts, so we handle that.
    """
    
    def __init__(self, *args, **kwargs) -> None:
        df = load_csv_with_error_handling("./data/Leukemia")
        
        # All columns except the last are features
        self.df = df.iloc[:, :-1]
        
        # Last column is the target (ALL or AML)
        self.target = df.iloc[:, -1]
        
        print(f"✅ Leukemia dataset loaded: {self.df.shape[0]} samples, {self.df.shape[1]} features, {self.target.nunique()} classes")
        super().__init__(*args, **kwargs)



class SRBCT(Test):
    """SRBCT dataset from local file.
    
    83 samples, 2308 features, 4-class classification.
    """
    
    def __init__(self, *args, **kwargs) -> None:
        df = load_gene_expression_file("./data/SRBCT", 
                                       target_values=['EWS', 'BL', 'NB', 'RMS'])
        
        self.df = df.iloc[:, :-1]
        self.target = df.iloc[:, -1]
        
        print(f"✅ SRBCT: {self.df.shape[0]} samples, {self.df.shape[1]} features")
        class_counts = self.target.value_counts()
        print(f"   Class distribution: {dict(class_counts)}")
        
        if class_counts.min() < 2:
            print("⚠️ Warning: Some classes have < 2 samples.")
        
        super().__init__(*args, **kwargs)


class Madelon(Test):
    """Madelon dataset from local ARFF file (sparse format).
    
    2600 samples, 500 features, binary classification.
    Uses scipy.io.arff for reliable parsing.
    """

    def __init__(self, *args, **kwargs) -> None:
        from scipy.io import arff
        
        data, meta = arff.loadarff("./data/madelon.arff")
        
        # Convert to DataFrame and decode bytes
        df = pd.DataFrame(data)
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].apply(lambda x: x.decode() if isinstance(x, bytes) else x)
        
        # Separate features and target
        self.df = df.iloc[:, :-1]  # All columns except last are features
        self.target = df.iloc[:, -1].astype(int if df.iloc[:, -1].str.isnumeric().all() else str)  # Last column is target
        
        # Map string classes '1' and '2' to 0 and 1 if needed
        if self.target.dtype == object:
            unique_vals = self.target.unique()
            if len(unique_vals) == 2:
                # Binary classification: map to 0 and 1
                mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
                self.target = self.target.map(mapping)

        super().__init__(*args, **kwargs)


class ORL(Test):
    def __init__(self, *args, **kwargs) -> None:
        from scipy.io import arff

        data, meta = arff.loadarff("./data/AT&T.arff")

        df = pd.DataFrame(data)

        # convert bytes → int/str if needed
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].apply(
                    lambda x: x.decode() if isinstance(x, bytes) else x
                )

        self.df = df.iloc[:, :-1]
        self.target = df.iloc[:, -1].astype(int)

        print(f"✅ ORL loaded correctly: {self.df.shape}")

        super().__init__(*args, **kwargs)


class Isolet(Test):
    """ISOLET dataset from local ARFF file.

    617 features, 26 classes, one line per example.
    """

    def __init__(self, *args, **kwargs) -> None:
        data, meta = arff.loadarff("./data/Isolet.arff")
        df = pd.DataFrame(data)

        # Convert bytes to native Python strings for all object columns
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].apply(
                    lambda x: x.decode() if isinstance(x, bytes) else x
                )

        self.df = df.iloc[:, :-1]
        self.target = df.iloc[:, -1]

        # Convert quoted numeric class labels like '1'..'26' to integers
        try:
            self.target = self.target.astype(int)
        except (ValueError, TypeError):
            pass

        print(f"✅ Isolet loaded: {self.df.shape[0]} samples, {self.df.shape[1]} features, {self.target.nunique()} classes")

        super().__init__(*args, **kwargs)


# ---------------------------------------------------------------------------
# Future slots — add below as you test new papers
# ---------------------------------------------------------------------------
# class EHRAdmissions(Test):   ...
# class GenomicsCancer(Test):  ...
# class Alzheimers(Test):      ...



def parse_sparse_arff(filepath):
    """
    Parse sparse ARFF format where data is like:
    {0 2.08075, 1 1.09907, 2 0.927763, ...}
    
    This parses the sparse format and returns a dense DataFrame.
    """
    # Read the file
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find the @data section
    data_start = False
    data_lines = []
    attribute_names = []
    attribute_types = []
    num_attributes = 0
    target_idx = -1
    
    # First pass: parse header to get number of attributes
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.lower().startswith('@data'):
            data_start = True
            continue
            
        if not data_start:
            # Parse attributes
            if line.lower().startswith('@attribute'):
                parts = line.split()
                if len(parts) >= 2:
                    attr_name = parts[1].strip("'\"")
                    attr_type = parts[2] if len(parts) > 2 else 'numeric'
                    attribute_names.append(attr_name)
                    attribute_types.append(attr_type)
                    num_attributes += 1
                    # Track which attribute is the target (if it has class values)
                    if '{' in attr_type and 'class' in attr_name.lower():
                        target_idx = len(attribute_names) - 1
        else:
            # Data lines
            if line and not line.startswith('%'):
                data_lines.append(line)
    
    # If target not found, assume it's the last attribute
    if target_idx == -1:
        target_idx = num_attributes - 1
    
    # Parse data lines
    parsed_data = []
    
    for line in data_lines:
        # Remove the "{" and "}" if present
        line = line.strip()
        if line.startswith('{') and line.endswith('}'):
            line = line[1:-1]
        
        # Initialize row with zeros (for sparse format)
        row = np.zeros(num_attributes - 1)  # Excluding target
        
        # Parse sparse entries: "index value" pairs
        if line.strip():
            # Split by comma and parse each pair
            pairs = [p.strip() for p in line.split(',') if p.strip()]
            target_value = 0
            
            for pair in pairs:
                parts = pair.split()
                if len(parts) >= 2:
                    idx = int(parts[0])
                    val = float(parts[1])
                    
                    if idx == target_idx:
                        target_value = val
                    elif idx < target_idx:
                        row[idx] = val
                    else:
                        # For indices after target, adjust
                        row[idx - 1] = val
            
            parsed_data.append((row, target_value))
    
    # Create DataFrame
    feature_names = [name for i, name in enumerate(attribute_names) if i != target_idx]
    X = np.array([row for row, _ in parsed_data])
    y = np.array([target for _, target in parsed_data])
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    return df, feature_names, y


def load_csv_with_error_handling(filepath, **kwargs):
    """
    Load CSV file with error handling for inconsistent rows.
    Skips problematic rows and reports issues.
    """
    try:
        # First try: Read with pandas
        df = pd.read_csv(filepath, header=None, **kwargs)
        return df
    except pd.errors.ParserError as e:
        print(f"⚠️ ParserError: {e}")
        print("Trying alternative approach...")
        
        # Second try: Read as lines and parse manually
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        # Find the first line with the target label (ALL or AML)
        # The target is the last column
        parsed_rows = []
        target_values = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Split by comma
            parts = line.split(',')
            
            # The last part should be the target (ALL or AML)
            if len(parts) >= 2:
                target = parts[-1].strip()
                if target in ['ALL', 'AML']:
                    # Extract features (all except last)
                    try:
                        features = [float(x) for x in parts[:-1]]
                        parsed_rows.append(features)
                        target_values.append(target)
                    except ValueError:
                        # Skip rows with non-numeric features
                        print(f"Skipping row {i+1}: contains non-numeric values")
                        continue
        
        # Create DataFrame
        if parsed_rows:
            # Find max feature count
            max_features = max(len(row) for row in parsed_rows)
            
            # Pad rows with zeros if needed
            padded_rows = []
            for row in parsed_rows:
                if len(row) < max_features:
                    row = row + [0.0] * (max_features - len(row))
                padded_rows.append(row)
            
            df = pd.DataFrame(padded_rows)
            df['target'] = target_values
            print(f"✅ Loaded {len(df)} samples with {max_features} features")
            return df
        else:
            raise ValueError("No valid rows found in the file")
        


def load_gene_expression_file(filepath, target_values=None):
    """
    Universal loader for gene expression files.
    Handles inconsistent column counts by padding/truncating.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    print(f"Total lines in file: {len(lines)}")
    
    parsed_rows = []
    target_values_list = []
    
    # Known target labels for SRBCT
    srbct_targets = ['EWS', 'BL', 'NB', 'RMS']
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        
        # Try to split by comma or tab
        if ',' in line:
            parts = line.split(',')
        elif '\t' in line:
            parts = line.split('\t')
        else:
            # Try splitting by whitespace
            parts = line.split()
        
        if len(parts) < 2:
            continue
        
        # Check if last part is a target label
        last = parts[-1].strip().strip('"').strip("'")
        
        # Check if it's a known target
        is_target = False
        target_label = None
        
        if target_values and last in target_values:
            is_target = True
            target_label = last
        elif last in srbct_targets:
            is_target = True
            target_label = last
        elif last in ['ALL', 'AML']:
            is_target = True
            target_label = last
        
        # Try to convert to float
        try:
            if is_target:
                # Target is a string label
                row_values = []
                for x in parts[:-1]:
                    try:
                        row_values.append(float(x))
                    except ValueError:
                        continue
                
                if row_values:
                    parsed_rows.append(row_values)
                    target_values_list.append(target_label)
            else:
                # All values are numeric
                row_values = [float(x) for x in parts]
                parsed_rows.append(row_values[:-1])  # All but last
                target_values_list.append(row_values[-1])  # Last as target
        except ValueError:
            # Skip rows that can't be parsed
            continue
    
    if not parsed_rows:
        raise ValueError(f"Could not parse any rows from {filepath}")
    
    # Pad rows to same length
    max_len = max(len(row) for row in parsed_rows)
    padded_rows = [row + [0.0] * (max_len - len(row)) for row in parsed_rows]
    
    # Create DataFrame
    df = pd.DataFrame(padded_rows)
    df['target'] = target_values_list
    
    print(f"Loaded {len(df)} samples with {max_len} features")
    print(f"Class distribution: {dict(pd.Series(target_values_list).value_counts())}")
    
    return df

