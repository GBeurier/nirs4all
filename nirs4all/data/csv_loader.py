import csv
import io
import pandas as pd
import gzip
import zipfile
from pathlib import Path
import numpy as np


# =============================================================================
# Utility: Check if a string can be converted to float, given a decimal separator
# =============================================================================
def _can_be_float(value, decimal_sep):
    """Check if a string can be converted to a float."""
    if not isinstance(value, str):
        return False  # Only strings should be checked

    value = value.strip()
    if not value:
        return False

    try:
        # If there's 'e' or 'E', handle scientific notation
        if 'e' in value.lower() or 'E' in value:
            float(value)
            return True

        # Replace decimal separator if needed
        if decimal_sep == '.':
            float(value)
        else:
            float(value.replace(decimal_sep, '.', 1))
        return True
    except ValueError:
        return False


# =============================================================================
# Utility: Strip all quotes from the file content
# =============================================================================
def _strip_all_quotes(content):
    """
    Removes *all* single-quote and double-quote characters from the string.
    """
    return content.replace('"', '').replace("'", "")


# =============================================================================
# Detect delimiter (unless specified by user)
# =============================================================================
def _detect_delimiter(lines, possible_delimiters=None):
    """
    Attempt to detect the delimiter by looking at the consistency of
    the number of columns. Return the best candidate or None if no good guess.
    """
    if possible_delimiters is None:
        possible_delimiters = [';', ',', '\t', '|', ' ']

    best_delim = None
    max_consistent_cols = -1
    most_cols_at_max_consistency = 0
   
    # Join lines so csv.reader sees them as input
    content_for_test = "".join(lines)

    for delim_candidate in possible_delimiters:
        try:
            reader = csv.reader(io.StringIO(content_for_test), delimiter=delim_candidate)
            cols_counts = [len(row) for row in reader if row]

            if not cols_counts:
                continue

            # The number of columns that appears the most
            most_frequent_cols = max(set(cols_counts), key=cols_counts.count)
            # How many lines have that number of columns
            consistency = sum(1 for count in cols_counts if count == most_frequent_cols)

            # Choose the delimiter that maximizes the consistency, then the number of columns
            if consistency > max_consistent_cols:
                max_consistent_cols = consistency
                most_cols_at_max_consistency = most_frequent_cols
                best_delim = delim_candidate
            elif consistency == max_consistent_cols:
                # If same consistency, prefer the one with more columns
                if most_frequent_cols > most_cols_at_max_consistency:
                    most_cols_at_max_consistency = most_frequent_cols
                    best_delim = delim_candidate
        except (csv.Error, ValueError):
            continue  # ignore parse errors with this candidate

    return best_delim


# =============================================================================
# Detect decimal separator and header (unless specified by user)
# =============================================================================
def _detect_decimal_and_header(parsed_rows, data_type='x'):
    """
    Given a list of parsed_rows (already split by delimiter),
    try to determine the decimal separator and whether there's a header.

    Returns: (best_decimal_sep, best_has_header)
    """
    if not parsed_rows:
        return '.', False  # fallback

    # We'll guess it by looking at numeric vs. non-numeric content
    num_cols = len(parsed_rows[0])
    if num_cols == 0:
        return '.', False  # fallback

    best_decimal_sep = '.'
    best_has_header = False
    max_numeric_score = -1.0

    for decimal_sep in ['.', ',']:
        for has_header_option in [False, True]:
            first_data_row_index = 1 if has_header_option else 0
            if len(parsed_rows) <= first_data_row_index:
                # no data rows to evaluate
                current_score = 0.0
            else:
                data_rows = parsed_rows[first_data_row_index:]
                numeric_cells = 0
                total_cells = 0
        
                for row in data_rows:
                    # We only consider rows with at least close to the expected columns
                    if abs(len(row) - num_cols) <= 1:
                        for val in row:
                            total_cells += 1
                            if _can_be_float(val, decimal_sep):  # This would call the commented out function
                                numeric_cells += 1

                current_score = numeric_cells / total_cells if total_cells else 0.0

            # If we declared there's a header but that row also looks numeric,
            # apply a small penalty
            if has_header_option and parsed_rows:
                header_row = parsed_rows[0]
                if len(header_row) == num_cols:
                    header_numeric_cells = sum(_can_be_float(cell, decimal_sep) for cell in header_row)  # This would call the commented out function
                    header_score = header_numeric_cells / len(header_row) if header_row else 0.0
                    if current_score > 0.5 and header_score >= current_score:
                        current_score *= 0.5

            if current_score > max_numeric_score + 1e-6:
                max_numeric_score = current_score
                best_decimal_sep = decimal_sep
                best_has_header = has_header_option
            elif abs(current_score - max_numeric_score) < 1e-6:
                # Tie-break: prefer '.' over ',' and prefer has_header=False over True
                if best_decimal_sep == ',' and decimal_sep == '.':
                    best_decimal_sep = decimal_sep
                    best_has_header = has_header_option
                elif best_has_header and (not has_header_option):
                    best_decimal_sep = decimal_sep
                    best_has_header = has_header_option

    return best_decimal_sep, best_has_header


# =============================================================================
# Main routine: Determine CSV parameters, skipping detection if user param is given
# =============================================================================
def _determine_csv_parameters(csv_content: str,  # csv_content is not used anymore
                              sample_lines=20,  # sample_lines is not used anymore
                              data_type='x',  # data_type is not used anymore
                              user_params=None, *, bypass_auto_detection=True):
    """
    Sets default CSV parameters (delimiter, decimal separator, header)
    and allows them to be overridden by `user_params`.
    The auto-detection logic is commented out.
    """
    if user_params is None:
        user_params = {}

    # Default parameters
    delimiter = user_params.get('delimiter', ';')
    decimal_sep = user_params.get('decimal_separator', '.')
    has_header = user_params.get('has_header', True)

    if not bypass_auto_detection:
        lines = []
        with io.StringIO(csv_content) as f:
            for i, line in enumerate(f):
                if i >= sample_lines:
                    break
                if line.strip():
                    lines.append(line)
        
        if not lines:
            # no lines to parse
            return {
                'delimiter': user_params.get('delimiter', ';'),  # Default
                'decimal_separator': user_params.get('decimal_separator', '.'),  # Default
                'has_header': user_params.get('has_header', True)  # Default
            }
        
        # 1) Delimiter detection
        if 'delimiter' in user_params:
            delimiter = user_params['delimiter']
        else:
            delimiter = _detect_delimiter(lines)  # Auto-detection commented out
            # delimiter = ';'  # Default
        
        if not delimiter:
            delimiter = ';'  # Default
        
        # 2) Parse a small sample using the chosen delimiter to create parsed_rows
        sample_data = "".join(lines)
        parsed_rows_reader = csv.reader(io.StringIO(sample_data), delimiter=delimiter)
        parsed_rows = [row for row in parsed_rows_reader if any(cell.strip() for cell in row)]
        
        # 3) Detect decimal separator / header if not specified
        if 'decimal_separator' in user_params:
            decimal_sep = user_params['decimal_separator']
        else:
            decimal_sep, _ = _detect_decimal_and_header(parsed_rows, data_type=data_type)  # Auto-detection commented out
            # decimal_sep = '.'  # Default
        
        if 'has_header' in user_params:
            has_header = user_params['has_header']
        else:
            _, has_header = _detect_decimal_and_header(parsed_rows, data_type=data_type)  # Auto-detection commented out
            # has_header = True  # Default

    return {
        'delimiter': delimiter,
        'decimal_separator': decimal_sep,
        'has_header': has_header
    }


# =============================================================================
# Main function: load_csv
# =============================================================================
def load_csv(path, na_policy='auto', data_type='x', categorical_mode='auto', **user_params):
    """
    Loads a CSV file using specified or default parameters, cleans data,
    handles NA values, and performs type conversions.

    Args:
        path (str or Path): Path to the CSV file (.csv, .gz, .zip).
        na_policy (str): 'remove' or 'abort' (or 'auto' which acts like 'remove').
            This policy applies to row removal if NAs are found.
        data_type (str): 'x' or 'y'. Influences type conversion.
        categorical_mode (str): How to handle string columns in 'y' data:
            - 'auto': Convert string columns to numerical categories.
            - 'preserve': Keep string columns (will become NaN if not convertible by final astype).
            - 'none': Treat all columns as potentially numeric.
        **user_params: CSV parsing parameters (delimiter, decimal_separator, has_header)
            and other pandas.read_csv arguments.

    Returns:
        (pandas.DataFrame | None, dict, pandas.Series | None):
            - DataFrame with processed data (before NA row removal).
            - Report dictionary.
            - Boolean Series indicating rows with NAs (aligned with the returned DataFrame).
            None if an error occurs before this stage.
    """
    if na_policy == 'auto':
        na_policy = 'remove'

    if na_policy not in ['remove', 'abort']:
        raise ValueError("Invalid NA policy - only 'remove' or 'abort' (or 'auto') are supported.")    
    if categorical_mode not in ['auto', 'preserve', 'none']:
        raise ValueError("Invalid categorical mode - only 'auto', 'preserve', or 'none' are supported.")

    report = {
        'file_path': str(path),
        'detection_params': None,
        'delimiter': None,  # For backward compatibility
        'decimal_separator': None,  # If needed
        'has_header': None,  # If needed
        'initial_shape': None,
        'final_shape': None,
        'na_handling': {
            'strategy': na_policy,
            'na_detected': False,
            'nb_removed_rows': 0,
            'removed_rows_indices': []
        },
        'categorical_info': {},  # Store category mappings
        'warnings': [],  # Store warnings about ambiguous detections
        'error': None
    }

    try:
        file_path = Path(path)
        if not file_path.exists():
            # The test expects either "n'existe pas" or "not exist" in the error
            raise FileNotFoundError(f"Le fichier n'existe pas: {path}")

        # --- 1) Read file content ---
        if file_path.suffix == '.gz':
            with gzip.open(file_path, 'rt', encoding='utf-8') as f:
                content = f.read()
        elif file_path.suffix == '.zip':
            with zipfile.ZipFile(file_path, 'r') as z:
                csv_files_in_zip = [n for n in z.namelist() if n.lower().endswith('.csv')]
                if not csv_files_in_zip:
                    raise ValueError(f"No .csv found in zip: {path}")
                if len(csv_files_in_zip) > 1:
                    print(f"Warning: multiple .csv found in {path}. Using {csv_files_in_zip[0]}")
                content = z.read(csv_files_in_zip[0]).decode('utf-8')
        else:
            # plain text read
            try:
                with open(file_path, 'r', encoding='utf-8', newline='') as f:
                    content = f.read()
            except UnicodeDecodeError:
                print(f"Warning: failed reading {path} with UTF-8. Trying Latin-1.")
                with open(file_path, 'r', encoding='latin-1', newline='') as f:
                    content = f.read()

        if not content.strip():
            raise ValueError("File is empty or could not be read.")

        # --- 2) Determine CSV parameters (now uses defaults or user_params) ---
        # The csv_content argument to _determine_csv_parameters is not strictly needed anymore
        # as auto-detection is off, but kept for signature consistency for now.
        detection_params = _determine_csv_parameters(
            csv_content=content, 
            user_params=user_params
        )

        # Extract parameters to be used
        delimiter = detection_params['delimiter']
        decimal_sep = detection_params['decimal_separator']
        has_header = detection_params['has_header']
        
        # Update report with used parameters
        report['detection_params'] = {
            'delimiter': delimiter,
            'decimal_separator': decimal_sep,
            'has_header': has_header
        }
        report['delimiter'] = delimiter
        report['decimal_separator'] = decimal_sep
        report['has_header'] = has_header

        # --- 4) Load with pandas.read_csv using the determined parameters ---
        read_csv_kwargs = {
            'sep': delimiter,
            'decimal': decimal_sep,
            'header': 0 if has_header else None,
            'na_filter': True,
            'na_values': ['NA', 'N/A', ''],
            'keep_default_na': True,
            'engine': 'python',  # Start with python engine for better error reporting/flexibility
            'skip_blank_lines': True,
            'quoting': csv.QUOTE_MINIMAL,  # Default pandas behavior, explicit here
        }

        # Add user-provided read_csv args (besides the three we handled)
        for k, v in user_params.items():
            if k not in ['delimiter', 'decimal_separator', 'has_header']:
                read_csv_kwargs[k] = v

        try:
            # Pass the original content to StringIO
            data = pd.read_csv(io.StringIO(content), **read_csv_kwargs)
        except Exception as e1:
            print(f"Warning: read_csv with engine='python' failed: {e1}")
            try:
                read_csv_kwargs['engine'] = 'c'
                data = pd.read_csv(io.StringIO(content), **read_csv_kwargs)
            except Exception as e2:
                msg = f"Could not parse CSV. Python engine error: {e1} | C engine error: {e2}"
                report['error'] = msg
                return None, report, None        
        report['initial_shape'] = data.shape

        # ---> FIX: Ensure column names are strings <--- 
        data.columns = data.columns.astype(str)

        report['shape_after_all_na_col_removal'] = data.shape
        
        if data.empty:  # If all columns were NA or file was effectively empty after header
            report['warnings'].append("Data is empty after removing all-NA columns or due to empty content.")
            # Return empty DataFrame, report, and an empty Series for na_row_mask
            return pd.DataFrame(), report, pd.Series(dtype=bool)

        # --- 5) Handle type conversion based on data_type ---
        # Ensure categorical_info is reset for this run, it's part of the main report dict.
        report['categorical_info'] = {} 
        _local_categorical_mappings = {} # Use a local temporary dict for populating

        if data_type == 'y':
            for col in data.columns:
                _original_col_series = data[col].copy() # Keep original for astype(str) and original NaN count
                _numeric_representation = pd.to_numeric(data[col], errors='coerce') # For NaN comparison and default conversion
                
                _original_is_object = pd.api.types.is_object_dtype(data[col].dtype)
                _original_is_numeric = pd.api.types.is_numeric_dtype(data[col].dtype)

                if categorical_mode == 'auto':
                    should_treat_as_categorical = False
                    if _original_is_object:
                        should_treat_as_categorical = True
                    elif not _original_is_numeric:  # Catches mixed types, booleans etc.
                        # Heuristic: if to_numeric creates more NaNs than original, or low cardinality
                        if _numeric_representation.isna().sum() > _original_col_series.isna().sum() or \
                           (_original_col_series.nunique() < len(_original_col_series) * 0.8 and _original_col_series.nunique() < 50):
                            should_treat_as_categorical = True

                    if should_treat_as_categorical:
                        # Warning for ambiguous numeric-like headers
                        # Check if column name is purely numeric or float-like (one dot)
                        if col.isdigit() or (col.count('.') == 1 and col.replace('.', '', 1).isdigit()):
                            report['warnings'].append(f"Column '{col}' detected as categorical but has a numeric header")
                        
                        # Factorize using the original data treated as strings to ensure correct categories
                        _codes, _categories = pd.factorize(_original_col_series.astype(str))
                        data[col] = _codes
                        _local_categorical_mappings[col] = {'categories': _categories.tolist()}
                    else:
                        # Not deemed categorical by 'auto' logic, so make it numeric
                        data[col] = _numeric_representation
                
                elif categorical_mode == 'preserve':
                    # In 'preserve' mode, all columns are converted to numeric.
                    # String values will become NaN. Numerics stay. No factorization.
                    data[col] = _numeric_representation 
                    # _local_categorical_mappings remains empty for 'preserve'

                elif categorical_mode == 'none':
                    # In 'none' mode, all columns are converted to numeric.
                    # String values will become NaN. No factorization.
                    data[col] = _numeric_representation
                    # _local_categorical_mappings remains empty for 'none'
            
            report['categorical_info'] = _local_categorical_mappings # Assign collected mappings to report

        elif data_type == 'x':
            # For X data, all columns are converted to numeric, coercing errors
            for col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            # report['categorical_info'] remains empty for data_type 'x' as it was cleared/initialized above

        # --- 6) Identify rows with NA values (POST type conversion) ---
        # This mask reflects NAs *after* all above conversions.
        # This is the mask that should be returned as the third element for potential synchronization by the caller.
        na_mask_after_conversions = data.isna().any(axis=1)
        report['na_handling']['na_detected_in_rows'] = bool(na_mask_after_conversions.any())

        # --- Handle NA policy internally for load_csv ---
        # This affects the 'data' DataFrame that will be returned by this function.
        if report['na_handling']['na_detected_in_rows']: # Check if there are any NAs to handle
            if na_policy == 'abort':
                # Find first NA for error reporting
                first_na_row_label_in_current_data = data.index[na_mask_after_conversions][0]
                first_na_col_name = data.loc[first_na_row_label_in_current_data].isna().idxmax()
                error_msg = (f"NA values detected after processing and na_policy is 'abort'. "
                            f"First NA found in column '{first_na_col_name}' (row label: {first_na_row_label_in_current_data}) "
                            f"in file {path}.")
                report['error'] = error_msg
                report['na_handling']['na_detected'] = True
                # Return None for data, and the na_mask_after_conversions (though caller might not use if error)
                return None, report, na_mask_after_conversions 

            elif na_policy == 'remove':
                # Update report fields about the rows that are about to be removed
                report['na_handling']['na_detected'] = True  # NAs were found and are being handled by removal
                report['na_handling']['nb_removed_rows'] = int(na_mask_after_conversions.sum())
                report['na_handling']['removed_rows_indices'] = data.index[na_mask_after_conversions].tolist()
                
                # Actually modify the 'data' DataFrame
                data = data[~na_mask_after_conversions].copy() # Use .copy() to avoid SettingWithCopyWarning
        
        # If na_policy == 'remove' but no NAs were detected, report fields remain at their initialized values (0, [], False)

        # --- 7) Final preparation of return values ---
        # 'final_shape' should reflect the shape of the data being returned.
        report['final_shape'] = data.shape 
        report['final_column_names'] = data.columns.tolist()
            
        # Return the 'data' (possibly with rows removed by this function if na_policy='remove')
        # and 'na_mask_after_conversions' (which is the mask *before* this function's internal NA removal).
        # data_array = data.to_numpy().astype(np.float32)
        return data, report, na_mask_after_conversions

    except FileNotFoundError as e:
        report['error'] = str(e)
        return None, report, None
    except ValueError as e:
        report['error'] = f"ValueError during processing: {e}"
        return None, report, None
    except Exception as e:
        # Catch any other unexpected error during loading/processing
        import traceback
        report['error'] = f"Unexpected error in load_csv: {e}\n{traceback.format_exc()}"
        return None, report, None
