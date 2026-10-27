"""
app.py
------
Flask API server for ReviewAnalytics.
Provides endpoints for file upload, dataset preview, and statistical analysis.

Run with:
    python app.py
"""

import os
import io
import json
import uuid
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS
from werkzeug.utils import secure_filename

import analysis

# ── App setup ────────────────────────────────────────────────────────────────

class NumpyJSONProvider(DefaultJSONProvider):
    def default(self, o):
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, (np.integer, np.int64)):
            return int(o)
        if isinstance(o, (np.floating, np.float64)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)

app = Flask(__name__)
app.json = NumpyJSONProvider(app)

# Allow requests from Next.js dev server (port 3000)
CORS(app, origins=["http://localhost:3000", "http://127.0.0.1:3000"])

# Upload configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_EXTENSIONS = {"csv", "xlsx"}
MAX_FILE_SIZE_MB = 20
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# In-memory session store: maps session_id → pandas DataFrame
# (Single-user local app — no database needed)
SESSION_DATA = {}


# ── Utility functions ────────────────────────────────────────────────────────

def allowed_file(filename):
    """Check that the uploaded file has a supported extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def df_to_preview(df, rows=10):
    """Convert the first N rows of a DataFrame to a JSON-friendly list of dicts."""
    preview = df.head(rows).copy()
    # Replace NaN with None so JSON serialisation works
    preview = preview.where(pd.notnull(preview), other=None)
    return preview.to_dict(orient="records")


def detect_column_types(df):
    """
    Return a list of dicts describing each column:
    { name, dtype, sample_values, missing_count }
    """
    info = []
    for col in df.columns:
        series = df[col]
        numeric_series = pd.to_numeric(series, errors="coerce")
        if numeric_series.notna().sum() > len(series) * 0.8:
            col_type = "numeric"
        else:
            col_type = "categorical"

        info.append({
            "name": col,
            "dtype": col_type,
            "original_dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "unique_count": int(series.nunique()),
            "sample_values": series.dropna().astype(str).head(5).tolist(),
        })
    return info


def success_response(data):
    return jsonify({"status": "success", **data})


def error_response(message, code=400):
    return jsonify({"status": "error", "message": message}), code


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/api/health", methods=["GET"])
def health():
    """Simple health check so the frontend can verify the backend is running."""
    return success_response({"message": "ReviewAnalytics backend is running."})


@app.route("/api/upload", methods=["POST"])
def upload_file():
    """
    Accept a CSV or XLSX file upload.
    Validates the file, loads it into a DataFrame, and returns a preview.
    """
    if "file" not in request.files:
        return error_response("No file was included in the request.")

    file = request.files["file"]
    if file.filename == "":
        return error_response("No file was selected.")
    if not allowed_file(file.filename):
        return error_response("Unsupported file type. Please upload a CSV or XLSX file.")

    filename = secure_filename(file.filename)
    ext = filename.rsplit(".", 1)[1].lower()

    try:
        file_bytes = file.read()

        if ext == "csv":
            df = pd.read_csv(io.BytesIO(file_bytes))
        else:
            df = pd.read_excel(io.BytesIO(file_bytes), engine="openpyxl")

    except Exception as exc:
        return error_response(f"Could not parse the uploaded file: {str(exc)}")

    if df.empty or len(df.columns) == 0:
        return error_response("The uploaded file appears to be empty.")

    if len(df) > 50000:
        return error_response(
            f"Dataset has {len(df)} rows — maximum supported is 50,000 rows."
        )

    # Assign a session ID so subsequent /api/analyze calls reference the same data
    session_id = str(uuid.uuid4())
    SESSION_DATA[session_id] = df

    column_info = detect_column_types(df)
    preview = df_to_preview(df, rows=10)
    summary = analysis.calculate_summary(df)

    return success_response({
        "session_id": session_id,
        "filename": filename,
        "rows": len(df),
        "columns": len(df.columns),
        "column_info": column_info,
        "preview": preview,
        "summary": summary,
    })


@app.route("/api/sample", methods=["GET"])
def load_sample():
    """Load the bundled sample_data.csv and return it as a dataset session."""
    sample_path = os.path.join(os.path.dirname(__file__), "sample_data.csv")
    if not os.path.exists(sample_path):
        return error_response("Sample dataset not found on the server.")

    try:
        df = pd.read_csv(sample_path)
    except Exception as exc:
        return error_response(f"Could not read sample dataset: {str(exc)}")

    session_id = str(uuid.uuid4())
    SESSION_DATA[session_id] = df

    column_info = detect_column_types(df)
    preview = df_to_preview(df, rows=10)
    summary = analysis.calculate_summary(df)

    return success_response({
        "session_id": session_id,
        "filename": "sample_data.csv",
        "rows": len(df),
        "columns": len(df.columns),
        "column_info": column_info,
        "preview": preview,
        "summary": summary,
    })


@app.route("/api/dataset/preview", methods=["GET"])
def dataset_preview():
    """Return a preview of the currently loaded dataset for a given session."""
    session_id = request.args.get("session_id")
    if not session_id or session_id not in SESSION_DATA:
        return error_response("No dataset loaded. Please upload a file first.", 404)

    df = SESSION_DATA[session_id]
    return success_response({
        "rows": len(df),
        "columns": len(df.columns),
        "preview": df_to_preview(df, rows=10),
        "column_info": detect_column_types(df),
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Perform a statistical analysis on the loaded dataset.

    Expected JSON body:
    {
        "session_id": "...",
        "analysis_type": "mean" | "median" | "frequency" | "correlation" | "regression" | "chi_square",
        "params": {
            // Varies by analysis type (see below)
        }
    }

    Analysis-specific params:
        mean / median / frequency : { "column": "col_name" }
        correlation               : { "col_x": "...", "col_y": "...", "method": "pearson"|"spearman" }
        regression                : { "dep_col": "...", "indep_col": "..." }
        chi_square                : { "col1": "...", "col2": "...", "alpha": 0.05 }
    """
    body = request.get_json(silent=True)
    if not body:
        return error_response("Request body must be JSON.")

    session_id = body.get("session_id")
    if not session_id or session_id not in SESSION_DATA:
        return error_response("Session not found. Please upload a dataset first.", 404)

    df = SESSION_DATA[session_id]
    analysis_type = body.get("analysis_type", "").strip().lower()
    params = body.get("params", {})

    if not analysis_type:
        return error_response("'analysis_type' is required.")

    # ── Route to the correct analysis function ───────────────────────────────
    result = None

    if analysis_type == "mean":
        column = params.get("column")
        if not column:
            return error_response("'column' parameter is required for mean analysis.")
        result = analysis.calculate_mean(df, column)

    elif analysis_type == "median":
        column = params.get("column")
        if not column:
            return error_response("'column' parameter is required for median analysis.")
        result = analysis.calculate_median(df, column)

    elif analysis_type == "frequency":
        column = params.get("column")
        if not column:
            return error_response("'column' parameter is required for frequency analysis.")
        result = analysis.calculate_frequency(df, column)

    elif analysis_type == "correlation":
        col_x = params.get("col_x")
        col_y = params.get("col_y")
        method = params.get("method", "pearson")
        if not col_x or not col_y:
            return error_response("'col_x' and 'col_y' are required for correlation analysis.")
        if col_x == col_y:
            return error_response("Please select two different columns for correlation.")
        result = analysis.calculate_correlation(df, col_x, col_y, method)

    elif analysis_type == "regression":
        dep_col = params.get("dep_col")
        indep_col = params.get("indep_col")
        if not dep_col or not indep_col:
            return error_response("'dep_col' and 'indep_col' are required for regression analysis.")
        if dep_col == indep_col:
            return error_response("Dependent and independent columns must be different.")
        result = analysis.calculate_regression(df, dep_col, indep_col)

    elif analysis_type == "chi_square":
        col1 = params.get("col1")
        col2 = params.get("col2")
        alpha = float(params.get("alpha", 0.05))
        if not col1 or not col2:
            return error_response("'col1' and 'col2' are required for chi-square analysis.")
        if col1 == col2:
            return error_response("Please select two different columns for chi-square.")
        result = analysis.calculate_chi_square(df, col1, col2, alpha)

    else:
        return error_response(
            f"Unknown analysis type '{analysis_type}'. "
            "Supported types: mean, median, frequency, correlation, regression, chi_square."
        )

    # If the analysis function returned an error key, send a 400
    if result and "error" in result:
        return error_response(result["error"])

    return success_response({"result": result})


# ── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print(" ReviewAnalytics Backend — Flask API")
    print(" Running at http://localhost:5000")
    print("=" * 55)
    app.run(debug=True, host="0.0.0.0", port=5000)
