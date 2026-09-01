from __future__ import annotations

import html
import json
import math
import os
import tempfile

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any


# ============================================================
# PADDLE / PADDLEX CLOUD STARTUP
# MUST BE SET BEFORE invoice_engine IMPORT
# ============================================================

os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["PADDLE_PDX_EAGER_INIT"] = "False"


# ============================================================
# IMPORTS
# ============================================================

import fitz
import pandas as pd
import streamlit as st

from PIL import Image

from invoice_engine import (
    EXPECTED_FIELDS,
    load_v3_production_engine,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Invoice Intelligence AI",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# HTML HELPER
# IMPORTANT:
# Use st.html(), NOT st.markdown(), for custom HTML.
# ============================================================

def render_html(markup: str) -> None:
    st.html(
        markup.strip()
    )


# ============================================================
# PROFESSIONAL CSS
# ============================================================

st.html(
    """
    <style>

    .stApp {
        background:
            radial-gradient(
                circle at top right,
                rgba(220, 38, 38, 0.06),
                transparent 34rem
            ),
            #f7f8fb;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }

    .hero {
        background:
            linear-gradient(
                135deg,
                #111827 0%,
                #1f2937 64%,
                #6b1f2a 130%
            );

        color: white;

        border-radius: 22px;

        padding:
            1.65rem
            1.8rem;

        margin-bottom:
            1.35rem;

        box-shadow:
            0 12px 35px
            rgba(17, 24, 39, 0.12);
    }

    .hero-kicker {
        color: #fca5a5;

        font-size: 0.76rem;

        font-weight: 800;

        text-transform: uppercase;

        letter-spacing: 0.14em;

        margin-bottom: 0.45rem;
    }

    .hero-title {
        color: white;

        font-size: 2.15rem;

        font-weight: 800;

        line-height: 1.15;

        margin: 0;
    }

    .hero-sub {
        color: #d1d5db;

        font-size: 0.98rem;

        max-width: 880px;

        margin-top: 0.7rem;

        line-height: 1.58;
    }

    .status-pill {
        display: inline-flex;

        align-items: center;

        gap: 0.4rem;

        margin-top: 0.95rem;

        padding:
            0.38rem
            0.78rem;

        border-radius: 999px;

        border:
            1px solid
            rgba(255, 255, 255, 0.18);

        background:
            rgba(255, 255, 255, 0.08);

        font-size: 0.82rem;

        color: #f3f4f6;
    }

    .status-dot {
        width: 8px;

        height: 8px;

        border-radius: 50%;

        background: #34d399;

        box-shadow:
            0 0 0 3px
            rgba(52, 211, 153, 0.15);

        display: inline-block;
    }

    .section-card {
        background:
            rgba(255,255,255,0.97);

        border:
            1px solid #e5e7eb;

        border-radius: 17px;

        padding:
            1.05rem
            1.1rem;

        box-shadow:
            0 5px 18px
            rgba(17, 24, 39, 0.035);

        margin-bottom:
            0.9rem;
    }

    .mini-label {
        color: #6b7280;

        font-size: 0.72rem;

        text-transform: uppercase;

        letter-spacing: 0.08em;

        font-weight: 800;
    }

    .mini-value {
        color: #111827;

        font-size: 1.06rem;

        font-weight: 700;

        margin-top: 0.28rem;

        word-break: break-word;
    }

    .not-detected {
        color: #9ca3af;

        font-style: italic;

        font-weight: 500;
    }

    .ready-card {
        background: #ffffff;

        border:
            1px solid #e5e7eb;

        border-radius: 16px;

        padding:
            1.1rem
            1.2rem;

        box-shadow:
            0 5px 18px
            rgba(17,24,39,0.035);
    }

    .ready-title {
        color: #047857;

        font-size: 0.75rem;

        font-weight: 800;

        letter-spacing: 0.09em;

        text-transform: uppercase;
    }

    .ready-text {
        color: #374151;

        font-size: 0.98rem;

        margin-top: 0.3rem;
    }

    .preview-page {
        color: #6b7280;

        font-size: 0.8rem;

        font-weight: 700;

        margin:
            0.2rem
            0
            0.5rem
            0;
    }

    div[data-testid="stMetric"] {
        background:
            rgba(255,255,255,0.97);

        border:
            1px solid #e5e7eb;

        border-radius: 15px;

        padding:
            0.85rem
            0.95rem;

        box-shadow:
            0 5px 18px
            rgba(17,24,39,0.035);
    }

    div[data-testid="stFileUploader"] {
        background: #ffffff;

        border:
            1px solid #e5e7eb;

        border-radius: 17px;

        padding:
            0.35rem
            0.8rem
            0.8rem
            0.8rem;
    }

    div[data-testid="stDataFrame"] {
        border:
            1px solid #e5e7eb;

        border-radius: 14px;

        overflow: hidden;
    }

    div.stButton > button,
    div.stDownloadButton > button {
        border-radius: 11px;

        font-weight: 700;

        min-height: 44px;
    }

    [data-testid="stSidebar"] {
        border-right:
            1px solid #e5e7eb;
    }

    footer {
        visibility: hidden;
    }

    </style>
    """
)


# ============================================================
# CONSTANTS
# ============================================================

FIELD_LABELS = {
    "VENDOR_NAME": "Vendor Name",
    "INVOICE_NUMBER": "Invoice Number",
    "INVOICE_DATE": "Invoice Date",
    "DUE_DATE": "Due Date",
    "CUSTOMER_NAME": "Customer Name",
    "ADDRESS": "Address",
    "CURRENCY": "Currency",
    "LINE_ITEM_DESC": "Line Item Description",
    "LINE_ITEM_QTY": "Line Item Quantity",
    "LINE_ITEM_UNIT_PRICE": "Line Item Unit Price",
    "LINE_ITEM_AMOUNT": "Line Item Amount",
    "TAX": "Tax",
    "DISCOUNT": "Discount",
    "SUBTOTAL": "Subtotal",
    "TOTAL_AMOUNT": "Total Amount",
    "PAYMENT_TERMS": "Payment Terms",
}


TABLE_KEYS = (
    "line_items",
    "line_item_table",
    "line_items_table",
    "material_table",
    "materials",
    "items",
)


VALUE_KEYS = (
    "value",
    "text",
    "prediction",
    "normalized_value",
    "final_value",
)


# ============================================================
# ENGINE
# ============================================================

@st.cache_resource(show_spinner=False)
def get_engine():

    return (
        load_v3_production_engine()
    )


def initialize_engine():

    try:

        with st.spinner(
            "Initializing Invoice AI production engine..."
        ):

            return get_engine()

    except Exception as error:

        st.error(
            "❌ Production engine initialization failed"
        )

        st.exception(
            error
        )

        st.stop()


# ============================================================
# JSON SAFE CONVERSION
# ============================================================

def make_json_safe(
    value: Any,
) -> Any:

    if value is None:
        return None


    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )


    if isinstance(
        value,
        Decimal,
    ):
        return float(
            value
        )


    if isinstance(
        value,
        (
            datetime,
            date,
        ),
    ):
        return value.isoformat()


    if isinstance(
        value,
        bytes,
    ):
        return value.decode(
            "utf-8",
            errors="replace",
        )


    if isinstance(
        value,
        dict,
    ):

        return {
            str(key):
                make_json_safe(
                    item
                )

            for key, item
            in value.items()
        }


    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):

        return [
            make_json_safe(
                item
            )

            for item
            in value
        ]


    if hasattr(
        value,
        "item",
    ):

        try:

            return make_json_safe(
                value.item()
            )

        except Exception:

            pass


    if hasattr(
        value,
        "tolist",
    ):

        try:

            return make_json_safe(
                value.tolist()
            )

        except Exception:

            pass


    if (
        isinstance(
            value,
            float,
        )
        and
        not math.isfinite(
            value
        )
    ):

        return None


    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):

        return value


    return str(
        value
    )


# ============================================================
# FIELD NORMALIZATION
# ============================================================

def unwrap_field_value(
    value: Any,
) -> Any:

    if isinstance(
        value,
        dict,
    ):

        for key in VALUE_KEYS:

            if key in value:

                return unwrap_field_value(
                    value[key]
                )


        if len(
            value
        ) == 1:

            return unwrap_field_value(
                next(
                    iter(
                        value.values()
                    )
                )
            )


        return value


    if isinstance(
        value,
        list,
    ):

        cleaned = [
            unwrap_field_value(
                item
            )

            for item
            in value

            if item is not None
        ]


        cleaned = [
            item

            for item
            in cleaned

            if str(
                item
            ).strip()
        ]


        if not cleaned:

            return (
                "NOT_DETECTED"
            )


        if len(
            cleaned
        ) == 1:

            return (
                cleaned[0]
            )


        return (
            " | ".join(
                str(
                    item
                )

                for item
                in cleaned
            )
        )


    if value is None:

        return (
            "NOT_DETECTED"
        )


    if (
        isinstance(
            value,
            str,
        )
        and
        not value.strip()
    ):

        return (
            "NOT_DETECTED"
        )


    return value


# ============================================================
# FIND FIELD DICTIONARY
# ============================================================

def field_dict_score(
    candidate: dict,
) -> int:

    return sum(
        1

        for field
        in EXPECTED_FIELDS

        if field
        in candidate
    )


def find_best_field_dict(
    payload: Any,
) -> dict:

    best = {}

    best_score = 0


    def walk(
        node: Any,
    ):

        nonlocal best, best_score


        if isinstance(
            node,
            dict,
        ):

            score = (
                field_dict_score(
                    node
                )
            )


            if score > best_score:

                best = node

                best_score = score


            for value in (
                node.values()
            ):

                walk(
                    value
                )


        elif isinstance(
            node,
            list,
        ):

            for value in node:

                walk(
                    value
                )


    walk(
        payload
    )


    return best


# ============================================================
# NORMALIZE 16 FIELDS
# ============================================================

def normalized_fields(
    payload: Any,
) -> dict:

    source = (
        find_best_field_dict(
            payload
        )
    )


    return {

        field:
            unwrap_field_value(
                source.get(
                    field,
                    "NOT_DETECTED",
                )
            )

        for field
        in EXPECTED_FIELDS

    }


# ============================================================
# FIND LINE ITEM TABLE
# ============================================================

def find_table(
    payload: Any,
):

    found = None


    def walk(
        node: Any,
    ):

        nonlocal found


        if found is not None:

            return


        if isinstance(
            node,
            dict,
        ):

            for key in TABLE_KEYS:

                if key not in node:

                    continue


                candidate = (
                    node[key]
                )


                if (
                    isinstance(
                        candidate,
                        list,
                    )
                    and
                    candidate
                    and
                    all(
                        isinstance(
                            row,
                            dict,
                        )
                        for row
                        in candidate
                    )
                ):

                    found = (
                        candidate
                    )

                    return


                if (
                    isinstance(
                        candidate,
                        dict,
                    )
                    and
                    candidate
                ):

                    found = (
                        candidate
                    )

                    return


            for value in (
                node.values()
            ):

                walk(
                    value
                )


        elif isinstance(
            node,
            list,
        ):

            if (
                node
                and
                all(
                    isinstance(
                        row,
                        dict,
                    )
                    for row
                    in node
                )
            ):

                keys = {

                    str(
                        key
                    ).lower()

                    for row
                    in node

                    for key
                    in row.keys()

                }


                signals = (
                    "description",
                    "desc",
                    "qty",
                    "quantity",
                    "rate",
                    "price",
                    "amount",
                    "material",
                    "item",
                )


                if any(
                    token in key

                    for key
                    in keys

                    for token
                    in signals
                ):

                    found = (
                        node
                    )

                    return


            for value in node:

                walk(
                    value
                )


    walk(
        payload
    )


    return found


# ============================================================
# DISPLAY VALUE
# ============================================================

def display_value(
    value: Any,
) -> str:

    if value is None:

        return (
            "NOT_DETECTED"
        )


    if isinstance(
        value,
        (
            dict,
            list,
        ),
    ):

        return (
            json.dumps(
                make_json_safe(
                    value
                ),
                ensure_ascii=False,
            )
        )


    text = (
        str(
            value
        )
        .strip()
    )


    if (
        not text
        or
        text.lower()
        in {
            "none",
            "null",
            "nan",
        }
    ):

        return (
            "NOT_DETECTED"
        )


    return text


# ============================================================
# PDF PREVIEW
# ============================================================

def preview_pdf(
    uploaded_file,
):

    document = (
        fitz.open(
            stream=
                uploaded_file.getvalue(),
            filetype="pdf",
        )
    )


    try:

        if (
            document.page_count
            ==
            0
        ):

            st.warning(
                "PDF contains no pages."
            )

            return


        st.caption(
            f"{document.page_count} page"
            f"{'s' if document.page_count != 1 else ''}"
            " • rendered directly in the app"
        )


        for page_number in range(
            document.page_count
        ):

            page = (
                document.load_page(
                    page_number
                )
            )


            pixmap = (
                page.get_pixmap(
                    matrix=
                        fitz.Matrix(
                            1.35,
                            1.35,
                        ),
                    alpha=False,
                )
            )


            render_html(
                f"""
                <div class="preview-page">
                    Page {page_number + 1}
                </div>
                """
            )


            st.image(
                pixmap.tobytes(
                    "png"
                ),
                use_container_width=True,
            )


            if (
                page_number
                <
                document.page_count - 1
            ):

                st.divider()


    finally:

        document.close()


# ============================================================
# IMAGE PREVIEW
# ============================================================

def preview_image(
    uploaded_file,
):

    uploaded_file.seek(
        0
    )


    image = (
        Image.open(
            uploaded_file
        )
    )


    try:

        st.image(
            image,
            caption=
                uploaded_file.name,
            use_container_width=True,
        )


    finally:

        try:

            image.close()

        except Exception:

            pass


# ============================================================
# GENERAL PREVIEW
# ============================================================

def preview_uploaded_file(
    uploaded_file,
):

    suffix = (

        Path(
            uploaded_file.name
        )
        .suffix
        .lower()

    )


    try:

        if suffix == ".pdf":

            preview_pdf(
                uploaded_file
            )

        else:

            preview_image(
                uploaded_file
            )


    except Exception as error:

        st.warning(
            "Preview could not be displayed."
        )


        st.caption(
            f"Preview error: {error}"
        )


# ============================================================
# RUN INFERENCE
# ============================================================

def process_upload(
    uploaded_file,
    process_invoice_final,
):

    suffix = (

        Path(
            uploaded_file.name
        )
        .suffix
        .lower()

        or

        ".bin"

    )


    temp_path = None


    try:

        with tempfile.NamedTemporaryFile(

            mode="wb",

            suffix=suffix,

            delete=False,

        ) as temp_file:

            temp_file.write(
                uploaded_file.getvalue()
            )


            temp_path = (
                Path(
                    temp_file.name
                )
            )


        result = (
            process_invoice_final(
                str(
                    temp_path
                )
            )
        )


        return (
            make_json_safe(
                result
            )
        )


    finally:

        if (
            temp_path
            is not None
        ):

            try:

                temp_path.unlink(
                    missing_ok=True
                )

            except Exception:

                pass


# ============================================================
# SUMMARY CARDS
# ============================================================

def render_summary(
    fields: dict,
):

    st.subheader(
        "Invoice Summary"
    )


    cards = [
        (
            "Vendor",
            display_value(
                fields.get(
                    "VENDOR_NAME"
                )
            ),
        ),

        (
            "Invoice #",
            display_value(
                fields.get(
                    "INVOICE_NUMBER"
                )
            ),
        ),

        (
            "Invoice Date",
            display_value(
                fields.get(
                    "INVOICE_DATE"
                )
            ),
        ),

        (
            "Total Amount",
            display_value(
                fields.get(
                    "TOTAL_AMOUNT"
                )
            ),
        ),
    ]


    columns = (
        st.columns(
            4
        )
    )


    for (
        column,
        (
            label,
            value,
        ),
    ) in zip(
        columns,
        cards,
    ):

        with column:

            safe_label = (
                html.escape(
                    str(
                        label
                    )
                )
            )


            if (
                value
                ==
                "NOT_DETECTED"
            ):

                safe_value = (
                    '<span class="not-detected">'
                    'NOT_DETECTED'
                    '</span>'
                )

            else:

                safe_value = (
                    html.escape(
                        str(
                            value
                        )
                    )
                )


            render_html(
                f"""
                <div class="section-card">
                    <div class="mini-label">
                        {safe_label}
                    </div>

                    <div class="mini-value">
                        {safe_value}
                    </div>
                </div>
                """
            )


# ============================================================
# FIELD TABLE
# ============================================================

def render_fields(
    fields: dict,
):

    rows = [
        {
            "Field":
                FIELD_LABELS.get(
                    field,
                    field,
                ),

            "Value":
                display_value(
                    fields.get(
                        field
                    )
                ),
        }

        for field
        in EXPECTED_FIELDS
    ]


    dataframe = (
        pd.DataFrame(
            rows
        )
    )


    st.dataframe(
        dataframe,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Field":
                st.column_config.TextColumn(
                    "Field",
                    width="medium",
                ),

            "Value":
                st.column_config.TextColumn(
                    "Extracted Value",
                    width="large",
                ),
        },
    )


# ============================================================
# LINE ITEMS
# ============================================================

def render_line_items(
    result,
):

    table = (
        find_table(
            result
        )
    )


    if table is None:

        st.info(
            "No structured line-item table "
            "was returned for this invoice."
        )

        return


    if isinstance(
        table,
        dict,
    ):

        try:

            dataframe = (
                pd.DataFrame(
                    table
                )
            )

        except Exception:

            dataframe = (
                pd.DataFrame(
                    [
                        {
                            "Key":
                                key,

                            "Value":
                                display_value(
                                    value
                                ),
                        }

                        for key, value
                        in table.items()
                    ]
                )
            )


    else:

        dataframe = (
            pd.DataFrame(
                table
            )
        )


    if dataframe.empty:

        st.info(
            "No structured line-item rows were returned."
        )

        return


    st.dataframe(
        dataframe,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# RESULTS
# ============================================================

def render_results(
    result,
):

    fields = (
        normalized_fields(
            result
        )
    )


    st.success(
        "✅ Invoice processed successfully"
    )


    render_summary(
        fields
    )


    (
        tab_fields,
        tab_items,
        tab_structured,
        tab_json,
    ) = st.tabs(
        [
            "Extracted Fields",
            "Line Items",
            "Structured Result",
            "Raw JSON",
        ]
    )


    with tab_fields:

        st.markdown(
            "#### 16-field extraction"
        )

        render_fields(
            fields
        )


    with tab_items:

        st.markdown(
            "#### Material / line-item extraction"
        )

        render_line_items(
            result
        )


    with tab_structured:

        st.markdown(
            "#### Complete structured output"
        )

        st.write(
            result
        )


    with tab_json:

        st.markdown(
            "#### Raw JSON"
        )

        st.json(
            result
        )


    json_bytes = (
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
        .encode(
            "utf-8"
        )
    )


    st.download_button(
        "⬇ Download JSON Result",
        data=json_bytes,
        file_name=
            "invoice_extraction.json",
        mime=
            "application/json",
    )


# ============================================================
# START ENGINE
# ============================================================

engine = (
    initialize_engine()
)


process_invoice_final = (
    engine[
        "process_invoice_final"
    ]
)


# ============================================================
# HERO
# ============================================================

render_html(
    """
    <div class="hero">

        <div class="hero-kicker">
            Production Document Intelligence
        </div>

        <div class="hero-title">
            🧾 Invoice Intelligence AI
        </div>

        <div class="hero-sub">
            Upload an invoice and extract structured invoice fields,
            line items, tax information and financial values using
            the verified Invoice AI V3 production pipeline.
        </div>

        <div class="status-pill">

            <span class="status-dot"></span>

            Production engine online

        </div>

    </div>
    """
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🧾 Invoice AI V3"
    )


    st.caption(
        "Production Engine"
    )


    st.success(
        "● Engine Ready"
    )


    left, right = (
        st.columns(
            2
        )
    )


    with left:

        st.metric(
            "Fields",
            len(
                engine.get(
                    "fields",
                    EXPECTED_FIELDS,
                )
            ),
        )


    with right:

        st.metric(
            "BIO Labels",
            len(
                engine.get(
                    "label_list",
                    [],
                )
            ),
        )


    st.metric(
        "Parameters",
        f"{engine.get('parameter_count', 0):,}",
    )


    st.metric(
        "Device",
        str(
            engine.get(
                "device",
                "cpu",
            )
        ).upper(),
    )


    with st.expander(
        "Engine Details"
    ):

        st.write(
            {
                "Architecture":
                    engine[
                        "model"
                    ].__class__.__name__,

                "Runtime":
                    "V3 + V6.1",

                "Entry Point":
                    "process_invoice_final(path)",

                "OCR":
                    "Lazy fallback",

                "Model Reload":
                    "Disabled / Cached",
            }
        )


    st.divider()


    if st.button(
        "Clear Current Result",
        use_container_width=True,
    ):

        st.session_state.pop(
            "invoice_result",
            None,
        )


        st.session_state.pop(
            "invoice_result_name",
            None,
        )


        st.rerun()


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    "### 1. Upload Invoice"
)


uploaded_file = (
    st.file_uploader(
        "Choose a PDF or invoice image",

        type=[
            "pdf",
            "png",
            "jpg",
            "jpeg",
            "webp",
            "bmp",
            "tiff",
            "tif",
        ],

        help=(
            "Supported formats: "
            "PDF, PNG, JPG, JPEG, "
            "WEBP, BMP and TIFF."
        ),
    )
)


# ============================================================
# EMPTY STATE
# ============================================================

if uploaded_file is None:

    render_html(
        """
        <div class="ready-card">

            <div class="ready-title">
                Ready
            </div>

            <div class="ready-text">
                Upload an invoice to begin extraction.
            </div>

        </div>
        """
    )


    st.stop()


# ============================================================
# FILE INFORMATION
# ============================================================

file_size_mb = (

    len(
        uploaded_file.getvalue()
    )

    /

    (
        1024
        *
        1024
    )
)


file_col, size_col = (
    st.columns(
        [
            3,
            1,
        ]
    )
)


with file_col:

    st.caption(
        "FILE"
    )


    st.markdown(
        f"**{uploaded_file.name}**"
    )


with size_col:

    st.caption(
        "SIZE"
    )


    st.markdown(
        f"**{file_size_mb:.2f} MB**"
    )


# ============================================================
# PREVIEW + EXTRACT
# ============================================================

preview_col, action_col = (
    st.columns(
        [
            2.15,
            1,
        ],
        gap="large",
    )
)


with preview_col:

    st.markdown(
        "### 2. Invoice Preview"
    )


    with st.container(
        border=True
    ):

        preview_uploaded_file(
            uploaded_file
        )


with action_col:

    st.markdown(
        "### 3. Extract"
    )


    st.write(
        "The production engine automatically "
        "chooses native extraction or OCR "
        "based on the invoice."
    )


    st.markdown(
        "**Production Pipeline**"
    )


    st.markdown(
        """
        - Native PDF extraction
        - OCR fallback when required
        - LayoutLMv3 inference
        - Overlap-safe chunking
        - BIO field extraction
        - Anchor / regex recovery
        - GST reconciliation
        - Financial validation
        - Material / line-table extraction
        - V3 / V6.1 protection layers
        """
    )


    process_clicked = (
        st.button(
            "🚀 Process Invoice",
            type="primary",
            use_container_width=True,
        )
    )


    if process_clicked:

        try:

            with st.spinner(
                "Analyzing invoice..."
            ):

                result = (
                    process_upload(
                        uploaded_file,
                        process_invoice_final,
                    )
                )


            st.session_state[
                "invoice_result"
            ] = result


            st.session_state[
                "invoice_result_name"
            ] = (
                uploaded_file.name
            )


        except Exception as error:

            st.error(
                "❌ Invoice processing failed"
            )


            st.exception(
                error
            )


# ============================================================
# STORED RESULT
# ============================================================

current_result = (
    st.session_state.get(
        "invoice_result"
    )
)


current_name = (
    st.session_state.get(
        "invoice_result_name"
    )
)


if (
    current_result is not None
    and
    current_name
    ==
    uploaded_file.name
):

    st.divider()


    st.markdown(
        "## 4. Extraction Results"
    )


    render_results(
        current_result
    )