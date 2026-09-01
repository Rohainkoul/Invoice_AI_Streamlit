import base64
import json
import math
import os
import tempfile

from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import streamlit as st


# ============================================================
# PADDLE / PADDLEX STARTUP
# ============================================================

# Avoid the unnecessary Paddle model-source connectivity check
# during every fresh application launch.
os.environ.setdefault(
    "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK",
    "True",
)


from invoice_engine import load_v3_production_engine


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
# CONSTANTS
# ============================================================

SUPPORTED_TYPES = [
    "pdf",
    "png",
    "jpg",
    "jpeg",
    "webp",
    "bmp",
    "tiff",
    "tif",
]


IMAGE_TYPES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
}


EXPECTED_FIELD_ORDER = [
    "VENDOR_NAME",
    "INVOICE_NUMBER",
    "INVOICE_DATE",
    "DUE_DATE",
    "CUSTOMER_NAME",
    "ADDRESS",
    "CURRENCY",
    "LINE_ITEM_DESC",
    "LINE_ITEM_QTY",
    "LINE_ITEM_UNIT_PRICE",
    "LINE_ITEM_AMOUNT",
    "TAX",
    "DISCOUNT",
    "SUBTOTAL",
    "TOTAL_AMOUNT",
    "PAYMENT_TERMS",
]


# ============================================================
# ENGINE CACHE
# ============================================================

@st.cache_resource(
    show_spinner=False
)
def get_engine():

    return (
        load_v3_production_engine()
    )


# ============================================================
# JSON-SAFE CONVERSION
# ============================================================

def to_json_safe(
    value,
):
    """
    Convert runtime output into JSON-safe Python objects.

    Handles:
    - Path
    - Decimal
    - datetime/date
    - NumPy scalars/arrays
    - Torch scalars/tensors
    - tuples/sets
    - nested dictionaries/lists
    """

    if value is None:
        return None


    if isinstance(
        value,
        (
            str,
            int,
            bool,
        ),
    ):

        return value


    if isinstance(
        value,
        float,
    ):

        if (
            math.isnan(value)
            or
            math.isinf(value)
        ):

            return str(value)

        return value


    if isinstance(
        value,
        Decimal,
    ):

        return float(value)


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
        Path,
    ):

        return str(value)


    if isinstance(
        value,
        dict,
    ):

        return {

            str(key):
                to_json_safe(item)

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

            to_json_safe(item)

            for item
            in value

        ]


    # NumPy / Torch scalar.
    if hasattr(
        value,
        "item",
    ):

        try:

            return (
                to_json_safe(
                    value.item()
                )
            )

        except Exception:

            pass


    # NumPy / Torch array/tensor.
    if hasattr(
        value,
        "tolist",
    ):

        try:

            return (
                to_json_safe(
                    value.tolist()
                )
            )

        except Exception:

            pass


    return str(value)


def create_json_bytes(
    data,
):

    safe_data = (
        to_json_safe(
            data
        )
    )


    return json.dumps(
        safe_data,
        indent=2,
        ensure_ascii=False,
    ).encode(
        "utf-8"
    )


# ============================================================
# RESULT FIELD DISCOVERY
# ============================================================

def find_best_field_dict(
    result,
):
    """
    Finds the dictionary containing the 16 invoice fields.

    This deliberately supports multiple output shapes so the
    Streamlit UI does not become tightly coupled to one runtime
    wrapper format.
    """

    if not isinstance(
        result,
        dict,
    ):

        return None


    expected = set(
        EXPECTED_FIELD_ORDER
    )


    # --------------------------------------------------------
    # CASE 1
    # Fields are directly at root.
    # --------------------------------------------------------

    root_keys = {

        str(key).upper()

        for key
        in result.keys()

    }


    direct_matches = (

        expected
        &
        root_keys

    )


    if len(
        direct_matches
    ) >= 3:

        return result


    # --------------------------------------------------------
    # CASE 2
    # Common nested names.
    # --------------------------------------------------------

    preferred_keys = [

        "fields",
        "extracted_fields",
        "invoice_fields",
        "header_fields",
        "final_fields",
        "result",
        "data",
        "invoice",

    ]


    lower_lookup = {

        str(key).lower():
            key

        for key
        in result.keys()

    }


    for preferred_key in (
        preferred_keys
    ):

        actual_key = (
            lower_lookup.get(
                preferred_key
            )
        )


        if actual_key is None:
            continue


        candidate = (
            result.get(
                actual_key
            )
        )


        if not isinstance(
            candidate,
            dict,
        ):

            continue


        candidate_keys = {

            str(key).upper()

            for key
            in candidate.keys()

        }


        if (
            expected
            &
            candidate_keys
        ):

            return candidate


    # --------------------------------------------------------
    # CASE 3
    # Search one level deep.
    # --------------------------------------------------------

    for candidate in (
        result.values()
    ):

        if not isinstance(
            candidate,
            dict,
        ):

            continue


        candidate_keys = {

            str(key).upper()

            for key
            in candidate.keys()

        }


        if (
            expected
            &
            candidate_keys
        ):

            return candidate


    return None


def normalize_field_lookup(
    field_dictionary,
):

    if not isinstance(
        field_dictionary,
        dict,
    ):

        return {}


    return {

        str(key).upper():
            value

        for key, value
        in field_dictionary.items()

    }


def display_value(
    value,
):

    if value is None:

        return "NOT_DETECTED"


    if isinstance(
        value,
        str,
    ):

        cleaned = (
            value.strip()
        )


        if not cleaned:

            return (
                "NOT_DETECTED"
            )


        return cleaned


    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):

        if not value:

            return (
                "NOT_DETECTED"
            )


        return ", ".join(
            str(item)
            for item
            in value
        )


    return str(value)


# ============================================================
# LINE ITEM DISCOVERY
# ============================================================

def find_tabular_data(
    result,
):

    if not isinstance(
        result,
        dict,
    ):

        return (
            None,
            None,
        )


    possible_names = [

        "line_items",
        "items",
        "material_table",
        "materials",
        "table",
        "tables",
        "line_item_table",
        "invoice_lines",

    ]


    lower_lookup = {

        str(key).lower():
            key

        for key
        in result.keys()

    }


    for possible_name in (
        possible_names
    ):

        real_key = (
            lower_lookup.get(
                possible_name
            )
        )


        if real_key is None:
            continue


        candidate = (
            result.get(
                real_key
            )
        )


        # ----------------------------------------------------
        # Direct list of rows.
        # ----------------------------------------------------

        if isinstance(
            candidate,
            list,
        ):

            if (
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

                return (
                    str(real_key),
                    candidate,
                )


        # ----------------------------------------------------
        # One additional nesting level.
        # ----------------------------------------------------

        if isinstance(
            candidate,
            dict,
        ):

            for (
                nested_key,
                nested_value,
            ) in candidate.items():

                if (

                    isinstance(
                        nested_value,
                        list,
                    )

                    and
                    nested_value

                    and
                    all(
                        isinstance(
                            row,
                            dict,
                        )
                        for row
                        in nested_value
                    )

                ):

                    return (

                        (
                            f"{real_key}."
                            f"{nested_key}"
                        ),

                        nested_value,

                    )


    return (
        None,
        None,
    )


# ============================================================
# PDF / IMAGE PREVIEW
# ============================================================

def render_pdf_preview(
    file_bytes,
):

    encoded_pdf = (

        base64.b64encode(
            file_bytes
        )
        .decode(
            "utf-8"
        )

    )


    pdf_html = f"""

        <iframe
            src="data:application/pdf;base64,{encoded_pdf}"
            width="100%"
            height="700"
            type="application/pdf"
            style="
                border: 1px solid #dddddd;
                border-radius: 8px;
            "
        >
        </iframe>

    """


    st.markdown(
        pdf_html,
        unsafe_allow_html=True,
    )


def render_file_preview(
    uploaded_file,
):

    extension = (

        Path(
            uploaded_file.name
        )
        .suffix
        .lower()

    )


    file_bytes = (
        uploaded_file.getvalue()
    )


    if extension == ".pdf":

        render_pdf_preview(
            file_bytes
        )


    elif extension in (
        IMAGE_TYPES
    ):

        st.image(
            file_bytes,
            caption=
                uploaded_file.name,
            use_container_width=True,
        )


# ============================================================
# PROCESS UPLOADED INVOICE
# ============================================================

def process_uploaded_invoice(
    uploaded_file,
    process_function,
):

    extension = (

        Path(
            uploaded_file.name
        )
        .suffix
        .lower()

    )


    temporary_path = None


    try:

        # Runtime expects a physical file path.
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as temporary_file:

            temporary_file.write(
                uploaded_file.getvalue()
            )


            temporary_path = (
                temporary_file.name
            )


        # ====================================================
        # VERIFIED FINAL PRODUCTION ENTRY POINT
        # ====================================================

        result = process_function(
            temporary_path
        )


        return result


    finally:

        # Always remove temporary invoice copy.
        if temporary_path:

            try:

                os.remove(
                    temporary_path
                )

            except OSError:

                pass


# ============================================================
# INITIALIZE FINAL ENGINE
# ============================================================

try:

    with st.spinner(
        "Initializing Invoice AI V3..."
    ):

        engine = (
            get_engine()
        )


except Exception as error:

    st.title(
        "🧾 Invoice Intelligence AI"
    )


    st.error(
        "❌ Production engine initialization failed"
    )


    st.exception(
        error
    )


    st.stop()


# ============================================================
# VERIFIED FINAL ENTRY POINT
# ============================================================

process_invoice_final = (
    engine[
        "process_invoice_final"
    ]
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "🧾 Invoice AI V3"
    )


    st.success(
        "● Production Engine Ready"
    )


    st.metric(
        "Parameters",
        f"{engine['parameter_count']:,}",
    )


    field_column, label_column = (
        st.columns(2)
    )


    field_column.metric(
        "Fields",
        len(
            engine["fields"]
        ),
    )


    label_column.metric(
        "BIO Labels",
        len(
            engine[
                "label_list"
            ]
        ),
    )


    st.metric(
        "Device",
        str(
            engine[
                "device"
            ]
        ).upper(),
    )


    # --------------------------------------------------------
    # ENGINE DETAILS
    # --------------------------------------------------------

    with st.expander(
        "Engine Details"
    ):

        st.write(
            "**Architecture**"
        )


        st.code(
            engine[
                "model"
            ].__class__.__name__,
            language=None,
        )


        st.write(
            "**Runtime Loader**"
        )


        st.code(
            str(
                engine[
                    "runtime_loader"
                ]
            ),
            language=None,
        )


        st.write(
            "**Final Entry Point**"
        )


        st.code(
            "process_invoice_final(path)",
            language=None,
        )


        st.write(
            "**Runtime Stack**"
        )


        st.caption(
            "V3 Model → Base Processor → "
            "V3/V4/V5 → V6 → V6.1 → "
            "Final V3 Integration"
        )


    # --------------------------------------------------------
    # CLEAR RESULT
    # --------------------------------------------------------

    if st.button(
        "🗑️ Clear Current Result",
        use_container_width=True,
    ):

        for key in [

            "invoice_result",
            "processed_filename",

        ]:

            st.session_state.pop(
                key,
                None,
            )


        st.rerun()


# ============================================================
# MAIN HEADER
# ============================================================

st.title(
    "🧾 Invoice Intelligence AI"
)


st.caption(
    "Invoice AI V3 — "
    "Final Production Inference System"
)


st.success(
    "🔥 Production engine online"
)


# ============================================================
# UPLOAD AREA
# ============================================================

st.header(
    "1. Upload Invoice"
)


uploaded_file = (
    st.file_uploader(
        "Select an invoice",
        type=
            SUPPORTED_TYPES,
        help=(
            "Supported formats: "
            "PDF, PNG, JPG, JPEG, "
            "WEBP, BMP and TIFF"
        ),
    )
)


if uploaded_file is None:

    st.info(
        "Upload an invoice above to begin extraction."
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
    st.columns(2)
)


file_col.metric(
    "File",
    uploaded_file.name,
)


size_col.metric(
    "Size",
    f"{file_size_mb:.2f} MB",
)


# ============================================================
# PREVIEW + PROCESS CONTROLS
# ============================================================

preview_column, process_column = (

    st.columns(
        [
            1.45,
            0.80,
        ],
        gap="large",
    )

)


# ============================================================
# PREVIEW
# ============================================================

with preview_column:

    st.header(
        "2. Invoice Preview"
    )


    with st.container(
        border=True
    ):

        render_file_preview(
            uploaded_file
        )


# ============================================================
# PROCESS CONTROLS
# ============================================================

with process_column:

    st.header(
        "3. Extract"
    )


    st.write(
        "The production engine automatically chooses "
        "native extraction or OCR based on the invoice."
    )


    st.markdown(
        """
**Production Pipeline**

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


# ============================================================
# RUN INFERENCE
# ============================================================

if process_clicked:

    try:

        with st.spinner(
            "Processing invoice... "
            "OCR-heavy documents may take longer on CPU."
        ):

            raw_result = (
                process_uploaded_invoice(
                    uploaded_file,
                    process_invoice_final,
                )
            )


        safe_result = (
            to_json_safe(
                raw_result
            )
        )


        st.session_state[
            "invoice_result"
        ] = safe_result


        st.session_state[
            "processed_filename"
        ] = uploaded_file.name


        st.success(
            "✅ Invoice processed successfully"
        )


    except Exception as error:

        st.error(
            "❌ Invoice processing failed"
        )


        st.exception(
            error
        )


        st.stop()


# ============================================================
# GET EXISTING RESULT
# ============================================================

result = (
    st.session_state.get(
        "invoice_result"
    )
)


processed_filename = (
    st.session_state.get(
        "processed_filename"
    )
)


if result is None:

    st.stop()


# ============================================================
# DIFFERENT UPLOAD WARNING
# ============================================================

if (

    processed_filename

    and

    processed_filename
    !=
    uploaded_file.name

):

    st.warning(
        "The result below belongs to "
        f"`{processed_filename}`. "
        "Click **Process Invoice** to process "
        "the newly uploaded file."
    )


# ============================================================
# RESULTS
# ============================================================

st.divider()


st.header(
    "4. Extraction Results"
)


# ============================================================
# EXTRACT MAIN FIELD DICTIONARY
# ============================================================

field_dictionary = (
    find_best_field_dict(
        result
    )
)


field_lookup = (
    normalize_field_lookup(
        field_dictionary
    )
)


# ============================================================
# QUICK SUMMARY CARDS
# ============================================================

if field_lookup:

    vendor = display_value(
        field_lookup.get(
            "VENDOR_NAME"
        )
    )


    invoice_number = display_value(
        field_lookup.get(
            "INVOICE_NUMBER"
        )
    )


    invoice_date = display_value(
        field_lookup.get(
            "INVOICE_DATE"
        )
    )


    total_amount = display_value(
        field_lookup.get(
            "TOTAL_AMOUNT"
        )
    )


    summary1, summary2, summary3, summary4 = (
        st.columns(4)
    )


    summary1.metric(
        "Vendor",
        vendor,
    )


    summary2.metric(
        "Invoice Number",
        invoice_number,
    )


    summary3.metric(
        "Invoice Date",
        invoice_date,
    )


    summary4.metric(
        "Total Amount",
        total_amount,
    )


# ============================================================
# 16 PRODUCTION FIELDS
# ============================================================

if field_dictionary is not None:

    st.subheader(
        "Extracted Invoice Fields"
    )


    field_rows = []


    detected_count = 0


    for field in (
        EXPECTED_FIELD_ORDER
    ):

        value = (
            display_value(
                field_lookup.get(
                    field
                )
            )
        )


        if (
            value
            !=
            "NOT_DETECTED"
        ):

            detected_count += 1


        field_rows.append(
            {
                "Field":
                    field,

                "Value":
                    value,
            }
        )


    detected_col, missing_col = (
        st.columns(2)
    )


    detected_col.metric(
        "Detected Fields",
        f"{detected_count}/16",
    )


    missing_col.metric(
        "Not Detected",
        16 - detected_count,
    )


    st.dataframe(
        field_rows,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# LINE ITEMS / MATERIAL TABLE
# ============================================================

table_name, table_rows = (
    find_tabular_data(
        result
    )
)


if table_rows:

    st.subheader(
        "Line Items / Material Table"
    )


    st.caption(
        "Runtime section: "
        f"{table_name}"
    )


    st.dataframe(
        table_rows,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# COMPLETE STRUCTURED OUTPUT
# ============================================================

st.subheader(
    "Complete Production Output"
)


structured_tab, json_tab = (
    st.tabs(
        [
            "Structured View",
            "Raw JSON",
        ]
    )
)


with structured_tab:

    if isinstance(
        result,
        dict,
    ):

        st.json(
            result,
            expanded=False,
        )


    else:

        st.write(
            result
        )


with json_tab:

    json_text = (
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )


    st.code(
        json_text,
        language="json",
    )


# ============================================================
# DOWNLOAD RESULT
# ============================================================

st.header(
    "5. Export Result"
)


base_filename = (

    Path(
        processed_filename
        or
        uploaded_file.name
    )
    .stem

)


json_filename = (
    base_filename
    +
    "_Invoice_AI_V3.json"
)


st.download_button(
    label=
        "⬇️ Download Structured JSON",
    data=
        create_json_bytes(
            result
        ),
    file_name=
        json_filename,
    mime=
        "application/json",
    type=
        "primary",
    use_container_width=True,
)


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "Invoice AI V3 • "
    "LayoutLMv3 • "
    "Final V6.1 Production Runtime • "
    "No retraining during inference"
)