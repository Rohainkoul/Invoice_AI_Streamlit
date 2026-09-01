from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile

from functools import lru_cache
from pathlib import Path, PurePosixPath

import torch

from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
)


# ============================================================
# INVOICE AI V3 — FINAL PRODUCTION ENGINE
# ============================================================

EXPECTED_PARAMETER_COUNT = 125_942_945

EXPECTED_FIELDS = [
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


EXPECTED_LABELS = ["O"]

for field in EXPECTED_FIELDS:
    EXPECTED_LABELS.extend(
        [
            f"B-{field}",
            f"I-{field}",
        ]
    )


EXPECTED_NUM_LABELS = len(
    EXPECTED_LABELS
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
)


ARTIFACTS_DIR = (
    PROJECT_ROOT
    / "artifacts"
)


CACHE_ROOT = (
    PROJECT_ROOT
    / ".invoice_ai_cache"
)


MODEL_CACHE_ROOT = (
    CACHE_ROOT
    / "model"
)


RUNTIME_CACHE_ROOT = (
    CACHE_ROOT
    / "production_runtime"
)


COLAB_COMPAT_ROOT = (
    CACHE_ROOT
    / "colab_content"
)


RUNTIME_SIGNATURE_FILE = (
    RUNTIME_CACHE_ROOT
    / ".runtime_signature.json"
)


MODEL_ZIP_NAME = (
    "Invoice_AI_V3_AllRounder_FINAL.zip"
)


RUNTIME_ZIP_NAME = (
    "Invoice_AI_V3_FINAL_Production_Runtime.zip"
)


CANONICAL_RUNTIME_LOADER = (
    "load_v3_final_runtime.py"
)


# ============================================================
# ARTIFACT DISCOVERY
# ============================================================

def _find_artifact(
    exact_name: str,
    pattern: str,
) -> Path:

    exact_path = (
        ARTIFACTS_DIR
        / exact_name
    )


    if exact_path.is_file():
        return exact_path


    candidates = sorted(
        ARTIFACTS_DIR.glob(
            pattern
        )
    )


    if len(candidates) == 1:
        return candidates[0]


    if not candidates:

        raise FileNotFoundError(
            "Required artifact was not found.\n\n"
            f"Expected:\n{exact_path}"
        )


    raise RuntimeError(
        "Multiple matching artifact ZIPs were found.\n\n"
        +
        "\n".join(
            str(path)
            for path in candidates
        )
    )


def get_model_zip() -> Path:

    return _find_artifact(
        MODEL_ZIP_NAME,
        "Invoice_AI_V3_AllRounder_FINAL*.zip",
    )


def get_runtime_zip() -> Path:

    return _find_artifact(
        RUNTIME_ZIP_NAME,
        "Invoice_AI_V3_FINAL_Production_Runtime*.zip",
    )


# ============================================================
# SAFE ZIP EXTRACTION
# ============================================================

def _validate_zip_members(
    archive: zipfile.ZipFile,
):

    for member in archive.infolist():

        normalized = (
            member.filename
            .replace(
                "\\",
                "/",
            )
        )


        path = PurePosixPath(
            normalized
        )


        if (
            path.is_absolute()
            or
            ".." in path.parts
        ):

            raise RuntimeError(
                "Unsafe path inside ZIP:\n"
                f"{member.filename}"
            )


def _extract_fresh(
    zip_path: Path,
    destination: Path,
):

    temporary = (
        destination.parent
        /
        (
            destination.name
            + ".__extracting__"
        )
    )


    if temporary.exists():

        shutil.rmtree(
            temporary,
            ignore_errors=True,
        )


    temporary.mkdir(
        parents=True,
        exist_ok=True,
    )


    try:

        with zipfile.ZipFile(
            zip_path,
            "r",
        ) as archive:

            _validate_zip_members(
                archive
            )


            archive.extractall(
                temporary
            )


        if destination.exists():

            shutil.rmtree(
                destination,
                ignore_errors=True,
            )


        temporary.replace(
            destination
        )


    except Exception:

        shutil.rmtree(
            temporary,
            ignore_errors=True,
        )

        raise


# ============================================================
# MODEL DISCOVERY
# ============================================================

def _candidate_is_model(
    candidate: Path,
) -> bool:

    config_file = (
        candidate
        / "config.json"
    )


    if not config_file.is_file():
        return False


    return (
        (
            candidate
            / "model.safetensors"
        ).is_file()

        or

        (
            candidate
            / "pytorch_model.bin"
        ).is_file()
    )


def find_model_directory() -> Path:

    if not MODEL_CACHE_ROOT.exists():

        raise FileNotFoundError(
            "Model cache does not exist."
        )


    matches = []


    for config_file in (
        MODEL_CACHE_ROOT.rglob(
            "config.json"
        )
    ):

        candidate = (
            config_file.parent
        )


        if _candidate_is_model(
            candidate
        ):

            matches.append(
                candidate
            )


    if not matches:

        raise RuntimeError(
            "No model directory containing "
            "config.json + model weights "
            "was found."
        )


    matches.sort(
        key=lambda path: (
            len(
                path.relative_to(
                    MODEL_CACHE_ROOT
                ).parts
            ),
            str(path).lower(),
        )
    )


    return matches[0]


def prepare_model() -> Path:

    try:

        return (
            find_model_directory()
        )

    except (
        FileNotFoundError,
        RuntimeError,
    ):

        pass


    print(
        "\nExtracting Final V3 model..."
    )


    _extract_fresh(
        get_model_zip(),
        MODEL_CACHE_ROOT,
    )


    model_dir = (
        find_model_directory()
    )


    print(
        "✅ Model extraction complete"
    )


    return model_dir


# ============================================================
# FINAL V3 MODEL
# ============================================================

@lru_cache(maxsize=1)
def load_v3_model():

    model_dir = (
        prepare_model()
    )


    print(
        "\nLoading Final V3 model..."
    )


    processor = (
        LayoutLMv3Processor
        .from_pretrained(
            str(model_dir),
            apply_ocr=False,
            local_files_only=True,
        )
    )


    model = (
        LayoutLMv3ForTokenClassification
        .from_pretrained(
            str(model_dir),
            local_files_only=True,
        )
    )


    device = torch.device(
        "cuda:0"
        if torch.cuda.is_available()
        else "cpu"
    )


    model.to(
        device
    )


    model.eval()


    parameter_count = sum(
        parameter.numel()
        for parameter
        in model.parameters()
    )


    if (
        model.__class__.__name__
        !=
        "LayoutLMv3ForTokenClassification"
    ):

        raise RuntimeError(
            "Wrong model architecture.\n"
            f"Found: "
            f"{model.__class__.__name__}"
        )


    if (
        parameter_count
        !=
        EXPECTED_PARAMETER_COUNT
    ):

        raise RuntimeError(
            "Wrong model loaded.\n\n"
            f"Expected parameters: "
            f"{EXPECTED_PARAMETER_COUNT:,}\n"
            f"Found parameters: "
            f"{parameter_count:,}"
        )


    if (
        model.config.num_labels
        !=
        EXPECTED_NUM_LABELS
    ):

        raise RuntimeError(
            "Wrong BIO schema.\n\n"
            f"Expected labels: "
            f"{EXPECTED_NUM_LABELS}\n"
            f"Found labels: "
            f"{model.config.num_labels}"
        )


    id2label = {
        int(key): value
        for key, value
        in model.config.id2label.items()
    }


    label_list = [
        id2label[index]
        for index
        in range(
            model.config.num_labels
        )
    ]


    if (
        label_list
        !=
        EXPECTED_LABELS
    ):

        raise RuntimeError(
            "Final V3 label order does not "
            "match the expected schema."
        )


    label2id = {
        label: index
        for index, label
        in enumerate(
            label_list
        )
    }


    print(
        "✅ Final V3 model loaded"
    )


    return {

        "model":
            model,

        "processor":
            processor,

        "device":
            device,

        "model_dir":
            model_dir,

        "parameter_count":
            parameter_count,

        "fields":
            list(
                EXPECTED_FIELDS
            ),

        "label_list":
            label_list,

        "id2label":
            id2label,

        "label2id":
            label2id,

    }


# ============================================================
# RUNTIME SIGNATURE
# ============================================================

def _sha256(
    path: Path,
) -> str:

    digest = (
        hashlib.sha256()
    )


    with path.open(
        "rb"
    ) as handle:

        while True:

            chunk = handle.read(
                1024 * 1024
            )


            if not chunk:
                break


            digest.update(
                chunk
            )


    return (
        digest.hexdigest()
    )


def _runtime_signature():

    runtime_zip = (
        get_runtime_zip()
    )


    stat = (
        runtime_zip.stat()
    )


    return {

        "filename":
            runtime_zip.name,

        "size":
            stat.st_size,

        "sha256":
            _sha256(
                runtime_zip
            ),

    }


def _read_runtime_signature():

    if not (
        RUNTIME_SIGNATURE_FILE
        .exists()
    ):

        return None


    try:

        return json.loads(
            RUNTIME_SIGNATURE_FILE
            .read_text(
                encoding="utf-8"
            )
        )

    except Exception:

        return None


def _write_runtime_signature(
    signature,
):

    RUNTIME_SIGNATURE_FILE.write_text(

        json.dumps(
            signature,
            indent=2,
        ),

        encoding="utf-8",

    )


# ============================================================
# RUNTIME EXTRACTION
# ============================================================

def find_runtime_loader() -> Path:

    if not (
        RUNTIME_CACHE_ROOT
        .exists()
    ):

        raise FileNotFoundError(
            "Runtime cache does not exist."
        )


    loaders = list(
        RUNTIME_CACHE_ROOT.rglob(
            CANONICAL_RUNTIME_LOADER
        )
    )


    if len(loaders) != 1:

        raise RuntimeError(
            "Expected exactly one "
            f"{CANONICAL_RUNTIME_LOADER}, "
            f"found {len(loaders)}."
        )


    return loaders[0]


def prepare_runtime(
    force_refresh=False,
) -> Path:

    signature = (
        _runtime_signature()
    )


    cache_valid = False


    if (
        RUNTIME_CACHE_ROOT.exists()
        and
        not force_refresh
    ):

        try:

            loader = (
                find_runtime_loader()
            )


            cache_valid = (
                loader.is_file()
                and
                _read_runtime_signature()
                ==
                signature
            )


        except Exception:

            cache_valid = False


    if not cache_valid:

        print(
            "\nExtracting clean "
            "production runtime..."
        )


        _extract_fresh(
            get_runtime_zip(),
            RUNTIME_CACHE_ROOT,
        )


        _write_runtime_signature(
            signature
        )


        print(
            "✅ Runtime extraction complete"
        )


    return (
        find_runtime_loader()
    )


# ============================================================
# EXACT COLAB -> STREAMLIT CLOUD FIX
# ============================================================

def _patch_colab_runtime_paths():
    """
    The production runtime contains Google Colab paths.

    IMPORTANT:
    V6.1 loads executable source from the original .ipynb,
    therefore patching only *.py files is NOT enough.

    This patches:
      *.py
      *.ipynb
      *.json

    /content/... becomes:
      .invoice_ai_cache/colab_content/...
    """

    COLAB_COMPAT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


    replacement = (
        COLAB_COMPAT_ROOT
        .resolve()
        .as_posix()
    )


    patchable_extensions = {
        ".py",
        ".ipynb",
        ".json",
    }


    patched_files = []


    for runtime_file in (
        RUNTIME_CACHE_ROOT.rglob(
            "*"
        )
    ):

        if not runtime_file.is_file():
            continue


        if (
            runtime_file.suffix.lower()
            not in patchable_extensions
        ):

            continue


        try:

            source = (
                runtime_file.read_text(
                    encoding="utf-8"
                )
            )


        except (
            UnicodeDecodeError,
            OSError,
        ):

            continue


        patched = source


        # ----------------------------------------------------
        # Normal Python / JSON strings
        # ----------------------------------------------------

        patched = patched.replace(
            '"/content',
            f'"{replacement}',
        )


        patched = patched.replace(
            "'/content",
            f"'{replacement}",
        )


        # ----------------------------------------------------
        # Escaped strings inside .ipynb JSON
        #
        # Example:
        #   Path(\"/content\")
        # ----------------------------------------------------

        patched = patched.replace(
            '\\"/content',
            f'\\"{replacement}',
        )


        patched = patched.replace(
            "\\'/content",
            f"\\'{replacement}",
        )


        if patched != source:

            runtime_file.write_text(
                patched,
                encoding="utf-8",
            )


            patched_files.append(
                runtime_file.name
            )


    print(
        "✅ Colab filesystem compatibility ready"
    )


    print(
        "Patched runtime files:",
        len(patched_files),
    )


    for filename in patched_files:

        print(
            " •",
            filename,
        )


# ============================================================
# RUNTIME /CONTENT SAFETY NET
# ============================================================

def _redirect_colab_path(
    value,
) -> Path:

    normalized = (
        str(value)
        .replace(
            "\\",
            "/",
        )
    )


    if (
        normalized
        ==
        "/content"

        or

        normalized.startswith(
            "/content/"
        )
    ):

        relative = (
            normalized[
                len("/content"):
            ]
            .lstrip("/")
        )


        return (
            COLAB_COMPAT_ROOT
            / relative
        )


    return Path(
        value
    )


# ============================================================
# LEGACY SOURCE RESOLUTION
# ============================================================

def _suffix_match_score(
    requested: Path,
    candidate: Path,
):

    requested_parts = [
        part.lower()
        for part
        in requested.parts
    ]


    candidate_parts = [
        part.lower()
        for part
        in candidate.parts
    ]


    score = 0


    for left, right in zip(

        reversed(
            requested_parts
        ),

        reversed(
            candidate_parts
        ),

    ):

        if left != right:
            break


        score += 1


    return score


def _resolve_runtime_source(
    source,
    runtime_loader: Path,
) -> Path:

    redirected = (
        _redirect_colab_path(
            source
        )
    )


    if redirected.exists():

        return redirected


    original = Path(
        source
    )


    if not original.is_absolute():

        possibilities = [

            runtime_loader.parent
            / original,

            RUNTIME_CACHE_ROOT
            / original,

            PROJECT_ROOT
            / original,

            COLAB_COMPAT_ROOT
            / original,

        ]


        for candidate in possibilities:

            if candidate.exists():

                return candidate


    matches = [

        candidate

        for candidate
        in RUNTIME_CACHE_ROOT.rglob(
            original.name
        )

        if candidate.is_file()

    ]


    if not matches:

        raise FileNotFoundError(
            "Production runtime requested "
            "a source file that cannot be found.\n\n"
            f"Requested:\n{source}\n\n"
            f"Runtime root:\n"
            f"{RUNTIME_CACHE_ROOT}"
        )


    matches.sort(
        key=lambda candidate: (

            -_suffix_match_score(
                original,
                candidate,
            ),

            len(
                candidate.parts
            ),

            str(
                candidate
            ).lower(),

        )
    )


    if len(matches) > 1:

        first_score = (
            _suffix_match_score(
                original,
                matches[0],
            )
        )


        second_score = (
            _suffix_match_score(
                original,
                matches[1],
            )
        )


        if (
            first_score
            ==
            second_score
        ):

            raise RuntimeError(
                "Ambiguous runtime source path.\n\n"
                f"Requested:\n{source}\n\n"
                "Matches:\n"
                +
                "\n".join(
                    str(path)
                    for path
                    in matches
                )
            )


    return matches[0]


# ============================================================
# SAFE SHUTIL PATCHES
# ============================================================

def _build_safe_copy2(
    runtime_loader,
    original_copy2,
):

    def safe_copy2(
        source,
        destination,
        *,
        follow_symlinks=True,
    ):

        source_path = (
            _resolve_runtime_source(
                source,
                runtime_loader,
            )
        )


        destination_path = (
            _redirect_colab_path(
                destination
            )
        )


        if (
            destination_path.exists()
            and
            destination_path.is_dir()
        ):

            destination_path.mkdir(
                parents=True,
                exist_ok=True,
            )

        else:

            destination_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )


        try:

            final_destination = (
                destination_path
                / source_path.name

                if destination_path.is_dir()

                else destination_path
            )


            if (
                final_destination.exists()
                and
                source_path.resolve()
                ==
                final_destination.resolve()
            ):

                return str(
                    final_destination
                )


        except Exception:

            pass


        return original_copy2(
            str(source_path),
            str(destination_path),
            follow_symlinks=
                follow_symlinks,
        )


    return safe_copy2


def _build_safe_copy(
    runtime_loader,
    original_copy,
):

    def safe_copy(
        source,
        destination,
        *,
        follow_symlinks=True,
    ):

        source_path = (
            _resolve_runtime_source(
                source,
                runtime_loader,
            )
        )


        destination_path = (
            _redirect_colab_path(
                destination
            )
        )


        if (
            destination_path.exists()
            and
            destination_path.is_dir()
        ):

            destination_path.mkdir(
                parents=True,
                exist_ok=True,
            )

        else:

            destination_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )


        return original_copy(
            str(source_path),
            str(destination_path),
            follow_symlinks=
                follow_symlinks,
        )


    return safe_copy


def _build_safe_copyfile(
    runtime_loader,
    original_copyfile,
):

    def safe_copyfile(
        source,
        destination,
        *,
        follow_symlinks=True,
    ):

        source_path = (
            _resolve_runtime_source(
                source,
                runtime_loader,
            )
        )


        destination_path = (
            _redirect_colab_path(
                destination
            )
        )


        destination_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


        return original_copyfile(
            str(source_path),
            str(destination_path),
            follow_symlinks=
                follow_symlinks,
        )


    return safe_copyfile


# ============================================================
# COMPLETE PRODUCTION ENGINE
# ============================================================

@lru_cache(maxsize=2)
def load_v3_production_engine(
    force_runtime_refresh=False,
):

    # ========================================================
    # LOAD VERIFIED MODEL
    # ========================================================

    engine = (
        load_v3_model()
    )


    # ========================================================
    # EXTRACT / VERIFY RUNTIME
    # ========================================================

    runtime_loader = (
        prepare_runtime(
            force_refresh=
                force_runtime_refresh
        )
    )


    # ========================================================
    # CRITICAL:
    # PATCH .PY + .IPYNB + .JSON BEFORE EXECUTION
    # ========================================================

    _patch_colab_runtime_paths()


    print(
        "\nRuntime loader:"
    )


    print(
        runtime_loader
    )


    # ========================================================
    # VARIABLES EXPECTED BY PRODUCTION RUNTIME
    # ========================================================

    runtime_namespace = {

        "__name__":
            "invoice_ai_v3_production_runtime",

        "__file__":
            str(
                runtime_loader
            ),

        "__builtins__":
            __builtins__,


        "model":
            engine["model"],

        "processor":
            engine["processor"],

        "device":
            engine["device"],

        "id2label":
            engine["id2label"],

        "label2id":
            engine["label2id"],


        "MODEL":
            engine["model"],

        "PROCESSOR":
            engine["processor"],

        "DEVICE":
            engine["device"],

        "TARGET_FIELDS":
            list(
                EXPECTED_FIELDS
            ),

        "EXPECTED_FIELDS":
            list(
                EXPECTED_FIELDS
            ),

        "MODEL_ID2LABEL":
            engine["id2label"],

        "LABEL_LIST":
            engine["label_list"],

        "PARAMETERS":
            engine[
                "parameter_count"
            ],

    }


    runtime_source = (
        runtime_loader.read_text(
            encoding="utf-8"
        )
    )


    # ========================================================
    # ALLOW RUNTIME SIBLING IMPORTS
    # ========================================================

    runtime_directory = str(
        runtime_loader.parent
    )


    inserted_runtime_path = False


    if (
        runtime_directory
        not in sys.path
    ):

        sys.path.insert(
            0,
            runtime_directory,
        )


        inserted_runtime_path = True


    # ========================================================
    # TEMPORARY SHUTIL COMPATIBILITY
    # ========================================================

    original_copy2 = (
        shutil.copy2
    )


    original_copy = (
        shutil.copy
    )


    original_copyfile = (
        shutil.copyfile
    )


    shutil.copy2 = (
        _build_safe_copy2(
            runtime_loader,
            original_copy2,
        )
    )


    shutil.copy = (
        _build_safe_copy(
            runtime_loader,
            original_copy,
        )
    )


    shutil.copyfile = (
        _build_safe_copyfile(
            runtime_loader,
            original_copyfile,
        )
    )


    print(
        "\nLoading production runtime..."
    )


    try:

        exec(
            compile(
                runtime_source,
                str(
                    runtime_loader
                ),
                "exec",
            ),
            runtime_namespace,
            runtime_namespace,
        )


    finally:

        shutil.copy2 = (
            original_copy2
        )


        shutil.copy = (
            original_copy
        )


        shutil.copyfile = (
            original_copyfile
        )


        if inserted_runtime_path:

            try:

                sys.path.remove(
                    runtime_directory
                )


            except ValueError:

                pass


    # ========================================================
    # FINAL CONTRACT
    # ========================================================

    process_invoice_final = (
        runtime_namespace.get(
            "process_invoice_final"
        )
    )


    if not callable(
        process_invoice_final
    ):

        raise RuntimeError(
            "Production runtime executed, "
            "but process_invoice_final() "
            "was not created."
        )


    print(
        "✅ Production runtime loaded"
    )


    return {

        **engine,


        "runtime_loader":
            runtime_loader,


        "runtime_namespace":
            runtime_namespace,


        "process_invoice_final":
            process_invoice_final,


        "runtime_ready":
            True,


        "colab_compat_root":
            COLAB_COMPAT_ROOT,

    }


# ============================================================
# VERIFY MODEL
# ============================================================

def verify_model():

    print(
        "=" * 72
    )


    print(
        "INVOICE AI V3 — MODEL VERIFICATION"
    )


    print(
        "=" * 72
    )


    engine = (
        load_v3_model()
    )


    print(
        "\n✅ FINAL V3 MODEL READY"
    )


    print(
        "Architecture :",
        engine[
            "model"
        ].__class__.__name__,
    )


    print(
        "Parameters   :",
        f"{engine['parameter_count']:,}",
    )


    print(
        "Fields       :",
        len(
            engine["fields"]
        ),
    )


    print(
        "BIO Labels   :",
        len(
            engine["label_list"]
        ),
    )


    print(
        "Device       :",
        engine["device"],
    )


    print(
        "Model path   :",
        engine["model_dir"],
    )


    print(
        "\n"
        + "=" * 72
    )


    print(
        "🔥 FINAL V3 MODEL VERIFIED"
    )


    print(
        "=" * 72
    )


# ============================================================
# VERIFY COMPLETE PRODUCTION ENGINE
# ============================================================

def verify_production(
    rebuild_runtime=False,
):

    print(
        "=" * 72
    )


    print(
        "INVOICE AI V3 — "
        "COMPLETE PRODUCTION VERIFICATION"
    )


    print(
        "=" * 72
    )


    engine = (
        load_v3_production_engine(
            force_runtime_refresh=
                rebuild_runtime
        )
    )


    print(
        "\n✅ FINAL V3 MODEL"
    )


    print(
        "Architecture :",
        engine[
            "model"
        ].__class__.__name__,
    )


    print(
        "Parameters   :",
        f"{engine['parameter_count']:,}",
    )


    print(
        "Fields       :",
        len(
            engine["fields"]
        ),
    )


    print(
        "BIO Labels   :",
        len(
            engine["label_list"]
        ),
    )


    print(
        "Device       :",
        engine["device"],
    )


    print(
        "\n✅ PRODUCTION RUNTIME"
    )


    print(
        "Loader       :",
        engine[
            "runtime_loader"
        ],
    )


    print(
        "Writable root:",
        engine[
            "colab_compat_root"
        ],
    )


    print(
        "\n✅ FINAL INFERENCE FUNCTION"
    )


    print(
        "process_invoice_final callable :",
        callable(
            engine[
                "process_invoice_final"
            ]
        ),
    )


    print(
        "\n"
        + "=" * 72
    )


    print(
        "🔥 INVOICE AI V3 COMPLETE "
        "PRODUCTION ENGINE VERIFIED"
    )


    print(
        "=" * 72
    )


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    parser = (
        argparse.ArgumentParser(
            description=(
                "Invoice AI V3 "
                "Production Engine"
            )
        )
    )


    parser.add_argument(
        "--model-only",
        action="store_true",
    )


    parser.add_argument(
        "--rebuild-runtime",
        action="store_true",
    )


    args = (
        parser.parse_args()
    )


    if args.model_only:

        verify_model()


    else:

        verify_production(
            rebuild_runtime=
                args.rebuild_runtime
        )


if __name__ == "__main__":

    main()