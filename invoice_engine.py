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
    LayoutLMv3Processor,
    LayoutLMv3ForTokenClassification,
)


# ============================================================
# INVOICE AI V3 — FINAL PRODUCTION ENGINE
# ============================================================


# ============================================================
# FINAL V3 MODEL CONTRACT
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
    EXPECTED_LABELS.append(
        f"B-{field}"
    )

    EXPECTED_LABELS.append(
        f"I-{field}"
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


    if exact_path.exists():
        return exact_path


    candidates = sorted(
        ARTIFACTS_DIR.glob(
            pattern
        )
    )


    if len(candidates) == 1:
        return candidates[0]


    if len(candidates) == 0:

        raise FileNotFoundError(
            "\nRequired artifact ZIP was not found.\n\n"
            f"Expected location:\n"
            f"{exact_path}\n"
        )


    raise RuntimeError(
        "\nMultiple matching ZIP files were found.\n"
        "Keep only one correct production artifact:\n\n"
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
                "Unsafe path found inside ZIP:\n"
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
    """
    Discovery check ONLY.

    We intentionally do NOT inspect raw config.json for
    num_labels here.

    Hugging Face may derive num_labels from id2label instead
    of storing an explicit num_labels field.

    Strict V3 validation is performed AFTER the model loads.
    """

    config_file = (
        candidate
        / "config.json"
    )


    if not config_file.is_file():
        return False


    has_weights = (
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


    return has_weights


def find_model_directory() -> Path:

    if not MODEL_CACHE_ROOT.exists():

        raise FileNotFoundError(
            "Model cache does not exist:\n"
            f"{MODEL_CACHE_ROOT}"
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
            "was found in the cache."
        )


    # Prefer the shallowest valid model directory.
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

    # First try existing extracted cache.
    try:

        return find_model_directory()

    except (
        FileNotFoundError,
        RuntimeError,
    ):

        pass


    model_zip = (
        get_model_zip()
    )


    print(
        "\nExtracting Final V3 model..."
    )


    _extract_fresh(
        model_zip,
        MODEL_CACHE_ROOT,
    )


    model_directory = (
        find_model_directory()
    )


    print(
        "✅ Model extraction complete"
    )


    return model_directory


# ============================================================
# MODEL LOADER
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


    # ========================================================
    # STRICT MODEL AUDIT
    # ========================================================

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
            "Wrong model architecture loaded.\n"
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
            f"Found parameters:    "
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
            f"Found labels:    "
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
            "match the expected production schema."
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
# RUNTIME ZIP SIGNATURE
# ============================================================

def _sha256(
    file_path: Path,
) -> str:

    digest = (
        hashlib.sha256()
    )


    with file_path.open(
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


    return digest.hexdigest()


def _get_runtime_signature():

    runtime_zip = (
        get_runtime_zip()
    )


    stat = runtime_zip.stat()


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

    if not RUNTIME_SIGNATURE_FILE.exists():
        return None


    try:

        return json.loads(
            RUNTIME_SIGNATURE_FILE.read_text(
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
# RUNTIME DISCOVERY
# ============================================================

def find_runtime_loader() -> Path:

    if not RUNTIME_CACHE_ROOT.exists():

        raise FileNotFoundError(
            "Runtime cache does not exist:\n"
            f"{RUNTIME_CACHE_ROOT}"
        )


    loaders = list(
        RUNTIME_CACHE_ROOT.rglob(
            CANONICAL_RUNTIME_LOADER
        )
    )


    if len(loaders) != 1:

        raise RuntimeError(
            "Expected exactly ONE canonical "
            "production runtime loader.\n\n"
            f"Filename: "
            f"{CANONICAL_RUNTIME_LOADER}\n"
            f"Found: {len(loaders)}"
        )


    return loaders[0]


def prepare_runtime(
    force_refresh=False,
) -> Path:

    signature = (
        _get_runtime_signature()
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


            stored_signature = (
                _read_runtime_signature()
            )


            cache_valid = (
                loader.is_file()
                and
                stored_signature
                ==
                signature
            )


        except Exception:

            cache_valid = False


    if not cache_valid:

        print(
            "\nExtracting clean production runtime..."
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
# LEGACY RUNTIME PATH FIX
# ============================================================

def _suffix_match_score(
    requested: Path,
    candidate: Path,
) -> int:

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


    for requested_part, candidate_part in zip(
        reversed(
            requested_parts
        ),
        reversed(
            candidate_parts
        ),
    ):

        if (
            requested_part
            !=
            candidate_part
        ):

            break


        score += 1


    return score


def _resolve_runtime_source(
    source,
    runtime_loader: Path,
) -> Path:

    requested = Path(
        source
    )


    # Original path exists.
    if requested.exists():

        return requested


    # Relative path possibilities.
    if not requested.is_absolute():

        possibilities = [

            runtime_loader.parent
            / requested,

            RUNTIME_CACHE_ROOT
            / requested,

            PROJECT_ROOT
            / requested,

        ]


        for candidate in possibilities:

            if candidate.exists():

                return candidate


    # Legacy source path may point to an old folder.
    # Search the freshly extracted runtime by filename.
    matches = [

        candidate

        for candidate
        in RUNTIME_CACHE_ROOT.rglob(
            requested.name
        )

        if candidate.is_file()

    ]


    if not matches:

        raise FileNotFoundError(
            "\nProduction runtime requested a source "
            "file which cannot be located.\n\n"
            f"Requested source:\n"
            f"{requested}\n\n"
            f"Runtime root:\n"
            f"{RUNTIME_CACHE_ROOT}\n"
        )


    matches.sort(
        key=lambda candidate: (

            -_suffix_match_score(
                requested,
                candidate,
            ),

            len(
                candidate.parts
            ),

            str(candidate).lower(),

        )
    )


    # If multiple files have exactly the same quality match,
    # do not silently choose the wrong file.
    if len(matches) > 1:

        first_score = (
            _suffix_match_score(
                requested,
                matches[0],
            )
        )


        second_score = (
            _suffix_match_score(
                requested,
                matches[1],
            )
        )


        if (
            first_score
            ==
            second_score
        ):

            raise RuntimeError(
                "\nAmbiguous runtime source path.\n\n"
                f"Requested:\n"
                f"{requested}\n\n"
                "Possible matches:\n"
                +
                "\n".join(
                    str(path)
                    for path
                    in matches
                )
            )


    return matches[0]


# ============================================================
# SAFE SHUTIL WRAPPERS
# ============================================================

def _prepare_destination_parent(
    destination,
):

    destination_path = Path(
        destination
    )


    destination_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    return destination_path


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


        destination_path = Path(
            destination
        )


        # If destination already exists as a directory,
        # copy into that directory.
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


        # Protect against copying a file onto itself.
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


        destination_path = Path(
            destination
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
            _prepare_destination_parent(
                destination
            )
        )


        return original_copyfile(
            str(source_path),
            str(destination_path),
            follow_symlinks=
                follow_symlinks,
        )


    return safe_copyfile


# ============================================================
# COMPLETE V3 PRODUCTION ENGINE
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
    # PREPARE VERIFIED RUNTIME PACKAGE
    # ========================================================

    runtime_loader = (
        prepare_runtime(
            force_refresh=
                force_runtime_refresh
        )
    )


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


        # Lowercase names

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


        # Production uppercase names

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
    # ALLOW RUNTIME TO IMPORT SIBLING PYTHON FILES
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
    # TEMPORARY SAFE COPY PATCH
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

        # ALWAYS restore shutil.
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
    # FINAL RUNTIME CONTRACT
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
            "\nProduction runtime executed, "
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

    }


# ============================================================
# MODEL-ONLY VERIFICATION
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
# COMPLETE PRODUCTION VERIFICATION
# ============================================================

def verify_production(
    rebuild_runtime=False,
):

    print(
        "=" * 72
    )


    print(
        "INVOICE AI V3 — COMPLETE PRODUCTION VERIFICATION"
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
        "🔥 INVOICE AI V3 COMPLETE PRODUCTION ENGINE VERIFIED"
    )


    print(
        "=" * 72
    )


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Invoice AI V3 "
            "Production Engine"
        )
    )


    parser.add_argument(
        "--model-only",
        action="store_true",
        help=(
            "Verify only the Final V3 model."
        ),
    )


    parser.add_argument(
        "--rebuild-runtime",
        action="store_true",
        help=(
            "Force a clean production runtime extraction."
        ),
    )


    args = parser.parse_args()


    if args.model_only:

        verify_model()

    else:

        verify_production(
            rebuild_runtime=
                args.rebuild_runtime
        )


if __name__ == "__main__":

    main()