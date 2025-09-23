"""
FastAPI script for file validation using the existing FileValidator.
"""

import tempfile
import os
from typing import List
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from file_validator import FileValidator, FileValidationConfig, FileValidationResult

app = FastAPI(title="File Validation API", version="1.0.0")

# Initialize the file validator
validator = FileValidator(FileValidationConfig())


@app.post("/validate-file/", response_model=FileValidationResult)
async def validate_single_file(file: UploadFile = File(...)):
    """
    Validate a single uploaded file.

    Args:
        file: The uploaded file to validate

    Returns:
        FileValidationResult: Validation result with details
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Create a temporary file to save the uploaded content
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        try:
            # Read and write the uploaded file content
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()

            # Validate the file
            result = validator.validate_file(temp_file.name)

            # Override the filename with the original uploaded filename
            result.filename = file.filename

            return result

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file.name)
            except OSError:
                pass


# @app.post("/validate-files/", response_model=List[FileValidationResult])
# async def validate_multiple_files(files: List[UploadFile] = File(...)):
#     """
#     Validate multiple uploaded files.

#     Args:
#         files: List of uploaded files to validate

#     Returns:
#         List[FileValidationResult]: List of validation results
#     """
#     if not files:
#         raise HTTPException(status_code=400, detail="No files provided")

#     results = []
#     temp_files = []

#     try:
#         # Process each uploaded file
#         for file in files:
#             if not file.filename:
#                 results.append(FileValidationResult(
#                     filename="unknown",
#                     file_type=None,
#                     is_valid=False,
#                     file_size=0,
#                     error_message="No filename provided"
#                 ))
#                 continue

#             # Create temporary file
#             temp_file = tempfile.NamedTemporaryFile(
#                 delete=False,
#                 suffix=os.path.splitext(file.filename)[1]
#             )
#             temp_files.append(temp_file.name)

#             # Read and write content
#             content = await file.read()
#             temp_file.write(content)
#             temp_file.flush()
#             temp_file.close()

#             # Validate the file
#             result = validator.validate_file(temp_file.name)

#             # Override the filename with the original uploaded filename
#             result.filename = file.filename

#             results.append(result)

#         return results

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing files: {str(e)}")
#     finally:
#         # Clean up all temporary files
#         for temp_path in temp_files:
#             try:
#                 os.unlink(temp_path)
#             except OSError:
#                 pass


@app.get("/supported-types/")
async def get_supported_types():
    """
    Get information about supported file types and validation config.

    Returns:
        dict: Supported file types and configuration
    """
    return {
        "supported_extensions": validator.config.allowed_extensions,
        "max_file_size_mb": validator.config.max_file_size_mb,
        "supported_types": ["pdf", "docx", "txt"],
        "mime_types": list(validator.mime_type_mapping.keys())
    }


@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "File Validation API",
        "endpoints": {
            "POST /validate-file/": "Validate a single file",
            "POST /validate-files/": "Validate multiple files",
            "GET /supported-types/": "Get supported file types and config",
            "GET /docs": "API documentation"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)