"""API routes for the domain registry.

A domain is a named bucket of documents backed 1:1 by one pgvector chunk table.
These endpoints are additive: /tables and /table/{name} keep working unchanged, and
a domain name is always a valid chunk table name.
"""

import uuid
from typing import Optional

import logfire
from fastapi import APIRouter, Depends, HTTPException, Header

from app.api.dependencies import get_config, get_forget_pipeline, get_pipeline_factory
from app.api.routes.table_deletion import drop_chunk_table
from app.api.validators import require_access_password, validate_table_name
from app.infra.db import DomainRepository
from app.models.schemas import CreateDomainRequest, DomainDocument, DomainInfo

router = APIRouter()


def _repo(config) -> DomainRepository:
    return DomainRepository(connection_string=config.connection_string)


def _to_info(row: dict) -> DomainInfo:
    return DomainInfo(
        name=row["name"],
        display_name=row["display_name"],
        description=row.get("description"),
        table_name=row["table_name"],
        document_count=row.get("document_count") or 0,
    )


@router.get("/domains")
async def list_domains(config=Depends(get_config)):
    """List every domain with its document count."""
    try:
        rows = await _repo(config).list_domains()
        domains = [_to_info(r) for r in rows]
        return {"domains": domains, "total": len(domains)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list domains: {str(e)}")


@router.post("/domains")
async def create_domain(
    request: CreateDomainRequest,
    x_app_password: Optional[str] = Header(default=None),
    config=Depends(get_config),
):
    """Create a domain. Idempotent — an existing domain is returned unchanged."""
    require_access_password(x_app_password)
    # The domain name is also the chunk table name, so it must clear the same
    # identifier and reserved-name checks an upload would apply.
    validate_table_name(request.name)
    try:
        row = await _repo(config).create_domain(
            name=request.name,
            display_name=request.display_name,
            description=request.description,
        )
        return _to_info(row)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create domain: {str(e)}")


@router.get("/domains/{name}")
async def get_domain(name: str, config=Depends(get_config)):
    """Return one domain."""
    validate_table_name(name)
    row = await _repo(config).get_domain(name)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Domain '{name}' does not exist")
    return _to_info(row)


@router.get("/domains/{name}/documents")
async def list_domain_documents(name: str, config=Depends(get_config)):
    """List the documents in a domain, including ones still being ingested.

    Reads the ingestion status table, not the chunk table, so a document at
    stage='parsing' appears with its stage rather than being invisible until its
    chunks land.
    """
    validate_table_name(name)
    repo = _repo(config)
    if await repo.get_domain(name) is None:
        raise HTTPException(status_code=404, detail=f"Domain '{name}' does not exist")

    rows = await repo.list_documents(name)
    documents = [
        DomainDocument(
            document_id=str(r["id"]),
            doc_name=r.get("doc_name"),
            file_name=r["file_name"],
            stage=r["stage"],
            chunk_count=r.get("chunk_count"),
        )
        for r in rows
    ]
    return {"domain": name, "documents": documents, "total": len(documents)}


@router.delete("/domains/{name}")
async def delete_domain(
    name: str,
    x_app_password: Optional[str] = Header(default=None),
    config=Depends(get_config),
    get_pipeline=Depends(get_pipeline_factory),
    forget_pipeline=Depends(get_forget_pipeline),
):
    """Delete a domain: its chunk table, its ingestion status rows, and its registry row."""
    require_access_password(x_app_password)
    validate_table_name(name)

    with logfire.span("domain_deletion", domain=name):
        repo = _repo(config)
        domain = await repo.get_domain(name)
        if domain is None:
            raise HTTPException(status_code=404, detail=f"Domain '{name}' does not exist")

        try:
            # A domain created but never uploaded to has no chunk table yet, so a
            # missing table is not an error here (unlike DELETE /table).
            result = await drop_chunk_table(
                domain["table_name"],
                config=config,
                get_pipeline=get_pipeline,
                forget_pipeline=forget_pipeline,
                missing_table_ok=True,
            )
            # drop_chunk_table deletes the domain row by table_name. If the domain
            # name differs from the table_name (future schema allows this), delete
            # the domain row by name as well.
            if domain["name"] != domain["table_name"]:
                await repo.delete_domain(name)

            logfire.info(
                "Domain deletion completed",
                domain=name,
                table_existed=result["table_existed"],
                documents_removed=result["documents_removed"],
            )

            return {
                "status": "success",
                "message": f"Domain '{name}' deleted successfully",
                "domain": name,
                "table_name": domain["table_name"],
                "table_existed": result["table_existed"],
                "estimated_rows_deleted": result["estimated_rows_deleted"],
                "documents_removed": result["documents_removed"],
                "timestamp": str(uuid.uuid1().time),
            }
        except HTTPException:
            raise
        except Exception as e:
            logfire.error(
                "Domain deletion failed",
                domain=name,
                error=str(e),
                error_type=type(e).__name__,
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete domain '{name}': {str(e)}",
            )
