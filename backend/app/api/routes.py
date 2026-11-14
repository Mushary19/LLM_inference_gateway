"""API route definitions."""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database.connection import get_db
from app.database.models import InferenceRequest, RequestStatus
from app.schemas.inference import GenerateRequest, GenerateResponse
from app.services.admission import AdmissionController
from app.services.inference import InferenceEngine

logger = logging.getLogger(__name__)

router = APIRouter()


def get_admission_controller() -> AdmissionController:
    return AdmissionController(
        max_queued_requests=settings.max_queued_requests,
        max_active_requests=settings.max_active_requests,
    )


def get_inference_engine(request: Request) -> InferenceEngine:
    """The process-wide engine created at startup; reused here for its tokenizer."""
    return request.app.state.inference_engine


@router.post("/generate", status_code=status.HTTP_202_ACCEPTED, response_model=GenerateResponse)
def generate(
    payload: GenerateRequest,
    db: Session = Depends(get_db),
    admission: AdmissionController = Depends(get_admission_controller),
    engine: InferenceEngine = Depends(get_inference_engine),
) -> GenerateResponse:
    input_tokens = engine.count_input_tokens(payload.prompt)

    try:
        # The admission lock is held until this transaction commits or rolls back.
        result = admission.evaluate(db)
        if not result.accepted:
            db.rollback()
            logger.warning("Rejected inference request: %s", result.reason)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference system is temporarily at capacity",
            )

        request = InferenceRequest(
            prompt=payload.prompt,
            priority=payload.priority,
            max_new_tokens=payload.max_new_tokens,
            input_tokens=input_tokens,
            status=RequestStatus.QUEUED,
        )
        db.add(request)
        db.commit()
        db.refresh(request)
    except SQLAlchemyError:
        db.rollback()
        logger.exception("Failed to store inference request")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store inference request",
        )

    return GenerateResponse(request_id=request.id, status=request.status)
