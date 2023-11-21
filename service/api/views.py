from random import sample
from typing import Any, Dict, List, Union

import polars as pl
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from service.api.exceptions import BearerAccessTokenError, UserNotFoundError
from service.log import app_logger

data = pl.read_csv("data/items.csv")


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


responses: Dict[Union[int, str], Dict[str, Any]] = {
    404: {"description": "Model not found", "content": {"application/json": {"example": {"detail": "Model not found"}}}}
}


router = APIRouter()

bearer_scheme = HTTPBearer()


@router.get(
    path="/health",
    tags=["Health"],
)
async def health() -> str:
    return "I am alive"


@router.get(
    path="/reco/{model_name}/{user_id}",
    tags=["Recommendations"],
    response_model=RecoResponse,
    responses=responses,
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
    token: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

    if token.credentials != request.app.state.true_token:
        raise BearerAccessTokenError()

    if model_name != "random":
        raise HTTPException(status_code=404, detail="Model not found")

    if user_id > 10**9:
        raise UserNotFoundError(error_message=f"User {user_id} not found")

    k_recs = request.app.state.k_recs

    item_ids = data["item_id"].to_list()
    if len(item_ids) < k_recs:
        reco = item_ids
    else:
        reco = sample(item_ids, k_recs)

    return RecoResponse(user_id=user_id, items=reco)


def add_views(app: FastAPI) -> None:
    app.include_router(router)
