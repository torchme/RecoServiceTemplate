from random import sample
from typing import List

import polars as pl
from fastapi import APIRouter, FastAPI, HTTPException, Request
from pydantic import BaseModel

from service.api.exceptions import UserNotFoundError
from service.log import app_logger

data = pl.read_csv("data/items.csv")


class RecoResponse(BaseModel):
    user_id: int
    items: List[int]


router = APIRouter()


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
    responses={404: {"description": "Model not found"}},
)
async def get_reco(
    request: Request,
    model_name: str,
    user_id: int,
) -> RecoResponse:
    app_logger.info(f"Request for model: {model_name}, user_id: {user_id}")

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
