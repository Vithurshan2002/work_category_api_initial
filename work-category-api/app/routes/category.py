from fastapi import APIRouter, Request

from app.schemas import CategoryRequest, CategoryResponse

router = APIRouter(
    prefix="/category",
    tags=["Category"],
)


@router.post("/predict", response_model=CategoryResponse)
def predict_category(
    payload: CategoryRequest,
    request: Request,
):
    predictor = request.app.state.predictor

    category = predictor.predict(payload.text)

    return CategoryResponse(category=category)