from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from recommender_service import recommender_service
from schemas import ProductCard, ProductDetails, Review, UserProfile

router = APIRouter()


@router.get("/recommend/", response_model=List[ProductCard])
def get_recommendations(user_id: str, k: int = 10):
    """
    Returns top-k recommendations for a given user.
    """
    try:
        recommendations = recommender_service.recommend_for_user(user_id, k)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not recommendations:
        return []
    return recommendations


@router.get("/item/{item_id}", response_model=ProductDetails)
def get_item_details(item_id: str):
    """
    Returns details for a single item.
    """
    details = recommender_service.get_item_detail(item_id)
    if not details:
        raise HTTPException(status_code=404, detail="Item not found")
    return details


@router.get("/reviews/", response_model=List[Review])
def get_item_reviews(item_id: str):
    """
    Returns all reviews for a single item.
    """
    reviews = recommender_service.get_item_reviews(item_id)
    if not reviews:
        return []

    return reviews


@router.get("/users", response_model=List[str])
def get_all_users():
    """
    Returns a list of all available user IDs.
    """
    return recommender_service.get_all_users()


@router.get("/user/{user_id}", response_model=UserProfile)
def get_user_profile(user_id: str):
    """
    Returns user profile and recently reviewed items.
    """
    profile = recommender_service.get_user_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="User not found")
    return profile
