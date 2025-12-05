from pydantic import BaseModel
from typing import List, Optional


class ProductCard(BaseModel):
    item_id: str
    title: str
    price: Optional[float] = None
    cover_image: Optional[str] = None
    main_category: Optional[str] = None
    average_rating: Optional[float] = None
    rating_number: Optional[int] = None


class ProductDetails(ProductCard):
    images: Optional[List[str]] = None
    description: Optional[str] = None
    features: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    store: Optional[str] = None
    details: Optional[dict] = None


class Review(BaseModel):
    user_id: Optional[str] = None
    item_id: Optional[str] = None
    rating: Optional[int] = None
    review_title: Optional[str] = None
    review_text: Optional[str] = None
    timestamp: Optional[int] = None


class UserProfile(BaseModel):
    user_id: str
    items: List[Review]