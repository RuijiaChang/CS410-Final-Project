export interface ProductCard {
    item_id: string;
    title: string;
    price?: number;
    cover_image?: string;
    main_category?: string;
    average_rating?: number;
    rating_number?: number;
}

export interface ProductDetails extends ProductCard {
    images?: string[];
    description?: string;
    features?: string[];
    categories?: string[];
    store?: string;
    details?: Record<string, any>;
}

export interface Review {
    user_id?: string;
    item_id?: string;
    rating?: number;
    review_title?: string;
    review_text?: string;
    timestamp?: number;
}

export interface UserProfile {
    user_id: string;
    items: Review[];
}
