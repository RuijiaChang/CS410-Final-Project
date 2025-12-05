import React, { useEffect, useState } from 'react';
import { Link, useParams } from 'react-router-dom';
import { getItemDetails, getItemReviews } from '../services/api';
import type { ProductDetails, Review } from '../services/types';
import ReviewComponent from '../components/Review';
import { Carousel } from 'react-responsive-carousel';
import "react-responsive-carousel/lib/styles/carousel.min.css"; // requires a loader

const ProductDetailsPage: React.FC = () => {
  const { itemId } = useParams<{ itemId: string }>();
  const [product, setProduct] = useState<ProductDetails | null>(null);
  const [reviews, setReviews] = useState<Review[]>([]);

  useEffect(() => {
    if (itemId) {
      getItemDetails(itemId).then(setProduct);
      getItemReviews(itemId).then(setReviews);
    }
  }, [itemId]);

  if (!product) {
    return <div>Loading...</div>;
  }

  return (
    <div className="product-details-page">
        <Link to="/" className="back-link">← Back to Home</Link>
        <div className="product-header">
            <h2>{product.title}</h2>
            {product.store && <p className="store-info">Sold by: {product.store}</p>}
        </div>

        <Carousel showArrows={true} infiniteLoop={true} useKeyboardArrows={true} autoPlay={true}>
            {product.images?.map((img, index) => (
                <div key={index}>
                    <img src={img} alt={`${product.title} view ${index + 1}`} />
                </div>
            ))}
        </Carousel>
        
        
        <div className="product-content-grid">
            <div className="grid-item description-box">
                <h3>Description</h3>
                {product.description && product.description !== "[]" ? (
                    <p>{product.description}</p>
                ) : (
                    <p>No description available.</p>
                )}
            </div>
            <div className="grid-item features-box">
                <h3>Features</h3>
                <ul>
                    {product.features?.map((feature, index) => (
                        <li key={index}>{feature}</li>
                    ))}
                </ul>
            </div>
            <div className="grid-item categories-box">
                <h3>Categories</h3>
                <ul>
                    {product.categories?.map((cat, index) => <li key={index}>{cat}</li>)}
                </ul>
            </div>
            <div className="grid-item details-box">
                <h3>Details</h3>
                {product.details ? (
                    <ul>
                        {Object.entries(product.details).map(([key, value]) => (
                            <li key={key}><strong>{key}:</strong> {String(value)}</li>
                        ))}
                    </ul>
                ) : (
                    <p>No details available.</p>
                )}
            </div>
        </div>

        <div className="reviews-section">
            <h3>Customer Reviews</h3>
            <div className="reviews-grid">
                {reviews.length > 0 ? (
                reviews.map((review, index) => (
                    <div className="review-item" key={index}>
                        <ReviewComponent review={review} />
                    </div>
                ))
                ) : (
                <p>No reviews yet.</p>
                )}
            </div>
        </div>
    </div>
  );
};

export default ProductDetailsPage;
