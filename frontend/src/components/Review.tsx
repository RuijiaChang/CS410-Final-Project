import React from 'react';
import type { Review } from '../services/types';

interface ReviewProps {
  review: Review;
}

const ReviewComponent: React.FC<ReviewProps> = ({ review }) => {
  return (
    <div className="review">
      <h4>{review.review_title}</h4>
      <p>Rating: {review.rating}</p>
      <p>{review.review_text}</p>
      {review.timestamp && <p><small>By: {review.user_id} at {new Date(review.timestamp * 1000).toLocaleString()}</small></p>}
    </div>
  );
};

export default ReviewComponent;
