import React, { useState, useEffect } from 'react';
import { getRecommendations } from '../services/api';
import type { ProductCard } from '../services/types';
import ProductCardComponent from '../components/ProductCard';

const HomePage: React.FC = () => {
  const [userIdInput, setUserIdInput] = useState<string>('');
  const [submittedUserId, setSubmittedUserId] = useState<string>('');
  const [recommendations, setRecommendations] = useState<ProductCard[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    if (!submittedUserId) {
      setRecommendations([]);
      return;
    }

    const fetchRecommendations = async () => {
      setLoading(true);
      setError(null);
      try {
        const recs = await getRecommendations(submittedUserId);
        setRecommendations(recs);
        if (recs.length === 0) {
          setError("No recommendations found for this user.");
        }
      } catch (err) {
        setError('User not found. Please try a different ID.');
        setRecommendations([]);
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchRecommendations();
  }, [submittedUserId]);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    setSubmittedUserId(userIdInput);
  };

  return (
    <div className="home-page">
      <h1>Find Recommendations</h1>
      <form onSubmit={handleSearch} className="user-search-form">
        <input
          type="text"
          value={userIdInput}
          onChange={(e) => setUserIdInput(e.target.value)}
          placeholder="Enter User ID"
          className="user-id-input"
        />
        <button type="submit" className="search-button" disabled={loading}>
          {loading ? 'Loading...' : 'Get Recommendations'}
        </button>
      </form>

      {error && <p className="error-message">{error}</p>}

      <div className="recommendations">
        {recommendations.map((product) => (
          <ProductCardComponent key={product.item_id} product={product} />
        ))}
      </div>
    </div>
  );
};

export default HomePage;

