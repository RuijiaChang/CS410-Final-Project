import React from 'react';
import { Link } from 'react-router-dom';
import type { ProductCard } from '../services/types';

interface ProductCardProps {
  product: ProductCard;
}

const ProductCardComponent: React.FC<ProductCardProps> = ({ product }) => {
  return (
    <div className="product-card">
      <Link to={`/item/${product.item_id}`}>
        <img src={product.cover_image} alt={product.title} />
        <h3>{product.title}</h3>
        {product.main_category && <p className="category">{product.main_category}</p>}
        <p>{product.price}</p>
      </Link>
    </div>
  );
};

export default ProductCardComponent;
