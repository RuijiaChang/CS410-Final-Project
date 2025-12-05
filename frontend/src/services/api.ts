import axios from 'axios';
import type { ProductCard, ProductDetails, Review, UserProfile } from './types';

const API_URL = 'http://127.0.0.1:8000';

export const getRecommendations = async (userId: string, k: number = 10): Promise<ProductCard[]> => {
  const response = await axios.get(`${API_URL}/recommend/`, {
    params: { user_id: userId, k },
  });
  return response.data;
};

export const getItemDetails = async (itemId: string): Promise<ProductDetails> => {
  const response = await axios.get(`${API_URL}/item/${itemId}`);
  return response.data;
};

export const getItemReviews = async (itemId: string): Promise<Review[]> => {
  const response = await axios.get(`${API_URL}/reviews/`, {
    params: { item_id: itemId },
  });
  return response.data;
};

export const getAllUsers = async (): Promise<string[]> => {
  const response = await axios.get(`${API_URL}/users`);
  return response.data;
};

export const getUserProfile = async (userId: string): Promise<UserProfile> => {
    const response = await axios.get(`${API_URL}/user/${userId}`);
    return response.data;
};
