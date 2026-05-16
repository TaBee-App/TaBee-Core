import type { AuthResponse, UserProfile } from "../types/auth";

const TOKEN_KEY = "tabee.authToken";
const USER_KEY = "tabee.currentUser";

export function getAuthToken() {
  return localStorage.getItem(TOKEN_KEY);
}

export function getCurrentUser(): UserProfile | null {
  try {
    const storedUser = localStorage.getItem(USER_KEY);
    return storedUser ? (JSON.parse(storedUser) as UserProfile) : null;
  } catch {
    return null;
  }
}

export function saveAuthSession(response: AuthResponse) {
  localStorage.setItem(TOKEN_KEY, response.token);
  localStorage.setItem(USER_KEY, JSON.stringify(response.user));
}

export function clearAuthSession() {
  localStorage.removeItem(TOKEN_KEY);
  localStorage.removeItem(USER_KEY);
}

export function isAuthenticated() {
  return Boolean(getAuthToken());
}
