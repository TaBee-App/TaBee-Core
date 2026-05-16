import { apiFetch } from "./apiClient";
import { clearAuthSession, saveAuthSession } from "./authSession";
import type { AuthResponse, LoginRequest, PublicUserProfile, RegisterRequest } from "../types/auth";

export async function login(request: LoginRequest) {
  const response = await apiFetch<AuthResponse>("/api/auth/login", {
    method: "POST",
    authenticated: false,
    body: JSON.stringify(request)
  });

  saveAuthSession(response);
  return response;
}

export async function register(request: RegisterRequest) {
  const response = await apiFetch<AuthResponse>("/api/auth/register", {
    method: "POST",
    authenticated: false,
    body: JSON.stringify(request)
  });

  saveAuthSession(response);
  return response;
}

export function logout() {
  clearAuthSession();
}

export async function searchUsers(query: string) {
  return apiFetch<PublicUserProfile[]>(`/api/users/search?q=${encodeURIComponent(query)}`);
}

export async function getPublicUser(userId: string) {
  return apiFetch<PublicUserProfile>(`/api/users/${userId}`);
}

export async function followUser(userId: number) {
  return apiFetch<PublicUserProfile>(`/api/users/${userId}/follow`, {
    method: "POST"
  });
}

export async function unfollowUser(userId: number) {
  return apiFetch<PublicUserProfile>(`/api/users/${userId}/follow`, {
    method: "DELETE"
  });
}
