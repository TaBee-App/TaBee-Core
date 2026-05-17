import { apiFetch } from "./apiClient";
import { clearAuthSession, saveAuthSession, saveCurrentUser } from "./authSession";
import type { AuthResponse, LoginRequest, PublicUserProfile, RegisterRequest, UserUpdateRequest } from "../types/auth";

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

export async function updateMe(request: UserUpdateRequest) {
  const user = await apiFetch<PublicUserProfile & { email: string } & { updatedAt?: string }>("/api/users/me", {
    method: "PUT",
    body: JSON.stringify(request)
  });
  saveCurrentUser(user);
  return user;
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

export async function removeFollower(userId: number) {
  return apiFetch<PublicUserProfile>(`/api/users/${userId}/follower`, {
    method: "DELETE"
  });
}

export async function getUserFollowers(userId: string) {
  return apiFetch<PublicUserProfile[]>(`/api/users/${userId}/followers`);
}

export async function getUserFollowing(userId: string) {
  return apiFetch<PublicUserProfile[]>(`/api/users/${userId}/following`);
}
