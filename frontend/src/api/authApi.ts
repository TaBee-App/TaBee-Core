import { apiFetch } from "./apiClient";
import { clearAuthSession, saveAuthSession, saveCurrentUser } from "./authSession";
import type {
  AuthResponse,
  DeleteAccountRequest,
  EmailUpdateCodeRequest,
  EmailUpdateCodeResponse,
  EmailUpdateConfirmRequest,
  LoginRequest,
  PasswordResetCodeRequest,
  PasswordResetCodeResponse,
  PasswordResetConfirmRequest,
  RegisterCodeRequest,
  RegisterCodeResponse,
  PublicUserProfile,
  RegisterRequest,
  UserUpdateRequest
} from "../types/auth";

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

export async function requestRegisterCode(request: RegisterCodeRequest) {
  return apiFetch<RegisterCodeResponse>("/api/auth/register/code", {
    method: "POST",
    authenticated: false,
    body: JSON.stringify(request)
  });
}

export async function requestPasswordResetCode(request: PasswordResetCodeRequest) {
  return apiFetch<PasswordResetCodeResponse>("/api/auth/password-reset/code", {
    method: "POST",
    authenticated: false,
    body: JSON.stringify(request)
  });
}

export async function confirmPasswordReset(request: PasswordResetConfirmRequest) {
  await apiFetch<null>("/api/auth/password-reset/confirm", {
    method: "POST",
    authenticated: false,
    body: JSON.stringify(request)
  });
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

export async function requestEmailUpdateCode(request: EmailUpdateCodeRequest) {
  return apiFetch<EmailUpdateCodeResponse>("/api/users/me/email/code", {
    method: "POST",
    body: JSON.stringify(request)
  });
}

export async function confirmEmailUpdate(request: EmailUpdateConfirmRequest) {
  const user = await apiFetch<PublicUserProfile & { email: string } & { updatedAt?: string; usernameUpdatedAt?: string | null }>("/api/users/me/email/confirm", {
    method: "POST",
    body: JSON.stringify(request)
  });
  saveCurrentUser(user);
  return user;
}

export async function updateProfileImage(file: File) {
  const formData = new FormData();
  formData.append("file", file);
  const user = await apiFetch<PublicUserProfile & { email: string } & { updatedAt?: string; usernameUpdatedAt?: string | null; profileImageUrl?: string | null }>("/api/users/me/profile-image", {
    method: "POST",
    body: formData
  });
  saveCurrentUser(user);
  return user;
}

export async function removeProfileImage() {
  const user = await apiFetch<PublicUserProfile & { email: string } & { updatedAt?: string; usernameUpdatedAt?: string | null; profileImageUrl?: string | null }>("/api/users/me/profile-image", {
    method: "DELETE"
  });
  saveCurrentUser(user);
  return user;
}

export async function deleteMe(request: DeleteAccountRequest) {
  await apiFetch<null>("/api/users/me", {
    method: "DELETE",
    body: JSON.stringify(request)
  });
  clearAuthSession();
}

export async function searchUsers(query: string) {
  return apiFetch<PublicUserProfile[]>(`/api/users/search?q=${encodeURIComponent(query)}`);
}

export async function listDiscoveryUsers() {
  return apiFetch<PublicUserProfile[]>("/api/users/discovery");
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
