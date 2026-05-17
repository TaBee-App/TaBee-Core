export interface ApiErrorResponse {
  timestamp?: string;
  status?: number;
  error?: string;
  message?: string;
  path?: string;
  fieldErrors?: Array<{
    field: string;
    message: string;
  }>;
}

export interface UserProfile {
  id: number;
  username: string;
  email: string;
  fullName?: string | null;
  createdAt?: string;
  updatedAt?: string;
}

export interface PublicUserProfile {
  id: number;
  username: string;
  fullName?: string | null;
  createdAt?: string;
  followedByCurrentUser: boolean;
  followerCount: number;
  followingCount: number;
}

export interface AuthResponse {
  user: UserProfile;
  token: string;
  message: string;
}

export interface LoginRequest {
  usernameOrEmail: string;
  password: string;
}

export interface RegisterRequest {
  username: string;
  email: string;
  password: string;
  fullName?: string;
}

export interface UserUpdateRequest {
  currentPassword: string;
  username?: string;
  email?: string;
  password?: string;
  fullName?: string;
}

export interface DeleteAccountRequest {
  currentPassword: string;
  confirmation: string;
}
