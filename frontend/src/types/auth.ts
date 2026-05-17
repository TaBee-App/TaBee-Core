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
  profileImageUrl?: string | null;
  createdAt?: string;
  updatedAt?: string;
  usernameUpdatedAt?: string | null;
}

export interface PublicUserProfile {
  id: number;
  username: string;
  fullName?: string | null;
  profileImageUrl?: string | null;
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
  verificationCode: string;
}

export interface RegisterCodeRequest {
  username: string;
  email: string;
  password: string;
  fullName?: string;
}

export interface RegisterCodeResponse {
  message: string;
  devCode?: string | null;
}

export interface PasswordResetCodeRequest {
  email: string;
}

export interface PasswordResetCodeResponse {
  message: string;
  devCode?: string | null;
}

export interface PasswordResetConfirmRequest {
  email: string;
  verificationCode: string;
  newPassword: string;
}

export interface UserUpdateRequest {
  currentPassword: string;
  username?: string;
  email?: string;
  password?: string;
  fullName?: string;
}

export interface EmailUpdateCodeRequest {
  email: string;
  currentPassword: string;
}

export interface EmailUpdateCodeResponse {
  message: string;
  devCode?: string | null;
}

export interface EmailUpdateConfirmRequest {
  email: string;
  currentPassword: string;
  verificationCode: string;
}

export interface DeleteAccountRequest {
  currentPassword: string;
  confirmation: string;
}
