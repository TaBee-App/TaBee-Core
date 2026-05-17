package com.tabee.backend.auth;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.Pattern;
import jakarta.validation.constraints.Size;

import com.tabee.backend.user.UserDtos.UserRequest;
import com.tabee.backend.user.UserDtos.UserResponse;

public final class AuthDtos {
    private AuthDtos() {
    }

    public record RegisterRequest(
            @NotBlank @Size(max = 50) String username,
            @NotBlank @Email String email,
            @NotBlank @Size(min = 8, max = 100) String password,
            @Size(max = 100) String fullName,
            @NotBlank @Pattern(regexp = "\\d{6}", message = "Verification code must be 6 digits") String verificationCode
    ) {
        public UserRequest toUserRequest() {
            return new UserRequest(username, email, password, fullName);
        }
    }

    public record RegisterCodeRequest(
            @NotBlank @Size(max = 50) String username,
            @NotBlank @Email String email,
            @NotBlank @Size(min = 8, max = 100) String password,
            @Size(max = 100) String fullName
    ) {
        public UserRequest toUserRequest() {
            return new UserRequest(username, email, password, fullName);
        }
    }

    public record RegisterCodeResponse(
            String message,
            String devCode
    ) {
    }

    public record PasswordResetCodeRequest(
            @NotBlank @Email String email
    ) {
    }

    public record PasswordResetCodeResponse(
            String message,
            String devCode
    ) {
    }

    public record PasswordResetConfirmRequest(
            @NotBlank @Email String email,
            @NotBlank @Pattern(regexp = "\\d{6}", message = "Verification code must be 6 digits") String verificationCode,
            @NotBlank @Size(min = 8, max = 100) String newPassword
    ) {
    }

    public record LoginRequest(
            @NotBlank String usernameOrEmail,
            @NotBlank String password
    ) {
    }

    public record AuthResponse(
            UserResponse user,
            String token,
            String message
    ) {
    }
}
