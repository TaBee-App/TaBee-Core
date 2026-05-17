package com.tabee.backend.auth;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Email;
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
            @Size(max = 100) String fullName
    ) {
        public UserRequest toUserRequest() {
            return new UserRequest(username, email, password, fullName);
        }
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
