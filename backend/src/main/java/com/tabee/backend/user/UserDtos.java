package com.tabee.backend.user;

import java.time.OffsetDateTime;

import jakarta.validation.constraints.Email;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

public final class UserDtos {
    private UserDtos() {
    }

    public record UserRequest(
            @NotBlank @Size(max = 50) String username,
            @NotBlank @Email String email,
            @NotBlank @Size(min = 8, max = 100) String password,
            @Size(max = 100) String fullName
    ) {
    }

    public record UserUpdateRequest(
            @NotBlank @Size(min = 6, max = 100) String currentPassword,
            @Size(max = 50) String username,
            @Email String email,
            @Size(min = 8, max = 100) String password,
            @Size(max = 100) String fullName
    ) {
    }

    public record UserDeleteRequest(
            @NotBlank @Size(min = 6, max = 100) String currentPassword,
            @NotBlank String confirmation
    ) {
    }

    public record UserResponse(
            Long id,
            String username,
            String email,
            String fullName,
            boolean emailConfirmed,
            String profileImageUrl,
            OffsetDateTime usernameUpdatedAt,
            OffsetDateTime createdAt,
            OffsetDateTime updatedAt
    ) {
        public static UserResponse from(User user, String profileImageUrl) {
            return new UserResponse(
                    user.getId(),
                    user.getUsername(),
                    user.getEmail(),
                    user.getFullName(),
                    user.isEmailConfirmed(),
                    profileImageUrl,
                    user.getUsernameUpdatedAt(),
                    user.getCreatedAt(),
                    user.getUpdatedAt()
            );
        }
    }

    public record PublicUserResponse(
            Long id,
            String username,
            String fullName,
            String profileImageUrl,
            OffsetDateTime createdAt,
            boolean followedByCurrentUser,
            long followerCount,
            long followingCount
    ) {
        public static PublicUserResponse from(User user, String profileImageUrl, boolean followedByCurrentUser,
                                              long followerCount, long followingCount) {
            return new PublicUserResponse(
                    user.getId(),
                    user.getUsername(),
                    user.getFullName(),
                    profileImageUrl,
                    user.getCreatedAt(),
                    followedByCurrentUser,
                    followerCount,
                    followingCount
            );
        }
    }

    public record EmailUpdateCodeRequest(
            @NotBlank @Email String email,
            @NotBlank @Size(min = 6, max = 100) String currentPassword
    ) {
    }

    public record EmailUpdateCodeResponse(
            String message,
            String devCode
    ) {
    }

    public record EmailUpdateConfirmRequest(
            @NotBlank @Email String email,
            @NotBlank @Size(min = 6, max = 100) String currentPassword,
            @NotBlank @jakarta.validation.constraints.Pattern(regexp = "\\d{6}", message = "Verification code must be 6 digits")
            String verificationCode
    ) {
    }
}
