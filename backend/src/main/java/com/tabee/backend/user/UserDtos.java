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
            @NotBlank @Size(min = 6, max = 100) String password,
            @Size(max = 100) String fullName
    ) {
    }

    public record UserUpdateRequest(
            @Size(max = 50) String username,
            @Email String email,
            @Size(min = 6, max = 100) String password,
            @Size(max = 100) String fullName
    ) {
    }

    public record UserResponse(
            Long id,
            String username,
            String email,
            String fullName,
            OffsetDateTime createdAt,
            OffsetDateTime updatedAt
    ) {
        public static UserResponse from(User user) {
            return new UserResponse(
                    user.getId(),
                    user.getUsername(),
                    user.getEmail(),
                    user.getFullName(),
                    user.getCreatedAt(),
                    user.getUpdatedAt()
            );
        }
    }

    public record PublicUserResponse(
            Long id,
            String username,
            String fullName,
            OffsetDateTime createdAt,
            boolean followedByCurrentUser,
            long followerCount,
            long followingCount
    ) {
        public static PublicUserResponse from(User user, boolean followedByCurrentUser,
                                              long followerCount, long followingCount) {
            return new PublicUserResponse(
                    user.getId(),
                    user.getUsername(),
                    user.getFullName(),
                    user.getCreatedAt(),
                    followedByCurrentUser,
                    followerCount,
                    followingCount
            );
        }
    }
}
