package com.tabee.backend.user;

import java.util.List;

import org.springframework.http.HttpStatus;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.web.server.ResponseStatusException;

import com.tabee.backend.user.UserDtos.UserRequest;
import com.tabee.backend.user.UserDtos.UserDeleteRequest;
import com.tabee.backend.user.UserDtos.UserUpdateRequest;

@Service
public class UserService {
    private final UserRepository userRepository;
    private final UserFollowRepository userFollowRepository;
    private final PasswordEncoder passwordEncoder;

    public UserService(UserRepository userRepository,
                       UserFollowRepository userFollowRepository,
                       PasswordEncoder passwordEncoder) {
        this.userRepository = userRepository;
        this.userFollowRepository = userFollowRepository;
        this.passwordEncoder = passwordEncoder;
    }

    public List<User> findAll() {
        return userRepository.findAll();
    }

    public List<User> search(String query) {
        String normalized = query == null ? "" : query.trim();
        if (normalized.isBlank()) {
            return userRepository.findTop24ByOrderByUsernameAsc();
        }
        return userRepository.findTop24ByUsernameContainingIgnoreCaseOrFullNameContainingIgnoreCaseOrEmailContainingIgnoreCaseOrderByUsernameAsc(
                normalized,
                normalized,
                normalized
        );
    }

    public User findById(Long id) {
        return userRepository.findById(id)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "User not found"));
    }

    public User create(UserRequest request) {
        ensureUsernameAvailable(request.username(), null);
        ensureEmailAvailable(request.email(), null);
        PasswordPolicy.validate(request.password(), request.username(), request.email(), request.fullName());

        User user = new User();
        user.setUsername(request.username());
        user.setEmail(request.email());
        user.setFullName(request.fullName());
        user.setPasswordHash(passwordEncoder.encode(request.password()));
        return userRepository.save(user);
    }

    public User update(Long id, UserUpdateRequest request) {
        User user = findById(id);

        if (!passwordEncoder.matches(request.currentPassword(), user.getPasswordHash())) {
            throw new ResponseStatusException(HttpStatus.FORBIDDEN, "Current password is incorrect");
        }

        if (request.username() != null && !request.username().isBlank()) {
            ensureUsernameAvailable(request.username(), id);
            user.setUsername(request.username());
        }
        if (request.email() != null && !request.email().isBlank()) {
            ensureEmailAvailable(request.email(), id);
            user.setEmail(request.email());
        }
        if (request.fullName() != null) {
            user.setFullName(request.fullName());
        }
        if (request.password() != null && !request.password().isBlank()) {
            PasswordPolicy.validate(request.password(), user.getUsername(), user.getEmail(), user.getFullName());
            user.setPasswordHash(passwordEncoder.encode(request.password()));
        }

        return userRepository.save(user);
    }

    public void delete(Long id) {
        User user = findById(id);
        userRepository.delete(user);
    }

    public void delete(Long id, UserDeleteRequest request) {
        User user = findById(id);

        if (!passwordEncoder.matches(request.currentPassword(), user.getPasswordHash())) {
            throw new ResponseStatusException(HttpStatus.FORBIDDEN, "Current password is incorrect");
        }
        if (!"delete my account".equals(request.confirmation())) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Confirmation phrase does not match");
        }

        userRepository.delete(user);
    }

    public List<UserFollow> findFollowing(User user) {
        return userFollowRepository.findByFollower_IdOrderByCreatedAtDesc(user.getId());
    }

    public List<UserFollow> findFollowers(User user) {
        return userFollowRepository.findByFollowed_IdOrderByCreatedAtDesc(user.getId());
    }

    public boolean isFollowing(User follower, Long followedUserId) {
        return userFollowRepository.existsByFollower_IdAndFollowed_Id(follower.getId(), followedUserId);
    }

    public long followingCount(Long userId) {
        return userFollowRepository.countByFollower_Id(userId);
    }

    public long followerCount(Long userId) {
        return userFollowRepository.countByFollowed_Id(userId);
    }

    public User follow(User follower, Long followedUserId) {
        if (follower.getId().equals(followedUserId)) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "You cannot follow yourself");
        }

        User followed = findById(followedUserId);
        UserFollow follow = new UserFollow();
        follow.setId(new UserFollowId(follower.getId(), followed.getId()));
        follow.setFollower(follower);
        follow.setFollowed(followed);
        userFollowRepository.save(follow);
        return followed;
    }

    public User unfollow(User follower, Long followedUserId) {
        User followed = findById(followedUserId);
        userFollowRepository.deleteById(new UserFollowId(follower.getId(), followedUserId));
        return followed;
    }

    public User removeFollower(User current, Long followerUserId) {
        if (current.getId().equals(followerUserId)) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "You cannot remove yourself as a follower");
        }

        User follower = findById(followerUserId);
        userFollowRepository.deleteById(new UserFollowId(followerUserId, current.getId()));
        return follower;
    }

    private void ensureUsernameAvailable(String username, Long currentUserId) {
        userRepository.findByUsername(username).ifPresent(existing -> {
            if (!existing.getId().equals(currentUserId)) {
                throw new ResponseStatusException(HttpStatus.CONFLICT, "Username already exists");
            }
        });
    }

    private void ensureEmailAvailable(String email, Long currentUserId) {
        userRepository.findByEmail(email).ifPresent(existing -> {
            if (!existing.getId().equals(currentUserId)) {
                throw new ResponseStatusException(HttpStatus.CONFLICT, "Email already exists");
            }
        });
    }
}
