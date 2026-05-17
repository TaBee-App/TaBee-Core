package com.tabee.backend.user;

import java.util.List;
import java.time.OffsetDateTime;

import org.springframework.http.HttpStatus;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

import com.tabee.backend.common.ImageStorageService;
import com.tabee.backend.user.UserDtos.UserRequest;
import com.tabee.backend.user.UserDtos.UserDeleteRequest;
import com.tabee.backend.user.UserDtos.EmailUpdateCodeRequest;
import com.tabee.backend.user.UserDtos.EmailUpdateCodeResponse;
import com.tabee.backend.user.UserDtos.EmailUpdateConfirmRequest;
import com.tabee.backend.user.UserDtos.UserUpdateRequest;

@Service
public class UserService {
    private final UserRepository userRepository;
    private final UserFollowRepository userFollowRepository;
    private final PasswordEncoder passwordEncoder;
    private final EmailUpdateVerificationService emailUpdateVerificationService;
    private final ImageStorageService imageStorageService;

    public UserService(UserRepository userRepository,
                       UserFollowRepository userFollowRepository,
                       PasswordEncoder passwordEncoder,
                       EmailUpdateVerificationService emailUpdateVerificationService,
                       ImageStorageService imageStorageService) {
        this.userRepository = userRepository;
        this.userFollowRepository = userFollowRepository;
        this.passwordEncoder = passwordEncoder;
        this.emailUpdateVerificationService = emailUpdateVerificationService;
        this.imageStorageService = imageStorageService;
    }

    public List<User> findAll() {
        return userRepository.findAll();
    }

    public List<User> search(String query) {
        String normalized = query == null ? "" : query.trim();
        if (normalized.startsWith("@")) {
            normalized = normalized.substring(1).trim();
        }
        if (normalized.isBlank()) {
            return userRepository.findTop24ByOrderByUsernameAsc();
        }
        return userRepository.findTop24ByUsernameContainingIgnoreCaseOrderByUsernameAsc(normalized);
    }

    public List<User> findDiscoveryUsers() {
        return userRepository.findTop10ByOrderByCreatedAtDesc();
    }

    public User findById(Long id) {
        return userRepository.findById(id)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "User not found"));
    }

    public User create(UserRequest request) {
        validateNewUser(request);

        User user = new User();
        user.setUsername(request.username());
        user.setEmail(request.email());
        user.setFullName(request.fullName());
        user.setEmailConfirmed(true);
        user.setPasswordHash(passwordEncoder.encode(request.password()));
        return userRepository.save(user);
    }

    public void validateNewUser(UserRequest request) {
        ensureUsernameAvailable(request.username(), null);
        ensureEmailAvailable(request.email(), null);
        PasswordPolicy.validate(request.password(), request.username(), request.email(), request.fullName());
    }

    public User update(Long id, UserUpdateRequest request) {
        User user = findById(id);

        if (!passwordEncoder.matches(request.currentPassword(), user.getPasswordHash())) {
            throw new ResponseStatusException(HttpStatus.FORBIDDEN, "Current password is incorrect");
        }

        if (request.username() != null && !request.username().isBlank()) {
            updateUsername(user, request.username());
        }
        if (request.email() != null && !request.email().isBlank()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Use the email verification flow to change email");
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

    public EmailUpdateCodeResponse requestEmailUpdateCode(Long id, EmailUpdateCodeRequest request) {
        User user = findById(id);
        ensureCurrentPassword(user, request.currentPassword());
        ensureEmailAvailable(request.email(), id);
        return emailUpdateVerificationService.sendCode(id, request.email());
    }

    public User confirmEmailUpdate(Long id, EmailUpdateConfirmRequest request) {
        User user = findById(id);
        ensureCurrentPassword(user, request.currentPassword());
        ensureEmailAvailable(request.email(), id);
        emailUpdateVerificationService.verify(id, request.email(), request.verificationCode());
        user.setEmail(request.email().trim().toLowerCase(java.util.Locale.ROOT));
        user.setEmailConfirmed(true);
        return userRepository.save(user);
    }

    public void delete(Long id) {
        User user = findById(id);
        imageStorageService.deleteQuietly(user.getProfileImageFilename());
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

        imageStorageService.deleteQuietly(user.getProfileImageFilename());
        userRepository.delete(user);
    }

    public User updateProfileImage(User user, MultipartFile file) {
        String previous = user.getProfileImageFilename();
        String filename = imageStorageService.store(file, "user-" + user.getId());
        user.setProfileImageFilename(filename);
        User saved = userRepository.save(user);
        imageStorageService.deleteQuietly(previous);
        return saved;
    }

    public User removeProfileImage(User user) {
        String previous = user.getProfileImageFilename();
        user.setProfileImageFilename(null);
        User saved = userRepository.save(user);
        imageStorageService.deleteQuietly(previous);
        return saved;
    }

    public String profileImageUrl(User user) {
        return imageStorageService.url(user.getProfileImageFilename());
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

    private void updateUsername(User user, String username) {
        String nextUsername = username.trim();
        if (nextUsername.equals(user.getUsername())) {
            return;
        }
        OffsetDateTime lastUsernameChange = user.getUsernameUpdatedAt();
        if (lastUsernameChange != null && lastUsernameChange.plusMonths(1).isAfter(OffsetDateTime.now())) {
            throw new ResponseStatusException(HttpStatus.TOO_MANY_REQUESTS, "Username can be changed once per month");
        }
        ensureUsernameAvailable(nextUsername, user.getId());
        user.setUsername(nextUsername);
        user.setUsernameUpdatedAt(OffsetDateTime.now());
    }

    private void ensureEmailAvailable(String email, Long currentUserId) {
        userRepository.findByEmail(email).ifPresent(existing -> {
            if (!existing.getId().equals(currentUserId)) {
                throw new ResponseStatusException(HttpStatus.CONFLICT, "Email already exists");
            }
        });
    }

    private void ensureCurrentPassword(User user, String currentPassword) {
        if (!passwordEncoder.matches(currentPassword, user.getPasswordHash())) {
            throw new ResponseStatusException(HttpStatus.FORBIDDEN, "Current password is incorrect");
        }
    }
}
