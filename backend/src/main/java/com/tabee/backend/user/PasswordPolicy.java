package com.tabee.backend.user;

import java.util.List;
import java.util.Locale;
import java.util.regex.Pattern;

import org.springframework.http.HttpStatus;
import org.springframework.web.server.ResponseStatusException;

public final class PasswordPolicy {
    private static final Pattern LOWERCASE = Pattern.compile("[a-z]");
    private static final Pattern UPPERCASE = Pattern.compile("[A-Z]");
    private static final Pattern DIGIT = Pattern.compile("\\d");
    private static final Pattern SPECIAL = Pattern.compile("[^A-Za-z0-9]");
    private static final Pattern REPEATED_CHARACTER = Pattern.compile("(.)\\1\\1");
    private static final List<String> COMMON_SEQUENCES = List.of(
            "0123", "1234", "2345", "3456", "4567", "5678", "6789",
            "9876", "8765", "7654", "6543", "5432", "4321", "3210",
            "abcd", "bcde", "cdef", "qwer", "asdf", "zxcv"
    );

    private PasswordPolicy() {
    }

    public static void validate(String password, String username, String email, String fullName) {
        String issue = firstIssue(password, username, email, fullName);
        if (issue != null) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, issue);
        }
    }

    private static String firstIssue(String password, String username, String email, String fullName) {
        if (password == null || password.length() < 8) {
            return "Password must be at least 8 characters.";
        }
        if (password.length() > 100) {
            return "Password must be at most 100 characters.";
        }
        if (!LOWERCASE.matcher(password).find()) {
            return "Password must include a lowercase letter.";
        }
        if (!UPPERCASE.matcher(password).find()) {
            return "Password must include an uppercase letter.";
        }
        if (!DIGIT.matcher(password).find()) {
            return "Password must include a number.";
        }
        if (!SPECIAL.matcher(password).find()) {
            return "Password must include a special character.";
        }
        String normalizedPassword = password.toLowerCase(Locale.ROOT);
        if (REPEATED_CHARACTER.matcher(normalizedPassword).find()) {
            return "Password cannot repeat the same character 3 times in a row.";
        }
        if (containsCommonSequence(normalizedPassword)) {
            return "Password cannot contain simple keyboard or number sequences.";
        }
        if (containsIdentity(normalizedPassword, username, email, fullName)) {
            return "Password cannot include your username, email, or name.";
        }
        return null;
    }

    private static boolean containsCommonSequence(String password) {
        return COMMON_SEQUENCES.stream().anyMatch(password::contains);
    }

    private static boolean containsIdentity(String password, String username, String email, String fullName) {
        if (containsToken(password, username)) {
            return true;
        }
        if (email != null) {
            for (String token : email.split("[@._\\-+]+")) {
                if (containsToken(password, token)) {
                    return true;
                }
            }
        }
        if (fullName != null) {
            for (String token : fullName.split("\\s+")) {
                if (containsToken(password, token)) {
                    return true;
                }
            }
        }
        return false;
    }

    private static boolean containsToken(String password, String token) {
        if (token == null) {
            return false;
        }
        String normalized = token.trim().toLowerCase(Locale.ROOT);
        return normalized.length() >= 3 && password.contains(normalized);
    }
}
