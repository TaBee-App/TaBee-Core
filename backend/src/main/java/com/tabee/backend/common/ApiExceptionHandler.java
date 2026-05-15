package com.tabee.backend.common;

import java.time.OffsetDateTime;
import java.util.List;

import jakarta.servlet.http.HttpServletRequest;

import org.springframework.dao.DataIntegrityViolationException;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.validation.FieldError;
import org.springframework.web.HttpRequestMethodNotSupportedException;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.MissingServletRequestParameterException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.method.annotation.MethodArgumentTypeMismatchException;
import org.springframework.web.multipart.MaxUploadSizeExceededException;
import org.springframework.web.multipart.support.MissingServletRequestPartException;
import org.springframework.web.server.ResponseStatusException;

@RestControllerAdvice
public class ApiExceptionHandler {

    @ExceptionHandler(ResponseStatusException.class)
    public ResponseEntity<ApiErrorResponse> handleResponseStatus(ResponseStatusException exception,
                                                                 HttpServletRequest request) {
        HttpStatus status = HttpStatus.valueOf(exception.getStatusCode().value());
        return build(status, exception.getReason(), request.getRequestURI(), List.of());
    }

    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<ApiErrorResponse> handleValidation(MethodArgumentNotValidException exception,
                                                             HttpServletRequest request) {
        List<ApiFieldError> fields = exception.getBindingResult().getFieldErrors().stream()
                .map(this::toApiFieldError)
                .toList();
        return build(HttpStatus.BAD_REQUEST, "Request validation failed", request.getRequestURI(), fields);
    }

    @ExceptionHandler(MissingServletRequestParameterException.class)
    public ResponseEntity<ApiErrorResponse> handleMissingParameter(MissingServletRequestParameterException exception,
                                                                   HttpServletRequest request) {
        String message = "Missing required request parameter: " + exception.getParameterName();
        return build(HttpStatus.BAD_REQUEST, message, request.getRequestURI(), List.of());
    }

    @ExceptionHandler(MissingServletRequestPartException.class)
    public ResponseEntity<ApiErrorResponse> handleMissingPart(MissingServletRequestPartException exception,
                                                              HttpServletRequest request) {
        String message = "Missing required multipart field: " + exception.getRequestPartName();
        return build(HttpStatus.BAD_REQUEST, message, request.getRequestURI(), List.of());
    }

    @ExceptionHandler(MethodArgumentTypeMismatchException.class)
    public ResponseEntity<ApiErrorResponse> handleTypeMismatch(MethodArgumentTypeMismatchException exception,
                                                               HttpServletRequest request) {
        String message = "Invalid value for parameter: " + exception.getName();
        return build(HttpStatus.BAD_REQUEST, message, request.getRequestURI(), List.of());
    }

    @ExceptionHandler(HttpMessageNotReadableException.class)
    public ResponseEntity<ApiErrorResponse> handleUnreadableJson(HttpMessageNotReadableException exception,
                                                                 HttpServletRequest request) {
        return build(HttpStatus.BAD_REQUEST, "Malformed or unreadable request body", request.getRequestURI(), List.of());
    }

    @ExceptionHandler(HttpRequestMethodNotSupportedException.class)
    public ResponseEntity<ApiErrorResponse> handleMethodNotSupported(HttpRequestMethodNotSupportedException exception,
                                                                     HttpServletRequest request) {
        return build(HttpStatus.METHOD_NOT_ALLOWED, "HTTP method is not supported for this endpoint",
                request.getRequestURI(), List.of());
    }

    @ExceptionHandler(MaxUploadSizeExceededException.class)
    public ResponseEntity<ApiErrorResponse> handleMaxUploadSizeExceeded(HttpServletRequest request) {
        return build(HttpStatus.PAYLOAD_TOO_LARGE, "Audio file is too large", request.getRequestURI(), List.of());
    }

    @ExceptionHandler(DataIntegrityViolationException.class)
    public ResponseEntity<ApiErrorResponse> handleDataIntegrity(DataIntegrityViolationException exception,
                                                                HttpServletRequest request) {
        return build(HttpStatus.CONFLICT, "Database constraint violation", request.getRequestURI(), List.of());
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<ApiErrorResponse> handleUnexpected(Exception exception, HttpServletRequest request) {
        return build(HttpStatus.INTERNAL_SERVER_ERROR, "Unexpected server error", request.getRequestURI(), List.of());
    }

    private ApiFieldError toApiFieldError(FieldError fieldError) {
        return new ApiFieldError(fieldError.getField(), fieldError.getDefaultMessage());
    }

    private ResponseEntity<ApiErrorResponse> build(HttpStatus status, String message, String path,
                                                   List<ApiFieldError> fieldErrors) {
        ApiErrorResponse response = new ApiErrorResponse(
                OffsetDateTime.now(),
                status.value(),
                status.getReasonPhrase(),
                message,
                path,
                fieldErrors
        );
        return ResponseEntity.status(status).body(response);
    }

    public record ApiErrorResponse(
            OffsetDateTime timestamp,
            int status,
            String error,
            String message,
            String path,
            List<ApiFieldError> fieldErrors
    ) {
    }

    public record ApiFieldError(
            String field,
            String message
    ) {
    }
}
