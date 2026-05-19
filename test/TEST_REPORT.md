# TaBee Test Report

This report documents the automated tests, manual test expectations, commands, and measured statistics for the TaBee project.

## Test Commands

Run backend automated tests:

```powershell
cd backend
mvn test
```

Run frontend regression build:

```powershell
cd frontend
npm run build
```

Run the backend locally:

```powershell
cd backend
mvn spring-boot:run
```

Run the frontend locally:

```powershell
cd frontend
npm run dev
```

## Latest Automated Results

Environment:

- Operating system: Windows 11
- Java: 17
- Backend test runner: Maven Surefire / JUnit 5
- Frontend build tool: Vite / TypeScript

Latest backend result:

- Command: `mvn test`
- Total tests: `29`
- Failures: `0`
- Errors: `0`
- Skipped: `0`
- Result: `PASS`
- Total Maven time: `7.941 s`

Latest frontend result:

- Command: `npm run build`
- TypeScript build: `PASS`
- Vite production build: `PASS`
- Output JS bundle: `1,467.61 kB`
- Output JS gzip size: `365.73 kB`
- Note: Vite reports a large chunk warning. This is not a failing test, but future code splitting is recommended.

## Automated Test Files

### `backend/src/test/java/com/tabee/backend/user/PasswordPolicyTest.java`

Purpose: unit tests for password validation rules.

Covered behavior:

- Accepts a strong password.
- Rejects a password containing the username.
- Rejects common numeric or keyboard sequences.
- Rejects passwords shorter than 8 characters.
- Rejects passwords without a special character.
- Rejects passwords with the same character repeated 3 times in a row.
- Rejects passwords containing an email token.

Expected result:

- Valid passwords pass without exception.
- Invalid passwords return `400 BAD_REQUEST` with a clear reason.

Latest result:

- Tests: `7`
- Result: `PASS`

### `backend/src/test/java/com/tabee/backend/common/ImageStorageServiceTest.java`

Purpose: unit and security tests for profile image and playlist cover storage.

Covered behavior:

- Stores PNG images and returns `/api/media/{filename}` URLs.
- Stores JPEG images with normalized `.jpg` extension.
- Rejects non-image content types.
- Rejects image uploads whose extension does not match allowed PNG/JPEG formats.
- Rejects path traversal attempts when resolving image files.
- Returns `null` for blank media URLs.
- Ignores blank or missing filenames during quiet deletion.

Expected result:

- Only PNG and JPEG images are accepted.
- Unsafe paths never resolve outside the configured media directory.

Latest result:

- Tests: `7`
- Result: `PASS`

### `backend/src/test/java/com/tabee/backend/tab/TabProcessingServiceTest.java`

Purpose: unit and security tests for uploaded audio validation before tab generation.

Covered behavior:

- Rejects unsupported file extensions such as `.txt`.
- Rejects missing audio files.
- Rejects supported extensions with unsafe content types.
- Rejects audio content types with unsupported filename extensions.

Expected result:

- Only `.wav`, `.mp3`, and `.mp4` uploads are allowed into the processing pipeline.
- Invalid files fail before the Python audio processor starts.

Latest result:

- Tests: `4`
- Result: `PASS`

### `backend/src/test/java/com/tabee/backend/security/AuthTokenServiceTest.java`

Purpose: unit tests for in-memory authentication token behavior.

Covered behavior:

- Issued tokens resolve back to the authenticated user.
- Unknown tokens return an empty result.
- Multiple issued tokens are unique.

Expected result:

- Login tokens must be unique and must map to the correct user id.
- Invalid tokens must not authenticate anyone.

Latest result:

- Tests: `3`
- Result: `PASS`

### `backend/src/test/java/com/tabee/backend/tab/TabDtosTest.java`

Purpose: unit tests for tab response mapping.

Covered behavior:

- Maps owner id, username, avatar URL, title, artist, tuning, tempo, favorite count, and tab JSON.
- Marks `createdByCurrentUser` correctly for the owner.
- Marks tabs from other users correctly.

Expected result:

- Frontend receives complete and correctly shaped tab data.
- Ownership flags are accurate.

Latest result:

- Tests: `2`
- Result: `PASS`

### `backend/src/test/java/com/tabee/backend/user/UserDtosTest.java`

Purpose: unit tests for private and public user profile response mapping.

Covered behavior:

- Private profile response includes email and email confirmation status.
- Public profile response does not expose the email address.
- Public profile response includes follower/following statistics.

Expected result:

- Private profile data is only used for the authenticated user.
- Public profile data avoids leaking private email addresses.

Latest result:

- Tests: `2`
- Result: `PASS`

### `backend/src/test/java/com/tabee/backend/TabeeBackendApplicationTests.java`

Purpose: lightweight application entry point test.

Covered behavior:

- Verifies that `TabeeBackendApplication` is annotated as a Spring Boot application.

Expected result:

- The application entry class remains valid without forcing local tests to connect to the real Aiven database.

Latest result:

- Tests: `1`
- Result: `PASS`

### `backend/src/test/java/com/tabee/backend/performance/PerformanceStatisticsTest.java`

Purpose: CPU, RAM/memory, and serialization benchmark tests that produce measurable statistics.

Generated metric files:

- `backend/target/test-metrics/cpu-statistics.md`
- `backend/target/test-metrics/memory-statistics.md`
- `backend/target/test-metrics/json-statistics.md`

Covered behavior:

- CPU benchmark for password policy validation.
- RAM/JVM memory benchmark for image storage.
- JSON serialization benchmark for generated tab note data.

Expected result:

- Password validation average time remains under `500 microseconds/request`.
- Small image storage memory delta remains under `50 MB`.
- Small image storage average time remains under `10 ms/file`.
- Generated tab note JSON serialization average time remains under `500 microseconds/record`.

Latest result:

- Tests: `3`
- Result: `PASS`

## Statistical Measurements

These measurements were generated by `PerformanceStatisticsTest` during the latest `mvn test` run.

### CPU Statistics

- Scenario: password policy validation benchmark
- Iterations: `20,000`
- Total time: `85.263 ms`
- Average time: `4.263 microseconds/request`
- Process CPU load before: `3.71%`
- Process CPU load after: `3.71%`
- Threshold: `< 500 microseconds/request`
- Result: `PASS`

### RAM / Memory Statistics

- Scenario: storing small profile or playlist images
- Files stored: `200`
- Payload per file: `8,192 bytes`
- Total time: `157.608 ms`
- Average time: `0.788 ms/file`
- JVM used memory before: `33,532,160 bytes`
- JVM used memory after: `34,578,688 bytes`
- JVM used memory delta: `1,046,528 bytes`
- Threshold: memory delta `< 50 MB`, average store time `< 10 ms/file`
- Result: `PASS`

### JSON Serialization Statistics

- Scenario: generated tab note JSON serialization
- Records serialized: `5,000`
- Total serialized bytes: `236,202 bytes`
- Total time: `50.969 ms`
- Average time: `10.194 microseconds/record`
- JVM used memory before: `26,190,080 bytes`
- JVM used memory after: `33,532,160 bytes`
- JVM used memory delta: `7,342,080 bytes`
- Threshold: `< 500 microseconds/record`
- Result: `PASS`

## Test Categories

### Unit Tests

Unit tests validate isolated backend behavior without requiring a database or running server.

Current coverage:

- Password policy validation
- Image upload validation and media path safety
- Audio upload validation before tab generation
- Authentication token generation and lookup
- User DTO mapping
- Tab DTO mapping
- Spring Boot entry point validation

Expected outcome:

- All valid inputs should pass.
- Invalid inputs should fail with explicit error messages and correct HTTP status codes.

### Integration Tests

Integration tests should verify controller, service, repository, and database behavior together.

Current status:

- Real database integration tests are intentionally not included in `mvn test`, because the Aiven database may be paused or offline and should not break local automated testing.

Recommended staging integration checklist:

- Register -> email verification -> login.
- Upload profile image.
- Create playlist and upload cover image.
- Upload `.wav`, `.mp3`, or `.mp4` file and generate a tab.
- Add a public tab to a playlist.
- Remove a tab from a playlist.
- Delete only owned resources.

Expected outcome:

- Database rows are created in the correct tables.
- Authenticated ownership rules are enforced.
- Users can add public tabs to their own playlists but cannot delete another user's tab.

### System Tests

System tests validate the full frontend + backend user flow.

Manual flow:

1. Start backend with `mvn spring-boot:run`.
2. Start frontend with `npm run dev`.
3. Register a new user.
4. Verify the email code.
5. Log in.
6. Upload a profile image.
7. Generate a tab from `.mp3` or `.wav`.
8. Open the generated tab page.
9. Favorite a tab.
10. Create a playlist with a cover image.
11. Add and remove a tab from the playlist.
12. Search users, tabs, and playlists.
13. Open Discovery and verify limited result lists.

Expected outcome:

- The UI displays clear, non-technical messages.
- CRUD changes appear without confusing stale states.
- Unauthorized actions return `401` or `403`.

### Regression Tests

Regression tests verify that recent changes do not break existing functionality.

Commands:

```powershell
cd backend
mvn test
```

```powershell
cd frontend
npm run build
```

Expected outcome:

- Backend tests pass with `0 failures`.
- Frontend TypeScript and Vite production build pass.

### Performance Tests

Performance tests provide measurable statistics for project reporting.

Current automated coverage:

- CPU: password validation benchmark.
- RAM/memory: image storage memory delta.
- Serialization: generated tab JSON serialization benchmark.

Expected outcome:

- Local operations remain fast and memory usage remains bounded.
- Generated metric files under `backend/target/test-metrics` can be attached or copied into project reports.

### Security Tests

Security tests check unsafe input and access-control-sensitive behavior.

Current automated coverage:

- Weak password rejection.
- Identity-based password rejection.
- Unsafe image upload rejection.
- Image path traversal rejection.
- Unsupported audio file rejection.
- Invalid token lookup rejection.
- Public user DTO does not expose email.

Manual security checklist:

- Protected endpoints without token should return `401`.
- Deleting another user's playlist or tab should return `403`.
- Duplicate active email registration should fail.
- Unsupported files should fail before processing starts.

Expected outcome:

- Invalid or unauthorized actions fail safely.
- Error responses should be understandable for frontend display.

### Usability Tests

Usability tests verify that the application remains understandable and visually stable.

Manual checklist:

- Empty profile playlist/tab panels show friendly empty states.
- Loading messages avoid technical wording.
- Search result colors distinguish users, tabs, and playlists.
- Profile and playlist cards scroll inside their panels instead of overflowing the page.
- Logo, avatar, and cover images remain visible and correctly cropped.

Expected outcome:

- Users can understand what is loading, what is empty, and what actions succeeded or failed.
