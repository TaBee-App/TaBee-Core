import { clearAuthSession, getAuthToken } from "./authSession";
import type { ApiErrorResponse } from "../types/auth";

export class ApiError extends Error {
  status: number;
  details: ApiErrorResponse | null;

  constructor(message: string, status: number, details: ApiErrorResponse | null = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.details = details;
  }
}

type ApiFetchOptions = RequestInit & {
  authenticated?: boolean;
};

export async function apiFetch<T>(path: string, options: ApiFetchOptions = {}): Promise<T> {
  const { authenticated = true, headers, ...fetchOptions } = options;
  const requestHeaders = new Headers(headers);

  if (fetchOptions.body && !(fetchOptions.body instanceof FormData) && !requestHeaders.has("Content-Type")) {
    requestHeaders.set("Content-Type", "application/json");
  }

  const token = getAuthToken();
  if (authenticated && token && !requestHeaders.has("Authorization")) {
    requestHeaders.set("Authorization", `Bearer ${token}`);
  }

  const response = await fetch(path, {
    ...fetchOptions,
    headers: requestHeaders
  });

  const data = await readResponseBody(response);

  if (!response.ok) {
    if (response.status === 401) {
      clearAuthSession();
    }

    const details = isApiErrorResponse(data) ? data : null;
    throw new ApiError(details?.message || response.statusText || "Request failed.", response.status, details);
  }

  return data as T;
}

export async function apiFetchBlob(path: string, options: ApiFetchOptions = {}): Promise<Blob> {
  const { authenticated = true, headers, ...fetchOptions } = options;
  const requestHeaders = new Headers(headers);

  const token = getAuthToken();
  if (authenticated && token && !requestHeaders.has("Authorization")) {
    requestHeaders.set("Authorization", `Bearer ${token}`);
  }

  const response = await fetch(path, {
    ...fetchOptions,
    headers: requestHeaders
  });

  if (!response.ok) {
    if (response.status === 401) {
      clearAuthSession();
    }

    const data = await readResponseBody(response);
    const details = isApiErrorResponse(data) ? data : null;
    throw new ApiError(details?.message || response.statusText || "Request failed.", response.status, details);
  }

  return response.blob();
}

async function readResponseBody(response: Response): Promise<unknown> {
  if (response.status === 204) {
    return null;
  }

  const contentType = response.headers.get("Content-Type") || "";
  if (contentType.includes("application/json")) {
    return response.json();
  }

  return response.text();
}

function isApiErrorResponse(value: unknown): value is ApiErrorResponse {
  return Boolean(value && typeof value === "object" && "message" in value);
}
