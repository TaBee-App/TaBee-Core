import { ApiError } from "../api/apiClient";

export function errorMessage(caught: unknown, fallback: string) {
  if (caught instanceof ApiError) {
    const fieldMessages = caught.details?.fieldErrors
      ?.map((fieldError) => `${label(fieldError.field)}: ${fieldError.message}`)
      .join(" ");
    return fieldMessages || caught.message || fallback;
  }

  return caught instanceof Error ? caught.message : fallback;
}

function label(field: string) {
  return field
    .replace(/([A-Z])/g, " $1")
    .replace(/^./, (first) => first.toUpperCase());
}
