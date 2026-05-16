import { Navigate, Outlet, useLocation } from "react-router-dom";
import { isAuthenticated } from "../api/authSession";

export function RequireAuth() {
  const location = useLocation();

  if (!isAuthenticated()) {
    return <Navigate to="/auth" replace state={{ from: location }} />;
  }

  return <Outlet />;
}
