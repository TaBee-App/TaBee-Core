interface AvatarProps {
  src?: string | null;
  label?: string | null;
  size?: "sm" | "md" | "lg";
}

export function Avatar({ src, label, size = "sm" }: AvatarProps) {
  return (
    <span className={`avatar avatar-${size}`} title={label || "Profile"}>
      <img src={src || "/default-user-avatar.png"} alt={src ? label || "Profile" : "Default profile"} />
    </span>
  );
}
