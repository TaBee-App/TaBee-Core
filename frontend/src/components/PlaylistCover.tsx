import { ListMusic } from "lucide-react";

interface PlaylistCoverProps {
  src?: string | null;
  title?: string | null;
  size?: "sm" | "md" | "lg";
}

export function PlaylistCover({ src, title, size = "md" }: PlaylistCoverProps) {
  return (
    <div className={`playlist-cover playlist-cover-${size}`}>
      {src ? <img src={src} alt={title || "Playlist cover"} /> : <ListMusic size={size === "lg" ? 34 : 22} />}
    </div>
  );
}
