import { apiFetch } from "./apiClient";
import type {
  AudioUploadAndProcessResponse,
  GeneratedNoteEvent,
  GeneratedTab,
  GenerateTabRequest,
  PlaylistRequest,
  PlaylistResponse,
  TabMetadataUpdate,
  TabResponse
} from "../types/tab";

export async function generateTab(request: GenerateTabRequest): Promise<GeneratedTab> {
  if (request.instrument !== "bass") {
    throw new Error("The connected backend pipeline currently supports bass recordings only.");
  }
  if (!isSupportedAudioFile(request.file)) {
    throw new Error("Unsupported file format. Please upload a .wav, .mp3, or .mp4 file.");
  }

  const formData = new FormData();
  formData.append("file", request.file);

  const response = await apiFetch<AudioUploadAndProcessResponse>("/api/audio-files/upload-and-process", {
    method: "POST",
    body: formData
  });

  let tab = response.tab;
  const title = request.title.trim();

  if (title && title !== tab.title) {
    tab = await apiFetch<TabResponse>(`/api/tabs/${tab.id}`, {
      method: "PUT",
      body: JSON.stringify({ title })
    });
  }

  return toGeneratedTab(tab, request.file.name);
}

function isSupportedAudioFile(file: File) {
  const filename = file.name.toLowerCase();
  return filename.endsWith(".wav") || filename.endsWith(".mp3") || filename.endsWith(".mp4");
}

export async function listTabs(): Promise<GeneratedTab[]> {
  const tabs = await apiFetch<TabResponse[]>("/api/tabs");
  return tabs.map((tab) => toGeneratedTab(tab));
}

export async function listFavoriteTabs(): Promise<GeneratedTab[]> {
  const tabs = await apiFetch<TabResponse[]>("/api/tabs/favorites");
  return tabs.map((tab) => toGeneratedTab(tab, "uploaded-audio", true));
}

export async function listPublicTabs(): Promise<GeneratedTab[]> {
  const tabs = await apiFetch<TabResponse[]>("/api/tabs/public");
  return tabs.map((tab) => toGeneratedTab(tab, "uploaded-audio", true));
}

export async function listDiscoveryTabs(): Promise<GeneratedTab[]> {
  const tabs = await apiFetch<TabResponse[]>("/api/tabs/discovery");
  return tabs.map((tab) => toGeneratedTab(tab, "uploaded-audio", true));
}

export async function listPublicTabsByUser(userId: string): Promise<GeneratedTab[]> {
  const tabs = await listPublicTabs();
  return tabs.filter((tab) => String(tab.ownerUserId) === userId);
}

export async function getGeneratedTab(tabId: string): Promise<GeneratedTab> {
  const tab = await apiFetch<TabResponse>(`/api/tabs/${tabId}`);
  return toGeneratedTab(tab);
}

export async function getPublicGeneratedTab(tabId: string): Promise<GeneratedTab> {
  const tab = await apiFetch<TabResponse>(`/api/tabs/public/${tabId}`);
  return toGeneratedTab(tab, "uploaded-audio", true);
}

export async function deleteGeneratedTab(tabId: string): Promise<void> {
  await apiFetch<null>(`/api/tabs/${tabId}`, {
    method: "DELETE"
  });
}

export async function updateGeneratedTab(tabId: string, metadata: TabMetadataUpdate): Promise<GeneratedTab> {
  const tab = await apiFetch<TabResponse>(`/api/tabs/${tabId}`, {
    method: "PUT",
    body: JSON.stringify(metadata)
  });
  return toGeneratedTab(tab);
}

export async function favoriteTab(tabId: string): Promise<GeneratedTab> {
  const tab = await apiFetch<TabResponse>(`/api/tabs/${tabId}/favorite`, {
    method: "POST"
  });
  return toGeneratedTab(tab, "uploaded-audio", true);
}

export async function unfavoriteTab(tabId: string): Promise<GeneratedTab> {
  const tab = await apiFetch<TabResponse>(`/api/tabs/${tabId}/favorite`, {
    method: "DELETE"
  });
  return toGeneratedTab(tab, "uploaded-audio", true);
}

export async function listPlaylists(): Promise<PlaylistResponse[]> {
  return apiFetch<PlaylistResponse[]>("/api/playlists");
}

export async function listPlaylistArchive(): Promise<PlaylistResponse[]> {
  return apiFetch<PlaylistResponse[]>("/api/playlists/archive");
}

export async function listDiscoveryPlaylists(): Promise<PlaylistResponse[]> {
  return apiFetch<PlaylistResponse[]>("/api/playlists/discovery");
}

export async function listPlaylistArchiveByUser(userId: string): Promise<PlaylistResponse[]> {
  const playlists = await listPlaylistArchive();
  return playlists.filter((playlist) => String(playlist.ownerUserId) === userId);
}

export async function listSavedPlaylists(): Promise<PlaylistResponse[]> {
  return apiFetch<PlaylistResponse[]>("/api/playlists/saved");
}

export async function getPlaylist(playlistId: string): Promise<PlaylistResponse> {
  return apiFetch<PlaylistResponse>(`/api/playlists/${playlistId}`);
}

export async function createPlaylist(request: PlaylistRequest): Promise<PlaylistResponse> {
  return apiFetch<PlaylistResponse>("/api/playlists", {
    method: "POST",
    body: JSON.stringify(request)
  });
}

export async function updatePlaylistCover(playlistId: number, file: File): Promise<PlaylistResponse> {
  const formData = new FormData();
  formData.append("file", file);
  return apiFetch<PlaylistResponse>(`/api/playlists/${playlistId}/cover-image`, {
    method: "POST",
    body: formData
  });
}

export async function removePlaylistCover(playlistId: number): Promise<PlaylistResponse> {
  return apiFetch<PlaylistResponse>(`/api/playlists/${playlistId}/cover-image`, {
    method: "DELETE"
  });
}

export async function deletePlaylist(playlistId: number): Promise<void> {
  await apiFetch<null>(`/api/playlists/${playlistId}`, {
    method: "DELETE"
  });
}

export async function addTabToPlaylist(playlistId: number, tabId: string): Promise<PlaylistResponse> {
  return apiFetch<PlaylistResponse>(`/api/playlists/${playlistId}/tabs`, {
    method: "POST",
    body: JSON.stringify({ tabId: Number(tabId) })
  });
}

export async function removeTabFromPlaylist(playlistId: number, tabId: number): Promise<PlaylistResponse> {
  return apiFetch<PlaylistResponse>(`/api/playlists/${playlistId}/tabs/${tabId}`, {
    method: "DELETE"
  });
}

export async function savePlaylist(playlistId: number): Promise<PlaylistResponse> {
  return apiFetch<PlaylistResponse>(`/api/playlists/${playlistId}/save`, {
    method: "POST"
  });
}

export async function unsavePlaylist(playlistId: number): Promise<PlaylistResponse> {
  return apiFetch<PlaylistResponse>(`/api/playlists/${playlistId}/save`, {
    method: "DELETE"
  });
}

function toGeneratedTab(tab: TabResponse, uploadedFileName = "uploaded-audio", publicAudio = false): GeneratedTab {
  const jsonData = tab.jsonData || {};
  return {
    id: String(tab.id),
    ownerUserId: tab.ownerUserId,
    ownerUsername: tab.ownerUsername,
    ownerProfileImageUrl: tab.ownerProfileImageUrl,
    title: tab.title,
    fileName: sourceFileName(jsonData.sourceAudio) || uploadedFileName,
    instrument: jsonData.instrument || "bass",
    artist: tab.artist,
    tuning: tab.tuning ?? jsonData.tuning ?? null,
    alphaTex: toAlphaTex(tab),
    audioUrl: jsonData.sourceAudioFile ? `/api/tabs/${publicAudio ? "public/" : ""}${tab.id}/audio` : undefined,
    tempo: tab.estimatedTempo ?? jsonData.estimatedTempo ?? null,
    createdAt: new Date(tab.createdAt).toLocaleString(),
    createdByCurrentUser: tab.createdByCurrentUser,
    favoritedByCurrentUser: tab.favoritedByCurrentUser,
    favoriteCount: tab.favoriteCount
  };
}

function toAlphaTex(tab: TabResponse) {
  const jsonData = tab.jsonData || {};
  const tuning = (tab.tuning || jsonData.tuning || "EADG").toUpperCase();
  const tuningText = tuning === "BEADG" ? "(G2 D2 A1 E1 B0)" : "(G2 D2 A1 E1)";
  const notes = [...(jsonData.noteEvents || [])].sort((left, right) => Number(left.time) - Number(right.time));
  const playableNotes = notes.length
    ? toAlphaTexEvents(notes, tab.estimatedTempo ?? jsonData.estimatedTempo ?? null)
    : [{ token: ":8 r", beats: 0.5 }];
  const body = chunkAlphaTexEvents(playableNotes, 4)
    .map((line) => `${line.join(" ")} |`)
    .join("\n");

  return String.raw`\title "${escapeAlphaTexText(tab.title)}"
\artist "${escapeAlphaTexText(tab.artist || "TaBee")}"
\track "Bass"
\staff {tabs}
\tuning ${tuningText}
${body}`;
}

function toAlphaTexNote(note: GeneratedNoteEvent) {
  if (note.fret == null || note.stringNumber == null) {
    return "r";
  }

  return `${note.fret}.${note.stringNumber}`;
}

type AlphaTexEvent = {
  token: string;
  beats: number;
};

function toAlphaTexEvents(notes: GeneratedNoteEvent[], tempo: number | null): AlphaTexEvent[] {
  return notes.flatMap((note) => {
    const totalBeats = quantizeDurationBeats(noteDurationBeats(note, tempo));
    const noteBeats = notePlaybackBeats(totalBeats);
    const restBeats = roundBeatRemainder(totalBeats - noteBeats);
    const events: AlphaTexEvent[] = [
      {
        token: `${durationPrefix(noteBeats)} ${toAlphaTexNote(note)}`,
        beats: noteBeats
      }
    ];

    if (restBeats >= 0.25) {
      events.push({
        token: `${durationPrefix(restBeats)} r`,
        beats: restBeats
      });
    }

    return events;
  });
}

function noteDurationBeats(note: GeneratedNoteEvent, tempo: number | null) {
  if (!note.duration || !tempo || tempo <= 0) {
    return 0.5;
  }

  return note.duration * (tempo / 60);
}

function quantizeDurationBeats(beats: number) {
  if (beats >= 1.5) return 2;
  if (beats >= 0.75) return 1;
  if (beats >= 0.375) return 0.5;
  return 0.25;
}

function durationPrefix(beats: number) {
  if (beats >= 2) return ":2";
  if (beats >= 1) return ":4";
  if (beats >= 0.5) return ":8";
  return ":16";
}

function notePlaybackBeats(totalBeats: number) {
  if (totalBeats >= 2) return 1;
  if (totalBeats >= 1) return 0.5;
  return totalBeats;
}

function roundBeatRemainder(beats: number) {
  if (beats >= 0.75) return 1;
  if (beats >= 0.375) return 0.5;
  if (beats >= 0.1875) return 0.25;
  return 0;
}

function chunkAlphaTexEvents(items: AlphaTexEvent[], beatsPerBar: number) {
  const chunks: string[][] = [];
  let current: string[] = [];
  let currentBeats = 0;

  for (const item of items) {
    if (current.length && currentBeats + item.beats > beatsPerBar) {
      chunks.push(current);
      current = [];
      currentBeats = 0;
    }

    current.push(item.token);
    currentBeats += item.beats;
  }

  if (current.length) {
    chunks.push(current);
  }
  return chunks;
}

function sourceFileName(sourceAudio?: string) {
  if (!sourceAudio) {
    return "";
  }
  return sourceAudio.split(/[\\/]/).pop() || sourceAudio;
}

function escapeAlphaTexText(value: string) {
  return value.replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}
