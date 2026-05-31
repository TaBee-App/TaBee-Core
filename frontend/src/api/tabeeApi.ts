import { apiFetch } from "./apiClient";
import type {
  AudioUploadAndProcessResponse,
  GeneratedNoteEvent,
  GeneratedTab,
  GenerateTabRequest,
  PlaylistRequest,
  PlaylistResponse,
  TabMetadataUpdate,
  TabResponse,
  TabSyncPoint
} from "../types/tab";

type AlphaTexToken = {
  value: string;
  beats: number;
};

export async function generateTab(request: GenerateTabRequest): Promise<GeneratedTab> {
  if (request.instrument !== "bass") {
    throw new Error("The connected backend pipeline currently supports bass recordings only.");
  }
  if (!isSupportedAudioFile(request.file)) {
    throw new Error("Unsupported file format. Please upload a .wav, .mp3, or .mp4 file.");
  }

  const formData = new FormData();
  formData.append("file", request.file);
  formData.append("tuning", request.tuning);

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
  const rendered = toAlphaTex(tab);
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
    alphaTex: rendered.alphaTex,
    syncMap: rendered.syncMap,
    audioUrl: jsonData.sourceAudioFile ? `/api/tabs/${publicAudio ? "public/" : ""}${tab.id}/audio` : undefined,
    tempo: rendered.tempo,
    createdAt: new Date(tab.createdAt).toLocaleString(),
    createdByCurrentUser: tab.createdByCurrentUser,
    favoritedByCurrentUser: tab.favoritedByCurrentUser,
    favoriteCount: tab.favoriteCount
  };
}

function toAlphaTex(tab: TabResponse) {
  const jsonData = tab.jsonData || {};
  const tuning = (tab.tuning || jsonData.tuning || "BEADG").toUpperCase();
  const tuningText = alphaTexTuningText(tuning);
  const notes = [...(jsonData.noteEvents || [])].sort((left, right) => Number(left.time) - Number(right.time));
  const tempo = resolveRenderTempo(notes, tab.estimatedTempo ?? jsonData.estimatedTempo ?? 90);
  const beatsPerBar = resolveBeatsPerBar(jsonData.beatsPerBar);
  const rendered = toTimedAlphaTexTokens(notes, tempo || 90);
  const body = toBarAlignedAlphaTex(rendered.tokens.length ? rendered.tokens : [{ value: ":4 r", beats: 1 }], beatsPerBar);

  const alphaTex = String.raw`\title "${escapeAlphaTexText(tab.title)}"
\artist "${escapeAlphaTexText(tab.artist || "TaBee")}"
\tempo ${tempo || 90}
\ts ${beatsPerBar} 4
\track "Bass"
\staff {tabs}
\tuning ${tuningText}
${body}`;

  return {
    alphaTex,
    syncMap: rendered.syncMap,
    tempo
  };
}

function toTimedAlphaTexTokens(notes: GeneratedNoteEvent[], tempo: number) {
  const secondsPerBeat = tempo > 0 ? 60 / tempo : 60 / 90;
  const tokens: AlphaTexToken[] = [];
  const syncMap: TabSyncPoint[] = [];
  let scoreBeatCursor = 0;

  notes.forEach((note, index) => {
    const durationSeconds = inferEventDuration(note, notes[index + 1], secondsPerBeat);
    const beatDuration = Math.max(0.125, durationSeconds / secondsPerBeat);
    const durations = splitDurationToAlphaTex(beatDuration);
    const noteValue = toAlphaTexNoteValue(note);
    const scoreStartBeat = scoreBeatCursor;

    durations.forEach((duration) => {
      const beats = alphaTexDurationToBeats(duration);
      tokens.push({ value: `:${duration} ${noteValue}`, beats });
      scoreBeatCursor += beats;
    });

    syncMap.push({
      audioStartMs: Math.max(0, Number(note.time) * 1000),
      audioEndMs: Math.max(0, (Number(note.time) + durationSeconds) * 1000),
      scoreStartMs: scoreStartBeat * secondsPerBeat * 1000,
      scoreEndMs: scoreBeatCursor * secondsPerBeat * 1000
    });
  });

  return { tokens, syncMap };
}

function resolveRenderTempo(notes: GeneratedNoteEvent[], storedTempo: number) {
  const fallbackTempo = storedTempo && Number.isFinite(storedTempo) && storedTempo > 0 ? storedTempo : 90;
  const detectedTempo = estimateTempoFromNoteEvents(notes);

  if (!detectedTempo) {
    return Math.round(fallbackTempo);
  }

  return Math.abs(detectedTempo - fallbackTempo) >= 8 ? detectedTempo : Math.round(fallbackTempo);
}

function estimateTempoFromNoteEvents(notes: GeneratedNoteEvent[]) {
  const onsetTimes = notes
    .filter((note) => !note.isRest && Number.isFinite(note.time))
    .map((note) => Number(note.time))
    .sort((left, right) => left - right);

  const candidateBpms: number[] = [];

  for (let index = 1; index < onsetTimes.length; index += 1) {
    const interval = onsetTimes[index] - onsetTimes[index - 1];

    if (interval < 0.12 || interval > 2.5) {
      continue;
    }

    let bpm = 60 / interval;
    while (bpm > 150) bpm /= 2;
    while (bpm < 70) bpm *= 2;

    if (bpm >= 70 && bpm <= 150) {
      candidateBpms.push(bpm);
    }
  }

  if (candidateBpms.length < 4) {
    return null;
  }

  candidateBpms.sort((left, right) => left - right);
  const middle = Math.floor(candidateBpms.length / 2);
  const median = candidateBpms.length % 2
    ? candidateBpms[middle]
    : (candidateBpms[middle - 1] + candidateBpms[middle]) / 2;

  return Math.round(median);
}

function toBarAlignedAlphaTex(tokens: AlphaTexToken[], measureBeats = 4) {
  const barsPerLine = 2;
  const lines: string[] = [];
  let currentLine: string[] = [];
  let currentMeasureBeats = 0;
  let currentLineBars = 0;

  tokens.forEach((token) => {
    const tokenBeats = Math.max(0.25, token.beats || 1);

    if (currentMeasureBeats > 0 && currentMeasureBeats + tokenBeats > measureBeats + 1e-6) {
      currentLine.push("|");
      currentMeasureBeats = 0;
      currentLineBars += 1;

      if (currentLineBars >= barsPerLine) {
        lines.push(currentLine.join(" "));
        currentLine = [];
        currentLineBars = 0;
      }
    }

    currentLine.push(token.value);
    currentMeasureBeats += tokenBeats;

    if (currentMeasureBeats >= measureBeats - 1e-6) {
      currentLine.push("|");
      currentMeasureBeats = 0;
      currentLineBars += 1;

      if (currentLineBars >= barsPerLine) {
        lines.push(currentLine.join(" "));
        currentLine = [];
        currentLineBars = 0;
      }
    }
  });

  if (currentLine.length) {
    if (currentLine[currentLine.length - 1] !== "|") {
      currentLine.push("|");
    }
    lines.push(currentLine.join(" "));
  }

  return lines.join("\n");
}

function alphaTexTuningText(tuning: string) {
  switch (tuning) {
    case "BEADG":
      return "(G2 D2 A1 E1 B0)";
    case "CGCF":
      return "(F2 C2 G1 C1)";
    case "EBABDBGB":
      return "(Gb2 Db2 Ab1 Eb1)";
    case "EADG":
    default:
      return "(G2 D2 A1 E1)";
  }
}

function resolveBeatsPerBar(value?: number | null) {
  if (!value || !Number.isFinite(value)) {
    return 4;
  }
  return Math.min(12, Math.max(2, Math.round(value)));
}

function inferEventDuration(note: GeneratedNoteEvent, nextNote: GeneratedNoteEvent | undefined, secondsPerBeat: number) {
  const nextGap = nextNote && Number.isFinite(nextNote.time) && Number.isFinite(note.time) && nextNote.time > note.time
    ? nextNote.time - note.time
    : null;

  if (note.duration && Number.isFinite(note.duration) && note.duration > 0) {
    return nextGap == null ? note.duration : Math.min(note.duration, nextGap);
  }

  if (nextGap != null) {
    return nextGap;
  }

  return secondsPerBeat;
}

function toAlphaTexNoteValue(note: GeneratedNoteEvent) {
  if (note.isRest || note.fret == null || note.stringNumber == null) {
    return "r";
  }
  return `${note.fret}.${note.stringNumber}`;
}

function splitDurationToAlphaTex(beatDuration: number) {
  const available = [
    { beats: 4, value: 1 },
    { beats: 2, value: 2 },
    { beats: 1, value: 4 },
    { beats: 0.5, value: 8 },
    { beats: 0.25, value: 16 }
  ];
  const roundedBeats = Math.max(0.25, Math.round(beatDuration * 4) / 4);
  const result: number[] = [];
  let remaining = roundedBeats;

  for (const option of available) {
    while (remaining + 1e-6 >= option.beats) {
      result.push(option.value);
      remaining -= option.beats;
    }
  }

  return result.length ? result : [16];
}

function alphaTexDurationToBeats(duration: number) {
  switch (duration) {
    case 1:
      return 4;
    case 2:
      return 2;
    case 4:
      return 1;
    case 8:
      return 0.5;
    case 16:
      return 0.25;
    default:
      return 1;
  }
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
