export type Instrument = "bass" | "guitar";

export interface GeneratedTab {
  id: string;
  ownerUserId?: number;
  ownerUsername?: string;
  ownerProfileImageUrl?: string | null;
  title: string;
  fileName: string;
  instrument: Instrument;
  artist?: string | null;
  tuning?: string | null;
  alphaTex: string;
  syncMap?: TabSyncPoint[];
  audioUrl?: string;
  tempo?: number | null;
  createdAt: string;
  createdByCurrentUser?: boolean;
  favoritedByCurrentUser?: boolean;
  favoriteCount?: number;
}

export interface TabSyncPoint {
  audioStartMs: number;
  audioEndMs: number;
  scoreStartMs: number;
  scoreEndMs: number;
}

export interface GenerateTabRequest {
  file: File;
  title: string;
  instrument: Instrument;
  tuning: "EADG" | "BEADG";
}

export interface GenerateTabResponse {
  success: boolean;
  tex?: string;
  alphaTex?: string;
  audio_url?: string;
  audioUrl?: string;
  tempo?: number | null;
  message?: string;
}

export interface GeneratedNoteEvent {
  isRest?: boolean;
  time: number;
  duration?: number | null;
  frequency?: number | null;
  confidence?: number | null;
  noteName?: string | null;
  midiNumber?: number | null;
  fret?: number | null;
  stringNumber?: number | null;
}

export interface GeneratedTabJson {
  sourceAudio?: string;
  sourceAudioFile?: string;
  instrument?: Instrument;
  tuning?: string;
  estimatedTempo?: number | null;
  noteEvents?: GeneratedNoteEvent[];
  summary?: {
    detectedOnsets?: number;
    detectedNotes?: number;
    playableNotes?: number;
  };
}

export interface TabResponse {
  id: number;
  ownerUserId: number;
  ownerUsername: string;
  ownerProfileImageUrl?: string | null;
  tabDataId: number;
  title: string;
  artist?: string | null;
  tuning?: string | null;
  estimatedTempo?: number | null;
  createdAt: string;
  updatedAt: string;
  createdByCurrentUser: boolean;
  favoritedByCurrentUser: boolean;
  favoriteCount: number;
  jsonData: GeneratedTabJson;
}

export interface AudioUploadAndProcessResponse {
  message: string;
  tabId: number;
  tab: TabResponse;
}

export interface TabMetadataUpdate {
  title?: string;
  artist?: string | null;
  tuning?: string | null;
  estimatedTempo?: number | null;
}

export interface PlaylistTabResponse {
  tabId: number;
  ownerUserId: number;
  title: string;
  artist?: string | null;
  addedAt: string;
}

export interface PlaylistResponse {
  id: number;
  ownerUserId: number;
  ownerUsername: string;
  ownerProfileImageUrl?: string | null;
  name: string;
  description?: string | null;
  coverImageUrl?: string | null;
  createdAt: string;
  createdByCurrentUser: boolean;
  savedByCurrentUser: boolean;
  savedCount: number;
  tabs: PlaylistTabResponse[];
}

export interface PlaylistRequest {
  name: string;
  description?: string | null;
}
