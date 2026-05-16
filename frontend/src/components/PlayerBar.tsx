import { Pause, Play, Repeat, ScrollText } from "lucide-react";
import { useEffect, useState } from "react";
import { apiFetchBlob } from "../api/apiClient";

interface PlayerBarProps {
  title: string;
  meta: string;
  ready: boolean;
  playing: boolean;
  looping: boolean;
  autoScroll: boolean;
  speed: number;
  audioUrl?: string;
  onTogglePlay: () => void;
  onLoopChange: (value: boolean) => void;
  onAutoScrollChange: (value: boolean) => void;
  onSpeedChange: (value: number) => void;
}

export function PlayerBar({
  title,
  meta,
  ready,
  playing,
  looping,
  autoScroll,
  speed,
  audioUrl,
  onTogglePlay,
  onLoopChange,
  onAutoScrollChange,
  onSpeedChange
}: PlayerBarProps) {
  const [playableAudioUrl, setPlayableAudioUrl] = useState("");

  useEffect(() => {
    let active = true;
    let objectUrl = "";

    async function loadAudio() {
      if (!audioUrl) {
        setPlayableAudioUrl("");
        return;
      }

      try {
        const blob = await apiFetchBlob(audioUrl);
        if (!active) return;
        objectUrl = URL.createObjectURL(blob);
        setPlayableAudioUrl(objectUrl);
      } catch {
        if (active) {
          setPlayableAudioUrl("");
        }
      }
    }

    loadAudio();

    return () => {
      active = false;
      if (objectUrl) {
        URL.revokeObjectURL(objectUrl);
      }
    };
  }, [audioUrl]);

  return (
    <div className="player-bar">
      <div className="player-inner">
        <button className="icon-btn primary" disabled={!ready} onClick={onTogglePlay} title={playing ? "Pause" : "Play"}>
          {playing ? <Pause size={21} /> : <Play size={21} />}
        </button>

        <div className="now-playing">
          <div className="now-title">{title}</div>
          <div className="now-meta">{meta}</div>
        </div>

        <div className="player-controls">
          <button
            className={`icon-btn${looping ? " active" : ""}`}
            disabled={!ready}
            onClick={() => onLoopChange(!looping)}
            title="Loop"
          >
            <Repeat size={19} />
          </button>
          <button
            className={`icon-btn${autoScroll ? " active" : ""}`}
            disabled={!ready}
            onClick={() => onAutoScrollChange(!autoScroll)}
            title="Auto scroll"
          >
            <ScrollText size={19} />
          </button>
          <label className="mini-control">
            Speed
            <input
              type="number"
              min={25}
              max={200}
              step={5}
              value={speed}
              onChange={(event) => onSpeedChange(Number(event.target.value))}
            />
          </label>
          {playableAudioUrl ? <audio src={playableAudioUrl} controls /> : null}
        </div>
      </div>
    </div>
  );
}
