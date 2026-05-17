import * as alphaTab from "@coderline/alphatab";
import { useCallback, useEffect, useRef, useState } from "react";
import { errorMessage } from "../lib/errors";
import type { GeneratedTab } from "../types/tab";

type PlaybackBeat = {
  absolutePlaybackStart: number;
  playbackDuration: number;
  voice?: {
    bar?: {
      voices?: Array<{
        beats?: PlaybackBeat[];
      }>;
    };
  };
};

type LoopRange = {
  startTick: number;
  endTick: number;
  startBeat: PlaybackBeat;
  endBeat: PlaybackBeat;
};

interface TabRendererProps {
  tab: GeneratedTab | null;
  playing: boolean;
  looping: boolean;
  autoScroll: boolean;
  speed: number;
  onReadyChange: (ready: boolean) => void;
  onPlayingChange: (playing: boolean) => void;
}

export function TabRenderer({
  tab,
  playing,
  looping,
  autoScroll,
  speed,
  onReadyChange,
  onPlayingChange
}: TabRendererProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const apiRef = useRef<alphaTab.AlphaTabApi | null>(null);
  const loopingRef = useRef(looping);
  const playingRef = useRef(playing);
  const previousLoopingRef = useRef(looping);
  const hasHighlightedRangeRef = useRef(false);
  const currentBeatRef = useRef<PlaybackBeat | null>(null);
  const pendingLoopRangeRef = useRef<LoopRange | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    loopingRef.current = looping;
  }, [looping]);

  useEffect(() => {
    playingRef.current = playing;
  }, [playing]);

  const destroy = useCallback(() => {
    apiRef.current?.destroy();
    apiRef.current = null;
    currentBeatRef.current = null;
    hasHighlightedRangeRef.current = false;
    pendingLoopRangeRef.current = null;
    onReadyChange(false);
  }, [onReadyChange]);

  const getFirstBeat = useCallback((api: alphaTab.AlphaTabApi): PlaybackBeat | null => {
    const score = api.score as unknown as {
      tracks?: Array<{
        staves?: Array<{
          bars?: Array<{
            voices?: Array<{
              beats?: PlaybackBeat[];
            }>;
          }>;
        }>;
      }>;
    } | null;

    return score?.tracks?.[0]?.staves?.[0]?.bars?.[0]?.voices?.[0]?.beats?.[0] ?? null;
  }, []);

  const getBarLoopRange = useCallback((beat: PlaybackBeat | null): LoopRange | null => {
    const bar = beat?.voice?.bar;
    const beats = bar?.voices?.flatMap((voice) => voice.beats ?? []) ?? [];

    if (!beats.length) {
      return null;
    }

    const startTick = Math.min(...beats.map((item) => item.absolutePlaybackStart));
    const endTick = Math.max(...beats.map((item) => item.absolutePlaybackStart + item.playbackDuration));

    if (!Number.isFinite(startTick) || !Number.isFinite(endTick) || endTick <= startTick) {
      return null;
    }

    return {
      startTick,
      endTick,
      startBeat: beats[0],
      endBeat: beats[beats.length - 1]
    };
  }, []);

  const activateLoopRange = useCallback((api: alphaTab.AlphaTabApi, range: LoopRange) => {
    api.playbackRange = { startTick: range.startTick, endTick: range.endTick };
    api.highlightPlaybackRange(range.startBeat as never, range.endBeat as never);
  }, []);

  const applyPlaybackRangeForBeat = useCallback((api: alphaTab.AlphaTabApi, beat: PlaybackBeat | null) => {
    const range = getBarLoopRange(beat);

    if (!range) {
      api.playbackRange = null;
      pendingLoopRangeRef.current = null;
      return;
    }

    activateLoopRange(api, range);
    pendingLoopRangeRef.current = null;
  }, [activateLoopRange, getBarLoopRange]);

  const queueLoopRangeForBeat = useCallback((api: alphaTab.AlphaTabApi, beat: PlaybackBeat | null) => {
    const range = getBarLoopRange(beat);

    if (!range) {
      api.playbackRange = null;
      pendingLoopRangeRef.current = null;
      return;
    }

    pendingLoopRangeRef.current = range;
    api.playbackRange = null;
    api.highlightPlaybackRange(range.startBeat as never, range.endBeat as never);
  }, [getBarLoopRange]);

  const applyCurrentBarPlaybackRange = useCallback((api: alphaTab.AlphaTabApi) => {
    const beat = currentBeatRef.current ?? getFirstBeat(api);
    applyPlaybackRangeForBeat(api, beat);
  }, [applyPlaybackRangeForBeat, getFirstBeat]);

  const applyLoopPlaybackRange = useCallback((api: alphaTab.AlphaTabApi) => {
    if (hasHighlightedRangeRef.current) {
      api.applyPlaybackRangeFromHighlight();
    }

    if (!api.playbackRange) {
      applyCurrentBarPlaybackRange(api);
    }
  }, [applyCurrentBarPlaybackRange]);

  const activatePendingLoopRange = useCallback((api: alphaTab.AlphaTabApi, range: LoopRange) => {
    activateLoopRange(api, range);
    pendingLoopRangeRef.current = null;
  }, [activateLoopRange]);

  useEffect(() => {
    if (!tab || !hostRef.current) {
      destroy();
      return;
    }

    destroy();
    setError("");
    hostRef.current.innerHTML = "";

    try {
      const api = new alphaTab.AlphaTabApi(hostRef.current, {
        core: {
          engine: "html5"
        },
        notation: {
          rhythmMode: alphaTab.TabRhythmMode.ShowWithBars
        },
        display: {
          staveProfile: tab.instrument === "bass" ? alphaTab.StaveProfile.Tab : alphaTab.StaveProfile.ScoreTab,
          resources: {
            tablatureFont: "bold 14px Arial",
            staffLineColor: "#687478",
            barSeparatorColor: "#687478",
            mainGlyphColor: "#1b2023",
            secondaryGlyphColor: "#4b5559",
            scoreInfoColor: "#4b5559",
            barNumberColor: "#687478"
          }
        },
        player: {
          enablePlayer: true,
          enableCursor: true,
          enableAnimatedBeatCursor: true,
          enableUserInteraction: true,
          scrollMode: alphaTab.ScrollMode.Off,
          scrollOffsetY: -80,
          soundFont: "https://cdn.jsdelivr.net/npm/@coderline/alphatab@latest/dist/soundfont/sonivox.sf2"
        }
      });

      apiRef.current = api;
      api.scoreLoaded.on(() => {
        currentBeatRef.current = getFirstBeat(api);
        onReadyChange(true);
      });
      api.playedBeatChanged.on((beat) => {
        const currentBeat = beat as unknown as PlaybackBeat;
        currentBeatRef.current = currentBeat;

        const pendingRange = pendingLoopRangeRef.current;
        if (
          loopingRef.current &&
          pendingRange &&
          currentBeat.absolutePlaybackStart >= pendingRange.endTick
        ) {
          activatePendingLoopRange(api, pendingRange);
          api.stop();
          api.play();
        }
      });
      api.playerPositionChanged.on((position) => {
        const pendingRange = pendingLoopRangeRef.current;
        if (!loopingRef.current || !pendingRange) {
          return;
        }

        const activationLeadTicks = 120;
        if (
          position.currentTick >= pendingRange.endTick - activationLeadTicks &&
          position.currentTick < pendingRange.endTick
        ) {
          activatePendingLoopRange(api, pendingRange);
        }
      });
      api.beatMouseUp.on((beat) => {
        if (!loopingRef.current || !beat) {
          return;
        }

        const clickedBeat = beat as unknown as PlaybackBeat;
        currentBeatRef.current = clickedBeat;

        window.setTimeout(() => {
          const activeApi = apiRef.current;
          if (activeApi && loopingRef.current) {
            queueLoopRangeForBeat(activeApi, clickedBeat);
          }
        }, 0);
      });
      api.playbackRangeHighlightChanged.on((args) => {
        hasHighlightedRangeRef.current = Boolean(args.startBeat && args.endBeat);
      });
      api.playerFinished.on(() => {
        if (!loopingRef.current) {
          onPlayingChange(false);
        }
      });
      api.tex(tab.alphaTex);
    } catch (caught) {
      setError(errorMessage(caught, "Unable to render this tab."));
      onReadyChange(false);
    }

    return destroy;
  }, [activatePendingLoopRange, applyLoopPlaybackRange, queueLoopRangeForBeat, destroy, getFirstBeat, onPlayingChange, onReadyChange, tab]);

  useEffect(() => {
    const api = apiRef.current;
    if (!api) return;
    const normalizedSpeed = Math.min(200, Math.max(25, Number(speed) || 100));
    api.playbackSpeed = normalizedSpeed / 100;
    api.settings.player.scrollMode = autoScroll && playing ? alphaTab.ScrollMode.Continuous : alphaTab.ScrollMode.Off;
    api.updateSettings();
  }, [autoScroll, playing, speed]);

  useEffect(() => {
    const api = apiRef.current;
    if (!api || previousLoopingRef.current === looping) return;

    previousLoopingRef.current = looping;
    api.isLooping = looping;

    if (looping) {
      if (playingRef.current) {
        queueLoopRangeForBeat(api, currentBeatRef.current ?? getFirstBeat(api));
      } else {
        applyLoopPlaybackRange(api);
      }
    } else {
      api.playbackRange = null;
      pendingLoopRangeRef.current = null;
      api.clearPlaybackRangeHighlight();
    }
  }, [applyLoopPlaybackRange, getFirstBeat, looping, queueLoopRangeForBeat]);

  useEffect(() => {
    const api = apiRef.current;
    if (!api) return;
    if (playing) {
      api.play();
    } else {
      api.pause();
    }
  }, [playing]);

  return (
    <div className="score-body">
      {!tab ? (
        <div className="score-empty">
          <div className="score-empty-inner">
            <h2>Upload a recording to begin</h2>
            <p>Your generated notation will appear here as a playable tab.</p>
          </div>
        </div>
      ) : null}
      {error ? <div className="render-error">{error}</div> : null}
      <div ref={hostRef} className="alpha-tab-host" style={{ display: tab ? "block" : "none" }} />
    </div>
  );
}
