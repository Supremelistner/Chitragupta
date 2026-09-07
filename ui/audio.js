/* ===================================================================
   Chitragupta UI — TTS helper

   Plays text through the model service's /infer/synthesize-audio
   endpoint and exposes a tiny controller with play/stop. The audio
   is rendered as a single <audio> element that's reused across
   invocations so we don't leak DOM nodes.

   Usage:
       Audio.speak({text, lang, voice, onStart, onEnd, onError})

   ``lang`` is a BCP-47 code; the model service supports hi-IN,
   en-US, ta-IN, bn-IN, etc. ``voice`` is optional (defaults to the
   service's default Kore).
   =================================================================== */
(function () {
    'use strict';

    // The orchestrator exposes POST /api/tts on the same origin. It
    // proxies to the model service. Keeping the audio call on the
    // same origin avoids CORS and keeps the model service's URL an
    // internal detail.
    const ENDPOINT = '/api/tts';

    let audioEl = null;
    let objectUrl = null;
    let currentToken = 0;
    let playing = false;

    function getAudioEl() {
        if (!audioEl) {
            audioEl = new Audio();
            audioEl.preload = 'auto';
            audioEl.addEventListener('ended', () => {
                playing = false;
                if (currentCb && currentCb.onEnd) currentCb.onEnd();
                cleanup();
            });
            audioEl.addEventListener('error', (e) => {
                playing = false;
                if (currentCb && currentCb.onError) currentCb.onError(e);
                cleanup();
            });
        }
        return audioEl;
    }

    let currentCb = null;

    function cleanup() {
        if (objectUrl) {
            URL.revokeObjectURL(objectUrl);
            objectUrl = null;
        }
        currentCb = null;
    }

    async function speak({ text, lang, voice, onStart, onEnd, onError }) {
        if (!text || !text.trim()) return;
        stop();

        const token = ++currentToken;
        currentCb = { onStart, onEnd, onError };

        try {
            const url = ENDPOINT;
            const resp = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: text.slice(0, 4000),  // safety cap
                    language: lang || 'en-US',
                    ...(voice ? { voice } : {}),
                }),
            });
            if (token !== currentToken) return;  // superseded by another call
            if (!resp.ok) {
                throw new Error(`TTS HTTP ${resp.status}`);
            }
            const data = await resp.json();
            if (token !== currentToken) return;

            // The orchestrator's /api/tts returns the audio in
            // ``audio_base64``; the model service's /infer/synthesize-audio
            // returns it in ``output``. Check both.
            const b64 = data.audio_base64 || data.output;
            if (!b64) {
                throw new Error('TTS response missing audio payload');
            }
            const bytes = base64ToBytes(b64);
            const blob = new Blob([bytes], { type: 'audio/wav' });
            objectUrl = URL.createObjectURL(blob);

            const el = getAudioEl();
            el.src = objectUrl;
            await el.play();
            if (token !== currentToken) return;
            playing = true;
            if (onStart) onStart();
        } catch (err) {
            console.warn('TTS failed:', err);
            if (token === currentToken) {
                playing = false;
                if (onError) onError(err);
            }
            cleanup();
        }
    }

    function stop() {
        if (audioEl) {
            audioEl.pause();
            audioEl.currentTime = 0;
        }
        playing = false;
        // bump the token so any in-flight speak() aborts
        currentToken += 1;
        cleanup();
    }

    function isPlaying() {
        return playing;
    }

    function base64ToBytes(b64) {
        const bin = atob(b64);
        const out = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
        return out.buffer;
    }

    window.Audio = { speak, stop, isPlaying };
})();
