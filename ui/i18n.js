/* ===================================================================
   Chitragupta UI — i18n loader

   Loads string tables from /static/i18n/<lang>.json and exposes
   ``t(key, params)`` for templates. Falls back to English when a
   key is missing in the active language, and falls back to the key
   itself when missing in English too — so missing keys are obvious
   in the rendered UI rather than silently swallowed.
   =================================================================== */
(function () {
    'use strict';

    const SUPPORTED = ['en', 'hi', 'ta', 'bn'];
    const DEFAULT_LANG = 'en';
    const STORAGE_KEY = 'chitragupta.lang';

    const tables = {};
    let activeLang = DEFAULT_LANG;

    function detectInitialLang() {
        // Priority: explicit user choice > active session preference >
        // browser language > default.
        const stored = localStorage.getItem(STORAGE_KEY);
        if (stored && SUPPORTED.includes(stored)) {
            return stored;
        }
        const browser = (navigator.language || '').toLowerCase().split('-')[0];
        if (SUPPORTED.includes(browser)) {
            return browser;
        }
        return DEFAULT_LANG;
    }

    async function loadLang(lang) {
        if (tables[lang]) return tables[lang];
        try {
            // The orchestrator serves the JSON via /api/i18n/<lang> so
            // we don't depend on StaticFiles subdirectory serving.
            const resp = await fetch(`/api/i18n/${lang}`);
            if (!resp.ok) {
                throw new Error(`HTTP ${resp.status}`);
            }
            tables[lang] = await resp.json();
        } catch (err) {
            console.warn(`i18n: failed to load ${lang}, falling back to ${DEFAULT_LANG}`, err);
            if (lang === DEFAULT_LANG) {
                tables[lang] = {};  // prevent infinite retry
            } else {
                tables[lang] = {};
            }
        }
        return tables[lang];
    }

    async function setLang(lang) {
        if (!SUPPORTED.includes(lang)) {
            console.warn(`i18n: unsupported language ${lang}`);
            return;
        }
        // English is the lookup fallback for missing keys, so it must
        // always be loaded — otherwise a Hindi-first session renders raw
        // `[key]` placeholders for any key missing in Hindi.
        await loadLang(DEFAULT_LANG);
        await loadLang(lang);
        activeLang = lang;
        localStorage.setItem(STORAGE_KEY, lang);
        document.documentElement.lang = lang;
    }

    /**
     * Look up a key in the active language; if not found, fall back
     * to English; if still not found, return the key itself.
     * Substitutes ``{name}``-style placeholders from ``params``.
     */
    function t(key, params) {
        const table = tables[activeLang] || {};
        let raw = table[key];
        if (raw === undefined) {
            const fallback = tables[DEFAULT_LANG] || {};
            raw = fallback[key];
        }
        if (raw === undefined) {
            return `[${key}]`;
        }
        if (params) {
            return raw.replace(/\{(\w+)\}/g, (m, name) => {
                return name in params ? String(params[name]) : m;
            });
        }
        return raw;
    }

    function getLang() {
        return activeLang;
    }

    function listSupported() {
        return SUPPORTED.slice();
    }

    // Expose
    window.I18N = {
        setLang,
        getLang,
        t,
        listSupported,
        detectInitialLang,
    };
})();
