/* ===================================================================
   Chitragupta Chat UI — main controller

   Responsibilities:
   - Bootstrap the i18n table, load the active language
   - Manage the user profile (first-launch modal, settings drawer)
   - Run the chat loop: send messages, render bot replies, handle
     confirmation popups, manage sessions
   - TTS playback via window.Audio
   - Language switch persists per session via /api/language
   - Theme switch (light/dark) persists in localStorage

   The backend is unchanged: this file is the only consumer of the
   new /api/chat, /api/sessions, /api/confirm, /api/language, and
   /api/tts endpoints.
   =================================================================== */
(function () {
    'use strict';

    const API = '';  // same-origin
    const PROFILE_KEY = 'chitragupta.profile';
    const THEME_KEY = 'chitragupta.theme';
    const AUTOREAD_KEY = 'chitragupta.autoread';
    const TOKEN_KEY = 'chitragupta.token';
    const AUTH_USER_KEY = 'chitragupta.auth_user';

    // ─── Auth (V1 multi-user: JWT, 7d sliding) ──────────────
    // Single fetch wrapper: injects Bearer for same-origin /api/* calls,
    // retries once after a silent refresh on 401. Call sites unchanged.
    const Auth = {
        mode: 'login',
        getToken() { try { return localStorage.getItem(TOKEN_KEY); } catch { return null; } },
        setSession(token, user) {
            try {
                if (token) localStorage.setItem(TOKEN_KEY, token); else localStorage.removeItem(TOKEN_KEY);
                if (user) localStorage.setItem(AUTH_USER_KEY, JSON.stringify(user)); else localStorage.removeItem(AUTH_USER_KEY);
            } catch { /* private mode */ }
            const label = document.getElementById('auth-email-label');
            if (label) label.textContent = user && user.email ? user.email : '';
        },
        getUser() { try { const raw = localStorage.getItem(AUTH_USER_KEY); return raw ? JSON.parse(raw) : null; } catch { return null; } },
        async refresh() {
            const token = this.getToken();
            if (!token) return null;
            try {
                const r = await window.__rawFetch(API + '/api/auth/refresh', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + token },
                    body: JSON.stringify({ token }),
                });
                if (!r.ok) return null;
                const data = await r.json();
                this.setSession(data.token, { user_id: data.user_id, email: (this.getUser() || {}).email });
                return data.token;
            } catch { return null; }
        },
    };

    if (!window.__rawFetch) {
        window.__rawFetch = window.fetch.bind(window);
        window.fetch = async function (url, opts) {
            opts = opts || {};
            const token = Auth.getToken();
            if (typeof url === 'string' && url.includes('/api/') && token) {
                opts.headers = Object.assign({}, opts.headers, { 'Authorization': 'Bearer ' + token });
            }
            let r = await window.__rawFetch(url, opts);
            if (r.status === 401 && typeof url === 'string' && url.includes('/api/') && !url.includes('/api/auth/')) {
                const fresh = await Auth.refresh();
                if (fresh) {
                    opts.headers = Object.assign({}, opts.headers, { 'Authorization': 'Bearer ' + fresh });
                    r = await window.__rawFetch(url, opts);
                } else {
                    Auth.setSession(null, null);
                    const modal = document.getElementById('auth-modal');
                    if (modal) modal.hidden = false;
                }
            }
            return r;
        };
    }

    function setupAuth() {
        const modal = $('auth-modal') || document.getElementById('auth-modal');
        const form = $('auth-form') || document.getElementById('auth-form');
        const emailInput = document.getElementById('auth-email-input');
        const passInput = document.getElementById('auth-password-input');
        const errBox = document.getElementById('auth-modal-error');
        const modeToggle = document.getElementById('auth-mode-toggle');
        const submitBtn = document.getElementById('auth-submit-btn');
        const logoutBtn = document.getElementById('btn-logout');
        if (!modal || !form) return;
        const showError = (msg) => { if (errBox) { errBox.textContent = msg; errBox.hidden = false; } };
        if (!Auth.getToken()) modal.hidden = false;
        else { modal.hidden = true; const u = Auth.getUser(); if (u && u.email) { const l = document.getElementById('auth-email-label'); if (l) l.textContent = u.email; } }
        if (modeToggle) modeToggle.addEventListener('click', () => {
            Auth.mode = Auth.mode === 'login' ? 'register' : 'login';
            if (submitBtn) submitBtn.textContent = Auth.mode === 'login' ? t('auth_login') : t('auth_register');
            modeToggle.textContent = Auth.mode === 'login' ? t('auth_need_account') : t('auth_have_account');
        });
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            if (errBox) errBox.hidden = true;
            const email = (emailInput.value || '').trim();
            const password = passInput.value || '';
            try {
                const r = await window.__rawFetch(API + '/api/auth/' + Auth.mode, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ email, password }),
                });
                const data = await r.json().catch(() => ({}));
                if (!r.ok) throw new Error(data.detail || ('HTTP ' + r.status));
                Auth.setSession(data.token, { user_id: data.user_id, email: data.email });
                modal.hidden = true;
                // Fresh login on shared device: pull global manifest note.
                try {
                    const m = await window.__rawFetch(API + '/api/sync/manifest', {
                        headers: { 'Authorization': 'Bearer ' + data.token },
                    });
                    if (m.ok) { const man = await m.json(); console.info('sync manifest docs:', Object.keys(man.documents || {}).length); }
                } catch (err) { console.warn('sync manifest check failed', err); }
                setupProfile();
                await ensureSession();
                await refreshSessionList();
            } catch (err) {
                showError(String((err && err.message) || err));
            }
        });
        if (logoutBtn) logoutBtn.addEventListener('click', async () => {
            const token = Auth.getToken();
            const user = Auth.getUser();
            try {
                await window.__rawFetch(API + '/api/auth/logout', {
                    method: 'POST',
                    headers: Object.assign({ 'Content-Type': 'application/json' }, token ? { 'Authorization': 'Bearer ' + token } : {}),
                    body: JSON.stringify({ wipe: true, user_id: user && user.user_id }),
                });
            } catch { /* best-effort wipe */ }
            Auth.setSession(null, null);
            try { localStorage.removeItem(PROFILE_KEY); } catch { /* private mode */ }
            state.currentSessionId = null;
            state.profile = null;
            modal.hidden = false;
        });
    }

    // ─── State ─────────────────────────────────────────────────
    const state = {
        currentSessionId: null,
        profile: null,                 // {display_name}
        isLoading: false,
        pendingConfirmation: null,
        pendingFiles: [],
        fileInput: null,             // hidden <input type=file>, owned by setupComposer
        sessions: [],
        ttsPlayingFor: null,           // id of message currently being spoken
    };

    // ─── DOM ───────────────────────────────────────────────────
    const $ = (id) => document.getElementById(id);
    const els = {
        // Profile
        profileModal: $('profile-modal'),
        profileForm: $('profile-form'),
        profileNameInput: $('profile-name-input'),
        profileModalGreeting: $('profile-modal-greeting'),
        profileModalHelper: $('profile-modal-helper'),
        // Settings
        settingsDrawer: $('settings-drawer'),
        settingsForm: $('settings-form'),
        settingsNameInput: $('settings-name-input'),
        settingsClose: $('settings-close'),
        settingsCancel: $('settings-cancel'),
        themeToggle: $('theme-toggle'),
        autoreadToggle: $('autoread-toggle'),
        // Brand
        brandName: $('brand-name'),
        brandTagline: $('brand-tagline'),
        // Sidebar
        sidebar: $('sidebar'),
        sessionList: $('session-list'),
        btnNewChat: $('btn-new-chat'),
        btnSidebarToggle: $('btn-sidebar-toggle'),
        btnOpenSettings: $('btn-open-settings'),
        // Chat
        chatTitle: $('chat-title'),
        messages: $('messages'),
        welcome: $('welcome'),
        welcomeGreeting: $('welcome-greeting'),
        welcomeSubtitle: $('welcome-subtitle'),
        // Confirmation pill
        confirmPill: $('confirm-pill'),
        expiryBanner: $('expiry-banner'),
        expiryTitle: $('expiry-title'),
        expirySub: $('expiry-sub'),
        btnExpiryView: $('btn-expiry-view'),
        btnExpiryDismiss: $('btn-expiry-dismiss'),
        confirmMessage: $('confirm-message'),
        confirmCorrection: $('confirm-correction'),
        btnApprove: $('btn-approve'),
        btnDeny: $('btn-deny'),
        // Upload strip
        uploadStrip: $('upload-strip'),
        uploadStripContent: $('upload-strip-content'),
        btnUploadRemove: $('btn-upload-remove'),
        // Composer
        messageInput: $('message-input'),
        btnSend: $('btn-send'),
        btnAttach: $('btn-attach'),
        btnMic: $('btn-mic'),
        // Language switch
        langHi: $('lang-hi'),
        langEn: $('lang-en'),
    };

    // ─── i18n helpers ──────────────────────────────────────────
    function t(key, params) { return window.I18N.t(key, params); }

    // Friendly display name for a stored field ("aadhaar_number" →
    // "आधार नंबर"). Unknown fields fall back to the raw name so the
    // popup never renders a [missing-key] placeholder to the user.
    function fieldLabel(name) {
        if (!name) return '';
        const key = 'field_label_' + String(name).toLowerCase();
        const label = t(key);
        return (label === `[${key}]`) ? String(name) : label;
    }

    function applyI18nToStatic() {
        // Walk the document and replace any element with [data-i18n].
        document.querySelectorAll('[data-i18n]').forEach((el) => {
            el.textContent = t(el.dataset.i18n);
        });
        document.querySelectorAll('[data-i18n-aria]').forEach((el) => {
            el.setAttribute('aria-label', t(el.dataset.i18nAria));
        });
    }

    // ─── Theme ─────────────────────────────────────────────────
    function applyTheme(theme) {
        document.body.dataset.theme = theme;
        els.themeToggle.checked = (theme === 'dark');
    }

    function setupTheme() {
        const stored = localStorage.getItem(THEME_KEY);
        applyTheme(stored || 'light');
        els.themeToggle.addEventListener('change', () => {
            const next = els.themeToggle.checked ? 'dark' : 'light';
            localStorage.setItem(THEME_KEY, next);
            applyTheme(next);
        });
    }

    // ─── Auto-read replies aloud ───────────────────────────────
    // Explicit choice wins; otherwise Hindi sessions default ON
    // (elderly-first) and English sessions default OFF.
    function isAutoreadOn() {
        const stored = localStorage.getItem(AUTOREAD_KEY);
        if (stored === '1') return true;
        if (stored === '0') return false;
        return window.I18N.getLang() === 'hi';
    }

    function applyAutoreadToggle() {
        els.autoreadToggle.checked = isAutoreadOn();
    }

    function setupAutoread() {
        applyAutoreadToggle();
        els.autoreadToggle.addEventListener('change', () => {
            localStorage.setItem(AUTOREAD_KEY, els.autoreadToggle.checked ? '1' : '0');
        });
    }

    function stopAllTts() {
        if (!window.Audio.isPlaying()) return;
        window.Audio.stop();
        state.ttsPlayingFor = null;
        document.querySelectorAll('.tts-btn.playing').forEach((b) => {
            b.classList.remove('playing');
            b.innerHTML = `<span>${escapeHtml(t('bot_listen'))}</span>`;
        });
    }

    function autoSpeakRow(row) {
        if (!isAutoreadOn() || !row) return;
        const btn = row.querySelector('.tts-btn');
        // Small delay so the bubble paints first; a tap meanwhile wins
        // because onTtsClick toggles (a second click stops).
        if (btn) setTimeout(() => { if (document.contains(btn)) btn.click(); }, 350);
    }

    // ─── Profile ───────────────────────────────────────────────
    function loadProfile() {
        try {
            const raw = localStorage.getItem(PROFILE_KEY);
            if (!raw) return null;
            const p = JSON.parse(raw);
            return p && p.display_name ? p : null;
        } catch {
            return null;
        }
    }

    function saveProfile(p) {
        localStorage.setItem(PROFILE_KEY, JSON.stringify(p));
    }

    // Tell the server our display name so the persona addresses us by
    // name (it otherwise falls back to "friend"). Fire-and-forget: chat
    // must never block on this.
    async function pushProfileToServer(name) {
        const displayName = (name || '').trim();
        if (!displayName) return;
        try {
            await fetch(API + '/api/profile', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ display_name: displayName.slice(0, 100) }),
            });
        } catch (e) {
            console.warn('pushProfileToServer failed', e);
        }
    }

    function showProfileModal() {
        els.profileModal.hidden = false;
        els.profileNameInput.value = '';
        setTimeout(() => els.profileNameInput.focus(), 50);
    }
    function hideProfileModal() {
        els.profileModal.hidden = true;
    }

    function applyProfile() {
        const name = state.profile ? state.profile.display_name : '';
        // Greeting uses the user's name. The welcome screen says
        // "नमस्ते, शर्मा जी" in Hindi or "Hello, Sharma" in English.
        els.welcomeGreeting.textContent = t('welcome_greeting', { name });
        els.welcomeSubtitle.textContent = t('welcome_subtitle');
        // Modal helper: "मैं आपको शर्मा जी कहकर बुलाऊँगा।"
        els.profileModalHelper.textContent = t('profile_setup_helper', { name: name || '...' });
        if (state.profile) {
            els.settingsNameInput.value = state.profile.display_name;
        }
    }

    let _profileWired = false;
    function setupProfile() {
        state.profile = loadProfile();
        // Auth comes first: never stack the name modal under the login modal.
        const authed = Auth.getToken();
        if (!state.profile && authed) {
            showProfileModal();
            // One-tap language choice during onboarding: applies live so
            // the rest of the modal immediately speaks their language.
            const markOnboardingLang = () => {
                const cur = window.I18N.getLang();
                document.querySelectorAll('.profile-lang-switch .lang-chip').forEach((chip) => {
                    chip.classList.toggle('active', chip.dataset.langCode === cur);
                });
            };
            document.querySelectorAll('.profile-lang-switch .lang-chip').forEach((chip) => {
                chip.addEventListener('click', async () => {
                    await setLanguage(chip.dataset.langCode);
                    markOnboardingLang();
                });
            });
            markOnboardingLang();
        } else if (state.profile) {
            applyProfile();
        }
        // Idempotent: login calls setupProfile() again — never double-bind.
        if (_profileWired) return;
        _profileWired = true;
        els.profileForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const name = els.profileNameInput.value.trim();
            if (!name) return;
            state.profile = { display_name: name };
            saveProfile(state.profile);
            hideProfileModal();
            applyProfile();
            pushProfileToServer(name);
            ensureSession();
        });
    }

    // ─── Sessions ──────────────────────────────────────────────
    async function listSessions() {
        try {
            const r = await fetch(API + '/api/sessions');
            if (!r.ok) return [];
            const data = await r.json();
            return data.sessions || [];
        } catch {
            return [];
        }
    }

    function renderSessionList() {
        els.sessionList.innerHTML = '';
        if (!state.sessions.length) {
            const empty = document.createElement('div');
            empty.className = 'empty-sidebar';
            empty.textContent = t('empty_state');
            els.sessionList.appendChild(empty);
            return;
        }
        for (const s of state.sessions) {
            const item = document.createElement('div');
            item.className = 'session-item' + (s.session_id === state.currentSessionId ? ' active' : '');
            item.setAttribute('role', 'listitem');
            item.dataset.sessionId = s.session_id;

            const title = document.createElement('div');
            title.className = 'session-title';
            title.textContent = s.title || t('nav_new_chat');

            const time = document.createElement('div');
            time.className = 'session-time';
            time.textContent = formatRelativeTime(s.created_at);

            item.appendChild(title);
            item.appendChild(time);
            // Keyboard access: session rows are divs, so expose them
            // to Tab and activate on Enter/Space like a button.
            item.tabIndex = 0;
            item.setAttribute('aria-label', s.title || t('nav_new_chat'));
            const activate = () => selectSession(s.session_id);
            item.addEventListener('click', activate);
            item.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    activate();
                }
            });
            els.sessionList.appendChild(item);
        }
    }

    function formatRelativeTime(iso) {
        if (!iso) return '';
        // NOTE: do not name this `t` — it would shadow the i18n `t()`
        // helper used below (that shadowing once blanked the sidebar).
        const ts = new Date(iso).getTime();
        if (Number.isNaN(ts)) return '';
        const diffSec = Math.max(0, Math.floor((Date.now() - ts) / 1000));
        if (diffSec < 60) return t('session_just_now');
        if (diffSec < 3600) return t('session_minutes_ago', { n: Math.floor(diffSec / 60) });
        if (diffSec < 86400) return t('session_hours_ago', { n: Math.floor(diffSec / 3600) });
        return t('session_days_ago', { n: Math.floor(diffSec / 86400) });
    }

    async function ensureSession() {
        if (state.currentSessionId) return state.currentSessionId;
        const r = await fetch(API + '/api/sessions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: t('nav_new_chat') }),
        });
        const data = await r.json();
        state.currentSessionId = data.session_id;
        await refreshSessionList();
        await applyLanguagePreferenceToCurrentSession();
        return state.currentSessionId;
    }

    async function refreshSessionList() {
        state.sessions = await listSessions();
        renderSessionList();
    }

    async function selectSession(sessionId) {
        state.currentSessionId = sessionId;
        // Mobile: close the sidebar after selecting.
        els.sidebar.classList.remove('open');
        await loadSessionMessages(sessionId);
        await applyLanguagePreferenceToCurrentSession();
        renderSessionList();
        checkExpiring();
    }

    async function loadSessionMessages(sessionId) {
        try {
            const r = await fetch(API + '/api/sessions/' + sessionId);
            if (!r.ok) return;
            const data = await r.json();
            els.chatTitle.textContent = data.title || t('nav_new_chat');
            if (data.messages && data.messages.length) {
                hideWelcome();
                renderMessages(data.messages);
            } else {
                // Empty session (e.g. just created): keep the welcome
                // screen instead of showing a blank chat.
                renderMessages([]);
                showWelcome();
            }
        } catch (e) {
            console.warn('loadSessionMessages failed', e);
        }
    }

    // ─── Language switch ──────────────────────────────────────
    function setLanguageSwitchActive(code) {
        // Only hi/en have chips (ta/bn have backend translation but no
        // shipped string tables, so their chrome falls back to English).
        // Highlight EN in that case so the switch always reflects the
        // language the chrome is actually rendered in.
        const chips = [[els.langHi, 'hi'], [els.langEn, 'en']];
        const effective = chips.some(([, c]) => c === code) ? code : 'en';
        for (const [el, codeExpected] of chips) {
            if (!el) continue;
            el.classList.toggle('active', effective === codeExpected);
        }
    }

    async function applyLanguagePreferenceToCurrentSession() {
        if (!state.currentSessionId) return;
        const lang = window.I18N.getLang();
        // Sessions persist their preference server-side; skip the POST
        // when we already pushed this language for this session.
        state.pushedLangBySession = state.pushedLangBySession || {};
        if (state.pushedLangBySession[state.currentSessionId] === lang) return;
        try {
            const r = await fetch(API + '/api/language', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: state.currentSessionId,
                    source: lang,
                    target: lang,
                }),
            });
            if (r.ok) state.pushedLangBySession[state.currentSessionId] = lang;
        } catch (e) {
            console.warn('applyLanguagePreferenceToCurrentSession failed', e);
        }
    }

    async function setLanguage(code) {
        if (!window.I18N.listSupported().includes(code)) return;
        await window.I18N.setLang(code);
        setLanguageSwitchActive(code);
        applyI18nToStatic();
        applyProfile();
        applyAutoreadToggle();   // default follows language until chosen
        renderSessionList();   // re-render so localized time labels update
        if (state.currentSessionId) {
            await applyLanguagePreferenceToCurrentSession();
        }
    }

    function setupLanguageSwitch() {
        els.langHi.addEventListener('click', () => setLanguage('hi'));
        els.langEn.addEventListener('click', () => setLanguage('en'));
    }

    // ─── Welcome screen ───────────────────────────────────────
    function showWelcome() {
        els.welcome.hidden = false;
        els.messages.appendChild(els.welcome);
        els.chatTitle.textContent = t('nav_new_chat');
    }
    function hideWelcome() {
        els.welcome.hidden = true;
    }

    function setupLifeCards() {
        document.querySelectorAll('.life-card').forEach((card) => {
            card.addEventListener('click', () => {
                const prompt = card.dataset.quick;
                if (prompt) {
                    els.messageInput.value = prompt;
                    autosize();
                    sendMessage();
                }
            });
        });
    }

    // ─── Messages ─────────────────────────────────────────────
    function renderMessages(messages) {
        // Wipe everything except the welcome screen.
        Array.from(els.messages.querySelectorAll('.message, .thinking-row')).forEach((el) => el.remove());
        for (const m of messages) {
            appendMessage(m.role, m.content, m.message_id);
        }
        scrollToBottom();
    }

    function appendMessage(role, content, messageId) {
        if (els.welcome && !els.welcome.hidden) hideWelcome();
        const row = document.createElement('div');
        row.className = `message ${role}`;
        row.dataset.messageId = messageId || '';

        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = role === 'user' ? (state.profile ? state.profile.display_name[0] : 'U') : 'C';

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        bubble.textContent = content;

        row.appendChild(avatar);
        row.appendChild(bubble);

        if (role === 'assistant') {
            const actions = document.createElement('div');
            actions.className = 'message-actions';
            const ttsBtn = document.createElement('button');
            ttsBtn.className = 'tts-btn';
            ttsBtn.dataset.messageId = messageId || '';
            ttsBtn.innerHTML = `<span>${escapeHtml(t('bot_listen'))}</span>`;
            ttsBtn.addEventListener('click', () => onTtsClick(messageId, content, ttsBtn));
            actions.appendChild(ttsBtn);
            bubble.appendChild(actions);
        }

        els.messages.appendChild(row);
        scrollToBottom();
        return row;
    }

    function showThinkingRow() {
        const row = document.createElement('div');
        row.className = 'message assistant thinking-row';
        const avatar = document.createElement('div');
        avatar.className = 'message-avatar';
        avatar.textContent = 'C';
        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        const span = document.createElement('span');
        span.className = 'thinking';
        span.innerHTML = `<span>${escapeHtml(t('bot_thinking'))}</span><span class="thinking-dots"><span></span><span></span><span></span></span>`;
        bubble.appendChild(span);
        row.appendChild(avatar);
        row.appendChild(bubble);
        els.messages.appendChild(row);
        scrollToBottom();
        return row;
    }

    function removeThinkingRow() {
        const r = els.messages.querySelector('.thinking-row');
        if (r) r.remove();
    }

    function escapeHtml(s) {
        const d = document.createElement('div');
        d.textContent = s;
        return d.innerHTML;
    }

    function scrollToBottom() {
        requestAnimationFrame(() => {
            els.messages.scrollTop = els.messages.scrollHeight;
        });
    }

    // ─── TTS ───────────────────────────────────────────────────
    function onTtsClick(messageId, text, btn) {
        if (window.Audio.isPlaying() && state.ttsPlayingFor === messageId) {
            window.Audio.stop();
            btn.classList.remove('playing');
            btn.innerHTML = `<span>${escapeHtml(t('bot_listen'))}</span>`;
            state.ttsPlayingFor = null;
            return;
        }
        // Stop any other TTS first
        if (window.Audio.isPlaying()) {
            window.Audio.stop();
            document.querySelectorAll('.tts-btn.playing').forEach((b) => {
                b.classList.remove('playing');
                b.innerHTML = `<span>${escapeHtml(t('bot_listen'))}</span>`;
            });
        }
        const lang = window.I18N.getLang() === 'hi' ? 'hi-IN' : 'en-US';
        btn.classList.add('playing');
        btn.innerHTML = `<span>${escapeHtml(t('stop_speaking'))}</span>`;
        state.ttsPlayingFor = messageId;
        window.Audio.speak({
            text,
            lang,
            onEnd: () => {
                btn.classList.remove('playing');
                btn.innerHTML = `<span>${escapeHtml(t('bot_listen'))}</span>`;
                if (state.ttsPlayingFor === messageId) state.ttsPlayingFor = null;
            },
            onError: (err) => {
                btn.classList.remove('playing');
                btn.innerHTML = `<span>${escapeHtml(t('error_tts'))}</span>`;
                console.warn('TTS error:', err);
            },
        });
    }

    // ─── Send message ─────────────────────────────────────────
    async function sendMessage() {
            if (state.isLoading) return;
            const text = els.messageInput.value.trim();
            if (!text && !state.pendingFiles.length) return;

            state.isLoading = true;
            els.btnSend.disabled = true;
            els.messageInput.value = '';
            autosize();

            try {
                // Ensure a session exists BEFORE building the body, so the
                // first-ever message carries a real session_id (previously it
                // was sent with an empty id and rejected with 400).
                await ensureSession();

                // Prepare form data if there's a file
                let body;
                let headers;
                if (state.pendingFiles.length > 0) {
                    const file = state.pendingFiles[0];
                    const formData = new FormData();
                    formData.append('session_id', state.currentSessionId || '');
                    formData.append('message', text);
                    formData.append('file', file);
                    body = formData;
                    headers = {}; // browser sets content-type with boundary
                } else {
                    body = JSON.stringify({
                        session_id: state.currentSessionId,
                        message: text,
                    });
                    headers = { 'Content-Type': 'application/json' };
                }

                stopAllTts();
                appendMessage('user', text || '(photo attached)');
                showThinkingRow();
                scrollToBottom();

                const r = await fetch(API + '/api/chat', {
                    method: 'POST',
                    headers,
                    body,
                });
                // Clear pending files after sending
                state.pendingFiles = [];
                renderUploadStrip();
                updateSendButton();

                removeThinkingRow();
                if (!r.ok) throw new Error(`HTTP ${r.status}`);
                const data = await r.json();

                if (data.confirmation_required) {
                    showConfirmation(data.confirmation_required);
                } else {
                    hideConfirmation();
                }
                if (data.message) {
                    const messageId = 'msg-' + Date.now();
                    autoSpeakRow(appendMessage('assistant', data.message, messageId));
                }
                // No session-list refetch here: the backend never retitles
                // or reorders on new messages, and the session was already
                // listed by ensureSession() above. (Re-fetch on demand via
                // selectSession / new-chat instead of every turn.)
            } catch (e) {
                removeThinkingRow();
                console.error('sendMessage failed', e);
                appendMessage('assistant', t('error_network'));
            } finally {
                state.isLoading = false;
                updateSendButton();
            }
        }

    function updateSendButton() {
        const hasContent = els.messageInput.value.trim() || state.pendingFiles.length;
        els.btnSend.disabled = !hasContent || state.isLoading;
    }

    function autosize() {
        const ta = els.messageInput;
        ta.style.height = 'auto';
        ta.style.height = Math.min(ta.scrollHeight, 160) + 'px';
    }

    function setupComposer() {
            // File attach: open a hidden file input for image selection
            const fileInput = document.createElement('input');
            fileInput.type = 'file';
            fileInput.accept = 'image/*';
            fileInput.style.display = 'none';
            fileInput.addEventListener('change', (e) => handleFileSelect(e.target.files));
            document.body.appendChild(fileInput);
            state.fileInput = fileInput;

            els.btnAttach.addEventListener('click', () => fileInput.click());

            // The ✕ chip button was previously never wired up.
            els.btnUploadRemove.addEventListener('click', () => {
                state.pendingFiles = [];
                if (state.fileInput) state.fileInput.value = '';
                renderUploadStrip();
                updateSendButton();
                els.messageInput.focus();
            });

            els.btnSend.addEventListener('click', sendMessage);
            setupMic();
            els.messageInput.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    sendMessage();
                }
            });
            els.messageInput.addEventListener('input', () => {
                autosize();
                updateSendButton();
            });
            updateSendButton();
        }

        // ─── Voice input (Web Speech API; server STT reserved for later) ──
        function speechRecognitionClass() {
            return window.SpeechRecognition || window.webkitSpeechRecognition || null;
        }

        function setupMic() {
            const SR = speechRecognitionClass();
            if (!SR) {
                // No voice support (old browser): hide the button rather
                // than show a dead control to elderly users.
                els.btnMic.hidden = true;
                return;
            }
            els.btnMic.setAttribute('aria-label', t('input_mic'));
            els.btnMic.addEventListener('click', toggleMic);
        }

        function recognitionLang() {
            const lang = window.I18N.getLang();
            return lang === 'hi' ? 'hi-IN' : lang === 'en' ? 'en-US' : lang;
        }

        function toggleMic() {
            if (state.recognition) {
                state.recognition.stop();
                return;
            }
            const SR = speechRecognitionClass();
            if (!SR) return;
            const rec = new SR();
            rec.lang = recognitionLang();
            rec.interimResults = true;
            rec.maxAlternatives = 1;
            // Append (don't replace): the user may have typed + spoken.
            const base = els.messageInput.value;
            const basePrefix = base && !base.endsWith(' ') ? base + ' ' : base;
            rec.onresult = (e) => {
                let interim = '';
                let finalText = '';
                for (const res of e.results) {
                    if (res.isFinal) finalText += res[0].transcript;
                    else interim += res[0].transcript;
                }
                els.messageInput.value = basePrefix + (finalText || interim);
                autosize();
                updateSendButton();
            };
            rec.onerror = (e) => {
                if (e.error === 'not-allowed' || e.error === 'service-not-allowed') {
                    alert(t('error_mic'));
                }
                // 'no-speech'/'aborted' are normal; stay silent.
            };
            rec.onend = () => {
                state.recognition = null;
                els.btnMic.classList.remove('listening');
                els.btnMic.setAttribute('aria-label', t('input_mic'));
                els.messageInput.focus();
            };
            state.recognition = rec;
            els.btnMic.classList.add('listening');
            els.btnMic.setAttribute('aria-label', t('stop_speaking'));
            try {
                rec.start();
            } catch (e) {
                state.recognition = null;
                els.btnMic.classList.remove('listening');
            }
        }

        function handleFileSelect(files) {
            if (!files || !files.length) return;
            const file = files[0];
            // Basic validation: image only, max 10MB
            if (!file.type.startsWith('image/')) {
                alert(t('error_upload'));
                return;
            }
            if (file.size > 10 * 1024 * 1024) {
                alert(t('error_upload'));
                return;
            }
            // Store for sending with next message. Reset the input so
            // picking the same file again still fires a change event.
            state.pendingFiles = [file];
            if (state.fileInput) state.fileInput.value = '';
            renderUploadStrip();
            updateSendButton();
        }

        function renderUploadStrip() {
            const content = els.uploadStripContent;
            content.innerHTML = '';
            for (const file of state.pendingFiles) {
                const chip = document.createElement('div');
                chip.className = 'upload-chip';
                chip.textContent = file.name;
                content.appendChild(chip);
            }
            els.uploadStrip.hidden = state.pendingFiles.length === 0;
        }

    // ─── Expiry banner (proactive renewal reminder) ────────────
    function docDisplayName(d) {
        return d.description || d.summary
            || (d.document_id ? String(d.document_id).slice(0, 8) + '…' : '');
    }

    async function checkExpiring() {
        hideExpiryBanner();
        const sid = state.currentSessionId;
        if (!sid) return;
        // Once per session per pageload: quiet, never nagging.
        state.expiryChecked = state.expiryChecked || new Set();
        if (state.expiryChecked.has(sid)) return;
        state.expiryChecked.add(sid);
        let items = [];
        try {
            const r = await fetch(API + '/api/expiring?within_days=30');
            if (!r.ok) return;
            items = (await r.json()).results || [];
        } catch (e) {
            console.warn('checkExpiring failed', e);
            return;
        }
        if (!items.length) return;
        const first = items[0];
        const days = first.days_until_expiry;
        const name = docDisplayName(first);
        els.expiryTitle.textContent = t('expiry_banner_title', { count: items.length });
        els.expirySub.textContent = days < 0
            ? t('expiry_banner_overdue', { name, days: Math.abs(days) })
            : t('expiry_banner_soon', { name, days });
        els.btnExpiryView.textContent = t('expiry_banner_view');
        els.expiryBanner.hidden = false;
    }

    function hideExpiryBanner() {
        els.expiryBanner.hidden = true;
    }

    function setupExpiryBanner() {
        els.btnExpiryDismiss.addEventListener('click', hideExpiryBanner);
        els.btnExpiryView.addEventListener('click', () => {
            hideExpiryBanner();
            els.messageInput.value = t('expiry_banner_ask');
            autosize();
            sendMessage();
        });
    }

    // ─── Confirmation pill ───────────────────────────────────
    function showConfirmation(data) {
        state.pendingConfirmation = data;
        // Build a friendlier confirmation copy: "Should I show your aadhaar_number?"
        const toolName = data.type || data.tool_name || '';
        const fieldMatch = (data.tool_args && (data.tool_args.field || data.tool_args.field_name)) || '';
        const isField = (toolName === 'get_field_value') && fieldMatch;
        if (isField) {
            els.confirmMessage.textContent =
                `${t('confirmation_title')} ${fieldLabel(fieldMatch)} ${t('confirmation_field_label')}`;
        } else {
            els.confirmMessage.textContent = data.message || t('confirmation_title').trim();
        }
        els.confirmCorrection.value = '';
        els.confirmCorrection.placeholder = t('confirmation_correction_placeholder');
        state.confirmBusy = false;
        els.btnApprove.disabled = false;
        els.btnDeny.disabled = false;
        els.confirmPill.hidden = false;
        // Re-localize the buttons (they have static HTML labels)
        els.btnApprove.textContent = t('confirmation_yes');
        els.btnDeny.textContent = t('confirmation_no');
    }

    function hideConfirmation() {
        state.pendingConfirmation = null;
        els.confirmPill.hidden = true;
    }

    async function respondConfirmation(approved) {
        if (!state.pendingConfirmation || state.confirmBusy) return;
        state.confirmBusy = true;
        els.btnApprove.disabled = true;
        els.btnDeny.disabled = true;
        const correction = els.confirmCorrection.value.trim() || null;
        const body = {
            session_id: state.currentSessionId,
            request_id: state.pendingConfirmation.request_id,
            approved,
            correction,
        };
        try {
            const r = await fetch(API + '/api/confirm', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body),
            });
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
            const data = await r.json();
            hideConfirmation();
            if (data.message) {
                autoSpeakRow(appendMessage('assistant', data.message, 'msg-' + Date.now()));
            }
        } catch (e) {
            console.error('respondConfirmation failed', e);
        } finally {
            state.confirmBusy = false;
            els.btnApprove.disabled = false;
            els.btnDeny.disabled = false;
        }
    }

    function setupConfirmation() {
        els.btnApprove.addEventListener('click', () => respondConfirmation(true));
        els.btnDeny.addEventListener('click', () => respondConfirmation(false));
        // Enter in the correction box approves (it is a single-line input).
        els.confirmCorrection.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                respondConfirmation(true);
            }
        });
    }

    // ─── Settings drawer ─────────────────────────────────────
    function openSettings() {
        if (state.profile) els.settingsNameInput.value = state.profile.display_name;
        els.settingsDrawer.hidden = false;
        setTimeout(() => els.settingsNameInput.focus(), 50);
    }
    function closeSettings() {
        els.settingsDrawer.hidden = true;
    }

    function setupSettings() {
        els.btnOpenSettings.addEventListener('click', openSettings);
        els.settingsClose.addEventListener('click', closeSettings);
        els.settingsCancel.addEventListener('click', closeSettings);
        // Clicking the dimmed backdrop (not the card) closes the drawer.
        els.settingsDrawer.addEventListener('click', (e) => {
            if (e.target === els.settingsDrawer) closeSettings();
        });
        // Esc closes drawer first, then the mobile sidebar.
        document.addEventListener('keydown', (e) => {
            if (e.key !== 'Escape') return;
            if (!els.settingsDrawer.hidden) {
                closeSettings();
            } else if (els.sidebar.classList.contains('open')) {
                els.sidebar.classList.remove('open');
            }
        });
        els.settingsForm.addEventListener('submit', (e) => {
            e.preventDefault();
            const name = els.settingsNameInput.value.trim();
            if (name) {
                state.profile = { display_name: name };
                saveProfile(state.profile);
                applyProfile();
                pushProfileToServer(name);
            }
            closeSettings();
        });
    }

    // ─── Sidebar toggle (mobile) ─────────────────────────────
    function setupSidebarToggle() {
        els.btnSidebarToggle.addEventListener('click', () => {
            els.sidebar.classList.toggle('open');
        });
        els.btnNewChat.addEventListener('click', async () => {
                    state.currentSessionId = null;
                    state.pendingFiles = [];
                    renderUploadStrip();
                    els.messages.innerHTML = '';
                    showWelcome();
                    await ensureSession();
                    await applyLanguagePreferenceToCurrentSession();
                    checkExpiring();
                    // Flash the new chat button as feedback
                    els.btnNewChat.style.transition = 'transform 0.1s';
                    els.btnNewChat.style.transform = 'scale(0.9)';
                    setTimeout(() => {
                        els.btnNewChat.style.transform = '';
                    }, 100);
                });
    }

    // ─── Bootstrap ───────────────────────────────────────────
    async function main() {
        const initial = window.I18N.detectInitialLang();
        await setLanguage(initial);
        setupTheme();
        setupAutoread();
        setupAuth();
        setupProfile();
        setupLanguageSwitch();
        setupLifeCards();
        setupComposer();
        setupConfirmation();
        setupExpiryBanner();
        setupSettings();
        setupSidebarToggle();
        applyI18nToStatic();
        if (state.profile) {
            applyProfile();
            pushProfileToServer(state.profile.display_name);
            await ensureSession();
            await applyLanguagePreferenceToCurrentSession();
        }
        await refreshSessionList();
        if (state.profile && state.currentSessionId) {
            await loadSessionMessages(state.currentSessionId);
        }
    }

    document.addEventListener('DOMContentLoaded', main);
})();
