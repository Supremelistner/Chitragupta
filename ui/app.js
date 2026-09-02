/* ===================================================================
   Chitragupta Chat UI — app.js
   =================================================================== */

const API = '';  // Same origin
let currentSessionId = null;
let pendingConfirmation = null;
let isLoading = false;
let pendingFiles = [];  // Array of files ready for upload

// DOM
const messagesEl = document.getElementById('messages');
const welcomeEl = document.getElementById('welcome-screen');
const inputEl = document.getElementById('message-input');
const sendBtn = document.getElementById('btn-send');
const sessionListEl = document.getElementById('session-list');
const chatTitleEl = document.getElementById('chat-title');
const sessionStatusEl = document.getElementById('session-status');
const confirmBar = document.getElementById('confirmation-bar');
const confirmMsg = document.getElementById('confirmation-message');
const confirmCorrection = document.getElementById('confirmation-correction');
const sidebar = document.getElementById('sidebar');
const fileInput = document.getElementById('file-input');
const attachBtn = document.getElementById('btn-attach');
const uploadPreview = document.getElementById('upload-preview');
const uploadPreviewContent = document.getElementById('upload-preview-content');
const uploadPreviewCount = document.getElementById('upload-preview-count');
const uploadRemoveBtn = document.getElementById('btn-upload-remove');
const chatMain = document.querySelector('.chat-main');

// ─── Init ────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    loadSessions();
    setupEventListeners();
});

function setupEventListeners() {
    sendBtn.addEventListener('click', sendMessage);

    inputEl.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    inputEl.addEventListener('input', () => {
        sendBtn.disabled = !inputEl.value.trim() && pendingFiles.length === 0;
        autoResize(inputEl);
    });

    document.getElementById('btn-new-chat').addEventListener('click', createSession);
    document.getElementById('btn-delete-chat').addEventListener('click', deleteSession);
    document.getElementById('btn-approve').addEventListener('click', () => respondConfirmation(true));
    document.getElementById('btn-deny').addEventListener('click', () => respondConfirmation(false));

    document.getElementById('btn-sidebar-toggle').addEventListener('click', () => {
        sidebar.classList.toggle('open');
    });

    // Suggestion buttons
    document.querySelectorAll('.suggestion').forEach(btn => {
        btn.addEventListener('click', () => {
            inputEl.value = btn.dataset.message;
            sendBtn.disabled = false;
            sendMessage();
        });
    });

    // File attach button — supports multiple
    attachBtn.addEventListener('click', () => fileInput.click());

    // File input change — supports multiple
    fileInput.addEventListener('change', (e) => {
        for (const file of e.target.files) {
            handleFileSelect(file);
        }
        fileInput.value = '';
    });

    // Remove file preview
    uploadRemoveBtn.addEventListener('click', clearPendingFiles);

    // Drag and drop on chat area — supports multiple
    chatMain.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        chatMain.classList.add('drag-over');
    });

    chatMain.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        chatMain.classList.remove('drag-over');
    });

    chatMain.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        chatMain.classList.remove('drag-over');
        for (const file of e.dataTransfer.files) {
            handleFileSelect(file);
        }
    });
}

// ─── File Upload (multi-file) ────────────────────────────────────
function handleFileSelect(file) {
    const allowedTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/gif', 'image/tiff', 'application/pdf'];
    if (!allowedTypes.includes(file.type)) {
        appendMessage('assistant', 'Unsupported file type: ' + file.name + '. Please upload an image (JPEG, PNG, WebP) or PDF.');
        return;
    }
    if (file.size > 50 * 1024 * 1024) {
        appendMessage('assistant', file.name + ' is too large (max 50MB).');
        return;
    }
    pendingFiles.push(file);
    renderUploadPreview();
    sendBtn.disabled = false;
    inputEl.focus();
}

function renderUploadPreview() {
    if (pendingFiles.length === 0) {
        uploadPreview.style.display = 'none';
        return;
    }
    uploadPreview.style.display = 'flex';
    uploadPreviewContent.innerHTML = '';

    pendingFiles.forEach((file) => {
        const chip = document.createElement('div');
        chip.className = 'upload-file-chip';

        if (file.type.startsWith('image/')) {
            const img = document.createElement('img');
            const reader = new FileReader();
            reader.onload = (e) => { img.src = e.target.result; };
            reader.readAsDataURL(file);
            chip.appendChild(img);
        } else {
            const icon = document.createElement('div');
            icon.className = 'file-icon';
            icon.innerHTML = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8l-6-6z"/><path d="M14 2v6h6"/></svg>';
            chip.appendChild(icon);
        }

        const name = document.createElement('span');
        name.className = 'file-name';
        name.textContent = file.name;
        chip.appendChild(name);

        const size = document.createElement('span');
        size.className = 'file-size';
        size.textContent = formatFileSize(file.size);
        chip.appendChild(size);

        uploadPreviewContent.appendChild(chip);
    });

    const total = pendingFiles.length;
    const totalSize = pendingFiles.reduce((s, f) => s + f.size, 0);
    uploadPreviewCount.textContent = total + ' file' + (total > 1 ? 's' : '') + ' (' + formatFileSize(totalSize) + ')';
}

function clearPendingFiles() {
    pendingFiles = [];
    uploadPreview.style.display = 'none';
    uploadPreviewContent.innerHTML = '';
    uploadPreviewCount.textContent = '';
    sendBtn.disabled = !inputEl.value.trim();
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

async function uploadFiles(sessionId) {
    if (pendingFiles.length === 0) return null;

    const files = [...pendingFiles];
    clearPendingFiles();

    const results = [];
    for (const file of files) {
        const formData = new FormData();
        formData.append('file', file);
        if (sessionId) formData.append('session_id', sessionId);

        try {
            const resp = await fetch(API + '/api/upload', { method: 'POST', body: formData });
            if (!resp.ok) {
                const err = await resp.json();
                results.push({ filename: file.name, error: err.detail || 'Upload failed' });
            } else {
                const data = await resp.json();
                results.push({ filename: file.name, ...data });
            }
        } catch (e) {
            results.push({ filename: file.name, error: e.message });
        }
    }
    return results;
}

function autoResize(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 150) + 'px';
}

// ─── Sessions ────────────────────────────────────────────────────
async function loadSessions() {
    try {
        const res = await fetch(API + '/api/sessions');
        const data = await res.json();
        renderSessions(data.sessions || []);
    } catch (e) {
        console.error('Failed to load sessions:', e);
    }
}

function renderSessions(sessions) {
    if (sessions.length === 0) {
        sessionListEl.innerHTML = '<div class="empty-sidebar">No conversations yet.<br>Click + to start one.</div>';
        return;
    }

    sessionListEl.innerHTML = sessions.map(s => {
        const active = s.session_id === currentSessionId ? ' active' : '';
        const time = timeAgo(s.created_at);
        const initial = (s.title || 'New')[0].toUpperCase();
        return `
            <div class="session-item${active}" data-id="${s.session_id}" onclick="selectSession('${s.session_id}')">
                <div class="session-icon">${initial}</div>
                <div class="session-info">
                    <div class="session-title">${escapeHtml(s.title || 'New conversation')}</div>
                    <div class="session-time">${time} &middot; ${s.message_count} messages</div>
                </div>
            </div>`;
    }).join('');
}

async function createSession() {
    try {
        const res = await fetch(API + '/api/sessions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: 'New conversation' }),
        });
        const data = await res.json();
        currentSessionId = data.session_id;
        chatTitleEl.textContent = 'New conversation';
        sessionStatusEl.textContent = '';
        welcomeEl.style.display = 'flex';
        messagesEl.innerHTML = '';
        messagesEl.appendChild(welcomeEl);
        confirmBar.style.display = 'none';
        await loadSessions();
        inputEl.focus();
    } catch (e) {
        console.error('Failed to create session:', e);
    }
}

async function selectSession(sessionId) {
    currentSessionId = sessionId;
    welcomeEl.style.display = 'none';

    try {
        const res = await fetch(API + '/api/sessions/' + sessionId);
        const data = await res.json();
        chatTitleEl.textContent = data.title || 'Conversation';

        // Render messages
        messagesEl.innerHTML = '';
        (data.messages || []).forEach(m => appendMessage(m.role, m.content));
        scrollToBottom();
    } catch (e) {
        console.error('Failed to load session:', e);
    }

    await loadSessions();
    sidebar.classList.remove('open');
}

async function deleteSession() {
    if (!currentSessionId) return;
    if (!confirm('Delete this conversation?')) return;

    try {
        await fetch(API + '/api/sessions/' + currentSessionId, { method: 'DELETE' });
        currentSessionId = null;
        chatTitleEl.textContent = 'New conversation';
        sessionStatusEl.textContent = '';
        messagesEl.innerHTML = '';
        messagesEl.appendChild(welcomeEl);
        welcomeEl.style.display = 'flex';
        confirmBar.style.display = 'none';
        await loadSessions();
    } catch (e) {
        console.error('Failed to delete session:', e);
    }
}

// ─── Messages ────────────────────────────────────────────────────
function sendMessage() {
    const text = inputEl.value.trim();
    const hasFiles = pendingFiles.length > 0;
    if ((!text && !hasFiles) || isLoading) return;

    if (!currentSessionId) {
        // Auto-create session
        createSession().then(() => {
            inputEl.value = text;
            sendBtn.disabled = false;
            sendMessage();
        });
        return;
    }

    // Hide welcome
    welcomeEl.style.display = 'none';

    // Show user message (with file attachment info)
    let userMsg = text || '';
    if (hasFiles) {
        const names = pendingFiles.map(f => f.name).join(', ');
        const label = '\ud83d\udcce Attached: ' + names;
        userMsg = userMsg ? userMsg + '\n' + label : label;
    }
    appendMessage('user', userMsg);
    inputEl.value = '';
    inputEl.style.height = 'auto';
    sendBtn.disabled = true;

    // Upload files first if present
    if (hasFiles) {
        const count = pendingFiles.length;
        const uploadId = showUploadProgress('Uploading ' + count + ' file' + (count > 1 ? 's' : '') + '...');
        isLoading = true;

        uploadFiles(currentSessionId).then(uploadResults => {
            removeTyping(uploadId);

            if (uploadResults && uploadResults.length > 0) {
                const succeeded = uploadResults.filter(r => !r.error);
                const failed = uploadResults.filter(r => r.error);
                if (succeeded.length > 0) {
                    appendToolInfo(succeeded.length, 'file upload');
                    const names = succeeded.map(r => r.filename).join(', ');
                    const plural = succeeded.length > 1 ? 's' : '';
                    let msg = 'Upload successful: ' + succeeded.length + ' file' + plural;
                    if (names) msg += '\n' + names;
                    appendMessage('assistant', msg);
                }
                if (failed.length > 0) {
                    const errMsgs = failed.map(f => f.filename + ': ' + f.error).join('\n');
                    appendMessage('assistant', 'Failed to upload:\n' + errMsgs);
                }
            }

            if (text) {
                sendChatMessage(text);
            } else {
                isLoading = false;
                scrollToBottom();
                loadSessions();
            }
        }).catch(e => {
            removeTyping(uploadId);
            isLoading = false;
            appendMessage('assistant', 'Upload failed. Please try again.');
        });
    } else {
        sendChatMessage(text);
    }
}

function sendChatMessage(text) {
    const typingId = showTyping();
    isLoading = true;

    fetch(API + '/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: currentSessionId, message: text }),
    })
    .then(r => r.json())
    .then(data => {
        removeTyping(typingId);
        isLoading = false;

        if (data.tool_calls_count > 0) {
            appendToolInfo(data.tool_calls_count);
        }

        if (data.message) {
            appendMessage('assistant', data.message);
        }

        if (data.confirmation_required) {
            showConfirmation(data.confirmation_required);
        }

        if (data.metadata && data.metadata.session_title) {
            chatTitleEl.textContent = data.metadata.session_title;
        }

        scrollToBottom();
        loadSessions();
    })
    .catch(e => {
        removeTyping(typingId);
        isLoading = false;
        appendMessage('assistant', 'Sorry, something went wrong. Please try again.');
        console.error('Chat error:', e);
    });
}

function appendMessage(role, content) {
    const div = document.createElement('div');
    div.className = `message ${role}`;

    const avatar = role === 'user' ? 'You' : 'C';
    const formatted = formatContent(content);

    div.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-bubble">${formatted}</div>`;

    messagesEl.appendChild(div);
    scrollToBottom();
}

function appendToolInfo(count, label) {
    const div = document.createElement('div');
    div.className = 'message assistant';
    const displayLabel = label || 'tool call';
    div.innerHTML = `
        <div class="message-avatar">C</div>
        <div class="message-bubble">
            <div class="tool-call">
                <span class="tool-name">${count} ${displayLabel}${count > 1 ? 's' : ''}</span>
            </div>
        </div>`;
    messagesEl.appendChild(div);
}

function showUploadProgress(msg) {
    const id = 'upload-' + Date.now();
    const div = document.createElement('div');
    div.id = id;
    div.className = 'upload-progress';
    div.innerHTML = `<div class="spinner"></div><span>${msg}</span>`;
    messagesEl.appendChild(div);
    scrollToBottom();
    return id;
}

function showTyping() {
    const id = 'typing-' + Date.now();
    const div = document.createElement('div');
    div.id = id;
    div.className = 'message assistant';
    div.innerHTML = `
        <div class="message-avatar">C</div>
        <div class="message-bubble">
            <div class="typing-indicator">
                <span></span><span></span><span></span>
            </div>
        </div>`;
    messagesEl.appendChild(div);
    scrollToBottom();
    return id;
}

function removeTyping(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

// ─── Confirmation ────────────────────────────────────────────────
function showConfirmation(conf) {
    pendingConfirmation = conf;
    confirmMsg.textContent = conf.message || 'This action requires your approval.';
    if (confirmCorrection) confirmCorrection.value = '';
    confirmBar.style.display = 'flex';
}

async function respondConfirmation(approved) {
    if (!pendingConfirmation) return;

    confirmBar.style.display = 'none';
    const conf = pendingConfirmation;
    pendingConfirmation = null;

    // On deny, grab the correction input (if any) so the LLM retries
    // with the right field name on the next turn.
    const correction = !approved && confirmCorrection
        ? confirmCorrection.value.trim()
        : '';
    if (confirmCorrection) confirmCorrection.value = '';

    const typingId = showTyping();
    isLoading = true;

    try {
        const payload = {
            session_id: currentSessionId,
            request_id: conf.request_id,
            approved: approved,
        };
        if (correction) payload.correction = correction;
        const res = await fetch(API + '/api/confirm', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });
        const data = await res.json();
        removeTyping(typingId);
        isLoading = false;

        if (data.message) {
            appendMessage('assistant', data.message);
        }
        scrollToBottom();
    } catch (e) {
        removeTyping(typingId);
        isLoading = false;
        appendMessage('assistant', 'Failed to process your response. Please try again.');
    }
}

// ─── Formatting ──────────────────────────────────────────────────
function formatContent(text) {
    if (!text) return '';

    // Escape HTML
    let html = escapeHtml(text);

    // Code blocks
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre><code>$2</code></pre>');

    // Inline code
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    // Bold
    html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');

    // Line breaks
    html = html.replace(/\n/g, '<br>');

    return html;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function scrollToBottom() {
    requestAnimationFrame(() => {
        messagesEl.scrollTop = messagesEl.scrollHeight;
    });
}

function timeAgo(iso) {
    const seconds = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
    if (seconds < 60) return 'just now';
    if (seconds < 3600) return Math.floor(seconds / 60) + 'm ago';
    if (seconds < 86400) return Math.floor(seconds / 3600) + 'h ago';
    return Math.floor(seconds / 86400) + 'd ago';
}
