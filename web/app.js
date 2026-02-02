// Tongue Web App

const API_BASE = '';  // Same origin

// State
let currentUser = null;
let currentSentence = null;
let currentStory = null;
let storySentences = [];
let hintUsed = false;
let hintWords = [];  // Words that were given as hints
let isWordChallenge = false;  // Track if current task is a word challenge
let isVocabChallenge = false;  // Track if current task is a vocab challenge
let isVerbChallenge = false;  // Track if current task is a verb challenge
let isSynonymChallenge = false; // Track if current task is a synonym/antonym challenge
let isMultiVocab = false;     // Track if current task is a multi-word vocab challenge
let isReverseVocab = false;   // Track if current task is reverse direction
let multiVocabWords = [];     // Array of {word, translation} for multi-word challenges
let currentDirection = 'normal'; // 'normal' (ES→EN) or 'reverse' (EN→ES)
let currentLanguageCode = 'es';
let currentLanguageName = 'Spanish';
let availableLanguages = [];
let currentTenses = [];

// Cookie helpers
function setCookie(name, value, days = 365) {
    const expires = new Date(Date.now() + days * 864e5).toUTCString();
    document.cookie = `${name}=${encodeURIComponent(value)}; expires=${expires}; path=/`;
}

function getCookie(name) {
    const cookies = document.cookie.split('; ');
    for (const cookie of cookies) {
        const [cookieName, cookieValue] = cookie.split('=');
        if (cookieName === name) {
            return decodeURIComponent(cookieValue);
        }
    }
    return null;
}

function deleteCookie(name) {
    document.cookie = `${name}=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/`;
}

// DOM Elements
const elements = {
    // Start screen
    startScreen: document.getElementById('start-screen'),
    startForm: document.getElementById('start-form'),
    usernameInput: document.getElementById('username-input'),
    pinInput: document.getElementById('pin-input'),
    usernameError: document.getElementById('username-error'),
    startBtn: document.getElementById('start-btn'),

    // Game screen
    gameScreen: document.getElementById('game-screen'),
    userDisplay: document.getElementById('user-display'),
    newGameBtn: document.getElementById('new-game-btn'),

    statusBar: document.getElementById('status-bar'),
    levelDisplay: document.getElementById('level-display'),
    progressDisplay: document.getElementById('progress-display'),
    challengeDisplay: document.getElementById('challenge-display'),
    completedDisplay: document.getElementById('completed-display'),
    practiceTimeDisplay: document.getElementById('practice-time-display'),
    statusBtn: document.getElementById('status-btn'),

    storySection: document.getElementById('story-section'),
    storyLevel: document.getElementById('story-level'),
    storyContent: document.getElementById('story-content'),

    previousEval: document.getElementById('previous-eval'),
    prevChallengeType: document.getElementById('prev-challenge-type'),
    prevSentenceLabel: document.getElementById('prev-sentence-label'),
    prevSentence: document.getElementById('prev-sentence'),
    prevTranslation: document.getElementById('prev-translation'),
    prevCorrect: document.getElementById('prev-correct'),
    prevScore: document.getElementById('prev-score'),
    prevReason: document.getElementById('prev-reason'),
    prevReasonRow: document.getElementById('prev-reason-row'),

    loading: document.getElementById('loading'),
    currentTask: document.getElementById('current-task'),
    currentSentence: document.getElementById('current-sentence'),
    translationForm: document.getElementById('translation-form'),
    translationInput: document.getElementById('translation-input'),
    submitBtn: document.getElementById('submit-btn'),
    hintBtn: document.getElementById('hint-btn'),
    hintDisplay: document.getElementById('hint-display'),
    reviewNotice: document.getElementById('review-notice'),
    wordChallengeNotice: document.getElementById('word-challenge-notice'),
    wordType: document.getElementById('word-type'),
    vocabChallengeNotice: document.getElementById('vocab-challenge-notice'),
    vocabCategory: document.getElementById('vocab-category'),
    vocabDirection: document.getElementById('vocab-direction'),
    multiVocabInputs: document.getElementById('multi-vocab-inputs'),
    verbChallengeNotice: document.getElementById('verb-challenge-notice'),
    synonymChallengeNotice: document.getElementById('synonym-challenge-notice'),
    synonymChallengeLabel: document.getElementById('synonym-challenge-label'),
    synonymWordType: document.getElementById('synonym-word-type'),
    tenseSelectRow: document.getElementById('tense-select-row'),
    tenseSelect: document.getElementById('tense-select'),
    taskPrompt: document.getElementById('task-prompt'),

    validationResult: document.getElementById('validation-result'),
    resultScore: document.getElementById('result-score'),
    resultStudent: document.getElementById('result-student'),
    resultCorrect: document.getElementById('result-correct'),
    resultReason: document.getElementById('result-reason'),
    resultReasonRow: document.getElementById('result-reason-row'),
    levelChange: document.getElementById('level-change'),
    autoAdvance: document.getElementById('auto-advance'),

    menuBtn: document.getElementById('menu-btn'),
    menuDropdown: document.getElementById('menu-dropdown'),

    statusModal: document.getElementById('status-modal'),
    statusDetails: document.getElementById('status-details'),

    masteredBtn: document.getElementById('mastered-btn'),
    masteredModal: document.getElementById('mastered-modal'),
    masteredCount: document.getElementById('mastered-count'),
    masteredTbody: document.getElementById('mastered-tbody'),

    learningBtn: document.getElementById('learning-btn'),
    switchDirectionBtn: document.getElementById('switch-direction-btn'),
    switchLanguageBtn: document.getElementById('switch-language-btn'),
    languageSelect: document.getElementById('language-select'),
    languageModal: document.getElementById('language-modal'),
    languageOptions: document.getElementById('language-options'),
    downgradeBtn: document.getElementById('downgrade-btn'),
    learningModal: document.getElementById('learning-modal'),
    learningCount: document.getElementById('learning-count'),
    learningTbody: document.getElementById('learning-tbody'),

    // API Stats
    apiStats: document.getElementById('api-stats'),
    statsTotalCalls: document.getElementById('stats-total-calls'),
    statsTotalTokens: document.getElementById('stats-total-tokens'),
    statsAvgMs: document.getElementById('stats-avg-ms'),
    statsDetails: document.getElementById('stats-details'),
    statsTableBody: document.getElementById('stats-table-body')
};

// Error log state
const errorLog = [];

function addErrorToLog(endpoint, error, status = 0) {
    const entry = {
        time: new Date().toLocaleTimeString(),
        endpoint,
        error: typeof error === 'string' ? error : (error.message || String(error)),
        status
    };
    errorLog.unshift(entry);
    if (errorLog.length > 50) errorLog.pop();
    renderErrorLog();

    // Report to server (fire and forget)
    fetch(`${API_BASE}/api/error`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            error: entry.error,
            endpoint,
            user_id: currentUser || '',
            status,
            context: `page=${document.hidden ? 'hidden' : 'visible'}, online=${navigator.onLine}`
        })
    }).catch(() => {}); // ignore reporting failures
}

function renderErrorLog() {
    const panel = document.getElementById('error-log');
    const entries = document.getElementById('error-log-entries');
    if (!panel || !entries) return;

    if (errorLog.length === 0) {
        panel.classList.add('hidden');
        return;
    }

    panel.classList.remove('hidden');
    entries.innerHTML = errorLog.map(e =>
        `<div class="error-entry">
            <span class="error-time">${e.time}</span>
            <span class="error-endpoint">${e.endpoint}</span>
            ${e.status ? `<span class="error-status">${e.status}</span>` : ''}
            <span class="error-message">${e.error}</span>
        </div>`
    ).join('');
}

// API Functions
async function api(endpoint, options = {}) {
    let response;
    try {
        response = await fetch(`${API_BASE}${endpoint}`, {
            headers: { 'Content-Type': 'application/json' },
            ...options
        });
    } catch (fetchError) {
        const msg = navigator.onLine
            ? `Network error: ${fetchError.message}`
            : 'You appear to be offline';
        addErrorToLog(endpoint, msg, 0);
        throw new Error(msg);
    }
    if (!response.ok) {
        let detail = `API error: ${response.status}`;
        try {
            const data = await response.json();
            if (data.detail) {
                // FastAPI validation errors return detail as an array of objects
                if (Array.isArray(data.detail)) {
                    detail = data.detail.map(e => e.msg || JSON.stringify(e)).join('; ');
                } else if (typeof data.detail === 'string') {
                    detail = data.detail;
                } else {
                    detail = JSON.stringify(data.detail);
                }
            }
        } catch (e) {}
        addErrorToLog(endpoint, detail, response.status);
        throw new Error(detail);
    }
    return response.json();
}

// User management APIs
async function checkUserExists(userId) {
    return api(`/api/users/${encodeURIComponent(userId)}/exists`);
}

async function createUser(userId, pin, language = 'es') {
    return api(`/api/users/${encodeURIComponent(userId)}`, {
        method: 'POST',
        body: JSON.stringify({ pin, language })
    });
}

async function loginUser(userId, pin) {
    return api(`/api/users/${encodeURIComponent(userId)}/login`, {
        method: 'POST',
        body: JSON.stringify({ pin })
    });
}

// Game APIs (all include user_id)
async function getStatus() {
    return api(`/api/status?user_id=${encodeURIComponent(currentUser)}`);
}

async function getNextSentence() {
    return api(`/api/next?user_id=${encodeURIComponent(currentUser)}`);
}

async function submitTranslation(sentence, translation, selectedTense = null, translations = []) {
    return api('/api/translate', {
        method: 'POST',
        body: JSON.stringify({
            sentence,
            translation,
            user_id: currentUser,
            hint_used: hintUsed,
            hint_words: hintWords,
            selected_tense: selectedTense,
            translations
        })
    });
}

async function getHint(sentence) {
    return api('/api/hint', {
        method: 'POST',
        body: JSON.stringify({ sentence, user_id: currentUser, partial_translation: elements.translationInput.value.trim() })
    });
}

async function getVerbHint(sentence) {
    return api('/api/verb-hint', {
        method: 'POST',
        body: JSON.stringify({ sentence, user_id: currentUser })
    });
}

async function getMasteredWords() {
    return api(`/api/mastered-words?user_id=${encodeURIComponent(currentUser)}`);
}

async function getLearningWords() {
    return api(`/api/learning-words?user_id=${encodeURIComponent(currentUser)}`);
}

async function getLanguages() {
    return api('/api/languages');
}

async function switchLanguage(languageCode) {
    return api(`/api/switch-language?user_id=${encodeURIComponent(currentUser)}&language=${encodeURIComponent(languageCode)}`, {
        method: 'POST'
    });
}

async function getApiStats() {
    return api('/api/stats');
}

async function updateApiStats() {
    try {
        const stats = await getApiStats();
        if (stats.total && stats.total.calls > 0) {
            elements.apiStats.classList.remove('hidden');
            elements.statsTotalCalls.textContent = `${stats.total.calls} calls`;
            elements.statsTotalTokens.textContent = `${stats.total.total_tokens.toLocaleString()} tokens`;
            elements.statsAvgMs.textContent = `${stats.total.avg_ms}ms avg`;

            // Populate stats table
            const callTypes = ['story', 'validate', 'hint', 'word_translation', 'verb_analysis'];
            const callLabels = {
                'story': 'Story (Pro)',
                'validate': 'Validate (Flash)',
                'hint': 'Hint (Flash)',
                'word_translation': 'Word (Flash)',
                'verb_analysis': 'Verb (Flash)'
            };

            elements.statsTableBody.innerHTML = '';
            for (const callType of callTypes) {
                const s = stats[callType];
                if (s && s.calls > 0) {
                    const row = document.createElement('tr');
                    row.innerHTML = `
                        <td>${callLabels[callType]}</td>
                        <td>${s.calls}</td>
                        <td>${s.avg_tokens}</td>
                        <td>${s.avg_ms}</td>
                    `;
                    elements.statsTableBody.appendChild(row);
                }
            }
        }
    } catch (e) {
        console.error('Failed to fetch API stats:', e);
    }
}

// UI Functions
function updateStatusBar(status) {
    const dir = status.direction || 'normal';
    currentDirection = dir;
    currentLanguageCode = status.language_code || 'es';
    currentLanguageName = status.language || 'Spanish';

    // Update tenses if provided
    if (status.tenses && status.tenses.length > 0) {
        currentTenses = status.tenses;
        updateTenseOptions(status.tenses);
    }

    const langPrefix = currentLanguageCode !== 'es' ? currentLanguageCode.toUpperCase() + ' ' : '';
    const prefix = dir === 'reverse' ? 'R' : 'L';
    elements.levelDisplay.textContent = `${langPrefix}${prefix}${status.difficulty}`;
    elements.progressDisplay.textContent = `${status.good_score_count}/7`;
    elements.challengeDisplay.textContent = status.challenge_stats_display || '0/0';
    elements.completedDisplay.textContent = `${status.total_completed}`;
    elements.practiceTimeDisplay.textContent = status.practice_time_display || '0s';
    elements.switchDirectionBtn.textContent = dir === 'reverse' ? 'Switch to Normal' : 'Switch to Reverse';
}

function updateTenseOptions(tenses) {
    const select = elements.tenseSelect;
    // Keep the placeholder option
    select.innerHTML = '<option value="">-- Choose tense --</option>';
    const tenseLabels = {
        'present': 'Present',
        'preterite': 'Preterite (Simple Past)',
        'imperfect': 'Imperfect',
        'future': 'Future',
        'conditional': 'Conditional',
        'subjunctive': 'Subjunctive',
        'past': 'Past',
        'imperative': 'Imperative'
    };
    for (const tense of tenses) {
        const option = document.createElement('option');
        option.value = tense;
        option.textContent = tenseLabels[tense] || tense.charAt(0).toUpperCase() + tense.slice(1);
        select.appendChild(option);
    }
}

function splitIntoSentences(text) {
    return text.split(/(?<=[.!?])\s+/).filter(s => s.trim().length > 3);
}

function renderStory(story, difficulty, currentSentenceText) {
    elements.storySection.classList.remove('hidden');
    const prefix = currentDirection === 'reverse' ? 'R' : 'L';
    elements.storyLevel.textContent = `(${prefix}${difficulty})`;

    const sentences = splitIntoSentences(story);
    let foundCurrent = false;

    elements.storyContent.innerHTML = sentences.map(sentence => {
        const isCurrent = sentence.trim() === currentSentenceText.trim();
        if (isCurrent) foundCurrent = true;

        let className = '';
        if (isCurrent) {
            className = 'current';
        } else if (!foundCurrent) {
            className = 'completed';
        }

        return `<p class="${className}">${sentence}</p>`;
    }).join('');

    // Scroll to current sentence
    const currentEl = elements.storyContent.querySelector('.current');
    if (currentEl) {
        currentEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}

function showPreviousEvaluation(eval_data) {
    if (!eval_data) {
        elements.previousEval.classList.add('hidden');
        return;
    }

    elements.previousEval.classList.remove('hidden');

    // Show challenge type indicator if it was a challenge
    if (eval_data.challenge_type) {
        const challengeLabels = {
            'word': 'Word Challenge',
            'vocab': 'Vocabulary Quiz',
            'verb': 'Verb Challenge',
            'synonym': 'Synonym/Antonym Challenge'
        };
        let label = challengeLabels[eval_data.challenge_type] || '';
        if (eval_data.challenge_direction) {
            label += ` (${eval_data.challenge_direction})`;
        }
        elements.prevChallengeType.textContent = label;
        elements.prevChallengeType.classList.remove('hidden');
        elements.prevSentenceLabel.textContent = 'Word:';
    } else {
        elements.prevChallengeType.classList.add('hidden');
        elements.prevSentenceLabel.textContent = 'Sentence:';
    }

    elements.prevSentence.textContent = eval_data.sentence;
    elements.prevTranslation.textContent = eval_data.translation;
    elements.prevCorrect.textContent = eval_data.correct_translation;
    elements.prevScore.textContent = eval_data.score;

    // Apply score styling
    elements.prevScore.className = '';
    if (eval_data.score >= 90) {
        elements.prevScore.classList.add('score-excellent');
    } else if (eval_data.score >= 70) {
        elements.prevScore.classList.add('score-good');
    } else {
        elements.prevScore.classList.add('score-poor');
    }

    if (eval_data.evaluation && eval_data.score < 100) {
        elements.prevReasonRow.classList.remove('hidden');
        elements.prevReason.textContent = eval_data.evaluation;
    } else {
        elements.prevReasonRow.classList.add('hidden');
    }
}

function showCurrentTask(sentence, isReview = false, isWordChal = false, challengeWord = null, isVocabChal = false, vocabChallenge = null, isVerbChal = false, verbChallenge = null, isSynChal = false, synChallenge = null) {
    elements.loading.classList.add('hidden');
    elements.currentTask.classList.remove('hidden');
    elements.validationResult.classList.add('hidden');
    elements.translationInput.value = '';
    elements.hintDisplay.classList.add('hidden');
    elements.submitBtn.disabled = false;
    hintUsed = false;  // Reset hint usage for new sentence
    hintWords = [];    // Reset hint words for new sentence

    // Reset multi-word state
    isMultiVocab = false;
    isReverseVocab = false;
    multiVocabWords = [];
    elements.multiVocabInputs.classList.add('hidden');
    elements.translationInput.classList.remove('hidden');
    elements.currentSentence.classList.remove('hidden');
    elements.vocabDirection.textContent = '';

    // Show review notice if this is a review sentence
    if (isReview) {
        elements.reviewNotice.classList.remove('hidden');
    } else {
        elements.reviewNotice.classList.add('hidden');
    }

    // Reset challenge notices and tense dropdown
    elements.wordChallengeNotice.classList.add('hidden');
    elements.vocabChallengeNotice.classList.add('hidden');
    elements.verbChallengeNotice.classList.add('hidden');
    elements.synonymChallengeNotice.classList.add('hidden');
    elements.tenseSelectRow.classList.add('hidden');
    elements.tenseSelect.value = '';
    elements.currentSentence.classList.remove('word-challenge');

    // Show appropriate challenge notice
    if (isSynChal && synChallenge) {
        const isSynonym = synChallenge.challenge_type === 'SYN';
        const typeLabel = isSynonym ? 'Synonym' : 'Antonym';
        elements.synonymChallengeNotice.classList.remove('hidden');
        elements.synonymChallengeNotice.className = `synonym-challenge-notice ${isSynonym ? 'synonym-type' : 'antonym-type'}`;
        elements.synonymChallengeLabel.textContent = `${typeLabel} Challenge!`;
        elements.synonymWordType.textContent = synChallenge.type || '';
        elements.taskPrompt.textContent = `Type a ${typeLabel.toLowerCase()} for:`;
        elements.currentSentence.textContent = sentence;
        elements.currentSentence.classList.add('word-challenge');
        elements.translationInput.placeholder = `Enter a ${typeLabel.toLowerCase()} in ${currentLanguageName}...`;
        elements.hintBtn.classList.add('hidden');
        elements.translationInput.focus();
    } else if (isVerbChal && verbChallenge) {
        elements.verbChallengeNotice.classList.remove('hidden');
        elements.tenseSelectRow.classList.remove('hidden');
        elements.taskPrompt.textContent = currentDirection === 'reverse'
            ? `Type the ${currentLanguageName} verb form:` : 'Translate this verb:';
        elements.currentSentence.textContent = sentence;
        elements.currentSentence.classList.add('word-challenge');
        elements.translationInput.placeholder = currentDirection === 'reverse'
            ? `Enter the ${currentLanguageName} verb form...`
            : 'Enter your English translation...';
        elements.translationInput.focus();
    } else if (isVocabChal && vocabChallenge && vocabChallenge.is_multi) {
        // Multi-word vocab challenge
        isMultiVocab = true;
        isReverseVocab = vocabChallenge.is_reverse || false;
        multiVocabWords = vocabChallenge.words;

        elements.vocabChallengeNotice.classList.remove('hidden');
        elements.vocabCategory.textContent = vocabChallenge.category_name;
        const langCode = currentLanguageCode.toUpperCase();
        elements.vocabDirection.textContent = isReverseVocab ? `EN \u2192 ${langCode}` : `${langCode} \u2192 EN`;
        elements.taskPrompt.textContent = isReverseVocab
            ? `Type the ${currentLanguageName} word for each:`
            : 'Translate each word:';
        elements.currentSentence.classList.add('hidden');
        elements.translationInput.classList.add('hidden');
        elements.hintBtn.classList.add('hidden');

        // Show multi-word inputs
        elements.multiVocabInputs.classList.remove('hidden');
        for (let i = 0; i < 4; i++) {
            const wordEl = document.getElementById(`multi-word-${i}`);
            const inputEl = document.getElementById(`multi-input-${i}`);
            const resultEl = document.getElementById(`multi-result-${i}`);

            if (i < multiVocabWords.length) {
                const item = multiVocabWords[i];
                // Reverse: show English, expect Spanish
                // Forward: show Spanish, expect English
                wordEl.textContent = isReverseVocab ? item.translation : item.word;
                inputEl.value = '';
                inputEl.placeholder = isReverseVocab ? `Translate to ${currentLanguageName}` : 'Translate to English';
                resultEl.textContent = '';
            }
        }
        // Focus first input
        document.getElementById('multi-input-0').focus();
    } else if (isVocabChal && vocabChallenge) {
        // Single-word vocab challenge
        isReverseVocab = vocabChallenge.is_reverse || false;
        elements.vocabChallengeNotice.classList.remove('hidden');
        elements.vocabCategory.textContent = vocabChallenge.category_name;
        const langCode2 = currentLanguageCode.toUpperCase();
        elements.vocabDirection.textContent = isReverseVocab ? `EN \u2192 ${langCode2}` : `${langCode2} \u2192 EN`;
        elements.taskPrompt.textContent = isReverseVocab
            ? `Type the ${currentLanguageName} word for:`
            : 'Translate this word:';
        elements.currentSentence.textContent = sentence;
        elements.currentSentence.classList.add('word-challenge');
        elements.translationInput.placeholder = isReverseVocab
            ? `Enter your ${currentLanguageName} translation...`
            : 'Enter your English translation...';
        elements.hintBtn.classList.add('hidden');
        elements.translationInput.focus();
    } else if (isWordChal && challengeWord) {
        elements.wordChallengeNotice.classList.remove('hidden');
        elements.wordType.textContent = challengeWord.type;
        elements.taskPrompt.textContent = currentDirection === 'reverse'
            ? `Type the ${currentLanguageName} word for:`
            : 'Translate this word:';
        elements.currentSentence.textContent = sentence;
        elements.currentSentence.classList.add('word-challenge');
        elements.translationInput.placeholder = currentDirection === 'reverse'
            ? `Enter your ${currentLanguageName} translation...`
            : 'Enter your English translation...';
        elements.hintBtn.classList.add('hidden');
        elements.translationInput.focus();
    } else {
        if (currentDirection === 'reverse') {
            elements.taskPrompt.textContent = `Translate to ${currentLanguageName}:`;
            elements.translationInput.placeholder = `Enter your ${currentLanguageName} translation...`;
        } else {
            elements.taskPrompt.textContent = 'Translate this sentence:';
            elements.translationInput.placeholder = 'Enter your English translation...';
        }
        elements.currentSentence.textContent = sentence;
        elements.hintBtn.classList.remove('hidden');
        elements.translationInput.focus();
    }
}

function showValidationResult(result, studentTranslation) {
    elements.currentTask.classList.add('hidden');
    elements.validationResult.classList.remove('hidden');

    // Score styling
    elements.resultScore.textContent = result.score;
    elements.resultScore.className = 'score-value';
    if (result.score >= 90) {
        elements.resultScore.classList.add('excellent');
    } else if (result.score >= 70) {
        elements.resultScore.classList.add('good');
    } else {
        elements.resultScore.classList.add('poor');
    }

    elements.resultStudent.textContent = studentTranslation;
    elements.resultCorrect.textContent = result.correct_translation;

    if (result.evaluation && result.score < 100) {
        elements.resultReasonRow.classList.remove('hidden');
        elements.resultReason.textContent = result.evaluation;
    } else {
        elements.resultReasonRow.classList.add('hidden');
    }

    // Per-word results for multi-word challenges
    const existingGrid = document.querySelector('.multi-result-grid');
    if (existingGrid) existingGrid.remove();

    if (result.word_results && result.word_results.length > 0) {
        const grid = document.createElement('div');
        grid.className = 'multi-result-grid';

        result.word_results.forEach(wr => {
            const item = document.createElement('div');
            item.className = `multi-result-item ${wr.is_correct ? 'correct' : 'incorrect'}`;

            const icon = wr.is_correct ? '\u2713' : '\u2717';
            const iconColor = wr.is_correct ? '#2ecc71' : '#e74c3c';

            let html = `<span class="multi-result-icon" style="color:${iconColor}">${icon}</span>`;
            html += `<span class="multi-result-word">${wr.word}</span>`;
            html += `<span class="multi-result-answer">${wr.student_answer || '(empty)'}</span>`;
            if (!wr.is_correct) {
                html += `<span class="multi-result-correct">\u2192 ${wr.correct_answer}</span>`;
            }

            item.innerHTML = html;
            grid.appendChild(item);
        });

        // Insert after the score display
        const resultContent = elements.validationResult.querySelector('.result-content');
        if (resultContent) {
            resultContent.appendChild(grid);
        }
    }

    // Level change
    if (result.level_changed) {
        elements.levelChange.classList.remove('hidden');
        if (result.change_type === 'advanced') {
            elements.levelChange.className = 'level-change advanced';
            elements.levelChange.textContent = `Level Up! Now at Level ${result.new_level}`;
        } else {
            elements.levelChange.className = 'level-change demoted';
            elements.levelChange.textContent = `Level Down. Now at Level ${result.new_level}`;
        }
    } else {
        elements.levelChange.classList.add('hidden');
    }

    // Auto-advance after delay (longer for multi-word to read results)
    const delay = result.word_results ? 5000 : 3000;
    setTimeout(() => {
        loadNextSentence();
    }, delay);
}

function formatPracticeTime(seconds) {
    seconds = Math.floor(seconds);
    if (seconds < 60) return `${seconds}s`;
    if (seconds < 600) return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    return `${hours}h ${minutes}m`;
}

async function showStatusModal() {
    try {
        const status = await getStatus();

        const avgScore = status.level_scores.length > 0
            ? (status.level_scores.reduce((a, b) => a + b, 0) / status.level_scores.length).toFixed(1)
            : 'N/A';

        // Build practice time breakdown HTML
        let practiceBreakdownHtml = '';
        if (status.practice_times && Object.keys(status.practice_times).length > 0) {
            const entries = Object.entries(status.practice_times)
                .filter(([, secs]) => secs > 0)
                .sort((a, b) => b[1] - a[1]);
            if (entries.length > 0) {
                practiceBreakdownHtml = '<p><strong>Practice Time Breakdown:</strong></p><ul style="margin:4px 0 8px 20px">';
                for (const [key, secs] of entries) {
                    const [lang, dir] = key.split(':');
                    const label = `${lang} ${dir}`;
                    const display = formatPracticeTime(secs);
                    practiceBreakdownHtml += `<li>${label}: ${display}</li>`;
                }
                practiceBreakdownHtml += '</ul>';
            }
        }

        elements.statusDetails.innerHTML = `
            <p><strong>Language:</strong> ${status.language}</p>
            <p><strong>Current Level:</strong> ${status.difficulty}/${status.max_difficulty}</p>
            <p><strong>Total Completed:</strong> ${status.total_completed}</p>
            <p><strong>Story Progress:</strong> ${status.story_sentences_remaining} sentences remaining</p>
            <p><strong>Recent Average:</strong> ${avgScore}</p>
            <p><strong>Credits (≥80 score):</strong> ${status.good_score_count}/7 needed to advance</p>
            <p><strong>Poor Scores (<50):</strong> ${status.poor_score_count}/4 triggers demotion</p>
            <p><strong>Total Practice Time:</strong> ${status.practice_time_display}</p>
            ${practiceBreakdownHtml}
        `;

        elements.statusModal.classList.remove('hidden');
    } catch (error) {
        console.error('Error fetching status:', error);
        alert('Failed to load status');
    }
}

function hideModal(modalId) {
    document.getElementById(modalId).classList.add('hidden');
}

function hideStatusModal() {
    elements.statusModal.classList.add('hidden');
}

function renderWordsTable(tbody, words) {
    if (words.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="empty-message">No words yet</td></tr>';
        return;
    }

    tbody.innerHTML = words.map(w => `
        <tr>
            <td class="word-cell">${w.word}</td>
            <td>${w.type}</td>
            <td>${w.translation}</td>
            <td class="count-correct">${w.correct_count}</td>
            <td class="count-incorrect">${w.incorrect_count}</td>
            <td class="success-rate ${w.success_rate >= 80 ? 'rate-good' : w.success_rate >= 50 ? 'rate-medium' : 'rate-poor'}">${w.success_rate}%</td>
        </tr>
    `).join('');
}

async function showMasteredModal() {
    try {
        const data = await getMasteredWords();
        elements.masteredCount.textContent = `(${data.total})`;
        renderWordsTable(elements.masteredTbody, data.words);
        elements.masteredModal.classList.remove('hidden');
    } catch (error) {
        console.error('Error fetching mastered words:', error);
        alert('Failed to load mastered words');
    }
}

async function showLearningModal() {
    try {
        const data = await getLearningWords();
        elements.learningCount.textContent = `(${data.total})`;
        renderWordsTable(elements.learningTbody, data.words);
        elements.learningModal.classList.remove('hidden');
    } catch (error) {
        console.error('Error fetching learning words:', error);
        alert('Failed to load learning words');
    }
}

async function loadNextSentence() {
    elements.loading.classList.remove('hidden');
    elements.loading.innerHTML = '<p>Loading story — this can take up to 30 seconds...</p>';
    elements.currentTask.classList.add('hidden');
    elements.validationResult.classList.add('hidden');

    try {
        const data = await getNextSentence();

        currentSentence = data.sentence;
        currentStory = data.story;
        isWordChallenge = data.is_word_challenge;
        isVocabChallenge = data.is_vocab_challenge;
        isVerbChallenge = data.is_verb_challenge;
        isSynonymChallenge = data.is_synonym_challenge || false;
        currentDirection = data.direction || 'normal';

        // Update status bar
        const status = await getStatus();
        updateStatusBar(status);

        // Render story (hide for challenges or when no story)
        if (data.is_word_challenge || data.is_vocab_challenge || data.is_verb_challenge || data.is_synonym_challenge || !data.story) {
            elements.storySection.classList.add('hidden');
        } else {
            renderStory(data.story, data.difficulty, data.sentence);
        }

        // Show previous evaluation if any
        if (data.has_previous_evaluation && data.previous_evaluation) {
            showPreviousEvaluation(data.previous_evaluation);
        } else {
            elements.previousEval.classList.add('hidden');
        }

        // Show current task
        showCurrentTask(data.sentence, data.is_review, data.is_word_challenge, data.challenge_word, data.is_vocab_challenge, data.vocab_challenge, data.is_verb_challenge, data.verb_challenge, data.is_synonym_challenge, data.synonym_challenge);

        // Update API stats
        updateApiStats();

    } catch (error) {
        console.error('Error loading sentence:', error);
        elements.loading.innerHTML = `<p>Error loading: ${error.message || error}</p><p><button onclick="loadNextSentence()">Retry</button></p>`;
    }
}

async function handleSubmit(e) {
    e.preventDefault();

    let translation = '';
    let translations = [];

    if (isMultiVocab) {
        // Collect all 4 inputs
        for (let i = 0; i < 4; i++) {
            const input = document.getElementById(`multi-input-${i}`);
            translations.push(input ? input.value.trim() : '');
        }
        // Check at least one is filled
        if (translations.every(t => !t)) {
            document.getElementById('multi-input-0').focus();
            return;
        }
        translation = translations.join(', ');
    } else {
        translation = elements.translationInput.value.trim();
        if (!translation) {
            elements.translationInput.focus();
            return;
        }
    }

    // For verb challenges, require tense selection
    let selectedTense = null;
    if (isVerbChallenge) {
        selectedTense = elements.tenseSelect.value;
        if (!selectedTense) {
            alert('Please select the tense for this verb.');
            elements.tenseSelect.focus();
            return;
        }
    }

    elements.submitBtn.disabled = true;
    elements.submitBtn.textContent = 'Validating...';

    try {
        const result = await submitTranslation(currentSentence, translation, selectedTense, translations);

        // Update status bar
        const status = await getStatus();
        updateStatusBar(status);

        // Show result
        showValidationResult(result, translation);

        // Update API stats
        updateApiStats();

    } catch (error) {
        console.error('Error submitting:', error);
        showSubmitError(error.message || 'Unknown error');
        elements.submitBtn.disabled = false;
    } finally {
        elements.submitBtn.textContent = 'Submit';
    }
}

function showSubmitError(message) {
    // Show error in the validation result area so it's visible on mobile
    elements.validationResult.style.display = 'block';
    elements.validationResult.className = 'result-box error';
    elements.validationResult.innerHTML = `
        <h3>Submission Failed</h3>
        <p><strong>Error:</strong> ${message}</p>
        <p style="margin-top: 10px; font-size: 0.9em; color: #666;">
            Try submitting again. If the problem persists, check your connection.
        </p>
    `;
}

async function handleHint() {
    elements.hintBtn.disabled = true;
    elements.hintBtn.textContent = 'Loading...';

    try {
        if (isVerbChallenge) {
            // Verb challenge: show conjugation rules
            const verbHint = await getVerbHint(currentSentence);
            let hintHtml = `<strong>${verbHint.tense.charAt(0).toUpperCase() + verbHint.tense.slice(1)} tense conjugation rules:</strong><br>`;
            hintHtml += verbHint.rules.replace(/\n/g, '<br>');
            elements.hintDisplay.innerHTML = hintHtml;
            elements.hintDisplay.classList.remove('hidden');
            hintUsed = true;
            updateApiStats();
            return;
        }

        const hint = await getHint(currentSentence);

        // Filter out entries where the AI returned null values inside the array
        const validHint = (h) => Array.isArray(h) && h[0] && h[0] !== 'null';

        let hintHtml = '<strong>Hint:</strong><br>';
        if (validHint(hint.noun)) {
            hintHtml += `Noun: <em>${hint.noun[0]}</em> = ${hint.noun[1]}<br>`;
            hintWords.push(hint.noun[0]);  // Track hint word
        }
        if (validHint(hint.verb)) {
            hintHtml += `Verb: <em>${hint.verb[0]}</em> = ${hint.verb[1]}<br>`;
            hintWords.push(hint.verb[0]);  // Track hint word
        }
        if (validHint(hint.adjective)) {
            hintHtml += `Adjective: <em>${hint.adjective[0]}</em> = ${hint.adjective[1]}<br>`;
            hintWords.push(hint.adjective[0]);  // Track hint word
        }
        if (validHint(hint.adverb)) {
            hintHtml += `Adverb: <em>${hint.adverb[0]}</em> = ${hint.adverb[1]}`;
            hintWords.push(hint.adverb[0]);  // Track hint word
        }
        if (!validHint(hint.noun) && !validHint(hint.verb) && !validHint(hint.adjective) && !validHint(hint.adverb)) {
            hintHtml = 'No hint available';
        }

        elements.hintDisplay.innerHTML = hintHtml;
        elements.hintDisplay.classList.remove('hidden');
        hintUsed = true;  // Mark that hint was used

        // Update API stats
        updateApiStats();

    } catch (error) {
        console.error('Error getting hint:', error);
        alert('Failed to get hint');
    } finally {
        elements.hintBtn.disabled = false;
        elements.hintBtn.textContent = 'Hint';
    }
}

// Start Screen Functions
function showStartScreen() {
    elements.startScreen.classList.remove('hidden');
    elements.gameScreen.classList.add('hidden');
    elements.usernameInput.value = '';
    elements.pinInput.value = '';
    elements.usernameError.classList.add('hidden');
    elements.usernameInput.focus();
    // Populate language selector for new users
    populateLoginLanguageSelect();
}

async function populateLoginLanguageSelect() {
    try {
        const data = await getLanguages();
        availableLanguages = data.languages || [];
        if (elements.languageSelect && availableLanguages.length > 0) {
            elements.languageSelect.innerHTML = '';
            for (const lang of availableLanguages) {
                const option = document.createElement('option');
                option.value = lang.code;
                option.textContent = lang.english_name;
                if (lang.code === 'es') option.selected = true;
                elements.languageSelect.appendChild(option);
            }
        }
    } catch (e) {
        console.error('Failed to load languages for selector:', e);
    }
}

function showGameScreen() {
    elements.startScreen.classList.add('hidden');
    elements.gameScreen.classList.remove('hidden');
    elements.userDisplay.textContent = currentUser;
    loadNextSentence();
}

async function handleStartForm(e) {
    e.preventDefault();

    const username = elements.usernameInput.value.trim().toLowerCase();
    const pin = elements.pinInput.value.trim();

    if (!username) {
        showUsernameError('Please enter a name');
        return;
    }

    // Validate username (alphanumeric and spaces only)
    if (!/^[a-zA-Z0-9 ]+$/.test(username)) {
        showUsernameError('Name can only contain letters, numbers, and spaces');
        return;
    }

    // Validate PIN (exactly 4 digits)
    if (!pin || pin.length !== 4 || !/^\d{4}$/.test(pin)) {
        showUsernameError('PIN must be exactly 4 digits');
        return;
    }

    elements.startBtn.disabled = true;
    elements.startBtn.textContent = 'Starting...';

    try {
        // Check if user exists
        const { exists } = await checkUserExists(username);

        if (exists) {
            // User exists - try to login with PIN
            const result = await loginUser(username, pin);
            if (result.success) {
                currentUser = username;
                setCookie('tongue_user', username);
                showGameScreen();
            } else {
                showUsernameError(result.error || 'Invalid PIN');
            }
        } else {
            // New user - create them with selected language
            const selectedLanguage = elements.languageSelect ? elements.languageSelect.value : 'es';
            const result = await createUser(username, pin, selectedLanguage);
            if (result.success) {
                currentUser = username;
                setCookie('tongue_user', username);
                showGameScreen();
            } else {
                showUsernameError(result.error || 'Failed to create user');
            }
        }
    } catch (error) {
        console.error('Error starting game:', error);
        showUsernameError('Failed to start. Please try again.');
    } finally {
        elements.startBtn.disabled = false;
        elements.startBtn.textContent = 'Start Practice';
    }
}

function showUsernameError(message) {
    elements.usernameError.textContent = message;
    elements.usernameError.classList.remove('hidden');
}

async function handleDowngrade() {
    if (!confirm('Go back to the previous level? This will reset your current score progress.')) return;
    try {
        const data = await api(`/api/downgrade?user_id=${currentUser}`, { method: 'POST' });
        if (data.success) {
            const status = await api(`/api/status?user_id=${currentUser}`);
            updateStatusBar(status);
            loadNextSentence();
        } else {
            alert(data.error || 'Cannot downgrade further.');
        }
    } catch (e) {
        alert('Failed to downgrade level.');
    }
}

async function handleSwitchDirection() {
    try {
        const data = await api(`/api/switch-direction?user_id=${encodeURIComponent(currentUser)}`, { method: 'POST' });
        if (data.success) {
            currentDirection = data.direction;
            const status = await getStatus();
            updateStatusBar(status);
            loadNextSentence();
        }
    } catch (e) {
        alert('Failed to switch direction.');
    }
}

async function handleSwitchLanguage() {
    // Populate language options
    if (availableLanguages.length === 0) {
        try {
            const data = await getLanguages();
            availableLanguages = data.languages || [];
        } catch (e) {
            alert('Failed to load languages.');
            return;
        }
    }

    elements.languageOptions.innerHTML = '';
    for (const lang of availableLanguages) {
        const btn = document.createElement('button');
        btn.className = 'language-option-btn';
        if (lang.code === currentLanguageCode) {
            btn.classList.add('active');
        }
        btn.textContent = `${lang.name} (${lang.english_name})`;
        btn.addEventListener('click', async () => {
            if (lang.code === currentLanguageCode) {
                elements.languageModal.classList.add('hidden');
                return;
            }
            try {
                const result = await switchLanguage(lang.code);
                if (result.success) {
                    elements.languageModal.classList.add('hidden');
                    const status = await getStatus();
                    updateStatusBar(status);
                    loadNextSentence();
                } else {
                    alert(result.error || 'Failed to switch language.');
                }
            } catch (e) {
                alert('Failed to switch language.');
            }
        });
        elements.languageOptions.appendChild(btn);
    }

    elements.languageModal.classList.remove('hidden');
}

function handleNewGame() {
    if (confirm('Start a new game? This will take you back to the name selection screen.')) {
        deleteCookie('tongue_user');
        currentUser = null;
        showStartScreen();
    }
}

// Menu dropdown toggle
function toggleMenu() {
    elements.menuDropdown.classList.toggle('hidden');
}

function closeMenu() {
    elements.menuDropdown.classList.add('hidden');
}

// Event Listeners
elements.startForm.addEventListener('submit', handleStartForm);
elements.translationForm.addEventListener('submit', handleSubmit);
elements.hintBtn.addEventListener('click', handleHint);
elements.menuBtn.addEventListener('click', toggleMenu);
elements.statusBtn.addEventListener('click', () => { closeMenu(); showStatusModal(); });
elements.masteredBtn.addEventListener('click', () => { closeMenu(); showMasteredModal(); });
elements.learningBtn.addEventListener('click', () => { closeMenu(); showLearningModal(); });
elements.switchDirectionBtn.addEventListener('click', () => { closeMenu(); handleSwitchDirection(); });
elements.switchLanguageBtn.addEventListener('click', () => { closeMenu(); handleSwitchLanguage(); });
elements.downgradeBtn.addEventListener('click', () => { closeMenu(); handleDowngrade(); });
elements.newGameBtn.addEventListener('click', () => { closeMenu(); handleNewGame(); });

// Enter-key navigation for multi-word inputs
for (let i = 0; i < 4; i++) {
    const input = document.getElementById(`multi-input-${i}`);
    if (input) {
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                if (i < 3) {
                    // Move to next input
                    document.getElementById(`multi-input-${i + 1}`).focus();
                } else {
                    // Last input - submit form
                    elements.translationForm.dispatchEvent(new Event('submit', { cancelable: true }));
                }
            }
        });
    }
}

// Close menu when clicking outside
document.addEventListener('click', (e) => {
    if (!elements.menuBtn.contains(e.target) && !elements.menuDropdown.contains(e.target)) {
        closeMenu();
    }
});

// Toggle API stats details on click
elements.apiStats.addEventListener('click', () => {
    elements.statsDetails.classList.toggle('hidden');
});

// Close buttons for all modals
document.querySelectorAll('.close-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        const modalId = btn.dataset.modal;
        if (modalId) {
            hideModal(modalId);
        }
    });
});

// Click outside modal to close
[elements.statusModal, elements.masteredModal, elements.learningModal, elements.languageModal].forEach(modal => {
    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.classList.add('hidden');
        }
    });
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        elements.statusModal.classList.add('hidden');
        elements.masteredModal.classList.add('hidden');
        elements.learningModal.classList.add('hidden');
        elements.languageModal.classList.add('hidden');
    }
});

// Initialize - check for existing user cookie
function init() {
    const savedUser = (getCookie('tongue_user') || '').toLowerCase() || null;
    if (savedUser) {
        // Verify user still exists
        checkUserExists(savedUser).then(({ exists }) => {
            if (exists) {
                currentUser = savedUser;
                showGameScreen();
            } else {
                // User was deleted, show start screen
                deleteCookie('tongue_user');
                showStartScreen();
            }
        }).catch(() => {
            showStartScreen();
        });
    } else {
        showStartScreen();
    }
}

init();
