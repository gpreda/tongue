// Tongue Web App

const API_BASE = '';  // Same origin

// State
let currentSentence = null;
let currentStory = null;
let storySentences = [];

// DOM Elements
const elements = {
    statusBar: document.getElementById('status-bar'),
    levelDisplay: document.getElementById('level-display'),
    progressDisplay: document.getElementById('progress-display'),
    completedDisplay: document.getElementById('completed-display'),
    statusBtn: document.getElementById('status-btn'),

    storySection: document.getElementById('story-section'),
    storyLevel: document.getElementById('story-level'),
    storyContent: document.getElementById('story-content'),

    previousEval: document.getElementById('previous-eval'),
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

    validationResult: document.getElementById('validation-result'),
    resultScore: document.getElementById('result-score'),
    resultStudent: document.getElementById('result-student'),
    resultCorrect: document.getElementById('result-correct'),
    resultReason: document.getElementById('result-reason'),
    resultReasonRow: document.getElementById('result-reason-row'),
    levelChange: document.getElementById('level-change'),
    nextBtn: document.getElementById('next-btn'),

    statusModal: document.getElementById('status-modal'),
    statusDetails: document.getElementById('status-details'),
    closeBtn: document.querySelector('.close-btn')
};

// API Functions
async function api(endpoint, options = {}) {
    const response = await fetch(`${API_BASE}${endpoint}`, {
        headers: { 'Content-Type': 'application/json' },
        ...options
    });
    if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
    }
    return response.json();
}

async function getStatus() {
    return api('/api/status');
}

async function getNextSentence() {
    return api('/api/next');
}

async function submitTranslation(sentence, translation) {
    return api('/api/translate', {
        method: 'POST',
        body: JSON.stringify({ sentence, translation })
    });
}

async function getHint(sentence) {
    return api('/api/hint', {
        method: 'POST',
        body: JSON.stringify({ sentence })
    });
}

async function getMissedWords() {
    return api('/api/missed-words?limit=15');
}

async function getMasteredWords() {
    return api('/api/mastered-words?limit=20');
}

// UI Functions
function updateStatusBar(status) {
    elements.levelDisplay.textContent = `Level ${status.difficulty}/${status.max_difficulty}`;
    elements.progressDisplay.textContent = `Progress: ${status.good_score_count}/${4} good scores`;
    elements.completedDisplay.textContent = `Completed: ${status.total_completed}`;
}

function splitIntoSentences(text) {
    return text.split(/(?<=[.!?])\s+/).filter(s => s.trim().length > 3);
}

function renderStory(story, difficulty, currentSentenceText) {
    elements.storySection.classList.remove('hidden');
    elements.storyLevel.textContent = `(Level ${difficulty})`;

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
    elements.prevSentence.textContent = eval_data.sentence;
    elements.prevTranslation.textContent = eval_data.translation;
    elements.prevCorrect.textContent = eval_data.correct_translation;
    elements.prevScore.textContent = eval_data.score;

    if (eval_data.evaluation && eval_data.score < 100) {
        elements.prevReasonRow.classList.remove('hidden');
        elements.prevReason.textContent = eval_data.evaluation;
    } else {
        elements.prevReasonRow.classList.add('hidden');
    }
}

function showCurrentTask(sentence) {
    elements.loading.classList.add('hidden');
    elements.currentTask.classList.remove('hidden');
    elements.validationResult.classList.add('hidden');
    elements.currentSentence.textContent = sentence;
    elements.translationInput.value = '';
    elements.translationInput.focus();
    elements.hintDisplay.classList.add('hidden');
    elements.submitBtn.disabled = false;
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
}

async function showStatusModal() {
    try {
        const [status, missed, mastered] = await Promise.all([
            getStatus(),
            getMissedWords(),
            getMasteredWords()
        ]);

        const avgScore = status.level_scores.length > 0
            ? (status.level_scores.reduce((a, b) => a + b, 0) / status.level_scores.length).toFixed(1)
            : 'N/A';

        elements.statusDetails.innerHTML = `
            <p><strong>Language:</strong> ${status.language}</p>
            <p><strong>Current Level:</strong> ${status.difficulty}/${status.max_difficulty}</p>
            <p><strong>Total Completed:</strong> ${status.total_completed}</p>
            <p><strong>Story Progress:</strong> ${status.story_sentences_remaining} sentences remaining</p>
            <p><strong>Recent Average:</strong> ${avgScore}</p>
            <p><strong>Good Scores (≥80):</strong> ${status.good_score_count}/4 needed to advance</p>
            <p><strong>Poor Scores (<50):</strong> ${status.poor_score_count}/4 triggers demotion</p>
            <p><strong>Mastered Words:</strong> ${mastered.total}</p>
            ${mastered.words.length > 0 ? `<div class="words-list">${mastered.words.join(', ')}</div>` : ''}
            <p><strong>Words to Practice:</strong> ${missed.total}</p>
            ${missed.words.length > 0 ? `<div class="words-list">${missed.words.map(w => `${w.word} (${w.english})`).join(', ')}</div>` : ''}
        `;

        elements.statusModal.classList.remove('hidden');
    } catch (error) {
        console.error('Error fetching status:', error);
        alert('Failed to load status');
    }
}

function hideStatusModal() {
    elements.statusModal.classList.add('hidden');
}

async function loadNextSentence() {
    elements.loading.classList.remove('hidden');
    elements.currentTask.classList.add('hidden');
    elements.validationResult.classList.add('hidden');

    try {
        const data = await getNextSentence();

        currentSentence = data.sentence;
        currentStory = data.story;

        // Update status bar
        const status = await getStatus();
        updateStatusBar(status);

        // Render story
        renderStory(data.story, data.difficulty, data.sentence);

        // Show previous evaluation if any
        if (data.has_previous_evaluation && data.previous_evaluation) {
            showPreviousEvaluation(data.previous_evaluation);
        } else {
            elements.previousEval.classList.add('hidden');
        }

        // Show current task
        showCurrentTask(data.sentence);

    } catch (error) {
        console.error('Error loading sentence:', error);
        elements.loading.innerHTML = '<p>Error loading. <button onclick="loadNextSentence()">Retry</button></p>';
    }
}

async function handleSubmit(e) {
    e.preventDefault();

    const translation = elements.translationInput.value.trim();
    if (!translation) {
        elements.translationInput.focus();
        return;
    }

    elements.submitBtn.disabled = true;
    elements.submitBtn.textContent = 'Validating...';

    try {
        const result = await submitTranslation(currentSentence, translation);

        // Update status bar
        const status = await getStatus();
        updateStatusBar(status);

        // Show result
        showValidationResult(result, translation);

    } catch (error) {
        console.error('Error submitting:', error);
        alert('Failed to submit translation');
        elements.submitBtn.disabled = false;
    } finally {
        elements.submitBtn.textContent = 'Submit';
    }
}

async function handleHint() {
    elements.hintBtn.disabled = true;
    elements.hintBtn.textContent = 'Loading...';

    try {
        const hint = await getHint(currentSentence);

        let hintHtml = '<strong>Hint:</strong><br>';
        if (hint.noun) {
            hintHtml += `Noun: <em>${hint.noun[0]}</em> = ${hint.noun[1]}<br>`;
        }
        if (hint.verb) {
            hintHtml += `Verb: <em>${hint.verb[0]}</em> = ${hint.verb[1]}<br>`;
        }
        if (hint.adjective) {
            hintHtml += `Adjective: <em>${hint.adjective[0]}</em> = ${hint.adjective[1]}`;
        }
        if (!hint.noun && !hint.verb && !hint.adjective) {
            hintHtml = 'No hint available';
        }

        elements.hintDisplay.innerHTML = hintHtml;
        elements.hintDisplay.classList.remove('hidden');

    } catch (error) {
        console.error('Error getting hint:', error);
        alert('Failed to get hint');
    } finally {
        elements.hintBtn.disabled = false;
        elements.hintBtn.textContent = 'Hint';
    }
}

// Event Listeners
elements.translationForm.addEventListener('submit', handleSubmit);
elements.hintBtn.addEventListener('click', handleHint);
elements.nextBtn.addEventListener('click', loadNextSentence);
elements.statusBtn.addEventListener('click', showStatusModal);
elements.closeBtn.addEventListener('click', hideStatusModal);
elements.statusModal.addEventListener('click', (e) => {
    if (e.target === elements.statusModal) {
        hideStatusModal();
    }
});

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
        hideStatusModal();
    }
});

// Initialize
loadNextSentence();
