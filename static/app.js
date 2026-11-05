/**
 * Клиентска логика за Whisper BG Transcriber
 */

document.addEventListener("DOMContentLoaded", () => {
    // Елементи от DOM
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const fileInfoContainer = document.getElementById("file-info-container");
    const filenameDisplay = document.getElementById("filename-display");
    const filesizeDisplay = document.getElementById("filesize-display");
    const durationDisplay = document.getElementById("duration-display");
    const audioPreview = document.getElementById("audio-preview");
    const removeFileBtn = document.getElementById("remove-file-btn");
    const youtubeUrlContainer = document.getElementById("youtube-url-container");
    const youtubeUrlInput = document.getElementById("youtube-url-input");
    const youtubeFetchBtn = document.getElementById("youtube-fetch-btn");
    const youtubeFetchStatus = document.getElementById("youtube-fetch-status");

    const startBtn = document.getElementById("start-btn");
    const preprocessToggle = document.getElementById("preprocess-toggle");
    const beamToggle = document.getElementById("beam-toggle");
    const isolateToggle = document.getElementById("isolate-toggle");
    const timestampsToggle = document.getElementById("timestamps-toggle");
    const vadToggle = document.getElementById("vad-toggle");
    const debugModeToggle = document.getElementById("debug-mode-toggle");
    const promptInput = document.getElementById("prompt-input");

    // Елементи за Дебъг панела (KDiff3 / Meld)
    const debugDiffPanel = document.getElementById("debug-diff-panel");
    const debugModelBadge = document.getElementById("debug-model-badge");
    const debugFolderPath = document.getElementById("debug-folder-path");
    const copyDebugPathBtn = document.getElementById("copy-debug-path-btn");
    const launchDiff2WayBtn = document.getElementById("launch-diff-2way-btn");
    const launchDiff3WayBtn = document.getElementById("launch-diff-3way-btn");
    const openDebugFolderBtn = document.getElementById("open-debug-folder-btn");
    const dlDebugWhisperBtn = document.getElementById("dl-debug-whisper-btn");
    const dlDebugFinalBtn = document.getElementById("dl-debug-final-btn");
    const debugViewTabs = document.querySelectorAll(".debug-view-tabs .btn-tab");
    let currentViewStage = "final"; // "final", "rules", "whisper"
    let lastCompletedData = null;

    // Хардуерни индикатори и енджин
    const hardwareBadge = document.getElementById("hardware-badge");
    const hardwareText = document.getElementById("hardware-text");
    const engineSelect = document.getElementById("engine-select");
    const engineHint = document.getElementById("engine-hint");
    const statsDevice = document.getElementById("stats-device");

    // Зареждане на хардуерна информация от системата
    async function loadSystemHardwareInfo() {
        try {
            const res = await fetch("/api/system-info");
            if (res.ok) {
                const data = await res.json();
                if (hardwareBadge && hardwareText) {
                    hardwareText.textContent = data.label || data.device_name;
                    hardwareBadge.className = `badge badge-hardware ${data.badge_class || ""}`;
                }
                if (engineHint) {
                    if (data.device_type === "directml") {
                        engineHint.innerHTML = `⚡ <strong>Открит хардуер:</strong> ${data.device_name} (DirectML ускорение за AMD Radeon / Intel GPU).`;
                    } else if (data.device_type === "cuda") {
                        engineHint.innerHTML = `⚡ <strong>Открит хардуер:</strong> ${data.device_name} (NVIDIA CUDA ускорение).`;
                    } else {
                        engineHint.innerHTML = `⚙️ <strong>Хардуер:</strong> Изчисленията ще се изпълняват на процесора (${data.device_name}).`;
                    }
                }
            }
        } catch (err) {
            console.warn("Грешка при зареждане на хардуерна информация:", err);
        }
    }
    loadSystemHardwareInfo();

    // Елементи за Пост-процесинг, Локален AI (BgGPT/Qwen) и Gemini
    const postprocessRadios = document.querySelectorAll('input[name="postprocess_mode"]');
    const localAiConfigBox = document.getElementById("local-ai-config-box");
    const localAiModelSelect = document.getElementById("local-ai-model-select");
    const geminiConfigBox = document.getElementById("gemini-config-box");
    const geminiKeySourceRadios = document.querySelectorAll('input[name="gemini_key_source"]');
    const geminiKeyContainer = document.getElementById("gemini-key-input-container");
    const geminiEnvInfo = document.getElementById("gemini-env-info");
    const geminiApiKeyInput = document.getElementById("gemini-api-key");
    const toggleKeyVisibilityBtn = document.getElementById("toggle-key-visibility");
    const fetchGeminiModelsBtn = document.getElementById("fetch-gemini-models-btn");
    const geminiModelSelect = document.getElementById("gemini-model-select");
    const testGeminiBtn = document.getElementById("test-gemini-btn");
    const geminiTestStatus = document.getElementById("gemini-test-status");

    // Зареждане на запазен локален AI модел и Gemini ключ от localStorage
    const savedLocalAiModel = localStorage.getItem("whisper_local_ai_model");
    if (savedLocalAiModel && localAiModelSelect) {
        const hasOption = Array.from(localAiModelSelect.options).some(opt => opt.value === savedLocalAiModel);
        if (hasOption) {
            localAiModelSelect.value = savedLocalAiModel;
        }
    }
    if (localAiModelSelect) {
        localAiModelSelect.addEventListener("change", () => {
            localStorage.setItem("whisper_local_ai_model", localAiModelSelect.value);
        });
    }

    const savedGeminiKey = localStorage.getItem("whisper_gemini_api_key");
    if (savedGeminiKey && geminiApiKeyInput) {
        geminiApiKeyInput.value = savedGeminiKey;
    }
    const savedGeminiModel = localStorage.getItem("whisper_gemini_model");
    if (savedGeminiModel && geminiModelSelect) {
        geminiModelSelect.value = savedGeminiModel;
    }

    // Превключване на режима за пост-процесинг
    function applyPostprocessModeUI(mode) {
        if (mode === "gemini") {
            geminiConfigBox.classList.remove("hidden");
            if (localAiConfigBox) localAiConfigBox.classList.add("hidden");
        } else if (mode === "qwen") {
            geminiConfigBox.classList.add("hidden");
            if (localAiConfigBox) localAiConfigBox.classList.remove("hidden");
        } else {
            geminiConfigBox.classList.add("hidden");
            if (localAiConfigBox) localAiConfigBox.classList.add("hidden");
        }
    }

    const savedPostprocessMode = localStorage.getItem("whisper_postprocess_mode");
    if (savedPostprocessMode) {
        const matchingRadio = Array.from(postprocessRadios).find(r => r.value === savedPostprocessMode);
        if (matchingRadio) {
            matchingRadio.checked = true;
            applyPostprocessModeUI(savedPostprocessMode);
        }
    } else {
        const checkedRadio = document.querySelector('input[name="postprocess_mode"]:checked');
        if (checkedRadio) applyPostprocessModeUI(checkedRadio.value);
    }

    postprocessRadios.forEach(radio => {
        radio.addEventListener("change", () => {
            localStorage.setItem("whisper_postprocess_mode", radio.value);
            applyPostprocessModeUI(radio.value);
        });
    });

    // Превключване на източника на API ключ (въвеждане vs $GEMINI_API_KEY)
    geminiKeySourceRadios.forEach(radio => {
        radio.addEventListener("change", () => {
            if (radio.value === "input") {
                geminiKeyContainer.classList.remove("hidden");
                geminiEnvInfo.classList.add("hidden");
            } else {
                geminiKeyContainer.classList.add("hidden");
                geminiEnvInfo.classList.remove("hidden");
            }
        });
    });

    // Бутон за показване/скриване на паролата
    if (toggleKeyVisibilityBtn && geminiApiKeyInput) {
        toggleKeyVisibilityBtn.addEventListener("click", () => {
            const isPassword = geminiApiKeyInput.type === "password";
            geminiApiKeyInput.type = isPassword ? "text" : "password";
            toggleKeyVisibilityBtn.textContent = isPassword ? "🙈" : "👁️";
        });
    }

    // Запазване в localStorage при промяна
    if (geminiApiKeyInput) {
        geminiApiKeyInput.addEventListener("input", () => {
            localStorage.setItem("whisper_gemini_api_key", geminiApiKeyInput.value.trim());
        });
    }
    if (geminiModelSelect) {
        geminiModelSelect.addEventListener("change", () => {
            localStorage.setItem("whisper_gemini_model", geminiModelSelect.value);
        });
    }

    // Бутон 1: Извличане на наличните модели от Google API
    if (fetchGeminiModelsBtn) {
        fetchGeminiModelsBtn.addEventListener("click", async () => {
            const keySource = document.querySelector('input[name="gemini_key_source"]:checked').value;
            const manualKey = geminiApiKeyInput.value.trim();

            if (keySource === "input" && !manualKey) {
                geminiTestStatus.className = "gemini-test-status status-error";
                geminiTestStatus.textContent = "✗ Моля, първо въведете вашия Google Gemini API ключ в полето по-горе.";
                geminiTestStatus.classList.remove("hidden");
                geminiApiKeyInput.focus();
                return;
            }

            geminiTestStatus.className = "gemini-test-status status-loading";
            geminiTestStatus.textContent = "Свързване с Google AI Studio и извличане на модели за вашия акаунт...";
            geminiTestStatus.classList.remove("hidden");
            fetchGeminiModelsBtn.disabled = true;

            try {
                const resp = await fetch("/api/gemini/models", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        key_source: keySource,
                        api_key: manualKey,
                    }),
                });

                const data = await resp.json();
                if (data.success && data.models && data.models.length > 0) {
                    geminiModelSelect.innerHTML = "";
                    const currentSaved = localStorage.getItem("whisper_gemini_model") || data.default_model || "";
                    let selectedMatched = false;

                    data.models.forEach(m => {
                        const opt = document.createElement("option");
                        const modelId = typeof m === "string" ? m : m.id;
                        const modelLabel = typeof m === "string" ? m : (m.label || m.name || m.id);
                        opt.value = modelId;
                        opt.textContent = modelLabel;
                        if (modelId === currentSaved) {
                            opt.selected = true;
                            selectedMatched = true;
                        }
                        geminiModelSelect.appendChild(opt);
                    });

                    if (!selectedMatched && geminiModelSelect.options.length > 0) {
                        geminiModelSelect.options[0].selected = true;
                    }

                    localStorage.setItem("whisper_gemini_model", geminiModelSelect.value);

                    geminiTestStatus.className = "gemini-test-status status-success";
                    geminiTestStatus.textContent = `✓ Намерени са ${data.models.length} валидни модела от Google API! Изберете модел от списъка по-долу.`;
                    showToast(`Заредени са ${data.models.length} модела от Google!`);
                } else {
                    geminiTestStatus.className = "gemini-test-status status-error";
                    geminiTestStatus.textContent = `✗ ${data.error || "Не бяха открити валидни модели за този ключ."}`;
                }
            } catch (err) {
                geminiTestStatus.className = "gemini-test-status status-error";
                geminiTestStatus.textContent = `✗ Грешка при връзка с локалния сървър: ${err.message}`;
            } finally {
                fetchGeminiModelsBtn.disabled = false;
            }
        });
    }

    // Бутон 2: Тестване на избрания Gemini модел
    if (testGeminiBtn) {
        testGeminiBtn.addEventListener("click", async () => {
            const keySource = document.querySelector('input[name="gemini_key_source"]:checked').value;
            const manualKey = geminiApiKeyInput.value.trim();
            const modelName = geminiModelSelect.value;

            if (keySource === "input" && !manualKey) {
                geminiTestStatus.className = "gemini-test-status status-error";
                geminiTestStatus.textContent = "✗ Моля, първо въведете вашия Google Gemini API ключ.";
                geminiTestStatus.classList.remove("hidden");
                geminiApiKeyInput.focus();
                return;
            }

            if (!modelName) {
                geminiTestStatus.className = "gemini-test-status status-error";
                geminiTestStatus.textContent = "✗ Моля, първо заредете или изберете модел от списъка.";
                geminiTestStatus.classList.remove("hidden");
                return;
            }

            geminiTestStatus.className = "gemini-test-status status-loading";
            geminiTestStatus.textContent = `Тестване на модел '${modelName}' с пробна заявка...`;
            geminiTestStatus.classList.remove("hidden");
            testGeminiBtn.disabled = true;

            try {
                const resp = await fetch("/api/test-gemini", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        key_source: keySource,
                        api_key: manualKey,
                        model_name: modelName,
                    }),
                });

                const data = await resp.json();
                if (data.success) {
                    geminiTestStatus.className = "gemini-test-status status-success";
                    geminiTestStatus.textContent = `✓ ${data.message}`;
                    showToast(`Моделът ${modelName} работи отлично!`);
                } else {
                    geminiTestStatus.className = "gemini-test-status status-error";
                    geminiTestStatus.textContent = `✗ ${data.error || "Грешка при тестване на избрания модел."}`;
                }
            } catch (err) {
                geminiTestStatus.className = "gemini-test-status status-error";
                geminiTestStatus.textContent = `✗ Грешка в мрежата: ${err.message}`;
            } finally {
                testGeminiBtn.disabled = false;
            }
        });
    }

    const resultSection = document.getElementById("result-section");
    const statusBadge = document.getElementById("status-badge");
    const progressContainer = document.getElementById("progress-container");
    const progressText = document.getElementById("progress-text");
    const progressPct = document.getElementById("progress-pct");
    const progressFill = document.getElementById("progress-fill");

    const exportActions = document.getElementById("export-actions");
    const downloadDocxBtn = document.getElementById("download-docx-btn");
    const downloadTxtBtn = document.getElementById("download-txt-btn");
    const downloadSrtBtn = document.getElementById("download-srt-btn");
    const copyBtn = document.getElementById("copy-btn");

    const statsWords = document.getElementById("stats-words");
    const statsChars = document.getElementById("stats-chars");
    const statsTime = document.getElementById("stats-time");
    const resultTextArea = document.getElementById("result-text-area");
    const segmentsAccordion = document.getElementById("segments-accordion");
    const segmentsList = document.getElementById("segments-list");
    const toast = document.getElementById("toast");

    let currentJobId = null;
    let pollInterval = null;
    let lastUploadedFile = null;

    // Toast помощник
    function showToast(message, duration = 3000) {
        toast.textContent = message;
        toast.classList.remove("hidden");
        setTimeout(() => {
            toast.classList.add("hidden");
        }, duration);
    }

    // Форматиране на секунди във времеви маркер
    function formatTime(seconds) {
        const m = Math.floor(seconds / 60);
        const s = Math.floor(seconds % 60);
        return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    }

    // Drag & Drop събития
    dropZone.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0) {
            handleFileUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            handleFileUpload(fileInput.files[0]);
        }
    });

    removeFileBtn.addEventListener("click", () => {
        resetApp();
    });

    // Функция за качване на аудио файл
    async function handleFileUpload(file) {
        lastUploadedFile = file;
        const formData = new FormData();
        formData.append("file", file);

        showToast("Качване на аудио файла...");
        startBtn.disabled = true;

        try {
            const response = await fetch("/api/upload", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || "Грешка при качване.");
            }

            const data = await response.json();
            currentJobId = data.job_id;

            // Обновяване на интерфейса
            filenameDisplay.textContent = data.filename;
            filesizeDisplay.textContent = `${data.file_size_mb} MB`;
            durationDisplay.textContent = `${data.duration_formatted} мин.`;
            
            // Настройване на аудио плеъра
            audioPreview.src = `/api/audio/${currentJobId}`;
            fileInfoContainer.classList.remove("hidden");
            dropZone.classList.add("hidden");
            if (youtubeUrlContainer) youtubeUrlContainer.classList.add("hidden");
            startBtn.disabled = false;

            showToast("Файлът е готов за транскрипция!");
        } catch (error) {
            alert(`Грешка: ${error.message}`);
            startBtn.disabled = true;
        }
    }

    // Извличане на аудио от YouTube URL
    async function handleYouTubeFetch() {
        if (!youtubeUrlInput) return;
        const url = youtubeUrlInput.value.trim();
        if (!url) {
            if (youtubeFetchStatus) {
                youtubeFetchStatus.className = "gemini-test-status status-error";
                youtubeFetchStatus.textContent = "✗ Моля, поставете валиден линк от YouTube в полето.";
                youtubeFetchStatus.classList.remove("hidden");
            }
            youtubeUrlInput.focus();
            return;
        }

        if (youtubeFetchStatus) {
            youtubeFetchStatus.className = "gemini-test-status status-loading";
            youtubeFetchStatus.textContent = "⏳ Извличане на аудио потока от YouTube (yt-dlp)... Моля, изчакайте няколко секунди.";
            youtubeFetchStatus.classList.remove("hidden");
        }
        if (youtubeFetchBtn) youtubeFetchBtn.disabled = true;
        startBtn.disabled = true;

        try {
            const response = await fetch("/api/youtube", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ url }),
            });

            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.detail || "Неуспешно извличане от YouTube.");
            }

            currentJobId = data.job_id;
            filenameDisplay.textContent = data.filename;
            filesizeDisplay.textContent = `${data.file_size_mb} MB`;
            durationDisplay.textContent = `${data.duration_formatted} мин.`;

            audioPreview.src = `/api/audio/${currentJobId}`;
            fileInfoContainer.classList.remove("hidden");
            dropZone.classList.add("hidden");
            if (youtubeUrlContainer) youtubeUrlContainer.classList.add("hidden");
            if (youtubeFetchStatus) youtubeFetchStatus.classList.add("hidden");
            startBtn.disabled = false;

            showToast("✓ Аудиото от YouTube е заредено и готово за транскрипция!");
        } catch (error) {
            if (youtubeFetchStatus) {
                youtubeFetchStatus.className = "gemini-test-status status-error";
                youtubeFetchStatus.textContent = `✗ ${error.message}`;
                youtubeFetchStatus.classList.remove("hidden");
            } else {
                alert(`Грешка: ${error.message}`);
            }
        } finally {
            if (youtubeFetchBtn) youtubeFetchBtn.disabled = false;
        }
    }

    if (youtubeFetchBtn) {
        youtubeFetchBtn.addEventListener("click", handleYouTubeFetch);
    }
    if (youtubeUrlInput) {
        youtubeUrlInput.addEventListener("keydown", (e) => {
            if (e.key === "Enter") {
                e.preventDefault();
                handleYouTubeFetch();
            }
        });
    }

    // Стартиране на транскрибирането
    startBtn.addEventListener("click", async () => {
        if (!currentJobId) return;

        const selectedModel = document.querySelector('input[name="whisper_model"]:checked').value;
        const includeTimestamps = timestampsToggle.checked;
        const vadFilter = vadToggle.checked;
        const isolateErrors = isolateToggle ? isolateToggle.checked : true;
        const audioPreprocess = preprocessToggle ? preprocessToggle.checked : true;
        const expandedBeam = beamToggle ? beamToggle.checked : true;
        
        // Пост-процесинг настройки
        const selectedPostprocessRadio = document.querySelector('input[name="postprocess_mode"]:checked');
        const postprocessMode = selectedPostprocessRadio ? selectedPostprocessRadio.value : "local";
        const localAiModel = localAiModelSelect ? localAiModelSelect.value : "zealod/BgGPT-Gemma-2-2.6B-IT-v1.0-int4-onnx-directml";
        const selectedKeySourceRadio = document.querySelector('input[name="gemini_key_source"]:checked');
        const geminiKeySource = selectedKeySourceRadio ? selectedKeySourceRadio.value : "input";
        const geminiApiKey = geminiApiKeyInput ? geminiApiKeyInput.value.trim() : "";
        const geminiModel = geminiModelSelect ? geminiModelSelect.value : "gemini-3.8-flash";

        const customPrompt = promptInput.value.trim();
        const selectedEngine = engineSelect ? engineSelect.value : "auto";
        const debugMode = debugModeToggle ? debugModeToggle.checked : true;

        // Подготовка на интерфейса
        startBtn.disabled = true;
        resultSection.classList.remove("hidden");
        exportActions.classList.add("hidden");
        if (debugDiffPanel) debugDiffPanel.classList.add("hidden");
        segmentsAccordion.classList.add("hidden");
        resultTextArea.textContent = "";
        currentViewStage = "final";
        debugViewTabs.forEach(btn => btn.classList.toggle("active", btn.dataset.view === "final"));
        
        statusBadge.textContent = "Инициализиране...";
        statusBadge.className = "status-badge badge-processing";
        progressFill.style.width = "5%";
        progressPct.textContent = "5%";
        progressText.textContent = `Зареждане на модел Whisper ${selectedModel}...`;

        if (statsDevice && hardwareText) {
            statsDevice.textContent = `Хардуер: ${hardwareText.textContent || "Подготовка..."}`;
        }

        // Превъртане към резултатите
        resultSection.scrollIntoView({ behavior: "smooth", block: "nearest" });

        try {
            const resp = await fetch("/api/transcribe", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    job_id: currentJobId,
                    model_name: selectedModel,
                    language: "bg",
                    vad_filter: vadFilter,
                    include_timestamps: includeTimestamps,
                    condition_on_previous_text: !isolateErrors,
                    audio_preprocessing: audioPreprocess,
                    expanded_beam_search: expandedBeam,
                    apply_postprocessing: postprocessMode !== "none",
                    postprocess_mode: postprocessMode,
                    local_ai_model: localAiModel,
                    gemini_api_key_source: geminiKeySource,
                    gemini_api_key: geminiApiKey,
                    gemini_model: geminiModel,
                    custom_prompt: customPrompt,
                    engine: selectedEngine,
                    debug_mode: debugMode,
                }),
            });

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || "Грешка при стартиране.");
            }

            // Започване на периодично питане за статус (polling)
            startStatusPolling();
        } catch (error) {
            statusBadge.textContent = "Грешка";
            statusBadge.className = "status-badge badge-error";
            progressText.textContent = error.message;
            startBtn.disabled = false;
        }
    });

    // Проследяване на напредъка
    function startStatusPolling() {
        if (pollInterval) clearInterval(pollInterval);

        pollInterval = setInterval(async () => {
            if (!currentJobId) return;

            try {
                const res = await fetch(`/api/status/${currentJobId}`);
                if (!res.ok) return;

                const data = await res.json();

                // Обновяване на прогрес бара
                const pct = Math.max(5, Math.min(100, data.progress || 0));
                progressFill.style.width = `${pct}%`;
                progressPct.textContent = `${pct}%`;
                progressText.textContent = data.message || "Обработва се...";

                // Обновяване на статуса в бейджа
                if (data.status === "transcribing") {
                    statusBadge.textContent = "Транскрибиране...";
                    statusBadge.className = "status-badge badge-processing";
                } else if (data.status === "loading_model") {
                    statusBadge.textContent = "Зареждане на модел...";
                    statusBadge.className = "status-badge badge-processing";
                } else if (data.status === "postprocessing") {
                    statusBadge.textContent = "Редактиране...";
                    statusBadge.className = "status-badge badge-processing";
                } else if (data.status === "fallback") {
                    statusBadge.textContent = "CPU Fallback";
                    statusBadge.className = "status-badge badge-warning";
                }

                // Обновяване на хардуерен статус в реално време
                if (data.device_label && statsDevice) {
                    statsDevice.textContent = `Хардуер: ${data.device_label}`;
                }
                if (data.badge_class && hardwareBadge) {
                    hardwareBadge.className = `badge badge-hardware ${data.badge_class}`;
                    if (data.device_label && hardwareText) {
                        hardwareText.textContent = data.device_label;
                    }
                }

                // Междинно обновяване на текста
                if (data.segments && data.segments.length > 0) {
                    renderText(data.segments, timestampsToggle.checked);
                    updateWordStats(data.segments);
                }

                if (data.status === "completed") {
                    clearInterval(pollInterval);
                    finishTranscription(data);
                } else if (data.status === "error") {
                    clearInterval(pollInterval);
                    statusBadge.textContent = "Грешка";
                    statusBadge.className = "status-badge badge-error";
                    progressText.textContent = data.error || "Възникна грешка при транскрибирането.";
                    startBtn.disabled = false;
                }
            } catch (e) {
                console.error("Грешка при проверка на статус:", e);
            }
        }, 800);
    }

    // Завършване на транскрипцията
    function finishTranscription(data) {
        lastCompletedData = data;
        statusBadge.textContent = "Завършено";
        statusBadge.className = "status-badge badge-completed";
        progressFill.style.width = "100%";
        progressPct.textContent = "100%";
        progressText.textContent = `Успешно транскрибирано за ${data.elapsed_seconds || 0} сек.!`;

        statsTime.textContent = `Време за обработка: ${data.elapsed_seconds || 0} сек.`;
        if (statsDevice) {
            statsDevice.textContent = data.device_label ? `Хардуер: ${data.device_label}` : (data.device_name ? `Хардуер: ${data.device_name}` : "Хардуер: Успешно");
        }

        exportActions.classList.remove("hidden");
        startBtn.disabled = false;

        // Показване на Дебъг панела за KDiff3 / Meld
        const isDebugEnabled = debugModeToggle ? debugModeToggle.checked : true;
        if (debugDiffPanel && isDebugEnabled) {
            debugDiffPanel.classList.remove("hidden");
            if (debugModelBadge) {
                debugModelBadge.textContent = data.ai_model_used && data.ai_model_used !== "none"
                    ? `Коректор: ${data.ai_model_used}`
                    : "Суров Whisper";
            }
            if (debugFolderPath && data.debug_files && data.debug_files.folder) {
                debugFolderPath.textContent = data.debug_files.folder;
            }
        }

        renderText(data.segments, timestampsToggle.checked, currentViewStage);
        updateWordStats(data.segments, currentViewStage);
        renderSegmentsAccordion(data.segments);

        showToast("Транскрипцията е готова! Дебъг файловете за KDiff3 / Meld са записани.");
    }

    function getSegmentTextByStage(seg, stage) {
        if (stage === "whisper") {
            return seg.raw_text || seg.text || "";
        } else if (stage === "rules") {
            return seg.pre_ai_text || seg.text || "";
        }
        return seg.text || "";
    }

    // Изчертаване на текста в главното поле
    function renderText(segments, withTimestamps, stage = "final") {
        if (!segments || segments.length === 0) return;

        let output = "";
        if (withTimestamps) {
            output = segments.map(s => `[${formatTime(s.start)} - ${formatTime(s.end)}]  ${getSegmentTextByStage(s, stage)}`).join("\n\n");
        } else {
            output = segments.map(s => getSegmentTextByStage(s, stage)).join("\n\n");
        }
        resultTextArea.textContent = output;
    }

    // Обновяване на статистиката за думи и символи
    function updateWordStats(segments, stage = "final") {
        const fullText = segments.map(s => getSegmentTextByStage(s, stage)).join(" ").trim();
        const words = fullText ? fullText.split(/\s+/).length : 0;
        const chars = fullText.length;
        statsWords.textContent = `Думи: ${words}`;
        statsChars.textContent = `Символи: ${chars}`;
    }

    // Превключване между табовете в дебъг панела (Финален / Преди AI / След Whisper)
    debugViewTabs.forEach(tabBtn => {
        tabBtn.addEventListener("click", () => {
            currentViewStage = tabBtn.dataset.view || "final";
            debugViewTabs.forEach(b => b.classList.toggle("active", b === tabBtn));
            if (lastCompletedData && lastCompletedData.segments) {
                renderText(lastCompletedData.segments, timestampsToggle.checked, currentViewStage);
                updateWordStats(lastCompletedData.segments, currentViewStage);
            }
        });
    });

    // Бутони за стартиране на KDiff3 / Meld и отваряне на папка debug_diff
    async function triggerDiffLaunch(mode = "2way") {
        if (!currentJobId) return;
        try {
            const res = await fetch(`/api/debug/launch-diff/${currentJobId}?mode=${mode}`, { method: "POST" });
            if (res.ok) {
                const info = await res.json();
                showToast(info.message || "Стартиране на инструмент за сравнение...", 4000);
            }
        } catch (err) {
            showToast("Грешка при стартиране на KDiff3 / Meld.");
        }
    }

    if (launchDiff2WayBtn) {
        launchDiff2WayBtn.addEventListener("click", () => triggerDiffLaunch("2way"));
    }
    if (launchDiff3WayBtn) {
        launchDiff3WayBtn.addEventListener("click", () => triggerDiffLaunch("3way"));
    }
    if (openDebugFolderBtn) {
        openDebugFolderBtn.addEventListener("click", async () => {
            if (!currentJobId) return;
            try {
                const res = await fetch(`/api/debug/open-folder/${currentJobId}`, { method: "POST" });
                if (res.ok) {
                    showToast("📂 Папката с дебъг файловете е отворена във File Explorer!");
                }
            } catch (err) {
                showToast("Неуспешно отваряне на папката.");
            }
        });
    }
    if (dlDebugWhisperBtn) {
        dlDebugWhisperBtn.addEventListener("click", () => {
            if (!currentJobId) return;
            window.location.href = `/api/download/${currentJobId}/debug_whisper`;
        });
    }
    if (dlDebugFinalBtn) {
        dlDebugFinalBtn.addEventListener("click", () => {
            if (!currentJobId) return;
            window.location.href = `/api/download/${currentJobId}/debug_final`;
        });
    }
    if (copyDebugPathBtn && debugFolderPath) {
        copyDebugPathBtn.addEventListener("click", async () => {
            try {
                await navigator.clipboard.writeText(debugFolderPath.textContent.trim());
                showToast("✓ Пътят до дебъг папката е копиран в клипборда!");
            } catch (e) {
                showToast("Неуспешно копиране на пътя.");
            }
        });
    }

    // Изчертаване на списъка с времеви интервали
    function renderSegmentsAccordion(segments) {
        if (!segments || segments.length === 0) return;

        segmentsList.innerHTML = "";
        segments.forEach(seg => {
            const item = document.createElement("div");
            item.className = "segment-item";
            item.innerHTML = `
                <span class="segment-timestamp">${formatTime(seg.start)} - ${formatTime(seg.end)}</span>
                <span class="segment-text">${seg.text}</span>
            `;
            // При клик превърта аудио плеъра до съответния начален момент
            item.addEventListener("click", () => {
                audioPreview.currentTime = seg.start;
                audioPreview.play();
            });
            segmentsList.appendChild(item);
        });

        segmentsAccordion.classList.remove("hidden");
    }

    // Превключвател на времеви маркери в реално време
    timestampsToggle.addEventListener("change", async () => {
        if (!currentJobId) return;
        try {
            const res = await fetch(`/api/status/${currentJobId}`);
            if (res.ok) {
                const data = await res.json();
                if (data.segments && data.segments.length > 0) {
                    renderText(data.segments, timestampsToggle.checked, currentViewStage);
                }
            }
        } catch (e) {
            console.error(e);
        }
    });

    // Бутони за сваляне
    downloadDocxBtn.addEventListener("click", () => {
        if (!currentJobId) return;
        const ts = timestampsToggle.checked;
        window.location.href = `/api/download/${currentJobId}/docx?timestamps=${ts}`;
    });

    downloadTxtBtn.addEventListener("click", () => {
        if (!currentJobId) return;
        const ts = timestampsToggle.checked;
        window.location.href = `/api/download/${currentJobId}/txt?timestamps=${ts}`;
    });

    downloadSrtBtn.addEventListener("click", () => {
        if (!currentJobId) return;
        window.location.href = `/api/download/${currentJobId}/srt`;
    });

    // Бутон за копиране
    copyBtn.addEventListener("click", async () => {
        const text = resultTextArea.textContent;
        if (!text) return;

        try {
            await navigator.clipboard.writeText(text);
            showToast("✓ Текстът е копиран в клипборда!");
        } catch (err) {
            showToast("Неуспешно автоматично копиране.");
        }
    });

    // Нулиране
    function resetApp() {
        if (pollInterval) clearInterval(pollInterval);
        currentJobId = null;
        lastUploadedFile = null;
        lastCompletedData = null;
        fileInput.value = "";
        audioPreview.pause();
        audioPreview.src = "";
        fileInfoContainer.classList.add("hidden");
        dropZone.classList.remove("hidden");
        if (youtubeUrlContainer) youtubeUrlContainer.classList.remove("hidden");
        if (youtubeUrlInput) youtubeUrlInput.value = "";
        if (youtubeFetchStatus) youtubeFetchStatus.classList.add("hidden");
        startBtn.disabled = true;
        resultSection.classList.add("hidden");
        if (debugDiffPanel) debugDiffPanel.classList.add("hidden");
    }
});
